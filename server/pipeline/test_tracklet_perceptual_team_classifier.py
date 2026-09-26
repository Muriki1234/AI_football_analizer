"""
test_tracklet_perceptual_team_classifier.py
===========================================
Unit test suite for TrackletPerceptualTeamClassifier.
"""

import unittest
import numpy as np
import cv2

from server.pipeline.tracklet_perceptual_team_classifier import (
    TrackletPerceptualTeamClassifier,
    delta_e_cie76,
    lab_to_hex,
    run_kmeans_pp,
)


class TestTrackletPerceptualTeamClassifier(unittest.TestCase):
    def setUp(self):
        # Create a synthetic football field image (green turf)
        self.img_h, self.img_w = 720, 1280
        self.frame = np.full((self.img_h, self.img_w, 3), (34, 139, 34), dtype=np.uint8)  # Forest Green BGR

    def _draw_player(self, frame, bbox, kit_bgr):
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        # Draw player body
        cv2.rectangle(frame, (x1, y1), (x2, y2), kit_bgr, -1)
        # Add green turf border inside bbox edges to simulate realistic imperfect crops
        cv2.rectangle(frame, (x1, y1 + int(bh * 0.6)), (x2, y2), (34, 139, 34), -1)

    def test_torso_extraction_and_grass_rejection(self):
        classifier = TrackletPerceptualTeamClassifier()
        # Red kit player: BGR (20, 20, 220)
        red_bgr = (20, 20, 220)
        bbox = [100, 200, 160, 360]
        self._draw_player(self.frame, bbox, red_bgr)

        success = classifier.add_observation(
            frame_idx=10,
            track_id=1,
            bbox=bbox,
            frame_bgr=self.frame,
        )
        self.assertTrue(success)
        self.assertIn(1, classifier.tracklets)
        profile = classifier.tracklets[1]
        self.assertEqual(len(profile.samples), 1)

        # The extracted sample should be predominantly reddish (high a* in Lab)
        lab = profile.samples[0].lab_median
        # In Lab OpenCV: L in [0, 255], a in [0, 255] where > 128 is reddish
        self.assertGreater(lab[1], 140, "Red kit should have elevated a* (red-green axis)")

    def test_bimodal_clustering_cie_lab(self):
        classifier = TrackletPerceptualTeamClassifier()
        blue_bgr = (220, 50, 20)   # Blue kit
        white_bgr = (240, 240, 240) # White kit

        # 5 Blue players (IDs 1-5)
        for tid in range(1, 6):
            bbox = [50 * tid, 100, 50 * tid + 40, 220]
            self._draw_player(self.frame, bbox, blue_bgr)
            classifier.add_observation(frame_idx=1, track_id=tid, bbox=bbox, frame_bgr=self.frame)

        # 5 White players (IDs 6-10)
        for tid in range(6, 11):
            bbox = [50 * tid, 300, 50 * tid + 40, 420]
            self._draw_player(self.frame, bbox, white_bgr)
            classifier.add_observation(frame_idx=1, track_id=tid, bbox=bbox, frame_bgr=self.frame)

        result = classifier.fit()
        self.assertTrue(result["success"])
        self.assertEqual(result["total_tracklets"], 10)
        self.assertGreater(result["inter_team_delta_e"], 25.0)

        # Verify all blue players belong to one team, white to the other
        blue_teams = [classifier.predict(tid)["team_id"] for tid in range(1, 6)]
        white_teams = [classifier.predict(tid)["team_id"] for tid in range(6, 11)]

        self.assertEqual(len(set(blue_teams)), 1, "All blue players must have the same team ID")
        self.assertEqual(len(set(white_teams)), 1, "All white players must have the same team ID")
        self.assertNotEqual(blue_teams[0], white_teams[0], "Blue and white teams must be distinct")

    def test_goalkeeper_referee_outlier_detection(self):
        classifier = TrackletPerceptualTeamClassifier(outlier_cluster_threshold_de=30.0)
        # Outfield: Red (Team 1) and Navy Blue (Team 2)
        red_bgr = (20, 20, 210)
        navy_bgr = (180, 40, 10)
        # Goalkeeper: Bright Neon Yellow BGR (0, 240, 240)
        gk_yellow_bgr = (0, 240, 240)
        # Referee: Solid Black BGR (15, 15, 15)
        ref_black_bgr = (15, 15, 15)

        for tid in range(1, 5):
            bbox = [60 * tid, 50, 60 * tid + 40, 180]
            self._draw_player(self.frame, bbox, red_bgr)
            classifier.add_observation(frame_idx=1, track_id=tid, bbox=bbox, frame_bgr=self.frame)

        for tid in range(5, 9):
            bbox = [60 * tid, 200, 60 * tid + 40, 330]
            self._draw_player(self.frame, bbox, navy_bgr)
            classifier.add_observation(frame_idx=1, track_id=tid, bbox=bbox, frame_bgr=self.frame)

        # 1 Goalkeeper (tid=99)
        gk_bbox = [100, 500, 140, 630]
        self._draw_player(self.frame, gk_bbox, gk_yellow_bgr)
        classifier.add_observation(frame_idx=1, track_id=99, bbox=gk_bbox, frame_bgr=self.frame)

        # 1 Referee (tid=100)
        ref_bbox = [250, 500, 290, 630]
        self._draw_player(self.frame, ref_bbox, ref_black_bgr)
        classifier.add_observation(frame_idx=1, track_id=100, bbox=ref_bbox, frame_bgr=self.frame)

        result = classifier.fit()
        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["outliers_detected"], 1)

        gk_pred = classifier.predict(99)
        ref_pred = classifier.predict(100)
        self.assertTrue(gk_pred["is_goalkeeper"] or gk_pred["is_referee"])
        self.assertTrue(ref_pred["is_referee"] or ref_pred["is_goalkeeper"])

    def test_spatial_temporal_fallback_continuity(self):
        classifier = TrackletPerceptualTeamClassifier()
        # Add 4 players so classifier can fit
        for tid in [1, 2]:
            bbox = [100 * tid, 100, 100 * tid + 40, 220]
            self._draw_player(self.frame, bbox, (200, 30, 20))
            classifier.add_observation(frame_idx=1, track_id=tid, bbox=bbox, frame_bgr=self.frame)

        for tid in [3, 4]:
            bbox = [100 * tid, 300, 100 * tid + 40, 420]
            self._draw_player(self.frame, bbox, (240, 240, 240))
            classifier.add_observation(frame_idx=1, track_id=tid, bbox=bbox, frame_bgr=self.frame)

        classifier.fit()

        # Completely unobserved tracklet (ID=999)
        pred_unobserved = classifier.predict(999)
        self.assertTrue(pred_unobserved["fallback"])
        self.assertEqual(pred_unobserved["confidence"], 0.0)
        self.assertEqual(pred_unobserved["fallback_mode"], "unobserved_zero_confidence")

    def test_kmeans_pp_numerical_stability(self):
        # 100 data points in 2 clearly separated clusters
        cluster_a = np.random.normal(loc=[50, 100, 100], scale=2.0, size=(50, 3)).astype(np.float32)
        cluster_b = np.random.normal(loc=[180, 150, 150], scale=2.0, size=(50, 3)).astype(np.float32)
        data = np.vstack([cluster_a, cluster_b])

        centroids, labels, inertia = run_kmeans_pp(data, k=2, n_init=10, random_seed=42)
        self.assertEqual(centroids.shape, (2, 3))
        self.assertEqual(len(labels), 100)
        self.assertLess(inertia, 1600.0)

    def test_excessive_grass_occlusion_rejection(self):
        classifier = TrackletPerceptualTeamClassifier()
        # Almost pure grass bbox (90% green turf)
        bbox = [100, 100, 150, 200]
        # Frame is already pure green turf
        success = classifier.add_observation(frame_idx=1, track_id=55, bbox=bbox, frame_bgr=self.frame)
        self.assertFalse(success, "Bbox with >85% grass should be rejected")

    def test_multi_frame_reservoir_quality_upgrade(self):
        classifier = TrackletPerceptualTeamClassifier()
        # Tracklet 1 observed across 20 frames with improving quality
        for f in range(20):
            bbox = [100, 100, 100 + 40 + f, 100 + 80 + f * 2]
            self._draw_player(self.frame, bbox, (210, 30, 30))
            classifier.add_observation(frame_idx=f, track_id=1, bbox=bbox, frame_bgr=self.frame)

        profile = classifier.tracklets[1]
        self.assertLessEqual(len(profile.samples), profile.max_samples)
        # Verify the aggregate Lab is non-empty and red-dominant
        agg_lab = profile.get_aggregate_lab()
        self.assertIsNotNone(agg_lab)
        self.assertGreater(agg_lab[1], 135)


if __name__ == "__main__":
    unittest.main()
