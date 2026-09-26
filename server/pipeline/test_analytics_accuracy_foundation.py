"""
test_analytics_accuracy_foundation.py — Unit Tests for Unified 5-Stage Analytics Accuracy Engine
"""

import unittest
from server.pipeline.analytics_accuracy_foundation import (
    TeamIdentityMetrics,
    PitchHomographyMetrics,
    TrajectoryKinematicsMetrics,
    UnifiedAnalyticsAccuracyScorecard,
)


class TestAnalyticsAccuracyFoundation(unittest.TestCase):
    def test_01_team_identity_perfect_purity(self):
        # 2 tracks across 15 frames, perfectly stable team assignments
        tracks = []
        for _ in range(15):
            tracks.append({
                1: {"team": 0},
                2: {"team": 1},
            })

        res = TeamIdentityMetrics.evaluate(tracks, min_track_length=5)
        self.assertEqual(res["total_tracks"], 2)
        self.assertEqual(res["mean_team_purity"], 1.0)
        self.assertEqual(res["team_flip_count"], 0)
        self.assertEqual(res["team_flip_rate"], 0.0)
        self.assertEqual(res["stable_tracks_pct"], 100.0)
        self.assertAlmostEqual(res["score"], 1.0)

    def test_02_team_identity_with_flips(self):
        # Track 1 has severe flickering (flips between 0 and 1)
        tracks = []
        teams_1 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        for t in teams_1:
            tracks.append({1: {"team": t}})

        res = TeamIdentityMetrics.evaluate(tracks, min_track_length=5)
        self.assertEqual(res["total_tracks"], 1)
        self.assertEqual(res["team_flip_count"], 9)
        self.assertEqual(res["mean_team_purity"], 0.5)
        self.assertEqual(res["stable_tracks_pct"], 0.0)
        self.assertLess(res["score"], 0.6)

    def test_03_pitch_homography_bounds_and_teleportation(self):
        # Normal in-bounds trajectory
        tracks_valid = [
            {1: {"position_transformed": (50.0, 34.0)}},
            {1: {"position_transformed": (50.2, 34.1)}},
            {1: {"position_transformed": (50.4, 34.2)}},
        ]
        res = PitchHomographyMetrics.evaluate(tracks_valid, fps=25.0)
        self.assertEqual(res["bounds_validity_pct"], 100.0)
        self.assertEqual(res["teleport_jumps_count"], 0)
        self.assertAlmostEqual(res["score"], 1.0)

        # Out-of-bounds point + teleportation jump
        tracks_teleport = [
            {1: {"position_transformed": (50.0, 34.0)}},
            {1: {"position_transformed": (80.0, 34.0)}},  # 30m jump in 0.04s = 750 m/s!
            {1: {"position_transformed": (200.0, 150.0)}}, # way out of pitch bounds
        ]
        res_bad = PitchHomographyMetrics.evaluate(tracks_teleport, fps=25.0)
        self.assertEqual(res_bad["teleport_jumps_count"], 2)  # both jumps are extreme
        self.assertLess(res_bad["bounds_validity_pct"], 100.0)
        self.assertLess(res_bad["score"], 0.8)

    def test_04_trajectory_kinematics_speed_and_acceleration(self):
        # Realistic running: ~18 km/h (5 m/s) -> 0.2m per frame at 25 fps
        tracks_normal = [
            {1: {"position_transformed": (10.0 + i * 0.2, 20.0)}}
            for i in range(25)
        ]
        res = TrajectoryKinematicsMetrics.evaluate(tracks_normal, fps=25.0)
        self.assertEqual(res["speeding_violations"], 0)
        self.assertEqual(res["accel_violations"], 0)
        self.assertAlmostEqual(res["mean_speed_kmh"], 18.0, delta=0.5)
        self.assertAlmostEqual(res["score"], 1.0)

        # Unphysical sprint: 50 km/h (> 37 km/h limit)
        # 50 km/h = 13.88 m/s -> ~0.55m per frame at 25 fps
        tracks_superhuman = [
            {1: {"position_transformed": (10.0 + i * 0.55, 20.0)}}
            for i in range(25)
        ]
        res_fast = TrajectoryKinematicsMetrics.evaluate(tracks_superhuman, fps=25.0)
        self.assertGreater(res_fast["speeding_violations"], 0)
        self.assertGreater(res_fast["peak_speed_kmh"], 37.0)
        self.assertLess(res_fast["score"], 1.0)

    def test_05_unified_scorecard(self):
        det_m = {"f1_score": 0.90}
        track_m = {"HOTA": 0.80}
        team_m = {"score": 0.95}
        homo_m = {"score": 0.92}
        kin_m = {"score": 0.88}

        card = UnifiedAnalyticsAccuracyScorecard.evaluate(
            det_m, track_m, team_m, homo_m, kin_m
        )
        # 0.90*0.2 + 0.80*0.25 + 0.95*0.15 + 0.92*0.25 + 0.88*0.15
        # = 0.18 + 0.20 + 0.1425 + 0.23 + 0.132 = 0.8845
        self.assertAlmostEqual(card["holistic_accuracy_index"], 0.8845, places=3)
        self.assertEqual(card["status"], "EXCELLENT")

    def test_06_e2e_full_chain_integration(self):
        from server.pipeline.detection_tracking_evaluator import (
            DetectionMetrics,
            TrackingMetrics,
        )

        # Build realistic 20-frame match data
        gt_frames = []
        pred_frames = []
        tracks_players = []

        for fi in range(20):
            # 2 players in frame
            p1_box = [100.0 + fi * 2.0, 100.0, 140.0 + fi * 2.0, 180.0]
            p2_box = [300.0, 200.0 + fi * 1.5, 340.0, 280.0 + fi * 1.5]

            gt_frames.append({
                1: {"bbox": p1_box},
                2: {"bbox": p2_box},
            })
            pred_frames.append({
                1: {"bbox": p1_box},
                2: {"bbox": p2_box},
            })
            tracks_players.append({
                1: {
                    "bbox": p1_box,
                    "team": 0,
                    "position_transformed": (20.0 + fi * 0.1, 30.0),
                },
                2: {
                    "bbox": p2_box,
                    "team": 1,
                    "position_transformed": (60.0, 10.0 + fi * 0.08),
                },
            })

        det = DetectionMetrics.evaluate(gt_frames, pred_frames)
        track = TrackingMetrics.evaluate(gt_frames, pred_frames)
        team = TeamIdentityMetrics.evaluate(tracks_players)
        homo = PitchHomographyMetrics.evaluate(tracks_players)
        kin = TrajectoryKinematicsMetrics.evaluate(tracks_players)

        scorecard = UnifiedAnalyticsAccuracyScorecard.evaluate(
            det, track, team, homo, kin
        )

        self.assertEqual(det["f1_score"], 1.0)
        self.assertEqual(track["HOTA"], 1.0)
        self.assertEqual(team["score"], 1.0)
        self.assertEqual(homo["score"], 1.0)
        self.assertEqual(kin["score"], 1.0)
        self.assertEqual(scorecard["holistic_accuracy_index"], 1.0)
        self.assertEqual(scorecard["status"], "EXCELLENT")


if __name__ == "__main__":
    unittest.main()

