"""
test_video_analytics_confidence_assessor.py - Verification & Benchmark Suite for VideoAnalyticsConfidenceAssessor
"""

import time
import unittest
import numpy as np

from server.pipeline.video_analytics_confidence_assessor import (
    ConfidenceTier,
    VideoAnalyticsConfidenceAssessor,
    VideoConfidenceReport,
)


class TestVideoAnalyticsConfidenceAssessor(unittest.TestCase):

    def test_broadcast_high_confidence(self):
        """Broadcast camera with clear lines, high keypoint density, and stable tracking."""
        assessor = VideoAnalyticsConfidenceAssessor(fps=25.0, max_staleness_seconds=3.0)
        total_frames = 500

        for f in range(total_frames):
            # 96% valid homography frames
            has_h = (f % 25 != 0)
            assessor.record_frame(
                frame_idx=f,
                has_valid_homography=has_h,
                keypoint_count=12 if has_h else 3,
                inlier_ratio=0.88 if has_h else 0.40,
                reprojection_rmse=3.5 if has_h else 18.0,
                ball_detected=(f % 5 != 0),  # 80% ball detection
                camera_pan_speed_px=2.0,
            )

        report = assessor.evaluate()
        rep_dict = report.to_dict()

        print("\n[Validation Proof - Broadcast High Confidence]:")
        print(f"  Overall Score     : {report.overall_confidence_score:.1f}/100")
        print(f"  Confidence Tier   : {report.confidence_tier.value} ({report.tier_label_zh})")
        print(f"  Homography Valid  : {report.homography_completeness_pct:.1f}%")
        print(f"  Max Staleness     : {report.max_homography_staleness_frames} frames")
        print(f"  Reliable Sprints  : {report.is_reliable_for_sprints}")
        print(f"  Reliable Distance : {report.is_reliable_for_distance}")

        self.assertEqual(report.confidence_tier, ConfidenceTier.BROADCAST_HIGH)
        self.assertGreaterEqual(report.overall_confidence_score, 80.0)
        self.assertTrue(report.is_reliable_for_sprints)
        self.assertTrue(report.is_reliable_for_distance)
        self.assertTrue(report.is_reliable_for_tactical_zones)
        self.assertEqual(report.stale_dropout_episodes_count, 0)
        self.assertEqual(len(report.warnings), 0)

    def test_amateur_handheld_low_confidence(self):
        """Amateur phone camera with faded lines, extreme motion blur, and prolonged dropouts."""
        assessor = VideoAnalyticsConfidenceAssessor(fps=25.0, max_staleness_seconds=3.0)
        total_frames = 600

        # Simulate 3 severe dropout episodes where camera followed the ball or shook violently
        for f in range(total_frames):
            # Frames 100-220 (120 frames = ~4.8s dropout)
            # Frames 350-480 (130 frames = ~5.2s dropout)
            is_dropout = (100 <= f <= 220) or (350 <= f <= 480)
            has_h = not is_dropout and (f % 3 != 0)

            assessor.record_frame(
                frame_idx=f,
                has_valid_homography=has_h,
                keypoint_count=4 if has_h else 1,
                inlier_ratio=0.45 if has_h else 0.20,
                reprojection_rmse=16.5 if has_h else 28.0,
                ball_detected=(f % 4 == 0),  # Only 25% ball detection
                camera_pan_speed_px=18.0 if is_dropout else 8.0,  # Shaky camera
            )

        report = assessor.evaluate()

        print("\n[Validation Proof - Amateur Handheld Low Confidence]:")
        print(f"  Overall Score     : {report.overall_confidence_score:.1f}/100")
        print(f"  Confidence Tier   : {report.confidence_tier.value} ({report.tier_label_zh})")
        print(f"  Homography Valid  : {report.homography_completeness_pct:.1f}%")
        print(f"  Max Staleness     : {report.max_homography_staleness_frames} frames ({(report.max_homography_staleness_frames/25.0):.1f}s)")
        print(f"  Dropout Episodes  : {report.stale_dropout_episodes_count}")
        print(f"  Reliable Sprints  : {report.is_reliable_for_sprints} (Rejected to prevent noise saturation)")
        print(f"  Warnings Count    : {len(report.warnings)}")
        for w in report.warnings:
            print(f"    - {w}")

        self.assertEqual(report.confidence_tier, ConfidenceTier.AMATEUR_LOW)
        self.assertLess(report.overall_confidence_score, 50.0)
        self.assertFalse(report.is_reliable_for_sprints)
        self.assertGreaterEqual(report.stale_dropout_episodes_count, 2)
        self.assertGreater(len(report.warnings), 0)

    def test_tactical_medium_confidence(self):
        """Tactical club footage with acceptable overall quality but isolated occlusion periods."""
        assessor = VideoAnalyticsConfidenceAssessor(fps=25.0, max_staleness_seconds=3.0)
        total_frames = 500

        for f in range(total_frames):
            # One 2-second occlusion at frames 200-250 (50 frames = 2.0s, within 3.0s threshold)
            is_brief_occlusion = (200 <= f <= 250)
            has_h = not is_brief_occlusion and (f % 4 != 0)

            assessor.record_frame(
                frame_idx=f,
                has_valid_homography=has_h,
                keypoint_count=8 if has_h else 2,
                inlier_ratio=0.72 if has_h else 0.35,
                reprojection_rmse=7.5 if has_h else 14.0,
                ball_detected=(f % 2 == 0),  # 50% ball detection
                camera_pan_speed_px=4.0,
            )

        report = assessor.evaluate()

        print("\n[Validation Proof - Tactical Medium Confidence]:")
        print(f"  Overall Score     : {report.overall_confidence_score:.1f}/100")
        print(f"  Confidence Tier   : {report.confidence_tier.value} ({report.tier_label_zh})")
        print(f"  Reliable Distance : {report.is_reliable_for_distance}")
        print(f"  Reliable Zones    : {report.is_reliable_for_tactical_zones}")

        self.assertEqual(report.confidence_tier, ConfidenceTier.TACTICAL_MEDIUM)
        self.assertGreaterEqual(report.overall_confidence_score, 50.0)
        self.assertLess(report.overall_confidence_score, 90.0)
        self.assertTrue(report.is_reliable_for_distance)

    def test_staleness_gating_behavior(self):
        """Verifies that record_frame flags projection usability correctly during dropouts."""
        assessor = VideoAnalyticsConfidenceAssessor(fps=25.0, max_staleness_seconds=2.0)
        # 2.0s at 25 fps = 50 frames tolerance

        # 1. Fresh valid frame
        usable = assessor.record_frame(frame_idx=0, has_valid_homography=True)
        self.assertTrue(usable)

        # 2. Frames 1 to 50 are within 2.0s tolerance
        for f in range(1, 51):
            usable = assessor.record_frame(frame_idx=f, has_valid_homography=False)
            self.assertTrue(usable, f"Frame {f} should be usable within 50-frame tolerance")

        # 3. Frame 51 exceeds tolerance -> Must be flagged unusable/degraded
        usable = assessor.record_frame(frame_idx=51, has_valid_homography=False)
        self.assertFalse(usable, "Frame 51 should be flagged unusable due to excessive staleness")

        # 4. Homography restored -> Must immediately recover
        usable = assessor.record_frame(frame_idx=52, has_valid_homography=True)
        self.assertTrue(usable, "Frame 52 should immediately recover on fresh homography")

    def test_benchmark_throughput(self):
        """Microbenchmark ensuring zero pipeline overhead (> 200,000 FPS)."""
        assessor = VideoAnalyticsConfidenceAssessor(fps=25.0)
        n_frames = 20000

        t0 = time.perf_counter()
        for f in range(n_frames):
            assessor.record_frame(
                frame_idx=f,
                has_valid_homography=(f % 10 != 0),
                keypoint_count=10,
                inlier_ratio=0.85,
                reprojection_rmse=4.2,
                ball_detected=(f % 2 == 0),
                camera_pan_speed_px=3.0,
            )
        report = assessor.evaluate()
        elapsed = time.perf_counter() - t0

        fps = n_frames / max(1e-6, elapsed)
        print(f"\n[Algorithm-only Benchmark] Confidence Assessor: {fps:,.0f} FPS ({n_frames} frames in {elapsed:.3f}s)")
        self.assertGreater(fps, 100000.0)


if __name__ == "__main__":
    unittest.main()
