"""
test_e2e_video_detection_profiler.py — Unit tests for E2E Video Detection Profiler

Verifies:
1. StageTimer timing accumulation and stage percentage properties
2. Vectorized track interpolation logic
3. End-to-end profiler execution and contract structure
"""

import os
import unittest
from server.pipeline.e2e_video_detection_profiler import (
    StageTimer,
    E2EVideoDetectionProfiler,
)


class TestE2EVideoDetectionProfiler(unittest.TestCase):
    def test_01_stage_timer_accumulation(self):
        timer = StageTimer()
        timer.add("video_decode", 0.010)
        timer.add("video_decode", 0.015)
        timer.add("model_inference", 0.050)

        self.assertAlmostEqual(timer.timings["video_decode"], 0.025)
        self.assertEqual(timer.counts["video_decode"], 2)
        self.assertAlmostEqual(timer.timings["model_inference"], 0.050)
        self.assertEqual(timer.counts["model_inference"], 1)

    def test_02_track_interpolation(self):
        profiler = E2EVideoDetectionProfiler.__new__(E2EVideoDetectionProfiler)

        # 5 frames: player 1 detected at frame 0 and frame 4, missing at 1, 2, 3
        player_tracks = [
            {1: {"bbox": [10.0, 10.0, 20.0, 30.0]}},
            {},
            {},
            {},
            {1: {"bbox": [50.0, 50.0, 60.0, 70.0]}},
        ]

        profiler._interpolate_player_tracks(player_tracks, total_frames=5)

        # Frame 2 is halfway: x1 should be 30.0, y1 should be 30.0
        self.assertIn(1, player_tracks[2])
        interp_box = player_tracks[2][1]["bbox"]
        self.assertAlmostEqual(interp_box[0], 30.0)
        self.assertAlmostEqual(interp_box[1], 30.0)
        self.assertAlmostEqual(interp_box[2], 40.0)
        self.assertAlmostEqual(interp_box[3], 50.0)
        self.assertTrue(player_tracks[2][1].get("interpolated", False))

    def test_03_profiler_contract_smoke(self):
        video_path = "backend/uploads/fe7f8619b7ea_test_17.mp4"
        if not os.path.exists(video_path):
            self.skipTest(f"Test video {video_path} not found")
        profiler = E2EVideoDetectionProfiler(
            video_path=video_path,
            detector_weights="backend/weights/football/best.pt",
            keypoint_weights="backend/weights/keypoints/best.pt",
            device="cpu",
        )
        res = profiler.run_profile(
            max_frames=10,
            imgsz=640,
            stride=3,
            enable_keypoints=False,
        )
        self.assertIn("total_video_duration_s", res)
        self.assertIn("total_frames", res)
        self.assertIn("effective_e2e_fps", res)
        self.assertIn("realtime_factor_rtf", res)
        self.assertIn("stage_breakdown", res)
        self.assertIn("primary_bottleneck", res)
        self.assertEqual(res["total_frames"], 10)
        self.assertGreater(res["effective_e2e_fps"], 0.0)
        self.assertGreater(res["realtime_factor_rtf"], 0.0)


if __name__ == "__main__":
    unittest.main()
