"""
test_bytetrack_adaptive_compensator.py - Unit and Regression Tests for AdaptiveByteTracker
"""

import unittest
import numpy as np
import supervision as sv

from server.pipeline.bytetrack_adaptive_compensator import (
    AdaptiveByteTracker,
    interpolate_sparse_tracks,
)


class TestAdaptiveByteTracker(unittest.TestCase):

    def test_adaptive_bytetracker_initialization(self):
        tracker = AdaptiveByteTracker(
            base_fps=25.0,
            base_stride=3,
            track_activation_threshold=0.45,
            minimum_matching_threshold=0.80,
            base_lost_buffer_frames=15,
        )

        # step_buffer = max(4, round(15 / 3)) = 5
        self.assertEqual(tracker.step_buffer, 5)
        self.assertEqual(tracker.tracker.max_time_lost, 5)

    def test_adaptive_bytetracker_stride_update(self):
        tracker = AdaptiveByteTracker(base_fps=25.0, base_stride=1, base_lost_buffer_frames=16)
        self.assertEqual(tracker.step_buffer, 16)
        self.assertEqual(tracker.tracker.max_time_lost, 16)

        tracker.update_stride(4)
        # step_buffer = max(4, round(16 / 4)) = 4
        self.assertEqual(tracker.step_buffer, 4)
        self.assertEqual(tracker.tracker.max_time_lost, 4)

    def test_adaptive_bytetracker_synthetic_tracking(self):
        tracker = AdaptiveByteTracker(base_fps=25.0, base_stride=2)

        # 3 moving targets
        t0_boxes = np.array([
            [100.0, 100.0, 150.0, 200.0],
            [300.0, 100.0, 350.0, 200.0],
            [500.0, 100.0, 550.0, 200.0],
        ])
        ds0 = sv.Detections(
            xyxy=t0_boxes,
            confidence=np.array([0.9, 0.85, 0.95]),
            class_id=np.array([0, 0, 0]),
        )
        tracked0 = tracker.update_with_detections(ds0)
        self.assertEqual(len(tracked0), 3)

        # Small displacement at step 1
        t1_boxes = t0_boxes + np.array([[5.0, 2.0, 5.0, 2.0]])
        ds1 = sv.Detections(
            xyxy=t1_boxes,
            confidence=np.array([0.9, 0.85, 0.95]),
            class_id=np.array([0, 0, 0]),
        )
        tracked1 = tracker.update_with_detections(ds1)
        self.assertEqual(len(tracked1), 3)

        # Track IDs should remain consistent
        ids0 = sorted(int(d[4]) if len(d) > 4 else int(d[1]) for d in tracked0)
        ids1 = sorted(int(d[4]) if len(d) > 4 else int(d[1]) for d in tracked1)
        self.assertEqual(ids0, ids1)

    def test_interpolate_sparse_tracks(self):
        sparse_tracks = [{} for _ in range(10)]
        # Player 1 detected at frames 0, 3, 6
        sparse_tracks[0][1] = {"bbox": [10.0, 20.0, 30.0, 40.0], "class_id": 0}
        sparse_tracks[3][1] = {"bbox": [16.0, 26.0, 36.0, 46.0], "class_id": 0}
        sparse_tracks[6][1] = {"bbox": [22.0, 32.0, 42.0, 52.0], "class_id": 0}

        dense_tracks = interpolate_sparse_tracks(sparse_tracks, total_frames=10, max_gap_frames=5)

        self.assertEqual(len(dense_tracks), 10)
        # Frame 0 should match exactly
        self.assertEqual(dense_tracks[0][1]["bbox"], [10.0, 20.0, 30.0, 40.0])
        # Frame 1 should be interpolated (alpha = 1/3)
        self.assertEqual(dense_tracks[1][1]["bbox"], [12.0, 22.0, 32.0, 42.0])
        # Frame 2 should be interpolated (alpha = 2/3)
        self.assertEqual(dense_tracks[2][1]["bbox"], [14.0, 24.0, 34.0, 44.0])
        # Frame 3 should match exactly
        self.assertEqual(dense_tracks[3][1]["bbox"], [16.0, 26.0, 36.0, 46.0])
        # Frame 6 should match exactly
        self.assertEqual(dense_tracks[6][1]["bbox"], [22.0, 32.0, 42.0, 52.0])
        # Frame 7 should not be extrapolated
        self.assertNotIn(1, dense_tracks[7])


if __name__ == "__main__":
    unittest.main()
