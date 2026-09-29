"""
test_motion_guided_crop_ball_detector.py
========================================
Unit tests for MotionGuidedCropBallDetector
"""

import unittest
from server.pipeline.motion_guided_crop_ball_detector import (
    MotionGuidedCropBallDetector,
    BallObservation,
)


class TestMotionGuidedCropBallDetector(unittest.TestCase):
    def setUp(self):
        self.detector = MotionGuidedCropBallDetector(crop_size=384, conf_threshold=0.35)

    def test_roi_prediction_follows_ball_velocity(self):
        # Frame 0: Ball at (500, 500)
        self.detector.update_observation(
            frame_idx=0,
            detected_box=(493.0, 493.0, 507.0, 507.0),
            confidence=0.85,
            is_crop=False,
        )
        self.assertEqual(self.detector.last_pos, (500.0, 500.0))

        # Frame 1: Ball moved to (520, 510) -> v ~ (20, 10)
        self.detector.update_observation(
            frame_idx=1,
            detected_box=(513.0, 503.0, 527.0, 517.0),
            confidence=0.82,
            is_crop=True,
        )

        # Predict Frame 2 ROI
        roi = self.detector.predict_next_roi(frame_w=1920, frame_h=1080)
        self.assertIsNotNone(roi)
        x1, y1, x2, y2 = roi
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Center of ROI should lead the ball in direction of motion (x > 520, y > 510)
        self.assertGreater(cx, 520.0)
        self.assertGreater(cy, 510.0)
        self.assertEqual(x2 - x1, 384)
        self.assertEqual(y2 - y1, 384)

    def test_crop_coordinate_mapping_accuracy(self):
        roi = (500, 300, 884, 684)  # 384x384 crop
        # Inside the crop, ball is at local (50, 60, 64, 74)
        crop_box = (50.0, 60.0, 64.0, 74.0)

        full_box = self.detector.map_crop_coords_to_full(crop_box, roi)
        expected = (550.0, 360.0, 564.0, 374.0)
        self.assertEqual(full_box, expected)

    def test_dropout_expansion_and_graceful_loss(self):
        self.detector.update_observation(
            frame_idx=0,
            detected_box=(500.0, 500.0, 514.0, 514.0),
            confidence=0.90,
        )

        # 1st failure: expands crop size
        roi_1 = self.detector.predict_next_roi(1920, 1080)
        self.detector.update_observation(frame_idx=1, detected_box=None, confidence=0.0)
        roi_2 = self.detector.predict_next_roi(1920, 1080)
        
        size_1 = roi_1[2] - roi_1[0]
        size_2 = roi_2[2] - roi_2[0]
        self.assertGreater(size_2, size_1, "Crop size must expand during dropouts to reacquire ball")

        # 4 consecutive failures: ball marked as lost -> returns None for global re-acquisition
        for fi in range(2, 6):
            self.detector.update_observation(frame_idx=fi, detected_box=None, confidence=0.0)
        
        roi_lost = self.detector.predict_next_roi(1920, 1080)
        self.assertIsNone(roi_lost, "Should return None when ball is lost, triggering global search fallback")


if __name__ == "__main__":
    unittest.main()
