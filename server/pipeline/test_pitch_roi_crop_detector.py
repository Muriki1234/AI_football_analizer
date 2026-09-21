"""
test_pitch_roi_crop_detector.py - Unit Tests for PitchROICropDetector
"""

import unittest
import numpy as np

from server.pipeline.pitch_roi_crop_detector import PitchROICropDetector


class TestPitchROICropDetector(unittest.TestCase):

    def test_pitch_roi_crop_bounds_and_alignment(self):
        detector = PitchROICropDetector(min_crop_height=480, pad_px=32, stride_alignment=32)

        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Pitch keypoints placed between y=200 and y=900
        kpts = {
            0: (100.0, 250.0),
            1: (1800.0, 250.0),
            2: (200.0, 850.0),
            3: (1700.0, 850.0),
        }

        y_min, y_max = detector.estimate_pitch_vertical_bounds(dummy_frame, keypoints=kpts)

        self.assertTrue(0 <= y_min < y_max <= 1080)
        self.assertGreaterEqual((y_max - y_min), 480)
        self.assertEqual((y_max - y_min) % 32, 0)
        # Should crop out top stands (y < 200)
        self.assertGreaterEqual(y_min, 250 - 32 - 32)

    def test_pitch_roi_crop_and_restore_boxes(self):
        detector = PitchROICropDetector(min_crop_height=480, pad_px=32, stride_alignment=32)

        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        y_min, y_max = 200, 840

        cropped = detector.crop_frame(frame, y_min, y_max)
        self.assertEqual(cropped.shape, (640, 1920, 3))

        # Detected box in cropped coordinates
        boxes_crop = np.array([
            [100.0, 50.0, 150.0, 150.0],
            [300.0, 200.0, 350.0, 300.0],
        ])

        restored = detector.restore_boxes(boxes_crop, y_offset=y_min)

        self.assertEqual(restored.shape, (2, 4))
        # y coordinates should be shifted by y_min (200)
        self.assertEqual(restored[0, 1], 250.0)
        self.assertEqual(restored[0, 3], 350.0)
        self.assertEqual(restored[1, 1], 400.0)
        self.assertEqual(restored[1, 3], 500.0)
        # x coordinates remain untouched
        self.assertEqual(restored[0, 0], 100.0)
        self.assertEqual(restored[0, 2], 150.0)


if __name__ == "__main__":
    unittest.main()
