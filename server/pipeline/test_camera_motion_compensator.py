"""
test_camera_motion_compensator.py
=================================
Unit tests for CameraMotionCompensator
"""

import unittest
import numpy as np
from server.pipeline.camera_motion_compensator import CameraMotionCompensator


class TestCameraMotionCompensator(unittest.TestCase):
    def setUp(self):
        self.gmc = CameraMotionCompensator()

    def test_synthetic_translation_recovery(self):
        # Create synthetic textured background
        np.random.seed(42)
        h, w = 300, 400
        bg = np.random.randint(50, 200, (h, w), dtype=np.uint8)

        # Frame 1: Original background
        frame1 = bg.copy()

        # Frame 2: Shifted by dx = 15.0, dy = -8.0
        dx, dy = 15.0, -8.0
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        import cv2
        frame2 = cv2.warpAffine(frame1, M, (w, h))

        # First frame: warmup
        A1, v1 = self.gmc.estimate_camera_motion(frame1)
        self.assertFalse(v1)

        # Second frame: estimate motion
        A2, v2 = self.gmc.estimate_camera_motion(frame2)
        self.assertTrue(v2)
        
        # Check translation parameters
        recovered_dx = A2[0, 2]
        recovered_dy = A2[1, 2]
        self.assertAlmostEqual(recovered_dx, dx, delta=1.5)
        self.assertAlmostEqual(recovered_dy, dy, delta=1.5)

    def test_homography_propagation_invariance(self):
        # If camera translates by dx, dy, point P in real world has pitch coords X_pitch
        # H_prev @ x_img1 = X_pitch
        # Under camera translation, x_img2 = A @ x_img1
        # Propagated H_curr @ x_img2 should equal X_pitch!
        H_prev = np.array([
            [1.2, 0.1, 50.0],
            [-0.05, 1.1, 100.0],
            [0.0001, 0.0002, 1.0]
        ], dtype=np.float64)

        A = np.array([
            [1.0, 0.0, 25.0],
            [0.0, 1.0, 10.0]
        ], dtype=np.float32)

        x_img1 = np.array([300.0, 200.0, 1.0], dtype=np.float64)
        X_pitch_expected = np.dot(H_prev, x_img1)
        X_pitch_expected /= X_pitch_expected[2]

        # Shifted image point
        x_img2 = np.array([325.0, 210.0, 1.0], dtype=np.float64)

        H_curr = CameraMotionCompensator.propagate_homography(H_prev, A)
        X_pitch_actual = np.dot(H_curr, x_img2)
        X_pitch_actual /= X_pitch_actual[2]

        np.testing.assert_allclose(X_pitch_actual, X_pitch_expected, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
