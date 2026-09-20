import unittest
import numpy as np
import cv2

from server.pipeline.analysis_core import (
    _linear_fill_keypoints,
    ViewTransformer,
    RobustPitchTransformer,
)


class TestPitchHomographyRobustness(unittest.TestCase):
    def test_linear_fill_no_ghost_extrapolation(self):
        sampled = {
            10: {0: [100.0, 100.0]},
            20: {0: [110.0, 105.0]},
            100: {0: [500.0, 500.0]},
        }
        filled = _linear_fill_keypoints(sampled, 120, max_gap=50)
        self.assertNotIn(0, filled[0])
        self.assertNotIn(0, filled[9])
        self.assertIn(0, filled[15])
        self.assertAlmostEqual(filled[15][0][0], 105.0)
        self.assertNotIn(0, filled[50])
        self.assertNotIn(0, filled[115])

    def test_linear_fill_single_detection_isolation(self):
        sampled = {5: {1: [42.0, 84.0]}}
        filled = _linear_fill_keypoints(sampled, 10)
        self.assertEqual(filled[5][1], [42.0, 84.0])
        for i in range(10):
            if i != 5:
                self.assertNotIn(1, filled[i])

    def test_ransac_homography_outlier_rejection(self):
        vt = ViewTransformer()
        H_true = np.array([
            [10.0, 2.0, -1000.0],
            [0.5, 8.0, 500.0],
            [0.0001, 0.0005, 1.0]
        ], dtype=np.float32)
        H_true = H_true / H_true[2, 2]

        dst_pts = np.array([
            [0.0, 1450.0], [2015.0, 1450.0], [0.0, 5550.0], [2015.0, 5550.0],
            [0.0, 2584.0], [550.0, 2584.0], [0.0, 4416.0], [550.0, 4416.0]
        ], dtype=np.float32)

        H_inv = np.linalg.inv(H_true)
        src_pts = cv2.perspectiveTransform(dst_pts.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        src_corrupted = np.vstack([src_pts, [[300.0, 400.0], [500.0, 200.0]]])
        dst_corrupted = np.vstack([dst_pts, [[6000.0, 3500.0], [12000.0, 7000.0]]])

        H_est = vt.compute_homography_ransac(src_corrupted, dst_corrupted)
        self.assertIsNotNone(H_est)
        test_p = np.array([[[400.0, 300.0]]], dtype=np.float32)
        p_true = cv2.perspectiveTransform(test_p, H_true)[0][0]
        p_est = cv2.perspectiveTransform(test_p, H_est)[0][0]
        error_m = np.linalg.norm(p_true - p_est) / 100.0
        self.assertLess(error_m, 0.5)

    def test_collinear_points_rejected(self):
        vt = ViewTransformer()
        p_collinear_src = np.array([[100, 100], [100, 200], [100, 300], [100, 400]])
        p_collinear_dst = np.array([[6000, 1000], [6000, 2000], [6000, 3000], [6000, 4000]])
        self.assertIsNone(vt.compute_homography_ransac(p_collinear_src, p_collinear_dst))

    def test_collapsed_homography_rejected(self):
        vt = ViewTransformer()
        H_broken = np.array([
            [14.7753, 64.9754, -19300.5449],
            [-2.3383, 25.9147, -1005.1526],
            [-0.0011, 0.0069, 1.0]
        ])
        self.assertFalse(vt.is_homography_valid(H_broken))

    def test_valid_perspective_accepted(self):
        vt = ViewTransformer()
        src_good = np.array([[200, 150], [1080, 150], [1200, 680], [80, 680]], dtype=np.float32)
        dst_good = np.array([[3000, 1000], [9000, 1000], [9000, 6000], [3000, 6000]], dtype=np.float32)
        H_good = vt.compute_homography_ransac(src_good, dst_good)
        self.assertIsNotNone(H_good)
        self.assertTrue(vt.is_homography_valid(H_good))

    def test_robust_pitch_transformer(self):
        H = np.eye(3, dtype=np.float64)
        transformer = RobustPitchTransformer(H)
        pts = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
        res = transformer.transform_points(pts)
        np.testing.assert_allclose(res, pts)


if __name__ == '__main__':
    unittest.main()
