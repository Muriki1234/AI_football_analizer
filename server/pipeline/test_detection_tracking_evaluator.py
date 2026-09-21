"""
test_detection_tracking_evaluator.py — Unit Tests for Detection & Tracking Metrics Engine

Verifies:
1. DetectionMetrics: Perfect match, partial match, small-target recall, crowded-player recall
2. TrackingMetrics: HOTA, DetA, AssA, IDF1, ID switches (IDSW), Fragmentations (Frag), MOTA
3. Degraded tracking: ID swap correctly increments IDSW, track loss increments Frag and drops IDF1/HOTA
"""

import unittest
from server.pipeline.detection_tracking_evaluator import (
    compute_iou,
    DetectionMetrics,
    TrackingMetrics,
)


class TestDetectionTrackingEvaluator(unittest.TestCase):
    def test_01_compute_iou_basic(self):
        boxA = [0.0, 0.0, 10.0, 10.0]
        boxB = [0.0, 0.0, 10.0, 10.0]
        self.assertAlmostEqual(compute_iou(boxA, boxB), 1.0)

        boxC = [5.0, 0.0, 15.0, 10.0]
        # Inter = 5 * 10 = 50. BoxA=100, BoxC=100, Union=150. IoU = 50/150 = 1/3
        self.assertAlmostEqual(compute_iou(boxA, boxC), 1.0 / 3.0)

        boxD = [20.0, 20.0, 30.0, 30.0]
        self.assertEqual(compute_iou(boxA, boxD), 0.0)

    def test_02_detection_metrics_perfect_and_partial(self):
        # Frame 0: 2 players, 1 small, 1 normal
        gt_frames = [
            {
                1: {"bbox": [10.0, 10.0, 30.0, 40.0]},  # w=20, h=30 -> small
                2: {"bbox": [50.0, 50.0, 90.0, 120.0]}, # w=40, h=70 -> normal
            },
            {
                1: {"bbox": [12.0, 12.0, 32.0, 42.0]},
                2: {"bbox": [52.0, 52.0, 92.0, 122.0]},
            }
        ]

        # Perfect prediction
        pred_perfect = [
            {
                101: {"bbox": [10.0, 10.0, 30.0, 40.0]},
                102: {"bbox": [50.0, 50.0, 90.0, 120.0]},
            },
            {
                101: {"bbox": [12.0, 12.0, 32.0, 42.0]},
                102: {"bbox": [52.0, 52.0, 92.0, 122.0]},
            }
        ]

        res = DetectionMetrics.evaluate(gt_frames, pred_perfect)
        self.assertEqual(res["precision"], 1.0)
        self.assertEqual(res["recall"], 1.0)
        self.assertEqual(res["f1_score"], 1.0)
        self.assertEqual(res["small_player_recall"], 1.0)
        self.assertEqual(res["tp"], 4)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["fn"], 0)

        # Partial prediction: misses the small player in frame 1, adds false positive
        pred_partial = [
            {
                101: {"bbox": [10.0, 10.0, 30.0, 40.0]},
                102: {"bbox": [50.0, 50.0, 90.0, 120.0]},
            },
            {
                102: {"bbox": [52.0, 52.0, 92.0, 122.0]},
                103: {"bbox": [200.0, 200.0, 240.0, 280.0]}, # FP
            }
        ]
        res_part = DetectionMetrics.evaluate(gt_frames, pred_partial)
        self.assertEqual(res_part["tp"], 3)
        self.assertEqual(res_part["fp"], 1)
        self.assertEqual(res_part["fn"], 1)
        self.assertEqual(res_part["precision"], 0.75)
        self.assertEqual(res_part["recall"], 0.75)
        self.assertEqual(res_part["small_player_recall"], 0.5)

    def test_03_crowded_player_recall(self):
        # 2 overlapping players in box
        gt_crowded = [
            {
                1: {"bbox": [100.0, 100.0, 150.0, 200.0]},
                2: {"bbox": [120.0, 100.0, 170.0, 200.0]}, # overlapping IoU > 0.3
            }
        ]
        pred_one = [
            {
                1: {"bbox": [100.0, 100.0, 150.0, 200.0]}, # detects only 1
            }
        ]
        res = DetectionMetrics.evaluate(gt_crowded, pred_one)
        self.assertEqual(res["total_gt_crowded"], 2)
        self.assertEqual(res["tp_crowded"], 1)
        self.assertEqual(res["crowded_player_recall"], 0.5)

    def test_04_tracking_metrics_perfect_tracking(self):
        gt = [
            {1: {"bbox": [10, 10, 30, 50]}, 2: {"bbox": [100, 100, 130, 150]}},
            {1: {"bbox": [12, 12, 32, 52]}, 2: {"bbox": [102, 102, 132, 152]}},
            {1: {"bbox": [14, 14, 34, 54]}, 2: {"bbox": [104, 104, 134, 154]}},
        ]
        pred = [
            {10: {"bbox": [10, 10, 30, 50]}, 20: {"bbox": [100, 100, 130, 150]}},
            {10: {"bbox": [12, 12, 32, 52]}, 20: {"bbox": [102, 102, 132, 152]}},
            {10: {"bbox": [14, 14, 34, 54]}, 20: {"bbox": [104, 104, 134, 154]}},
        ]
        metrics = TrackingMetrics.evaluate(gt, pred)
        self.assertEqual(metrics["HOTA"], 1.0)
        self.assertEqual(metrics["DetA"], 1.0)
        self.assertEqual(metrics["AssA"], 1.0)
        self.assertEqual(metrics["IDF1"], 1.0)
        self.assertEqual(metrics["MOTA"], 1.0)
        self.assertEqual(metrics["IDSW"], 0)
        self.assertEqual(metrics["Frag"], 0)

    def test_05_tracking_metrics_id_switch_and_fragmentation(self):
        gt = [
            {1: {"bbox": [10, 10, 30, 50]}},
            {1: {"bbox": [12, 12, 32, 52]}},
            {1: {"bbox": [14, 14, 34, 54]}},
            {1: {"bbox": [16, 16, 36, 56]}},
        ]
        # Frame 0: tracked as ID 10
        # Frame 1: missed (lost track -> fragmentation)
        # Frame 2: tracked as ID 20 (ID switch from 10 to 20!)
        # Frame 3: tracked as ID 20
        pred = [
            {10: {"bbox": [10, 10, 30, 50]}},
            {}, # lost
            {20: {"bbox": [14, 14, 34, 54]}},
            {20: {"bbox": [16, 16, 36, 56]}},
        ]
        metrics = TrackingMetrics.evaluate(gt, pred)
        self.assertEqual(metrics["IDSW"], 1, "Should count exactly 1 ID switch")
        self.assertGreaterEqual(metrics["Frag"], 1, "Should count at least 1 fragmentation")
        self.assertLess(metrics["HOTA"], 1.0)
        self.assertLess(metrics["IDF1"], 1.0)
        self.assertLess(metrics["MOTA"], 1.0)


if __name__ == "__main__":
    unittest.main()
