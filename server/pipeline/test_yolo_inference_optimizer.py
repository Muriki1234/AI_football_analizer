"""
test_yolo_inference_optimizer.py - Unit Tests for YOLOInferenceOptimizer
"""

import os
import unittest
import numpy as np

from server.pipeline.yolo_inference_optimizer import YOLOInferenceOptimizer

WEIGHTS_PATH = "backend/weights/football/best.pt"


class TestYOLOInferenceOptimizer(unittest.TestCase):

    def test_yolo_inference_optimizer_initialization(self):
        if not os.path.exists(WEIGHTS_PATH):
            self.skipTest(f"Weights {WEIGHTS_PATH} not found")

        optimizer = YOLOInferenceOptimizer(
            model_path=WEIGHTS_PATH,
            conf=0.25,
            target_imgsz=640,
            batch_size=2,
        )

        specs = optimizer.get_hardware_specs()
        self.assertEqual(specs["imgsz"], 640)
        self.assertEqual(specs["batch_size"], 2)
        self.assertIn("device", specs)
        self.assertIsNotNone(optimizer.model)

    def test_yolo_inference_optimizer_predict_batch(self):
        if not os.path.exists(WEIGHTS_PATH):
            self.skipTest(f"Weights {WEIGHTS_PATH} not found")

        optimizer = YOLOInferenceOptimizer(
            model_path=WEIGHTS_PATH,
            conf=0.25,
            target_imgsz=640,
            batch_size=2,
        )

        # 2 synthetic dummy frames
        f1 = np.zeros((480, 640, 3), dtype=np.uint8)
        f2 = np.zeros((480, 640, 3), dtype=np.uint8)

        results = optimizer.predict_batch([f1, f2])
        self.assertEqual(len(results), 2)
        self.assertTrue(hasattr(results[0], "boxes"))
        self.assertTrue(hasattr(results[1], "boxes"))


if __name__ == "__main__":
    unittest.main()
