"""
test_gpu_hardware_verifier.py — Unit tests for GPUHardwareVerifier
"""

import unittest
from unittest.mock import MagicMock, patch
from server.pipeline.gpu_hardware_verifier import GPUHardwareVerifier


class TestGPUHardwareVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = GPUHardwareVerifier()

    def test_query_hardware_basic_structure(self):
        spec = self.verifier.query_hardware(force_refresh=True)
        self.assertIn("is_cuda_available", spec)
        self.assertIn("exact_gpu_model", spec)
        self.assertIn("architecture", spec)
        self.assertIn("system_ram_gb", spec)
        self.assertIn("recommended_yolo_batch", spec)
        self.assertIn("recommended_samurai_workers", spec)
        self.assertGreater(spec["system_ram_gb"], 0.0)

    @patch("server.pipeline.gpu_hardware_verifier.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("server.pipeline.gpu_hardware_verifier.subprocess.run")
    def test_nvidia_smi_parsing(self, mock_run, mock_which):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "NVIDIA GeForce RTX 4090, 535.129.03, 24564, GPU-12345\n"
        mock_run.return_value = mock_proc

        spec = self.verifier.query_hardware(force_refresh=True)
        self.assertEqual(spec["exact_gpu_model"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(spec["driver_version"], "535.129.03")
        self.assertEqual(spec["total_vram_gb"], 23.99)
        self.assertEqual(spec["recommended_yolo_batch"], 64)

    def test_format_diagnostic_report(self):
        report = self.verifier.format_diagnostic_report()
        self.assertIn("GPU HARDWARE & ACCELERATION SPECIFICATION REPORT", report)
        self.assertIn("Exact GPU Model", report)
        self.assertIn("Recommended Batch", report)


if __name__ == "__main__":
    unittest.main()
