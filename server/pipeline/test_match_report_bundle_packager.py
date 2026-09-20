"""
test_match_report_bundle_packager.py - Unit Test & Algorithm Benchmark Suite
for MatchReportBundlePackager & Checksum Manifest Generator.
"""

import hashlib
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
import zipfile

from server.pipeline.match_report_bundle_packager import (
    ArtifactManifestEntry,
    MatchDossierManifest,
    MatchReportBundlePackager,
)


class TestMatchReportBundlePackager(unittest.TestCase):

    def setUp(self):
        self.packager = MatchReportBundlePackager()

    def test_compute_sha256(self):
        temp_dir = tempfile.mkdtemp()
        try:
            fpath = Path(temp_dir) / "test.txt"
            content = b"Antigravity Match Analytics SOTA"
            with open(fpath, "wb") as f:
                f.write(content)

            expected_hash = hashlib.sha256(content).hexdigest()
            computed_hash = self.packager.compute_sha256(fpath)
            self.assertEqual(computed_hash, expected_hash)
            self.assertEqual(len(computed_hash), 64)
        finally:
            shutil.rmtree(temp_dir)

    def test_classify_artifacts(self):
        cat, mime, desc = self.packager.classify_artifact("shot_map.png")
        self.assertEqual(cat, "tactical_visual")
        self.assertEqual(mime, "image/png")

        cat, mime, desc = self.packager.classify_artifact("speed_telemetry.json")
        self.assertEqual(cat, "kinematics")
        self.assertEqual(mime, "application/json")

        cat, mime, desc = self.packager.classify_artifact("minimap_replay.mp4")
        self.assertEqual(cat, "video_replay")
        self.assertEqual(mime, "video/mp4")

        cat, mime, desc = self.packager.classify_artifact("ai_summary.md")
        self.assertEqual(cat, "report")

    def test_build_manifest(self):
        temp_dir = tempfile.mkdtemp()
        try:
            out_dir = Path(temp_dir) / "session_123"
            out_dir.mkdir(parents=True)

            (out_dir / "shot_map.png").write_bytes(b"FAKE_PNG_BYTES_123")
            (out_dir / "ai_summary.md").write_text("# Tactical AI Coach Report", encoding="utf-8")
            (out_dir / "pressing_ppda_summary.json").write_text(
                json.dumps({"team1": {"ppda": 7.8}, "team2": {"ppda": 14.2}}), encoding="utf-8"
            )

            session = {
                "total_frames": 500,
                "video_fps": 25.0,
                "video_resolution": "1920x1080",
            }
            cache_data = {
                "team_control": [1, 1, 1, 2, 2],
            }

            manifest = self.packager.build_manifest("session_123", out_dir, session, cache_data)
            self.assertEqual(manifest.session_id, "session_123")
            self.assertEqual(manifest.artifacts_count, 3)
            self.assertGreater(manifest.total_size_bytes, 0)
            self.assertEqual(manifest.match_metadata["duration_sec"], 20.0)
            self.assertEqual(manifest.executive_kpis["possession_pct"]["team1"], 60.0)
            self.assertEqual(manifest.executive_kpis["pressing_ppda"]["team1_ppda"], 7.8)

            # Check individual artifacts
            names = [a["name"] for a in manifest.artifacts]
            self.assertIn("shot_map.png", names)
            self.assertIn("ai_summary.md", names)
            self.assertIn("pressing_ppda_summary.json", names)
        finally:
            shutil.rmtree(temp_dir)

    def test_package_bundle_zip_integrity(self):
        temp_dir = tempfile.mkdtemp()
        try:
            out_dir = Path(temp_dir) / "session_456"
            out_dir.mkdir(parents=True)

            (out_dir / "heatmap.png").write_bytes(b"PNG_HEATMAP_DATA")
            (out_dir / "replay.mp4").write_bytes(b"H264_STREAM_DATA")
            (out_dir / "report.md").write_text("Detailed Coach Report", encoding="utf-8")

            session = {"total_frames": 250, "video_fps": 25.0}
            zip_path, manifest = self.packager.package_bundle("session_456", out_dir, session)

            self.assertTrue(zip_path.exists())
            self.assertTrue((out_dir / "manifest.json").exists())

            # Test ZIP CRC32 integrity
            with zipfile.ZipFile(zip_path, "r") as zf:
                bad_file = zf.testzip()
                self.assertIsNone(bad_file, f"Zip integrity error on: {bad_file}")

                namelist = zf.namelist()
                self.assertIn("manifest.json", namelist)
                self.assertIn("heatmap.png", namelist)
                self.assertIn("replay.mp4", namelist)
                self.assertIn("report.md", namelist)

                # Verify compression mode
                mp4_info = zf.getinfo("replay.mp4")
                self.assertEqual(mp4_info.compress_type, zipfile.ZIP_STORED)

                txt_info = zf.getinfo("report.md")
                self.assertEqual(txt_info.compress_type, zipfile.ZIP_DEFLATED)
        finally:
            shutil.rmtree(temp_dir)

    def test_algorithm_benchmark(self):
        temp_dir = tempfile.mkdtemp()
        try:
            out_dir = Path(temp_dir) / "bench_session"
            out_dir.mkdir(parents=True)

            # Generate 20 files of 500KB each (10 MB total)
            chunk = b"A" * (500 * 1024)
            for i in range(20):
                (out_dir / f"artifact_{i:02d}.json").write_bytes(chunk)

            session = {"total_frames": 1000, "video_fps": 25.0}

            start_t = time.perf_counter()
            zip_path, manifest = self.packager.package_bundle("bench_session", out_dir, session)
            elapsed = time.perf_counter() - start_t

            mb_processed = 10.0
            throughput_mb_s = mb_processed / elapsed
            print(f"\n[Algorithm-only Benchmark] MatchReportBundlePackager: {throughput_mb_s:,.1f} MB/s ({mb_processed} MB in {elapsed:.3f}s)")
            self.assertGreater(throughput_mb_s, 20.0)
        finally:
            shutil.rmtree(temp_dir)



    def test_run_match_bundle_task_integration(self):
        from server.pipeline.tasks import run_match_bundle
        from unittest.mock import MagicMock

        temp_dir = tempfile.mkdtemp()
        try:
            session_id = "test_bundle_session"
            task_id = "task_bundle_001"

            out_dir = Path(temp_dir) / session_id
            out_dir.mkdir(parents=True, exist_ok=True)

            (out_dir / "heatmap.png").write_bytes(b"PNG_DATA_1")
            (out_dir / "shot_map.png").write_bytes(b"PNG_DATA_2")
            (out_dir / "speed_telemetry.json").write_text("{}", encoding="utf-8")

            sm = MagicMock()
            sm.session_output_dir.return_value = out_dir

            session = {
                "total_frames": 500,
                "video_fps": 25.0,
            }

            run_match_bundle(session_id, session, task_id, sm)

            zip_file = out_dir / "match_analysis_bundle.zip"
            manifest_file = out_dir / "manifest.json"
            self.assertTrue(zip_file.exists())
            self.assertTrue(manifest_file.exists())

            with open(manifest_file, "r") as f:
                manifest_data = json.load(f)
            self.assertEqual(manifest_data["session_id"], session_id)
            self.assertGreaterEqual(manifest_data["artifacts_count"], 3)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
