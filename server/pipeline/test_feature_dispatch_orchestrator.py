from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Provide lightweight runtime shims for web framework modules if running outside Docker
if "fastapi" not in sys.modules:
    fastapi_mock = MagicMock()
    fastapi_mock.APIRouter = MagicMock
    fastapi_mock.Depends = lambda x: x
    fastapi_mock.HTTPException = Exception
    fastapi_mock.status.HTTP_501_NOT_IMPLEMENTED = 501
    fastapi_mock.Query = lambda default=None: default
    sys.modules["fastapi"] = fastapi_mock
    sys.modules["fastapi.responses"] = MagicMock()

if "pydantic" not in sys.modules:
    pydantic_mock = MagicMock()
    class DummyModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    pydantic_mock.BaseModel = DummyModel
    pydantic_mock.Field = lambda default=None, **kwargs: default
    sys.modules["pydantic"] = pydantic_mock

if "pydantic_settings" not in sys.modules:
    ps_mock = MagicMock()
    class DummySettings:
        API_KEY = ""
        output_root = Path(tempfile.gettempdir())
        SAMURAI_SCRIPT = ""
        SUPABASE_URL = ""
        SUPABASE_SERVICE_KEY = ""
        def __init__(self, **kwargs):
            pass
    ps_mock.BaseSettings = DummySettings
    ps_mock.SettingsConfigDict = lambda **kwargs: None
    sys.modules["pydantic_settings"] = ps_mock

if "supabase" not in sys.modules:
    sys.modules["supabase"] = MagicMock()

if "huggingface_hub" not in sys.modules:
    hf_mock = MagicMock()
    sys.modules["huggingface_hub"] = hf_mock
    sys.modules["huggingface_hub.utils"] = MagicMock()

from server.pipeline.feature_registry import (
    FEATURE_ALIASES,
    FEATURE_SPECS,
    build_feature_dispatch_table,
    get_all_artifact_files,
    get_cpu_features,
    get_feature_spec,
    get_feature_task,
    get_stats_only_features,
    resolve_canonical_feature,
)
from server.routes.analysis import (
    FEATURE_TASKS as ROUTES_FEATURE_TASKS,
    _ARTIFACT_FILES as ROUTES_ARTIFACT_FILES,
)
from server.handler import (
    _CPU_FEATURES as HANDLER_CPU_FEATURES,
    _STATS_ONLY_FEATURES as HANDLER_STATS_ONLY_FEATURES,
    _action_feature,
)


class TestFeatureDispatchOrchestrator(unittest.TestCase):
    def test_01_canonical_and_alias_resolution(self):
        """Verifies resolution of canonical names, aliases, and unknown features."""
        # Canonicals
        self.assertEqual(resolve_canonical_feature("heatmap"), "heatmap")
        self.assertEqual(resolve_canonical_feature("spatial_radar"), "spatial_radar")
        self.assertEqual(resolve_canonical_feature("voronoi"), "voronoi")
        self.assertEqual(resolve_canonical_feature("pass_network"), "pass_network")
        self.assertEqual(resolve_canonical_feature("vertical_crop"), "vertical_crop")
        self.assertEqual(resolve_canonical_feature("full_replay"), "full_replay")

        # Aliases
        self.assertEqual(resolve_canonical_feature("18_zone_radar"), "spatial_radar")
        self.assertEqual(resolve_canonical_feature("pitch_control"), "voronoi")
        self.assertEqual(resolve_canonical_feature("passes"), "pass_network")
        self.assertEqual(resolve_canonical_feature("vertical_crop_916"), "vertical_crop")
        self.assertEqual(resolve_canonical_feature("replay"), "full_replay")

        # Unknown
        self.assertIsNone(resolve_canonical_feature("non_existent_feature"))
        self.assertIsNone(resolve_canonical_feature(""))

    def test_02_dispatch_table_completeness(self):
        """Verifies every registered feature in routes and registry is callable."""
        dispatch = build_feature_dispatch_table()
        self.assertGreaterEqual(len(dispatch), len(FEATURE_SPECS))

        for name, spec in FEATURE_SPECS.items():
            self.assertIn(name, dispatch)
            fn = dispatch[name]
            self.assertTrue(callable(fn), f"Task {spec.task_fn_name} is not callable")

        # Check synchronization with routes/analysis.py
        for name in FEATURE_SPECS:
            self.assertIn(name, ROUTES_FEATURE_TASKS)

    def test_03_cpu_and_stats_worker_classification(self):
        """Verifies CPU workers accept analytical tasks and reject heavy GPU video tasks."""
        cpu_features = get_cpu_features()
        stats_features = get_stats_only_features()

        # Analytical stats features must be CPU-supported and video-free
        for feat in ["heatmap", "speed_chart", "possession", "spatial_radar", "voronoi", "pass_network"]:
            self.assertIn(feat, cpu_features)
            self.assertIn(feat, stats_features)

        # Video encoding features require GPU or full video download
        self.assertNotIn("vertical_crop", cpu_features)
        self.assertNotIn("full_replay", cpu_features)
        self.assertNotIn("vertical_crop", stats_features)
        self.assertNotIn("full_replay", stats_features)

        # Handler sync check
        self.assertEqual(cpu_features, HANDLER_CPU_FEATURES)
        self.assertEqual(stats_features, HANDLER_STATS_ONLY_FEATURES)

    def test_04_artifact_registry_sync(self):
        """Verifies artifact registry includes all outputs from newly integrated tasks."""
        artifacts = get_all_artifact_files()
        expected = {
            "spatial_radar.png",
            "voronoi_pitch_control.png",
            "pass_network.png",
            "vertical_crop_916.mp4",
            "full_replay.mp4",
            "heatmap.png",
            "speed_chart.png",
            "possession_chart.png",
            "sprint_analysis.png",
            "defensive_line.png",
            "ai_summary.md",
        }
        for exp in expected:
            self.assertIn(exp, artifacts)
            self.assertIn(exp, ROUTES_ARTIFACT_FILES)

    def test_05_handler_action_feature_dispatch(self):
        """Verifies RunPod Serverless _action_feature dispatches properly and handles aliases."""
        mock_sm = MagicMock()
        mock_sm.create_task.return_value = "task_test_123"
        mock_sm.get_task.return_value = {"task_id": "task_test_123", "status": "done"}

        mock_session = {"session_id": "sess_001", "tracks_cache_path": "dummy"}

        # Dispatch with alias "18_zone_radar"
        payload = {"feature": "18_zone_radar"}
        with patch("server.pipeline.tasks.run_spatial_zone_radar") as mock_radar:
            res = _action_feature("sess_001", mock_session, payload, mock_sm)
            self.assertTrue(res.get("ok"))
            mock_radar.assert_called_once_with("sess_001", mock_session, "task_test_123", mock_sm)

        # Dispatch with unknown feature returns clean error
        bad_res = _action_feature("sess_001", mock_session, {"feature": "unknown_task"}, mock_sm)
        self.assertIn("error", bad_res)
        self.assertIn("unknown feature", bad_res["error"])

    def test_06_microbenchmark_feature_dispatch(self):
        """[Component Benchmark] Evaluates dispatch table lookup and canonical resolution throughput."""
        start = time.perf_counter()
        iterations = 50000
        test_queries = ["heatmap", "18_zone_radar", "passes", "voronoi", "replay", "vertical_crop_916"]
        
        for i in range(iterations):
            q = test_queries[i % len(test_queries)]
            spec = get_feature_spec(q)
            self.assertIsNotNone(spec)

        elapsed = time.perf_counter() - start
        lookups_per_sec = iterations / max(1e-6, elapsed)
        print(f"\n[Component Benchmark: Feature Dispatch Orchestrator]")
        print(f"Processed {iterations} feature resolutions in {elapsed * 1000:.2f} ms ({lookups_per_sec:,.0f} lookups/sec)")
        self.assertGreater(lookups_per_sec, 50000)




    def test_07_synthetic_feature_tasks_execution(self):
        """[Component Benchmark] Verifies that newly integrated feature tasks run and produce artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            mock_sm = MagicMock()
            mock_sm.session_output_dir.return_value = out_dir

            # Build synthetic tracks
            synthetic_tracks = {
                "players": [
                    {
                        1: {"position_minimap": (52.5, 34.0), "team": 1},
                        2: {"position_minimap": (25.0, 15.0), "team": 1},
                        3: {"position_minimap": (80.0, 45.0), "team": 2},
                    }
                    for _ in range(10)
                ],
                "ball": [
                    {1: {"bbox": [500, 300, 520, 320]}}
                    for _ in range(10)
                ]
            }
            synthetic_tracked_bboxes = {
                i: (500, 300, 40, 80) for i in range(10)
            }
            synthetic_cache = {
                "tracks": synthetic_tracks,
                "tracked_bboxes": synthetic_tracked_bboxes
            }

            session = {
                "session_id": "test_e2e_sess",
                "tracks_cache_path": str(out_dir / "tracks.pkl"),
                "video_path": None,
            }

            from server.pipeline import tasks as pipeline_tasks

            with patch("server.pipeline.tasks._load_cache", return_value=synthetic_cache), \
                 patch("server.pipeline.tasks._finish_task") as mock_finish:

                # 1. run_spatial_zone_radar
                pipeline_tasks.run_spatial_zone_radar("test_e2e_sess", session, "task_radar", mock_sm)
                self.assertTrue(mock_finish.called)
                self.assertTrue((out_dir / "spatial_radar.png").exists())
                mock_finish.reset_mock()

                # 2. run_pitch_control_voronoi
                pipeline_tasks.run_pitch_control_voronoi("test_e2e_sess", session, "task_voronoi", mock_sm)
                self.assertTrue(mock_finish.called)
                self.assertTrue((out_dir / "voronoi_pitch_control.png").exists())
                mock_finish.reset_mock()

                # 3. run_pass_network
                pipeline_tasks.run_pass_network("test_e2e_sess", session, "task_passes", mock_sm)
                self.assertTrue(mock_finish.called)
                self.assertTrue((out_dir / "pass_network.png").exists())
                mock_finish.reset_mock()

                # 4. run_vertical_crop
                pipeline_tasks.run_vertical_crop("test_e2e_sess", session, "task_crop", mock_sm)
                self.assertTrue(mock_finish.called)
                self.assertTrue((out_dir / "vertical_crop_manifest.json").exists())
                mock_finish.reset_mock()

                # 5. run_full_replay (with mock gemini_video present)
                (out_dir / "gemini_video.mp4").write_bytes(b"dummy_video_bytes")
                pipeline_tasks.run_full_replay("test_e2e_sess", session, "task_replay", mock_sm)
                self.assertTrue(mock_finish.called)
                self.assertTrue((out_dir / "full_replay.mp4").exists())


if __name__ == "__main__":
    unittest.main()
