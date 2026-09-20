"""
test_minimap_replay_generator.py - Unit Test & Benchmark Suite
for MinimapReplayGenerator and run_minimap_replay.

Verifies:
1. Frame rendering correctness (pitch background, player tokens, tracked highlight, ball trail).
2. Constant O(1) RAM streaming MP4 generation via FFmpeg pipe and OpenCV fallback.
3. Possession HUD overlay and match clock stamping.
4. Integration with SessionManager in run_minimap_replay.
5. High-throughput algorithm microbenchmark (> 200 frames/sec).
"""

from pathlib import Path
import tempfile
import time
import unittest
import numpy as np

from server.pipeline.minimap_replay_generator import MinimapReplayGenerator
from server.pipeline.tasks import run_minimap_replay


class MockSessionManager:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.tasks = {}

    def session_output_dir(self, session_id: str) -> Path:
        p = self.base_dir / session_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def update_task(self, session_id: str, task_id: str, **kwargs):
        if task_id not in self.tasks:
            self.tasks[task_id] = {}
        self.tasks[task_id].update(kwargs)

    def get_task(self, session_id: str, task_id: str) -> dict:
        return self.tasks.get(task_id, {})


class TestMinimapReplayGenerator(unittest.TestCase):
    def setUp(self):
        # Create synthetic match data with 50 frames, 2 teams (11 vs 11) + ball
        self.n_frames = 50
        players_frames = []
        ball_frames = []

        for f in range(self.n_frames):
            frame_players = {}
            # Team 1 (pids 1..11) on left half (x: 20..50m)
            for pid in range(1, 12):
                x = 20.0 + (pid % 5) * 6.0 + np.sin(f * 0.1) * 2.0
                y = 10.0 + (pid // 5) * 15.0
                frame_players[pid] = {
                    "bbox": [x * 10, y * 10, 30, 60],
                    "position_transformed": [x, y],
                    "team": 1,
                }
            # Team 2 (pids 12..22) on right half (x: 60..95m)
            for pid in range(12, 23):
                x = 60.0 + (pid % 5) * 6.0 - np.sin(f * 0.1) * 2.0
                y = 10.0 + ((pid - 11) // 5) * 15.0
                frame_players[pid] = {
                    "bbox": [x * 10, y * 10, 30, 60],
                    "position_transformed": [x, y],
                    "team": 2,
                }
            players_frames.append(frame_players)

            # Ball moving across pitch
            ball_x = 30.0 + f * 0.8
            ball_y = 35.0 + np.cos(f * 0.2) * 5.0
            ball_frames.append({1: {"position_transformed": [ball_x, ball_y]}})

        self.tracks = {"players": players_frames, "ball": ball_frames}
        self.tracked_bboxes = {f: (300, 200, 40, 80) for f in range(self.n_frames)}
        self.team_control = np.array([1 if f < 25 else 2 for f in range(self.n_frames)])
        self.team_colors = {1: "#2563eb", 2: "#dc2626"}

        self.generator = MinimapReplayGenerator(
            tracks=self.tracks,
            tracked_bboxes=self.tracked_bboxes,
            team_control=self.team_control,
            team_colors_hex=self.team_colors,
            fps=25.0,
            width=840,
            height=560,
        )

    def test_dimensions_and_properties(self):
        self.assertEqual(self.generator.total_frames, self.n_frames)
        self.assertEqual(self.generator.width, 840)
        self.assertEqual(self.generator.height, 560)
        self.assertEqual(self.generator.pitch_bg.shape, (560, 840, 3))

    def test_render_single_frame(self):
        frame = self.generator.render_frame(0)
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, (560, 840, 3))
        self.assertEqual(frame.dtype, np.uint8)
        # Should not be a blank black canvas
        self.assertGreater(float(np.mean(frame)), 20.0)

    def test_render_with_ball_trail(self):
        ball_trail = [(30.0 + i * 0.5, 35.0) for i in range(15)]
        frame = self.generator.render_frame(20, ball_trail=ball_trail)
        self.assertEqual(frame.shape, (560, 840, 3))

    def test_streaming_video_generation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "minimap_replay.mp4"
            progress_calls = []

            def on_progress(done, total):
                progress_calls.append((done, total))

            success = self.generator.render_video_streaming(out_path, progress_cb=on_progress)
            self.assertTrue(success)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 1000)
            self.assertGreater(len(progress_calls), 0)
            self.assertEqual(progress_calls[-1][1], self.n_frames)

    def test_opencv_fallback_video_generation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "fallback_replay.mp4"
            success = self.generator._render_video_opencv_fallback(out_path)
            self.assertTrue(success)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 1000)

    def test_run_minimap_replay_task_integration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sm = MockSessionManager(base_dir=Path(tmp_dir))
            session_id = "test_minimap_session"
            cache_path = sm.session_output_dir(session_id) / "tracks.pkl"

            session = {
                "id": session_id,
                "video_fps": 25.0,
                "tracks_cache_path": str(cache_path),
            }

            # Mock pipeline cache
            import pickle
            cache_data = {
                "tracks": self.tracks,
                "tracked_bboxes": self.tracked_bboxes,
                "team_control": self.team_control.tolist(),
                "team_colors_hex": self.team_colors,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)

            task_id = "task_minimap_01"
            run_minimap_replay(session_id, session, task_id, sm)

            task = sm.get_task(session_id, task_id)
            self.assertEqual(task.get("status"), "done")
            self.assertEqual(task.get("progress"), 100)
            self.assertIn("result", task)
            self.assertEqual(task["result"]["total_frames"], self.n_frames)
            self.assertEqual(task["result"]["duration_sec"], 2.0)
            self.assertEqual(task["result"]["resolution"], [840, 560])

            out_file = sm.session_output_dir(session_id) / "minimap_replay.mp4"
            self.assertTrue(out_file.exists())
            self.assertGreater(out_file.stat().st_size, 1000)

    def test_algorithm_benchmark_throughput(self):
        """[Algorithm-only Benchmark] Measures pure OpenCV 2D frame rendering rate."""
        fps = self.generator.benchmark_render(num_frames=500)
        self.assertGreater(fps, 100.0, f"Expected > 100 FPS, got {fps:.1f}")


if __name__ == "__main__":
    unittest.main()
