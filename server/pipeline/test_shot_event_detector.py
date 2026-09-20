"""
test_shot_event_detector.py - Unit Test & Algorithm Benchmark Suite
for ShotEventDetector & Freeze-Frame Expected Goals (xG) Physics Evaluator.
"""

import math
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

from server.pipeline.shot_event_detector import ShotEvent, ShotEventDetector


class TestShotEventDetector(unittest.TestCase):

    def setUp(self):
        self.detector = ShotEventDetector(fps=25.0)

    def test_coordinate_normalization(self):
        # 1. Corner-based coordinates [0, 105] x [0, 68]
        corner_ball = {0: (100.0, 34.0), 1: (102.0, 34.0)}
        corner_players = {0: {1: (95.0, 34.0), 2: (103.0, 34.0)}}
        norm_b, norm_p = self.detector.normalize_coordinates(corner_ball, corner_players)

        # Center should be shifted by (52.5, 34.0)
        self.assertAlmostEqual(norm_b[0][0], 100.0 - 52.5, places=2)
        self.assertAlmostEqual(norm_b[0][1], 0.0, places=2)
        self.assertAlmostEqual(norm_p[0][1][0], 95.0 - 52.5, places=2)

        # 2. Centimeter coordinates (x > 200)
        cm_ball = {0: (10000.0, 3400.0)}
        cm_players = {0: {1: (9500.0, 3400.0)}}
        norm_b2, norm_p2 = self.detector.normalize_coordinates(cm_ball, cm_players)
        self.assertAlmostEqual(norm_b2[0][0], 100.0 - 52.5, places=2)

    def test_subtended_goal_angle(self):
        # Central shot at penalty spot (11m from right goal line x=52.5)
        # Shot location: x = 52.5 - 11.0 = 41.5, y = 0.0
        angle_rad, angle_deg = ShotEventDetector.compute_subtended_goal_angle(
            shot_xy=(41.5, 0.0),
            goal_x=52.5,
            post1_y=-3.66,
            post2_y=3.66,
        )
        # tan(theta/2) = 3.66 / 11.0 = 0.3327 => theta/2 = 18.40 deg => theta = 36.8 deg
        self.assertAlmostEqual(angle_deg, 36.8, delta=0.5)
        self.assertAlmostEqual(angle_rad, math.radians(36.8), delta=0.02)

        # Tight angle shot near the byline (x=50.0, y=25.0)
        tight_rad, tight_deg = ShotEventDetector.compute_subtended_goal_angle(
            shot_xy=(50.0, 25.0),
            goal_x=52.5,
            post1_y=-3.66,
            post2_y=3.66,
        )
        self.assertLess(tight_deg, 10.0)

    def test_point_in_triangle_and_shot_cone(self):
        # Shooter at (35.0, 0.0), goal at x=52.5, posts at y=-3.66, +3.66
        # Defender 1 inside the cone: (45.0, 0.0)
        # Defender 2 inside the cone: (48.0, 1.0)
        # Defender 3 outside the cone (wide): (45.0, 8.0)
        # Defender 4 behind the shooter: (30.0, 0.0)
        defenders = [(45.0, 0.0), (48.0, 1.0), (45.0, 8.0), (30.0, 0.0)]
        in_cone, blockers = self.detector.count_defenders_in_shot_cone(
            shot_xy=(35.0, 0.0),
            target_goal_x=52.5,
            post1_y=-3.66,
            post2_y=3.66,
            defenders_xy=defenders,
        )
        self.assertEqual(in_cone, 2)
        # Defender 1 at (45.0, 0.0) is directly in line of sight (blocker)
        self.assertGreaterEqual(blockers, 1)

    def test_calibrated_xg_physics_model(self):
        # 1. Penalty spot: 11m, angle ~0.64 rad, 0 defenders, GK on line
        penalty_rad = math.radians(36.8)
        penalty_xg = self.detector.compute_xg(
            distance_m=11.0,
            angle_rad=penalty_rad,
            defenders_in_cone=0,
            blockers_in_lane=0,
            gk_positioning_score=0.5,
            shot_speed_mps=25.0,
        )
        # Historical penalty benchmark is ~0.76 (0.70 - 0.82)
        self.assertGreaterEqual(penalty_xg, 0.72)
        self.assertLessEqual(penalty_xg, 0.82)

        # 2. Central 6-yard tap-in: 4m, angle ~1.2 rad, open goal
        tapin_rad = math.radians(70.0)
        tapin_xg = self.detector.compute_xg(
            distance_m=4.0,
            angle_rad=tapin_rad,
            defenders_in_cone=0,
            blockers_in_lane=0,
            gk_positioning_score=0.0,
            shot_speed_mps=15.0,
        )
        self.assertGreaterEqual(tapin_xg, 0.85)

        # 3. Contested edge-of-box shot: 18m, angle ~0.4 rad, 2 defenders, 1 blocker
        box_edge_rad = math.radians(23.0)
        contested_xg = self.detector.compute_xg(
            distance_m=18.0,
            angle_rad=box_edge_rad,
            defenders_in_cone=2,
            blockers_in_lane=1,
            gk_positioning_score=1.0,
            shot_speed_mps=20.0,
        )
        # Real-world benchmark: ~0.08 - 0.14
        self.assertGreaterEqual(contested_xg, 0.05)
        self.assertLessEqual(contested_xg, 0.20)

        # 4. Long-range 30m contested screamer
        long_rad = math.radians(14.0)
        long_xg = self.detector.compute_xg(
            distance_m=30.0,
            angle_rad=long_rad,
            defenders_in_cone=3,
            blockers_in_lane=1,
            gk_positioning_score=1.0,
            shot_speed_mps=28.0,
        )
        self.assertLess(long_xg, 0.05)

    def test_detect_shots_goal_outcome(self):
        # Construct synthetic trajectory:
        # Player 9 on Team 1 (attacking right) shoots from (35.0, 0.0) at frame 10
        # Ball travels fast towards (52.5, 0.0) and crosses goal line between posts
        ball_traj = {}
        player_traj = {}
        teams = {9: 1, 10: 1, 1: 2, 4: 2, 5: 2}

        # Frames 0-9: ball dribbled near player 9 at (34.0, 0.0)
        for f in range(10):
            ball_traj[f] = (34.0 + f * 0.1, 0.0)
            player_traj[f] = {
                9: (34.0 + f * 0.1, 0.0),
                1: (51.0, 0.0),  # Opponent GK near goal
                4: (44.0, 2.0),  # Opponent defender
            }

        # Frame 10: Player 9 strikes ball at (35.0, 0.0)
        # Velocity: travels 17.5m in 15 frames (0.6s) => ~29 m/s (~105 km/h)
        for f in range(10, 26):
            frac = (f - 10) / 15.0
            bx = 35.0 + frac * 18.0  # reaches 53.0m (past 52.5m goal line)
            by = 0.0 + frac * 1.0    # crosses at y = 1.0 (well within ±3.66m)
            ball_traj[f] = (bx, by)
            player_traj[f] = {
                9: (35.0, 0.0),
                1: (51.0, -1.5),  # GK dives to wrong side
                4: (44.0, 2.0),
            }

        shots = self.detector.detect_shots(
            ball_trajectory=ball_traj,
            player_trajectories=player_traj,
            teams=teams,
            attacking_directions={1: "right", 2: "left"},
        )

        self.assertEqual(len(shots), 1)
        shot = shots[0]
        self.assertEqual(shot.shooter_id, 9)
        self.assertEqual(shot.shooter_team, 1)
        self.assertEqual(shot.outcome, "goal")
        self.assertTrue(shot.is_goal)
        self.assertTrue(shot.is_on_target)
        self.assertGreater(shot.xg, 0.10)
        self.assertAlmostEqual(shot.distance_m, 17.5, delta=1.0)

    def test_detect_shots_saved_by_keeper(self):
        # Ball aimed at goal but intercepted by opposing GK at x=50.0
        ball_traj = {}
        player_traj = {}
        teams = {9: 1, 1: 2}

        for f in range(10):
            ball_traj[f] = (35.0, 0.0)
            player_traj[f] = {9: (35.0, 0.0), 1: (50.0, 0.0)}

        # Shot from frame 10 towards goal, but at frame 20 GK at (50.0, 0.0) stops it
        for f in range(10, 21):
            frac = (f - 10) / 10.0
            bx = 35.0 + frac * 15.0  # reaches 50.0 at frame 20
            ball_traj[f] = (bx, 0.0)
            player_traj[f] = {9: (35.0, 0.0), 1: (50.0, 0.0)}

        shots = self.detector.detect_shots(
            ball_trajectory=ball_traj,
            player_trajectories=player_traj,
            teams=teams,
            attacking_directions={1: "right", 2: "left"},
        )
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].outcome, "saved_by_keeper")
        self.assertFalse(shots[0].is_goal)
        self.assertTrue(shots[0].is_on_target)

    def test_detect_shots_off_target(self):
        # Shot flies wide (y reaches 12.0m at goal line)
        ball_traj = {}
        player_traj = {}
        teams = {9: 1}

        for f in range(10):
            ball_traj[f] = (35.0, 0.0)
            player_traj[f] = {9: (35.0, 0.0)}

        for f in range(10, 25):
            frac = (f - 10) / 14.0
            bx = 35.0 + frac * 18.0
            by = 0.0 + frac * 10.0  # crosses at y=10.0 (wide of ±3.66m post)
            ball_traj[f] = (bx, by)
            player_traj[f] = {9: (35.0, 0.0)}

        shots = self.detector.detect_shots(
            ball_trajectory=ball_traj,
            player_trajectories=player_traj,
            teams=teams,
            attacking_directions={1: "right", 2: "left"},
        )
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].outcome, "off_target")
        self.assertFalse(shots[0].is_goal)
        self.assertFalse(shots[0].is_on_target)

    def test_summarize_shooting_intelligence(self):
        shot1 = ShotEvent(
            shot_id=1, shooter_id=9, shooter_team=1,
            start_frame=10, end_frame=25, start_xy=(41.5, 0.0), end_xy=(52.5, 1.0),
            target_goal="right", distance_m=11.0, angle_rad=0.64, angle_deg=36.8,
            xg=0.76, outcome="goal", is_goal=True, is_on_target=True,
            defenders_in_cone=0, blockers_in_lane=0, gk_distance_to_goal=1.0,
            gk_positioning_score=0.5, shot_speed_mps=25.0, shot_speed_kmh=90.0,
            timestamp_sec=0.4,
        )
        shot2 = ShotEvent(
            shot_id=2, shooter_id=10, shooter_team=1,
            start_frame=50, end_frame=65, start_xy=(34.5, 5.0), end_xy=(50.0, 1.0),
            target_goal="right", distance_m=18.0, angle_rad=0.38, angle_deg=22.0,
            xg=0.12, outcome="saved_by_keeper", is_goal=False, is_on_target=True,
            defenders_in_cone=2, blockers_in_lane=1, gk_distance_to_goal=2.0,
            gk_positioning_score=0.8, shot_speed_mps=22.0, shot_speed_kmh=79.2,
            timestamp_sec=2.0,
        )

        summary = self.detector.summarize_shooting_intelligence([shot1, shot2])
        self.assertEqual(summary["total_shots"], 2)
        self.assertEqual(summary["shots_on_target"], 2)
        self.assertEqual(summary["goals"], 1)
        self.assertAlmostEqual(summary["total_xg"], 0.88, places=2)
        self.assertAlmostEqual(summary["xg_per_shot"], 0.44, places=2)
        self.assertAlmostEqual(summary["goals_minus_xg"], 0.12, places=2)
        self.assertEqual(len(summary["top_shooters"]), 2)
        self.assertEqual(summary["top_shooters"][0]["player_id"], 9)

    def test_render_shot_map(self):
        shot = ShotEvent(
            shot_id=1, shooter_id=9, shooter_team=1,
            start_frame=10, end_frame=25, start_xy=(41.5, 0.0), end_xy=(52.5, 1.0),
            target_goal="right", distance_m=11.0, angle_rad=0.64, angle_deg=36.8,
            xg=0.76, outcome="goal", is_goal=True, is_on_target=True,
            defenders_in_cone=0, blockers_in_lane=0, gk_distance_to_goal=1.0,
            gk_positioning_score=0.5, shot_speed_mps=25.0, shot_speed_kmh=90.0,
            timestamp_sec=0.4,
        )
        temp_dir = tempfile.mkdtemp()
        try:
            out_file = Path(temp_dir) / "shot_map.png"
            self.detector.render_shot_map([shot], out_file)
            self.assertTrue(out_file.exists())
            self.assertGreater(out_file.stat().st_size, 10000)
        finally:
            shutil.rmtree(temp_dir)

    def test_algorithm_benchmark(self):
        # Generate 1,000 frames of tracking data and measure pure algorithm execution time
        ball_traj = {f: (30.0 + (f % 50) * 0.4, (f % 20) - 10.0) for f in range(1000)}
        player_traj = {
            f: {
                pid: (30.0 + (f % 50) * 0.4 + pid, (f % 20) - 10.0 + pid)
                for pid in range(1, 23)
            }
            for f in range(1000)
        }
        teams = {pid: (1 if pid <= 11 else 2) for pid in range(1, 23)}

        # Warmup
        self.detector.detect_shots(ball_traj, player_traj, teams)

        iters = 5
        start_t = time.perf_counter()
        for _ in range(iters):
            self.detector.detect_shots(ball_traj, player_traj, teams)
        elapsed = time.perf_counter() - start_t

        total_frames = 1000 * iters
        fps = total_frames / elapsed
        print(f"\n[Algorithm-only Benchmark] ShotEventDetector: {fps:,.0f} FPS ({total_frames} frames in {elapsed:.3f}s)")
        self.assertGreater(fps, 10000.0)



    def test_run_shot_xg_task_integration(self):
        from server.pipeline.tasks import run_shot_xg, _compute_shot_xg_for_ai
        from unittest.mock import MagicMock

        temp_dir = tempfile.mkdtemp()
        try:
            session_id = "test_shot_session"
            task_id = "task_shot_001"

            # Create mock session output dir
            out_dir = Path(temp_dir) / session_id
            out_dir.mkdir(parents=True, exist_ok=True)

            sm = MagicMock()
            sm.session_output_dir.return_value = out_dir

            # Synthetic match cache data
            # Player 9 shoots towards right goal
            ball_traj = [{1: {"position_minimap": [35.0 + f * 1.0, 0.5]}} for f in range(20)]
            players_traj = [
                {
                    9: {"position_minimap": [35.0, 0.0], "team": 1, "track_id": 9},
                    1: {"position_minimap": [51.0, 0.0], "team": 2, "track_id": 1},
                }
                for _ in range(20)
            ]
            cache_data = {
                "tracks": {
                    "ball": ball_traj,
                    "players": players_traj,
                }
            }
            cache_file = out_dir / "tracks.pkl"
            import pickle
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)

            session = {
                "tracks_cache_path": str(cache_file),
                "video_fps": 25.0,
            }

            run_shot_xg(session_id, session, task_id, sm)

            # Check that shot_map.png and shot_xg_summary.json exist
            shot_map_file = out_dir / "shot_map.png"
            summary_json_file = out_dir / "shot_xg_summary.json"
            self.assertTrue(shot_map_file.exists())
            self.assertTrue(summary_json_file.exists())

            import json
            with open(summary_json_file, "r") as f:
                summary_data = json.load(f)
            self.assertIn("total_shots", summary_data)
            self.assertIn("total_xg", summary_data)
            self.assertIn("top_shooters", summary_data)

            # Verify _compute_shot_xg_for_ai helper
            ai_summary = _compute_shot_xg_for_ai(
                data=cache_data,
                tracks=cache_data["tracks"],
                tracked_bboxes={0: (350, 0, 40, 80)},
                fps=25.0,
            )
            self.assertIn("total_shots", ai_summary)
            self.assertIn("total_xg", ai_summary)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
