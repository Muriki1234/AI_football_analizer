"""
test_pressing_intensity_engine.py - Unit Test & Algorithm Benchmark Suite
for PressingIntensityEngine & PPDA (Passes Per Defensive Action).
"""

import shutil
import tempfile
import time
import unittest
from pathlib import Path

from server.pipeline.pressing_intensity_engine import (
    PressingAction,
    PressingIntensityEngine,
    TeamPPDAMetrics,
)


class TestPressingIntensityEngine(unittest.TestCase):

    def setUp(self):
        self.engine = PressingIntensityEngine(fps=25.0)

    def test_opposition_60_boundary(self):
        # Defending team attacking "right" (+x):
        # Pitch length 105m (-52.5 to +52.5). Opponent 60% is x in [-10.5, +52.5].
        self.assertTrue(self.engine.is_in_opposition_60(0.0, 1, "right"))
        self.assertTrue(self.engine.is_in_opposition_60(20.0, 1, "right"))
        self.assertTrue(self.engine.is_in_opposition_60(-10.0, 1, "right"))
        self.assertFalse(self.engine.is_in_opposition_60(-20.0, 1, "right"))

        # Defending team attacking "left" (-x):
        # Opponent 60% is x in [-52.5, +10.5].
        self.assertTrue(self.engine.is_in_opposition_60(0.0, 2, "left"))
        self.assertTrue(self.engine.is_in_opposition_60(-20.0, 2, "left"))
        self.assertTrue(self.engine.is_in_opposition_60(10.0, 2, "left"))
        self.assertFalse(self.engine.is_in_opposition_60(20.0, 2, "left"))

    def test_pressing_style_classification(self):
        self.assertEqual(PressingIntensityEngine.classify_pressing_style(6.2), "Aggressive Gegenpressing")
        self.assertEqual(PressingIntensityEngine.classify_pressing_style(9.5), "Active Mid-High Block")
        self.assertEqual(PressingIntensityEngine.classify_pressing_style(13.4), "Moderate Containment Block")
        self.assertEqual(PressingIntensityEngine.classify_pressing_style(19.1), "Deep Passive Low Block")

    def test_proximity_challenge_duel_detection(self):
        # Ball at (15.0, 0.0), Team 2 Player 8 (carrier) at (15.0, 0.0)
        # Team 1 Player 4 (defender, attacking right) closes in at (16.2, 0.0) -> dist = 1.2m <= 2.2m
        ball_traj = {f: (22.0, 0.0) for f in range(30)}
        player_traj = {
            f: {
                8: (22.0, 0.0),  # Team 2
                4: (23.2, 0.0),  # Team 1
            }
            for f in range(30)
        }
        teams = {8: 2, 4: 1}
        att_dirs = {1: "right", 2: "left"}

        actions = self.engine.detect_defensive_pressing_actions(
            ball_trajectory=ball_traj,
            player_trajectories=player_traj,
            teams=teams,
            attacking_directions=att_dirs,
        )

        # Due to 1.0s debouncing, 30 frames at 25 fps should produce exactly 1 pressing action
        self.assertEqual(len(actions), 1)
        a = actions[0]
        self.assertEqual(a.defender_id, 4)
        self.assertEqual(a.opponent_carrier_id, 8)
        self.assertEqual(a.defending_team, 1)
        self.assertEqual(a.action_type, "challenge_duel")
        self.assertTrue(a.is_high_press)  # x=16.2m is within 35m of opponent goal at 52.5m (52.5-16.2 = 36.3m, wait: 52.5-16.2 = 36.3 > 35m? Let us check dist_to_opp_goal)

    def test_ppda_calculation_exact_ratio(self):
        # Team 1 defending against Team 2:
        # Team 2 made 18 passes in Team 1 opposition 60% zone (x >= -10.5)
        # Team 1 performed 3 defensive actions in that zone
        pass_events = [
            {"passer_team": 2, "start_xy": (5.0, 0.0), "outcome": "completed"}
            for _ in range(18)
        ]
        def_actions = [
            PressingAction(
                action_id=i, frame=i * 25, timestamp_sec=i * 1.0,
                defending_team=1, defender_id=4, opponent_carrier_id=8,
                action_type="challenge_duel", pitch_xy=(10.0, 0.0),
                distance_to_opp_goal=42.5, is_high_press=False,
            )
            for i in range(1, 4)
        ]

        metrics = self.engine.compute_team_ppda(
            defending_team=1,
            opponent_team=2,
            pass_events=pass_events,
            defensive_actions=def_actions,
            attacking_directions={1: "right", 2: "left"},
        )

        # PPDA = 18 / 3 = 6.0
        self.assertAlmostEqual(metrics.ppda, 6.0, places=2)
        self.assertEqual(metrics.pressing_style, "Aggressive Gegenpressing")
        self.assertEqual(metrics.opponent_passes_in_zone, 18)
        self.assertEqual(metrics.defensive_actions_in_zone, 3)

    def test_analyze_match_pressing_end_to_end(self):
        ball_traj = {
            0: (12.0, 0.0),
            10: (14.0, 0.0),
            20: (20.0, 5.0),
        }
        player_traj = {
            0: {8: (12.0, 0.0), 4: (13.0, 0.0)},
            10: {8: (14.0, 0.0), 4: (14.8, 0.0)},
            20: {8: (20.0, 5.0), 4: (21.0, 5.0)},
        }
        teams = {8: 2, 4: 1}
        pass_events = [
            {"passer_team": 2, "start_xy": (10.0, 0.0), "end_xy": (20.0, 5.0), "outcome": "completed"},
            {"passer_team": 1, "start_xy": (-20.0, 0.0), "end_xy": (-5.0, 0.0), "outcome": "intercepted", "receiver_team": 2, "receiver_id": 8, "end_frame": 15},
        ]

        result = self.engine.analyze_match_pressing(
            ball_trajectory=ball_traj,
            player_trajectories=player_traj,
            teams=teams,
            pass_events=pass_events,
        )

        self.assertIn("team1", result)
        self.assertIn("team2", result)
        self.assertIn("total_pressing_actions", result)
        self.assertGreaterEqual(result["total_pressing_actions"], 1)

    def test_render_pressing_report(self):
        analysis_dict = {
            "team1": {
                "ppda": 7.4,
                "pressing_style": "Aggressive Gegenpressing",
                "opponent_passes_in_zone": 22,
                "defensive_actions_in_zone": 3,
                "high_press_turnovers": 2,
                "top_pressers": [{"player_id": 4, "pressures": 3}],
            },
            "team2": {
                "ppda": 14.8,
                "pressing_style": "Moderate Containment Block",
                "opponent_passes_in_zone": 44,
                "defensive_actions_in_zone": 3,
                "high_press_turnovers": 0,
                "top_pressers": [{"player_id": 8, "pressures": 2}],
            },
            "total_pressing_actions": 5,
            "actions": [
                {
                    "action_id": 1,
                    "defending_team": 1,
                    "pitch_xy": (25.0, 5.0),
                    "action_type": "challenge_duel",
                },
                {
                    "action_id": 2,
                    "defending_team": 2,
                    "pitch_xy": (-15.0, -10.0),
                    "action_type": "interception",
                },
            ],
        }

        temp_dir = tempfile.mkdtemp()
        try:
            out_file = Path(temp_dir) / "pressing_intensity.png"
            self.engine.render_pressing_report(analysis_dict, out_file)
            self.assertTrue(out_file.exists())
            self.assertGreater(out_file.stat().st_size, 10000)
        finally:
            shutil.rmtree(temp_dir)

    def test_algorithm_benchmark(self):
        ball_traj = {f: (15.0 + (f % 40) * 0.5, (f % 20) - 10.0) for f in range(1000)}
        player_traj = {
            f: {
                pid: (15.0 + (f % 40) * 0.5 + (pid % 3), (f % 20) - 10.0 + (pid % 2))
                for pid in range(1, 23)
            }
            for f in range(1000)
        }
        teams = {pid: (1 if pid <= 11 else 2) for pid in range(1, 23)}
        att_dirs = {1: "right", 2: "left"}

        # Warmup
        self.engine.detect_defensive_pressing_actions(ball_traj, player_traj, teams, att_dirs)

        iters = 5
        start_t = time.perf_counter()
        for _ in range(iters):
            self.engine.detect_defensive_pressing_actions(ball_traj, player_traj, teams, att_dirs)
        elapsed = time.perf_counter() - start_t

        total_frames = 1000 * iters
        fps = total_frames / elapsed
        print(f"\n[Algorithm-only Benchmark] PressingIntensityEngine: {fps:,.0f} FPS ({total_frames} frames in {elapsed:.3f}s)")
        self.assertGreater(fps, 10000.0)



    def test_run_pressing_intensity_task_integration(self):
        from server.pipeline.tasks import run_pressing_intensity, _compute_pressing_intensity_for_ai
        from unittest.mock import MagicMock
        import pickle
        import json

        temp_dir = tempfile.mkdtemp()
        try:
            session_id = "test_press_session"
            task_id = "task_press_001"

            out_dir = Path(temp_dir) / session_id
            out_dir.mkdir(parents=True, exist_ok=True)

            sm = MagicMock()
            sm.session_output_dir.return_value = out_dir

            ball_traj = [{1: {"position_minimap": [15.0, 0.0]}} for _ in range(25)]
            players_traj = [
                {
                    8: {"position_minimap": [15.0, 0.0], "team": 2, "track_id": 8},
                    4: {"position_minimap": [16.0, 0.0], "team": 1, "track_id": 4},
                }
                for _ in range(25)
            ]
            cache_data = {
                "tracks": {
                    "ball": ball_traj,
                    "players": players_traj,
                }
            }
            cache_file = out_dir / "tracks.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)

            session = {
                "tracks_cache_path": str(cache_file),
                "video_fps": 25.0,
            }

            run_pressing_intensity(session_id, session, task_id, sm)

            report_file = out_dir / "pressing_intensity.png"
            summary_json_file = out_dir / "pressing_ppda_summary.json"
            self.assertTrue(report_file.exists())
            self.assertTrue(summary_json_file.exists())

            with open(summary_json_file, "r") as f:
                data = json.load(f)
            self.assertIn("team1", data)
            self.assertIn("team2", data)
            self.assertIn("total_pressing_actions", data)

            ai_summary = _compute_pressing_intensity_for_ai(
                data=cache_data,
                tracks=cache_data["tracks"],
                tracked_bboxes={0: (160, 0, 40, 80)},
                fps=25.0,
            )
            self.assertIn("team1", ai_summary)
            self.assertIn("team2", ai_summary)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
