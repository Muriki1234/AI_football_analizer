"""
test_turnover_transition_spotter.py - Unit Test & Algorithm Benchmark Suite
for TurnoverTransitionSpotter & Immediate Counter-Press Transition Analyzer.
"""

import shutil
import tempfile
import time
import unittest
from pathlib import Path

from server.pipeline.turnover_transition_spotter import (
    TurnoverEvent,
    TurnoverTransitionSpotter,
)


class TestTurnoverTransitionSpotter(unittest.TestCase):

    def setUp(self):
        self.spotter = TurnoverTransitionSpotter(fps=25.0)

    def test_pitch_zone_classification(self):
        # Right attacking:
        # x >= 17.5: attacking third
        # x in [17.5, 36.0] and abs(y) <= 13.5: zone_14
        # x <= -17.5: defensive third
        self.assertEqual(self.spotter.classify_pitch_zone((25.0, 0.0), 1, "right"), "zone_14")
        self.assertEqual(self.spotter.classify_pitch_zone((40.0, 20.0), 1, "right"), "attacking_third")
        self.assertEqual(self.spotter.classify_pitch_zone((-25.0, 0.0), 1, "right"), "defensive_third")
        self.assertEqual(self.spotter.classify_pitch_zone((0.0, 0.0), 1, "right"), "middle_third")

        # Left attacking:
        self.assertEqual(self.spotter.classify_pitch_zone((-25.0, 0.0), 2, "left"), "zone_14")
        self.assertEqual(self.spotter.classify_pitch_zone((-40.0, 20.0), 2, "left"), "attacking_third")
        self.assertEqual(self.spotter.classify_pitch_zone((25.0, 0.0), 2, "left"), "defensive_third")

    def test_turnover_detection_and_reaction(self):
        # Frames 0-9: Team 1 Player 10 in possession at (10.0, 0.0)
        # Frames 10-35: Team 2 Player 4 takes possession at (10.0, 0.0) -> Turnover!
        # At Frame 15 (5 frames = 0.20s later), Team 1 Player 8 challenges at (11.5, 0.0) (dist 1.5m <= 2.8m)
        ball_traj = {}
        player_traj = {}
        teams = {10: 1, 8: 1, 4: 2}

        for f in range(10):
            ball_traj[f] = (10.0, 0.0)
            player_traj[f] = {10: (10.0, 0.0), 8: (5.0, 0.0), 4: (20.0, 0.0)}

        for f in range(10, 40):
            ball_traj[f] = (10.0, 0.0)
            # Player 8 moves from 5.0 to 11.5 at frame 15
            p8_x = 5.0 if f < 15 else 11.5
            player_traj[f] = {10: (0.0, 0.0), 8: (p8_x, 0.0), 4: (10.0, 0.0)}

        turnovers = self.spotter.spot_turnovers(
            ball_trajectory=ball_traj,
            player_trajectories=player_traj,
            teams=teams,
            attacking_directions={1: "right", 2: "left"},
        )

        self.assertEqual(len(turnovers), 1)
        t = turnovers[0]
        self.assertEqual(t.losing_team, 1)
        self.assertEqual(t.winning_team, 2)
        self.assertEqual(t.winning_player_id, 4)
        self.assertTrue(t.counter_press_reacted)
        self.assertAlmostEqual(t.counter_press_reaction_sec, 0.20, places=2)

    def test_fast_break_counter_attack(self):
        # Possession switches to Team 1 at frame 10 at (-10.0, 0.0)
        # Team 1 drives forward to (15.0, 0.0) -> 25m progression within 50 frames (2.0s)
        ball_traj = {}
        player_traj = {}
        teams = {4: 2, 9: 1}

        for f in range(10):
            ball_traj[f] = (-10.0, 0.0)
            player_traj[f] = {4: (-10.0, 0.0), 9: (0.0, 0.0)}

        for f in range(10, 60):
            frac = (f - 10) / 50.0
            bx = -10.0 + frac * 25.0  # reaches 15.0m
            ball_traj[f] = (bx, 0.0)
            player_traj[f] = {4: (-10.0, 0.0), 9: (bx, 0.0)}

        turnovers = self.spotter.spot_turnovers(
            ball_trajectory=ball_traj,
            player_trajectories=player_traj,
            teams=teams,
            attacking_directions={1: "right", 2: "left"},
        )

        self.assertEqual(len(turnovers), 1)
        t = turnovers[0]
        self.assertEqual(t.winning_team, 1)
        self.assertTrue(t.is_counter_attack)
        self.assertGreaterEqual(t.progression_distance_5s, 18.0)

    def test_summarize_transitions(self):
        t1 = TurnoverEvent(
            turnover_id=1, frame=25, timestamp_sec=1.0, timestamp_mmss="00:01",
            losing_team=1, losing_player_id=10, winning_team=2, winning_player_id=4,
            turnover_xy=(20.0, 0.0), zone="attacking_third", is_high_turnover=True,
            is_dangerous_turnover=False, counter_press_reacted=True,
            counter_press_reaction_sec=1.2, is_counter_attack=True, progression_distance_5s=22.0,
        )
        summary = self.spotter.summarize_transitions([t1])
        self.assertEqual(summary["total_turnovers"], 1)
        self.assertEqual(summary["team2"]["turnovers_won"], 1)
        self.assertEqual(summary["team1"]["turnovers_lost"], 1)
        self.assertEqual(summary["team2"]["high_turnovers_won"], 1)
        self.assertEqual(summary["team1"]["avg_reaction_sec"], 1.2)

    def test_render_turnover_map(self):
        t1 = TurnoverEvent(
            turnover_id=1, frame=25, timestamp_sec=1.0, timestamp_mmss="00:01",
            losing_team=1, losing_player_id=10, winning_team=2, winning_player_id=4,
            turnover_xy=(20.0, 0.0), zone="attacking_third", is_high_turnover=True,
            is_dangerous_turnover=False, counter_press_reacted=True,
            counter_press_reaction_sec=1.2, is_counter_attack=True, progression_distance_5s=22.0,
        )
        temp_dir = tempfile.mkdtemp()
        try:
            out_file = Path(temp_dir) / "turnover_map.png"
            self.spotter.render_turnover_map([t1], out_file)
            self.assertTrue(out_file.exists())
            self.assertGreater(out_file.stat().st_size, 10000)
        finally:
            shutil.rmtree(temp_dir)

    def test_algorithm_benchmark(self):
        ball_traj = {f: (10.0 + (f % 30) * 0.5, (f % 10) - 5.0) for f in range(1000)}
        player_traj = {
            f: {
                pid: (10.0 + (f % 30) * 0.5 + (pid % 2), (f % 10) - 5.0 + (pid % 3))
                for pid in range(1, 23)
            }
            for f in range(1000)
        }
        teams = {pid: (1 if pid <= 11 else 2) for pid in range(1, 23)}

        # Warmup
        self.spotter.spot_turnovers(ball_traj, player_traj, teams)

        iters = 5
        start_t = time.perf_counter()
        for _ in range(iters):
            self.spotter.spot_turnovers(ball_traj, player_traj, teams)
        elapsed = time.perf_counter() - start_t

        total_frames = 1000 * iters
        fps = total_frames / elapsed
        print(f"\n[Algorithm-only Benchmark] TurnoverTransitionSpotter: {fps:,.0f} FPS ({total_frames} frames in {elapsed:.3f}s)")
        self.assertGreater(fps, 10000.0)

    def test_run_turnover_transition_task_integration(self):
        from server.pipeline.tasks import run_turnover_transition, _compute_turnover_transitions_for_ai
        from unittest.mock import MagicMock
        import pickle
        import json

        temp_dir = tempfile.mkdtemp()
        try:
            session_id = "test_turnover_session"
            task_id = "task_turnover_001"

            out_dir = Path(temp_dir) / session_id
            out_dir.mkdir(parents=True, exist_ok=True)

            sm = MagicMock()
            sm.session_output_dir.return_value = out_dir

            # Frames 0-9: Team 1 player 10 has ball at (10, 0)
            # Frames 10-24: Team 2 player 4 has ball at (10, 0)
            ball_list = []
            players_list = []
            for f in range(25):
                ball_list.append({1: {"position_minimap": [10.0, 0.0]}})
                if f < 10:
                    players_list.append({
                        10: {"position_minimap": [10.0, 0.0], "team": 1, "track_id": 10},
                        4: {"position_minimap": [20.0, 0.0], "team": 2, "track_id": 4},
                    })
                else:
                    players_list.append({
                        10: {"position_minimap": [0.0, 0.0], "team": 1, "track_id": 10},
                        4: {"position_minimap": [10.0, 0.0], "team": 2, "track_id": 4},
                    })

            cache_data = {
                "tracks": {
                    "ball": ball_list,
                    "players": players_list,
                }
            }
            cache_file = out_dir / "tracks.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)

            session = {
                "tracks_cache_path": str(cache_file),
                "video_fps": 25.0,
            }

            run_turnover_transition(session_id, session, task_id, sm)

            report_file = out_dir / "turnover_transitions.png"
            summary_json_file = out_dir / "turnover_transitions.json"
            self.assertTrue(report_file.exists())
            self.assertTrue(summary_json_file.exists())

            with open(summary_json_file, "r") as f:
                data = json.load(f)
            self.assertIn("team1", data)
            self.assertIn("team2", data)
            self.assertIn("total_turnovers", data)
            self.assertEqual(data["total_turnovers"], 1)

            ai_summary = _compute_turnover_transitions_for_ai(
                data=cache_data,
                tracks=cache_data["tracks"],
                tracked_bboxes={0: (100, 0, 40, 80)},
                fps=25.0,
            )
            self.assertIn("team1", ai_summary)
            self.assertIn("team2", ai_summary)
            self.assertEqual(ai_summary["total_turnovers"], 1)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()

