import unittest
import time
from server.pipeline.pitch_control import (
    PlayerState,
    calculate_time_to_intercept,
    calculate_point_pitch_control,
    evaluate_pitch_control_grid,
    evaluate_passing_lane_openness,
)

class TestPitchControl(unittest.TestCase):
    def test_intercept_time_stationary(self):
        p = PlayerState(id=1, team_id=0, x=0.0, y=0.0, vx=0.0, vy=0.0, v_max=7.0, reaction_time=0.7)
        # Target 14m away -> 0.7 + 14/7 = 2.7s
        t = calculate_time_to_intercept(p, target_x=14.0, target_y=0.0)
        self.assertAlmostEqual(t, 2.7, places=2)

    def test_intercept_time_with_momentum(self):
        # Target at x=20.
        # Player A moving towards target at 5m/s
        p_towards = PlayerState(id=1, team_id=0, x=0.0, y=0.0, vx=5.0, vy=0.0, v_max=7.0, reaction_time=0.7)
        # Player B moving away from target at -5m/s
        p_away = PlayerState(id=2, team_id=0, x=0.0, y=0.0, vx=-5.0, vy=0.0, v_max=7.0, reaction_time=0.7)
        
        t_towards = calculate_time_to_intercept(p_towards, 20.0, 0.0)
        t_away = calculate_time_to_intercept(p_away, 20.0, 0.0)
        self.assertLess(t_towards, t_away)

    def test_point_pitch_control_dominance(self):
        # Home player right at (50, 34), Away player far at (90, 34)
        home = PlayerState(id=1, team_id=0, x=50.0, y=34.0)
        away = PlayerState(id=2, team_id=1, x=90.0, y=34.0)
        prob = calculate_point_pitch_control([home, away], target_x=51.0, target_y=34.0)
        self.assertGreater(prob, 0.95)

    def test_symmetric_midfield_split(self):
        # Home at x=40, y=34. Away at x=65, y=34. Midpoint is x=52.5, y=34.
        home = PlayerState(id=1, team_id=0, x=40.0, y=34.0)
        away = PlayerState(id=2, team_id=1, x=65.0, y=34.0)
        prob = calculate_point_pitch_control([home, away], target_x=52.5, target_y=34.0)
        self.assertAlmostEqual(prob, 0.5, delta=0.05)

    def test_grid_evaluation_structure(self):
        players = [
            PlayerState(id=1, team_id=0, x=30.0, y=34.0),
            PlayerState(id=2, team_id=1, x=75.0, y=34.0),
        ]
        res = evaluate_pitch_control_grid(players, grid_x=21, grid_y=14)
        self.assertEqual(res["grid_shape"], (14, 21))
        self.assertEqual(len(res["grid"]), 14)
        self.assertEqual(len(res["grid"][0]), 21)
        total_pct = res["home_territory_pct"] + res["away_territory_pct"] + res["contested_territory_pct"]
        self.assertAlmostEqual(total_pct, 100.0, delta=0.5)

    def test_passing_lane_openness_clear(self):
        # Pass between (20, 20) and (40, 20). Defender far away at (80, 50)
        passer = PlayerState(id=1, team_id=0, x=20.0, y=20.0)
        receiver = PlayerState(id=2, team_id=0, x=40.0, y=20.0)
        defender = PlayerState(id=3, team_id=1, x=80.0, y=50.0)
        
        openness = evaluate_passing_lane_openness([passer, receiver, defender], 20.0, 20.0, 40.0, 20.0)
        self.assertGreater(openness, 0.90)

    def test_passing_lane_blocked(self):
        # Pass between (20, 30) and (50, 30). Defender lurking right at (35, 30)
        passer = PlayerState(id=1, team_id=0, x=20.0, y=30.0)
        receiver = PlayerState(id=2, team_id=0, x=50.0, y=30.0)
        defender = PlayerState(id=3, team_id=1, x=35.0, y=30.0, vx=0.0, vy=0.0)

        openness = evaluate_passing_lane_openness([passer, receiver, defender], 20.0, 30.0, 50.0, 30.0)
        self.assertLess(openness, 0.20)

    def test_performance_benchmark(self):
        # 22 players on pitch
        players = []
        for i in range(11):
            players.append(PlayerState(id=i, team_id=0, x=10.0 + i*6.0, y=10.0 + (i%5)*10.0))
            players.append(PlayerState(id=i+11, team_id=1, x=50.0 + i*4.0, y=10.0 + (i%5)*10.0))

        t0 = time.perf_counter()
        res = evaluate_pitch_control_grid(players, grid_x=21, grid_y=14)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertLess(elapsed_ms, 50.0, f"Grid evaluation took {elapsed_ms:.2f}ms, expected < 50ms")

if __name__ == "__main__":
    unittest.main()
