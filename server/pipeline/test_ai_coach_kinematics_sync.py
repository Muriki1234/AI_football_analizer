"""
test_ai_coach_kinematics_sync.py - Unit Test & Benchmark Suite
for AI Coach Kinematics & Debounced Tactical Telemetry Synchronization.

Verifies:
1. _compute_sprint_stats_for_ai: debounces tracking dropouts and uses FIFA >= 0.8s threshold.
2. _compute_defensive_breakthroughs_for_ai: integrates DefensiveLineAnalyzer with bidirectional support.
3. _compute_speed_telemetry_for_ai: produces 5-zone athletic intensity breakdown and distance/duration.
4. JSON payload serialization stability.
5. High-throughput algorithm microbenchmark (> 10,000 frames/sec).
"""

import json
import time
import unittest
import numpy as np

from server.pipeline.tasks import (
    _compute_sprint_stats_for_ai,
    _compute_defensive_breakthroughs_for_ai,
    _compute_speed_telemetry_for_ai,
)


class TestAICoachKinematicsSync(unittest.TestCase):
    def setUp(self):
        # 100 frames match simulation @ 25 FPS (4.0s)
        self.fps = 25.0
        self.n_frames = 100

        # Tracked player (Team 1) sprints from frame 20 to 55 (35 frames = 1.4s)
        # with a 2-frame tracking dip at frames 32-33 (21.5 km/h)
        self.players_frames = []
        self.tracked_bboxes = {}

        for f in range(self.n_frames):
            frame_players = {}
            if 20 <= f <= 55:
                spd = 21.5 if (f in (32, 33)) else 27.5
            else:
                spd = 12.0

            x = 20.0 + f * 0.5
            y = 34.0
            bx1, by1 = x * 10, y * 10
            bx2, by2 = bx1 + 30, by1 + 60
            frame_players[7] = {
                "bbox": [bx1, by1, bx2, by2],
                "position_transformed": [x, y],
                "speed": spd,
                "distance": f * 0.5,
                "team": 1,
            }

            # Opponent team (Team 2) 4 defenders at x=75.0m
            for opp_id in (101, 102, 103, 104):
                frame_players[opp_id] = {
                    "bbox": [750, opp_id * 5, 780, opp_id * 5 + 60],
                    "position_transformed": [75.0, 10.0 + (opp_id - 100) * 10.0],
                    "team": 2,
                }

            self.players_frames.append(frame_players)
            self.tracked_bboxes[f] = (int(bx1), int(by1), 30, 60)

        self.tracks = {"players": self.players_frames}

    def test_sprint_stats_debouncing_resilience(self):
        """Verifies sprint burst is detected even with 2-frame tracking drop dip."""
        sprints = _compute_sprint_stats_for_ai(self.tracks, self.tracked_bboxes, self.fps)
        # Should detect 1 clean stitched sprint burst
        self.assertEqual(sprints["count"], 1)
        self.assertAlmostEqual(sprints["avg_duration_s"], 36 / 25.0, delta=0.2)
        self.assertEqual(sprints["peak_kmh"], 27.5)
        self.assertGreater(sprints["total_distance_m"], 10.0)
        self.assertEqual(len(sprints["events"]), 1)
        ev = sprints["events"][0]
        self.assertIn("time_mm_ss", ev)
        self.assertGreater(ev["duration_s"], 1.0)

    def test_defensive_breakthroughs_analyzer_integration(self):
        """Verifies player passing defense line triggers penetration event."""
        def_line = _compute_defensive_breakthroughs_for_ai(self.tracks, self.tracked_bboxes, self.fps)
        self.assertEqual(def_line["tracked_team"], 1)
        self.assertEqual(def_line["attacking_direction"], "right")
        self.assertEqual(def_line["count"], 0)

        # Extend player to cross defense line at x=82m for frames 80..95 (> 75m)
        for f in range(80, 95):
            self.tracks["players"][f][7]["position_transformed"] = [82.0, 34.0]

        def_line_breached = _compute_defensive_breakthroughs_for_ai(self.tracks, self.tracked_bboxes, self.fps)
        self.assertGreaterEqual(def_line_breached["count"], 1)
        self.assertGreater(def_line_breached["max_depth_m"], 5.0)
        self.assertGreater(len(def_line_breached["events"]), 0)

    def test_speed_telemetry_5_zone_breakdown(self):
        """Verifies FIFA 5-zone athletic workload breakdown for LLM context."""
        telemetry = _compute_speed_telemetry_for_ai(self.tracks, self.tracked_bboxes, self.fps)
        self.assertIn("max_speed_kmh", telemetry)
        self.assertIn("avg_speed_kmh", telemetry)
        self.assertIn("total_distance_m", telemetry)
        self.assertIn("zone_breakdown", telemetry)

        zb = telemetry["zone_breakdown"]
        self.assertIn("zone_1_walking", zb)
        self.assertIn("zone_2_jogging", zb)
        self.assertIn("zone_3_running", zb)
        self.assertIn("zone_4_hsr", zb)
        self.assertIn("zone_5_sprinting", zb)

        # Sprint zone should have captured non-zero distance and percentage
        self.assertGreater(zb["zone_5_sprinting"]["distance_m"], 0.0)
        self.assertGreater(zb["zone_5_sprinting"]["percentage"], 0.0)

    def test_json_payload_serializability(self):
        """Ensures all computed stats serialize cleanly without NaN or non-standard types."""
        sprints = _compute_sprint_stats_for_ai(self.tracks, self.tracked_bboxes, self.fps)
        def_line = _compute_defensive_breakthroughs_for_ai(self.tracks, self.tracked_bboxes, self.fps)
        telemetry = _compute_speed_telemetry_for_ai(self.tracks, self.tracked_bboxes, self.fps)

        payload = {
            "sprints": sprints,
            "defensive_breakthroughs": def_line,
            "speed_telemetry": telemetry,
        }
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertIsInstance(serialized, str)
        self.assertIn("zone_breakdown", serialized)
        self.assertIn("time_mm_ss", serialized)

    def test_algorithm_benchmark_throughput(self):
        """[Algorithm-only Benchmark] Measures pure synchronized telemetry extraction rate."""
        n_evals = 20
        t0 = time.perf_counter()
        for _ in range(n_evals):
            _ = _compute_sprint_stats_for_ai(self.tracks, self.tracked_bboxes, self.fps)
            _ = _compute_defensive_breakthroughs_for_ai(self.tracks, self.tracked_bboxes, self.fps)
            _ = _compute_speed_telemetry_for_ai(self.tracks, self.tracked_bboxes, self.fps)
        elapsed = time.perf_counter() - t0

        total_frames_processed = self.n_frames * n_evals
        fps = total_frames_processed / max(elapsed, 1e-6)
        print(f"\n[Algorithm-only Benchmark] AICoachKinematicsSync: {fps:,.0f} frames/sec ({elapsed*1000:.2f} ms for {total_frames_processed} frames)")
        self.assertGreater(fps, 10_000, f"Expected > 10,000 FPS, got {fps:.1f}")


if __name__ == "__main__":
    unittest.main()
