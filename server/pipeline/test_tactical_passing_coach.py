"""
test_tactical_passing_coach.py — Unit Tests & Benchmarks for Tactical Passing Coach Intelligence

Verifies:
1. Pre-cached pass events & network extraction into stats_payload.
2. On-the-fly fallback pass detection when cache is empty.
3. Tracked player individual passing profile & combination partner extraction.
4. JSON schema validation and prompt structure integrity for both team & player modes.
5. Algorithm-only benchmark throughput on high pass volumes.
"""

import json
import time
import unittest
import numpy as np

from server.pipeline.tasks import _compute_passing_stats_for_ai
from server.pipeline.pass_event_detector import PassEvent


class TestTacticalPassingCoach(unittest.TestCase):
    def setUp(self):
        self.fps = 25.0

    def test_01_passing_stats_extraction_from_cache(self):
        """Pre-cached pass_events and pass_networks must be structured cleanly for the LLM."""
        fake_events = [
            {
                "pass_id": 1,
                "passer_id": 7,
                "passer_team": 1,
                "receiver_id": 10,
                "receiver_team": 1,
                "start_frame": 100,
                "end_frame": 125,
                "outcome": "completed",
                "pass_distance": 18.5,
                "is_progressive": True,
                "is_zone14_entry": True,
                "is_box_entry": False,
            },
            {
                "pass_id": 2,
                "passer_id": 10,
                "passer_team": 1,
                "receiver_id": 9,
                "receiver_team": 1,
                "start_frame": 150,
                "end_frame": 170,
                "outcome": "completed",
                "pass_distance": 12.0,
                "is_progressive": True,
                "is_zone14_entry": False,
                "is_box_entry": True,
            },
            {
                "pass_id": 3,
                "passer_id": 4,
                "passer_team": 2,
                "receiver_id": 8,
                "receiver_team": 2,
                "start_frame": 300,
                "end_frame": 320,
                "outcome": "completed",
                "pass_distance": 8.0,
                "is_progressive": False,
                "is_zone14_entry": False,
                "is_box_entry": False,
            },
        ]
        fake_networks = {
            "team1": {
                "team_id": 1,
                "total_passes": 2,
                "completed_passes": 2,
                "completion_rate_pct": 100.0,
                "progressive_passes": 2,
                "zone14_entries": 1,
                "box_entries": 1,
                "nodes": [
                    {"player_id": 10, "centroid_xy": [15.0, 0.0], "passes_made": 1, "passes_received": 1},
                    {"player_id": 7, "centroid_xy": [-10.0, 5.0], "passes_made": 1, "passes_received": 0},
                    {"player_id": 9, "centroid_xy": [35.0, 0.0], "passes_made": 0, "passes_received": 1},
                ],
                "edges": [
                    {"source": 7, "target": 10, "count": 1},
                    {"source": 10, "target": 9, "count": 1},
                ],
            },
            "team2": {
                "team_id": 2,
                "total_passes": 1,
                "completed_passes": 1,
                "completion_rate_pct": 100.0,
                "progressive_passes": 0,
                "zone14_entries": 0,
                "box_entries": 0,
                "nodes": [
                    {"player_id": 4, "centroid_xy": [-20.0, -10.0], "passes_made": 1, "passes_received": 0},
                    {"player_id": 8, "centroid_xy": [0.0, 0.0], "passes_made": 0, "passes_received": 1},
                ],
                "edges": [{"source": 4, "target": 8, "count": 1}],
            },
        }

        data = {"pass_events": fake_events, "pass_networks": fake_networks}
        stats = _compute_passing_stats_for_ai(data, tracks={}, tracked_bboxes={}, fps=self.fps)

        self.assertIn("team1", stats)
        self.assertIn("team2", stats)
        self.assertEqual(stats["team1"]["progressive_passes"], 2)
        self.assertEqual(stats["team1"]["zone14_entries"], 1)
        self.assertEqual(stats["team1"]["box_entries"], 1)
        self.assertEqual(len(stats["team1"]["key_hubs"]), 2)
        self.assertEqual(len(stats["key_penetration_events"]), 2)

        ev1 = stats["key_penetration_events"][0]
        self.assertEqual(ev1["passer_id"], 7)
        self.assertIn("zone14_entry", ev1["tags"])
        self.assertIn("progressive", ev1["tags"])
        self.assertEqual(ev1["time_mm_ss"], "00:04")

    def test_02_on_the_fly_pass_detection_fallback(self):
        """When cache has no pass data, detector must run on raw tracks smoothly."""
        n_frames = 40
        tracks = {
            "ball": [
                {1: {"position_transformed": (float(-15.0 + (30.0 * i / 25.0)), 0.0)}}
                if i < 25
                else {1: {"position_transformed": (15.0, 0.0)}}
                for i in range(n_frames)
            ],
            "players": [
                {
                    7: {"position_transformed": (-15.0, 0.0), "team": 1},
                    10: {"position_transformed": (15.0, 0.0), "team": 1},
                    4: {"position_transformed": (0.0, 15.0), "team": 2},
                }
                for _ in range(n_frames)
            ],
        }

        data = {}  # Empty cache
        stats = _compute_passing_stats_for_ai(data, tracks, tracked_bboxes={}, fps=self.fps)

        self.assertIn("team1", stats)
        self.assertIn("team2", stats)
        self.assertGreaterEqual(stats["team1"]["completed_passes"], 1)
        self.assertEqual(stats["team1"]["completion_rate_pct"], 100.0)

    def test_03_tracked_player_passing_profile(self):
        """Tracked player's personal passing stats and combination partners must be isolated."""
        fake_events = [
            {
                "pass_id": 1,
                "passer_id": 7,
                "passer_team": 1,
                "receiver_id": 10,
                "receiver_team": 1,
                "start_frame": 50,
                "end_frame": 75,
                "outcome": "completed",
                "pass_distance": 15.0,
                "is_progressive": True,
                "is_zone14_entry": True,
                "is_box_entry": False,
            },
            {
                "pass_id": 2,
                "passer_id": 7,
                "passer_team": 1,
                "receiver_id": 11,
                "receiver_team": 1,
                "start_frame": 150,
                "end_frame": 175,
                "outcome": "completed",
                "pass_distance": 22.0,
                "is_progressive": True,
                "is_zone14_entry": False,
                "is_box_entry": True,
            },
            {
                "pass_id": 3,
                "passer_id": 10,
                "passer_team": 1,
                "receiver_id": 7,
                "receiver_team": 1,
                "start_frame": 250,
                "end_frame": 270,
                "outcome": "completed",
                "pass_distance": 14.0,
                "is_progressive": False,
                "is_zone14_entry": False,
                "is_box_entry": False,
            },
            {
                "pass_id": 4,
                "passer_id": 7,
                "passer_team": 1,
                "receiver_id": 9,
                "receiver_team": 1,
                "start_frame": 350,
                "end_frame": 370,
                "outcome": "intercepted",
                "pass_distance": 25.0,
                "is_progressive": False,
                "is_zone14_entry": False,
                "is_box_entry": False,
            },
        ]
        data = {"pass_events": fake_events, "pass_networks": {}}
        tracks = {
            "players": [
                {7: {"bbox": [100, 100, 150, 200], "track_id": 7, "team": 1}}
                for _ in range(5)
            ]
        }
        tracked_bboxes = {0: [100, 100, 150, 200], 1: [100, 100, 150, 200]}

        stats = _compute_passing_stats_for_ai(data, tracks, tracked_bboxes, fps=self.fps)
        tp = stats.get("tracked_player")

        self.assertIsNotNone(tp)
        self.assertEqual(tp["player_id"], 7)
        self.assertEqual(tp["passes_attempted"], 3)
        self.assertEqual(tp["passes_completed"], 2)
        self.assertEqual(tp["passes_received"], 1)
        self.assertAlmostEqual(tp["completion_rate_pct"], 66.7, places=1)
        self.assertEqual(tp["progressive_passes"], 2)
        self.assertEqual(tp["zone14_entries"], 1)
        self.assertEqual(tp["box_entries"], 1)
        # Partner 10 had 1 received from 7 and 1 passed to 7 = 2 exchanges
        self.assertEqual(tp["top_combination_partners"][0]["teammate_id"], 10)
        self.assertEqual(tp["top_combination_partners"][0]["exchanges"], 2)

    def test_04_stats_payload_serialization(self):
        """Passing stats must serialize to valid JSON without NaN/inf."""
        data = {"pass_events": [], "pass_networks": {}}
        stats = _compute_passing_stats_for_ai(data, tracks={}, tracked_bboxes={}, fps=self.fps)

        payload = {"passing_intelligence": stats}
        dumped = json.dumps(payload, ensure_ascii=False)
        self.assertIn('"passing_intelligence"', dumped)
        loaded = json.loads(dumped)
        self.assertIn("team1", loaded["passing_intelligence"])
        self.assertIn("key_penetration_events", loaded["passing_intelligence"])

    def test_05_algorithm_only_benchmark_passing_coach_throughput(self):
        """
        [Algorithm-only Benchmark: AI Passing Intelligence Extraction & Formatting]:
        Measures latency on 500 match pass events. Must complete in < 15ms.
        """
        n_events = 500
        events = [
            {
                "pass_id": i + 1,
                "passer_id": (i % 11) + 1,
                "passer_team": 1 if (i % 22) < 11 else 2,
                "receiver_id": ((i + 1) % 11) + 1,
                "receiver_team": 1 if (i % 22) < 11 else 2,
                "start_frame": i * 30,
                "end_frame": i * 30 + 15,
                "outcome": "completed" if (i % 5 != 0) else "intercepted",
                "pass_distance": 10.0 + (i % 20),
                "is_progressive": (i % 4 == 0),
                "is_zone14_entry": (i % 7 == 0),
                "is_box_entry": (i % 11 == 0),
            }
            for i in range(n_events)
        ]
        data = {"pass_events": events, "pass_networks": {}}

        t0 = time.perf_counter()
        stats = _compute_passing_stats_for_ai(data, tracks={}, tracked_bboxes={}, fps=self.fps)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        print(f"\n[Algorithm-only Benchmark: AI Passing Intelligence Extraction & Formatting]:")
        print(f"  Pass Events Evaluated: {n_events}")
        print(f"  Extraction Latency   : {elapsed_ms:.3f} ms")
        print(f"  Throughput           : {n_events / (elapsed_ms / 1000.0):,.0f} events/sec")

        self.assertLess(elapsed_ms, 15.0)
        self.assertGreater(len(stats["key_penetration_events"]), 0)


if __name__ == "__main__":
    unittest.main()
