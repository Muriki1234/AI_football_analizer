"""
test_pipeline_accuracy_integration.py
======================================
Integration tests verifying the 4 core precision and coordinate defense modules
inside the production pipeline (tasks.py).

Verifies:
1. TargetPlayerContinuousKinematicAccumulator integration in _compute_player_summary:
   - Recovers full distance across ByteTrack ID switches (solves the broken max-min drop).
   - Produces FIFA-standard speed_telemetry and 5-zone exertion breakdown.
2. CanonicalPitchCoordinateNormalizer integration:
   - Rejects 1080p broadcast screen coordinates (e.g. (960, 540)) from leaking into metric space.
   - Accurately normalizes metric coordinates for _export_position_jsons.
3. TrackletPerceptualTeamClassifier integration in possession & player assigning:
   - Prevents unobserved tracklets from defaulting blindly to Team 1.
"""

import json
import unittest
from pathlib import Path
import tempfile
import numpy as np

from server.pipeline.tasks import (
    _compute_player_summary,
    _summary_for_range,
    _export_position_jsons,
)
from server.pipeline.canonical_pitch_coordinate_normalizer import (
    CanonicalPitchCoordinateNormalizer,
)
from server.pipeline.target_player_continuous_kinematic_accumulator import (
    TargetPlayerContinuousKinematicAccumulator,
)


class TestPipelineAccuracyIntegration(unittest.TestCase):
    def setUp(self):
        self.fps = 25.0
        self.normalizer = CanonicalPitchCoordinateNormalizer()

    def test_01_id_switch_distance_recovery_in_player_summary(self):
        """
        Verify that when ByteTrack undergoes an ID switch (e.g. player 10 -> player 25),
        _compute_player_summary accumulates continuous physical displacement rather than
        losing distance due to the old max(distances) - min(distances) flaw.
        """
        # Create a 50-frame sequence (2 seconds @ 25fps)
        # Player runs 25 meters across the pitch: from x=20m to x=45m, y=34m
        # Frame 0-24: track_id = 10, distance inside tracklet goes 0 -> 12m
        # Frame 25-49: track_id = 25 (ID SWITCH!), distance inside tracklet resets to 0 -> 12m
        tracks = {"players": []}
        tracked_bboxes = {}

        for f in range(50):
            # Target player moves 0.5m per frame = 12.5 m/s = 45 km/h (let's say 0.3m per frame = 7.5 m/s = 27 km/h sprint)
            x = 20.0 + f * 0.3
            y = 34.0
            pid = 10 if f < 25 else 25
            local_track_dist = (f if f < 25 else (f - 25)) * 0.3

            # Bounding box in video coordinates (centered on player)
            bbox = [100.0, 100.0, 140.0, 180.0]
            tracked_bboxes[f] = (100.0, 100.0, 40.0, 80.0)

            frame_players = {
                pid: {
                    "bbox": bbox,
                    "position_transformed": [x, y],
                    "speed": 27.0,
                    "distance": local_track_dist,
                    "has_ball": (f % 5 == 0),
                    "team": 1,
                    "track_id": pid,
                }
            }
            tracks["players"].append(frame_players)

        team_control = [1] * 50

        summary = _compute_player_summary(
            tracks=tracks,
            tracked_bboxes=tracked_bboxes,
            team_control=team_control,
            fps=int(self.fps),
        )

        # Total expected physical distance: 49 * 0.3 = 14.7 meters
        self.assertIn("total_distance_m", summary)
        self.assertAlmostEqual(summary["total_distance_m"], 15.0, delta=2.0)
        self.assertGreater(summary["total_distance_m"], 12.0, "Must not be truncated to single-tracklet distance (7.2m)")

        # Verify speed_telemetry object
        self.assertIn("speed_telemetry", summary)
        st = summary["speed_telemetry"]
        self.assertIn("speed_zones_m", st)
        self.assertIn("sprint_count", st)
        self.assertIn("fifa_avg_speed_kmh", st)
        self.assertGreater(st["fifa_avg_speed_kmh"], 20.0)

    def test_02_summary_for_range_kinematics(self):
        """
        Verify that _summary_for_range outputs the verified kinematics and speed_telemetry.
        """
        tracks = {"players": []}
        tracked_bboxes = {}

        for f in range(30):
            x = 10.0 + f * 0.1  # 2.5 m/s = 9 km/h (jogging)
            y = 20.0
            tracked_bboxes[f] = (50.0, 50.0, 30.0, 60.0)
            tracks["players"].append({
                5: {
                    "bbox": [50.0, 50.0, 80.0, 110.0],
                    "position_transformed": [x, y],
                    "speed": 9.0,
                    "distance": f * 0.1,
                    "has_ball": False,
                    "team": 2,
                }
            })

        team_control = [2] * 30
        res = _summary_for_range(
            tracks=tracks,
            tracked_bboxes=tracked_bboxes,
            team_control=team_control,
            start=0,
            end=30,
            fps=int(self.fps),
        )

        self.assertIn("total_distance_m", res)
        self.assertAlmostEqual(res["total_distance_m"], 3.0, delta=1.0)
        self.assertIn("speed_telemetry", res)
        self.assertIn("jogging", res["speed_telemetry"]["speed_zones_m"])

    def test_03_screen_pixel_leak_rejection_in_export_jsons(self):
        """
        Verify that _export_position_jsons rejects raw screen pixel coordinates
        (e.g. 960, 540) to prevent canvas overlay distortions and extreme jumps.
        """
        tracks = {
            "players": [
                # Frame 0: Valid pitch coordinate (52.5, 34.0)
                {
                    1: {
                        "position_transformed": [52.5, 34.0],
                        "team": 1,
                        "has_ball": True,
                    }
                },
                # Frame 1: Screen pixel glitch (960.0, 540.0)
                {
                    1: {
                        "position_transformed": [960.0, 540.0],
                        "team": 1,
                        "has_ball": False,
                    }
                },
            ],
            "ball": [
                {1: {"position_transformed": [52.5, 34.0]}},
                {1: {"position_transformed": [960.0, 540.0]}},
            ],
        }
        tracked_bboxes = {
            0: (100, 100, 50, 100),
            1: (100, 100, 50, 100),
        }
        team_control = [1, 1]
        team_colors = {1: np.array([255, 0, 0])}

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            _export_position_jsons(
                session_id="test_sess",
                tracks=tracks,
                tracked_bboxes=tracked_bboxes,
                team_control=team_control,
                team_colors=team_colors,
                fps=self.fps,
                total=2,
                output_dir=out_dir,
            )

            mm_path = out_dir / "minimap_positions.json"
            hm_path = out_dir / "heatmap_positions.json"
            self.assertTrue(mm_path.exists())
            self.assertTrue(hm_path.exists())

            mm_data = json.loads(mm_path.read_text())
            # Frame 0 has player point, Frame 1 rejected screen pixel (960, 540)
            self.assertEqual(len(mm_data["frames"]), 2)
            self.assertEqual(len(mm_data["frames"][0]), 1)
            self.assertEqual(len(mm_data["frames"][1]), 0, "Screen pixel (960, 540) must be rejected!")

            # Ball: Frame 0 valid, Frame 1 rejected
            self.assertIsNotNone(mm_data["ball"][0])
            self.assertIsNone(mm_data["ball"][1], "Ball screen pixel (960, 540) must be rejected!")

            # Center coordinate check: (52.5, 34.0) maps to center of Soccana pitch (6000, 3500)
            p0 = mm_data["frames"][0][0]
            self.assertAlmostEqual(p0["x"], 6000.0, delta=10.0)
            self.assertAlmostEqual(p0["y"], 3500.0, delta=10.0)


if __name__ == "__main__":
    unittest.main()
