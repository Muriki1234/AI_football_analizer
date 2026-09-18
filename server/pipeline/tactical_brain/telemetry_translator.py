"""
telemetry_translator.py
Translates raw football tracking telemetry (coordinates, velocities, possession)
into semantic tactical metrics:
- 18-Zone Juego de Posición (JDP) pitch channel mapping
- Rest Defense (Restabsicherung) structural integrity
- Team compactness (bounding envelope & depth)
- Transition latency and counter-press triggers
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import math

# Standard pitch dimensions in meters
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# 5 Vertical Channels (Width: 0 to 68m)
VERTICAL_CHANNELS = [
    ("left_flank", 0.0, 13.6),
    ("left_half_space", 13.6, 27.2),
    ("center", 27.2, 40.8),
    ("right_half_space", 40.8, 54.4),
    ("right_flank", 54.4, 68.0),
]

# 4 Horizontal Lines (Length: 0 to 105m)
HORIZONTAL_LINES = [
    ("defensive_third", 0.0, 35.0),
    ("middle_third", 35.0, 70.0),
    ("attacking_third", 70.0, 88.5),
    ("penalty_area", 88.5, 105.0),
]


def resolve_jdp_zone(x: float, y: float) -> str:
    """
    Resolves an (x, y) pitch coordinate in meters to its Juego de Posición zone.
    x in [0, 105], y in [0, 68].
    """
    x = max(0.0, min(x, PITCH_LENGTH_M))
    y = max(0.0, min(y, PITCH_WIDTH_M))

    h_name = "middle_third"
    for name, x_min, x_max in HORIZONTAL_LINES:
        if x_min <= x <= x_max:
            h_name = name
            break

    v_name = "center"
    for name, y_min, y_max in VERTICAL_CHANNELS:
        if y_min <= y <= y_max:
            v_name = name
            break

    # Golden zone: Zone 14 is center attacking third (outside penalty area)
    if h_name == "attacking_third" and v_name == "center":
        return "zone_14"

    return f"{h_name}:{v_name}"


def compute_team_compactness(player_coords: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Calculates team envelope metrics: longitudinal depth, horizontal width, and bounding area.
    """
    if not player_coords or len(player_coords) < 2:
        return {"depth_m": 0.0, "width_m": 0.0, "area_m2": 0.0}

    xs = [p[0] for p in player_coords]
    ys = [p[1] for p in player_coords]

    depth = max(xs) - min(xs)
    width = max(ys) - min(ys)
    area = depth * width

    return {
        "depth_m": round(depth, 1),
        "width_m": round(width, 1),
        "area_m2": round(area, 1),
    }


def analyze_rest_defense(
    defending_players_coords: List[Tuple[float, float]],
    ball_coord: Optional[Tuple[float, float]],
    attacking_direction: int = 1  # +1: attacking towards x=105; -1: towards x=0
) -> Dict[str, Any]:
    """
    Analyzes rest defense (Restabsicherung) structure behind the ball.
    Returns player count, spatial distribution (e.g. '3+2', '2+3'), and vulnerability score.
    """
    if not defending_players_coords or ball_coord is None:
        return {"players_behind_ball": 0, "structure": "unknown", "stability_score": 50.0}

    ball_x = ball_coord[0]

    behind = []
    for p in defending_players_coords:
        if attacking_direction > 0 and p[0] < ball_x:
            behind.append(p)
        elif attacking_direction < 0 and p[0] > ball_x:
            behind.append(p)

    count = len(behind)

    if count >= 5:
        score = 95.0
        struct = "3+2"
    elif count == 4:
        score = 80.0
        struct = "3+1"
    elif count == 3:
        score = 60.0
        struct = "2+1"
    elif count == 2:
        score = 35.0
        struct = "2+0"
    else:
        score = 15.0
        struct = "exposed"

    return {
        "players_behind_ball": count,
        "structure": struct,
        "stability_score": score,
    }


def synthesize_telemetry_digest(
    match_duration_sec: float,
    fps: float,
    speed_reliability: str,
    possession_team1_pct: float,
    possession_team2_pct: float,
    possession_switches: int,
    pitch_positions: List[Tuple[float, float]],
    key_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Synthesizes complete telemetry into the standardized TacticalTelemetryDigest schema.
    """
    zone_counts: Dict[str, int] = {}
    half_space_count = 0
    total_positions = max(len(pitch_positions), 1)

    for x, y in pitch_positions:
        zone = resolve_jdp_zone(x, y)
        zone_counts[zone] = zone_counts.get(zone, 0) + 1
        if "half_space" in zone:
            half_space_count += 1

    half_space_pct = round((half_space_count / total_positions) * 100, 1)

    def_count = sum(c for z, c in zone_counts.items() if z.startswith("defensive_third"))
    mid_count = sum(c for z, c in zone_counts.items() if z.startswith("middle_third"))
    att_count = sum(c for z, c in zone_counts.items() if z.startswith("attacking_third") or z.startswith("penalty_area") or z == "zone_14")

    return {
        "match_metadata": {
            "duration_sec": round(match_duration_sec, 1),
            "fps": round(fps, 1),
            "speed_reliability": speed_reliability,
        },
        "possession_dynamics": {
            "team1_pct": round(possession_team1_pct, 1),
            "team2_pct": round(possession_team2_pct, 1),
            "neutral_pct": round(max(0.0, 100.0 - possession_team1_pct - possession_team2_pct), 1),
            "switches_count": possession_switches,
        },
        "spatial_distribution": {
            "defensive_third_pct": round((def_count / total_positions) * 100, 1),
            "middle_third_pct": round((mid_count / total_positions) * 100, 1),
            "attacking_third_pct": round((att_count / total_positions) * 100, 1),
            "half_space_occupancy_pct": half_space_pct,
            "zone14_entries_count": zone_counts.get("zone_14", 0),
        },
        "key_events": key_events,
    }
