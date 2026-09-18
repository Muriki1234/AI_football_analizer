"""
pitch_control.py - Lightweight Spearman Pitch Control & Voronoi Territory Estimator
Implements William Spearman's "Beyond Expected Goals" time-to-intercept model
to calculate continuous spatial pitch control and passing lane openness.
Zero external dependencies (pure Python standard library).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

PITCH_LENGTH_METERS = 105.0
PITCH_WIDTH_METERS = 68.0

@dataclass
class PlayerState:
    id: int
    team_id: int  # 0 for Home / Attacking, 1 for Away / Defending
    x: float      # meters [0.0, 105.0]
    y: float      # meters [0.0, 68.0]
    vx: float = 0.0  # m/s
    vy: float = 0.0  # m/s
    v_max: float = 7.0         # max sprint speed m/s
    reaction_time: float = 0.7  # seconds


def calculate_time_to_intercept(
    player: PlayerState,
    target_x: float,
    target_y: float
) -> float:
    """
    Computes time required for player to intercept a point (target_x, target_y),
    accounting for current velocity and initial cognitive reaction latency.
    """
    # Position after reaction time
    rx = player.x + player.vx * player.reaction_time
    ry = player.y + player.vy * player.reaction_time
    
    # Distance to target from post-reaction point
    dist = math.hypot(target_x - rx, target_y - ry)
    
    # Total intercept time
    return player.reaction_time + (dist / max(0.1, player.v_max))


def calculate_point_pitch_control(
    players: List[PlayerState],
    target_x: float,
    target_y: float,
    steepness: float = 4.3
) -> float:
    """
    Calculates the probability P(Home team controls point (target_x, target_y))
    using logistic time-difference formulation.
    Returns float in [0.0, 1.0].
    """
    home_times = [
        calculate_time_to_intercept(p, target_x, target_y)
        for p in players if p.team_id == 0
    ]
    away_times = [
        calculate_time_to_intercept(p, target_x, target_y)
        for p in players if p.team_id == 1
    ]

    if not home_times and not away_times:
        return 0.5
    if not home_times:
        return 0.0
    if not away_times:
        return 1.0

    min_home = min(home_times)
    min_away = min(away_times)

    time_diff = min_away - min_home  # Positive -> Home arrives first
    # Logistic sigmoid
    prob = 1.0 / (1.0 + math.exp(-steepness * time_diff))
    return max(0.0, min(1.0, prob))


def evaluate_pitch_control_grid(
    players: List[PlayerState],
    grid_x: int = 21,
    grid_y: int = 14,
    length: float = PITCH_LENGTH_METERS,
    width: float = PITCH_WIDTH_METERS
) -> Dict[str, Any]:
    """
    Evaluates pitch control across a 2D spatial grid.
    Returns summary metrics and 2D probability matrix.
    """
    step_x = length / (grid_x - 1) if grid_x > 1 else length
    step_y = width / (grid_y - 1) if grid_y > 1 else width

    grid: List[List[float]] = []
    home_controlled_cells = 0
    away_controlled_cells = 0
    total_cells = grid_x * grid_y

    for j in range(grid_y):
        row: List[float] = []
        y = j * step_y
        for i in range(grid_x):
            x = i * step_x
            p_home = calculate_point_pitch_control(players, x, y)
            row.append(round(p_home, 3))
            if p_home >= 0.55:
                home_controlled_cells += 1
            elif p_home <= 0.45:
                away_controlled_cells += 1
        grid.append(row)

    home_pct = round((home_controlled_cells / total_cells) * 100.0, 1)
    away_pct = round((away_controlled_cells / total_cells) * 100.0, 1)
    contested_pct = round(100.0 - home_pct - away_pct, 1)

    return {
        "grid": grid,
        "grid_shape": (grid_y, grid_x),
        "home_territory_pct": home_pct,
        "away_territory_pct": away_pct,
        "contested_territory_pct": contested_pct
    }


def evaluate_passing_lane_openness(
    players: List[PlayerState],
    passer_x: float,
    passer_y: float,
    receiver_x: float,
    receiver_y: float,
    pass_speed: float = 15.0,  # m/s
    samples: int = 10
) -> float:
    """
    Evaluates whether a passing vector is open or likely intercepted by defenders.
    Returns openness score [0.0, 1.0] (1.0 = completely free, 0.0 = directly blocked).
    """
    dx = receiver_x - passer_x
    dy = receiver_y - passer_y
    pass_dist = math.hypot(dx, dy)
    if pass_dist < 1.0:
        return 1.0

    total_flight_time = pass_dist / max(1.0, pass_speed)
    defenders = [p for p in players if p.team_id == 1]
    if not defenders:
        return 1.0

    intercept_risks = []
    for step in range(1, samples + 1):
        fraction = step / (samples + 1)
        ball_x = passer_x + dx * fraction
        ball_y = passer_y + dy * fraction
        ball_arrival_time = total_flight_time * fraction

        min_def_time = min(
            calculate_time_to_intercept(d, ball_x, ball_y)
            for d in defenders
        )
        min_dist_to_defender = min(
            math.hypot(d.x - ball_x, d.y - ball_y)
            for d in defenders
        )

        # Direct interception radius (1.5m stretch reach)
        if min_dist_to_defender <= 1.5 and ball_arrival_time >= 0.3:
            risk = 0.95
        else:
            time_margin = min_def_time - ball_arrival_time
            risk = 1.0 / (1.0 + math.exp(5.0 * time_margin))

        intercept_risks.append(risk)

    max_risk = max(intercept_risks) if intercept_risks else 0.0
    return round(max(0.0, min(1.0, 1.0 - max_risk)), 3)
