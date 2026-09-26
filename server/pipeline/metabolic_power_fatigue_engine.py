"""
metabolic_power_fatigue_engine.py - Osgnach (2010) & di Prampero Metabolic Power & HMLD Kinematics Engine

Physiological & Mathematical Framework:
- Osgnach et al. (2010) "Energy Cost and Metabolic Power in Elite Soccer: A New Approach to Field Monitoring"
  (Medicine & Science in Sports & Exercise, 42(1):170-178).
- Di Prampero et al. (2005) Equivalent Slope (ES) Model:
    Accelerating on flat ground is energetically equivalent to running at constant speed up an incline:
      g' = sqrt(a_f^2 + g^2)
      ES = tan(alpha) = a_f / g
- Incline Energy Cost (EC, J/(kg*m)) from Minetti et al. (2002):
    EC = (155.4*ES^5 - 30.4*ES^4 - 32.8*ES^3 + 43.3*ES^2 + 39.7*ES + 3.6) * (g'/g) * KT
  where KT = 1.29 is the turf factor for natural football grass.
- Instantaneous Metabolic Power (P_met, W/kg):
    P_met = EC * v
- Key Metrics Derived:
    1. High Metabolic Load Distance (HMLD, m): distance covered at P_met > 25.5 W/kg (captures low-speed accelerations).
    2. Equivalent Distance (ED, m): hypothetical steady-state flat distance = Total Energy / (3.6 * KT).
    3. Equivalent Distance Index (EDI): ED / Actual Distance (typically 1.15 - 1.30 in football).
    4. Anaerobic Work Ratio: energy expended above 20 W/kg.
    5. Explosive Accelerations (a > +3.0 m/s^2) & Decelerations (a < -3.0 m/s^2).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

GRAVITY_M_S2 = 9.80665
DEFAULT_TURF_FACTOR = 1.29  # Standard football grass turf factor (di Prampero 2005)
HIGH_METABOLIC_THRESHOLD_W_KG = 25.5  # FIFA / Catapult standard HMLD threshold
ANAEROBIC_CRITICAL_POWER_W_KG = 20.0  # Aerobic critical power threshold
BASELINE_COST_FLAT_J_KG_M = 3.6  # Standard baseline cost on flat ground without turf factor


@dataclass
class MetabolicPowerConfig:
    turf_factor: float = DEFAULT_TURF_FACTOR
    hmld_threshold: float = HIGH_METABOLIC_THRESHOLD_W_KG
    anaerobic_threshold: float = ANAEROBIC_CRITICAL_POWER_W_KG
    accel_explosive_threshold: float = 3.0  # m/s^2
    decel_explosive_threshold: float = -3.0  # m/s^2
    min_speed_deadband: float = 0.2  # m/s (ignore standstill micro-jitter)


@dataclass
class PlayerMetabolicSummary:
    player_id: Union[int, str]
    team_id: Union[int, str]
    duration_sec: float
    total_distance_m: float
    hmld_m: float  # High Metabolic Load Distance (> 25.5 W/kg)
    equivalent_distance_m: float  # Theoretical constant flat distance
    equivalent_distance_index: float  # ED / Actual Distance
    total_energy_kj: float  # Total energy in kJ (assuming 75kg player baseline if weight unspecified)
    total_energy_j_per_kg: float  # Total energy in J/kg
    mean_metabolic_power_w_kg: float
    peak_metabolic_power_w_kg: float
    anaerobic_energy_ratio_pct: float  # % of energy expended > 20 W/kg
    explosive_accel_count: int  # a > +3.0 m/s^2
    explosive_decel_count: int  # a < -3.0 m/s^2
    power_zone_distances_m: Dict[str, float] = field(default_factory=dict)
    power_zone_time_pct: Dict[str, float] = field(default_factory=dict)


class MetabolicPowerEngine:
    """
    Computes professional sports science metabolic power, HMLD, and energetic loading
    from kinematic player trajectory coordinates and velocities.
    """

    def __init__(self, config: Optional[MetabolicPowerConfig] = None) -> None:
        self.cfg = config or MetabolicPowerConfig()

    @staticmethod
    def compute_energy_cost(
        accelerations: np.ndarray,
        turf_factor: float = DEFAULT_TURF_FACTOR,
    ) -> np.ndarray:
        """
        Computes instantaneous energy cost EC (J/(kg*m)) given forward acceleration vector.
        Uses di Prampero / Minetti 5th-order polynomial with Equivalent Slope (ES).
        """
        a = np.asarray(accelerations, dtype=np.float64)
        g = GRAVITY_M_S2
        es = a / g  # Equivalent slope tan(alpha)

        # Canonical Minetti et al. (2002) polynomial:
        # C(es) = 155.4*es^5 - 30.4*es^4 - 43.3*es^3 + 46.3*es^2 + 19.5*es + 3.6
        poly = (
            155.4 * (es ** 5)
            - 30.4 * (es ** 4)
            - 43.3 * (es ** 3)
            + 46.3 * (es ** 2)
            + 19.5 * es
            + 3.6
        )

        # Equivalent body mass ratio EM = g'/g = sqrt((a/g)^2 + 1)
        em = np.sqrt(es ** 2 + 1.0)

        # Base energy cost clamped to physical bounds [0.5, 90.0] J/(kg*m)
        ec = poly * em * float(turf_factor)
        return np.clip(ec, 0.5, 90.0)

    def evaluate_trajectory(
        self,
        times: np.ndarray,
        positions_x: np.ndarray,
        positions_y: np.ndarray,
        player_id: Union[int, str] = 0,
        team_id: Union[int, str] = 0,
        player_mass_kg: float = 75.0,
    ) -> PlayerMetabolicSummary:
        """
        Evaluates full kinematic and metabolic profile from time-series position coordinates (x, y, t).
        """
        n_points = len(times)
        if n_points < 3:
            return PlayerMetabolicSummary(
                player_id=player_id,
                team_id=team_id,
                duration_sec=0.0,
                total_distance_m=0.0,
                hmld_m=0.0,
                equivalent_distance_m=0.0,
                equivalent_distance_index=1.0,
                total_energy_kj=0.0,
                total_energy_j_per_kg=0.0,
                mean_metabolic_power_w_kg=0.0,
                peak_metabolic_power_w_kg=0.0,
                anaerobic_energy_ratio_pct=0.0,
                explosive_accel_count=0,
                explosive_decel_count=0,
            )

        t = np.asarray(times, dtype=np.float64)
        x = np.asarray(positions_x, dtype=np.float64)
        y = np.asarray(positions_y, dtype=np.float64)

        dt = np.diff(t)
        dt = np.where(dt <= 0, 1e-3, dt)  # guard against zero dt

        # Step displacement
        dx = np.diff(x)
        dy = np.diff(y)
        dist_steps = np.hypot(dx, dy)
        total_dist = float(np.sum(dist_steps))
        duration = float(t[-1] - t[0])

        # Instantaneous speeds at intervals: v = dist / dt
        speeds = dist_steps / dt
        # Apply speed deadband
        speeds = np.where(speeds < self.cfg.min_speed_deadband, 0.0, speeds)

        # Acceleration: a = dv / dt
        # For length matching, pad or central-difference
        # Interleaved speeds: pad edges
        v_padded = np.pad(speeds, (1, 1), mode="edge")
        dt_full = np.pad(dt, (0, 1), mode="edge")
        accels = np.diff(v_padded) / np.maximum(1e-3, dt_full)
        accels = accels[:len(speeds)]  # align with steps

        # Clamp physically impossible acceleration/deceleration spikes on video tracking
        accels = np.clip(accels, -8.0, 8.0)

        # Instantaneous Energy Cost EC
        ec = self.compute_energy_cost(accels, turf_factor=self.cfg.turf_factor)

        # Instantaneous Metabolic Power P = EC * v (W/kg)
        power = ec * speeds

        # 1. HMLD: distance where P_met >= 25.5 W/kg
        hmld_mask = power >= self.cfg.hmld_threshold
        hmld_distance = float(np.sum(dist_steps[hmld_mask]))

        # 2. Total Energy: integral of P_met * dt = J/kg
        energy_steps_j_kg = power * dt
        total_energy_j_kg = float(np.sum(energy_steps_j_kg))
        total_energy_kj = (total_energy_j_kg * float(player_mass_kg)) / 1000.0

        # 3. Equivalent Distance: ED = Total Energy / (3.6 * KT)
        flat_cost = BASELINE_COST_FLAT_J_KG_M * self.cfg.turf_factor
        equiv_dist = total_energy_j_kg / flat_cost if flat_cost > 0 else total_dist
        edi = (equiv_dist / total_dist) if total_dist > 1.0 else 1.0

        # 4. Anaerobic Ratio: energy > 20 W/kg
        anaerobic_mask = power >= self.cfg.anaerobic_threshold
        anaerobic_energy = float(np.sum(energy_steps_j_kg[anaerobic_mask]))
        anaerobic_pct = (anaerobic_energy / total_energy_j_kg * 100.0) if total_energy_j_kg > 0 else 0.0

        # 5. Explosive Accels & Decels counts (debounced, lasting at least 0.2s)
        exp_accel_events = int(np.sum(accels >= self.cfg.accel_explosive_threshold))
        exp_decel_events = int(np.sum(accels <= self.cfg.decel_explosive_threshold))

        # 6. Power Zones Breakdown:
        # Z1: 0 - 10 W/kg
        # Z2: 10 - 20 W/kg
        # Z3: 20 - 25.5 W/kg
        # Z4: 25.5 - 35 W/kg
        # Z5: > 35 W/kg
        z1 = power < 10.0
        z2 = (power >= 10.0) & (power < 20.0)
        z3 = (power >= 20.0) & (power < 25.5)
        z4 = (power >= 25.5) & (power < 35.0)
        z5 = power >= 35.0

        total_time = max(1e-3, float(np.sum(dt)))
        zone_distances = {
            "Z1_Low_0_10": round(float(np.sum(dist_steps[z1])), 2),
            "Z2_Medium_10_20": round(float(np.sum(dist_steps[z2])), 2),
            "Z3_High_20_25.5": round(float(np.sum(dist_steps[z3])), 2),
            "Z4_Elevated_25.5_35": round(float(np.sum(dist_steps[z4])), 2),
            "Z5_Maximal_gt_35": round(float(np.sum(dist_steps[z5])), 2),
        }
        zone_time_pct = {
            "Z1_Low_0_10": round(float(np.sum(dt[z1])) / total_time * 100.0, 1),
            "Z2_Medium_10_20": round(float(np.sum(dt[z2])) / total_time * 100.0, 1),
            "Z3_High_20_25.5": round(float(np.sum(dt[z3])) / total_time * 100.0, 1),
            "Z4_Elevated_25.5_35": round(float(np.sum(dt[z4])) / total_time * 100.0, 1),
            "Z5_Maximal_gt_35": round(float(np.sum(dt[z5])) / total_time * 100.0, 1),
        }

        mean_power = float(np.mean(power)) if len(power) > 0 else 0.0
        peak_power = float(np.max(power)) if len(power) > 0 else 0.0

        return PlayerMetabolicSummary(
            player_id=player_id,
            team_id=team_id,
            duration_sec=round(duration, 2),
            total_distance_m=round(total_dist, 2),
            hmld_m=round(hmld_distance, 2),
            equivalent_distance_m=round(equiv_dist, 2),
            equivalent_distance_index=round(edi, 3),
            total_energy_kj=round(total_energy_kj, 1),
            total_energy_j_per_kg=round(total_energy_j_kg, 1),
            mean_metabolic_power_w_kg=round(mean_power, 2),
            peak_metabolic_power_w_kg=round(peak_power, 2),
            anaerobic_energy_ratio_pct=round(anaerobic_pct, 1),
            explosive_accel_count=exp_accel_events,
            explosive_decel_count=exp_decel_events,
            power_zone_distances_m=zone_distances,
            power_zone_time_pct=zone_time_pct,
        )

    def render_metabolic_chart(
        self,
        summary: PlayerMetabolicSummary,
        output_path: str,
        title: Optional[str] = None,
    ) -> str:
        """
        Renders an athletic performance dashboard showing power zone distribution and EDI.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0B132B")
        for ax in (ax1, ax2):
            ax.set_facecolor("#0B132B")

        # 1. Bar chart: Power zone distance breakdown
        zones = list(summary.power_zone_distances_m.keys())
        zone_labels = ["Z1 (<10W)", "Z2 (10-20W)", "Z3 (20-25.5W)", "Z4 (25.5-35W)", "Z5 (>35W)"]
        dists = [summary.power_zone_distances_m[z] for z in zones]
        colors = ["#4A90E2", "#50E3C2", "#F5A623", "#FF6B6B", "#D0021B"]

        bars = ax1.bar(zone_labels, dists, color=colors, edgecolor="#FFFFFF", lw=0.8)
        ax1.set_title("Metabolic Power Distance Breakdown (m)", color="#FFFFFF", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Distance (m)", color="#E0E6ED", fontsize=10)
        ax1.tick_params(colors="#E0E6ED", labelsize=9)
        ax1.grid(axis="y", color="#2A3B5C", linestyle="--", alpha=0.5)

        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 5.0, f"{int(h)}m",
                         ha="center", va="bottom", color="#FFFFFF", fontsize=9, fontweight="bold")

        # 2. Key Metrics Summary Panel
        ax2.axis("off")
        chart_title = title or f"Player {summary.player_id} - Metabolic Power & Workload Audit"
        fig.suptitle(chart_title, color="#FFFFFF", fontsize=14, fontweight="bold", y=0.98)

        text_content = (
            f"⚡ Workload & Fatigue Summary:\n"
            f"─────────────────────────────────────\n"
            f"• Actual Odometer Distance : {summary.total_distance_m:,.1f} m\n"
            f"• High Metabolic Load (HMLD) : {summary.hmld_m:,.1f} m ({(summary.hmld_m/max(1.0, summary.total_distance_m)*100.0):.1f}%)\n"
            f"• Equivalent Distance (ED)  : {summary.equivalent_distance_m:,.1f} m\n"
            f"• Energy Cost Inflation (EDI): {summary.equivalent_distance_index:.2f}x\n"
            f"• Total Energy Expended     : {summary.total_energy_kj:,.1f} kJ ({summary.total_energy_j_per_kg:,.0f} J/kg)\n"
            f"• Mean Power / Peak Power   : {summary.mean_metabolic_power_w_kg:.1f} / {summary.peak_metabolic_power_w_kg:.1f} W/kg\n"
            f"• Anaerobic Energy Fraction : {summary.anaerobic_energy_ratio_pct:.1f}%\n"
            f"• Explosive Accelerations   : {summary.explosive_accel_count} (a > +3.0 m/s²)\n"
            f"• Explosive Decelerations   : {summary.explosive_decel_count} (a < -3.0 m/s²)\n"
        )
        ax2.text(
            0.05, 0.50, text_content,
            transform=ax2.transAxes,
            va="center", ha="left",
            color="#E0E6ED",
            fontsize=11,
            family="monospace",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#16223F", edgecolor="#2A3B5C", lw=1.5),
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return output_path
