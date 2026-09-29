"""
grassroots_graceful_degradation_engine.py
=========================================
Grassroots & Amateur Video Analytics Reliability & Graceful Degradation Engine

Implements the 5-Tier Video Quality Taxonomy and Graceful Degradation Contract:
1. BROADCAST_PRO (Tier 1): Full physical metric analytics (speed, distance, pitch control, minimap).
2. TACTICAL_MAST (Tier 2): High mast / Veo camera, high spatial reliability.
3. AMATEUR_STAND (Tier 3): Bleacher/stand mount, requires motion-guided crop & kinematic smoothing.
4. PITCHSIDE_HANDHELD (Tier 4): Pitch level, perspective compression, camera shake.
   -> Triggers Partial Degradation: Metric speed/distance labeled SUSPECT/BOUNDED; minimap gated.
5. SEVERELY_DEGRADED (Tier 5): Faded lines, digital zoom, <4 keypoints persistently.
   -> Triggers Hard Degradation: Physical spatial metrics EXPLICITLY DISABLED (None/unavailable).
      Screen-space analytics (possession, duels, events) preserved without outputting fake numbers.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class FootageQualityTier(str, Enum):
    TIER_1_BROADCAST = "TIER_1_BROADCAST"
    TIER_2_TACTICAL_MAST = "TIER_2_TACTICAL_MAST"
    TIER_3_AMATEUR_STAND = "TIER_3_AMATEUR_STAND"
    TIER_4_PITCHSIDE_HANDHELD = "TIER_4_PITCHSIDE_HANDHELD"
    TIER_5_SEVERELY_DEGRADED = "TIER_5_SEVERELY_DEGRADED"


class MetricAvailability(str, Enum):
    FULLY_AVAILABLE = "FULLY_AVAILABLE"
    REDUCED_ACCURACY = "REDUCED_ACCURACY"
    EXPLICITLY_UNAVAILABLE = "EXPLICITLY_UNAVAILABLE"


class GrassrootsGracefulDegradationEngine:
    """
    Evaluates video telemetry and enforces the Graceful Degradation Contract.
    Prevents outputting deceptive 'fake-precise' physical metrics on low-grade footage.
    """

    def __init__(
        self,
        min_keypoints_for_homography: int = 4,
        max_staleness_frames_for_metric: int = 60,  # 2.4s at 25fps
        max_camera_jitter_px: float = 25.0,
    ) -> None:
        self.min_keypoints = min_keypoints_for_homography
        self.max_staleness = max_staleness_frames_for_metric
        self.max_jitter = max_camera_jitter_px

    def classify_footage_tier(
        self,
        homography_completeness_pct: float,
        avg_keypoint_count: float,
        camera_jitter_px: float,
        ball_coverage_pct: float,
        has_faded_markings: bool = False,
    ) -> FootageQualityTier:
        """
        Classifies input video into the 5-Tier Empirical Taxonomy.
        """
        # Tier 5: Severely Degraded
        if (
            homography_completeness_pct < 20.0
            or avg_keypoint_count < 2.0
            or (has_faded_markings and homography_completeness_pct < 35.0)
        ):
            return FootageQualityTier.TIER_5_SEVERELY_DEGRADED

        # Tier 4: Pitchside Handheld
        if (
            homography_completeness_pct < 55.0
            or avg_keypoint_count < 4.5
            or camera_jitter_px > self.max_jitter
        ):
            return FootageQualityTier.TIER_4_PITCHSIDE_HANDHELD

        # Tier 3: Amateur Stand / Bleacher
        if homography_completeness_pct < 80.0 or avg_keypoint_count < 6.5:
            return FootageQualityTier.TIER_3_AMATEUR_STAND

        # Tier 2: Tactical Mast / Veo
        if homography_completeness_pct < 95.0 or camera_jitter_px > 8.0:
            return FootageQualityTier.TIER_2_TACTICAL_MAST

        # Tier 1: Broadcast Pro
        return FootageQualityTier.TIER_1_BROADCAST

    def evaluate_metric_availability(
        self,
        tier: FootageQualityTier,
        homography_completeness_pct: float,
        max_staleness_frames: int,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Determines which metrics are valid, reduced, or explicitly disabled.
        """
        contract = {}

        # 1. Physical Speed & Sprints (km/h)
        if tier == FootageQualityTier.TIER_5_SEVERELY_DEGRADED or max_staleness_frames > 100:
            contract["max_speed_kmh"] = {
                "status": MetricAvailability.EXPLICITLY_UNAVAILABLE,
                "reason_zh": "由于球场标线不可辨识或摄像机剧烈位移，未形成稳定的真实坐标投影，物理速度已自动屏蔽以避免虚假数据。",
                "reason_en": "Pitch markings degraded or camera unstable; metric physical speed disabled to prevent deceptive numbers.",
            }
            contract["sprints_count"] = {
                "status": MetricAvailability.EXPLICITLY_UNAVAILABLE,
                "reason_zh": "冲刺次数依赖连续高精度米制速度序列，当前视频条件下不可靠。",
                "reason_en": "Sprint count requires high-fidelity metric coordinates; unavailable on degraded footage.",
            }
        elif tier == FootageQualityTier.TIER_4_PITCHSIDE_HANDHELD or max_staleness_frames > self.max_staleness:
            contract["max_speed_kmh"] = {
                "status": MetricAvailability.REDUCED_ACCURACY,
                "reason_zh": "受限于边线低角度透视，物理速度已应用生理加速度边界（6.5 m/s²）平滑，精度为评估级。",
                "reason_en": "Bounded by physiological acceleration filter due to pitchside perspective.",
            }
            contract["sprints_count"] = {
                "status": MetricAvailability.REDUCED_ACCURACY,
                "reason_zh": "冲刺次数已执行时域防抖去噪。",
                "reason_en": "Sprint count debounced against optical tracking jumps.",
            }
        else:
            contract["max_speed_kmh"] = {"status": MetricAvailability.FULLY_AVAILABLE}
            contract["sprints_count"] = {"status": MetricAvailability.FULLY_AVAILABLE}

        # 2. Total Distance (meters)
        if tier == FootageQualityTier.TIER_5_SEVERELY_DEGRADED:
            contract["total_distance_m"] = {
                "status": MetricAvailability.EXPLICITLY_UNAVAILABLE,
                "reason_zh": "全场跑动米数在无标线投影下会产生严重漂移膨胀，已关闭。",
                "reason_en": "Total distance disabled due to projection drift.",
            }
        elif tier == FootageQualityTier.TIER_4_PITCHSIDE_HANDHELD:
            contract["total_distance_m"] = {
                "status": MetricAvailability.REDUCED_ACCURACY,
                "reason_zh": "跑动距离已结合目标球员连续运动学累加器（消除镜头抖动导致的虚假位移）。",
                "reason_en": "Distance filtered via continuous kinematic accumulator.",
            }
        else:
            contract["total_distance_m"] = {"status": MetricAvailability.FULLY_AVAILABLE}

        # 3. 2D Minimap & Pitch Control (Voronoi)
        if tier in [FootageQualityTier.TIER_4_PITCHSIDE_HANDHELD, FootageQualityTier.TIER_5_SEVERELY_DEGRADED]:
            contract["2d_minimap"] = {
                "status": MetricAvailability.REDUCED_ACCURACY if tier == FootageQualityTier.TIER_4_PITCHSIDE_HANDHELD else MetricAvailability.EXPLICITLY_UNAVAILABLE,
                "reason_zh": "小地图俯视图在低机位或标线缺失下远端球员透视拉伸明显，仅展示相对站位。",
                "reason_en": "Minimap shows qualitative positioning only due to low-elevation perspective.",
            }
        else:
            contract["2d_minimap"] = {"status": MetricAvailability.FULLY_AVAILABLE}

        # 4. Screen-Space Analytics (Team Possession, Pass Counts, Action Spotting)
        # These operate in image space and REMAIN FULLY AVAILABLE even when spatial homography fails!
        contract["team_possession_pct"] = {
            "status": MetricAvailability.FULLY_AVAILABLE,
            "note": "Screen-space ball-to-player continuous state machine remains robust even without pitch homography.",
        }
        contract["pass_events"] = {
            "status": MetricAvailability.FULLY_AVAILABLE,
            "note": "Pass spotting utilizes temporal ball velocity and player proximity.",
        }

        return contract

    def apply_degradation_to_summary(
        self,
        summary: Dict[str, Any],
        telemetry: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Enforces the contract directly onto the final analytics summary dictionary.
        Replaces fake-precise metrics with None/Unavailable markers when required.
        """
        pct = telemetry.get("homography_completeness_pct", 100.0)
        kps = telemetry.get("avg_keypoint_count", 8.0)
        jitter = telemetry.get("camera_jitter_px", telemetry.get("camera_jitter_score", 5.0))
        ball_cov = telemetry.get("ball_coverage_pct", 50.0)
        staleness = telemetry.get("max_homography_staleness_frames", 0)
        faded = telemetry.get("has_faded_markings", False)

        tier = self.classify_footage_tier(pct, kps, jitter, ball_cov, faded)
        contract = self.evaluate_metric_availability(tier, pct, staleness)

        sanitized = dict(summary)
        sanitized["footage_quality_tier"] = tier.value
        sanitized["analytics_contract"] = {k: v["status"].value for k, v in contract.items()}

        # Enforce Explicit Unavailability
        if contract["max_speed_kmh"]["status"] == MetricAvailability.EXPLICITLY_UNAVAILABLE:
            sanitized["max_speed_kmh"] = None
            sanitized["max_speed_unavailable_reason"] = contract["max_speed_kmh"]["reason_zh"]

        if contract["sprints_count"]["status"] == MetricAvailability.EXPLICITLY_UNAVAILABLE:
            sanitized["sprint_count"] = None
            sanitized["sprints_unavailable_reason"] = contract["sprints_count"]["reason_zh"]

        if contract["total_distance_m"]["status"] == MetricAvailability.EXPLICITLY_UNAVAILABLE:
            sanitized["total_distance_m"] = None
            sanitized["distance_unavailable_reason"] = contract["total_distance_m"]["reason_zh"]

        return sanitized
