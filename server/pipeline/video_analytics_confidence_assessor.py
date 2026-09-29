"""
video_analytics_confidence_assessor.py - Video Analytics Confidence, Homography Completeness & Amateur Footage Robustness Assessor

Architecture & Theoretical Grounding:
1. SoccerNet Camera Calibration & BroadTrack SOTA Standards:
   - Evaluates camera calibration completeness rate (fraction of frames with valid homography).
   - Tracks reprojection RMSE and RANSAC inlier ratios across 29 canonical pitch keypoints.
   - Monitors temporal staleness (consecutive frames relying on historical projection without ground truth anchors).
2. Amateur / Grassroots Footage Awareness:
   - Distinguishes high-mount broadcast cameras from handheld smartphone footage.
   - Detects low pitch line visibility, violent camera pans, and prolonged homography blackouts.
   - Computes an end-to-end 0-100 Confidence Index across 4 orthogonal dimensions:
     * Calibration Completeness (45%)
     * Keypoint Reprojection Fidelity (25%)
     * Ball Tracking Continuity (15%)
     * Camera Motion Stability (15%)
3. Graceful Degradation & Transparency:
   - Flags stale projections when staleness > MAX_HOLD_FRAMES (default 75 frames / ~3.0s) to prevent distance inflation.
   - Generates transparent diagnostics and actionable warnings for frontend reporting.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ConfidenceTier(str, Enum):
    BROADCAST_HIGH = "BROADCAST_HIGH"
    TACTICAL_MEDIUM = "TACTICAL_MEDIUM"
    AMATEUR_LOW = "AMATEUR_LOW"


@dataclass
class FrameCalibrationObservation:
    """Per-frame telemetry collected during video pipeline execution."""
    frame_idx: int
    has_valid_homography: bool
    keypoint_count: int
    inlier_ratio: float = 0.0
    reprojection_rmse: float = 0.0
    ball_detected: bool = False
    camera_pan_speed_px: float = 0.0


@dataclass
class VideoConfidenceReport:
    """Comprehensive video quality, calibration fidelity, and analytics confidence scorecard."""
    total_frames: int
    valid_homography_frames: int
    homography_completeness_pct: float
    max_homography_staleness_frames: int
    stale_dropout_episodes_count: int
    avg_keypoint_count: float
    avg_inlier_ratio: float
    avg_reprojection_rmse: float
    ball_coverage_pct: float
    camera_jitter_score: float
    
    # 0-100 Sub-scores
    score_homography: float
    score_reprojection: float
    score_ball: float
    score_stability: float
    overall_confidence_score: float
    
    # Categorization & User Guidance
    confidence_tier: ConfidenceTier
    tier_label_zh: str
    tier_label_en: str
    is_reliable_for_sprints: bool
    is_reliable_for_distance: bool
    is_reliable_for_tactical_zones: bool
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "valid_homography_frames": self.valid_homography_frames,
            "homography_completeness_pct": round(self.homography_completeness_pct, 1),
            "max_homography_staleness_frames": self.max_homography_staleness_frames,
            "stale_dropout_episodes_count": self.stale_dropout_episodes_count,
            "avg_keypoint_count": round(self.avg_keypoint_count, 1),
            "avg_inlier_ratio": round(self.avg_inlier_ratio, 3),
            "avg_reprojection_rmse": round(self.avg_reprojection_rmse, 1),
            "ball_coverage_pct": round(self.ball_coverage_pct, 1),
            "camera_jitter_score": round(self.camera_jitter_score, 1),
            "sub_scores": {
                "homography": round(self.score_homography, 1),
                "reprojection": round(self.score_reprojection, 1),
                "ball": round(self.score_ball, 1),
                "stability": round(self.score_stability, 1),
            },
            "overall_confidence_score": round(self.overall_confidence_score, 1),
            "confidence_tier": self.confidence_tier.value,
            "tier_label_zh": self.tier_label_zh,
            "tier_label_en": self.tier_label_en,
            "reliability_gates": {
                "sprints": self.is_reliable_for_sprints,
                "distance": self.is_reliable_for_distance,
                "tactical_zones": self.is_reliable_for_tactical_zones,
            },
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }


class VideoAnalyticsConfidenceAssessor:
    """
    Evaluates end-to-end video analytics health, homography stability,
    and amateur footage degradation.
    """

    def __init__(
        self,
        fps: float = 25.0,
        max_staleness_seconds: float = 3.0,
        min_keypoints_for_full_credit: int = 8,
        acceptable_rmse_px: float = 12.0,
    ):
        self.fps = max(1.0, float(fps))
        self.max_staleness_frames = int(round(max_staleness_seconds * self.fps))
        self.min_keypoints_for_full_credit = min_keypoints_for_full_credit
        self.acceptable_rmse_px = acceptable_rmse_px

        self._observations: List[FrameCalibrationObservation] = []
        self._current_staleness: int = 0

    def reset(self) -> None:
        self._observations.clear()
        self._current_staleness = 0

    def record_frame(
        self,
        frame_idx: int,
        has_valid_homography: bool,
        keypoint_count: int = 0,
        inlier_ratio: float = 0.0,
        reprojection_rmse: float = 0.0,
        ball_detected: bool = False,
        camera_pan_speed_px: float = 0.0,
    ) -> bool:
        """
        Records a frame observation.
        Returns whether the frame's projection is fresh (True) or stale/degraded (False).
        """
        if has_valid_homography:
            self._current_staleness = 0
            is_usable = True
        else:
            self._current_staleness += 1
            is_usable = (self._current_staleness <= self.max_staleness_frames)

        obs = FrameCalibrationObservation(
            frame_idx=frame_idx,
            has_valid_homography=has_valid_homography,
            keypoint_count=keypoint_count,
            inlier_ratio=inlier_ratio,
            reprojection_rmse=reprojection_rmse,
            ball_detected=ball_detected,
            camera_pan_speed_px=camera_pan_speed_px,
        )
        self._observations.append(obs)
        return is_usable

    def evaluate(self) -> VideoConfidenceReport:
        """
        Aggregates frame observations into a comprehensive VideoConfidenceReport.
        """
        total = len(self._observations)
        if total == 0:
            return VideoConfidenceReport(
                total_frames=0,
                valid_homography_frames=0,
                homography_completeness_pct=0.0,
                max_homography_staleness_frames=0,
                stale_dropout_episodes_count=0,
                avg_keypoint_count=0.0,
                avg_inlier_ratio=0.0,
                avg_reprojection_rmse=0.0,
                ball_coverage_pct=0.0,
                camera_jitter_score=0.0,
                score_homography=0.0,
                score_reprojection=0.0,
                score_ball=0.0,
                score_stability=0.0,
                overall_confidence_score=0.0,
                confidence_tier=ConfidenceTier.AMATEUR_LOW,
                tier_label_zh="数据不足",
                tier_label_en="Insufficient Data",
                is_reliable_for_sprints=False,
                is_reliable_for_distance=False,
                is_reliable_for_tactical_zones=False,
                warnings=["未录入任何分析帧数据。"],
                recommendations=["请检查视频解码与帧读取是否正常。"],
            )

        valid_h_count = sum(1 for o in self._observations if o.has_valid_homography)
        completeness_pct = (valid_h_count / total) * 100.0

        # Calculate staleness runs and dropout episodes
        max_staleness = 0
        current_stale = 0
        dropout_episodes = 0
        in_dropout = False

        for o in self._observations:
            if o.has_valid_homography:
                if in_dropout:
                    in_dropout = False
                current_stale = 0
            else:
                current_stale += 1
                if current_stale > max_staleness:
                    max_staleness = current_stale
                if current_stale > self.max_staleness_frames and not in_dropout:
                    dropout_episodes += 1
                    in_dropout = True

        # Keypoints & inliers across frames that had keypoints
        kp_counts = [o.keypoint_count for o in self._observations]
        avg_kp = float(np.mean(kp_counts)) if kp_counts else 0.0

        inliers = [o.inlier_ratio for o in self._observations if o.has_valid_homography]
        avg_inlier = float(np.mean(inliers)) if inliers else 0.0

        rmses = [o.reprojection_rmse for o in self._observations if o.has_valid_homography and o.reprojection_rmse > 0]
        avg_rmse = float(np.mean(rmses)) if rmses else 0.0

        ball_count = sum(1 for o in self._observations if o.ball_detected)
        ball_coverage_pct = (ball_count / total) * 100.0

        pan_speeds = [o.camera_pan_speed_px for o in self._observations if o.camera_pan_speed_px > 0]
        avg_pan = float(np.mean(pan_speeds)) if pan_speeds else 0.0
        # High pan speed or jitter degrades stability score
        jitter_penalty = min(50.0, avg_pan * 1.5)

        # ── 1. Sub-score: Homography Completeness (Weight: 45%) ──────────────
        # 85%+ completeness gets full 100 score; prolonged dropouts penalize heavily
        raw_h_score = min(100.0, (completeness_pct / 85.0) * 100.0)
        dropout_penalty = min(35.0, dropout_episodes * 8.0)
        score_homography = max(0.0, raw_h_score - dropout_penalty)

        # ── 2. Sub-score: Keypoint & Reprojection Fidelity (Weight: 25%) ─────
        # Average keypoints >= min_keypoints gets full credit; inlier ratio scales it
        kp_factor = min(1.0, avg_kp / max(1.0, float(self.min_keypoints_for_full_credit)))
        inlier_factor = max(0.0, min(1.0, avg_inlier / 0.70)) if avg_inlier > 0 else 0.5
        rmse_penalty = max(0.0, min(30.0, (avg_rmse - self.acceptable_rmse_px) * 2.5)) if avg_rmse > self.acceptable_rmse_px else 0.0
        score_reprojection = max(0.0, min(100.0, (0.5 * kp_factor + 0.5 * inlier_factor) * 100.0 - rmse_penalty))

        # ── 3. Sub-score: Ball Tracking Continuity (Weight: 15%) ─────────────
        # Ball detected >= 70% of match gets 100
        score_ball = min(100.0, (ball_coverage_pct / 70.0) * 100.0)

        # ── 4. Sub-score: Camera Stability (Weight: 15%) ─────────────────────
        score_stability = max(0.0, 100.0 - jitter_penalty)

        # ── Composite Overall Confidence Score ──────────────────────────────
        overall_score = (
            0.45 * score_homography
            + 0.25 * score_reprojection
            + 0.15 * score_ball
            + 0.15 * score_stability
        )
        overall_score = max(0.0, min(100.0, overall_score))

        # ── Categorization into Tiers ─────────────────────────────────────────
        warnings: List[str] = []
        recommendations: List[str] = []

        if overall_score >= 80.0 and completeness_pct >= 85.0 and dropout_episodes == 0:
            tier = ConfidenceTier.BROADCAST_HIGH
            label_zh = "专业转播级 / 高置信度"
            label_en = "Broadcast Pro / High Confidence"
            reliable_sprints = True
            reliable_dist = True
            reliable_zones = True
        elif overall_score >= 50.0:
            tier = ConfidenceTier.TACTICAL_MEDIUM
            label_zh = "战术机位 / 中置信度"
            label_en = "Tactical Mount / Medium Confidence"
            reliable_sprints = (max_staleness <= self.max_staleness_frames * 1.5)
            reliable_dist = True
            reliable_zones = True
        else:
            tier = ConfidenceTier.AMATEUR_LOW
            label_zh = "业余手持 / 低置信度（降级容错）"
            label_en = "Amateur Handheld / Degraded Tolerance"
            reliable_sprints = False
            reliable_dist = (completeness_pct >= 35.0)
            reliable_zones = (completeness_pct >= 40.0)

        # Diagnose root causes and produce helpful warnings
        if completeness_pct < 60.0:
            warnings.append(
                f"球场透视校准覆盖率偏低（仅 {completeness_pct:.1f}% 帧满足几何标定）。"
            )
            recommendations.append(
                "建议拍摄时使用高位看台固定机位，尽可能包含边线、大禁区或中圈白线。"
            )

        if dropout_episodes > 0:
            staleness_sec = max_staleness / self.fps
            warnings.append(
                f"检测到 {dropout_episodes} 次较长时间校准脱节（最长单次失锁 {staleness_sec:.1f} 秒）。"
            )
            recommendations.append(
                "镜头快速摇摄或大幅度特写导致球场参考线短暂移出视野，已自动启用时空死区保护以防跑动距离虚增。"
            )

        if avg_kp < 5.0 and total > 30:
            warnings.append(
                f"单帧有效角点/交点数量偏少（平均 {avg_kp:.1f} 个，标准需 ≥ 6 个）。"
            )

        if ball_coverage_pct < 35.0:
            warnings.append(
                f"足球检测连续率不足（仅 {ball_coverage_pct:.1f}%），控球率可能偏向中立状态。"
            )

        if jitter_penalty > 20.0:
            warnings.append("检测到手持拍摄高频抖动，瞬时峰值速度已被生理滤波抑制。")

        return VideoConfidenceReport(
            total_frames=total,
            valid_homography_frames=valid_h_count,
            homography_completeness_pct=completeness_pct,
            max_homography_staleness_frames=max_staleness,
            stale_dropout_episodes_count=dropout_episodes,
            avg_keypoint_count=avg_kp,
            avg_inlier_ratio=avg_inlier,
            avg_reprojection_rmse=avg_rmse,
            ball_coverage_pct=ball_coverage_pct,
            camera_jitter_score=jitter_penalty,
            score_homography=score_homography,
            score_reprojection=score_reprojection,
            score_ball=score_ball,
            score_stability=score_stability,
            overall_confidence_score=overall_score,
            confidence_tier=tier,
            tier_label_zh=label_zh,
            tier_label_en=label_en,
            is_reliable_for_sprints=reliable_sprints,
            is_reliable_for_distance=reliable_dist,
            is_reliable_for_tactical_zones=reliable_zones,
            warnings=warnings,
            recommendations=recommendations,
        )
