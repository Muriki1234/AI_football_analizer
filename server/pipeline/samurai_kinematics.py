"""Metric target trajectory independent of YOLO identity matching."""
import math
import numpy as np
from .target_player_continuous_kinematic_accumulator import TargetPlayerContinuousKinematicAccumulator


def project_target_positions(bboxes, homographies):
    """Project original-image footpoints using validated image-to-pitch-cm H.

    The existing pitch model is padded 120x70m, with a 105x68m playing area.
    Never infer a meter scale from pixel coordinates or invent missing H.
    """
    points = {}
    for frame, box in bboxes.items():
        if not (0 <= frame < len(homographies)) or homographies[frame] is None:
            continue
        try:
            x, y, w, h = map(float, box)
            matrix = np.asarray(homographies[frame], dtype=float)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                continue
            if w <= 0 or h <= 0 or not all(math.isfinite(v) for v in (x, y, w, h)):
                continue
            mapped = matrix @ np.array([x + w / 2, y + h, 1.0])
            if not np.isfinite(mapped).all() or abs(mapped[2]) < 1e-8:
                continue
            px = mapped[0] / mapped[2] / 100.0 - 7.5
            py = mapped[1] / mapped[2] / 100.0 - 1.0
            if -4 <= px <= 109 and -4 <= py <= 72:
                points[frame] = (float(px), float(py))
        except (TypeError, ValueError):
            continue
    return points


def target_motion_summary(bboxes, points, ranges, fps):
    """Report observed calibrated motion; missing calibration is unavailable.

    Separate accumulators prevent bridging skipped periods. Existing temporal
    and jitter filters operate in meters after projection, never in pixels.
    """
    summaries = []
    tracked = calibrated = samples = clamped = 0
    for start, end in ranges:
        acc = TargetPlayerContinuousKinematicAccumulator(fps=float(fps), reject_outliers=True)
        for frame in sorted(f for f in bboxes if start <= f < end):
            tracked += 1
            point = points.get(frame)
            if point is None:
                continue
            calibrated += 1
            acc.add_point(frame, frame / float(fps), *point)
        summary = acc.get_summary()
        valid_steps = sum(not p.is_gap and not p.is_clamped for p in acc.history[1:])
        samples += valid_steps
        clamped += acc.velocity_clamped_count
        if valid_steps:
            summaries.append(summary)

    coverage = calibrated / max(1, tracked)
    result = {
        'kinematics_source': 'samurai_pitch_projection',
        'tracked_frames': tracked,
        'calibrated_target_frames': calibrated,
        'pitch_coordinate_coverage_pct': round(coverage * 100, 1),
        'speed_samples': samples,
        'speed_rejected_count': clamped,
    }
    if not summaries or clamped > max(1, calibrated * 0.1):
        reason = ('已跟踪到球员，但缺少连续有效的球场坐标，无法可靠计算速度和距离。'
                  if not summaries else '球场坐标跳变过多，速度和距离已停用。')
        result.update({
            'total_distance_m': None, 'avg_speed_kmh': None,
            'max_speed_kmh': None, 'sprint_count': None,
            'speed_reliability': 'suspect',
            'max_speed_unavailable_reason': reason,
            'distance_unavailable_reason': reason,
            'sprints_unavailable_reason': reason,
            'analytics_contract': {'total_distance_m': 'EXPLICITLY_UNAVAILABLE',
                                   'sprints_count': 'EXPLICITLY_UNAVAILABLE'},
        })
        return result

    distance = sum(s['total_distance_m'] for s in summaries)
    duration = sum(s['duration_s'] for s in summaries)
    result.update({
        'total_distance_m': round(distance, 1),
        'avg_speed_kmh': round(distance / max(0.001, duration) * 3.6, 1),
        'max_speed_kmh': round(max(s['max_speed_kmh'] for s in summaries), 1),
        'sprint_count': sum(s['sprint_count'] for s in summaries),
        'speed_reliability': 'ok' if coverage >= 0.8 and not clamped else 'suspect',
    })
    if coverage < 0.8:
        result['distance_unavailable_reason'] = (
            f'仅统计有效球场坐标覆盖的片段（覆盖率 {coverage * 100:.1f}%），'
            '距离不是整段视频的完整跑动距离。')
    return result
