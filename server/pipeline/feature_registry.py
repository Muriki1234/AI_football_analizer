"""
feature_registry.py - Unified Serverless Feature Dispatch & Artifact Registry

Single Source of Truth (SSOT) for all pipeline features across:
- server/routes/analysis.py (REST / Pod endpoint)
- server/handler.py (RunPod Serverless endpoint)
- server/pipeline/tasks.py (Task execution entrypoints)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Optional, Set

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSpec:
    canonical_name: str
    task_fn_name: str
    artifact_filename: str
    is_cpu_supported: bool
    needs_video_file: bool
    description: str


FEATURE_SPECS: dict[str, FeatureSpec] = {
    "heatmap": FeatureSpec(
        canonical_name="heatmap",
        task_fn_name="run_heatmap",
        artifact_filename="heatmap.png",
        is_cpu_supported=True,
        needs_video_file=False,
        description="Spatial density heatmap of tracked player",
    ),
    "speed_chart": FeatureSpec(
        canonical_name="speed_chart",
        task_fn_name="run_speed_chart",
        artifact_filename="speed_chart.png",
        is_cpu_supported=True,
        needs_video_file=False,
        description="Speed trajectory and sprint velocity profile",
    ),
    "possession": FeatureSpec(
        canonical_name="possession",
        task_fn_name="run_possession_stats",
        artifact_filename="possession_chart.png",
        is_cpu_supported=True,
        needs_video_file=False,
        description="Team possession percentage and breakdown",
    ),
    "sprint_analysis": FeatureSpec(
        canonical_name="sprint_analysis",
        task_fn_name="run_sprint_analysis",
        artifact_filename="sprint_analysis.png",
        is_cpu_supported=True,
        needs_video_file=False,
        description="High-speed running distance and sprint count",
    ),
    "defensive_line": FeatureSpec(
        canonical_name="defensive_line",
        task_fn_name="run_defensive_line",
        artifact_filename="defensive_line.png",
        is_cpu_supported=True,
        needs_video_file=False,
        description="Opposition defensive line penetration analysis",
    ),
    "ai_summary": FeatureSpec(
        canonical_name="ai_summary",
        task_fn_name="run_ai_summary",
        artifact_filename="ai_summary.md",
        is_cpu_supported=True,
        needs_video_file=True,
        description="Multimodal AI coaching tactical analysis",
    ),
    "spatial_radar": FeatureSpec(
        canonical_name="spatial_radar",
        task_fn_name="run_spatial_zone_radar",
        artifact_filename="spatial_radar.png",
        is_cpu_supported=True,
        needs_video_file=False,
        description="20-Zone Juego de Posición (JDP) tactical distribution radar",
    ),
    "voronoi": FeatureSpec(
        canonical_name="voronoi",
        task_fn_name="run_pitch_control_voronoi",
        artifact_filename="voronoi_pitch_control.png",
        is_cpu_supported=True,
        needs_video_file=False,
        description="Spearman pitch control and Voronoi territory model",
    ),
    "pass_network": FeatureSpec(
        canonical_name="pass_network",
        task_fn_name="run_pass_network",
        artifact_filename="pass_network.png",
        is_cpu_supported=True,
        needs_video_file=False,
        description="Passing network graph and ball action spotting",
    ),
    "vertical_crop": FeatureSpec(
        canonical_name="vertical_crop",
        task_fn_name="run_vertical_crop",
        artifact_filename="vertical_crop_916.mp4",
        is_cpu_supported=False,
        needs_video_file=True,
        description="9:16 vertical highlight video with cinematic tracking",
    ),
    "full_replay": FeatureSpec(
        canonical_name="full_replay",
        task_fn_name="run_full_replay",
        artifact_filename="full_replay.mp4",
        is_cpu_supported=False,
        needs_video_file=True,
        description="Showcase annotated full match replay video",
    ),
    "minimap_replay": FeatureSpec(
        canonical_name="minimap_replay",
        task_fn_name="run_minimap_replay",
        artifact_filename="minimap_replay.mp4",
        is_cpu_supported=True,
        needs_video_file=False,
        description="2D animated tactical pitch minimap video",
    ),
    "shot_xg": FeatureSpec(
        canonical_name="shot_xg",
        task_fn_name="run_shot_xg",
        artifact_filename="shot_map.png",
        is_cpu_supported=True,
        needs_video_file=False,
        description="Tactical shot event spotting and freeze-frame Expected Goals (xG) evaluator",
    ),
    "pressing_intensity": FeatureSpec(
        canonical_name="pressing_intensity",
        task_fn_name="run_pressing_intensity",
        artifact_filename="pressing_intensity.png",
        is_cpu_supported=True,
        needs_video_file=False,
        description="Passes Per Defensive Action (PPDA) and spatial pressing intensity analyzer",
    ),
    "match_bundle": FeatureSpec(
        canonical_name="match_bundle",
        task_fn_name="run_match_bundle",
        artifact_filename="match_analysis_bundle.zip",
        is_cpu_supported=True,
        needs_video_file=False,
        description="Single-call packaging of all match tactical artifacts and cryptographic manifest into a zip dossier",
    ),
    "turnover_transition": FeatureSpec(
        canonical_name="turnover_transition",
        task_fn_name="run_turnover_transition",
        artifact_filename="turnover_transitions.png",
        is_cpu_supported=True,
        needs_video_file=False,
        description="Defensive turnover spotting, 5s transition counter-press reaction latency, and counter-attack spotter",
    ),
    "offside_var": FeatureSpec(
        canonical_name="offside_var",
        task_fn_name="run_offside_var",
        artifact_filename="var_offside_map.png",
        is_cpu_supported=True,
        needs_video_file=False,
        description="Automated VAR offside line evaluation, second-last defender tracking, and metric margin analysis",
    ),
    "team_compactness": FeatureSpec(
        canonical_name="team_compactness",
        task_fn_name="run_team_compactness",
        artifact_filename="team_compactness.png",
        is_cpu_supported=True,
        needs_video_file=False,
        description="Dynamic 2D outfield player Convex Hull area (m^2), tactical stretch index, and team centroid distance",
    ),
}

FEATURE_ALIASES: dict[str, str] = {
    "18_zone_radar": "spatial_radar",
    "radar": "spatial_radar",
    "pitch_control": "voronoi",
    "pitch_control_voronoi": "voronoi",
    "passes": "pass_network",
    "passing_network": "pass_network",
    "ball_action": "pass_network",
    "vertical_crop_916": "vertical_crop",
    "replay": "full_replay",
    "minimap": "minimap_replay",
    "tactical_minimap": "minimap_replay",
    "shot_xg_analysis": "shot_xg",
    "xg": "shot_xg",
    "shots": "shot_xg",
    "shot_map": "shot_xg",
    "expected_goals": "shot_xg",
    "ppda": "pressing_intensity",
    "pressing": "pressing_intensity",
    "gegenpressing": "pressing_intensity",
    "counterpress": "pressing_intensity",
    "pressing_ppda": "pressing_intensity",
    "bundle": "match_bundle",
    "export_bundle": "match_bundle",
    "dossier": "match_bundle",
    "zip": "match_bundle",
    "all_artifacts": "match_bundle",
    "turnover": "turnover_transition",
    "turnovers": "turnover_transition",
    "transition_spotter": "turnover_transition",
    "turnover_spotter": "turnover_transition",
    "counterpress_transition": "turnover_transition",
    "turnover_transitions": "turnover_transition",
    "offside": "offside_var",
    "var_offside": "offside_var",
    "var": "offside_var",
    "saot": "offside_var",
    "compactness": "team_compactness",
    "convex_hull": "team_compactness",
    "stretch_index": "team_compactness",
    "team_shape": "team_compactness",
    "dispersion": "team_compactness",
}


def resolve_canonical_feature(name: str) -> str | None:
    """Resolves a feature name or alias to its canonical name."""
    if not name:
        return None
    cleaned = name.strip()
    if cleaned in FEATURE_SPECS:
        return cleaned
    return FEATURE_ALIASES.get(cleaned)


def get_feature_spec(name: str) -> FeatureSpec | None:
    """Retrieves the FeatureSpec for a canonical name or alias."""
    canon = resolve_canonical_feature(name)
    if not canon:
        return None
    return FEATURE_SPECS.get(canon)


def get_feature_task(name: str) -> Callable[..., Any] | None:
    """
    Dynamically loads and returns the task function from server.pipeline.tasks.
    Avoids eager circular dependencies at import time.
    """
    spec = get_feature_spec(name)
    if not spec:
        return None
    from . import tasks as pipeline_tasks
    return getattr(pipeline_tasks, spec.task_fn_name, None)


def build_feature_dispatch_table() -> dict[str, Callable[..., Any]]:
    """
    Builds the complete dispatch table mapping all canonical names and aliases
    to their executable task functions.
    """
    from . import tasks as pipeline_tasks
    dispatch: dict[str, Callable[..., Any]] = {}

    for name, spec in FEATURE_SPECS.items():
        fn = getattr(pipeline_tasks, spec.task_fn_name, None)
        if fn is not None:
            dispatch[name] = fn

    # Add aliases
    for alias, canon in FEATURE_ALIASES.items():
        if canon in dispatch:
            dispatch[alias] = dispatch[canon]

    return dispatch


def get_cpu_features() -> frozenset[str]:
    """Features permitted to run on CPU workers."""
    cpu_features = {
        name for name, spec in FEATURE_SPECS.items() if spec.is_cpu_supported
    }
    for alias, canon in FEATURE_ALIASES.items():
        if canon in cpu_features:
            cpu_features.add(alias)
    return frozenset(cpu_features)


def get_stats_only_features() -> frozenset[str]:
    """Features that only read tracks.pkl and do not require raw video download."""
    stats_features = {
        name for name, spec in FEATURE_SPECS.items() if not spec.needs_video_file
    }
    for alias, canon in FEATURE_ALIASES.items():
        if canon in stats_features:
            stats_features.add(alias)
    return frozenset(stats_features)


def get_all_artifact_files() -> set[str]:
    """All artifact filenames that can be cleared or registered."""
    base_artifacts = {
        "samurai_tracking.pkl",
        "tracks.pkl",
        "gemini_video.mp4",
        "samurai_temp.mp4",
        "vertical_crop_manifest.json",
    }
    spec_artifacts = {spec.artifact_filename for spec in FEATURE_SPECS.values()}
    return base_artifacts | spec_artifacts
