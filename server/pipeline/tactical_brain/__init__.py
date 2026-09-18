"""
tactical_brain - 18-Zone Juego de Posición (JDP) and Spatial Analytics
"""

from .telemetry_translator import (
    resolve_jdp_zone,
    compute_team_compactness,
    analyze_rest_defense,
    synthesize_telemetry_digest,
)

__all__ = [
    "resolve_jdp_zone",
    "compute_team_compactness",
    "analyze_rest_defense",
    "synthesize_telemetry_digest",
]
