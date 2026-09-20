"""
match_report_bundle_packager.py - Unified Single-Call Match Dossier Zip Packager & Checksum Manifest Generator

Architectural Foundations:
1. FIFA EPTS Match Data Exchange Standard:
   Every exported match dossier contains a root manifest.json detailing match metadata,
   coordinate reference system, executive tactical KPIs, and SHA256 cryptographic hashes
   for all included visual, event, and telemetry artifacts.

2. Dual-Mode Streaming Zip Packaging:
   - Uses ZIP_DEFLATED for highly compressible JSON telemetry, Markdown coaching reports, and PNG charts.
   - Uses ZIP_STORED for pre-compressed H.264 MP4 video streams, eliminating redundant CPU re-compression.

3. Complete Artifact Discovery:
   Aggregates up to 14 tactical match assets:
   - Visual charts: heatmap.png, speed_chart.png, possession_chart.png, sprint_analysis.png,
     defensive_line.png, spatial_radar.png, voronoi_pitch_control.png, pass_network.png,
     shot_map.png, pressing_intensity.png
   - Telemetry data: speed_telemetry.json, shot_xg_summary.json, pressing_ppda_summary.json
   - AI Coaching: ai_summary.md
   - Replay videos: minimap_replay.mp4, vertical_crop_916.mp4, full_replay.mp4
"""

from __future__ import annotations

import datetime
import hashlib
import json
import mimetypes
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import zipfile


@dataclass
class ArtifactManifestEntry:
    name: str
    category: str  # "tactical_visual", "kinematics", "event_data", "video_replay", "report"
    file_size_bytes: int
    sha256_hash: str
    mime_type: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MatchDossierManifest:
    session_id: str
    generated_at: str
    match_metadata: Dict[str, Any]
    executive_kpis: Dict[str, Any]
    artifacts_count: int
    total_size_bytes: int
    artifacts: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MatchReportBundlePackager:
    """
    Aggregates match artifacts, builds a cryptographic manifest, and packages
    them into a single-call downloadable zip container.
    """

    KNOWN_ARTIFACTS: Dict[str, Tuple[str, str]] = {
        "heatmap.png": ("tactical_visual", "Spatial density heatmap of tracked player"),
        "speed_chart.png": ("kinematics", "Speed trajectory and sprint velocity profile"),
        "possession_chart.png": ("tactical_visual", "Team possession percentage and breakdown"),
        "sprint_analysis.png": ("kinematics", "High-speed running distance and sprint burst analysis"),
        "defensive_line.png": ("tactical_visual", "Opposition defensive line penetration analysis"),
        "spatial_radar.png": ("tactical_visual", "20-Zone Juego de Posición (JDP) tactical distribution radar"),
        "voronoi_pitch_control.png": ("tactical_visual", "Spearman pitch control and Voronoi territory model"),
        "pass_network.png": ("tactical_visual", "Passing network graph and ball action spotting"),
        "shot_map.png": ("tactical_visual", "Tactical shot event spotting and Expected Goals (xG) map"),
        "pressing_intensity.png": ("tactical_visual", "PPDA pressing intensity and high-press duel zones"),
        "turnover_transitions.png": ("tactical_visual", "Defensive turnover spatial map and counter-attack transitions"),
        "ai_summary.md": ("report", "Multimodal AI coaching tactical analysis and evaluation report"),
        "speed_telemetry.json": ("kinematics", "FIFA 5-zone athletic speed kinematics and timeline telemetry"),
        "shot_xg_summary.json": ("event_data", "Detailed shot events and freeze-frame Expected Goals calculations"),
        "pressing_ppda_summary.json": ("event_data", "PPDA pressing actions and duel transition events"),
        "turnover_transitions.json": ("event_data", "Possession turnover logs, counter-press latencies, and transition metrics"),
        "minimap_replay.mp4": ("video_replay", "2D animated tactical pitch minimap replay video"),
        "vertical_crop_916.mp4": ("video_replay", "9:16 mobile vertical highlight video"),
        "full_replay.mp4": ("video_replay", "Showcase annotated full match replay video"),
    }

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Computes SHA-256 cryptographic hash of a file in 64KB blocks."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def classify_artifact(self, file_name: str) -> Tuple[str, str, str]:
        """
        Returns (category, mime_type, description) for a given artifact filename.
        """
        mime_type, _ = mimetypes.guess_type(file_name)
        mime_type = mime_type or "application/octet-stream"

        if file_name in self.KNOWN_ARTIFACTS:
            cat, desc = self.KNOWN_ARTIFACTS[file_name]
            return cat, mime_type, desc

        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            return "tactical_visual", mime_type, "Custom tactical visual export"
        elif file_name.endswith(".json"):
            return "event_data", mime_type, "Match data telemetry export"
        elif file_name.endswith(".mp4"):
            return "video_replay", mime_type, "Video replay export"
        elif file_name.endswith(".md"):
            return "report", mime_type, "Tactical coaching report"
        return "other", mime_type, "Match analysis asset"

    def build_manifest(
        self,
        session_id: str,
        output_dir: Path,
        session: Dict[str, Any],
        cache_data: Optional[Dict[str, Any]] = None,
    ) -> MatchDossierManifest:
        """
        Scans output_dir for all present match artifacts, computes hashes,
        extracts executive KPIs, and returns MatchDossierManifest.
        """
        artifacts_entries: List[ArtifactManifestEntry] = []
        total_size = 0

        # Scan all files in output_dir
        if output_dir.exists():
            for p in sorted(output_dir.iterdir()):
                if p.is_file() and not p.name.startswith(".") and p.name not in ("manifest.json", "match_analysis_bundle.zip"):
                    # Check if recognized artifact or standard output
                    cat, mime, desc = self.classify_artifact(p.name)
                    size = p.stat().st_size
                    h = self.compute_sha256(p)
                    entry = ArtifactManifestEntry(
                        name=p.name,
                        category=cat,
                        file_size_bytes=size,
                        sha256_hash=h,
                        mime_type=mime,
                        description=desc,
                    )
                    artifacts_entries.append(entry)
                    total_size += size

        # Extract metadata
        total_frames = int(session.get("total_frames") or 0)
        fps = float(session.get("video_fps") or 25.0)
        duration_sec = round(total_frames / fps, 1) if fps > 0 else 0.0

        match_meta = {
            "session_id": session_id,
            "total_frames": total_frames,
            "fps": fps,
            "duration_sec": duration_sec,
            "resolution": session.get("video_resolution") or "1920x1080",
        }

        # Extract executive match KPIs from cache_data if available
        cache = cache_data or {}
        kpis: Dict[str, Any] = {}

        team_ctrl = cache.get("team_control", [])
        if team_ctrl:
            t1 = sum(1 for c in team_ctrl if c == 1)
            t2 = sum(1 for c in team_ctrl if c == 2)
            tot = t1 + t2 or 1
            kpis["possession_pct"] = {"team1": round(t1 / tot * 100.0, 1), "team2": round(t2 / tot * 100.0, 1)}

        # Check for shot summary
        shot_json_path = output_dir / "shot_xg_summary.json"
        if shot_json_path.exists():
            try:
                with open(shot_json_path, "r", encoding="utf-8") as f:
                    s_data = json.load(f)
                    kpis["shooting"] = {
                        "total_shots": s_data.get("total_shots", 0),
                        "goals": s_data.get("goals", 0),
                        "total_xg": s_data.get("total_xg", 0.0),
                    }
            except Exception:
                pass

        # Check for PPDA summary
        ppda_json_path = output_dir / "pressing_ppda_summary.json"
        if ppda_json_path.exists():
            try:
                with open(ppda_json_path, "r", encoding="utf-8") as f:
                    p_data = json.load(f)
                    t1_p = p_data.get("team1") or {}
                    t2_p = p_data.get("team2") or {}
                    kpis["pressing_ppda"] = {
                        "team1_ppda": t1_p.get("ppda", 12.0),
                        "team2_ppda": t2_p.get("ppda", 12.0),
                    }
            except Exception:
                pass

        manifest = MatchDossierManifest(
            session_id=session_id,
            generated_at=datetime.datetime.utcnow().isoformat() + "Z",
            match_metadata=match_meta,
            executive_kpis=kpis,
            artifacts_count=len(artifacts_entries),
            total_size_bytes=total_size,
            artifacts=[a.to_dict() for a in artifacts_entries],
        )
        return manifest

    def package_bundle(
        self,
        session_id: str,
        output_dir: Path,
        session: Dict[str, Any],
        cache_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, MatchDossierManifest]:
        """
        Compiles manifest.json and packages all match artifacts into
        output_dir / match_analysis_bundle.zip.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.build_manifest(session_id, output_dir, session, cache_data)

        # 1. Write manifest.json to output directory
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)

        # 2. Package everything into zip
        zip_path = output_dir / "match_analysis_bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as zipf:
            # Include manifest.json first
            zipf.write(manifest_path, arcname="manifest.json", compress_type=zipfile.ZIP_DEFLATED)

            # Include each artifact
            for p in sorted(output_dir.iterdir()):
                if p.is_file() and p.name not in ("match_analysis_bundle.zip", "manifest.json") and not p.name.startswith("."):
                    # Use ZIP_STORED for pre-compressed videos, ZIP_DEFLATED for others
                    compress_mode = zipfile.ZIP_STORED if p.name.endswith(".mp4") else zipfile.ZIP_DEFLATED
                    zipf.write(p, arcname=p.name, compress_type=compress_mode)

        return zip_path, manifest
