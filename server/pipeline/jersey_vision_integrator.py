"""
jersey_vision_integrator.py — Tracklet Jersey Vision Spotter & Pipeline Integrator

Connects torso visual extraction with TrackletJerseyVotingEngine to:
1. Select high-clarity keyframes (largest bounding box area + vertical aspect ratio).
2. Extract upper-torso ROIs and apply contrast normalization.
3. Aggregate multi-frame visual digit observations using temporal consensus voting.
4. Annotate tracks["players"] in place with resolved jersey numbers and confidence.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False

from .jersey_voting import (
    TrackletJerseyVotingEngine,
    extract_torso_roi,
    sanitize_jersey_number,
)


class JerseyVisionIntegrator:
    """
    Orchestrates keyframe selection, torso extraction, and temporal consensus
    voting to resolve player jersey numbers across video tracklets.
    """

    def __init__(
        self,
        max_keyframes_per_player: int = 8,
        min_bbox_height: float = 30.0,
        decay_rate: float = 0.001,
        torso_ratio: float = 0.45,
    ) -> None:
        self.max_keyframes = max_keyframes_per_player
        self.min_bbox_height = float(min_bbox_height)
        self.decay_rate = float(decay_rate)
        self.torso_ratio = float(torso_ratio)
        self.engine = TrackletJerseyVotingEngine(decay_rate=self.decay_rate)

    def select_keyframe_candidates(
        self,
        tracks: Dict[str, Any],
    ) -> Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]]:
        """
        Selects top-K highest clarity frames for each player tracklet.
        Returns: {player_id: [(frame_idx, (x1, y1, x2, y2)), ...]}
        """
        player_frames = tracks.get("players", [])
        tracklet_candidates: Dict[int, List[Tuple[float, int, Tuple[float, float, float, float]]]] = {}

        for fi, f_dict in enumerate(player_frames):
            if not f_dict:
                continue
            for pid, info in f_dict.items():
                if not info or "bbox" not in info:
                    continue
                bbox = info["bbox"]
                if len(bbox) != 4:
                    continue
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if h < self.min_bbox_height or w <= 0:
                    continue

                # Player aspect ratio (typical upright athlete: 0.25 to 0.65)
                aspect = w / h
                if not (0.20 <= aspect <= 0.80):
                    continue

                # Score based on area (larger crop = higher resolution digits)
                area = w * h
                tracklet_candidates.setdefault(pid, []).append((area, fi, tuple(bbox)))

        # Sort by area descending and pick top-K
        selected: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
        for pid, cand_list in tracklet_candidates.items():
            cand_list.sort(key=lambda x: x[0], reverse=True)
            selected[pid] = [(fi, bbox) for _, fi, bbox in cand_list[:self.max_keyframes]]

        return selected

    def extract_torso_patch(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """Extracts and contrast-enhances the upper-torso region of a player."""
        if not _HAS_CV2 or frame is None:
            return None

        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = extract_torso_roi(bbox, self.torso_ratio)
        ix1 = max(0, min(w_img - 1, int(x1)))
        iy1 = max(0, min(h_img - 1, int(y1)))
        ix2 = max(0, min(w_img, int(x2)))
        iy2 = max(0, min(h_img, int(y2)))

        if ix2 <= ix1 or iy2 <= iy1:
            return None

        crop = frame[iy1:iy2, ix1:ix2]
        if crop.size == 0:
            return None

        # Standardize size & normalize contrast for digit readability
        target_h, target_w = 64, 48
        resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        return enhanced

    def record_observation(
        self,
        track_id: int,
        frame_idx: int,
        raw_number: Any,
        confidence: float,
    ) -> bool:
        """Sanitizes and registers a candidate jersey number observation."""
        num = sanitize_jersey_number(str(raw_number))
        if num is not None:
            self.engine.add_observation(track_id, frame_idx, num, confidence)
            return True
        return False

    def resolve_all_tracklets(
        self,
        min_support: int = 1,
        confidence_threshold: float = 0.50,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Resolves the consensus jersey number and confidence for all tracked players.
        Returns: {track_id: {"jersey_number": int, "confidence": float, "votes": int, "status": str}}
        """
        results: Dict[int, Dict[str, Any]] = {}
        for pid in list(self.engine._tracklets.keys()):
            res = self.engine.resolve_tracklet(
                pid, min_support=min_support, confidence_threshold=confidence_threshold
            )
            if res.get("number") is not None:
                results[pid] = {
                    "jersey_number": res["number"],
                    "confidence": round(float(res["confidence"]), 3),
                    "votes": res.get("support", 0),
                    "status": res.get("status", "CONFIRMED"),
                }
        return results

    def annotate_tracks(
        self,
        tracks: Dict[str, Any],
        resolved_dict: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Applies resolved jersey numbers to every frame of the player tracklet in tracks["players"].
        Mutates tracks in place and returns the summary dictionary.
        """
        if resolved_dict is None:
            resolved_dict = self.resolve_all_tracklets()

        player_frames = tracks.get("players", [])
        for f_dict in player_frames:
            if not f_dict:
                continue
            for pid, info in f_dict.items():
                if pid in resolved_dict:
                    res = resolved_dict[pid]
                    info["jersey_number"] = res["jersey_number"]
                    info["jersey_confidence"] = res["confidence"]

        return resolved_dict
