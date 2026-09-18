"""
jersey_voting.py - Jersey Number Temporal Tracklet Voting Engine
SOTA 2025/2026 Sports Computer Vision Architecture.
Resolves moving player jersey numbers across video tracklets by filtering
upper-torso ROIs and aggregating noisy per-frame OCR predictions using
temporal weighted Dirichlet confidence voting.
Zero external dependencies (pure Python standard library).
"""

from __future__ import annotations
import math
import re
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


def extract_torso_roi(
    bbox: Tuple[float, float, float, float],
    torso_ratio: float = 0.45
) -> Tuple[float, float, float, float]:
    """
    Extracts the upper-torso region of interest from a full player bounding box.
    Filters out legs, feet, and turf distractions.

    Args:
        bbox: (x1, y1, x2, y2) in pixel coordinates.
        torso_ratio: Fractional height allocated to the jersey torso (default: 0.45).

    Returns:
        (torso_x1, torso_y1, torso_x2, torso_y2)
    """
    x1, y1, x2, y2 = bbox
    height = max(0.0, y2 - y1)
    torso_y2 = y1 + height * torso_ratio
    return (x1, y1, x2, torso_y2)


def sanitize_jersey_number(raw_text: str) -> Optional[int]:
    """
    Validates and normalizes candidate jersey OCR text.
    Only valid soccer jersey numbers (1-99) are accepted.

    Args:
        raw_text: Raw string extracted by OCR model.

    Returns:
        Integer jersey number (1-99) or None if invalid/unreadable.
    """
    s = str(raw_text).strip()
    if not s or s.startswith("-"):
        return None
    if s.startswith("#"):
        s = s[1:].strip()
    if not s.isdigit():
        return None
    val = int(s)
    if 1 <= val <= 99:
        return val
    return None


class TrackletJerseyVotingEngine:
    """
    Accumulates noisy single-frame OCR observations per player tracklet and
    computes a temporal consensus jersey number with confidence estimation.
    """

    def __init__(self, decay_rate: float = 0.001):
        """
        Args:
            decay_rate: Temporal decay factor per frame difference to prioritize recency.
        """
        self.decay_rate = decay_rate
        # track_id -> List of (frame_idx, number, confidence)
        self._tracklets: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)

    def add_observation(
        self,
        track_id: int,
        frame_idx: int,
        raw_number: Any,
        confidence: float
    ) -> bool:
        """
        Adds a single-frame detection to the tracklet history.
        """
        if confidence <= 0.0:
            return False

        number = sanitize_jersey_number(str(raw_number))
        if number is None:
            return False

        self._tracklets[track_id].append((frame_idx, number, float(confidence)))
        return True

    def resolve_tracklet(
        self,
        track_id: int,
        min_support: int = 3,
        confidence_threshold: float = 0.60
    ) -> Dict[str, Any]:
        """
        Resolves the consensus jersey number for the specified tracklet.

        Returns:
            Dict containing:
                - number: Confirmed jersey number (int) or None
                - confidence: Aggregated confidence score (0.0 - 1.0)
                - support: Total valid frame observations
                - status: "CONFIRMED" | "TENTATIVE" | "UNRESOLVED"
        """
        obs = self._tracklets.get(track_id, [])
        if not obs:
            return {
                "number": None,
                "confidence": 0.0,
                "support": 0,
                "status": "UNRESOLVED",
            }

        max_frame = max(frame for frame, _, _ in obs)

        # Weighted vote accumulation: conf * exp(-decay * (max_frame - frame))
        vote_scores: Dict[int, float] = defaultdict(float)
        vote_counts: Dict[int, int] = defaultdict(int)
        total_weight = 0.0

        for frame, num, conf in obs:
            delta_t = max(0, max_frame - frame)
            weight = conf * math.exp(-self.decay_rate * delta_t)
            vote_scores[num] += weight
            vote_counts[num] += 1
            total_weight += weight

        if total_weight <= 0.0:
            return {
                "number": None,
                "confidence": 0.0,
                "support": len(obs),
                "status": "UNRESOLVED",
            }

        # Find candidate with highest score
        best_num = max(vote_scores, key=vote_scores.get)
        best_score = vote_scores[best_num]
        normalized_conf = round(best_score / total_weight, 3)
        best_support = vote_counts[best_num]

        if best_support >= min_support and normalized_conf >= confidence_threshold:
            status = "CONFIRMED"
        elif best_support >= 1 and normalized_conf >= 0.30:
            status = "TENTATIVE"
        else:
            status = "UNRESOLVED"

        return {
            "number": best_num,
            "confidence": normalized_conf,
            "support": best_support,
            "status": status,
        }
