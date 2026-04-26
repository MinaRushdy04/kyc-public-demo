from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    return float(np.dot(a, b) / denom)


def cosine_to_score(cosine: float) -> float:
    return round(max(0.0, min(100.0, ((float(cosine) + 1.0) / 2.0) * 100.0)), 4)


def decision_from_score(score: float, auto_reject_below: float, auto_accept_at_or_above: float) -> tuple[str, str]:
    if score < auto_reject_below:
        return "FACE_MISMATCH", "AUTO_REJECT"
    if score < auto_accept_at_or_above:
        return "FACE_UNCERTAIN", "MANUAL_REVIEW"
    return "FACE_MATCH", "AUTO_ACCEPT"
