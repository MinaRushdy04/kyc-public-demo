from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from .similarity import cosine_to_score, decision_from_score


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def calibration_path_for_cfg(cfg: dict) -> Path | None:
    profile = str(cfg.get("face_model_profile", "")).lower()
    path = cfg.get(f"face_calibration_path_{profile}") if profile else None
    path = path or cfg.get("face_calibration_path")
    if not path:
        return None
    return Path(str(path))


@lru_cache(maxsize=8)
def _load_json_cached(path: str, mtime_ns: int) -> dict:
    del mtime_ns
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_calibration(cfg: dict) -> dict | None:
    path = calibration_path_for_cfg(cfg)
    if path is None or not path.is_file():
        return None
    try:
        data = _load_json_cached(str(path), path.stat().st_mtime_ns)
    except Exception:
        return None
    expected_profile = str(cfg.get("face_model_profile", "")).lower()
    data_profile = str(data.get("model_profile", "")).lower()
    if expected_profile and data_profile and expected_profile != data_profile:
        return None
    return data


def decision_thresholds(cfg: dict) -> dict:
    calibration = load_calibration(cfg)
    thresholds = {}
    if calibration:
        runtime_mode = str(cfg.get("face_runtime_score_mode", "raw_score_with_calibrated_thresholds"))
        if runtime_mode == "calibrated_match_probability":
            thresholds = calibration.get("probability_thresholds") or calibration.get("thresholds", {})
        else:
            thresholds = calibration.get("raw_score_thresholds") or calibration.get("thresholds", {})
    return {
        "auto_reject_below": _safe_float(thresholds.get("auto_reject_below"), cfg.get("auto_reject_below", 50)),
        "auto_accept_at_or_above": _safe_float(
            thresholds.get("auto_accept_at_or_above"),
            cfg.get("auto_accept_at_or_above", 70),
        ),
    }


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def calibrated_score_from_cosine(cosine: float, calibration: dict) -> float | None:
    calibrator = calibration.get("calibrator", {})
    if calibrator.get("type") != "logistic_cosine":
        return None
    coef = _safe_float(calibrator.get("coef"), 0.0)
    intercept = _safe_float(calibrator.get("intercept"), 0.0)
    probability = _sigmoid((coef * float(cosine)) + intercept)
    return round(max(0.0, min(100.0, probability * 100.0)), 4)


def score_similarity(cosine: float, cfg: dict) -> dict:
    raw_score = cosine_to_score(cosine)
    calibration = load_calibration(cfg)
    thresholds = decision_thresholds(cfg)
    score = raw_score
    mode = "raw_cosine_score"
    calibrated = False
    calibration_name = None

    if calibration:
        runtime_mode = str(cfg.get("face_runtime_score_mode", "raw_score_with_calibrated_thresholds"))
        calibration_name = calibration.get("name") or calibration.get("calibration_id")
        calibrated = True
        if runtime_mode == "calibrated_match_probability":
            calibrated_score = calibrated_score_from_cosine(cosine, calibration)
            if calibrated_score is None:
                calibrated = False
            else:
                score = calibrated_score
                mode = "calibrated_match_probability"
        else:
            score = raw_score
            mode = "raw_score_with_calibrated_thresholds"

    status, decision = decision_from_score(
        score,
        thresholds["auto_reject_below"],
        thresholds["auto_accept_at_or_above"],
    )
    return {
        "score": round(float(score), 4),
        "raw_score": round(float(raw_score), 4),
        "score_mode": mode,
        "calibrated": calibrated,
        "calibration_name": calibration_name,
        "thresholds": thresholds,
        "status": status,
        "decision": decision,
    }
