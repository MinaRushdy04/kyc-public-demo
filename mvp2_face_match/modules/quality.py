from __future__ import annotations

import cv2
import numpy as np

from .face_detector import FaceDetection


def brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))


def blur_variance(gray: np.ndarray) -> float:
    h, w = gray.shape[:2]
    scale = min(1.0, 256.0 / max(h, w, 1))
    if scale < 1.0:
        gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def assess_face_quality(face_crop_bgr: np.ndarray, detection: FaceDetection, cfg: dict, prefix: str) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    specific = "id_portrait" if prefix.startswith("id_portrait") else prefix
    min_face_size = float(cfg.get(f"{specific}_min_face_size_px", cfg.get("min_face_size_px", 48)))
    min_conf = float(cfg.get(f"{specific}_min_detector_confidence", cfg.get("min_detector_confidence", 0.45)))
    min_blur = float(cfg.get(f"{specific}_min_blur_variance", cfg.get("min_blur_variance", 18.0)))
    min_bright = float(cfg.get("min_brightness", 35.0))
    max_bright = float(cfg.get("max_brightness", 225.0))

    if min(detection.width, detection.height) < min_face_size:
        warnings.append(f"{prefix}_face_too_small")
    if detection.confidence < min_conf:
        warnings.append(f"{prefix}_detector_confidence_low")
    blur = blur_variance(gray)
    bright = brightness(gray)
    if blur < min_blur:
        warnings.append(f"{prefix}_face_blurry")
    if bright < min_bright:
        warnings.append(f"{prefix}_face_too_dark")
    if bright > max_bright:
        warnings.append(f"{prefix}_face_too_bright")
    return len(warnings) == 0, warnings
