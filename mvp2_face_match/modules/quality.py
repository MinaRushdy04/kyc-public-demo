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


def glare_ratio(image_bgr: np.ndarray, value_threshold: int = 245, saturation_threshold: int = 35) -> float:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    glare = (value >= int(value_threshold)) & (saturation <= int(saturation_threshold))
    return float(np.mean(glare))


def landmark_pose_warnings(detection: FaceDetection, cfg: dict, prefix: str) -> list[str]:
    warnings: list[str] = []
    landmarks = np.asarray(detection.landmarks, dtype=np.float32)
    if landmarks.shape != (5, 2):
        return [f"{prefix}_landmarks_invalid"]

    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
    eye_dx = float(right_eye[0] - left_eye[0])
    eye_dy = float(right_eye[1] - left_eye[1])
    eye_distance = float(np.hypot(eye_dx, eye_dy))
    if eye_distance <= 1e-3:
        return [f"{prefix}_landmarks_invalid"]

    roll = abs(float(np.degrees(np.arctan2(eye_dy, eye_dx))))
    max_roll = float(cfg.get(f"{prefix}_max_roll_degrees", cfg.get("max_face_roll_degrees", 18.0)))
    if roll > max_roll:
        warnings.append(f"{prefix}_face_roll_extreme")

    face_width = max(float(detection.width), 1.0)
    min_eye_ratio = float(cfg.get(f"{prefix}_min_eye_distance_ratio", cfg.get("min_eye_distance_ratio", 0.18)))
    if eye_distance / face_width < min_eye_ratio:
        warnings.append(f"{prefix}_face_pose_or_scale_unstable")

    eye_center_x = float((left_eye[0] + right_eye[0]) / 2.0)
    mouth_center_x = float((left_mouth[0] + right_mouth[0]) / 2.0)
    reference_x = (eye_center_x + mouth_center_x) / 2.0
    yaw_proxy = abs(float(nose[0] - reference_x)) / face_width
    max_yaw_proxy = float(cfg.get(f"{prefix}_max_yaw_proxy", cfg.get("max_yaw_proxy", 0.18)))
    if yaw_proxy > max_yaw_proxy:
        warnings.append(f"{prefix}_face_yaw_extreme")
    return warnings


def assess_id_card_quality(image_bgr: np.ndarray, cfg: dict, prefix: str = "id_card") -> tuple[bool, list[str]]:
    warnings: list[str] = []
    glare = glare_ratio(
        image_bgr,
        int(cfg.get("glare_value_threshold", 245)),
        int(cfg.get("glare_saturation_threshold", 35)),
    )
    warn_threshold = float(cfg.get("id_card_glare_warn_ratio", 0.035))
    reject_threshold = float(cfg.get("id_card_glare_reject_ratio", 0.12))
    if glare >= warn_threshold:
        warnings.append(f"{prefix}_glare_detected:{glare:.4f}")
    if glare >= reject_threshold:
        warnings.append(f"{prefix}_glare_too_high")
    return not any(w.endswith("_glare_too_high") for w in warnings), warnings


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
    warnings.extend(landmark_pose_warnings(detection, cfg, prefix))
    blur = blur_variance(gray)
    bright = brightness(gray)
    glare = glare_ratio(
        face_crop_bgr,
        int(cfg.get("glare_value_threshold", 245)),
        int(cfg.get("glare_saturation_threshold", 35)),
    )
    max_glare = float(cfg.get(f"{specific}_max_glare_ratio", cfg.get("max_face_glare_ratio", 0.10)))
    if blur < min_blur:
        warnings.append(f"{prefix}_face_blurry")
    if bright < min_bright:
        warnings.append(f"{prefix}_face_too_dark")
    if bright > max_bright:
        warnings.append(f"{prefix}_face_too_bright")
    if glare > max_glare:
        warnings.append(f"{prefix}_face_glare_high")
    return len(warnings) == 0, warnings
