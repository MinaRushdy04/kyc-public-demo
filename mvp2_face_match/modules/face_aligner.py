from __future__ import annotations

import cv2
import numpy as np


ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_face(image_bgr: np.ndarray, landmarks: np.ndarray, output_size: int = 112) -> tuple[np.ndarray | None, str]:
    if landmarks is None or np.asarray(landmarks).shape != (5, 2):
        return None, "missing_or_invalid_landmarks"
    src = np.asarray(landmarks, dtype=np.float32)
    dst = ARCFACE_TEMPLATE.copy()
    if output_size != 112:
        dst *= float(output_size) / 112.0
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        return None, "affine_estimation_failed"
    aligned = cv2.warpAffine(image_bgr, matrix, (output_size, output_size), borderValue=0.0)
    return aligned, ""


def crop_bbox(image_bgr: np.ndarray, bbox: np.ndarray, margin: float = 0.15) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    pad = max(bw, bh) * margin
    x1 = int(max(0, x1 - pad))
    y1 = int(max(0, y1 - pad))
    x2 = int(min(w - 1, x2 + pad))
    y2 = int(min(h - 1, y2 + pad))
    return image_bgr[y1 : y2 + 1, x1 : x2 + 1].copy()
