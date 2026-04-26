from __future__ import annotations

import cv2
import numpy as np


def _resize_by_scale(image_bgr: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 1.0:
        return image_bgr.copy()
    h, w = image_bgr.shape[:2]
    return cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def _resize_to_min_width(image_bgr: np.ndarray, min_width: int, max_scale: float) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if w >= min_width:
        return image_bgr.copy()
    scale = min(float(max_scale), float(min_width) / max(float(w), 1.0))
    return _resize_by_scale(image_bgr, scale)


def clahe_bgr(image_bgr: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_chan)
    return cv2.cvtColor(cv2.merge((enhanced_l, a_chan, b_chan)), cv2.COLOR_LAB2BGR)


def unsharp_mask(image_bgr: np.ndarray, amount: float = 0.55, sigma: float = 1.1) -> np.ndarray:
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), float(sigma))
    return cv2.addWeighted(image_bgr, 1.0 + float(amount), blurred, -float(amount), 0)


def mild_denoise(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image_bgr, None, 3, 3, 7, 21)


def id_card_enhancement_variants(image_bgr: np.ndarray, cfg: dict) -> list[tuple[str, np.ndarray]]:
    min_width = int(cfg.get("id_portrait_enhance_min_card_width", 700))
    max_scale = float(cfg.get("id_portrait_enhance_max_scale", 3.0))
    base = _resize_to_min_width(image_bgr, min_width=min_width, max_scale=max_scale)
    variants: list[tuple[str, np.ndarray]] = []

    def add(name: str, image: np.ndarray) -> None:
        if image is not None and image.size:
            variants.append((name, image))

    add("original", image_bgr.copy())
    if base.shape[:2] != image_bgr.shape[:2]:
        add("upscaled", base)

    contrast = clahe_bgr(base, clip_limit=float(cfg.get("id_portrait_clahe_clip_limit", 2.0)))
    add("clahe", contrast)
    add("clahe_sharp", unsharp_mask(contrast, amount=float(cfg.get("id_portrait_sharp_amount", 0.55))))

    denoised = mild_denoise(base)
    add("denoise_clahe_sharp", unsharp_mask(clahe_bgr(denoised), amount=0.45))

    return variants
