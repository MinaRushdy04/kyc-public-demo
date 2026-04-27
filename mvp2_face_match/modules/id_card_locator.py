from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


@dataclass
class IDCardCandidate:
    bbox: list[int]
    crop_bgr: np.ndarray
    area_ratio: float
    aspect_ratio: float
    strategy: str
    national_id_probability: float = 0.0


def order_points(points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_warp(image_bgr: np.ndarray, points: np.ndarray) -> np.ndarray:
    rect = order_points(points)
    tl, tr, br, bl = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_width = max(64, int(max(width_a, width_b)))
    max_height = max(64, int(max(height_a, height_b)))
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image_bgr, matrix, (max_width, max_height))


def _candidate_from_box(image_bgr: np.ndarray, box: np.ndarray, strategy: str) -> IDCardCandidate | None:
    h, w = image_bgr.shape[:2]
    xs = np.clip(box[:, 0], 0, w - 1)
    ys = np.clip(box[:, 1], 0, h - 1)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    if x2 <= x1 or y2 <= y1:
        return None
    crop = four_point_warp(image_bgr, box)
    ch, cw = crop.shape[:2]
    aspect = max(cw / max(ch, 1), ch / max(cw, 1))
    area_ratio = ((x2 - x1) * (y2 - y1)) / max(w * h, 1)
    return IDCardCandidate(
        bbox=[x1, y1, x2, y2],
        crop_bgr=crop,
        area_ratio=float(area_ratio),
        aspect_ratio=float(aspect),
        strategy=strategy,
    )


def _candidate_from_bbox(image_bgr: np.ndarray, bbox: tuple[int, int, int, int], strategy: str) -> IDCardCandidate | None:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image_bgr[y1 : y2 + 1, x1 : x2 + 1].copy()
    ch, cw = crop.shape[:2]
    aspect = max(cw / max(ch, 1), ch / max(cw, 1))
    area_ratio = ((x2 - x1) * (y2 - y1)) / max(w * h, 1)
    return IDCardCandidate(
        bbox=[x1, y1, x2, y2],
        crop_bgr=crop,
        area_ratio=float(area_ratio),
        aspect_ratio=float(aspect),
        strategy=strategy,
    )


def _add_unique(candidates: list[IDCardCandidate], seen: set, candidate: IDCardCandidate | None) -> None:
    if candidate is None:
        return
    key = tuple(int(round(v / 8.0)) for v in candidate.bbox)
    if key in seen:
        return
    seen.add(key)
    candidates.append(candidate)


def find_card_candidates(image_bgr: np.ndarray, cfg: dict) -> list[IDCardCandidate]:
    h, w = image_bgr.shape[:2]
    min_area = float(cfg.get("id_locator_min_area_ratio", 0.005)) * w * h
    max_area = float(cfg.get("id_locator_max_area_ratio", 0.85)) * w * h
    min_aspect = float(cfg.get("id_locator_min_aspect", 1.25))
    max_aspect = float(cfg.get("id_locator_max_aspect", 2.25))
    max_candidates = int(cfg.get("id_locator_max_candidates", 12))

    candidates: list[IDCardCandidate] = []
    seen = set()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variants = [
        ("canny_soft", cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 25, 100)),
        ("canny_normal", cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 140)),
        ("canny_hard", cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 70, 180)),
    ]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    for strategy, edges in variants:
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            rect = cv2.minAreaRect(contour)
            (_, _), (rw, rh), _ = rect
            if rw <= 0 or rh <= 0:
                continue
            aspect = max(rw / rh, rh / rw)
            if aspect < min_aspect or aspect > max_aspect:
                continue
            _add_unique(candidates, seen, _candidate_from_box(image_bgr, cv2.boxPoints(rect), strategy))
            x, y, bw, bh = cv2.boundingRect(contour)
            _add_unique(candidates, seen, _candidate_from_bbox(image_bgr, (x, y, x + bw, y + bh), f"{strategy}_upright_bbox"))

    # Last-resort sliding regions for webcam images where the card edge is weak.
    for scale in (0.28, 0.35, 0.45, 0.55, 0.70):
        crop_w = int(w * scale)
        crop_h = int(crop_w / 1.58)
        if crop_h <= 0 or crop_w <= 0 or crop_h > h:
            continue
        for cy in (0.45, 0.60, 0.75, 0.85):
            for cx in (0.35, 0.50, 0.65, 0.80, 0.90):
                x1 = int(w * cx - crop_w / 2)
                y1 = int(h * cy - crop_h / 2)
                _add_unique(candidates, seen, _candidate_from_bbox(image_bgr, (x1, y1, x1 + crop_w, y1 + crop_h), "sliding_card_region"))
    candidates = [
        c
        for c in candidates
        if min_aspect <= c.aspect_ratio <= max_aspect and min_area <= c.area_ratio * w * h <= max_area
    ]
    candidates.sort(key=lambda c: (c.area_ratio, -abs(c.aspect_ratio - 1.58)), reverse=True)
    candidates = candidates[: max_candidates * 2]
    return candidates


def candidate_selection_score(candidate: IDCardCandidate, cfg: dict) -> float:
    area_weight = float(cfg.get("id_locator_area_penalty_weight", 0.08))
    aspect_weight = float(cfg.get("id_locator_aspect_penalty_weight", 0.02))
    target_aspect = float(cfg.get("id_locator_target_aspect", 1.58))
    area_penalty = candidate.area_ratio * area_weight
    aspect_penalty = abs(candidate.aspect_ratio - target_aspect) * aspect_weight
    return float(candidate.national_id_probability) - area_penalty - aspect_penalty


def face_anchor_card_candidates(image_bgr: np.ndarray, face_bboxes: list[list[float]]) -> list[IDCardCandidate]:
    candidates: list[IDCardCandidate] = []
    seen = set()
    for bbox in face_bboxes:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        fw, fh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        fc_x, fc_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        for multiplier in (6.0, 7.5, 9.0, 10.5, 12.0):
            card_w = max(fw, fh) * multiplier
            card_h = card_w / 1.58
            for rel_x in (0.22, 0.28, 0.34, 0.42):
                for rel_y in (0.42, 0.50, 0.58):
                    left = fc_x - card_w * rel_x
                    top = fc_y - card_h * rel_y
                    candidate = _candidate_from_bbox(
                        image_bgr,
                        (int(left), int(top), int(left + card_w), int(top + card_h)),
                        "id_portrait_face_anchor",
                    )
                    _add_unique(candidates, seen, candidate)
    return candidates


def locate_id_card(
    image_bgr: np.ndarray,
    cfg: dict,
    score_fn: Callable[[np.ndarray], float],
) -> tuple[IDCardCandidate | None, list[IDCardCandidate], list[str]]:
    warnings: list[str] = []
    candidates = find_card_candidates(image_bgr, cfg)
    for candidate in candidates:
        candidate.national_id_probability = float(score_fn(candidate.crop_bgr))
    candidates.sort(key=lambda c: candidate_selection_score(c, cfg), reverse=True)

    threshold = float(cfg.get("mvp1_accept_threshold", cfg.get("accept_threshold", 0.95)))
    if candidates and candidates[0].national_id_probability >= threshold:
        return candidates[0], candidates, warnings

    if str(cfg.get("id_locator_fallback_full_image", "true")).lower() == "true":
        prob = float(score_fn(image_bgr))
        if prob >= threshold:
            h, w = image_bgr.shape[:2]
            warnings.append("LOW_CONFIDENCE_ID_LOCALIZATION")
            return (
                IDCardCandidate(
                    bbox=[0, 0, w - 1, h - 1],
                    crop_bgr=image_bgr.copy(),
                    area_ratio=1.0,
                    aspect_ratio=max(w / max(h, 1), h / max(w, 1)),
                    strategy="fallback_full_image",
                    national_id_probability=prob,
                ),
                candidates,
                warnings,
            )
        if not candidates:
            warnings.append("NO_RECTANGULAR_ID_CANDIDATES")
        else:
            warnings.append("ALL_ID_CANDIDATES_REJECTED_BY_MVP1")
    return None, candidates, warnings
