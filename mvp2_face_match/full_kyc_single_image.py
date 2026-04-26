from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from modules.face_aligner import align_face, crop_bbox
from modules.face_detector import SCRFDFaceDetector, read_bgr
from modules.face_embedder import ArcFaceEmbedder
from modules.id_card_locator import IDCardCandidate, face_anchor_card_candidates, locate_id_card
from modules.image_enhancement import id_card_enhancement_variants
from modules.quality import assess_face_quality
from modules.response_schema import LIMITATIONS
from modules.score_calibration import decision_thresholds, score_similarity
from modules.similarity import cosine_similarity
from modules.utils import apply_face_model_profile, ensure_dir, load_config, project_relative_default_config


SINGLE_IMAGE_LIMITATIONS = LIMITATIONS + [
    "ID localization is MVP-level and may fail if the card is too small, blurry, occluded, or reflective.",
]


def _load_python_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MVP1DocumentValidator:
    def __init__(self, cfg: dict):
        project_dir = Path(str(cfg["project_dir"]))
        mvp1_cfg = load_config(str(cfg.get("mvp1_config_path", project_dir / "config.yaml")))
        mvp1_cfg.setdefault("image_size", 224)
        mvp1_cfg.setdefault("accept_threshold", cfg.get("mvp1_accept_threshold", 0.95))
        self.threshold = float(cfg.get("mvp1_accept_threshold", mvp1_cfg.get("accept_threshold", 0.95)))
        ckpt_path = Path(str(cfg.get("mvp1_checkpoint_path", project_dir / "models" / "validator_best.pt")))
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"MVP1 checkpoint not found: {ckpt_path}")
        model_module = _load_python_module("mvp1_validator_model_module", project_dir / "modules" / "model.py")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        payload = torch.load(str(ckpt_path), map_location=self.device)
        model_name = str(payload.get("actual_model_name") or payload.get("requested_model_name") or mvp1_cfg.get("model_name", "efficientnetv2_b0"))
        self.model = model_module.build_model(model_name, pretrained=False).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.positive_class_name = str(payload.get("positive_class_name", "national_id"))
        _, self.eval_tf = model_module.get_transforms(int(mvp1_cfg.get("image_size", 224)))

    def probability(self, crop_bgr: np.ndarray) -> float:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        x = self.eval_tf(im).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw = float(torch.sigmoid(self.model(x)).item())
        return raw if self.positive_class_name == "national_id" else 1.0 - raw


def base_response(status: str, final_decision: str, cfg: dict, warnings: list[str]) -> dict:
    thresholds = decision_thresholds(cfg)
    return {
        "status": status,
        "final_decision": final_decision,
        "document": {
            "status": "NO_ID_CARD_FOUND",
            "national_id_probability": 0.0,
            "id_bbox": None,
            "id_crop_path": None,
        },
        "face_match": {
            "status": "NOT_RUN",
            "score": None,
            "cosine_similarity": None,
            "raw_score": None,
            "score_mode": "not_run",
            "calibrated": False,
            "thresholds": thresholds,
        },
        "quality": {"warnings": warnings},
        "limitations": SINGLE_IMAGE_LIMITATIONS,
        "mvp_version": "MVP2_SINGLE_IMAGE",
    }


def _bbox_intersection_ratio(face_bbox: np.ndarray, id_bbox: list[int]) -> float:
    fx1, fy1, fx2, fy2 = [float(v) for v in face_bbox]
    ix1, iy1, ix2, iy2 = [float(v) for v in id_bbox]
    x1, y1 = max(fx1, ix1), max(fy1, iy1)
    x2, y2 = min(fx2, ix2), min(fy2, iy2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    face_area = max(1.0, (fx2 - fx1) * (fy2 - fy1))
    return inter / face_area


def _face_center_inside(face_bbox: np.ndarray, id_bbox: list[int]) -> bool:
    x1, y1, x2, y2 = [float(v) for v in face_bbox]
    ix1, iy1, ix2, iy2 = [float(v) for v in id_bbox]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return ix1 <= cx <= ix2 and iy1 <= cy <= iy2


def _select_holder_face(faces, id_bbox: list[int], image_shape: tuple[int, int], cfg: dict):
    h, w = image_shape
    image_area = max(1, h * w)
    outside = [
        f
        for f in faces
        if not _face_center_inside(f.bbox, id_bbox) and _bbox_intersection_ratio(f.bbox, id_bbox) < 0.20
    ]
    outside.sort(key=lambda f: (f.area, f.confidence), reverse=True)
    large_ratio = float(cfg.get("holder_multiple_face_relative_area", 0.01))
    large = [f for f in outside if f.area / image_area >= large_ratio]
    if len(large) > 1:
        return None, len(large), "MULTIPLE_SELFIE_FACES"
    if large:
        return large[0], 1, ""
    if outside:
        return outside[0], 1, ""
    return None, 0, "NO_HOLDER_FACE"


def _candidate_allows_holder_face(candidate: IDCardCandidate, faces, image_shape: tuple[int, int], cfg: dict) -> bool:
    holder_face, _, error = _select_holder_face(faces, candidate.bbox, image_shape, cfg)
    return holder_face is not None and not error


def _choose_candidate_that_preserves_holder(
    selected: IDCardCandidate,
    candidates: list[IDCardCandidate],
    faces,
    image_shape: tuple[int, int],
    cfg: dict,
) -> tuple[IDCardCandidate, str | None]:
    threshold = float(cfg.get("mvp1_accept_threshold", 0.95))
    if not faces:
        return selected, None
    largest_face = max(faces, key=lambda f: f.area)
    ordered = [selected] + [c for c in candidates if c is not selected]
    for candidate in ordered:
        if candidate.national_id_probability < threshold:
            continue
        if _face_center_inside(largest_face.bbox, candidate.bbox) or _bbox_intersection_ratio(largest_face.bbox, candidate.bbox) > 0.15:
            continue
        if _candidate_allows_holder_face(candidate, faces, image_shape, cfg):
            if candidate is not selected:
                return candidate, "ID_CANDIDATE_RESELECTED_TO_KEEP_HOLDER_FACE"
            return selected, None
    return selected, None


def _save_debug_crop(path: Path, image_bgr: np.ndarray | None) -> str | None:
    if image_bgr is None:
        return None
    ensure_dir(path.parent)
    cv2.imwrite(str(path), image_bgr)
    return str(path)


def _quality_candidate_score(face, crop_bgr: np.ndarray, aligned_bgr: np.ndarray | None, quality_ok: bool, warnings: list[str]) -> float:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    size = min(float(face.width), float(face.height))
    score = float(face.confidence) * 240.0 + min(size, 180.0) * 1.3 + min(blur, 80.0) * 1.2
    if quality_ok:
        score += 500.0
    if aligned_bgr is None:
        score -= 350.0
    score -= len(warnings) * 45.0
    return score


def _detect_best_id_portrait(
    id_crop_bgr: np.ndarray,
    detector: SCRFDFaceDetector,
    cfg: dict,
    debug_root: Path,
) -> dict | None:
    best: dict | None = None
    attempts: list[dict] = []
    for variant_name, variant_bgr in id_card_enhancement_variants(id_crop_bgr, cfg):
        faces = detector.detect(variant_bgr)
        attempts.append(
            {
                "variant": variant_name,
                "shape": list(variant_bgr.shape[:2]),
                "faces": [
                    {
                        "confidence": round(float(face.confidence), 4),
                        "bbox": [round(float(v), 2) for v in face.bbox],
                        "size": [round(float(face.width), 2), round(float(face.height), 2)],
                    }
                    for face in faces[:5]
                ],
            }
        )
        for face in faces[:3]:
            face_crop = crop_bbox(variant_bgr, face.bbox)
            quality_ok, quality_warnings = assess_face_quality(face_crop, face, cfg, "id_portrait")
            aligned, align_error = align_face(variant_bgr, face.landmarks)
            candidate_warnings = list(quality_warnings)
            if aligned is None:
                candidate_warnings.append(f"id_portrait_alignment_failed:{align_error}")
            score = _quality_candidate_score(face, face_crop, aligned, quality_ok, candidate_warnings)
            candidate = {
                "variant": variant_name,
                "variant_bgr": variant_bgr,
                "face": face,
                "face_crop": face_crop,
                "aligned": aligned,
                "quality_ok": quality_ok,
                "warnings": candidate_warnings,
                "score": score,
            }
            if best is None or score > float(best["score"]):
                best = candidate

    (debug_root / "id_portrait_detection_attempts.json").write_text(json.dumps(attempts, indent=2), encoding="utf-8")
    if best is not None and str(best["variant"]) != "original":
        _save_debug_crop(debug_root / f"id_card_enhanced_{best['variant']}.jpg", best["variant_bgr"])
    return best


def load_face_models(cfg: dict) -> tuple[SCRFDFaceDetector, ArcFaceEmbedder]:
    detector = SCRFDFaceDetector(
        cfg["scrfd_model_path"],
        input_size=int(cfg.get("det_input_size", 640)),
        confidence_threshold=float(cfg.get("det_confidence_threshold", 0.45)),
        nms_threshold=float(cfg.get("nms_threshold", 0.4)),
    )
    embedder = ArcFaceEmbedder(cfg["arcface_model_path"])
    return detector, embedder


def run_single_image_pipeline(
    image_path: str | Path,
    cfg: dict,
    document_validator: MVP1DocumentValidator | None = None,
    detector: SCRFDFaceDetector | None = None,
    embedder: ArcFaceEmbedder | None = None,
    debug_dir: str | Path | None = None,
    return_debug: bool = False,
) -> dict:
    warnings: list[str] = []
    try:
        full_bgr = read_bgr(image_path)
    except Exception:
        return base_response("INVALID_IMAGE", "REJECT_RETRY", cfg, ["invalid_or_unreadable_image"])

    document_validator = document_validator or MVP1DocumentValidator(cfg)
    detector = detector or load_face_models(cfg)[0]
    embedder = embedder or ArcFaceEmbedder(cfg["arcface_model_path"])
    debug_root = Path(debug_dir) if debug_dir else Path(str(cfg["face_project_dir"])) / "outputs" / "debug" / "single_image_latest"
    ensure_dir(debug_root)

    selected, candidates, loc_warnings = locate_id_card(full_bgr, cfg, document_validator.probability)
    warnings.extend(loc_warnings)
    result = base_response("NO_ID_CARD_FOUND", "REJECT_RETRY", cfg, warnings)
    result["_candidate_count"] = len(candidates)
    for idx, candidate in enumerate(candidates[:8]):
        _save_debug_crop(debug_root / f"id_candidate_{idx:02d}_{candidate.national_id_probability:.4f}.jpg", candidate.crop_bgr)

    if selected is None:
        if return_debug:
            result["_debug"] = {"full_bgr": full_bgr, "id_crop": None, "holder_crop": None, "id_face_crop": None}
        return result

    full_faces = detector.detect(full_bgr)
    if full_faces:
        largest_face = max(full_faces, key=lambda f: f.area)
        small_face_bboxes = [
            [float(v) for v in f.bbox]
            for f in full_faces
            if f is not largest_face and f.area < largest_face.area * 0.35
        ]
        anchor_candidates = face_anchor_card_candidates(full_bgr, small_face_bboxes)
        for candidate in anchor_candidates:
            candidate.national_id_probability = float(document_validator.probability(candidate.crop_bgr))
        candidates.extend(anchor_candidates)
        candidates.sort(key=lambda c: c.national_id_probability, reverse=True)
        for idx, candidate in enumerate(anchor_candidates[:8]):
            _save_debug_crop(debug_root / f"id_anchor_candidate_{idx:02d}_{candidate.national_id_probability:.4f}.jpg", candidate.crop_bgr)
    selected, reselect_warning = _choose_candidate_that_preserves_holder(
        selected,
        candidates,
        full_faces,
        full_bgr.shape[:2],
        cfg,
    )
    if reselect_warning:
        warnings.append(reselect_warning)

    id_crop_path = _save_debug_crop(debug_root / "id_card_crop.jpg", selected.crop_bgr)
    doc_status = "NATIONAL_ID_ACCEPTED" if selected.national_id_probability >= float(cfg.get("mvp1_accept_threshold", 0.95)) else "DOCUMENT_REJECTED"
    result["document"] = {
        "status": doc_status,
        "national_id_probability": round(float(selected.national_id_probability), 6),
        "id_bbox": selected.bbox,
        "id_crop_path": id_crop_path,
    }

    if doc_status != "NATIONAL_ID_ACCEPTED":
        result["status"] = "DOCUMENT_REJECTED"
        if return_debug:
            result["_debug"] = {"full_bgr": full_bgr, "id_crop": selected.crop_bgr, "holder_crop": None, "id_face_crop": None}
        return result

    holder_face, holder_count, holder_error = _select_holder_face(full_faces, selected.bbox, full_bgr.shape[:2], cfg)
    if holder_error:
        result["status"] = holder_error
        result["final_decision"] = "REJECT_RETRY"
        warnings.append(holder_error.lower())
        result["quality"]["warnings"] = warnings
        if return_debug:
            result["_debug"] = {"full_bgr": full_bgr, "id_crop": selected.crop_bgr, "holder_crop": None, "id_face_crop": None}
        return result

    id_portrait = _detect_best_id_portrait(selected.crop_bgr, detector, cfg, debug_root)
    if id_portrait is None:
        result["status"] = "NO_FACE_IN_ID"
        result["final_decision"] = "REJECT_RETRY"
        warnings.append("no_face_in_id_portrait")
        result["quality"]["warnings"] = warnings
        if return_debug:
            holder_crop = crop_bbox(full_bgr, holder_face.bbox)
            result["_debug"] = {"full_bgr": full_bgr, "id_crop": selected.crop_bgr, "holder_crop": holder_crop, "id_face_crop": None}
        return result

    holder_crop = crop_bbox(full_bgr, holder_face.bbox)
    id_face_crop = id_portrait["face_crop"]
    _save_debug_crop(debug_root / "holder_face_crop.jpg", holder_crop)
    _save_debug_crop(debug_root / "id_portrait_face_crop.jpg", id_face_crop)
    if str(id_portrait["variant"]) != "original":
        warnings.append(f"id_portrait_enhancement_used:{id_portrait['variant']}")

    holder_quality_ok, holder_warnings = assess_face_quality(holder_crop, holder_face, cfg, "holder")
    id_quality_ok = bool(id_portrait["quality_ok"])
    id_warnings = list(id_portrait["warnings"])
    warnings.extend(holder_warnings + id_warnings)

    holder_aligned, holder_align_error = align_face(full_bgr, holder_face.landmarks)
    id_aligned = id_portrait["aligned"]
    if holder_aligned is None:
        warnings.append(f"holder_alignment_failed:{holder_align_error}")
    _save_debug_crop(debug_root / "holder_face_aligned.jpg", holder_aligned)
    _save_debug_crop(debug_root / "id_portrait_face_aligned.jpg", id_aligned)

    if not id_quality_ok or id_aligned is None:
        result["status"] = "LOW_QUALITY_ID_FACE"
        result["final_decision"] = "REJECT_RETRY"
        result["face_match"]["status"] = "NOT_RUN"
        result["quality"]["warnings"] = warnings
        if return_debug:
            result["_debug"] = {"full_bgr": full_bgr, "id_crop": selected.crop_bgr, "holder_crop": holder_crop, "id_face_crop": id_face_crop}
        return result
    if not holder_quality_ok or holder_aligned is None:
        result["status"] = "LOW_QUALITY_SELFIE_FACE"
        result["final_decision"] = "REJECT_RETRY"
        result["face_match"]["status"] = "NOT_RUN"
        result["quality"]["warnings"] = warnings
        if return_debug:
            result["_debug"] = {"full_bgr": full_bgr, "id_crop": selected.crop_bgr, "holder_crop": holder_crop, "id_face_crop": id_face_crop}
        return result

    id_emb = embedder.embed(id_aligned)
    holder_emb = embedder.embed(holder_aligned)
    cosine = cosine_similarity(id_emb, holder_emb)
    scoring = score_similarity(cosine, cfg)
    face_status = scoring["status"]
    face_decision = scoring["decision"]
    result["face_match"] = {
        "status": face_status,
        "score": scoring["score"],
        "cosine_similarity": round(float(cosine), 6),
        "raw_score": scoring["raw_score"],
        "score_mode": scoring["score_mode"],
        "calibrated": scoring["calibrated"],
        "calibration_name": scoring["calibration_name"],
        "thresholds": scoring["thresholds"],
    }
    if face_decision == "AUTO_ACCEPT":
        result["status"] = "FINAL_ACCEPT"
        result["final_decision"] = "ACCEPT_FOR_NEXT_STEP"
    elif face_decision == "MANUAL_REVIEW":
        result["status"] = "FACE_UNCERTAIN"
        result["final_decision"] = "MANUAL_REVIEW"
    else:
        result["status"] = "FACE_MISMATCH"
        result["final_decision"] = "REJECT_RETRY"
    result["quality"]["warnings"] = warnings

    if return_debug:
        result["_debug"] = {
            "full_bgr": full_bgr,
            "id_crop": selected.crop_bgr,
            "holder_crop": holder_crop,
            "id_face_crop": id_face_crop,
            "holder_aligned": holder_aligned,
            "id_aligned": id_aligned,
        }
    return result


def _strip_debug(result: dict) -> dict:
    clean = dict(result)
    clean.pop("_debug", None)
    clean.pop("_candidate_count", None)
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MVP2 single-image selfie-holding-ID pipeline.")
    parser.add_argument("--config", default=str(project_relative_default_config()))
    parser.add_argument("--image", required=True)
    parser.add_argument("--debug-dir", default="")
    parser.add_argument("--face-profile", choices=["stable", "accurate"], default=None)
    args = parser.parse_args()

    cfg = apply_face_model_profile(load_config(args.config), args.face_profile)
    result = run_single_image_pipeline(args.image, cfg, debug_dir=args.debug_dir or None)
    print(json.dumps(_strip_debug(result), indent=2))


if __name__ == "__main__":
    main()
