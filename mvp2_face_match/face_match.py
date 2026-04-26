from __future__ import annotations

import argparse
import json
from pathlib import Path

from modules.face_aligner import align_face, crop_bbox
from modules.face_detector import SCRFDFaceDetector, read_bgr
from modules.face_embedder import ArcFaceEmbedder
from modules.quality import assess_face_quality
from modules.response_schema import response
from modules.score_calibration import score_similarity
from modules.similarity import cosine_similarity
from modules.utils import apply_face_model_profile, load_config, project_relative_default_config


def load_models(cfg: dict) -> tuple[SCRFDFaceDetector, ArcFaceEmbedder]:
    detector = SCRFDFaceDetector(
        cfg["scrfd_model_path"],
        input_size=int(cfg.get("det_input_size", 640)),
        confidence_threshold=float(cfg.get("det_confidence_threshold", 0.45)),
        nms_threshold=float(cfg.get("nms_threshold", 0.4)),
    )
    embedder = ArcFaceEmbedder(cfg["arcface_model_path"])
    return detector, embedder


def match_faces(id_image: str | Path, selfie_image: str | Path, cfg: dict, detector=None, embedder=None, return_debug: bool = False) -> dict:
    try:
        id_bgr = read_bgr(id_image)
        selfie_bgr = read_bgr(selfie_image)
    except Exception:
        return response("INVALID_IMAGE", "AUTO_REJECT", 0.0, 0.0, cfg, False, False, 0, False, False, ["invalid_or_unreadable_image"])

    try:
        detector = detector or load_models(cfg)[0]
        embedder = embedder or ArcFaceEmbedder(cfg["arcface_model_path"])
    except FileNotFoundError:
        return response("MODEL_NOT_FOUND", "AUTO_REJECT", 0.0, 0.0, cfg, False, False, 0, False, False, ["model_file_not_found"])

    id_faces = detector.detect(id_bgr)
    selfie_faces = detector.detect(selfie_bgr)
    large_area_ratio = float(cfg.get("selfie_multiple_face_relative_area", 0.04))
    selfie_area = selfie_bgr.shape[0] * selfie_bgr.shape[1]
    large_selfie_faces = [f for f in selfie_faces if f.area / max(selfie_area, 1) >= large_area_ratio]
    selfie_face_count = len(large_selfie_faces) if large_selfie_faces else len(selfie_faces)

    if not id_faces:
        return response("NO_FACE_IN_ID", "AUTO_REJECT", 0.0, 0.0, cfg, False, bool(selfie_faces), selfie_face_count, False, bool(selfie_faces), ["no_face_in_id"])
    if not selfie_faces:
        return response("NO_FACE_IN_SELFIE", "AUTO_REJECT", 0.0, 0.0, cfg, True, False, 0, True, False, ["no_face_in_selfie"])
    if selfie_face_count > 1:
        return response("MULTIPLE_SELFIE_FACES", "AUTO_REJECT", 0.0, 0.0, cfg, True, True, selfie_face_count, True, False, ["multiple_large_faces_in_selfie"])

    id_face = id_faces[0]
    selfie_face = large_selfie_faces[0] if large_selfie_faces else selfie_faces[0]
    id_crop = crop_bbox(id_bgr, id_face.bbox)
    selfie_crop = crop_bbox(selfie_bgr, selfie_face.bbox)
    id_quality_ok, id_warnings = assess_face_quality(id_crop, id_face, cfg, "id")
    selfie_quality_ok, selfie_warnings = assess_face_quality(selfie_crop, selfie_face, cfg, "selfie")
    warnings = id_warnings + selfie_warnings

    id_aligned, id_align_error = align_face(id_bgr, id_face.landmarks)
    selfie_aligned, selfie_align_error = align_face(selfie_bgr, selfie_face.landmarks)
    if id_aligned is None:
        warnings.append(f"id_alignment_failed:{id_align_error}")
    if selfie_aligned is None:
        warnings.append(f"selfie_alignment_failed:{selfie_align_error}")
    if not id_quality_ok or id_aligned is None:
        result = response("LOW_QUALITY_ID_FACE", "AUTO_REJECT", None, None, cfg, True, True, selfie_face_count, id_quality_ok, selfie_quality_ok, warnings)
    elif not selfie_quality_ok or selfie_aligned is None:
        result = response("LOW_QUALITY_SELFIE_FACE", "AUTO_REJECT", None, None, cfg, True, True, selfie_face_count, id_quality_ok, selfie_quality_ok, warnings)
    else:
        id_emb = embedder.embed(id_aligned)
        selfie_emb = embedder.embed(selfie_aligned)
        cosine = cosine_similarity(id_emb, selfie_emb)
        scoring = score_similarity(cosine, cfg)
        result = response(scoring["status"], scoring["decision"], scoring["score"], cosine, cfg, True, True, selfie_face_count, True, True, warnings)
        result["raw_score"] = scoring["raw_score"]
        result["score_mode"] = scoring["score_mode"]
        result["calibrated"] = scoring["calibrated"]
        result["calibration_name"] = scoring["calibration_name"]
        result["thresholds"] = scoring["thresholds"]

    if return_debug:
        result["_debug"] = {
            "id_bgr": id_bgr,
            "selfie_bgr": selfie_bgr,
            "id_crop": id_crop,
            "selfie_crop": selfie_crop,
            "id_aligned": id_aligned,
            "selfie_aligned": selfie_aligned,
            "id_detections": id_faces,
            "selfie_detections": selfie_faces,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP2 face similarity matching.")
    parser.add_argument("--config", default=str(project_relative_default_config()))
    parser.add_argument("--id-image", required=True)
    parser.add_argument("--selfie-image", required=True)
    parser.add_argument("--face-profile", choices=["stable", "accurate"], default=None)
    args = parser.parse_args()

    cfg = apply_face_model_profile(load_config(args.config), args.face_profile)
    result = match_faces(args.id_image, args.selfie_image, cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
