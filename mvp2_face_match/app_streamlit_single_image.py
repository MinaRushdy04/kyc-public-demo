from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import streamlit as st

from modules.utils import ensure_dir, load_config, project_relative_default_config


def save_upload(uploaded, out_dir: Path) -> Path:
    path = out_dir / uploaded.name
    path.write_bytes(uploaded.getvalue())
    return path


def upload_hash(uploaded) -> str:
    return hashlib.sha256(uploaded.getvalue()).hexdigest()


def read_bgr_if_exists(path: str | Path | None):
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return cv2.imread(str(p), cv2.IMREAD_COLOR)


def show_bgr(title: str, image_bgr):
    if image_bgr is None:
        st.info(f"{title}: not available")
        return
    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption=title, width="stretch")


def show_decision_banner(result: dict) -> None:
    status = str(result.get("status", ""))
    decision = str(result.get("final_decision", ""))
    if status == "FINAL_ACCEPT":
        st.success("Document accepted for the next KYC stage. This is not full identity verification.")
    elif decision == "MANUAL_REVIEW":
        st.warning("Manual review recommended: document passed, but face similarity is not strong enough for auto-accept.")
    elif status == "LOW_QUALITY_ID_FACE":
        st.warning("ID portrait was detected but is too small/blurry for a safe auto decision. Try holding the ID closer, flatter, and steadier.")
    elif status == "NO_ID_CARD_FOUND":
        st.error("No reliable ID card crop was found. Move the card closer, reduce glare, and keep all four card edges visible.")
    elif status in {"NO_HOLDER_FACE", "NO_FACE_IN_ID", "MULTIPLE_SELFIE_FACES"}:
        st.error("Face extraction failed for this image. Check the detailed status and warning list below.")
    else:
        st.error("Rejected for retry. Check the document, face, and quality sections below.")


def draw_id_bbox(image_bgr, bbox):
    if image_bgr is None:
        return None
    canvas = image_bgr.copy()
    if bbox:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 220, 30), 4)
        cv2.putText(canvas, "ID candidate", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 220, 30), 2)
    return canvas


def parse_json_from_stdout(stdout: str) -> dict:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Pipeline did not return JSON.")
    return json.loads(stdout[start : end + 1])


def run_pipeline_subprocess(image_path: Path, cfg: dict, debug_dir: Path, face_profile: str) -> dict:
    script = Path(str(cfg["face_project_dir"])) / "full_kyc_single_image.py"
    cmd = [
        sys.executable,
        str(script),
        "--config",
        str(project_relative_default_config()),
        "--image",
        str(image_path),
        "--debug-dir",
        str(debug_dir),
        "--face-profile",
        face_profile,
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    completed = subprocess.run(
        cmd,
        cwd=str(Path(str(cfg["project_dir"]))),
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    if completed.returncode != 0:
        return {
            "status": "PIPELINE_ERROR",
            "final_decision": "REJECT_RETRY",
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
                "thresholds": {"auto_reject_below": 50, "auto_accept_at_or_above": 70},
            },
            "quality": {
                "warnings": [
                    "pipeline_subprocess_failed",
                    completed.stderr[-2000:] if completed.stderr else completed.stdout[-2000:],
                ]
            },
            "limitations": [
                "No liveness or anti-spoofing is included.",
                "This compares face similarity only.",
                "Printed-photo and screen-replay attacks are not reliably prevented.",
                "ID localization is MVP-level and may fail if the card is too small, blurry, occluded, or reflective.",
            ],
            "mvp_version": "MVP2_SINGLE_IMAGE",
        }
    return parse_json_from_stdout(completed.stdout)


def save_debug_image(path: Path, image_bgr):
    if image_bgr is not None:
        cv2.imwrite(str(path), image_bgr)


def save_hard_case(case_type: str, uploaded, result: dict, debug_dir: Path, cfg: dict) -> Path:
    if case_type == "save failed ID localization":
        base = Path(str(cfg["face_project_dir"])) / "outputs" / "failed_id_localization"
    else:
        base = Path(str(cfg["face_project_dir"])) / "outputs" / "hard_pairs"
    root = ensure_dir(base / f"{int(time.time())}_{case_type.replace(' ', '_')}")
    save_upload(uploaded, root)
    for name in [
        "id_card_crop.jpg",
        "id_portrait_detection_attempts.json",
        "holder_face_crop.jpg",
        "id_portrait_face_crop.jpg",
        "holder_face_aligned.jpg",
        "id_portrait_face_aligned.jpg",
    ]:
        src = debug_dir / name
        if src.is_file():
            (root / name).write_bytes(src.read_bytes())
    for src in debug_dir.glob("id_card_enhanced_*.jpg"):
        (root / src.name).write_bytes(src.read_bytes())
    (root / "manifest.json").write_text(
        json.dumps({"case_type": case_type, "result": result}, indent=2),
        encoding="utf-8",
    )
    return root


def main() -> None:
    st.set_page_config(page_title="MVP2 Single Image KYC", layout="wide")
    st.title("MVP2 Single-Image Selfie Holding ID")
    st.warning(
        "Internal testing only. This accepts a KYC step for the next stage; it does not prove full identity, "
        "does not perform OCR/database matching, and includes no liveness or anti-spoofing."
    )

    cfg = load_config(project_relative_default_config())
    default_profile = str(cfg.get("face_model_profile", "accurate")).lower()
    profile_options = ["accurate", "stable"]
    if default_profile not in profile_options:
        default_profile = "accurate"
    face_profile = st.radio(
        "Face embedding profile",
        profile_options,
        index=profile_options.index(default_profile),
        horizontal=True,
        help="accurate uses the stronger ArcFace model in an isolated subprocess; stable uses the lightweight MobileFaceNet model.",
    )
    if face_profile == "accurate":
        st.caption("Accurate profile: stronger embeddings, higher RAM use. Best for quality testing after memory is clean.")
    else:
        st.caption("Stable profile: lower RAM use. Best when Windows memory/pagefile pressure is high.")

    uploaded = st.file_uploader("Upload one image: selfie while holding Egyptian National ID", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is None:
        st.info("Upload a single full image containing both the holder face and the ID card.")
        return

    file_hash = upload_hash(uploaded)
    cache = st.session_state.get("single_image_pipeline_cache", {})
    force_run = st.button("Run / rerun pipeline", type="primary")
    needs_run = force_run or cache.get("file_hash") != file_hash or cache.get("face_profile") != face_profile

    if needs_run:
        run_id = f"{int(time.time() * 1000)}_{face_profile}_{file_hash[:10]}"
        temp_dir = ensure_dir(Path(str(cfg["face_project_dir"])) / "outputs" / "debug" / f"single_image_streamlit_{run_id}")
        image_path = save_upload(uploaded, temp_dir)
        with st.spinner("Running one-image KYC pipeline in an isolated model process..."):
            result = run_pipeline_subprocess(image_path, cfg, temp_dir, face_profile)
        st.session_state["single_image_pipeline_cache"] = {
            "file_hash": file_hash,
            "face_profile": face_profile,
            "result": result,
            "temp_dir": str(temp_dir),
            "image_path": str(image_path),
            "uploaded_name": uploaded.name,
        }
    else:
        st.caption("Using cached result for this uploaded image. Click Run / rerun pipeline to force a fresh pass.")

    cache = st.session_state["single_image_pipeline_cache"]
    result = cache["result"]
    temp_dir = Path(cache["temp_dir"])
    image_path = Path(cache["image_path"])

    original_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    full_with_bbox = draw_id_bbox(original_bgr, result["document"].get("id_bbox"))
    id_crop = read_bgr_if_exists(result["document"].get("id_crop_path"))
    if id_crop is None:
        id_crop = read_bgr_if_exists(temp_dir / "id_card_crop.jpg")
    holder_crop = read_bgr_if_exists(temp_dir / "holder_face_crop.jpg")
    id_face_crop = read_bgr_if_exists(temp_dir / "id_portrait_face_crop.jpg")
    enhanced_paths = sorted(temp_dir.glob("id_card_enhanced_*.jpg"))
    enhanced_id_crop = read_bgr_if_exists(enhanced_paths[0]) if enhanced_paths else None

    show_decision_banner(result)

    left, right = st.columns(2)
    with left:
        show_bgr("Original image with ID bbox", full_with_bbox)
    with right:
        show_bgr("Cropped ID card", id_crop)

    if enhanced_id_crop is not None:
        show_bgr("Enhanced ID card used for portrait detection", enhanced_id_crop)

    face_a, face_b = st.columns(2)
    with face_a:
        show_bgr("Holder/selfie face crop", holder_crop)
    with face_b:
        show_bgr("ID portrait face crop", id_face_crop)

    st.subheader("Document Result")
    st.write(f"Document status: `{result['document']['status']}`")
    st.metric("National ID probability", f"{float(result['document']['national_id_probability']) * 100:.2f}%")
    st.write(f"ID bbox: `{result['document']['id_bbox']}`")

    st.subheader("Face Result")
    st.write(f"Embedding profile: `{cache.get('face_profile', face_profile)}`")
    st.write(f"Face status: `{result['face_match']['status']}`")
    score = result["face_match"].get("score")
    score_mode = result["face_match"].get("score_mode", "raw_cosine_score")
    score_label = "Calibrated match probability" if score_mode == "calibrated_match_probability" else "Raw similarity score"
    st.metric(score_label, "not computed" if score is None else f"{float(score):.2f} / 100")
    raw_score = result["face_match"].get("raw_score")
    if raw_score is not None:
        st.write(f"Raw cosine-derived score: `{float(raw_score):.2f} / 100`")
    st.write(f"Score mode: `{score_mode}`")
    st.write(f"Calibration: `{result['face_match'].get('calibration_name') or 'not available'}`")
    st.write(f"Cosine similarity: `{result['face_match']['cosine_similarity']}`")

    st.subheader("Final Decision")
    st.write(f"Status: `{result['status']}`")
    st.write(f"Final decision: `{result['final_decision']}`")

    st.subheader("Warnings And Limitations")
    st.json(result["quality"])
    for item in result["limitations"]:
        st.caption(f"- {item}")

    st.subheader("Full JSON")
    st.json(result)

    st.subheader("Save Hard Case")
    case_type = st.selectbox(
        "Choose hard-case label",
        [
            "save failed ID localization",
            "save false document accept",
            "save false document reject",
            "save false face match",
            "save false face mismatch",
        ],
    )
    if st.button("Save hard case"):
        saved = save_hard_case(case_type, uploaded, result, temp_dir, cfg)
        st.success(f"Saved hard case to: {saved}")


if __name__ == "__main__":
    main()
