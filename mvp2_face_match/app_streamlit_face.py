from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import streamlit as st

from face_match import load_models, match_faces
from modules.response_schema import LIMITATIONS
from modules.utils import ensure_dir, load_config, project_relative_default_config


@st.cache_resource(show_spinner=False)
def cached_models(config_path: str):
    cfg = load_config(config_path)
    return cfg, *load_models(cfg)


def save_upload(uploaded, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    path.write_bytes(uploaded.getvalue())
    return path


def save_debug_image(path: Path, bgr):
    if bgr is not None:
        cv2.imwrite(str(path), bgr)


def save_hard_case(case_type: str, id_upload, selfie_upload, result: dict, debug: dict, cfg: dict) -> Path:
    root = ensure_dir(Path(str(cfg["face_project_dir"])) / "outputs" / "hard_pairs" / f"{int(time.time())}_{case_type}")
    id_path = save_upload(id_upload, root, f"id_{id_upload.name}")
    selfie_path = save_upload(selfie_upload, root, f"selfie_{selfie_upload.name}")
    save_debug_image(root / "id_face_crop.jpg", debug.get("id_crop"))
    save_debug_image(root / "selfie_face_crop.jpg", debug.get("selfie_crop"))
    save_debug_image(root / "id_face_aligned.jpg", debug.get("id_aligned"))
    save_debug_image(root / "selfie_face_aligned.jpg", debug.get("selfie_aligned"))
    manifest = {"case_type": case_type, "id_image": str(id_path), "selfie_image": str(selfie_path), "result": result}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def main() -> None:
    st.set_page_config(page_title="MVP2 Face Match", layout="wide")
    st.title("MVP2 Face Similarity: ID Photo vs Selfie")
    st.warning("MVP2 compares face similarity only. It does not perform liveness, anti-spoofing, OCR, database matching, or full KYC approval.")
    for item in LIMITATIONS:
        st.caption(f"- {item}")

    config_path = str(project_relative_default_config())
    cfg, detector, embedder = cached_models(config_path)

    col_a, col_b = st.columns(2)
    id_upload = col_a.file_uploader("Upload ID image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    selfie_upload = col_b.file_uploader("Upload selfie image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if not id_upload or not selfie_upload:
        return

    temp_dir = ensure_dir(Path(str(cfg["face_project_dir"])) / "outputs" / "debug" / "streamlit_tmp")
    id_path = save_upload(id_upload, temp_dir, f"id_{id_upload.name}")
    selfie_path = save_upload(selfie_upload, temp_dir, f"selfie_{selfie_upload.name}")

    result = match_faces(id_path, selfie_path, cfg, detector=detector, embedder=embedder, return_debug=True)
    debug = result.pop("_debug", {})

    col_a.image(id_upload, caption="Uploaded ID image", use_container_width=True)
    col_b.image(selfie_upload, caption="Uploaded selfie image", use_container_width=True)

    crop_a, crop_b = st.columns(2)
    if debug.get("id_crop") is not None:
        crop_a.image(cv2.cvtColor(debug["id_crop"], cv2.COLOR_BGR2RGB), caption="Detected ID face crop", use_container_width=True)
    else:
        crop_a.error("No ID face crop available")
    if debug.get("selfie_crop") is not None:
        crop_b.image(cv2.cvtColor(debug["selfie_crop"], cv2.COLOR_BGR2RGB), caption="Detected selfie face crop", use_container_width=True)
    else:
        crop_b.error("No selfie face crop available")

    st.subheader("Result")
    st.metric("Similarity score", f"{result['score']:.2f} / 100")
    st.write(f"Cosine similarity: `{result['cosine_similarity']}`")
    st.write(f"Status: `{result['status']}`")
    st.write(f"Decision: `{result['decision']}`")
    st.write("Quality warnings:")
    st.json(result["quality"])
    st.write("Full JSON response:")
    st.json(result)

    st.subheader("Save Hard Case")
    case_type = st.selectbox(
        "Case type",
        ["false_match", "false_mismatch", "failed_id_face_detection", "failed_selfie_face_detection"],
    )
    if st.button("Save hard case"):
        saved = save_hard_case(case_type, id_upload, selfie_upload, result, debug, cfg)
        st.success(f"Saved hard case to: {saved}")


if __name__ == "__main__":
    main()
