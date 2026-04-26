import tempfile

import streamlit as st

from modules.face_similarity import ArcFaceOnnxEmbedder, MobileNetFaceBaseline, score_pair
from modules.utils import simple_yaml_load


@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str, image_size: int, arcface_path: str):
    if model_name.lower() == "arcface_onnx":
        return ArcFaceOnnxEmbedder(arcface_path)
    return MobileNetFaceBaseline(image_size=image_size)


def main() -> None:
    st.set_page_config(page_title="MVP2 Face Similarity", layout="centered")
    st.title("MVP2 ID Photo vs Selfie Similarity")
    st.warning("MVP2 compares face similarity only. It does not perform liveness, anti-spoofing, OCR, or backend identity checks.")

    cfg = simple_yaml_load("config_mvp2.yaml")
    low = float(cfg.get("manual_review_low", 0.50))
    high = float(cfg.get("manual_review_high", 0.70))
    embedder = load_embedder(
        str(cfg.get("face_match_model", "mobilenet_v3_small_imagenet_baseline")),
        int(cfg.get("image_size", 224)),
        str(cfg.get("arcface_onnx_path", "")),
    )

    id_upload = st.file_uploader("Upload ID/document face image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    selfie_upload = st.file_uploader("Upload selfie image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if not id_upload or not selfie_upload:
        return

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as id_tmp:
        id_tmp.write(id_upload.getvalue())
        id_path = id_tmp.name
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as selfie_tmp:
        selfie_tmp.write(selfie_upload.getvalue())
        selfie_path = selfie_tmp.name

    col1, col2 = st.columns(2)
    col1.image(id_upload, caption="ID/document image", use_container_width=True)
    col2.image(selfie_upload, caption="Selfie image", use_container_width=True)

    result = score_pair(id_path, selfie_path, embedder)
    score = float(result["similarity"])
    if score >= high:
        status = "FACE_MATCH_ACCEPTED"
        decision = "ACCEPT_FOR_NEXT_STEP"
    elif score >= low:
        status = "FACE_MATCH_MANUAL_REVIEW"
        decision = "MANUAL_REVIEW"
    else:
        status = "FACE_MATCH_REJECTED"
        decision = "REJECT_RETRY"

    st.subheader("Result")
    st.metric("Similarity", f"{score * 100:.2f}%")
    st.write(f"Status: `{status}`")
    st.write(f"Decision: `{decision}`")
    st.write(f"Manual review range: `{low:.2f}` to `{high:.2f}`")
    st.write(f"ID face detected: `{result['id_face_found']}`")
    st.write(f"Selfie face detected: `{result['selfie_face_found']}`")


if __name__ == "__main__":
    main()
