import json
import os
from typing import Dict, Optional

import streamlit as st
import torch
from PIL import Image, ImageOps

from modules.model import build_model, get_transforms
from modules.utils import simple_yaml_load


def load_cfg(path: str = "config.yaml") -> Dict:
    cfg = simple_yaml_load(path)
    defaults = {
        "project_dir": r"D:\Projects\KYC\kyc-validator-training",
        "model_name": "efficientnetv2_b0",
        "image_size": 224,
        "accept_threshold": 0.95,
        "mvp_version": "MVP1",
    }
    defaults.update(cfg)
    return defaults


@st.cache_resource(show_spinner=False)
def load_model(ckpt_path: str, model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(ckpt_path, map_location=device)
    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload, device


def predict_image(
    im: Image.Image,
    model,
    payload: Dict,
    image_size: int,
    threshold: float,
    mvp_version: str,
    device: str,
) -> Dict:
    _, eval_tf = get_transforms(int(image_size))
    x = eval_tf(im).unsqueeze(0).to(device)
    positive_class_name = str(payload.get("positive_class_name", "national_id"))
    with torch.no_grad():
        logit = model(x)
        p_raw = float(torch.sigmoid(logit).item())
    p_nat = p_raw if positive_class_name == "national_id" else 1.0 - p_raw
    p_not = 1.0 - p_nat
    accepted = p_nat >= threshold
    return {
        "status": "NATIONAL_ID_ACCEPTED" if accepted else "DOCUMENT_REJECTED",
        "decision": "ACCEPT_FOR_NEXT_STEP" if accepted else "REJECT_RETRY",
        "label": "national_id" if accepted else "not_national_id",
        "national_id_probability": round(float(p_nat), 6),
        "not_national_id_probability": round(float(p_not), 6),
        "threshold": float(threshold),
        "mvp_version": mvp_version,
    }


def open_image_from_upload(uploaded_file) -> Optional[Image.Image]:
    try:
        im = Image.open(uploaded_file)
        im = ImageOps.exif_transpose(im).convert("RGB")
        return im
    except Exception:
        return None


def open_image_from_path(path: str) -> Optional[Image.Image]:
    if not path or not os.path.isfile(path):
        return None
    try:
        im = Image.open(path)
        im = ImageOps.exif_transpose(im).convert("RGB")
        return im
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title="KYC MVP1 Validator", layout="wide")
    st.title("KYC MVP1 Document Validator")
    st.caption("Binary classifier: Egyptian National ID (front) vs Not Egyptian National ID")

    cfg = load_cfg("config.yaml")
    project_dir = os.path.abspath(cfg["project_dir"])
    ckpt_path = os.path.join(project_dir, "models", "validator_best.pt")
    default_threshold = float(cfg.get("accept_threshold", 0.95))
    mvp_version = str(cfg.get("mvp_version", "MVP1"))

    col_a, col_b = st.columns([2, 1])
    with col_b:
        st.subheader("Settings")
        threshold = st.slider("Decision threshold", min_value=0.50, max_value=0.99, value=float(default_threshold), step=0.01)
        st.write(f"Checkpoint: `{ckpt_path}`")
        st.write(f"Model: `{cfg.get('model_name', 'efficientnetv2_b0')}`")

    if not os.path.isfile(ckpt_path):
        st.error("Model checkpoint not found. Train first: python train_validator.py --config config.yaml")
        return

    model, payload, device = load_model(ckpt_path, str(cfg.get("model_name", "efficientnetv2_b0")))
    st.success(f"Model loaded on {device}.")

    with col_a:
        st.subheader("Input")
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"])
        local_path = st.text_input("Or local image path", value="")

    image = None
    source = ""
    if uploaded is not None:
        image = open_image_from_upload(uploaded)
        source = uploaded.name
    elif local_path.strip():
        image = open_image_from_path(local_path.strip())
        source = local_path.strip()

    if image is None:
        st.info("Provide an image to run prediction.")
        return

    st.image(image, caption=f"Input: {source}", use_column_width=True)
    result = predict_image(
        image,
        model=model,
        payload=payload,
        image_size=int(cfg.get("image_size", 224)),
        threshold=float(threshold),
        mvp_version=mvp_version,
        device=device,
    )

    st.subheader("Prediction")
    st.json(result)
    st.progress(float(result["national_id_probability"]))

    debug_col1, debug_col2 = st.columns(2)
    with debug_col1:
        st.metric("National ID Probability", f"{100.0 * result['national_id_probability']:.2f}%")
    with debug_col2:
        st.metric("Not National ID Probability", f"{100.0 * result['not_national_id_probability']:.2f}%")


if __name__ == "__main__":
    main()

