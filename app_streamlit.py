import os
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
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
def load_checkpoint(ckpt_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(ckpt_path, map_location=device)
    model_name = str(payload.get("actual_model_name") or payload.get("requested_model_name") or "efficientnetb0")
    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload, model_name, device


def open_uploaded_image(uploaded_file) -> Optional[Image.Image]:
    try:
        return ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    except Exception:
        return None


def save_hard_negative_candidate(uploaded_file, cfg: Dict, result: Dict) -> str:
    project_dir = Path(os.path.abspath(str(cfg["project_dir"])))
    prepared = Path(os.path.abspath(str(cfg["prepared_dataset_path"])))
    target_dir = prepared / "not_national_id" / "manual_hard_negatives"
    target_dir.mkdir(parents=True, exist_ok=True)

    data = uploaded_file.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    suffix = Path(uploaded_file.name or "upload.jpg").suffix.lower() or ".jpg"
    stem = Path(uploaded_file.name or "upload").stem
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)[:80]
    target_path = target_dir / f"manual_{safe_stem}_{sha[:10]}{suffix}"

    if not target_path.exists():
        target_path.write_bytes(data)

    log_path = project_dir / "reports" / "hard_negative_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "created_utc",
                "source_name",
                "saved_path",
                "sha256",
                "national_id_probability",
                "not_national_id_probability",
                "status",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_name": uploaded_file.name,
                "saved_path": str(target_path),
                "sha256": sha,
                "national_id_probability": f"{result['national_id_probability']:.8f}",
                "not_national_id_probability": f"{result['not_national_id_probability']:.8f}",
                "status": "saved_for_next_retrain",
            }
        )
    return str(target_path)


def predict(im: Image.Image, model, payload: Dict, image_size: int, threshold: float, device: str) -> Dict:
    _, eval_tf = get_transforms(image_size)
    x = eval_tf(im).unsqueeze(0).to(device)
    positive_class_name = str(payload.get("positive_class_name", "national_id"))
    with torch.no_grad():
        raw_prob = float(torch.sigmoid(model(x)).item())
    p_nat = raw_prob if positive_class_name == "national_id" else 1.0 - raw_prob
    p_not = 1.0 - p_nat
    accepted = p_nat >= threshold
    return {
        "label": "national_id" if accepted else "not_national_id",
        "national_id_probability": p_nat,
        "not_national_id_probability": p_not,
        "threshold": threshold,
        "decision": "ACCEPT_FOR_NEXT_STEP" if accepted else "REJECT_RETRY",
    }


def main() -> None:
    st.set_page_config(page_title="MVP1 KYC Validator", layout="centered")
    st.title("MVP1 KYC Document Validator")
    st.warning("MVP1 only validates document type. It does not verify user identity.")

    cfg = load_cfg("config.yaml")
    project_dir = os.path.abspath(str(cfg["project_dir"]))
    ckpt_path = os.path.join(project_dir, "models", "validator_best.pt")
    threshold = float(cfg.get("accept_threshold", 0.95))

    if not os.path.isfile(ckpt_path):
        st.error("No trained model found. Expected models/validator_best.pt")
        return

    model, payload, model_name, device = load_checkpoint(ckpt_path)
    st.caption(f"Model: {model_name} | Threshold: {threshold:.2f} | Device: {device}")

    uploaded = st.file_uploader("Upload one image", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"])
    if uploaded is None:
        return

    im = open_uploaded_image(uploaded)
    if im is None:
        st.error("Invalid or unreadable image.")
        return

    st.image(im, caption="Uploaded image", use_container_width=True)
    result = predict(
        im,
        model=model,
        payload=payload,
        image_size=int(cfg.get("image_size", 224)),
        threshold=threshold,
        device=device,
    )

    st.subheader("Result")
    st.metric("Predicted label", result["label"])
    st.metric("National ID probability", f"{result['national_id_probability'] * 100:.2f}%")
    st.metric("Not national ID probability", f"{result['not_national_id_probability'] * 100:.2f}%")
    st.write(f"Threshold: `{result['threshold']:.2f}`")
    st.write(f"Decision: `{result['decision']}`")
    if result["decision"] == "ACCEPT_FOR_NEXT_STEP":
        st.info(
            "If this image is not a real Egyptian National ID front, save it as a hard negative "
            "before the next retrain. This is how we teach the model this failure mode."
        )
        if st.button("Save as hard negative for next retrain"):
            saved_path = save_hard_negative_candidate(uploaded, cfg, result)
            st.success(f"Saved hard negative candidate: {saved_path}")


if __name__ == "__main__":
    main()
