import argparse
import json
import os
from typing import Dict

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from modules.model import build_model, get_transforms
from modules.utils import simple_yaml_load


def load_cfg(path: str) -> Dict:
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


def result_template(
    status: str,
    decision: str,
    label: str,
    p_nat: float,
    p_not: float,
    threshold: float,
    mvp_version: str,
) -> Dict:
    return {
        "status": status,
        "decision": decision,
        "label": label,
        "national_id_probability": round(float(p_nat), 6),
        "not_national_id_probability": round(float(p_not), 6),
        "threshold": float(threshold),
        "mvp_version": mvp_version,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for one image.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    project_dir = os.path.abspath(cfg["project_dir"])
    ckpt = args.checkpoint or os.path.join(project_dir, "models", "validator_best.pt")
    threshold = float(cfg.get("accept_threshold", 0.95))
    mvp_version = str(cfg.get("mvp_version", "MVP1"))

    if not os.path.isfile(ckpt):
        res = result_template(
            status="MODEL_NOT_FOUND",
            decision="REJECT_RETRY",
            label="not_national_id",
            p_nat=0.0,
            p_not=1.0,
            threshold=threshold,
            mvp_version=mvp_version,
        )
        print(json.dumps(res, indent=2))
        return

    image_path = os.path.abspath(args.image)
    if not os.path.isfile(image_path):
        res = result_template(
            status="INVALID_IMAGE",
            decision="REJECT_RETRY",
            label="not_national_id",
            p_nat=0.0,
            p_not=1.0,
            threshold=threshold,
            mvp_version=mvp_version,
        )
        print(json.dumps(res, indent=2))
        return

    try:
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, RuntimeError):
        res = result_template(
            status="INVALID_IMAGE",
            decision="REJECT_RETRY",
            label="not_national_id",
            p_nat=0.0,
            p_not=1.0,
            threshold=threshold,
            mvp_version=mvp_version,
        )
        print(json.dumps(res, indent=2))
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(ckpt, map_location=device)
    model_name = str(payload.get("actual_model_name") or payload.get("requested_model_name") or cfg["model_name"])
    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    positive_class_name = str(payload.get("positive_class_name", "national_id"))

    _, eval_tf = get_transforms(int(cfg["image_size"]))
    x = eval_tf(im).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(x)
        p_raw = float(torch.sigmoid(logit).item())
        p_nat = p_raw if positive_class_name == "national_id" else 1.0 - p_raw
    p_not = 1.0 - p_nat
    accepted = p_nat >= threshold

    res = result_template(
        status="NATIONAL_ID_ACCEPTED" if accepted else "DOCUMENT_REJECTED",
        decision="ACCEPT_FOR_NEXT_STEP" if accepted else "REJECT_RETRY",
        label="national_id" if accepted else "not_national_id",
        p_nat=p_nat,
        p_not=p_not,
        threshold=threshold,
        mvp_version=mvp_version,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
