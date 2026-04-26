import argparse
import os

import torch

from modules.model import build_model
from modules.utils import simple_yaml_load


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained validator checkpoint to ONNX.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cfg = simple_yaml_load(args.config)
    project_dir = os.path.abspath(cfg.get("project_dir", r"D:\Projects\KYC\kyc-validator-training"))
    model_name = cfg.get("model_name", "efficientnetv2_b0")
    image_size = int(cfg.get("image_size", 224))

    ckpt = args.checkpoint or os.path.join(project_dir, "models", "validator_best.pt")
    out_path = args.output or os.path.join(project_dir, "models", "validator.onnx")
    if not os.path.isfile(ckpt):
        raise SystemExit("Checkpoint not found. Train model first, then run ONNX export.")

    payload = torch.load(ckpt, map_location="cpu")
    model = build_model(model_name, pretrained=False)
    model.load_state_dict(payload["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["image"],
        output_names=["logit"],
        dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX exported to: {out_path}")


if __name__ == "__main__":
    main()

