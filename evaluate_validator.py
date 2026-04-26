import argparse
import json
import os
import shutil
from typing import Dict, List, Tuple

import torch
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from modules.metrics import binary_metrics, confusion_counts, roc_auc_score_safe
from modules.model import build_model, get_transforms
from modules.utils import simple_yaml_load


def load_cfg(path: str) -> Dict:
    cfg = simple_yaml_load(path)
    defaults = {
        "prepared_dataset_path": r"D:\Projects\KYC\kyc-validator-training\prepared_validator_dataset",
        "project_dir": r"D:\Projects\KYC\kyc-validator-training",
        "model_name": "efficientnetv2_b0",
        "image_size": 224,
        "batch_size": 16,
        "accept_threshold": 0.95,
    }
    defaults.update(cfg)
    return defaults


def empty_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isfile(full):
            os.remove(full)


def evaluate_subset(
    model,
    dataset,
    indices: List[int],
    batch_size: int,
    positive_class_name: str,
    threshold: float,
    device: str,
) -> Tuple[Dict, List[int], List[int], List[float], List[str]]:
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    class_to_idx = dataset.class_to_idx
    national_idx = int(class_to_idx["national_id"])
    sample_paths = [dataset.samples[i][0] for i in indices]

    y_true: List[int] = []
    y_prob: List[float] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs_raw = torch.sigmoid(logits).squeeze(1).cpu().numpy().tolist()
            labels = y.numpy().tolist()
            for prob_raw, label in zip(probs_raw, labels):
                p_nat = float(prob_raw) if positive_class_name == "national_id" else 1.0 - float(prob_raw)
                true_bin = 1 if int(label) == national_idx else 0
                pred_bin = 1 if p_nat >= threshold else 0
                y_true.append(true_bin)
                y_prob.append(p_nat)
                y_pred.append(pred_bin)

    metrics = binary_metrics(y_true, y_pred)
    conf = confusion_counts(y_true, y_pred)
    auc = roc_auc_score_safe(y_true, y_prob)
    result = {
        "count": len(indices),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": auc,
        "confusion_matrix": conf,
        "threshold": threshold,
    }
    return result, y_true, y_pred, y_prob, sample_paths


def save_confusion_matrix_png(path: str, conf: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w, h = 600, 420
    im = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(im)
    draw.text((30, 20), "Confusion Matrix (Test, threshold=0.95)", fill=(20, 20, 20))
    labels = [["TN", "FP"], ["FN", "TP"]]
    values = [[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]]
    x0, y0 = 130, 90
    cell_w, cell_h = 170, 120
    for r in range(2):
        for c in range(2):
            x = x0 + c * cell_w
            y = y0 + r * cell_h
            fill = (232, 242, 252) if (r == c) else (252, 236, 232)
            draw.rectangle((x, y, x + cell_w, y + cell_h), fill=fill, outline=(70, 70, 70), width=2)
            draw.text((x + 20, y + 25), labels[r][c], fill=(20, 20, 20))
            draw.text((x + 20, y + 60), str(values[r][c]), fill=(20, 20, 20))
    draw.text((x0, y0 - 25), "Pred not_id", fill=(20, 20, 20))
    draw.text((x0 + cell_w, y0 - 25), "Pred national_id", fill=(20, 20, 20))
    draw.text((20, y0 + 45), "True not_id", fill=(20, 20, 20))
    draw.text((20, y0 + cell_h + 45), "True national_id", fill=(20, 20, 20))
    im.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MVP1 validator model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    project_dir = os.path.abspath(cfg["project_dir"])
    ckpt = args.checkpoint or os.path.join(project_dir, "models", "validator_best.pt")
    if not os.path.isfile(ckpt):
        raise SystemExit("Model checkpoint not found. Train first before evaluation.")

    payload = torch.load(ckpt, map_location="cpu")
    split_path = payload.get("split_path") or os.path.join(project_dir, "models", "mvp1_split.json")
    if not os.path.isfile(split_path):
        raise SystemExit(f"Split file not found: {split_path}")
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    prepared = os.path.abspath(cfg["prepared_dataset_path"])
    _, eval_tf = get_transforms(int(cfg["image_size"]))
    ds = ImageFolder(prepared, transform=eval_tf)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = str(payload.get("actual_model_name") or payload.get("requested_model_name") or cfg["model_name"])
    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    positive_class_name = str(payload.get("positive_class_name", "national_id"))
    threshold = float(cfg.get("accept_threshold", 0.95))

    split_results = {}
    raw_test = None
    for split_name, key in [("train", "train_indices"), ("validation", "val_indices"), ("test", "test_indices")]:
        result, y_true, y_pred, y_prob, paths = evaluate_subset(
            model,
            ds,
            split[key],
            batch_size=int(cfg["batch_size"]),
            positive_class_name=positive_class_name,
            threshold=threshold,
            device=device,
        )
        split_results[split_name] = result
        if split_name == "test":
            raw_test = (y_true, y_pred, y_prob, paths)

    out_debug = os.path.join(project_dir, "outputs", "debug")
    out_false_accepts = os.path.join(project_dir, "outputs", "false_accepts")
    out_false_rejects = os.path.join(project_dir, "outputs", "false_rejects")
    os.makedirs(out_debug, exist_ok=True)
    empty_dir(out_false_accepts)
    empty_dir(out_false_rejects)

    assert raw_test is not None
    y_true, y_pred, y_prob, sample_paths = raw_test
    for src, truth, pred, prob in zip(sample_paths, y_true, y_pred, y_prob):
        if truth == 0 and pred == 1:
            dst = os.path.join(out_false_accepts, f"{prob:.4f}_{os.path.basename(src)}")
            shutil.copy2(src, dst)
        elif truth == 1 and pred == 0:
            dst = os.path.join(out_false_rejects, f"{prob:.4f}_{os.path.basename(src)}")
            shutil.copy2(src, dst)

    target_names = ["not_national_id", "national_id"]
    report_text = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    metrics_payload = {
        "model_used": model_name,
        "threshold": threshold,
        "splits": split_results,
        "classification_report_test": report_text,
        "confusion_matrix_test": conf_matrix,
        "false_accepts_count": split_results["test"]["confusion_matrix"]["fp"],
        "false_rejects_count": split_results["test"]["confusion_matrix"]["fn"],
    }

    metrics_json = os.path.join(out_debug, "evaluation_metrics.json")
    report_path = os.path.join(out_debug, "classification_report.txt")
    summary_path = os.path.join(out_debug, "final_metrics_summary.md")
    matrix_json = os.path.join(out_debug, "confusion_matrix.json")
    matrix_png = os.path.join(out_debug, "confusion_matrix.png")

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    with open(matrix_json, "w", encoding="utf-8") as f:
        json.dump({"labels": target_names, "matrix": conf_matrix}, f, indent=2)

    save_confusion_matrix_png(matrix_png, split_results["test"]["confusion_matrix"])

    lines = [
        "# MVP1 Baseline Metrics Summary",
        "",
        f"- Model used: **{model_name}**",
        f"- Threshold: **{threshold}**",
        f"- False accepts (test): **{metrics_payload['false_accepts_count']}**",
        f"- False rejects (test): **{metrics_payload['false_rejects_count']}**",
        "",
        "## Split Metrics",
    ]
    for name, res in split_results.items():
        lines.append(
            f"- {name}: count={res['count']}, accuracy={res['accuracy']:.4f}, "
            f"precision={res['precision']:.4f}, recall={res['recall']:.4f}, "
            f"f1={res['f1']:.4f}, roc_auc={res['roc_auc']}"
        )
    lines.extend(["", "## Test Classification Report", "", "```text", report_text.rstrip(), "```"])
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps(metrics_payload, indent=2))
    print(f"Saved metrics summary at: {summary_path}")


if __name__ == "__main__":
    main()
