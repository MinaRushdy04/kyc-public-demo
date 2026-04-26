import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from modules.face_similarity import ArcFaceOnnxEmbedder, MobileNetFaceBaseline, score_pair
from modules.utils import simple_yaml_load


def choose_threshold(labels: list[int], scores: list[float], target_far: float = 0.01) -> float:
    negatives = sorted([s for y, s in zip(labels, scores) if y == 0], reverse=True)
    if not negatives:
        return 0.70
    idx = min(len(negatives) - 1, max(0, int(np.floor(target_far * len(negatives)))))
    return float(negatives[idx] + 1e-6)


def main() -> None:
    cfg = simple_yaml_load("config_mvp2.yaml")
    project_dir = Path(str(cfg["project_dir"]))
    pairs_csv = Path(str(cfg["prepared_pairs_csv"]))
    reports = project_dir / "reports"
    outputs = project_dir / "outputs" / "mvp2"
    reports.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)

    with pairs_csv.open("r", newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    if str(cfg.get("face_match_model", "")).lower() == "arcface_onnx":
        embedder = ArcFaceOnnxEmbedder(str(cfg["arcface_onnx_path"]))
    else:
        embedder = MobileNetFaceBaseline(image_size=int(cfg.get("image_size", 224)))
    scored = []
    for i, row in enumerate(pairs, start=1):
        try:
            res = score_pair(row["id_image"], row["selfie_image"], embedder)
            status = "ok"
        except Exception as exc:
            res = {"similarity": -1.0, "id_face_found": False, "selfie_face_found": False}
            status = f"error:{type(exc).__name__}"
        scored.append({**row, **res, "status": status})
        if i % 50 == 0:
            print(f"Scored {i}/{len(pairs)} pairs")

    scored_csv = reports / "mvp2_similarity_scores.csv"
    with scored_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(scored[0].keys()) if scored else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored)

    valid = [r for r in scored if r["status"] == "ok"]
    labels = [int(r["label"]) for r in valid]
    scores = [float(r["similarity"]) for r in valid]
    threshold_70 = float(cfg.get("accept_threshold", 0.70))
    threshold_far1 = choose_threshold(labels, scores, target_far=0.01)

    def metrics_at(threshold: float) -> dict:
        preds = [1 if s >= threshold else 0 for s in scores]
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        false_accepts = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
        false_rejects = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
        negatives = sum(1 for y in labels if y == 0)
        positives = sum(1 for y in labels if y == 1)
        return {
            "threshold": threshold,
            "accuracy": accuracy_score(labels, preds),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_accepts": false_accepts,
            "false_rejects": false_rejects,
            "far": false_accepts / max(negatives, 1),
            "frr": false_rejects / max(positives, 1),
        }

    metrics = {
        "model": str(cfg["face_match_model"]),
        "valid_pairs": len(valid),
        "same_person_pairs": sum(labels),
        "different_person_pairs": len(labels) - sum(labels),
        "roc_auc": roc_auc_score(labels, scores) if len(set(labels)) == 2 else None,
        "threshold_0_70": metrics_at(threshold_70),
        "threshold_target_far_1pct": metrics_at(threshold_far1),
        "id_face_miss_count": sum(1 for r in valid if not r["id_face_found"]),
        "selfie_face_miss_count": sum(1 for r in valid if not r["selfie_face_found"]),
    }

    metrics_json = reports / "mvp2_similarity_metrics.json"
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lines = [
        "# MVP2 Face Similarity Baseline Report",
        "",
        f"- Dataset: `{cfg['dataset_repo']}`",
        f"- Dataset URL: {cfg['dataset_url']}",
        f"- License: `{cfg['dataset_license']}`",
        f"- Baseline model: `{cfg['face_match_model']}`",
        f"- Valid pairs scored: **{metrics['valid_pairs']}**",
        f"- Same-person pairs: **{metrics['same_person_pairs']}**",
        f"- Different-person pairs: **{metrics['different_person_pairs']}**",
        f"- ROC-AUC: **{metrics['roc_auc']}**",
        f"- ID face detection misses: **{metrics['id_face_miss_count']}**",
        f"- Selfie face detection misses: **{metrics['selfie_face_miss_count']}**",
        "",
        "## Threshold 0.70",
        "",
    ]
    for key, value in metrics["threshold_0_70"].items():
        lines.append(f"- {key}: **{value}**")
    lines += [
        "",
        "## Threshold Targeting ~1% FAR On This Tiny Sample",
        "",
    ]
    for key, value in metrics["threshold_target_far_1pct"].items():
        lines.append(f"- {key}: **{value}**")
    lines += [
        "",
        "## Verdict",
        "",
        "- This is a baseline MVP2 harness, not a production biometric model.",
        "- For production, replace this ImageNet embedding baseline with ArcFace/InsightFace or an equivalent face-recognition embedding model, then calibrate on consented KYC-style Egyptian/region-relevant samples.",
        "- Liveness/anti-spoofing remains out of scope per current backend constraints.",
    ]
    (reports / "mvp2_similarity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {reports / 'mvp2_similarity_report.md'}")


if __name__ == "__main__":
    main()
