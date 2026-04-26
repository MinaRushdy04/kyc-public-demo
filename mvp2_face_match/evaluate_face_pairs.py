from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from face_match import load_models, match_faces
from modules.score_calibration import calibrated_score_from_cosine
from modules.utils import apply_face_model_profile, ensure_dir, load_config, project_relative_default_config


SCORABLE_STATUSES = {"FACE_MATCH", "FACE_UNCERTAIN", "FACE_MISMATCH"}
FIXED_THRESHOLDS = [50, 60, 70, 80]


def metrics_at(labels: list[int], scores: list[float], threshold: float) -> dict:
    preds = [1 if s >= threshold else 0 for s in scores]
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "far_fpr": float(fp / max(fp + tn, 1)),
        "frr_fnr": float(fn / max(fn + tp, 1)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_curve(labels: list[int], scores: list[float]) -> list[dict]:
    return [metrics_at(labels, scores, threshold) for threshold in range(0, 101, 5)]


def recommend_threshold(curve: list[dict], max_far: float = 0.02) -> dict:
    candidates = [row for row in curve if row["far_fpr"] <= max_far]
    if candidates:
        return max(candidates, key=lambda row: (row["recall"], row["accuracy"], -row["threshold"]))
    return min(curve, key=lambda row: (row["far_fpr"], row["frr_fnr"], -row["accuracy"]))


def reject_metrics_at(labels: list[int], scores: list[float], threshold: float) -> dict:
    positives = max(sum(labels), 1)
    negatives = max(len(labels) - sum(labels), 1)
    auto_reject = [s < threshold for s in scores]
    false_rejects = sum(1 for y, reject in zip(labels, auto_reject) if y == 1 and reject)
    true_rejects = sum(1 for y, reject in zip(labels, auto_reject) if y == 0 and reject)
    return {
        "threshold": float(threshold),
        "auto_reject_count": int(sum(auto_reject)),
        "true_rejects": int(true_rejects),
        "false_rejects": int(false_rejects),
        "auto_reject_coverage": float(sum(auto_reject) / max(len(labels), 1)),
        "true_negative_auto_reject_rate": float(true_rejects / negatives),
        "frr_fnr": float(false_rejects / positives),
    }


def recommend_triage_thresholds(
    labels: list[int],
    scores: list[float],
    max_far: float,
    max_frr: float,
) -> dict:
    thresholds = [round(float(t), 2) for t in np.arange(0.0, 100.01, 0.5)]
    accept_curve = [metrics_at(labels, scores, t) for t in thresholds]
    accept_candidates = [row for row in accept_curve if row["far_fpr"] <= max_far and row["confusion_matrix"]["tp"] > 0]
    if accept_candidates:
        accept = max(accept_candidates, key=lambda row: (row["recall"], row["accuracy"], -row["threshold"]))
    else:
        accept = min(accept_curve, key=lambda row: (row["far_fpr"], row["frr_fnr"], -row["accuracy"]))

    reject_curve = [reject_metrics_at(labels, scores, t) for t in thresholds if t <= accept["threshold"]]
    reject_candidates = [row for row in reject_curve if row["frr_fnr"] <= max_frr]
    if reject_candidates:
        reject = max(
            reject_candidates,
            key=lambda row: (row["true_negative_auto_reject_rate"], -row["frr_fnr"], row["threshold"]),
        )
    else:
        reject = min(reject_curve, key=lambda row: (row["frr_fnr"], -row["true_negative_auto_reject_rate"]))
    return {
        "auto_reject_below": float(reject["threshold"]),
        "auto_accept_at_or_above": float(accept["threshold"]),
        "accept_threshold_metrics": accept,
        "reject_threshold_metrics": reject,
        "target_max_far": float(max_far),
        "target_max_frr": float(max_frr),
    }


def fit_logistic_calibration(labels: list[int], cosines: list[float], cfg: dict, profile: str) -> dict | None:
    if len(labels) < 8 or len(set(labels)) < 2:
        return None
    x = np.asarray(cosines, dtype=np.float32).reshape(-1, 1)
    y = np.asarray(labels, dtype=np.int32)
    model = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=42)
    model.fit(x, y)
    calibration = {
        "name": f"mvp2_{profile}_logistic_cosine",
        "version": 1,
        "model_profile": profile,
        "raw_metric": "cosine_similarity",
        "score_output": "calibrated_match_probability_0_100",
        "calibrator": {
            "type": "logistic_cosine",
            "coef": float(model.coef_[0][0]),
            "intercept": float(model.intercept_[0]),
        },
        "training_pairs": int(len(labels)),
        "positive_pairs": int(sum(labels)),
        "negative_pairs": int(len(labels) - sum(labels)),
    }
    calibrated_scores = [calibrated_score_from_cosine(cos, calibration) for cos in cosines]
    max_far = float(cfg.get("calibration_target_max_far", 0.02))
    max_frr = float(cfg.get("calibration_target_max_frr", 0.05))
    calibration["thresholds"] = recommend_triage_thresholds(labels, calibrated_scores, max_far, max_frr)
    calibration["score_distribution"] = score_distribution(labels, calibrated_scores)
    calibration["roc_auc"] = roc_auc_score(labels, calibrated_scores) if len(set(labels)) == 2 else None
    return calibration


def score_distribution(labels: list[int], scores: list[float]) -> dict:
    def dist(vals: list[float]) -> dict:
        if not vals:
            return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
        return {
            "count": len(vals),
            "min": float(np.min(vals)),
            "p25": float(np.percentile(vals, 25)),
            "median": float(np.median(vals)),
            "p75": float(np.percentile(vals, 75)),
            "max": float(np.max(vals)),
        }

    genuine = [s for y, s in zip(labels, scores) if y == 1]
    impostor = [s for y, s in zip(labels, scores) if y == 0]
    return {"all": dist(scores), "genuine_same_person": dist(genuine), "impostor_different_person": dist(impostor)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate/calibrate MVP2 face-pair similarity.")
    parser.add_argument("--config", default=str(project_relative_default_config()))
    parser.add_argument("--pairs", default="")
    parser.add_argument("--det-input-size", type=int, default=320)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit for quick validation.")
    parser.add_argument("--face-profile", choices=["stable", "accurate"], default=None)
    parser.add_argument("--write-calibration", action="store_true", help="Save calibrated scoring artifact for this face profile.")
    args = parser.parse_args()

    cfg = apply_face_model_profile(load_config(args.config), args.face_profile)
    cfg["det_input_size"] = int(args.det_input_size)
    pairs_path = Path(args.pairs or str(cfg["pairs_csv"]))
    reports_dir = ensure_dir(Path(str(cfg["face_project_dir"])) / "reports")
    failed_dir = ensure_dir(Path(str(cfg["face_project_dir"])) / "outputs" / "failed_detections")

    with pairs_path.open("r", newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    if int(args.limit) > 0:
        pairs = pairs[: int(args.limit)]

    detector, embedder = load_models(cfg)
    rows = []
    for i, row in enumerate(pairs, start=1):
        result = match_faces(row["id_image"], row["selfie_image"], cfg, detector=detector, embedder=embedder)
        score = result["score"]
        cosine = result["cosine_similarity"]
        out = {
            **row,
            "score": "" if score is None else score,
            "raw_score": "" if result.get("raw_score") is None else result.get("raw_score", score),
            "score_mode": result.get("score_mode", "raw_cosine_score"),
            "calibrated": result.get("calibrated", False),
            "cosine_similarity": "" if cosine is None else cosine,
            "status": result["status"],
            "decision": result["decision"],
            "id_face_detected": result["faces"]["id_face_detected"],
            "selfie_face_detected": result["faces"]["selfie_face_detected"],
            "selfie_face_count": result["faces"]["selfie_face_count"],
            "id_face_quality_ok": result["quality"]["id_face_quality_ok"],
            "selfie_face_quality_ok": result["quality"]["selfie_face_quality_ok"],
            "warnings": "|".join(result["quality"]["warnings"]),
        }
        rows.append(out)
        if result["status"] not in SCORABLE_STATUSES:
            safe_id = row["pair_id"].replace("/", "_").replace("\\", "_")
            (failed_dir / f"{safe_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if i % 50 == 0:
            print(f"Evaluated {i}/{len(pairs)} pairs")
            gc.collect()

    scores_csv = reports_dir / "face_similarity_scores.csv"
    with scores_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    scorable = [r for r in rows if r["status"] in SCORABLE_STATUSES and r["score"] != ""]
    labels = [int(r["label"]) for r in scorable]
    scores = [float(r["raw_score"] or r["score"]) for r in scorable]
    cosines = [float(r["cosine_similarity"]) for r in scorable]
    curve = threshold_curve(labels, scores) if scores else []
    fixed = {str(t): metrics_at(labels, scores, t) for t in FIXED_THRESHOLDS} if scores else {}
    recommendation = recommend_threshold(curve, max_far=0.02) if curve else {}
    profile = str(cfg.get("face_model_profile", "stable"))
    calibration = fit_logistic_calibration(labels, cosines, cfg, profile) if scores else None
    calibrated_scores = [calibrated_score_from_cosine(cos, calibration) for cos in cosines] if calibration else []
    calibrated_fixed = {str(t): metrics_at(labels, calibrated_scores, t) for t in FIXED_THRESHOLDS} if calibrated_scores else {}
    calibrated_triage = calibration.get("thresholds") if calibration else {}
    raw_triage = recommend_triage_thresholds(
        labels,
        scores,
        float(cfg.get("calibration_target_max_far", 0.02)),
        float(cfg.get("calibration_target_max_frr", 0.05)),
    ) if scores else {}
    if calibration:
        calibration["runtime_score_mode"] = "raw_score_with_calibrated_thresholds"
        calibration["raw_score_thresholds"] = raw_triage
        calibration["probability_thresholds"] = calibrated_triage
        calibration["thresholds"] = raw_triage

    calibration_path = None
    if calibration and args.write_calibration:
        calibration_path = Path(str(cfg.get(f"face_calibration_path_{profile}", reports_dir / f"face_score_calibration_{profile}.json")))
        ensure_dir(calibration_path.parent)
        calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    metrics = {
        "note": "Suitable for prototype testing and threshold calibration only. Do not train ArcFace from this report.",
        "pairs_total": len(rows),
        "pairs_scorable_for_calibration": len(scorable),
        "pairs_excluded_failed_or_low_quality": len(rows) - len(scorable),
        "positive_pairs_scorable": sum(labels),
        "negative_pairs_scorable": len(labels) - sum(labels),
        "roc_auc_scorable": roc_auc_score(labels, scores) if len(set(labels)) == 2 else None,
        "model_profile": profile,
        "score_distribution": score_distribution(labels, scores),
        "metrics_at_thresholds": fixed,
        "recommended_threshold_max_far_2pct": recommendation,
        "calibration_written": str(calibration_path) if calibration_path else None,
        "calibrated_score_distribution": score_distribution(labels, calibrated_scores) if calibrated_scores else {},
        "calibrated_metrics_at_thresholds": calibrated_fixed,
        "calibrated_recommended_triage_thresholds": calibrated_triage,
        "raw_score_recommended_triage_thresholds": raw_triage,
        "threshold_curve": curve,
        "excluded_status_counts": {
            status: sum(1 for r in rows if r["status"] == status and r not in scorable)
            for status in sorted({r["status"] for r in rows if r not in scorable})
        },
    }
    (reports_dir / "face_eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lines = [
        "# MVP2 Face Pair Calibration Evaluation",
        "",
        "**Suitable for prototype testing and threshold calibration only. Do not train ArcFace from this report.**",
        "",
        f"- Total pairs: **{metrics['pairs_total']}**",
        f"- Scorable pairs used for calibration: **{metrics['pairs_scorable_for_calibration']}**",
        f"- Excluded failed/low-quality pairs: **{metrics['pairs_excluded_failed_or_low_quality']}**",
        f"- Positive scorable pairs: **{metrics['positive_pairs_scorable']}**",
        f"- Negative scorable pairs: **{metrics['negative_pairs_scorable']}**",
        f"- ROC-AUC on scorable pairs: **{metrics['roc_auc_scorable']}**",
        "",
        "## Fixed Thresholds",
        "",
        "| threshold | accuracy | FAR/FPR | FRR/FNR | precision | recall | TP | FP | TN | FN |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in FIXED_THRESHOLDS:
        row = fixed.get(str(t), {})
        cm = row.get("confusion_matrix", {})
        lines.append(
            f"| {t} | {row.get('accuracy', 0):.4f} | {row.get('far_fpr', 0):.4f} | {row.get('frr_fnr', 0):.4f} | "
            f"{row.get('precision', 0):.4f} | {row.get('recall', 0):.4f} | {cm.get('tp', 0)} | {cm.get('fp', 0)} | {cm.get('tn', 0)} | {cm.get('fn', 0)} |"
        )
    lines += [
        "",
        "## Calibrated Match Probability",
        "",
        f"- Model profile: **{profile}**",
        f"- Calibration artifact: **{calibration_path or 'not written; rerun with --write-calibration'}**",
        f"- Score mode after calibration: **calibrated_match_probability**",
        f"- Auto-reject below: **{calibrated_triage.get('auto_reject_below')}**",
        f"- Auto-accept at or above: **{calibrated_triage.get('auto_accept_at_or_above')}**",
        "",
        "Runtime uses raw similarity score with calibrated thresholds by default because it is easier for testers to interpret.",
        "",
        f"- Runtime raw-score auto-reject below: **{raw_triage.get('auto_reject_below')}**",
        f"- Runtime raw-score auto-accept at or above: **{raw_triage.get('auto_accept_at_or_above')}**",
        "",
        "| calibrated threshold | accuracy | FAR/FPR | FRR/FNR | precision | recall | TP | FP | TN | FN |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in FIXED_THRESHOLDS:
        row = calibrated_fixed.get(str(t), {})
        cm = row.get("confusion_matrix", {})
        lines.append(
            f"| {t} | {row.get('accuracy', 0):.4f} | {row.get('far_fpr', 0):.4f} | {row.get('frr_fnr', 0):.4f} | "
            f"{row.get('precision', 0):.4f} | {row.get('recall', 0):.4f} | {cm.get('tp', 0)} | {cm.get('fp', 0)} | {cm.get('tn', 0)} | {cm.get('fn', 0)} |"
        )
    lines += [
        "",
        "## Recommended Threshold",
        "",
        f"- Target: highest recall while keeping FAR/FPR <= 2% on this calibration set.",
        f"- Recommendation: **{recommendation.get('threshold')}**",
        f"- FAR/FPR: **{recommendation.get('far_fpr')}**",
        f"- FRR/FNR: **{recommendation.get('frr_fnr')}**",
        f"- Accuracy: **{recommendation.get('accuracy')}**",
        "",
        "## Quality Handling",
        "",
        "- Failed detections and low-quality faces are excluded from threshold math.",
        "- `LOW_QUALITY_ID_FACE` and `LOW_QUALITY_SELFIE_FACE` should trigger `REJECT_RETRY`, not silent zero-similarity scoring.",
        "",
        "## Production Warning",
        "",
        "- No liveness or anti-spoofing is included.",
        "- Printed-photo and screen-replay attacks are not reliably prevented.",
        "- Production calibration requires consented, representative ID/selfie pairs from the real capture flow.",
    ]
    (reports_dir / "face_eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {scores_csv}")
    print(f"Wrote {reports_dir / 'face_eval_report.md'}")


if __name__ == "__main__":
    main()
