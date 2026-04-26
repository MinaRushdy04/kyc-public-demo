import argparse
import csv
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import ImageFolder

from modules.model import build_model, get_transforms
from modules.utils import simple_yaml_load


def load_cfg(path: str) -> Dict:
    cfg = simple_yaml_load(path)
    defaults = {
        "prepared_dataset_path": r"D:\Projects\KYC\kyc-validator-training\prepared_validator_dataset",
        "model_name": "efficientnetv2_b0",
        "image_size": 224,
        "batch_size": 16,
        "epochs_head": 10,
        "epochs_finetune": 10,
        "learning_rate_head": 0.001,
        "learning_rate_finetune": 0.00001,
        "accept_threshold": 0.95,
        "project_dir": r"D:\Projects\KYC\kyc-validator-training",
    }
    defaults.update(cfg)
    return defaults


def resolve_model_name(requested_name: str) -> str:
    import torchvision.models as models

    name = (requested_name or "").lower()
    if name == "efficientnetv2_b0":
        if hasattr(models, "efficientnet_v2_b0") and hasattr(models, "EfficientNet_V2_B0_Weights"):
            return "efficientnetv2_b0"
        try:
            import timm

            if "tf_efficientnetv2_b0" in timm.list_models("tf_efficientnetv2_b0"):
                return "tf_efficientnetv2_b0"
        except Exception:
            pass
        return "efficientnetb0"
    return name


def stratified_three_way_split(
    labels: List[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    by_class: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(idx)

    g = torch.Generator().manual_seed(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for _, indices in sorted(by_class.items()):
        perm = torch.randperm(len(indices), generator=g).tolist()
        shuffled = [indices[i] for i in perm]
        n_train = max(1, int(len(shuffled) * train_ratio))
        n_val = max(1, int(len(shuffled) * val_ratio))
        train_idx.extend(shuffled[:n_train])
        val_idx.extend(shuffled[n_train : n_train + n_val])
        test_idx.extend(shuffled[n_train + n_val :])
    return train_idx, val_idx, test_idx


def force_manual_hard_negatives_into_train(
    samples: List[Tuple[str, int]],
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
) -> Tuple[List[int], List[int], List[int], int]:
    markers = {
        f"{os.sep}manual_hard_negatives{os.sep}",
        f"{os.sep}manual_hard_negative_augmented{os.sep}",
    }
    forced = {
        idx
        for idx, (path, _) in enumerate(samples)
        if any(marker in os.path.abspath(path) for marker in markers)
    }
    if not forced:
        return train_idx, val_idx, test_idx, 0
    train_set = set(train_idx) | forced
    val_idx = [idx for idx in val_idx if idx not in forced]
    test_idx = [idx for idx in test_idx if idx not in forced]
    return sorted(train_set), val_idx, test_idx, len(forced)


def set_backbone_trainable(model, trainable: bool) -> None:
    for name, p in model.named_parameters():
        if "classifier" in name:
            p.requires_grad = True
        else:
            p.requires_grad = trainable


def extract_features(model, loader, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    features_out: List[torch.Tensor] = []
    labels_out: List[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if hasattr(model, "forward_features"):
                feats = model.forward_features(x)
                if hasattr(model, "global_pool"):
                    feats = model.global_pool(feats)
                else:
                    feats = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(feats, 1), 1)
            else:
                feats = model.features(x)
                feats = model.avgpool(feats)
                feats = torch.flatten(feats, 1)
            features_out.append(feats.cpu())
            labels_out.append(y.cpu())
    return torch.cat(features_out, dim=0), torch.cat(labels_out, dim=0)


def classifier_epoch(
    classifier,
    loader,
    criterion,
    positive_class_idx: int,
    optimizer=None,
    device: str = "cpu",
) -> Tuple[float, float]:
    is_train = optimizer is not None
    classifier.train(is_train)
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.set_grad_enabled(is_train):
        for feats, y in loader:
            feats = feats.to(device)
            y_bin = (y == positive_class_idx).float().unsqueeze(1).to(device)
            logits = classifier(feats)
            loss = criterion(logits, y_bin)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * feats.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += int((preds == y_bin.long()).sum().item())
            total += feats.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def positive_class_weight(labels: List[int], positive_class_idx: int) -> float:
    positives = sum(1 for y in labels if int(y) == positive_class_idx)
    negatives = len(labels) - positives
    if positives <= 0:
        return 1.0
    return max(1.0, negatives / positives)


def save_history_csv(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["stage", "epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_training_curve(path: str, history: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w, h = 900, 520
    margin_l, margin_r, margin_t, margin_b = 70, 30, 35, 65
    im = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(im)
    x0, y0 = margin_l, margin_t
    x1, y1 = w - margin_r, h - margin_b
    draw.rectangle((x0, y0, x1, y1), outline=(25, 25, 25), width=1)
    draw.text((x0, 10), "MVP1 Training Curves", fill=(20, 20, 20))

    if not history:
        draw.text((x0 + 20, y0 + 20), "No history", fill=(20, 20, 20))
        im.save(path)
        return

    epochs = list(range(1, len(history) + 1))
    train_loss = [float(r["train_loss"]) for r in history]
    val_loss = [float(r["val_loss"]) for r in history]
    max_loss = max(max(train_loss), max(val_loss), 1e-6)

    def point(i: int, value: float) -> Tuple[int, int]:
        px = x0 + int((epochs[i] - 1) / max(1, len(epochs) - 1) * (x1 - x0))
        py = y1 - int(value / max_loss * (y1 - y0))
        return px, py

    for i in range(len(history) - 1):
        draw.line((point(i, train_loss[i]), point(i + 1, train_loss[i + 1])), fill=(30, 120, 190), width=3)
        draw.line((point(i, val_loss[i]), point(i + 1, val_loss[i + 1])), fill=(210, 80, 60), width=3)
    draw.text((x0, y1 + 15), "blue=train loss, red=validation loss", fill=(20, 20, 20))
    im.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MVP1 validator baseline.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--epochs-head", type=int, default=None)
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="Optional existing validator checkpoint to warm-start from instead of downloading pretrained weights.",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    project_dir = os.path.abspath(cfg["project_dir"])
    prepared = os.path.abspath(cfg["prepared_dataset_path"])
    models_dir = os.path.join(project_dir, "models")
    outputs_debug = os.path.join(project_dir, "outputs", "debug")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_debug, exist_ok=True)

    train_tf, eval_tf = get_transforms(int(cfg["image_size"]))
    base_ds = ImageFolder(prepared, transform=None)
    if len(base_ds.classes) != 2:
        raise SystemExit("Expected exactly two class folders under prepared dataset.")

    class_to_idx = base_ds.class_to_idx
    positive_class_name = "national_id"
    if positive_class_name not in class_to_idx:
        raise SystemExit(f"Missing required class: {positive_class_name}")
    positive_class_idx = int(class_to_idx[positive_class_name])

    train_idx, val_idx, test_idx = stratified_three_way_split(base_ds.targets, seed=42)
    train_idx, val_idx, test_idx, forced_train_count = force_manual_hard_negatives_into_train(
        base_ds.samples, train_idx, val_idx, test_idx
    )
    split_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classes": base_ds.classes,
        "class_to_idx": class_to_idx,
        "positive_class_name": positive_class_name,
        "positive_class_idx": positive_class_idx,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "test_count": len(test_idx),
        "forced_manual_hard_negative_train_count": forced_train_count,
    }
    split_path = os.path.join(models_dir, "mvp1_split.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)

    train_ds = ImageFolder(prepared, transform=train_tf)
    eval_ds = ImageFolder(prepared, transform=eval_tf)
    train_img_loader = DataLoader(Subset(train_ds, train_idx), batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)
    val_img_loader = DataLoader(Subset(eval_ds, val_idx), batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)

    requested_model = str(cfg["model_name"])
    actual_model = resolve_model_name(requested_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_checkpoint = os.path.abspath(args.init_checkpoint) if args.init_checkpoint else ""
    if init_checkpoint and os.path.isfile(init_checkpoint):
        payload = torch.load(init_checkpoint, map_location=device)
        warm_model_name = str(payload.get("actual_model_name") or payload.get("requested_model_name") or requested_model)
        actual_model = warm_model_name
        model = build_model(warm_model_name, pretrained=False).to(device)
        model.load_state_dict(payload["model_state"])
        init_source = init_checkpoint
    else:
        model = build_model(requested_model, pretrained=True).to(device)
        init_source = "pretrained_imagenet_weights"
    set_backbone_trainable(model, trainable=False)

    print(f"Requested model: {requested_model}")
    print(f"Model used: {actual_model}")
    print(f"Initialization source: {init_source}")
    print(f"Device: {device}")
    print(f"Split counts: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"Forced manual hard negatives into train: {forced_train_count}")
    print("Extracting frozen EfficientNet features...")
    train_feats, train_labels = extract_features(model, train_img_loader, device=device)
    val_feats, val_labels = extract_features(model, val_img_loader, device=device)

    feature_batch = min(256, max(16, int(cfg["batch_size"]) * 8))
    train_loader = DataLoader(TensorDataset(train_feats, train_labels), batch_size=feature_batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_feats, val_labels), batch_size=feature_batch, shuffle=False)

    classifier = model.classifier.to(device)
    pos_weight_value = positive_class_weight([base_ds.targets[i] for i in train_idx], positive_class_idx)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device))
    print(f"Positive class loss weight: {pos_weight_value:.4f}")
    optimizer = optim.Adam(classifier.parameters(), lr=float(cfg["learning_rate_head"]))
    epochs = int(args.epochs_head if args.epochs_head is not None else cfg["epochs_head"])
    best_val_loss = float("inf")
    best_classifier_state = None
    patience = 0
    history: List[Dict] = []

    for epoch in range(epochs):
        tr_loss, tr_acc = classifier_epoch(
            classifier,
            train_loader,
            criterion,
            positive_class_idx=positive_class_idx,
            optimizer=optimizer,
            device=device,
        )
        va_loss, va_acc = classifier_epoch(
            classifier,
            val_loader,
            criterion,
            positive_class_idx=positive_class_idx,
            optimizer=None,
            device=device,
        )
        row = {
            "stage": "frozen_feature_head",
            "epoch": epoch + 1,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_acc": tr_acc,
            "val_acc": va_acc,
        }
        history.append(row)
        print(
            f"[head][{epoch + 1}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"train_acc={tr_acc:.4f} val_acc={va_acc:.4f}",
            flush=True,
        )
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_classifier_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(args.early_stop_patience):
                print("Early stopping triggered.")
                break

    if best_classifier_state is not None:
        classifier.load_state_dict(best_classifier_state)
        model.classifier.load_state_dict(best_classifier_state)

    ckpt_path = os.path.join(models_dir, "validator_best.pt")
    torch.save(
        {
            "model_state": model.cpu().state_dict(),
            "classes": base_ds.classes,
            "class_to_idx": class_to_idx,
            "positive_class_name": positive_class_name,
            "positive_class_idx": positive_class_idx,
            "requested_model_name": requested_model,
            "actual_model_name": actual_model,
            "config": cfg,
            "training_mode": "frozen_feature_head_transfer_learning",
            "split_path": split_path,
            "positive_class_loss_weight": pos_weight_value,
            "initialization_source": init_source,
        },
        ckpt_path,
    )

    history_json = os.path.join(models_dir, "training_history.json")
    history_csv = os.path.join(models_dir, "training_history.csv")
    curve_png = os.path.join(outputs_debug, "training_curves.png")
    with open(history_json, "w", encoding="utf-8") as f:
        json.dump({"history": history, "model_used": actual_model, "split": split_payload}, f, indent=2)
    save_history_csv(history_csv, history)
    save_training_curve(curve_png, history)

    print(f"Best checkpoint saved at: {ckpt_path}")
    print(f"Training history saved at: {history_json}")
    print(f"Training curves saved at: {curve_png}")


if __name__ == "__main__":
    main()
