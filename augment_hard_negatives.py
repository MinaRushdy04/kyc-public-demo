import argparse
import csv
import hashlib
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError

from modules.utils import simple_yaml_load


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_cfg(path: str) -> dict:
    cfg = simple_yaml_load(path)
    defaults = {
        "project_dir": r"D:\Projects\KYC\kyc-validator-training",
        "prepared_dataset_path": r"D:\Projects\KYC\kyc-validator-training\prepared_validator_dataset",
    }
    defaults.update(cfg)
    return defaults


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_image(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, RuntimeError):
        return None


def jitter_image(im: Image.Image, rng: random.Random) -> Image.Image:
    out = im.copy()
    w, h = out.size

    # Keep the document identity intact while varying capture-like conditions.
    scale = rng.uniform(0.92, 1.0)
    crop_w = max(16, int(w * scale))
    crop_h = max(16, int(h * scale))
    left = rng.randint(0, max(0, w - crop_w))
    top = rng.randint(0, max(0, h - crop_h))
    out = out.crop((left, top, left + crop_w, top + crop_h)).resize((w, h), Image.Resampling.BICUBIC)

    angle = rng.uniform(-5.0, 5.0)
    out = out.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(255, 255, 255))

    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.82, 1.18))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.85, 1.22))
    out = ImageEnhance.Color(out).enhance(rng.uniform(0.85, 1.15))

    if rng.random() < 0.30:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.75)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create safe augmented variants of reviewed hard negatives.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--variants-per-image", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    project_dir = Path(str(cfg["project_dir"]))
    prepared = Path(str(cfg["prepared_dataset_path"]))
    source_dir = prepared / "not_national_id" / "manual_hard_negatives"
    target_dir = prepared / "not_national_id" / "manual_hard_negative_augmented"
    target_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rows = []
    copied = 0
    skipped = 0
    seen_hashes = set()

    for src in sorted(source_dir.iterdir() if source_dir.exists() else []):
        if not src.is_file() or src.suffix.lower() not in IMAGE_EXTS:
            continue
        im = load_image(src)
        if im is None:
            skipped += 1
            continue
        for i in range(args.variants_per_image):
            aug = jitter_image(im, rng)
            tmp_path = target_dir / f"aug_{src.stem}_{i:03d}.jpg"
            aug.save(tmp_path, quality=92)
            data = tmp_path.read_bytes()
            digest = sha256_bytes(data)
            if digest in seen_hashes:
                tmp_path.unlink(missing_ok=True)
                skipped += 1
                continue
            seen_hashes.add(digest)
            copied += 1
            rows.append(
                {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "source_path": str(src),
                    "saved_path": str(tmp_path),
                    "sha256": digest,
                }
            )

    log_path = project_dir / "reports" / "hard_negative_augmentation_report.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["created_utc", "source_path", "saved_path", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Augmented hard negatives written: {copied}")
    print(f"Skipped: {skipped}")
    print(f"Output dir: {target_dir}")
    print(f"Report: {log_path}")


if __name__ == "__main__":
    main()
