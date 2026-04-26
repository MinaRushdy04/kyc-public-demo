import argparse
import csv
import hashlib
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

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


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            ImageOps.exif_transpose(im).convert("RGB").verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError, RuntimeError):
        return False


def existing_hashes(root: Path) -> set[str]:
    hashes = set()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            try:
                hashes.add(file_sha256(path))
            except OSError:
                continue
    return hashes


def append_log(log_path: Path, row: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "created_utc",
                "source_path",
                "saved_path",
                "sha256",
                "reason",
                "status",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add manually reviewed false accepts as hard negatives.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--reason", default="manual_false_accept")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    prepared = Path(str(cfg["prepared_dataset_path"]))
    target_dir = prepared / "not_national_id" / "manual_hard_negatives"
    target_dir.mkdir(parents=True, exist_ok=True)

    src = Path(args.image).expanduser().resolve()
    log_path = Path(str(cfg["project_dir"])) / "reports" / "hard_negative_log.csv"

    if not src.is_file() or src.suffix.lower() not in IMAGE_EXTS:
        raise SystemExit(f"Not a supported image file: {src}")
    if not is_readable_image(src):
        append_log(
            log_path,
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_path": str(src),
                "saved_path": "",
                "sha256": "",
                "reason": args.reason,
                "status": "corrupted_or_unreadable",
            },
        )
        raise SystemExit(f"Unreadable image skipped: {src}")

    sha = file_sha256(src)
    all_negative_hashes = existing_hashes(prepared / "not_national_id")
    if sha in all_negative_hashes:
        append_log(
            log_path,
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_path": str(src),
                "saved_path": "",
                "sha256": sha,
                "reason": args.reason,
                "status": "duplicate_skipped",
            },
        )
        print(f"Duplicate hard negative skipped: {src}")
        return

    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in src.stem)[:80]
    dst = target_dir / f"manual_{safe_stem}_{sha[:10]}{src.suffix.lower()}"
    shutil.copy2(src, dst)

    append_log(
        log_path,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_path": str(src),
            "saved_path": str(dst),
            "sha256": sha,
            "reason": args.reason,
            "status": "copied",
        },
    )
    print(f"Copied hard negative: {dst}")


if __name__ == "__main__":
    main()
