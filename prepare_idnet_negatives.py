import argparse
import os
import random
import shutil
import tarfile
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

from modules.report import save_markdown
from modules.utils import ensure_dir, is_image_file, list_files, relpath, sha256_file, slugify_filename


FRAUD_HINTS = ("fraud", "inpaint", "rewrite", "crop_and_replace", "replace", "manip")
POSITIVE_HINTS = ("positive", "bonafide", "original")


def safe_extract(tar_path: str, extract_dir: str) -> None:
    ensure_dir(extract_dir)
    target_abs = os.path.abspath(extract_dir)
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            member_path = os.path.abspath(os.path.join(extract_dir, member.name))
            if not member_path.startswith(target_abs + os.sep) and member_path != target_abs:
                raise RuntimeError(f"Blocked unsafe archive member: {member.name}")
        tar.extractall(extract_dir)


def is_readable(path: str) -> bool:
    try:
        with Image.open(path) as im:
            ImageOps.exif_transpose(im).convert("RGB").resize((8, 8))
            return True
    except (AttributeError, UnidentifiedImageError, OSError, ValueError, RuntimeError):
        return False


def classify_idnet(path: str) -> str:
    p = path.lower().replace("\\", "/")
    if any(k in p for k in FRAUD_HINTS):
        return "idnet_fraud"
    if any(k in p for k in POSITIVE_HINTS):
        return "idnet_non_egyptian_positive"
    return "idnet_other"


def collect_existing_hashes(output_root: str) -> Tuple[set, int, int]:
    hashes = set()
    duplicates = 0
    corrupted = 0
    if not os.path.isdir(output_root):
        return hashes, duplicates, corrupted
    for p in list_files(output_root):
        if not is_image_file(p):
            continue
        if not is_readable(p):
            corrupted += 1
            continue
        digest = sha256_file(p)
        if digest in hashes:
            duplicates += 1
        hashes.add(digest)
    return hashes, duplicates, corrupted


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare IDNet-2025 hard negatives for MVP1.")
    parser.add_argument("--archive", default=r"D:\Datasets\IDNet_2025_downloads\EST.tar.gz")
    parser.add_argument("--extract-dir", default=r"D:\Datasets\IDNet_2025_extracted\EST")
    parser.add_argument(
        "--output-dir",
        default=r"D:\Projects\KYC\kyc-validator-training\prepared_validator_dataset\not_national_id\idnet2025_selected",
    )
    parser.add_argument(
        "--report-path",
        default=r"D:\Projects\KYC\kyc-validator-training\reports\idnet2025_merge_report.md",
    )
    parser.add_argument("--target-add", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    archive = os.path.abspath(args.archive)
    extract_dir = os.path.abspath(args.extract_dir)
    output_dir = os.path.abspath(args.output_dir)
    report_path = os.path.abspath(args.report_path)
    ensure_dir(output_dir)

    lines: List[str] = [
        "# IDNet-2025 Merge Report",
        "",
        f"- Time (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        f"- Archive: `{archive}`",
        f"- Extract dir: `{extract_dir}`",
        f"- Output dir: `{output_dir}`",
        "",
    ]

    if not os.path.isfile(archive):
        lines.append("## Result")
        lines.append("- Archive not found. Nothing merged.")
        save_markdown(report_path, lines)
        print(f"Wrote {report_path}")
        return

    if not os.path.isdir(extract_dir) or not any(is_image_file(p) for p in list_files(extract_dir)):
        safe_extract(archive, extract_dir)

    images = [p for p in list_files(extract_dir) if is_image_file(p)]
    buckets: Dict[str, List[str]] = {"idnet_fraud": [], "idnet_non_egyptian_positive": [], "idnet_other": []}
    for p in images:
        buckets[classify_idnet(p)].append(p)

    rng = random.Random(args.seed)
    for paths in buckets.values():
        rng.shuffle(paths)

    not_national_root = os.path.abspath(os.path.join(output_dir, os.pardir))
    existing_hashes, existing_duplicates, existing_corrupted = collect_existing_hashes(not_national_root)
    copied = 0
    duplicates_skipped = 0
    corrupted_skipped = 0
    copied_by_bucket = Counter()

    ordered = buckets["idnet_fraud"] + buckets["idnet_non_egyptian_positive"] + buckets["idnet_other"]
    for src in ordered:
        if copied >= int(args.target_add):
            break
        if not is_readable(src):
            corrupted_skipped += 1
            continue
        digest = sha256_file(src)
        if digest in existing_hashes:
            duplicates_skipped += 1
            continue
        existing_hashes.add(digest)
        bucket = classify_idnet(src)
        src_rel = relpath(src, extract_dir)
        out_name = slugify_filename(f"{bucket}_{os.path.splitext(os.path.basename(src))[0]}_{digest[:10]}.jpg")
        out_path = os.path.join(output_dir, out_name)
        shutil.copy2(src, out_path)
        copied += 1
        copied_by_bucket[bucket] += 1

    lines.extend(
        [
            "## Discovery",
            f"- Images discovered: **{len(images)}**",
            f"- Fraud/manipulated candidates: **{len(buckets['idnet_fraud'])}**",
            f"- Non-Egyptian positive ID candidates: **{len(buckets['idnet_non_egyptian_positive'])}**",
            f"- Other image candidates: **{len(buckets['idnet_other'])}**",
            "",
            "## Merge",
            f"- Target to add: **{args.target_add}**",
            f"- Copied this run: **{copied}**",
            f"- Copied fraud/manipulated: **{copied_by_bucket['idnet_fraud']}**",
            f"- Copied non-Egyptian positive IDs: **{copied_by_bucket['idnet_non_egyptian_positive']}**",
            f"- Copied other: **{copied_by_bucket['idnet_other']}**",
            f"- Duplicates skipped this run: **{duplicates_skipped}**",
            f"- Corrupted skipped this run: **{corrupted_skipped}**",
            f"- Existing duplicate hashes observed under not_national_id: **{existing_duplicates}**",
            f"- Existing corrupted files observed under not_national_id: **{existing_corrupted}**",
        ]
    )

    save_markdown(report_path, lines)
    print(f"Wrote {report_path}")
    print(f"Copied {copied} IDNet negatives.")


if __name__ == "__main__":
    main()
