import argparse
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

from modules.report import save_markdown
from modules.utils import (
    contains_arabic,
    is_image_file,
    keyword_hit,
    list_files,
    relpath,
    sha256_file,
    slugify_filename,
)


ARABIC_HINTS = [
    "arabic",
    "arab",
    "ar_",
    "_ar",
    "egypt",
    "egy",
    "msa",
]
FORGE_HINTS = [
    "forg",
    "fake",
    "attack",
    "manip",
    "spoof",
    "tamper",
    "counterfeit",
    "morph",
]
BONAFIDE_HINTS = [
    "bonafide",
    "bona_fide",
    "genuine",
    "authentic",
    "real",
]


def classify_priority(rel_image_path: str) -> Tuple[str, bool, bool, bool]:
    low = rel_image_path.lower()
    is_ar = any(h in low for h in ARABIC_HINTS) or contains_arabic(rel_image_path)
    is_forge = keyword_hit(low, FORGE_HINTS)
    is_bona = keyword_hit(low, BONAFIDE_HINTS)
    if is_ar and is_forge:
        return "priority_1_arabic_forged", is_ar, is_forge, is_bona
    if is_ar and is_bona:
        return "priority_2_arabic_bonafide", is_ar, is_forge, is_bona
    if (not is_ar) and is_forge:
        return "priority_3_nonarabic_forged", is_ar, is_forge, is_bona
    return "other", is_ar, is_forge, is_bona


def is_readable_image(path: str) -> bool:
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError, RuntimeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Select FantasyID negatives for MVP1.")
    parser.add_argument("--extract-dir", default=r"D:\Datasets\FantasyID_extracted")
    parser.add_argument(
        "--output-dir",
        default=r"D:\Projects\KYC\kyc-validator-training\prepared_validator_dataset\not_national_id\fantasyid_selected",
    )
    parser.add_argument(
        "--report-path",
        default=r"D:\Projects\KYC\kyc-validator-training\reports\fantasyid_selection_report.md",
    )
    parser.add_argument("--target-total", type=int, default=950)
    args = parser.parse_args()

    extract_dir = os.path.abspath(args.extract_dir)
    output_dir = os.path.abspath(args.output_dir)
    report_path = os.path.abspath(args.report_path)
    target_total = int(args.target_total)
    os.makedirs(output_dir, exist_ok=True)

    lines: List[str] = []
    lines.append("# FantasyID Selection Report")
    lines.append("")
    lines.append(f"- Extract dir: `{extract_dir}`")
    lines.append(f"- Output dir: `{output_dir}`")
    lines.append(f"- Selection time (UTC): `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("")

    if not os.path.isdir(extract_dir):
        lines.append("## Result")
        lines.append("- Extraction directory not found. No negatives selected.")
        save_markdown(report_path, lines)
        print(f"Wrote {report_path}")
        return

    effective_dir = extract_dir
    root_files = list_files(extract_dir)
    root_images = [p for p in root_files if is_image_file(p)]
    if len(root_images) == 0:
        try:
            subdirs = [
                os.path.join(extract_dir, d)
                for d in os.listdir(extract_dir)
                if os.path.isdir(os.path.join(extract_dir, d))
            ]
        except OSError:
            subdirs = []
        if len(subdirs) == 1:
            candidate_files = list_files(subdirs[0])
            if any(is_image_file(p) for p in candidate_files):
                effective_dir = subdirs[0]

    image_files = [p for p in list_files(effective_dir) if is_image_file(p)]
    p1: List[Tuple[str, str]] = []
    p2: List[Tuple[str, str]] = []
    p3: List[Tuple[str, str]] = []
    others: List[Tuple[str, str]] = []
    corrupted = 0
    duplicate_skipped = 0
    copied = 0  # newly added in this run
    selected_bucket_counts = Counter()

    by_sha_selected = set()
    found_stats = Counter()

    for ap in image_files:
        rp = relpath(ap, effective_dir)
        priority, is_ar, is_forge, is_bona = classify_priority(rp)
        if is_ar:
            found_stats["arabic_total"] += 1
        if is_ar and is_forge:
            found_stats["arabic_forged"] += 1
        if is_ar and is_bona:
            found_stats["arabic_bonafide"] += 1
        if priority == "priority_1_arabic_forged":
            p1.append((ap, rp))
        elif priority == "priority_2_arabic_bonafide":
            p2.append((ap, rp))
        elif priority == "priority_3_nonarabic_forged":
            p3.append((ap, rp))
        else:
            others.append((ap, rp))

    existing_unique_count = 0
    existing_duplicate_files = 0
    existing_corrupted_files = 0

    # Keep existing negatives but remove exact duplicates by hash.
    existing_files = [p for p in list_files(output_dir) if is_image_file(p)]
    for ep in existing_files:
        if not is_readable_image(ep):
            existing_corrupted_files += 1
            continue
        sha = sha256_file(ep)
        if sha in by_sha_selected:
            existing_duplicate_files += 1
            try:
                os.remove(ep)
            except OSError:
                pass
            continue
        by_sha_selected.add(sha)
        existing_unique_count += 1

    needed_to_add = max(0, target_total - existing_unique_count)

    def maybe_copy(ap: str, rp: str, bucket: str) -> None:
        nonlocal corrupted, duplicate_skipped, copied
        if not is_readable_image(ap):
            corrupted += 1
            return
        sha = sha256_file(ap)
        if sha in by_sha_selected:
            duplicate_skipped += 1
            return
        by_sha_selected.add(sha)
        stem = os.path.splitext(os.path.basename(rp))[0]
        out_name = slugify_filename(f"{bucket}_{stem}_{sha[:10]}.jpg")
        out_path = os.path.join(output_dir, out_name)
        shutil.copy2(ap, out_path)
        copied += 1
        selected_bucket_counts[bucket] += 1

    if needed_to_add > 0:
        for ap, rp in p1:
            if copied >= needed_to_add:
                break
            maybe_copy(ap, rp, "priority_1_arabic_forged")
        for ap, rp in p2:
            if copied >= needed_to_add:
                break
            maybe_copy(ap, rp, "priority_2_arabic_bonafide")
        for ap, rp in p3:
            if copied >= needed_to_add:
                break
            maybe_copy(ap, rp, "priority_3_nonarabic_forged")

    final_total = sum(1 for p in list_files(output_dir) if is_image_file(p))

    lines.append("## Discovery")
    lines.append(f"- Effective content root: `{effective_dir}`")
    lines.append(f"- Total FantasyID images discovered: **{len(image_files)}**")
    lines.append(f"- Arabic images found: **{found_stats['arabic_total']}**")
    lines.append(f"- Arabic forged/manipulated images found: **{found_stats['arabic_forged']}**")
    lines.append(f"- Arabic bonafide fantasy images found: **{found_stats['arabic_bonafide']}**")
    lines.append(f"- Non-Arabic forged fallback candidates: **{len(p3)}**")
    lines.append("")
    lines.append("## Selection")
    lines.append(f"- Existing unique negatives before update: **{existing_unique_count}**")
    lines.append(f"- Existing duplicate files removed: **{existing_duplicate_files}**")
    lines.append(f"- Existing corrupted files observed (kept): **{existing_corrupted_files}**")
    lines.append(f"- Target total negatives: **{target_total}**")
    lines.append(f"- Needed to add in this run: **{needed_to_add}**")
    lines.append(f"- Number copied in this run: **{copied}**")
    lines.append(f"- Final negatives in output_dir: **{final_total}**")
    lines.append(
        f"- Added Arabic forged/manipulated: **{selected_bucket_counts['priority_1_arabic_forged']}**"
    )
    lines.append(
        f"- Added Arabic bonafide: **{selected_bucket_counts['priority_2_arabic_bonafide']}**"
    )
    lines.append(
        f"- Added non-Arabic forged fallback: **{selected_bucket_counts['priority_3_nonarabic_forged']}**"
    )
    lines.append(f"- Duplicates skipped: **{duplicate_skipped}**")
    lines.append(f"- Corrupted skipped: **{corrupted}**")
    lines.append("")
    enough = final_total >= max(700, int(0.9 * target_total))
    lines.append("## MVP1 Negative Sufficiency")
    lines.append(f"- FantasyID enough for MVP1 negatives: **{'yes' if enough else 'no'}**")
    if not enough:
        lines.append("- Reason: selected count is too low for robust negative coverage.")
    if len(image_files) == 0:
        lines.append("- Reason: no extracted FantasyID images were found.")

    save_markdown(report_path, lines)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
