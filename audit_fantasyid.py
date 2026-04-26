import argparse
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List

from PIL import Image, ImageOps, UnidentifiedImageError

from modules.report import save_markdown
from modules.utils import (
    contains_arabic,
    is_image_file,
    keyword_hit,
    list_files,
    read_json,
    relpath,
)


ARABIC_HINTS = [
    "arabic",
    "ar_",
    "_ar",
    "ara",
    "egypt",
    "egy",
    "arab",
    "msa",
]
FORGE_HINTS = [
    "forg",
    "fake",
    "attack",
    "manip",
    "spoof",
    "tamper",
    "morph",
    "counterfeit",
]
BONAFIDE_HINTS = [
    "bonafide",
    "bona_fide",
    "genuine",
    "real",
    "authentic",
]
METADATA_EXTS = {".json", ".csv", ".txt", ".yaml", ".yml", ".xml"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit FantasyID extracted dataset.")
    parser.add_argument("--extract-dir", default=r"D:\Datasets\FantasyID_extracted")
    parser.add_argument(
        "--report-path",
        default=r"D:\Projects\KYC\kyc-validator-training\reports\fantasyid_audit.md",
    )
    args = parser.parse_args()

    extract_dir = os.path.abspath(args.extract_dir)
    report_path = os.path.abspath(args.report_path)
    manifest_path = os.path.join(extract_dir, "_extraction_manifest.json")
    manifest = read_json(manifest_path) or {}

    lines: List[str] = []
    lines.append("# FantasyID Audit")
    lines.append("")
    lines.append(f"- Extract dir: `{extract_dir}`")
    lines.append(f"- Audit time (UTC): `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("")
    lines.append("## Extraction Manifest")
    if manifest:
        for k in [
            "status",
            "archive_path",
            "extract_dir",
            "extracted_image_count",
            "timestamp_utc",
        ]:
            if k in manifest:
                lines.append(f"- {k}: `{manifest[k]}`")
    else:
        lines.append("- No extraction manifest found.")

    if not os.path.isdir(extract_dir):
        lines.append("")
        lines.append("## Result")
        lines.append("- Extraction directory does not exist.")
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

    all_files = list_files(effective_dir)
    image_files = [p for p in all_files if is_image_file(p)]
    metadata_files = [p for p in all_files if os.path.splitext(p)[1].lower() in METADATA_EXTS]

    fmt_counts = Counter()
    corrupt = []
    top_folders = Counter()
    arabic_candidates = 0
    forged_candidates = 0
    bonafide_candidates = 0
    arabic_forged_candidates = 0
    arabic_bonafide_candidates = 0

    for p in all_files:
        rp = relpath(p, effective_dir)
        top = rp.split("/", 1)[0] if "/" in rp else "__root__"
        top_folders[top] += 1

    for p in image_files:
        rp = relpath(p, effective_dir)
        low = rp.lower()
        if any(h in low for h in ARABIC_HINTS) or contains_arabic(rp):
            arabic_candidates += 1
        has_forge = keyword_hit(low, FORGE_HINTS)
        has_bona = keyword_hit(low, BONAFIDE_HINTS)
        if has_forge:
            forged_candidates += 1
        if has_bona:
            bonafide_candidates += 1
        if (any(h in low for h in ARABIC_HINTS) or contains_arabic(rp)) and has_forge:
            arabic_forged_candidates += 1
        if (any(h in low for h in ARABIC_HINTS) or contains_arabic(rp)) and has_bona:
            arabic_bonafide_candidates += 1
        try:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im)
                im.verify()
                fmt = (im.format or os.path.splitext(p)[1].lstrip(".")).upper()
                fmt_counts[fmt] += 1
        except (UnidentifiedImageError, OSError, ValueError, RuntimeError):
            corrupt.append(rp)

    lines.append("")
    lines.append("## Structure")
    lines.append(f"- Effective content root: `{effective_dir}`")
    lines.append(f"- Total files: **{len(all_files)}**")
    lines.append(f"- Total images: **{len(image_files)}**")
    lines.append(f"- Metadata/label files: **{len(metadata_files)}**")
    lines.append("- Top folders by file count:")
    for k, v in top_folders.most_common(20):
        lines.append(f"  - `{k}`: {v}")

    lines.append("")
    lines.append("## Quality")
    lines.append(
        "- Image formats: "
        + (", ".join([f"{k}:{v}" for k, v in fmt_counts.most_common()]) if fmt_counts else "none")
    )
    lines.append(f"- Corrupted/unreadable images: **{len(corrupt)}**")
    if corrupt:
        lines.append("- Sample corrupted files:")
        for rp in corrupt[:30]:
            lines.append(f"  - `{rp}`")

    lines.append("")
    lines.append("## Label / Language Indicators")
    lines.append(f"- Arabic-related samples (path/name heuristic): **{arabic_candidates}**")
    lines.append(f"- Forged/manipulated/attack samples (heuristic): **{forged_candidates}**")
    lines.append(f"- Bonafide/genuine samples (heuristic): **{bonafide_candidates}**")
    lines.append(f"- Arabic forged/manipulated/attack samples: **{arabic_forged_candidates}**")
    lines.append(f"- Arabic bonafide samples: **{arabic_bonafide_candidates}**")
    lines.append("")
    lines.append("## Notes")
    lines.append(
        "- Counts above are heuristic unless explicit metadata labels are present. The selection script applies the same priority logic for MVP1 negatives."
    )

    save_markdown(report_path, lines)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
