import argparse
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from modules.report import build_preview_grid, save_markdown
from modules.utils import is_image_file, list_files, relpath, sha256_file, write_csv


def compute_dhash(gray_array: np.ndarray) -> int:
    diff = gray_array[:, 1:] > gray_array[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(bool(b))
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def summarize(values: List[int]) -> str:
    if not values:
        return "n/a"
    v = sorted(values)
    return f"min={v[0]}, median={int(statistics.median(v))}, max={v[-1]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit prepared validator dataset.")
    parser.add_argument(
        "--prepared-root",
        default=r"D:\Projects\KYC\kyc-validator-training\prepared_validator_dataset",
    )
    parser.add_argument(
        "--report-path",
        default=r"D:\Projects\KYC\kyc-validator-training\reports\prepared_validator_audit.md",
    )
    parser.add_argument(
        "--distribution-csv",
        default=r"D:\Projects\KYC\kyc-validator-training\reports\class_distribution.csv",
    )
    parser.add_argument(
        "--sample-grid-dir",
        default=r"D:\Projects\KYC\kyc-validator-training\outputs\sample_grids",
    )
    parser.add_argument(
        "--source-mapping",
        default=r"D:\Projects\KYC\kyc-validator-training\reports\source_mapping.csv",
    )
    args = parser.parse_args()

    prepared_root = os.path.abspath(args.prepared_root)
    report_path = os.path.abspath(args.report_path)
    distribution_csv = os.path.abspath(args.distribution_csv)
    sample_grid_dir = os.path.abspath(args.sample_grid_dir)
    source_mapping = os.path.abspath(args.source_mapping)
    os.makedirs(sample_grid_dir, exist_ok=True)

    class_dirs = {
        "national_id": os.path.join(prepared_root, "national_id"),
        "not_national_id": os.path.join(prepared_root, "not_national_id"),
    }
    class_images: Dict[str, List[str]] = {"national_id": [], "not_national_id": []}
    for cls, d in class_dirs.items():
        if os.path.isdir(d):
            class_images[cls] = [p for p in list_files(d) if is_image_file(p)]

    corrupted = []
    sha_map = defaultdict(list)
    path_sha: Dict[str, str] = {}
    dhash_index = defaultdict(list)
    widths: List[int] = []
    heights: List[int] = []
    paths_all: List[str] = []
    class_of_path: Dict[str, str] = {}
    back_contamination_name_hits = 0

    for cls, paths in class_images.items():
        for p in paths:
            rp = relpath(p, prepared_root)
            if "back" in rp.lower():
                back_contamination_name_hits += 1
            paths_all.append(p)
            class_of_path[p] = cls
            try:
                digest = sha256_file(p)
                sha_map[digest].append(p)
                path_sha[p] = digest
                with Image.open(p) as im:
                    im = ImageOps.exif_transpose(im)
                    w, h = im.size
                    widths.append(w)
                    heights.append(h)
                    gray = im.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
                    arr = np.asarray(gray, dtype=np.float32)
                    d = compute_dhash(arr)
                    dhash_index[d].append(p)
            except (UnidentifiedImageError, OSError, ValueError, RuntimeError):
                corrupted.append(rp)

    exact_dup_groups = [g for g in sha_map.values() if len(g) > 1]
    exact_dup_count = sum(len(g) for g in exact_dup_groups)

    # near duplicates by hash buckets
    buckets = defaultdict(list)
    path_to_hash = {}
    for d, plist in dhash_index.items():
        for p in plist:
            path_to_hash[p] = d
            for band in range(4):
                key = (band, (d >> (16 * band)) & 0xFFFF)
                buckets[key].append(p)
    candidate_pairs = set()
    for plist in buckets.values():
        if len(plist) < 2 or len(plist) > 2000:
            continue
        plist_sorted = sorted(set(plist))
        for a, b in combinations(plist_sorted, 2):
            candidate_pairs.add((a, b))
    near_dup_pairs = []
    for a, b in candidate_pairs:
        d = hamming(path_to_hash[a], path_to_hash[b])
        if d <= 6 and path_sha.get(a) != path_sha.get(b):
            near_dup_pairs.append((a, b, d))

    cross_class_exact = []
    for g in exact_dup_groups:
        cls = {class_of_path[p] for p in g}
        if len(cls) > 1:
            cross_class_exact.append(g)

    cross_class_near = []
    for a, b, d in near_dup_pairs:
        if class_of_path[a] != class_of_path[b]:
            cross_class_near.append((a, b, d))

    rows = [
        {
            "class_name": "national_id",
            "count": len(class_images["national_id"]),
        },
        {
            "class_name": "not_national_id",
            "count": len(class_images["not_national_id"]),
        },
    ]
    write_csv(distribution_csv, rows, fieldnames=["class_name", "count"])

    nat = len(class_images["national_id"])
    neg = len(class_images["not_national_id"])
    imbalance = (max(nat, neg) / min(nat, neg)) if min(nat, neg) > 0 else 9999.0
    ready_to_train = (
        nat >= 500
        and neg >= 500
        and len(corrupted) == 0
        and imbalance <= 4.0
    )

    missing: List[str] = []
    if nat < 500:
        missing.append("more national_id positives")
    if neg < 500:
        missing.append("more negatives")
        missing.append("manual negatives")
    if len(corrupted) > 0:
        missing.append("corrupted cleanup")
    if len(cross_class_exact) > 0 or len(cross_class_near) > 0:
        missing.append("duplicates cleanup")
    if back_contamination_name_hits > 0:
        missing.append("wrong samples")

    national_grid = build_preview_grid(
        class_images["national_id"][:16],
        os.path.join(sample_grid_dir, "national_id_grid.png"),
    )
    negative_grid = build_preview_grid(
        class_images["not_national_id"][:16],
        os.path.join(sample_grid_dir, "not_national_id_grid.png"),
    )

    # Optional check from source mapping.
    back_from_mapping = 0
    if os.path.isfile(source_mapping):
        try:
            import csv

            with open(source_mapping, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cname = (row.get("class_name") or "").strip().lower()
                    if cname == "back_egy_id":
                        back_from_mapping += 1
        except Exception:
            pass

    lines = [
        "# Prepared Validator Dataset Audit",
        "",
        f"- Prepared root: `{prepared_root}`",
        f"- Audit time (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Counts",
        f"- national_id count: **{nat}**",
        f"- not_national_id count: **{neg}**",
        f"- class imbalance ratio (max/min): **{imbalance:.2f}x**" if min(nat, neg) > 0 else "- class imbalance ratio: undefined",
        "",
        "## Quality",
        f"- Corrupted/unreadable images: **{len(corrupted)}**",
        f"- Exact duplicate groups: **{len(exact_dup_groups)}**",
        f"- Exact duplicate file count: **{exact_dup_count}**",
        f"- Near duplicate pairs (dHash <= 6): **{len(near_dup_pairs)}**",
        f"- Cross-class exact duplicates: **{len(cross_class_exact)}**",
        f"- Cross-class near duplicates: **{len(cross_class_near)}**",
        f"- Width summary: **{summarize(widths)}**",
        f"- Height summary: **{summarize(heights)}**",
        "",
        "## Leakage and Contamination Checks",
        f"- Possible leakage (cross-class exact/near): **{len(cross_class_exact) + len(cross_class_near)}**",
        f"- Filename-based back_egy_id contamination hints: **{back_contamination_name_hits}**",
        f"- Source-mapping back_egy_id rows: **{back_from_mapping}**",
        "",
        "## Sample Grids",
        f"- national_id grid: `{national_grid or 'not generated'}`",
        f"- not_national_id grid: `{negative_grid or 'not generated'}`",
        "",
        "## Final Verdict",
        f"- READY_TO_TRAIN: **{'yes' if ready_to_train else 'no'}**",
    ]
    if not ready_to_train:
        lines.append("- Missing before training:")
        for m in sorted(set(missing)):
            lines.append(f"  - {m}")
    save_markdown(report_path, lines)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
