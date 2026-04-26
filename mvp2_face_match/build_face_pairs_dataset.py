from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from modules.utils import ensure_dir, load_config, project_relative_default_config


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _images_in(path: Path) -> list[Path]:
    return [p for p in sorted(path.iterdir() if path.exists() else []) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def load_calibration_identities(root: Path) -> dict[str, dict[str, list[Path]]]:
    identities_root = root / "identities"
    identities = {}
    for person_dir in sorted(identities_root.iterdir() if identities_root.exists() else []):
        if not person_dir.is_dir():
            continue
        id_images = _images_in(person_dir / "id")
        selfie_images = _images_in(person_dir / "selfies")
        if id_images and selfie_images:
            identities[person_dir.name] = {"ids": id_images, "selfies": selfie_images}
    return identities


def load_legacy_sample_identities(root: Path) -> dict[str, dict[str, list[Path]]]:
    files_root = root / "files"
    identities = {}
    for person_dir in sorted(files_root.iterdir() if files_root.exists() else []):
        if not person_dir.is_dir():
            continue
        ids = [p for p in _images_in(person_dir) if p.name.lower().startswith("id_")]
        selfies = [p for p in _images_in(person_dir) if p.name.lower().startswith("selfie_")]
        if ids and selfies:
            identities[person_dir.name] = {"ids": ids, "selfies": selfies}
    return identities


def build_pairs(identities: dict[str, dict[str, list[Path]]], out_csv: Path, seed: int = 42, negatives_per_positive: float = 1.0) -> tuple[int, int, int]:
    if len(identities) < 2:
        raise SystemExit("Need at least two identities with id/ and selfies/ images to build impostor pairs.")
    rng = random.Random(seed)
    rows = []

    for person_id, data in identities.items():
        for id_path in data["ids"]:
            for selfie_path in data["selfies"]:
                rows.append(
                    {
                        "pair_id": f"pos_{len(rows):06d}",
                        "id_person": person_id,
                        "selfie_person": person_id,
                        "id_image": str(id_path),
                        "selfie_image": str(selfie_path),
                        "label": 1,
                        "pair_type": "same_person",
                    }
                )

    positive_count = len(rows)
    people = list(identities.keys())
    negative_count = max(1, int(round(positive_count * negatives_per_positive)))
    for i in range(negative_count):
        id_person = rng.choice(people)
        selfie_person = rng.choice([p for p in people if p != id_person])
        rows.append(
            {
                "pair_id": f"neg_{i:06d}",
                "id_person": id_person,
                "selfie_person": selfie_person,
                "id_image": str(rng.choice(identities[id_person]["ids"])),
                "selfie_image": str(rng.choice(identities[selfie_person]["selfies"])),
                "label": 0,
                "pair_type": "different_person",
            }
        )

    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pair_id", "id_person", "selfie_person", "id_image", "selfie_image", "label", "pair_type"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(identities), positive_count, negative_count


def create_readme(root: Path) -> None:
    ensure_dir(root / "identities")
    readme = root / "README.md"
    if readme.exists():
        return
    readme.write_text(
        """# MVP2 Face Calibration Dataset

Use this folder for consented calibration/evaluation examples only. Do not use it to train ArcFace from scratch.

Expected structure:

```text
calibration_dataset/
  identities/
    person_001/
      id/
        id_front_001.jpg
      selfies/
        selfie_001.jpg
        selfie_002.jpg
    person_002/
      id/
      selfies/
```

The pair builder creates:

- positive pairs from the same identity
- negative/impostor pairs from different identities

This is for threshold calibration and failure analysis, not production approval by itself.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MVP2 face calibration pair CSV.")
    parser.add_argument("--config", default=str(project_relative_default_config()))
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--legacy-sample", action="store_true", help="Use the old downloaded Selfie_ID_Dataset_hf/files layout.")
    parser.add_argument("--negatives-per-positive", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    calibration_root = Path(args.dataset_root or str(cfg.get("calibration_dataset_root")))
    create_readme(calibration_root)
    out_csv = Path(args.out or str(cfg["pairs_csv"]))

    if args.legacy_sample:
        identities = load_legacy_sample_identities(Path(str(cfg["prototype_dataset_root"])))
        source = str(cfg["prototype_dataset_root"])
    else:
        identities = load_calibration_identities(calibration_root)
        source = str(calibration_root)

    identities_count, positives, negatives = build_pairs(
        identities,
        out_csv,
        negatives_per_positive=float(args.negatives_per_positive),
    )
    print(f"Wrote {out_csv}")
    print(f"Source: {source}")
    print("Suitable for prototype testing and threshold calibration only, not production calibration unless data is representative and consented.")
    print(f"Identities: {identities_count}")
    print(f"Positive pairs: {positives}")
    print(f"Negative pairs: {negatives}")


if __name__ == "__main__":
    main()
