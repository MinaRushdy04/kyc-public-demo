import csv
import random
from pathlib import Path

from modules.utils import simple_yaml_load


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main() -> None:
    cfg = simple_yaml_load("config_mvp2.yaml")
    root = Path(str(cfg["raw_dataset_path"])) / "files"
    out_csv = Path(str(cfg["prepared_pairs_csv"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    identities = {}
    for person_dir in sorted(root.iterdir() if root.exists() else []):
        if not person_dir.is_dir():
            continue
        id_files = [p for p in sorted(person_dir.iterdir()) if p.name.lower().startswith("id_") and p.suffix.lower() in IMAGE_EXTS]
        selfies = [
            p
            for p in sorted(person_dir.iterdir())
            if p.name.lower().startswith("selfie_") and p.suffix.lower() in IMAGE_EXTS
        ]
        if id_files and selfies:
            identities[person_dir.name] = {"ids": id_files, "selfies": selfies}

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

    person_ids = list(identities.keys())
    positive_count = len(rows)
    for i in range(positive_count):
        id_person = rng.choice(person_ids)
        selfie_person = rng.choice([p for p in person_ids if p != id_person])
        id_path = rng.choice(identities[id_person]["ids"])
        selfie_path = rng.choice(identities[selfie_person]["selfies"])
        rows.append(
            {
                "pair_id": f"neg_{i:06d}",
                "id_person": id_person,
                "selfie_person": selfie_person,
                "id_image": str(id_path),
                "selfie_image": str(selfie_path),
                "label": 0,
                "pair_type": "different_person",
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pair_id", "id_person", "selfie_person", "id_image", "selfie_image", "label", "pair_type"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Prepared pair CSV: {out_csv}")
    print(f"Identities: {len(identities)}")
    print(f"Positive pairs: {positive_count}")
    print(f"Negative pairs: {len(rows) - positive_count}")


if __name__ == "__main__":
    main()
