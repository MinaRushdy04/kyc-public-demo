import csv
import hashlib
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from modules.utils import simple_yaml_load


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_image(path: Path) -> tuple[bool, tuple[int, int] | None]:
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            return True, im.size
    except (UnidentifiedImageError, OSError, ValueError, RuntimeError):
        return False, None


def main() -> None:
    cfg = simple_yaml_load("config_mvp2.yaml")
    root = Path(str(cfg["raw_dataset_path"]))
    reports = Path(str(cfg["project_dir"])) / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    people = []
    rows = []
    corrupted = []
    hashes = {}
    formats = {}
    for person_dir in sorted((root / "files").iterdir() if (root / "files").exists() else []):
        if not person_dir.is_dir():
            continue
        files = [p for p in sorted(person_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        id_files = [p for p in files if p.name.lower().startswith("id_")]
        selfie_files = [p for p in files if p.name.lower().startswith("selfie_")]
        people.append((person_dir.name, len(id_files), len(selfie_files)))
        for p in files:
            ok, size = inspect_image(p)
            formats[p.suffix.lower()] = formats.get(p.suffix.lower(), 0) + 1
            digest = ""
            if ok:
                digest = sha256(p)
                hashes.setdefault(digest, []).append(str(p))
            else:
                corrupted.append(str(p))
            rows.append(
                {
                    "person_id": person_dir.name,
                    "kind": "id" if p in id_files else "selfie" if p in selfie_files else "other",
                    "path": str(p),
                    "format": p.suffix.lower(),
                    "readable": ok,
                    "width": size[0] if size else "",
                    "height": size[1] if size else "",
                    "sha256": digest,
                }
            )

    inventory_csv = reports / "mvp2_dataset_inventory.csv"
    with inventory_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["person_id", "kind", "path", "format", "readable", "width", "height", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    total_ids = sum(p[1] for p in people)
    total_selfies = sum(p[2] for p in people)
    lines = [
        "# MVP2 Selfie-ID Dataset Audit",
        "",
        f"- Source repo: `{cfg['dataset_repo']}`",
        f"- Source URL: {cfg['dataset_url']}",
        f"- License: `{cfg['dataset_license']}`",
        f"- Local path: `{root}`",
        f"- People/identities: **{len(people)}**",
        f"- ID images: **{total_ids}**",
        f"- Selfie images: **{total_selfies}**",
        f"- Total image-like files: **{len(rows)}**",
        f"- Corrupted/unreadable images: **{len(corrupted)}**",
        f"- Exact duplicate groups: **{len(duplicate_groups)}**",
        "",
        "## Format Counts",
        "",
    ]
    for ext, count in sorted(formats.items()):
        lines.append(f"- `{ext}`: {count}")
    lines += [
        "",
        "## Per-Person Counts",
        "",
        "| person_id | id_images | selfie_images |",
        "|---|---:|---:|",
    ]
    for person_id, id_count, selfie_count in people:
        lines.append(f"| {person_id} | {id_count} | {selfie_count} |")
    lines += [
        "",
        "## Business Notes",
        "",
        "- This public sample is useful to validate the MVP2 pipeline shape.",
        "- It is not enough to train or production-certify a biometric matcher.",
        "- The license is non-commercial/no-derivatives, so use it for internal prototyping only unless legal approves broader use.",
    ]
    (reports / "mvp2_dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {reports / 'mvp2_dataset_audit.md'}")
    print(f"Wrote {inventory_csv}")


if __name__ == "__main__":
    main()
