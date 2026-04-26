import argparse
import os
import tarfile
from datetime import datetime, timezone
from typing import List

from modules.utils import find_archives, is_image_file, list_files, read_json, write_json


def safe_extract(tar: tarfile.TarFile, target_dir: str) -> None:
    target_dir_abs = os.path.abspath(target_dir)
    for member in tar.getmembers():
        member_path = os.path.abspath(os.path.join(target_dir, member.name))
        if not member_path.startswith(target_dir_abs + os.sep) and member_path != target_dir_abs:
            raise RuntimeError(f"Blocked path traversal in archive member: {member.name}")
    tar.extractall(path=target_dir)


def count_images(root: str) -> int:
    return sum(1 for p in list_files(root) if is_image_file(p))


def pick_fantasy_archive(archives: List[str]) -> str:
    if not archives:
        return ""
    ranked = sorted(
        archives,
        key=lambda p: (
            0 if "fantasy" in os.path.basename(p).lower() else 1,
            os.path.basename(p).lower(),
        ),
    )
    return ranked[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract FantasyID archive safely.")
    parser.add_argument("--dataset-root", default=r"D:\Datasets")
    parser.add_argument("--extract-dir", default=r"D:\Datasets\FantasyID_extracted")
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    extract_dir = os.path.abspath(args.extract_dir)
    manifest_path = os.path.join(extract_dir, "_extraction_manifest.json")

    os.makedirs(extract_dir, exist_ok=True)
    archives = find_archives(dataset_root)
    archive_path = pick_fantasy_archive(archives)

    if not archive_path:
        payload = {
            "status": "archive_not_found",
            "dataset_root": dataset_root,
            "extract_dir": extract_dir,
            "archives_found": archives,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "message": "No .tgz/.tar.gz/.tar archive found under dataset root.",
        }
        write_json(manifest_path, payload)
        print("No FantasyID archive found. Wrote manifest with archive_not_found status.")
        return

    archive_stat = os.stat(archive_path)
    existing = read_json(manifest_path) or {}
    existing_ok = (
        existing.get("status") == "ok"
        and existing.get("archive_path") == archive_path
        and int(existing.get("archive_size", -1)) == int(archive_stat.st_size)
        and int(existing.get("archive_mtime", -1)) == int(archive_stat.st_mtime)
        and int(existing.get("extracted_image_count", 0)) > 0
        and os.path.isdir(extract_dir)
    )
    if existing_ok:
        print("Extraction already valid. Skipping.")
        return

    # Clean only extraction directory contents if needed, never touch archive.
    for root, dirs, files in os.walk(extract_dir):
        for fn in files:
            os.remove(os.path.join(root, fn))
        for dn in dirs:
            full = os.path.join(root, dn)
            # defer removing dirs after walk
    for root, dirs, _ in os.walk(extract_dir, topdown=False):
        for dn in dirs:
            os.rmdir(os.path.join(root, dn))

    with tarfile.open(archive_path, "r:*") as tar:
        safe_extract(tar, extract_dir)

    image_count = count_images(extract_dir)
    payload = {
        "status": "ok" if image_count > 0 else "empty_extraction",
        "dataset_root": dataset_root,
        "archive_path": archive_path,
        "archive_size": int(archive_stat.st_size),
        "archive_mtime": int(archive_stat.st_mtime),
        "extract_dir": extract_dir,
        "extracted_image_count": int(image_count),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(manifest_path, payload)
    print(f"Extraction complete. Images found: {image_count}")


if __name__ == "__main__":
    main()

