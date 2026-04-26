import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set


IMAGE_EXTS: Set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".jfif",
}

ARCHIVE_EXTS: Sequence[str] = (".tgz", ".tar.gz", ".tar")


@dataclass
class ScanSummary:
    total_files: int = 0
    total_images: int = 0
    corrupted_images: int = 0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def list_files(root: str) -> List[str]:
    out: List[str] = []
    for r, _, files in os.walk(root):
        for f in files:
            out.append(os.path.join(r, f))
    return out


def find_archives(root: str) -> List[str]:
    found: List[str] = []
    for p in list_files(root):
        p_low = p.lower()
        if any(p_low.endswith(ext) for ext in ARCHIVE_EXTS):
            found.append(p)
    return sorted(found)


def sha256_file(path: str, block_size: int = 64 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            try:
                chunk = f.read(block_size)
            except MemoryError:
                # Fallback to a smaller read size in constrained environments.
                block_size = 8 * 1024
                chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str, payload: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_json(path: str) -> Optional[Dict]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_csv(path: str, rows: List[Dict], fieldnames: Sequence[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def slugify_filename(name: str, max_len: int = 120) -> str:
    stem, ext = os.path.splitext(name)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    if not stem:
        stem = "file"
    stem = stem[:max_len]
    return stem + ext.lower()


def relpath(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace("\\", "/")


def infer_split_from_path(path: str) -> str:
    p = path.lower().replace("\\", "/")
    if "/train/" in p:
        return "train"
    if "/valid/" in p or "/val/" in p:
        return "valid"
    if "/test/" in p:
        return "test"
    return "unspecified"


ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06ff]")


def contains_arabic(text: str) -> bool:
    return bool(ARABIC_CHAR_RE.search(text or ""))


def keyword_hit(text: str, keywords: Sequence[str]) -> bool:
    t = (text or "").lower()
    return any(k in t for k in keywords)


def simple_yaml_load(path: str) -> Dict:
    """
    Minimal parser for the config/data yaml used in this project.
    Supports:
    - key: value
    - key: [a, b, c]
    """
    data: Dict = {}
    if not os.path.isfile(path):
        return data
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = k.strip()
        value = v.strip()
        if value.startswith("[") and value.endswith("]"):
            items = [x.strip().strip("'\"") for x in value[1:-1].split(",") if x.strip()]
            data[key] = items
        elif value.lower() in {"true", "false"}:
            data[key] = value.lower() == "true"
        else:
            value = value.strip().strip("'\"")
            if re.fullmatch(r"-?\d+", value):
                data[key] = int(value)
            elif re.fullmatch(r"-?\d+\.\d+", value):
                data[key] = float(value)
            else:
                data[key] = value
    return data
