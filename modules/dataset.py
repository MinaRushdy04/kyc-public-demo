import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .utils import simple_yaml_load


@dataclass
class YoloDatasetInfo:
    dataset_root: str
    data_yaml: str
    split_image_dirs: Dict[str, str]
    split_label_dirs: Dict[str, str]
    names: Dict[int, str]


def find_data_yaml(dataset_root: str) -> Optional[str]:
    for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
        p = os.path.join(dataset_root, name)
        if os.path.isfile(p):
            return p
    return None


def parse_names_from_data_yaml(data_yaml_path: str) -> Dict[int, str]:
    data = simple_yaml_load(data_yaml_path)
    names: Dict[int, str] = {}
    raw = data.get("names", [])
    if isinstance(raw, list):
        for i, name in enumerate(raw):
            names[i] = str(name)
    elif isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("[") and raw.endswith("]"):
            items = [x.strip().strip("'\"") for x in raw[1:-1].split(",") if x.strip()]
            for i, name in enumerate(items):
                names[i] = name
    return names


def detect_yolo_structure(dataset_root: str) -> Optional[YoloDatasetInfo]:
    data_yaml = find_data_yaml(dataset_root)
    if not data_yaml:
        return None

    split_image_dirs: Dict[str, str] = {}
    split_label_dirs: Dict[str, str] = {}
    for split in ("train", "valid", "test"):
        img_dir = os.path.join(dataset_root, split, "images")
        lbl_dir = os.path.join(dataset_root, split, "labels")
        if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
            split_image_dirs[split] = img_dir
            split_label_dirs[split] = lbl_dir

    if not split_image_dirs:
        return None

    names = parse_names_from_data_yaml(data_yaml)
    return YoloDatasetInfo(
        dataset_root=dataset_root,
        data_yaml=data_yaml,
        split_image_dirs=split_image_dirs,
        split_label_dirs=split_label_dirs,
        names=names,
    )


def find_class_index(names: Dict[int, str], class_name: str) -> Optional[int]:
    target = class_name.strip().lower()
    for idx, nm in names.items():
        if str(nm).strip().lower() == target:
            return idx
    return None


def find_image_for_label(label_path: str, image_dir: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(label_path))[0]
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif"):
        p = os.path.join(image_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def iter_yolo_label_rows(label_path: str) -> Iterable[Tuple[int, float, float, float, float]]:
    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
                vals = [float(x) for x in parts[1:5]]
            except ValueError:
                continue
            yield class_id, vals[0], vals[1], vals[2], vals[3]

