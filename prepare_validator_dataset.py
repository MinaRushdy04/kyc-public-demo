import argparse
import hashlib
import io
import os
from datetime import datetime, timezone
from typing import Dict, List

from PIL import Image

from modules.dataset import (
    detect_yolo_structure,
    find_class_index,
    find_image_for_label,
    iter_yolo_label_rows,
)
from modules.preprocessing import crop_bbox, expand_bbox, exif_open_rgb, yolo_to_xyxy
from modules.utils import ensure_dir, relpath, write_csv, write_text


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MVP1 validator dataset from Egyptian YOLO dataset.")
    parser.add_argument("--dataset-root", default=r"D:\Datasets")
    parser.add_argument(
        "--national-out",
        default=r"D:\Projects\KYC\kyc-validator-training\prepared_validator_dataset\national_id",
    )
    parser.add_argument(
        "--mapping-csv",
        default=r"D:\Projects\KYC\kyc-validator-training\reports\source_mapping.csv",
    )
    parser.add_argument(
        "--summary-path",
        default=r"D:\Projects\KYC\kyc-validator-training\reports\prepare_validator_summary.md",
    )
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    national_out = os.path.abspath(args.national_out)
    mapping_csv = os.path.abspath(args.mapping_csv)
    summary_path = os.path.abspath(args.summary_path)

    ensure_dir(national_out)
    ensure_dir(os.path.dirname(mapping_csv))

    yolo = detect_yolo_structure(dataset_root)
    if yolo is None:
        write_text(
            summary_path,
            "# Prepare Validator Summary\n\n- YOLO dataset structure not found.\n",
        )
        print("YOLO structure not found.")
        return

    front_idx = find_class_index(yolo.names, "front_egy_id")
    back_idx = find_class_index(yolo.names, "back_egy_id")
    if front_idx is None:
        write_text(
            summary_path,
            "# Prepare Validator Summary\n\n- Could not find class 'front_egy_id' in data.yaml names.\n",
        )
        print("front_egy_id class not found.")
        return

    rows: List[Dict] = []
    crop_hashes = set()
    saved_count = 0
    skipped_duplicate = 0
    ignored_back = 0
    label_files_total = 0
    front_boxes_total = 0

    for split in ("train", "valid", "test"):
        if split not in yolo.split_label_dirs:
            continue
        label_dir = yolo.split_label_dirs[split]
        image_dir = yolo.split_image_dirs[split]
        for fn in os.listdir(label_dir):
            if not fn.lower().endswith(".txt"):
                continue
            label_files_total += 1
            label_path = os.path.join(label_dir, fn)
            image_path = find_image_for_label(label_path, image_dir)
            if not image_path:
                rows.append(
                    {
                        "split": split,
                        "source_label": relpath(label_path, dataset_root),
                        "source_image": "",
                        "class_id": "",
                        "class_name": "",
                        "object_index": "",
                        "bbox_x1": "",
                        "bbox_y1": "",
                        "bbox_x2": "",
                        "bbox_y2": "",
                        "output_path": "",
                        "output_sha256": "",
                        "status": "missing_image_for_label",
                    }
                )
                continue

            try:
                im = exif_open_rgb(image_path)
            except Exception:
                rows.append(
                    {
                        "split": split,
                        "source_label": relpath(label_path, dataset_root),
                        "source_image": relpath(image_path, dataset_root),
                        "class_id": "",
                        "class_name": "",
                        "object_index": "",
                        "bbox_x1": "",
                        "bbox_y1": "",
                        "bbox_x2": "",
                        "bbox_y2": "",
                        "output_path": "",
                        "output_sha256": "",
                        "status": "image_read_error",
                    }
                )
                continue

            width, height = im.size
            for obj_idx, row in enumerate(iter_yolo_label_rows(label_path)):
                class_id, x, y, w, h = row
                if back_idx is not None and class_id == back_idx:
                    ignored_back += 1
                    continue
                if class_id != front_idx:
                    continue
                front_boxes_total += 1
                bbox = yolo_to_xyxy(width, height, x, y, w, h)
                bbox = expand_bbox(bbox, width, height, pad_ratio=0.02)
                crop = crop_bbox(im, bbox)
                if crop is None:
                    rows.append(
                        {
                            "split": split,
                            "source_label": relpath(label_path, dataset_root),
                            "source_image": relpath(image_path, dataset_root),
                            "class_id": class_id,
                            "class_name": "front_egy_id",
                            "object_index": obj_idx,
                            "bbox_x1": bbox[0],
                            "bbox_y1": bbox[1],
                            "bbox_x2": bbox[2],
                            "bbox_y2": bbox[3],
                            "output_path": "",
                            "output_sha256": "",
                            "status": "crop_invalid",
                        }
                    )
                    continue

                buf = io.BytesIO()
                crop.save(buf, format="JPEG", quality=95)
                img_bytes = buf.getvalue()
                crop_sha = sha256_bytes(img_bytes)
                if crop_sha in crop_hashes:
                    skipped_duplicate += 1
                    rows.append(
                        {
                            "split": split,
                            "source_label": relpath(label_path, dataset_root),
                            "source_image": relpath(image_path, dataset_root),
                            "class_id": class_id,
                            "class_name": "front_egy_id",
                            "object_index": obj_idx,
                            "bbox_x1": bbox[0],
                            "bbox_y1": bbox[1],
                            "bbox_x2": bbox[2],
                            "bbox_y2": bbox[3],
                            "output_path": "",
                            "output_sha256": crop_sha,
                            "status": "duplicate_crop_skipped",
                        }
                    )
                    continue
                crop_hashes.add(crop_sha)

                image_stem = os.path.splitext(os.path.basename(image_path))[0]
                out_name = f"{split}_{image_stem}_front_{obj_idx}_{crop_sha[:10]}.jpg"
                out_path = os.path.join(national_out, out_name)
                with open(out_path, "wb") as f:
                    f.write(img_bytes)
                saved_count += 1

                rows.append(
                    {
                        "split": split,
                        "source_label": relpath(label_path, dataset_root),
                        "source_image": relpath(image_path, dataset_root),
                        "class_id": class_id,
                        "class_name": "front_egy_id",
                        "object_index": obj_idx,
                        "bbox_x1": bbox[0],
                        "bbox_y1": bbox[1],
                        "bbox_x2": bbox[2],
                        "bbox_y2": bbox[3],
                        "output_path": out_path,
                        "output_sha256": crop_sha,
                        "status": "saved",
                    }
                )

    write_csv(
        mapping_csv,
        rows,
        fieldnames=[
            "split",
            "source_label",
            "source_image",
            "class_id",
            "class_name",
            "object_index",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "output_path",
            "output_sha256",
            "status",
        ],
    )

    summary = [
        "# Prepare Validator Summary",
        "",
        f"- Time (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        f"- Dataset root: `{dataset_root}`",
        f"- data.yaml: `{yolo.data_yaml}`",
        f"- front_egy_id class index: **{front_idx}**",
        f"- back_egy_id class index: **{back_idx if back_idx is not None else 'not found'}**",
        f"- Label files scanned: **{label_files_total}**",
        f"- front_egy_id boxes seen: **{front_boxes_total}**",
        f"- back_egy_id boxes ignored: **{ignored_back}**",
        f"- Saved national_id crops: **{saved_count}**",
        f"- Duplicate crops skipped: **{skipped_duplicate}**",
        f"- Source mapping CSV: `{mapping_csv}`",
    ]
    write_text(summary_path, "\n".join(summary) + "\n")
    print(f"Saved national_id crops: {saved_count}")


if __name__ == "__main__":
    main()

