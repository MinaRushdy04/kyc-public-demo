from typing import Optional, Tuple

from PIL import Image, ImageOps


def exif_open_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    return im


def yolo_to_xyxy(
    width: int,
    height: int,
    x_center: float,
    y_center: float,
    box_w: float,
    box_h: float,
) -> Tuple[int, int, int, int]:
    x_c = x_center * width
    y_c = y_center * height
    bw = box_w * width
    bh = box_h * height
    x1 = int(max(0, round(x_c - bw / 2)))
    y1 = int(max(0, round(y_c - bh / 2)))
    x2 = int(min(width, round(x_c + bw / 2)))
    y2 = int(min(height, round(y_c + bh / 2)))
    return x1, y1, x2, y2


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
    pad_ratio: float = 0.02,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(round(bw * pad_ratio))
    pad_y = int(round(bh * pad_ratio))
    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(width, x2 + pad_x)
    ny2 = min(height, y2 + pad_y)
    return nx1, ny1, nx2, ny2


def crop_bbox(im: Image.Image, bbox: Tuple[int, int, int, int]) -> Optional[Image.Image]:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    crop = im.crop((x1, y1, x2, y2))
    if crop.size[0] < 8 or crop.size[1] < 8:
        return None
    return crop

