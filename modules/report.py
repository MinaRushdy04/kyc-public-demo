import math
import os
from typing import List, Optional

from PIL import Image, ImageOps

from .utils import ensure_dir


def save_markdown(path: str, lines: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_preview_grid(
    image_paths: List[str],
    out_path: str,
    max_images: int = 16,
    tile_size: int = 224,
) -> Optional[str]:
    if not image_paths:
        return None

    ensure_dir(os.path.dirname(out_path))
    chosen = image_paths[:max_images]
    cols = min(4, max(1, len(chosen)))
    rows = int(math.ceil(len(chosen) / cols))
    margin = 10
    canvas_w = margin + cols * (tile_size + margin)
    canvas_h = margin + rows * (tile_size + margin)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (243, 245, 248))

    for i, path in enumerate(chosen):
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                thumb = ImageOps.fit(
                    im,
                    (tile_size, tile_size),
                    method=Image.Resampling.BICUBIC,
                )
        except Exception:
            thumb = Image.new("RGB", (tile_size, tile_size), (160, 60, 60))
        r = i // cols
        c = i % cols
        x = margin + c * (tile_size + margin)
        y = margin + r * (tile_size + margin)
        canvas.paste(thumb, (x, y))

    canvas.save(out_path, format="PNG")
    return out_path

