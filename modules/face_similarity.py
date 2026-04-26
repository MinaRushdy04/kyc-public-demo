from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageOps


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def crop_largest_face(im: Image.Image, margin: float = 0.25) -> Tuple[Image.Image, bool]:
    arr = np.array(im)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = []
    for cascade_name in [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_profileface.xml",
    ]:
        cascade_path = cv2.data.haarcascades + cascade_name
        detector = cv2.CascadeClassifier(cascade_path)
        found = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(32, 32))
        if len(found) > 0:
            faces = found
            break
    if len(faces) == 0:
        return im, False
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    pad = int(max(w, h) * margin)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(im.width, x + w + pad)
    y1 = min(im.height, y + h + pad)
    return im.crop((x0, y0, x1, y1)), True


class MobileNetFaceBaseline:
    def __init__(self, image_size: int = 224, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier = torch.nn.Identity()
        model.eval().to(self.device)
        self.model = model
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @torch.no_grad()
    def embed_image(self, im: Image.Image) -> torch.Tensor:
        x = self.transform(im).unsqueeze(0).to(self.device)
        emb = self.model(x)
        return F.normalize(emb.flatten(1), p=2, dim=1).cpu().squeeze(0)


class ArcFaceOnnxEmbedder:
    def __init__(self, model_path: str | Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def embed_image(self, im: Image.Image) -> torch.Tensor:
        arr = np.array(im.resize((112, 112), Image.Resampling.BICUBIC)).astype(np.float32)
        arr = (arr - 127.5) / 128.0
        arr = arr[np.newaxis, ...]
        emb = self.session.run([self.output_name], {self.input_name: arr})[0][0]
        emb = emb / max(np.linalg.norm(emb), 1e-12)
        return torch.from_numpy(emb.astype(np.float32))


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def score_pair(id_image: str | Path, selfie_image: str | Path, embedder: MobileNetFaceBaseline) -> dict:
    id_im = load_rgb(id_image)
    selfie_im = load_rgb(selfie_image)
    id_face, id_face_found = crop_largest_face(id_im)
    selfie_face, selfie_face_found = crop_largest_face(selfie_im)
    id_emb = embedder.embed_image(id_face)
    selfie_emb = embedder.embed_image(selfie_face)
    similarity = cosine_similarity(id_emb, selfie_emb)
    return {
        "similarity": similarity,
        "id_face_found": id_face_found,
        "selfie_face_found": selfie_face_found,
    }
