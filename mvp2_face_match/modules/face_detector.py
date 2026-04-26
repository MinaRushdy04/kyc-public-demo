from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

cv2.setNumThreads(1)


@dataclass
class FaceDetection:
    bbox: np.ndarray
    landmarks: np.ndarray
    confidence: float
    image_shape: tuple[int, int]

    @property
    def width(self) -> float:
        return float(max(0.0, self.bbox[2] - self.bbox[0]))

    @property
    def height(self) -> float:
        return float(max(0.0, self.bbox[3] - self.bbox[1]))

    @property
    def area(self) -> float:
        return self.width * self.height


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-12)
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    return keep


class SCRFDFaceDetector:
    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 640,
        confidence_threshold: float = 0.45,
        nms_threshold: float = 0.4,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"SCRFD model not found: {self.model_path}")
        self.input_size = int(input_size)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.strides = [8, 16, 32]
        self.num_anchors = 2
        self._anchor_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def _preprocess(self, bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        h, w = bgr.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...].astype(np.float32, copy=False)
        return blob.astype(np.float32), scale, (h, w)

    def _anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        if key in self._anchor_cache:
            return self._anchor_cache[key]
        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        if self.num_anchors > 1:
            anchor_centers = np.stack([anchor_centers] * self.num_anchors, axis=1).reshape((-1, 2))
        self._anchor_cache[key] = anchor_centers
        return anchor_centers

    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        blob, scale, original_shape = self._preprocess(image_bgr)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(self.strides):
            scores = outputs[idx].reshape(-1)
            bbox_preds = outputs[idx + 3] * stride
            kps_preds = outputs[idx + 6] * stride
            height = self.input_size // stride
            width = self.input_size // stride
            anchor_centers = self._anchor_centers(height, width, stride)
            pos_inds = np.where(scores >= self.confidence_threshold)[0]
            if len(pos_inds) == 0:
                continue
            bboxes = _distance2bbox(anchor_centers, bbox_preds).reshape(-1, 4)
            kpss = _distance2kps(anchor_centers, kps_preds).reshape(-1, 5, 2)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            kpss_list.append(kpss[pos_inds])

        if not scores_list:
            return []
        scores = np.concatenate(scores_list)
        bboxes = np.concatenate(bboxes_list) / scale
        kpss = np.concatenate(kpss_list) / scale
        h, w = original_shape
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, w - 1)
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, h - 1)
        kpss[:, :, 0] = np.clip(kpss[:, :, 0], 0, w - 1)
        kpss[:, :, 1] = np.clip(kpss[:, :, 1], 0, h - 1)
        keep = _nms(bboxes, scores, self.nms_threshold)
        faces = [
            FaceDetection(
                bbox=bboxes[i].astype(np.float32),
                landmarks=kpss[i].astype(np.float32),
                confidence=float(scores[i]),
                image_shape=(h, w),
            )
            for i in keep
        ]
        faces.sort(key=lambda f: (f.confidence, f.area), reverse=True)
        return faces


def read_bgr(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Invalid or unreadable image: {path}")
    return img
