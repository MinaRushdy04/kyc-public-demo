from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class ArcFaceEmbedder:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ArcFace model not found: {self.model_path}")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_shape = list(input_meta.shape)
        self.output_name = self.session.get_outputs()[0].name

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        arr = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        arr = (arr - 127.5) / 128.0
        if len(self.input_shape) == 4 and self.input_shape[1] == 3:
            arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
        else:
            arr = arr[np.newaxis, ...]
        emb = self.session.run([self.output_name], {self.input_name: arr})[0][0].astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm <= 1e-12:
            return emb
        return emb / norm
