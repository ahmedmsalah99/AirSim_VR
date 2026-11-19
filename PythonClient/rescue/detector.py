"""Person detector: the YOLO11 ONNX model run over image tiles.

The image is split into overlapping tiles so small people aren't shrunk
away when each tile is resized to the model's 640x640 input. Each tile is
letterboxed, run through the model, and its boxes are mapped back to full
image pixels; boxes from all tiles are then merged with NMS.
"""
from dataclasses import dataclass
import math
import time
from pathlib import Path

import cv2
import numpy as np

from survey_plan import tile_size

LETTERBOX_FILL = 114  # the grey Ultralytics pads with


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def center(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


def tile_origins(length, tiles, overlap):
    """Start pixel of each tile along one axis, and the tile size."""
    size = tile_size(length, tiles, overlap)
    if tiles == 1:
        return [0], length
    stride = (length - size) / (tiles - 1)
    return [round(i * stride) for i in range(tiles)], math.ceil(size)


def letterbox(image, size):
    """Resize keeping aspect ratio and pad to size x size. Returns image, scale, (pad_x, pad_y)."""
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    out = np.full((size, size, 3), LETTERBOX_FILL, dtype=np.uint8)
    out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return out, scale, (pad_x, pad_y)


class PersonDetector:
    def __init__(self, det_cfg, base_dir=Path(__file__).parent):
        import onnxruntime as ort

        self.cfg = det_cfg
        model = Path(det_cfg["model"])
        if not model.is_absolute():
            model = base_dir / model
        if not model.exists():
            raise FileNotFoundError(f"detector model not found: {model}")
        self.size = det_cfg["input_size"]
        self.session, self.device, self.seconds_per_tile = pick_session(ort, model, self.size,
                                                                        det_cfg.get("device", "auto"))
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, image_bgr, classes=None):
        """Detections in full-image pixels. classes: class ids to keep (default: person only)."""
        classes = [self.cfg["person_class"]] if classes is None else classes
        h, w = image_bgr.shape[:2]
        xs, tile_w = tile_origins(w, self.cfg["tiles_x"], self.cfg["tile_overlap"])
        ys, tile_h = tile_origins(h, self.cfg["tiles_y"], self.cfg["tile_overlap"])

        found = []
        for y0 in ys:
            for x0 in xs:
                tile = image_bgr[y0:y0 + tile_h, x0:x0 + tile_w]
                found += self._detect_tile(tile, x0, y0, classes)
        return self._nms(found)

    def _detect_tile(self, tile_bgr, x0, y0, classes):
        boxed, scale, (pad_x, pad_y) = letterbox(tile_bgr, self.size)
        blob = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        output = self.session.run(None, {self.input_name: blob})[0][0]  # (4 + classes, N)
        return decode(output, self.cfg["conf_threshold"], classes, scale, pad_x, pad_y, x0, y0,
                      tile_bgr.shape[1], tile_bgr.shape[0])

    def _nms(self, detections):
        kept = []
        for cls in {d.class_id for d in detections}:
            group = [d for d in detections if d.class_id == cls]
            boxes = [[d.x1, d.y1, d.width, d.height] for d in group]
            scores = [d.confidence for d in group]
            idx = cv2.dnn.NMSBoxes(boxes, scores, self.cfg["conf_threshold"], self.cfg["iou_threshold"])
            kept += merge_fragments([group[i] for i in np.array(idx).flatten()])
        return sorted(kept, key=lambda d: -d.confidence)


def _session_options(ort, provider):
    options = ort.SessionOptions()
    if provider == "DmlExecutionProvider":  # DirectML needs these
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


def _candidates(ort, device):
    """(name, providers) to try for device: "auto", "cpu", "cuda" or "dml:<n>" (a DirectML GPU)."""
    available = ort.get_available_providers()
    cpu = [("CPU", ["CPUExecutionProvider"])]
    if device == "cpu":
        return cpu
    if device == "cuda":
        return [("CUDA", ["CUDAExecutionProvider", "CPUExecutionProvider"])]
    if device.startswith("dml:"):
        n = int(device[4:])
        return [(f"DirectML GPU {n}", [("DmlExecutionProvider", {"device_id": n}), "CPUExecutionProvider"])]
    if device != "auto":
        raise ValueError(f"detector.device must be auto, cpu, cuda or dml:<n>, not {device!r}")
    found = []
    if "CUDAExecutionProvider" in available:
        found.append(("CUDA", ["CUDAExecutionProvider", "CPUExecutionProvider"]))
    if "DmlExecutionProvider" in available:
        # Every GPU DirectML sees: on a laptop GPU 0 is usually the integrated one.
        found += [(f"DirectML GPU {n}", [("DmlExecutionProvider", {"device_id": n}), "CPUExecutionProvider"])
                  for n in range(4)]
    return found + cpu


def pick_session(ort, model, size, device="auto"):
    """The fastest working ONNX Runtime session for device (see _candidates): each candidate
    runs the model on a blank tile and the quickest wins. Returns (session, name, s per tile)."""
    blank = np.zeros((1, 3, size, size), np.float32)
    best = None
    for name, providers in _candidates(ort, device):
        first = providers[0] if isinstance(providers[0], str) else providers[0][0]
        try:
            session = ort.InferenceSession(str(model), _session_options(ort, first), providers=providers)
            if session.get_providers()[0] != first:
                continue  # fell back to the CPU
            feed = {session.get_inputs()[0].name: blank}
            session.run(None, feed)  # the first run includes one-off setup
            start = time.time()
            for _ in range(3):
                session.run(None, feed)
            seconds = (time.time() - start) / 3
        except Exception:  # noqa: BLE001 - a GPU that doesn't exist or can't run the model
            continue
        if best is None or seconds < best[2]:
            best = (session, name, seconds)
    if best is None:
        raise RuntimeError(f"no ONNX Runtime device could run {model} (detector.device: {device})")
    return best


def merge_fragments(detections, min_overlap=0.6):
    """Drop boxes lying mostly inside a bigger one.

    A person cut by a tile seam comes back as a partial box from one tile and
    a full one from the tile that contains them; NMS keeps both because their
    IoU is small. Measuring the overlap against the smaller box catches it.
    """
    kept = []
    for d in sorted(detections, key=lambda d: -d.width * d.height):
        if all(_intersection(d, k) < min_overlap * d.width * d.height for k in kept):
            kept.append(d)
        else:
            big = max((k for k in kept if _intersection(d, k) >= min_overlap * d.width * d.height),
                      key=lambda k: k.width * k.height)
            big.confidence = max(big.confidence, d.confidence)
    return kept


def _intersection(a, b):
    w = min(a.x2, b.x2) - max(a.x1, b.x1)
    h = min(a.y2, b.y2) - max(a.y1, b.y1)
    return max(w, 0) * max(h, 0)


def decode(output, conf_threshold, classes, scale, pad_x, pad_y, x0, y0, tile_w, tile_h):
    """YOLO11 raw output (4 + classes, N) for one tile -> Detections in full-image pixels."""
    scores = output[4:]
    class_ids = scores.argmax(axis=0)
    confidences = scores[class_ids, np.arange(scores.shape[1])]
    keep = (confidences >= conf_threshold) & np.isin(class_ids, classes)

    detections = []
    for (cx, cy, bw, bh), conf, cls in zip(output[:4, keep].T, confidences[keep], class_ids[keep]):
        # Letterbox pixels -> tile pixels, clipped to the tile, then shifted to the full image.
        x1 = np.clip((cx - bw / 2 - pad_x) / scale, 0, tile_w) + x0
        y1 = np.clip((cy - bh / 2 - pad_y) / scale, 0, tile_h) + y0
        x2 = np.clip((cx + bw / 2 - pad_x) / scale, 0, tile_w) + x0
        y2 = np.clip((cy + bh / 2 - pad_y) / scale, 0, tile_h) + y0
        detections.append(Detection(float(x1), float(y1), float(x2), float(y2), float(conf), int(cls)))
    return detections


def draw(image_bgr, detections, color=(0, 0, 255)):
    out = image_bgr.copy()
    for d in detections:
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, color, 2)
        cv2.putText(out, f"{d.confidence:.2f}", (p1[0], max(p1[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out
