import logging
from percival.contracts import Detection

log = logging.getLogger(__name__)

def resolve_device(requested):
  if requested != "auto": return requested
  import torch
  return "cuda:0" if torch.cuda.is_available() else "cpu"

class Detector:
  def __init__(self, cfg, version):
    from ultralytics import YOLO

    self.cfg, self.version = cfg, version
    self.device = resolve_device(cfg.device)
    self.half = cfg.half and self.device != "cpu"
    self.model = YOLO(cfg.weights)
    self.names = dict(self.model.names)

    lookup = {name: i for i, name in self.names.items()}
    missing = [c for c in cfg.classes if c not in lookup]

    if missing: raise ValueError(f"Classes not supported by {cfg.weights}: {missing}")
    self.class_ids = [lookup[c] for c in cfg.classes]
    log.info("detector ready: weights=%s device=%s half=%s classes=%d", cfg.weights, self.device, self.half, len(self.class_ids))

  def detect(self, frame):
    c = self.cfg
    quantize = 16 if self.half else 32
    result = self.model.predict(frame.image, conf=c.conf, iou=c.iou, imgsz=c.imgsz, classes=self.class_ids, device=self.device, quantize=quantize, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0: return []
    xyxy, conf, cls = boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)
    return [self._detection(frame, b, float(s), int(k)) for b, s, k in zip(xyxy, conf, cls)]

  def _detection(self, frame, b, score, class_id):
    x1, y1, x2, y2 = (float(v) for v in b)
    return Detection(run_id=frame.run_id, clip_id=frame.clip_id, source_index=frame.source_index, sample_index=frame.sample_index, timestamp_s=frame.timestamp_s,
                     box=(x1, y1, x2 - x1, y2 - y1), class_id=class_id, class_name=self.names[class_id], confidence=score, detector_version=self.version)