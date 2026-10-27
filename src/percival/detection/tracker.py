import inspect
from types import SimpleNamespace
import numpy as np
from percival.contracts import Observation

MIN_MATCH_IOU = 0.3

def iou_matrix(a, b):
  ix1, iy1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
  ix2, iy2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])

  inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
  area = lambda m: (m[:, 2] - m[:, 0]) * (m[:, 3] - m[:, 1])

  return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)

def assign(rows, row_cls, dets, det_cls, min_iou=MIN_MATCH_IOU):
  if len(rows) == 0 or len(dets) == 0: return {}

  iou = iou_matrix(rows, dets)
  iou[row_cls[:, None] != det_cls[None, :]] = 0
  pairs = sorted(((iou[i, j], i, j) for i, j in zip(*np.nonzero(iou >= min_iou))), reverse=True)
  used_rows, used_dets, out = set(), set(), {}

  for _, i, j in pairs:
    if i in used_rows or j in used_dets: continue
    used_rows.add(i)
    used_dets.add(j)
    out[int(i)] = int(j)
  return out

class Detections:
  def __init__(self, conf, xywh, cls):
    self.conf, self.xywh, self.cls = conf, xywh, cls

  def __len__(self): return len(self.conf)
  def __getitem__(self, mask): return Detections(self.conf[mask], self.xywh[mask], self.cls[mask])

  @property
  def xyxy(self):
    half = self.xywh[:, 2:4] / 2
    return np.concatenate([self.xywh[:, :2] - half, self.xywh[:, :2] + half], axis=1)

class Tracker:
  def __init__(self, cfg, version, frame_rate=None):
    from ultralytics.trackers.byte_tracker import BYTETracker
    args = SimpleNamespace(track_high_thresh=cfg.track_high_thresh, track_low_thresh=cfg.track_low_thresh, new_track_thresh=cfg.new_track_thresh, track_buffer=cfg.track_buffer, match_thresh=cfg.match_thresh, fuse_score=True)
    legacy = "frame_rate" in inspect.signature(BYTETracker.__init__).parameters
    self.tracker = BYTETracker(args, frame_rate=30) if legacy else BYTETracker(args)
    self.version = version
    self.last_dropped = 0

  def update(self, frame, detections):
    n = len(detections)
    corners = np.array([[d.box[0], d.box[1], d.box[0] + d.box[2], d.box[1] + d.box[3]] for d in detections], dtype=np.float32).reshape(n, 4)
    centers = np.array([[d.box[0] + d.box[2] / 2, d.box[1] + d.box[3] / 2, d.box[2], d.box[3]] for d in detections], dtype=np.float32).reshape(n, 4)
    det_cls = np.array([d.class_id for d in detections], dtype=np.int64)
    dets = Detections(np.array([d.confidence for d in detections], dtype=np.float32), centers, det_cls.astype(np.float32))
    tracked = np.asarray(self.tracker.update(dets), dtype=np.float32)

    if tracked.size == 0:
      self.last_dropped = 0
      return []

    tracked = tracked.reshape(len(tracked), -1)
    matches = assign(tracked[:, :4], tracked[:, 6].astype(np.int64), corners, det_cls)
    self.last_dropped = len(tracked) - len(matches)

    return [self._observation(detections[j], int(tracked[i, 4])) for i, j in sorted(matches.items())]

  def _observation(self, d, track_id):
    return Observation(**d.model_dump(), track_id=track_id, tracker_version=self.version)