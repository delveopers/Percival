from types import SimpleNamespace
import numpy as np
from percival.contracts import Observation

class _TrackerDetections:
  """Minimal Ultralytics Results-like container for BYTETracker."""

  def __init__(self, conf, xywh, cls):
    self.conf = conf
    self.xywh = xywh
    self.cls = cls

  def __len__(self): return len(self.conf)

  def __getitem__(self, mask):
    return _TrackerDetections(
      conf=self.conf[mask],
      xywh=self.xywh[mask],
      cls=self.cls[mask],
    )

class Tracker:
  def __init__(self, cfg, version, frame_rate):
    from ultralytics.trackers.byte_tracker import BYTETracker
    args = SimpleNamespace(
      track_high_thresh=cfg.track_high_thresh,
      track_low_thresh=cfg.track_low_thresh,
      new_track_thresh=cfg.new_track_thresh,
      track_buffer=cfg.track_buffer,
      match_thresh=cfg.match_thresh,
      fuse_score=True,
    )

    self.tracker = BYTETracker(args)
    self.version = version

  def update(self, frame, detections):
    n = len(detections)
    if n == 0: return []

    boxes = np.asarray([
        [
          d.box[0] + d.box[2] / 2,
          d.box[1] + d.box[3] / 2,
          d.box[2],
          d.box[3],
          i,
        ]
        for i, d in enumerate(detections)
      ],
    dtype=np.float32)

    dets = _TrackerDetections(
      conf=np.asarray([d.confidence for d in detections], dtype=np.float32),
      xywh=boxes,
      cls=np.asarray([d.class_id for d in detections], dtype=np.float32),
    )
    tracked = self.tracker.update(dets)

    return [
      self._observation(detections[int(row[7])], int(row[4]))
      for row in tracked
      if int(row[7]) < n
    ]

  def _observation(self, d, track_id):
    return Observation(
      **d.model_dump(),
      track_id=track_id,
      tracker_version=self.version,
    )