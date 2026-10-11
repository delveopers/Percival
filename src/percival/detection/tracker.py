from types import SimpleNamespace
import numpy as np
from percival.contracts import Observation

class Tracker:
  def __init__(self, cfg, version, frame_rate):
    from ultralytics.trackers.byte_tracker import BYTETracker

    args = SimpleNamespace(track_high_thresh=cfg.track_high_thresh, track_low_thresh=cfg.track_low_thresh, new_track_thresh=cfg.new_track_thresh, track_buffer=cfg.track_buffer, match_thresh=cfg.match_thresh, fuse_score=True)
    self.tracker = BYTETracker(args, frame_rate=max(1, int(round(frame_rate))))
    self.version = version

  def update(self, frame, detections):
    n = len(detections)
    boxes = np.array([[d.box[0] + d.box[2] / 2, d.box[1] + d.box[3] / 2, d.box[2], d.box[3]] for d in detections], dtype=np.float32).reshape(n, 4)
    dets = SimpleNamespace(conf=np.array([d.confidence for d in detections], dtype=np.float32), xywh=boxes, cls=np.array([d.class_id for d in detections], dtype=np.float32))
    tracked = self.tracker.update(dets)
    return [self._observation(detections[int(row[7])], int(row[4])) for row in tracked if int(row[7]) < n]

  def _observation(self, d, track_id):
    return Observation(**d.model_dump(), track_id=track_id, tracker_version=self.version)