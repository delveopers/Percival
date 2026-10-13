from collections import Counter
from dataclasses import dataclass, field
from percival.contracts import TrackPoint, Tracklet

@dataclass(slots=True)
class OpenTrack:
  track_id: int
  run_id: str; clip_id: str
  detector_version: str; tracker_version: str
  points: list = field(default_factory=list)
  votes: Counter = field(default_factory=Counter)
  conf_sum: float = 0.0
  last_index: int = 0

class TrackletBuilder:
  def __init__(self, cfg):
    self.cfg = cfg
    self.tracks = {}

  def update(self, observations, sample_index):
    for o in observations:
      t = self.tracks.get(o.track_id)
      if t is None: t = self.tracks[o.track_id] = OpenTrack(o.track_id, o.run_id, o.clip_id, o.detector_version, o.tracker_version)
      t.points.append(TrackPoint(source_index=o.source_index, sample_index=o.sample_index, timestamp_s=o.timestamp_s, box=o.box, confidence=o.confidence))
      t.votes[o.class_name] += 1
      t.conf_sum += o.confidence
      t.last_index = o.sample_index
    stale = [i for i, t in self.tracks.items() if sample_index - t.last_index > self.cfg.max_lost_frames]
    return self._close(stale)

  def finish(self):
    return self._close(list(self.tracks))

  def _close(self, ids):
    closed = (self._to_tracklet(self.tracks.pop(i)) for i in ids)
    return [t for t in closed if t is not None]

  def _to_tracklet(self, t):
    if len(t.points) < self.cfg.min_track_length: return None
    first, last = t.points[0], t.points[-1]
    return Tracklet(run_id=t.run_id, clip_id=t.clip_id, track_id=t.track_id, class_name=t.votes.most_common(1)[0][0], start_s=first.timestamp_s, end_s=last.timestamp_s, start_index=first.sample_index, end_index=last.sample_index, mean_confidence=t.conf_sum / len(t.points), points=t.points, detector_version=t.detector_version, tracker_version=t.tracker_version)