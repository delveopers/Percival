from dataclasses import dataclass
import numpy as np
from percival.anomaly.features import center, label, median_size, still_limit, window
from percival.contracts import TrackSegment

@dataclass(slots=True)
class TrackState:
  state: str = None
  since: tuple = None
  anchor: tuple = None
  pending: str = None
  pending_since: tuple = None
  cls: str = ""
  last: tuple = None

class BehaviorStates:
  def __init__(self, cfg, run_id, clip_id):
    self.cfg, self.run_id, self.clip_id = cfg, run_id, clip_id
    self.states = {}

  def update(self, frame, tracks):
    c, now, idx, closed = self.cfg, frame.timestamp_s, frame.sample_index, []
    for tid, t in tracks.items():
      if t.last_index != idx: continue
      ts = self.states.setdefault(tid, TrackState())
      ts.last, ts.cls = (now, idx), label(t)
      got = self._classify(t, ts, now)
      if got is None: continue
      inst, first, median = got
      if ts.state is None:
        ts.state, ts.since, ts.anchor = inst, (first.timestamp_s, first.sample_index), median
      elif inst == ts.state: ts.pending = None
      else:
        if ts.pending != inst: ts.pending, ts.pending_since = inst, (now, idx)
        if now - ts.pending_since[0] >= c.state_min_seconds:
          closed.append(self._segment(tid, ts, ts.pending_since))
          ts.state, ts.since, ts.pending = inst, ts.pending_since, None
          ts.anchor = median if inst == "still" else None
    for tid in [k for k in self.states if k not in tracks]:
      seg = self._segment(tid, self.states.pop(tid), None)
      if seg: closed.append(seg)
    return closed

  def finish(self):
    out = [self._segment(tid, ts, None) for tid, ts in self.states.items()]
    self.states.clear()
    return [s for s in out if s]

  def _classify(self, t, ts, now):
    c = self.cfg
    pts = window(t, now, c.state_window_seconds)

    if len(pts) < 4 or pts[-1].timestamp_s - pts[0].timestamp_s < 0.75 * c.state_window_seconds: return None
    size_px = median_size(pts)
    if size_px < c.rule_min_box_px: return None
    centers = np.array([center(p.box) for p in pts])
    median = tuple(np.median(centers, axis=0))
    ref = np.asarray(ts.anchor) if ts.state == "still" and ts.anchor is not None else np.asarray(median)
    d = np.linalg.norm(centers - ref, axis=1)
    k = int(round(0.9 * (len(d) - 1)))
    return ("still" if float(np.partition(d, k)[k]) <= still_limit(c, ts.cls, size_px) else "moving"), pts[0], median

  def _segment(self, tid, ts, end):
    if ts.state is None: return None
    end = end or ts.last
    return TrackSegment(run_id=self.run_id, clip_id=self.clip_id, track_id=tid, class_name=ts.cls, state=ts.state, start_s=ts.since[0], end_s=end[0], start_index=ts.since[1], end_index=end[1])