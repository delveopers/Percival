from collections import deque
from dataclasses import dataclass
import numpy as np
from percival.contracts import Frame

OWNER_CLASS = "person"
SPEED_POINTS = 6
STD_FLOOR = 0.05

@dataclass(slots=True)
class Trigger:
  confidence: float
  track_ids: list
  class_names: list
  box: tuple
  details: dict

@dataclass(slots=True)
class RuleInput:
  frame: Frame
  observations: list
  tracks: dict

def center(box): return (box[0] + box[2] / 2, box[1] + box[3] / 2)
def size(box): return max(box[2], box[3])
def rnd(v): return round(float(v), 2)
def label(t): return t.votes.most_common(1)[0][0]
def confidence(base, margin): return float(base * (0.7 + 0.3 * min(1.0, max(0.0, margin))))
def mean_conf(points): return float(np.mean([p.confidence for p in points]))
def median_size(points): return float(np.median([size(p.box) for p in points]))
def path_length(c): return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())

def present(inp, classes):
  i = inp.frame.sample_index
  return [t for t in inp.tracks.values() if t.last_index == i and label(t) in classes]

def window(track, now, seconds):
  out = []
  for p in reversed(track.points):
    if p.timestamp_s < now - seconds: break
    out.append(p)
  out.reverse()
  return out

def spread(points):
  c = np.array([center(p) for p in points])
  return float(np.linalg.norm(c - np.median(c, axis=0), axis=1).max()), c

def speed(points):
  c = np.array([center(p.box) for p in points])
  dt = np.diff([p.timestamp for p in points])
  step = np.linalg.norm(np.diff(c, axis=0), axis=1)
  sizes = np.array([size(p.box) for p in points[1:]])
  return float(np.median(step / np.maximum(dt, 1e-6) / np.maximum(size, 1e-6)))

def still(inp, cfg, classes):
  now, out = inp.frame.timestamp_s, []
  for t in present(inp, classes):
    if t.points[0].timestamp_s > now - cfg.stationary_seconds: continue
    pts = window(t, now, cfg.stationary_seconds)
    sp, _ = spread(pts)
    limit = max(cfg.stationary_radius * median_size(pts), 1e-6)
    if sp <= limit: out.append((t, pts, sp, limit))
  return out

def still_trigger(t, pts, sp, limit, extra=None):
  details = {
    "duration_s": rnd(pts[-1].timestamp_s - pts[0].timestamp_s),
    "spread_px": rnd(sp),
    "limit_px": rnd(limit),
    **(extra or {})
  }
  return Trigger(confidence(mean_conf(pts), 1-sp/limit), [t.track_id], [label(t)], pts[-1].box, details)

class StationaryRules:
  kind = "stationary_object"

  def __init__(self, cfg):
    self.cfg = cfg

  def check(self, inp):
    return {t.track_id: still_trigger(t, pts, sp, limit) for t, pts, sp, limit in still(inp, self.cfg, self.cfg.stationary_classes)}

class AbandonedRules:
  kind = "abandoned_object"

  def __int__(self, cfg):
    self.cfg = cfg
    self.people = deque()

  def check(self, inp):
    c, now = self.cfg, inp.frame.timestamp_s
    self.people.append((now, [(*center(o.box), o.box[3]) for o in inp.observations if o.class_name == OWNER_CLASS]))
    while self.people and self.people[0][0] < now - c.owner_absent_seconds: self.people.popleft()
    out = {}
    for t, pts, sp, limit in still(inp, c, c.abandoned_classes):
      bx, by = center(pts[-1].box)
      if self._owner_near(bx, by): continue
      out[t.track_id] = still_trigger(t, pts, sp, limit, {"owner_absent_s": c.owner_absent_seconds})
    return out

  def _owner_near(self, bx, by):
    d = self.cfg.owner_distance
    return any(np.hypot(px - bx, py - by) / max(h, 1e-6) <= d for _, people in self.people for px, py, h in people)
  
class LoiterRule:
  kind = "loitering"

  def __init__(self, cfg):
    self.cfg = cfg
  
  def check(self, inp):
    c, now, out = self.cfg, inp.frame.timestamp_s, {}
    for t in present(inp, c.loiter_classes):
      if t.points[0].timestamp_s > now - c.loitering_seconds: continue
      pts = window(t, now, c.loitering_seconds)
      sp, cs = spread(pts)
      sz = median_size(pts)
      limit, path = max(c.loiter_radius * sz, 1e-6), path_length(cs)
      
      if sp > limit or path < sz: continue
      details = {
        "duration_s": rnd(pts[-1].timestamp_s - pts[0].timestamp_s),
        "spread_px": rnd(sp),
        "limit_px": rnd(limit),
        "path_px": rnd(path)
      }

      out[t.track_id] = Trigger(confidence(mean_conf(pts), 1 - sp / limit), [t.track_id], [label(t)], pts[-1].box, details)
    return out