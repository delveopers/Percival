import math
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
  since: float = None
  since_index: int = None

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
  c = np.array([center(p.box) for p in points])
  return float(np.linalg.norm(c - np.median(c, axis=0), axis=1).max()), c

def speed(points):
  c = np.array([center(p.box) for p in points])
  dt = np.diff([p.timestamp_s for p in points])
  step = np.linalg.norm(np.diff(c, axis=0), axis=1)
  sizes = np.array([size(p.box) for p in points[1:]])

  return float(np.median(step / np.maximum(dt, 1e-6) / np.maximum(sizes, 1e-6)))

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
  details = {"duration_s": rnd(pts[-1].timestamp_s - pts[0].timestamp_s), "spread_px": rnd(sp), "limit_px": rnd(limit), **(extra or {})}
  return Trigger(confidence(mean_conf(pts), 1 - sp / limit), [t.track_id], [label(t)], pts[-1].box, details)

class StationaryRule:
  kind = "stationary_object"

  def __init__(self, cfg): self.cfg = cfg

  def check(self, inp):
    return {t.track_id: still_trigger(t, pts, sp, limit) for t, pts, sp, limit in still(inp, self.cfg, self.cfg.stationary_classes)}

class AbandonedRule:
  kind = "abandoned_object"

  def __init__(self, cfg):
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

  def __init__(self, cfg): self.cfg = cfg

  def check(self, inp):
    c, now, out = self.cfg, inp.frame.timestamp_s, {}

    for t in present(inp, c.loiter_classes):
      if t.points[0].timestamp_s > now - c.loitering_seconds: continue
      pts = window(t, now, c.loitering_seconds)
      sp, cs = spread(pts)
      sz = median_size(pts)
      limit = max(c.loiter_radius * sz, 1e-6)
      path = path_length(cs)
      if sp > limit or sp < c.loiter_min_spread * sz: continue
      details = {"duration_s": rnd(pts[-1].timestamp_s - pts[0].timestamp_s), "spread_px": rnd(sp), "limit_px": rnd(limit), "min_spread_px": rnd(c.loiter_min_spread * sz), "path_px": rnd(path)}
      out[t.track_id] = Trigger(confidence(mean_conf(pts), 1 - sp / limit), [t.track_id], [label(t)], pts[-1].box, details)
    return out

class SpeedRule:
  kind = "speed_outlier"

  def __init__(self, cfg):
    self.cfg = cfg
    self.stats = {}
    self.since = {}

  def check(self, inp):
    c, now, idx = self.cfg, inp.frame.timestamp_s, inp.frame.sample_index
    out, updates, live = {}, [], set()

    for t in present(inp, c.speed_classes):
      live.add(t.track_id)
      if len(t.points) < SPEED_POINTS: continue
      pts = t.points[-SPEED_POINTS:]
      v, name = speed(pts), label(t)
      n, mean, m2 = self.stats.get(name, (0, 0.0, 0.0))
      z = (v - mean) / max(np.sqrt(m2 / (n - 1)), STD_FLOOR) if n >= c.speed_min_samples else 0.0
      if z < c.speed_zscore:
        self.since.pop(t.track_id, None)
        updates.append((name, v))
        continue

      start = self.since.setdefault(t.track_id, (now, idx))
      if now - start[0] < c.speed_min_seconds: continue
      details = {"speed": rnd(v), "mean": rnd(mean), "zscore": rnd(z), "threshold": c.speed_zscore, "sustained_s": rnd(now - start[0]), "unit": "box_sizes_per_s"}
      out[t.track_id] = Trigger(confidence(mean_conf(pts), (z - c.speed_zscore) / c.speed_zscore), [t.track_id], [name], pts[-1].box, details, start[0], start[1])
    for tid in [k for k in self.since if k not in live]: del self.since[tid]

    for name, v in updates:
      n, mean, m2 = self.stats.get(name, (0, 0.0, 0.0))
      n += 1
      d = v - mean
      mean += d / n
      self.stats[name] = (n, mean, m2 + d * (v - mean))
    return out

class CrowdRule:
  kind = "crowd"

  def __init__(self, cfg):
    self.cfg = cfg
    self.mean = self.var = self.last = None
    self.since = self.frozen = None

  def check(self, inp):
    c, now = self.cfg, inp.frame.timestamp_s
    members = [o for o in inp.observations if o.class_name in c.crowd_classes]
    n = len(members)
    mean, sigma, z, surge = self.mean, 0.0, 0.0, False

    if mean is not None:
      sigma = max(math.sqrt(self.var), 0.5 * math.sqrt(max(mean, 1.0)), 1.0)
      z = (n - mean) / sigma
      enter = n >= c.crowd_count and z >= c.crowd_zscore and n - mean >= c.crowd_min_increase
      stay = self.since is not None and n >= c.crowd_count and z >= c.crowd_zscore / 2 and n - mean >= c.crowd_min_increase / 2
      surge = enter or stay
    if surge and self.frozen is None: self.frozen = now
    if not surge: self.frozen = None
    self._learn(n, now, surge and now - self.frozen < c.crowd_baseline_seconds)

    if not surge:
      self.since = None
      return {}
    if self.since is None: self.since = (now, inp.frame.sample_index)
    if now - self.since[0] < c.crowd_min_seconds: return {}

    x1, y1 = min(o.box[0] for o in members), min(o.box[1] for o in members)
    x2, y2 = max(o.box[0] + o.box[2] for o in members), max(o.box[1] + o.box[3] for o in members)
    details = {"count": n, "baseline": rnd(mean), "sigma": rnd(sigma), "zscore": rnd(z), "threshold": c.crowd_zscore, "sustained_s": rnd(now - self.since[0])}
    trigger = Trigger(confidence(float(np.mean([o.confidence for o in members])), (z - c.crowd_zscore) / c.crowd_zscore), [o.track_id for o in members], sorted({o.class_name for o in members}), (x1, y1, x2 - x1, y2 - y1), details, self.since[0], self.since[1])

    return {"scene": trigger}

  def _learn(self, n, now, frozen):
    if self.mean is None:
      self.mean, self.var, self.last = float(n), 0.0, now
      return

    dt, self.last = max(now - self.last, 0.0), now
    if frozen: return
    alpha = 1 - math.exp(-dt / self.cfg.crowd_baseline_seconds)
    d = n - self.mean
    self.mean += alpha * d
    self.var = (1 - alpha) * (self.var + alpha * d * d)

RULES = {"stationary": StationaryRule, "abandoned": AbandonedRule, "loitering": LoiterRule, "speed": SpeedRule, "crowd": CrowdRule}