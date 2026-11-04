import math
from collections import deque
from dataclasses import dataclass, field
import numpy as np
from percival.contracts import Frame
from .features import anchor_spread, center, confidence, label, mean_conf, median_size, path_length, present, rnd, size, spread, speed, still_limit, window

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
  kind: str = None

@dataclass(slots=True)
class RuleInput:
  frame: Frame
  observations: list
  tracks: dict
  detections: list = None
  states: dict = None
  feats: dict = field(default_factory=dict)

  def feat(self, t):
    f = self.feats.get(t.track_id)
    if f is None: f = self.feats[t.track_id] = (label(t), median_size(t.points[-10:]))
    return f

def stills(inp, cfg, classes, seconds):
  now, states, out = inp.frame.timestamp_s, inp.states or {}, []
  for t in present(inp, classes, cfg.rule_min_box_px):
    s = states.get(t.track_id)
    if s is None or s.state != "still" or s.pending is not None or now - s.since[0] < seconds: continue
    out.append((t, s))
  return out

STILL_SLACK = 1.25

def still_trigger(cfg, t, s, inp, extra=None):
  now, name = inp.frame.timestamp_s, label(t)
  pts = window(t, now, now - s.since[0])
  sp = anchor_spread(pts, s.anchor)
  limit = still_limit(cfg, name, median_size(pts))
  if sp > limit * STILL_SLACK: return None
  details = {"duration_s": rnd(now - s.since[0]), "spread_px": rnd(sp), "limit_px": rnd(limit), **(extra or {})}
  return Trigger(confidence(mean_conf(pts[-30:]), 1 - sp / limit), [t.track_id], [name], pts[-1].box, details, s.since[0], s.since[1])

def still_triggers(cfg, inp, classes, seconds, extra=None):
  out = {}
  for t, s in stills(inp, cfg, classes, seconds):
    trig = still_trigger(cfg, t, s, inp, extra)
    if trig is not None: out[t.track_id] = trig
  return out

class StationaryRule:
  name, kind = "stationary", "stationary_object"
  def __init__(self, cfg): self.cfg = cfg

  def check(self, inp):
    c = self.cfg
    return still_triggers(c, inp, c.stationary_classes, c.stationary_seconds)
class DwellRule:
  name, kind = "dwell", "dwell"

  def __init__(self, cfg): self.cfg = cfg

  def check(self, inp):
    c = self.cfg
    return {t.track_id: still_trigger(c, t, s, inp) for t, s in stills(inp, c, c.dwell_classes, c.dwell_seconds)}

class AbandonedRule:
  name, kind = "abandoned", "abandoned_object"

  def __init__(self, cfg):
    self.cfg = cfg
    self.people = deque()

  def check(self, inp):
    c, now = self.cfg, inp.frame.timestamp_s
    self.people.append((now, [(*center(o.box), o.box[3]) for o in inp.observations if o.class_name == OWNER_CLASS]))
    while self.people and self.people[0][0] < now - c.owner_absent_seconds: self.people.popleft()
    out = {}

    for t, s in stills(inp, c, c.abandoned_classes, c.stationary_seconds):
      bx, by = center(t.points[-1].box)
      if self._owner_near(bx, by): continue
      trig = still_trigger(c, t, s, inp, {"owner_absent_s": c.owner_absent_seconds})
      trig.since = trig.since_index = None
      out[t.track_id] = trig
    return out

  def _owner_near(self, bx, by):
    d = self.cfg.owner_distance
    return any(np.hypot(px - bx, py - by) / max(h, 1e-6) <= d for _, people in self.people for px, py, h in people)

class LoiterRule:
  name, kind = "loitering", "loitering"

  def __init__(self, cfg): self.cfg = cfg

  def check(self, inp):
    c, now, out = self.cfg, inp.frame.timestamp_s, {}
    for t in present(inp, c.loiter_classes, c.rule_min_box_px):
      if t.points[0].timestamp_s > now - c.loitering_seconds: continue
      pts = window(t, now, c.loitering_seconds)
      sp, cs = spread(pts)
      sz = median_size(pts)
      limit, low = max(c.loiter_radius * sz, 1e-6), max(c.loiter_min_spread * sz, c.loiter_min_px)
      if sp > limit or sp < low: continue
      details = {"duration_s": rnd(pts[-1].timestamp_s - pts[0].timestamp_s), "spread_px": rnd(sp), "limit_px": rnd(limit), "min_spread_px": rnd(low), "path_px": rnd(path_length(cs))}
      out[t.track_id] = Trigger(confidence(mean_conf(pts), 1 - sp / limit), [t.track_id], [label(t)], pts[-1].box, details)
    return out

class SpeedRule:
  name, kind = "speed", "speed_outlier"

  def __init__(self, cfg):
    self.cfg = cfg
    self.stats = {}
    self.since = {}

  def check(self, inp):
    c, now, idx = self.cfg, inp.frame.timestamp_s, inp.frame.sample_index
    out, updates, live = {}, [], set()
    for t in present(inp, c.speed_classes, c.rule_min_box_px):
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

def clusters(centers, sizes, link):
  n = len(centers)
  if n == 0: return []
  dist = np.hypot(centers[:, None, 0] - centers[None, :, 0], centers[:, None, 1] - centers[None, :, 1])
  adj = dist <= link * np.maximum(sizes[:, None], sizes[None, :])
  seen, out = np.zeros(n, dtype=bool), []
  for i in range(n):
    if seen[i]: continue
    seen[i] = True
    stack, comp = [i], []
    while stack:
      j = stack.pop()
      comp.append(j)
      for k in np.nonzero(adj[j] & ~seen)[0]:
        seen[k] = True
        stack.append(int(k))
    out.append(comp)
  return out

class DensityRule:
  name = "density"
  kind = "crowd"

  def __init__(self, cfg):
    self.cfg = cfg
    self.cur = None
    self.n_groups = 0

  def check(self, inp):
    c, now = self.cfg, inp.frame.timestamp_s
    source = inp.observations if inp.detections is None else inp.detections
    people = [o for o in source if o.class_name in c.crowd_classes and o.confidence >= c.crowd_min_confidence and size(o.box) >= c.rule_min_box_px]
    level, via, members = self._level(people)
    if level is None:
      self.cur = None
      return {}
    if self.cur is None or self.cur[0] != level: self.cur = (level, now, inp.frame.sample_index)
    if now - self.cur[1] < c.density_min_seconds: return {}
    chosen = [people[i] for i in members]
    x1, y1 = min(o.box[0] for o in chosen), min(o.box[1] for o in chosen)
    x2, y2 = max(o.box[0] + o.box[2] for o in chosen), max(o.box[1] + o.box[3] for o in chosen)
    floor = c.crowd_min if level == "crowd" else c.group_min
    details = {"level": level, "via": via, "people": len(people), "in_cluster": len(chosen), "groups": self.n_groups, "sustained_s": rnd(now - self.cur[1])}
    trig = Trigger(confidence(float(np.mean([o.confidence for o in chosen])), (len(chosen) - floor) / floor), [], sorted({o.class_name for o in chosen}), (x1, y1, x2 - x1, y2 - y1), details, self.cur[1], self.cur[2], level)
    return {"scene": trig}

  def _level(self, people):
    c = self.cfg
    n = len(people)
    self.n_groups = 0

    if n < c.group_min: return None, None, None
    relax = 0.75 if self.cur is not None and self.cur[0] == "crowd" else 1.0
    cmin, total = max(c.group_min, math.ceil(c.crowd_min * relax)), max(c.group_min, math.ceil(c.crowd_total * relax))
    centers = np.array([center(o.box) for o in people])
    sizes = np.array([size(o.box) for o in people], dtype=float)
    near = clusters(centers, sizes, c.group_link)
    groups = [g for g in near if len(g) >= c.group_min]
    self.n_groups = len(groups)

    if not groups and n < total: return None, None, None
    biggest = max(near, key=len)
    if len(biggest) >= cmin: return "crowd", "cluster", biggest
    if n >= total: return "crowd", "total", list(range(n))

    if len(groups) >= 2:
      for comp in clusters(centers, sizes, c.crowd_link):
        inside = sum(1 for g in groups if g[0] in comp)
        if inside >= 2 and len(comp) >= cmin: return "crowd", "nearby_groups", comp
    return "group", "cluster", max(groups, key=len)

class SurgeRule:
  name, kind = "surge", "crowd_surge"

  def __init__(self, cfg):
    self.cfg = cfg
    self.mean = self.var = self.last = None
    self.since = self.frozen = None

  def check(self, inp):
    c, now = self.cfg, inp.frame.timestamp_s
    source = inp.observations if inp.detections is None else inp.detections
    members = [o for o in source if o.class_name in c.crowd_classes and o.confidence >= c.crowd_min_confidence and size(o.box) >= c.rule_min_box_px]
    ids = sorted(o.track_id for o in inp.observations if o.class_name in c.crowd_classes)
    n = len(members)
    mean, sigma, z, surge = self.mean, 0.0, 0.0, False

    if mean is not None:
      sigma = max(math.sqrt(self.var), 0.5 * math.sqrt(max(mean, 1.0)), 1.0)
      z = (n - mean) / sigma
      enter = n >= c.surge_count and z >= c.surge_zscore and n - mean >= c.surge_min_increase
      stay = self.since is not None and n >= c.surge_count and z >= c.surge_zscore / 2 and n - mean >= c.surge_min_increase / 2
      surge = enter or stay

    if surge and self.frozen is None: self.frozen = now
    if not surge: self.frozen = None
    self._learn(n, now, surge and now - self.frozen < c.surge_baseline_seconds)

    if not surge:
      self.since = None
      return {}

    if self.since is None: self.since = (now, inp.frame.sample_index)
    if now - self.since[0] < c.surge_min_seconds: return {}

    x1, y1 = min(o.box[0] for o in members), min(o.box[1] for o in members)
    x2, y2 = max(o.box[0] + o.box[2] for o in members), max(o.box[1] + o.box[3] for o in members)
    details = {"count": n, "baseline": rnd(mean), "sigma": rnd(sigma), "zscore": rnd(z), "threshold": c.surge_zscore, "sustained_s": rnd(now - self.since[0])}
    trigger = Trigger(confidence(float(np.mean([o.confidence for o in members])), (z - c.surge_zscore) / c.surge_zscore), ids, sorted({o.class_name for o in members}), (x1, y1, x2 - x1, y2 - y1), details, self.since[0], self.since[1])

    return {"scene": trigger}

  def _learn(self, n, now, frozen):
    if self.mean is None:
      self.mean, self.var, self.last = float(n), 0.0, now
      return

    dt, self.last = max(now - self.last, 0.0), now
    if frozen: return
    alpha = 1 - math.exp(-dt / self.cfg.surge_baseline_seconds)
    d = n - self.mean
    self.mean += alpha * d
    self.var = (1 - alpha) * (self.var + alpha * d * d)

RULES = {r.name: r for r in (StationaryRule, DwellRule, AbandonedRule, LoiterRule, SpeedRule, DensityRule, SurgeRule)}