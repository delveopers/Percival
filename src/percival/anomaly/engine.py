from dataclasses import dataclass
from percival.contracts import Anomaly
from .rules import RULES, RuleInput

@dataclass(slots=True)
class ActiveEvent:
  kind: str
  start_s: float; last_s: float
  start_index: int; last_index: int
  confidence: float
  track_ids: list
  class_names: list
  box: tuple
  details: dict

class AnomalyEngine:
  def __init__(self, cfg, run_id, clip_id):
    self.cfg, self.run_id, self.clip_id = cfg, run_id, clip_id
    self.rules = [RULES[n](cfg) for n in cfg.rules]
    self.events = {}
    self.start = None

  @property
  def active(self):
    return [self._anomaly(e) for e in self.events.values()]

  def update(self, frame, observations, tracks, scene_change=False, detections=None, states=None):
    c, now = self.cfg, frame.timestamp_s
    if self.start is None: self.start = now
    if scene_change:
      for e in self.events.values(): e.last_s = now
      return []

    live, seen, closed = now - self.start >= c.warmup_seconds - 1e-9, set(), []
    inp = RuleInput(frame, observations, tracks, detections, states)

    for rule in self.rules:
      for key, trig in rule.check(inp).items():
        if not live or trig.confidence < c.min_confidence: continue
        k = (rule.name, key)
        seen.add(k)
        old = self._apply(k, trig.kind or rule.kind, trig, frame)
        if old: closed.append(old)

    expired = [k for k, e in self.events.items() if k not in seen and now - e.last_s > c.cooldown_seconds]
    return closed + [self._anomaly(self.events.pop(k)) for k in expired]

  def finish(self):
    closed = [self._anomaly(e) for e in self.events.values()]
    self.events.clear()
    return closed

  def _apply(self, k, kind, trig, frame):
    e, replaced = self.events.get(k), None
    start_s, start_i = (frame.timestamp_s, frame.sample_index) if trig.since is None else (min(frame.timestamp_s, trig.since), min(frame.sample_index, trig.since_index))
    if e is not None and e.kind != kind:
      e.last_s, e.last_index = min(e.last_s, start_s), min(e.last_index, start_i)
      replaced = self._anomaly(self.events.pop(k))
      e = None

    if e is None:
      self.events[k] = ActiveEvent(kind, start_s, frame.timestamp_s, start_i, frame.sample_index, trig.confidence, sorted(trig.track_ids), sorted(trig.class_names), trig.box, trig.details)
      return replaced

    e.last_s, e.last_index = frame.timestamp_s, frame.sample_index
    e.confidence = max(e.confidence, trig.confidence)
    e.track_ids = sorted(set(e.track_ids) | set(trig.track_ids))
    e.class_names = sorted(set(e.class_names) | set(trig.class_names))
    e.box, e.details = trig.box, trig.details
    return replaced

  def _anomaly(self, e):
    return Anomaly(run_id=self.run_id, clip_id=self.clip_id, kind=e.kind, confidence=e.confidence, start_s=e.start_s, end_s=e.last_s, start_index=e.start_index, end_index=e.last_index, track_ids=e.track_ids, class_names=e.class_names, box=e.box, details=e.details)