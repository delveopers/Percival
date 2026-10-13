from dataclasses import dataclass
from percival.contracts import Anomaly
from .rules import RULES, RuleInput

@dataclass(slots=True)
class ActiveEvent:
  kind: str
  start_s: float; end_s: float
  start_index: int; end_index: int
  confidence: float
  track_ids: list
  class_names: list
  box: tuple
  details: dict

class AnomalyEngine:
  def __init__(self, cfg, run_id, clip_id):
    self.cfg, self.run_id, self.clip_id = cfg, run_id, clip_id
    self.rules = [RULES[n] for n in cfg.rules]
    self.events = []
    self.frames = 0

  @property
  def active(self):
    return [self._anomaly(e) for e in self.events.values()]

  def update(self, frame, observations, tracks, scene_change=False):
    c, now = self.cfg, frame.timestamp_s
    self.frames += 1

    if scene_change:
      for e in self.events.values(): e.last_s = now
      return []

    live, seen = self.frames > c.warmup_frames, set()
    inp = RuleInput(frame, observations, tracks)
    
    for rule in self.rules:
      for key, trig in rule.check(inp).items():
        if not live or trig.confidence < c.min_confidence: continue
        k = (rule.kind, key)
        seen.add(k)
        self._apply(k, rule.kind, trig, frame)
    expired = [k for k, e in self.events.items() if k not in seen and now - e.last_s > c.cooldown_seconds]
    return [self._anomaly(self.events.pop(k)) for k in expired]

  def finish(self):
    closed = [self._anomaly(e) for e in self.events.values()]
    self.events.clear()
    return closed

  def _apply(self, k, kind, trig, frame):
    e = self.events.get(k)
    
    if e is None:
      self.events[k] = ActiveEvent(kind, frame.timestamp_s, frame.timestamp_s, frame.sample_index, frame.sample_index, trig.confidence, sorted(trig.track_ids), sorted(trig.class_names), trig.box, trig.details)
      return
    
    e.last_s, e.last_index = frame.timestamp_s, frame.sample_index
    e.confidence = max(e.confidence, trig.confidence)
    e.track_ids = sorted(set(e.track_ids) | set(trig.track_ids))
    e.class_names = sorted(set(e.class_names) | set(trig.class_names))
    e.box, e.details = trig.box, trig.details

  def _anomaly(self, e):
    return Anomaly(run_id=self.run_id, clip_id=self.clip_id, kind=e.kind, confidence=e.confidence, start_s=e.start_s, end_s=e.last_s, start_index=e.start_index, end_index=e.last_index, track_ids=e.track_ids, class_names=e.class_names, box=e.box, details=e.details)