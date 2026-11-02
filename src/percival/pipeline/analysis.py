from percival.anomaly import AnomalyEngine
from percival.behavior import BehaviorStates
from percival.tracking import TrackletBuilder

class Analysis:
  def __init__(self, cfg, run_id, clip_id, tracker, writers, metrics):
    self.tracker, self.writers, self.m = tracker, writers, metrics
    self.builder = TrackletBuilder(cfg.tracking)
    self.engine = AnomalyEngine(cfg.anomaly, run_id, clip_id) if cfg.anomaly.enabled else None
    self.behavior = BehaviorStates(cfg.anomaly, run_id, clip_id) if cfg.anomaly.enabled else None

  def step(self, frame, dets, scene):
    m, w = self.m, self.writers
    with m.time("track"): obs = self.tracker.update(frame, dets)
    m.add("tracker_dropped", getattr(self.tracker, "last_dropped", 0))
    m.add("duplicate_box_frames", len({o.box for o in obs}) < len(obs))

    for o in obs: w["observations"].write(o)
    m.add("observations", len(obs))
    closed = self.builder.update(obs, frame.sample_index)
    active = []

    if self.engine:
      self._segments(self.behavior.update(frame, self.builder.tracks))
      with m.time("anomaly"): events = self.engine.update(frame, obs, self.builder.tracks, scene, dets, self.behavior.states)
      for a in events: w["anomalies"].write(a)
      m.add("anomalies", len(events))
      active = self.engine.active
    self._tracklets(closed)

    return obs, active

  def finish(self):
    self._tracklets(self.builder.finish())
    if self.engine:
      self._segments(self.behavior.finish())
      tail = self.engine.finish()
      for a in tail: self.writers["anomalies"].write(a)
      self.m.add("anomalies", len(tail))
    self.m.counts["orphan_observations"] = self.m.counts["observations"] - self.m.counts["tracklet_points"]

  def _tracklets(self, tracklets):
    for t in tracklets:
      self.writers["tracklets"].write(t)
      self.m.add("tracklet_points", len(t.points))
    self.m.add("tracklets", len(tracklets))

  def _segments(self, segments):
    for seg in segments: self.writers["states"].write(seg)
    self.m.add("state_segments", len(segments))