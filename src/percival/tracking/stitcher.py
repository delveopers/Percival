import math
from dataclasses import dataclass

@dataclass(slots=True)
class Memory:
  t: float
  x: float; y: float
  size: float
  cls: float

class Stitcher:
  def __init__(self, cfg):
    self.cfg = cfg
    self.alias, self.mem = {}, {}
    self.merges = 0

  def process(self, obs):
    self.merges = 0
    if not obs: return obs
    now = obs[0].timestamp_s
    known = {o.track_id: self.alias[o.track_id] for o in obs if o.track_id in self.alias}
    new = [o for o in obs if o.track_id not in self.alias]

    for raw, canon in self._match(new, now, set(known.values())).items():
      known[raw] = canon

    out = []
    for o in obs:
      canon = known.get(o.track_id, o.track_id)
      self.alias[o.track_id] = canon
      self.mem[canon] = Memory(o.timestamp_s, o.box[0] + o.box[2] / 2, o.box[1] + o.box[3] / 2, max(o.box[2], o.box[3]), o.class_name)
      out.append(o if canon == o.track_id else o.model_copy(update={"track_id": canon, "raw_track_id": o.track_id}))
    return out

  def _match(self, new, now, active):
    c = self.cfg
    for canon in [k for k, m in self.mem.items() if now - m.t > c.stitch_gap_seconds]: del self.mem[canon]
    cands = {k: m for k, m in self.mem.items() if k not in active and now - m.t > 0}
    if not new or not cands: return {}
    costs = {}

    for o in new:
      ox, oy, osize = o.box[0] + o.box[2] / 2, o.box[1] + o.box[3] / 2, max(o.box[2], o.box[3])
      for canon, m in cands.items():
        if m.cls != o.class_name: continue
        reach = (c.stitch_distance + c.stitch_speed * (now - m.t)) * max(osize, m.size)
        cost = math.hypot(ox - m.x, oy - m.y) / reach
        if cost <= 1.0: costs[(o.track_id, canon)] = cost

    by_new, by_cand = {}, {}
    for (raw, canon), cost in costs.items():
      by_new.setdefault(raw, []).append(cost)
      by_cand.setdefault(canon, []).append(cost)

    clear = lambda group, key, cost: len(group[key]) == 1 or cost <= c.stitch_ambiguity * sorted(group[key])[1]
    out, used = {}, set()

    for (raw, canon), cost in sorted(costs.items(), key=lambda kv: kv[1]):
      if raw in out or canon in used: continue
      if cost > min(by_new[raw]) or cost > min[by_cand[canon]]: continue
      if clear(by_new, raw, cost) and clear(by_cand, canon, cost):
        out[raw] = canon
        used.add(canon)
    self.merges = len(out)
    return out