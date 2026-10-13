from collections import Counter, defaultdict
from contextlib import contextmanager
from time import perf_counter


class Metrics:
  def __init__(self):
    self.counts, self.seconds = Counter(), defaultdict(float)

  def add(self, name, n=1): self.counts[name] += n

  @contextmanager
  def time(self, stage):
    t = perf_counter()
    try: yield
    finally: self.seconds[stage] += perf_counter() - t

  def merge(self, other):
    self.counts.update(other.counts)
    for k, v in self.seconds.items(): self.seconds[k] += v

  def snapshot(self):
    n = max(self.counts.get("frames", 0), 1)
    return {"counts": dict(self.counts), "seconds": {k: round(v, 4) for k, v in self.seconds.items()}, "avg_ms_per_frame": {k: round(v / n * 1000, 3) for k, v in self.seconds.items()}}