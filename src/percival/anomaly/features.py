from bisect import bisect_left
from collections import Counter
import numpy as np

def center(box): return (box[0] + box[2] / 2, box[1] + box[3] / 2)
def size(box): return max(box[2], box[3])
def rnd(v): return round(float(v), 2)
def confidence(base, margin): return float(base * (0.7 + 0.3 * min(1.0, max(0.0, margin))))
def mean_conf(points): return float(np.mean([p.confidence for p in points]))
def median_size(points):
  v = sorted(max(p.box[2], p.box[3]) for p in points)
  m = len(v) // 2
  return float(v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2)

def path_length(c): return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())

def label(t): return Counter(t.recent).most_common(1)[0][0] if t.recent else t.votes.most_common(1)[0][0]

def radius_for(cfg, name): return cfg.dwell_radius if name in cfg.dwell_classes else cfg.stationary_radius
def still_limit(cfg, name, size_px): return max(radius_for(cfg, name) * size_px, cfg.still_min_px)

def window(track, now, seconds): return track.points[bisect_left(track.times, now - seconds):]

def spread(points, q=90):
  c = np.array([center(p.box) for p in points])
  d = np.linalg.norm(c - np.median(c, axis=0), axis=1)
  k = int(round(q / 100 * (len(d) - 1)))
  return float(np.partition(d, k)[k]), c

def anchor_spread(points, anchor, q=90):
  c = np.array([center(p.box) for p in points])
  d = np.linalg.norm(c - np.asarray(anchor), axis=1)
  return float(np.partition(d, int(round(q / 100 * (len(d) - 1))))[int(round(q / 100 * (len(d) - 1)))])

def speed(points):
  c = np.array([center(p.box) for p in points])
  dt = np.diff([p.timestamp_s for p in points])
  step = np.linalg.norm(np.diff(c, axis=0), axis=1)
  sizes = np.array([size(p.box) for p in points[1:]])
  return float(np.median(step / np.maximum(dt, 1e-6) / np.maximum(sizes, 1e-6)))

def present(inp, classes, min_px=0.0):
  i, out = inp.frame.sample_index, []
  for t in inp.tracks.values():
    if t.last_index != i: continue
    name, px = inp.feat(t)
    if name in classes and px >= min_px: out.append(t)
  return out