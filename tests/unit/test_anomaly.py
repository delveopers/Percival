import math
import numpy as np
from percival.config.schema import AnomalyCfg, TrackingCfg
from percival.contracts import Frame, Observation
from percival.tracking import TrackletBuilder
from percival.anomaly import AnomalyEngine

CFG = dict(stationary_seconds=5, loitering_seconds=8, warmup_seconds=0, cooldown_seconds=1, crowd_count=3, crowd_min_seconds=1, min_confidence=0.3, owner_absent_seconds=3, speed_min_samples=20, speed_zscore=3)

def ob(i, tid, cls, box, conf=0.9):
  return Observation(run_id="r", clip_id="c", source_index=i * 2, sample_index=i, timestamp_s=i * 0.1, box=box, class_id=0, class_name=cls, confidence=conf, detector_version="v", track_id=tid, tracker_version="v")

def run(scene, n, flags=lambda i: False, finish=True, dets=None, **over):
  b, e, out = TrackletBuilder(TrackingCfg(max_lost_frames=30, min_track_length=3)), AnomalyEngine(AnomalyCfg(**{**CFG, **over}), "r", "c"), []
  for i in range(n):
    obs = [] if flags(i) else [ob(i, *s) for s in scene(i)]
    b.update(obs, i)
    raw = None if dets is None else [ob(i, *d) for d in dets(i)]
    out += e.update(Frame("r", "c", i * 2, i, i * 0.1, np.zeros((1, 1, 3), np.uint8), 1.0), obs, b.tracks, flags(i), raw)
  if finish: out += e.finish()
  return out, e

kinds = lambda out: sorted({a.kind for a in out})
car = lambda i: [(1, "car", (100 + i % 2, 100, 80, 50))]

def test_parked_car_is_flagged_once_after_the_required_time():
  out, _ = run(car, 80)
  assert kinds(out) == ["stationary_object"] and len(out) == 1
  assert 5.0 <= out[0].start_s <= 5.3 and out[0].track_ids == [1] and out[0].details["duration_s"] >= 5

def test_moving_car_is_not_stationary():
  out, _ = run(lambda i: [(1, "car", (100 + i * 20, 100, 80, 50))], 80)
  assert not out

def test_warmup_suppresses_events():
  out, _ = run(car, 80, warmup_seconds=1000)
  assert not out

def abandoned(stay):
  return lambda i: [(2, "backpack", (130, 200, 30, 30))] + ([(1, "person", (100, 100, 40, 120))] if (i < 40 or stay) else [])

def test_bag_is_abandoned_only_after_owner_leaves():
  out, _ = run(abandoned(False), 120)
  assert kinds(out) == ["abandoned_object"] and out[0].start_s >= 6.9 and out[0].track_ids == [2]

def test_bag_with_owner_nearby_is_not_abandoned():
  out, _ = run(abandoned(True), 120)
  assert not out

def test_pacing_person_is_loitering():
  out, _ = run(lambda i: [(1, "person", (100 + 200 * abs(math.sin(i * 0.2)), 100, 40, 120))], 120)
  assert kinds(out) == ["loitering"] and out[0].details["min_spread_px"] == 60.0

def test_person_standing_still_with_box_jitter_is_not_loitering():
  out, _ = run(lambda i: [(1, "person", (100 + 3 * (i % 2), 100 + 2 * (i % 3), 40, 120))], 150)
  assert not out

def test_small_sway_below_the_minimum_spread_is_not_loitering():
  out, _ = run(lambda i: [(1, "person", (100 + 40 * abs(math.sin(i * 0.2)), 100, 40, 120))], 120)
  assert not out

def test_person_walking_through_is_not_loitering():
  out, _ = run(lambda i: [(1, "person", (i * 10, 100, 40, 120))], 120)
  assert not out

def test_runner_among_walkers_is_a_speed_outlier():
  def scene(i):
    s = [(k, "person", (i * 15 + k * 500, 50 + k * 150, 40, 100)) for k in (1, 2, 3)]
    return s + [(9, "person", (i * 80, 900, 40, 100))] if i >= 60 else s
  out, _ = run(scene, 100, crowd_count=10)
  speed = [a for a in out if a.kind == "speed_outlier"]
  assert speed and all(a.track_ids == [9] for a in speed) and speed[0].details["zscore"] >= 3
  assert 6.4 <= speed[0].start_s <= 6.7 and speed[0].details["sustained_s"] >= 1.0

def people(n): return [(k, "person", (100 * k, 100, 40, 100)) for k in range(1, n + 1)]

def test_steady_crowd_is_baseline_not_an_anomaly():
  out, _ = run(lambda i: people(10), 100, crowd_count=8)
  assert not out

def test_sudden_surge_is_flagged_from_its_onset():
  out, _ = run(lambda i: people(5 if i < 60 else 15), 100, crowd_count=8)
  assert kinds(out) == ["crowd"] and len(out) == 1
  assert 6.0 <= out[0].start_s <= 6.1 and out[0].details["count"] == 15 and abs(out[0].details["baseline"] - 5) < 1

def test_small_increase_is_not_a_surge():
  out, _ = run(lambda i: people(10 if i < 60 else 12), 100, crowd_count=8)
  assert not out

def test_surge_ends_when_baseline_adapts_to_the_new_normal():
  out, _ = run(lambda i: people(5 if i < 60 else 15), 300, crowd_count=8, crowd_baseline_seconds=2)
  assert kinds(out) == ["crowd"] and out[0].end_s < 12

def test_brief_speed_burst_is_ignored():
  def scene(i):
    s = [(k, "person", (i * 15 + k * 500, 50 + k * 150, 40, 100)) for k in (1, 2, 3)]
    x = 15 * i if i < 60 else 900 + 80 * (i - 60) if i < 65 else 1300 + 15 * (i - 65)
    return s + [(9, "person", (x, 900, 40, 100))]
  out, _ = run(scene, 100)
  assert not [a for a in out if a.kind == "speed_outlier"]

def test_scene_change_pauses_open_events():
  out, e = run(car, 130, flags=lambda i: 80 <= i < 100, finish=False)
  assert out == [] and len(e.active) == 1 and len(e.finish()) == 1

def test_event_closes_after_cooldown_when_object_leaves():
  out, e = run(lambda i: car(i) if i < 70 else [], 100, finish=False)
  assert len(out) == 1 and out[0].end_s < 7.1 and not e.active

def test_sustained_surge_stays_flagged_inside_the_baseline_window():
  out, e = run(lambda i: people(5 if i < 60 else 15), 200, crowd_count=8, finish=False)
  assert out == [] and len(e.active) == 1 and e.active[0].end_s >= 19.8

def test_count_hovering_near_the_threshold_does_not_break_a_surge():
  scene = lambda i: people(5 if i < 60 else 8 if (i - 60) % 8 == 7 else 12)
  out, _ = run(scene, 120, crowd_count=6)
  assert kinds(out) == ["crowd"] and len(out) == 1 and 6.0 <= out[0].start_s <= 6.1 and out[0].end_s >= 11.5

def test_crowd_counts_raw_detections_not_tracked_objects():
  tracked = lambda i: people(5)
  raw = lambda i: people(5 if i < 60 else 15)
  out, _ = run(tracked, 100, dets=raw, crowd_count=8)
  assert kinds(out) == ["crowd"] and out[0].details["count"] == 15 and 6.0 <= out[0].start_s <= 6.1

def test_weak_detections_below_the_crowd_confidence_are_not_counted():
  weak = lambda i: [(k, "person", (100 * k, 100, 40, 100), 0.2) for k in range(1, 6 if i < 60 else 16)]
  out, _ = run(lambda i: people(5), 100, dets=weak, crowd_count=8)
  assert not out