import math
import numpy as np
from percival.config.schema import RULE_NAMES, AnomalyCfg, TrackingCfg
from percival.contracts import Frame, Observation
from percival.tracking import TrackletBuilder
from percival.anomaly import AnomalyEngine
from percival.behavior import BehaviorStates

CFG = dict(stationary_seconds=5, loitering_seconds=8, dwell_seconds=3, warmup_seconds=0, cooldown_seconds=1, surge_min_seconds=1, density_min_seconds=1, min_confidence=0.3, owner_absent_seconds=3, speed_min_samples=20, speed_zscore=3)

def ob(i, tid, cls, box, conf=0.9):
  return Observation(run_id="r", clip_id="c", source_index=i * 2, sample_index=i, timestamp_s=i * 0.1, box=box, class_id=0, class_name=cls, confidence=conf, detector_version="v", track_id=tid, tracker_version="v")

def run(scene, n, rules=None, flags=lambda i: False, finish=True, dets=None, **over):
  cfg = AnomalyCfg(**{**CFG, "rules": list(rules or RULE_NAMES), **over})
  b, bs, e, out = TrackletBuilder(TrackingCfg(max_lost_frames=30, min_track_length=3)), BehaviorStates(cfg, "r", "c"), AnomalyEngine(cfg, "r", "c"), []
  for i in range(n):
    obs = [] if flags(i) else [ob(i, *s) for s in scene(i)]
    raw = None if dets is None else [ob(i, *d) for d in dets(i)]
    frame = Frame("r", "c", i * 2, i, i * 0.1, np.zeros((1, 1, 3), np.uint8), 1.0)
    b.update(obs, i)
    bs.update(frame, b.tracks)
    out += e.update(frame, obs, b.tracks, flags(i), raw, bs.states)
  if finish: out += e.finish()
  return out, e

kinds = lambda out: sorted({a.kind for a in out})
car = lambda i: [(1, "car", (100 + i % 2, 100, 80, 50))]
walker = lambda x0, step: (lambda i: [(1, "person", (x0 + step * i, 100, 40, 120))])

def row(n, gap=60, x0=0, y=100, h=100): return [(x0 + gap * k, y, 40, h) for k in range(n)]
def crowd_of(boxes, conf=0.9): return [(k + 1, "person", b, conf) for k, b in enumerate(boxes)]

def test_parked_car_is_flagged_once_and_dated_from_when_it_stopped():
  out, _ = run(car, 80, ["stationary"])
  assert kinds(out) == ["stationary_object"] and len(out) == 1
  assert out[0].start_s <= 0.3 and out[0].end_s >= 7.5 and out[0].details["duration_s"] >= 5 and out[0].track_ids == [1]

def test_moving_car_is_not_stationary():
  out, _ = run(lambda i: [(1, "car", (100 + i * 20, 100, 80, 50))], 80, ["stationary"])
  assert not out

def test_warmup_suppresses_events():
  out, _ = run(car, 80, ["stationary"], warmup_seconds=1000)
  assert not out

def test_a_single_jolt_does_not_break_a_stationary_object():
  out, _ = run(lambda i: [(1, "car", (100 + (60 if i == 40 else 0), 100, 80, 50))], 80, ["stationary"])
  assert len(out) == 1 and out[0].end_s >= 7.5

def test_objects_below_the_minimum_box_size_are_ignored_by_the_shape_rules():
  out, _ = run(lambda i: [(1, "car", (100, 100, 8, 6))], 80, ["stationary"])
  assert not out

def abandoned(stay):
  return lambda i: [(2, "backpack", (130, 200, 30, 30))] + ([(1, "person", (100, 100, 40, 120))] if (i < 40 or stay) else [])

def test_bag_is_abandoned_only_after_owner_leaves():
  out, _ = run(abandoned(False), 120, ["abandoned"])
  assert kinds(out) == ["abandoned_object"] and out[0].start_s >= 6.9 and out[0].track_ids == [2]

def test_bag_with_owner_nearby_is_not_abandoned():
  out, _ = run(abandoned(True), 120, ["abandoned"])
  assert not out

def test_dwelling_lasts_exactly_as_long_as_the_person_stays():
  scene = lambda i: [(1, "person", (100 + (0 if i < 80 else 15 * (i - 80)), 100, 40, 120))]
  out, _ = run(scene, 140, ["dwell"])
  assert kinds(out) == ["dwell"] and len(out) == 1 and out[0].start_s <= 0.3 and 7.8 <= out[0].end_s <= 9.0

def test_a_person_shifting_their_weight_is_still_dwelling():
  out, _ = run(lambda i: [(1, "person", (100 + 20 * ((i // 10) % 2), 100, 40, 120))], 100, ["dwell"])
  assert kinds(out) == ["dwell"]

def test_walking_is_not_dwelling_and_short_stops_are_ignored():
  assert not run(walker(0, 10), 100, ["dwell"])[0]
  assert not run(lambda i: [(1, "person", (100 + (0 if i < 20 else 15 * (i - 20)), 100, 40, 120))], 100, ["dwell"])[0]

def test_pacing_person_is_loitering():
  out, _ = run(lambda i: [(1, "person", (100 + 200 * abs(math.sin(i * 0.2)), 100, 40, 120))], 120, ["loitering"])
  assert kinds(out) == ["loitering"] and out[0].details["min_spread_px"] == 60.0

def test_standing_with_box_jitter_is_dwelling_not_loitering():
  scene = lambda i: [(1, "person", (100 + 3 * (i % 2), 100 + 2 * (i % 3), 40, 120))]
  assert not run(scene, 150, ["loitering"])[0]
  assert kinds(run(scene, 150, ["dwell"])[0]) == ["dwell"]

def test_small_sway_is_not_loitering_and_walking_through_is_not_loitering():
  assert not run(lambda i: [(1, "person", (100 + 40 * abs(math.sin(i * 0.2)), 100, 40, 120))], 120, ["loitering"])[0]
  assert not run(walker(0, 15), 120, ["loitering"])[0]

def test_tiny_pacing_person_is_ignored():
  assert not run(lambda i: [(1, "person", (100 + 20 * abs(math.sin(i * 0.2)), 100, 4, 8))], 120, ["loitering"])[0]

def test_runner_among_walkers_is_a_speed_outlier_dated_from_its_onset():
  def scene(i):
    s = [(k, "person", (i * 15 + k * 500, 50 + k * 150, 40, 100)) for k in (1, 2, 3)]
    return s + [(9, "person", (i * 80, 900, 40, 100))] if i >= 60 else s
  out, _ = run(scene, 100, ["speed"])
  speed = [a for a in out if a.kind == "speed_outlier"]
  assert speed and all(a.track_ids == [9] for a in speed) and speed[0].details["zscore"] >= 3
  assert 6.4 <= speed[0].start_s <= 6.7 and speed[0].details["sustained_s"] >= 1.0

def test_brief_speed_burst_is_ignored():
  def scene(i):
    s = [(k, "person", (i * 15 + k * 500, 50 + k * 150, 40, 100)) for k in (1, 2, 3)]
    x = 15 * i if i < 60 else 900 + 80 * (i - 60) if i < 65 else 1300 + 15 * (i - 65)
    return s + [(9, "person", (x, 900, 40, 100))]
  assert not run(scene, 100, ["speed"])[0]

def test_three_close_people_are_a_group():
  out, _ = run(lambda i: crowd_of(row(3)), 40, ["density"])
  assert kinds(out) == ["group"] and out[0].start_s <= 0.2 and out[0].details["in_cluster"] == 3

def test_two_people_or_scattered_people_are_neither_group_nor_crowd():
  assert not run(lambda i: crowd_of(row(2)), 40, ["density"])[0]
  assert not run(lambda i: crowd_of(row(5, gap=500)), 40, ["density"])[0]

def test_a_big_tight_cluster_is_a_crowd():
  out, _ = run(lambda i: crowd_of(row(9)), 40, ["density"])
  assert kinds(out) == ["crowd"] and out[0].details["via"] == "cluster"

def test_several_groups_in_proximity_form_a_crowd_but_distant_groups_do_not():
  near = lambda i: crowd_of(row(4) + row(4, x0=480))
  far = lambda i: crowd_of(row(4) + row(4, x0=3000))
  out, _ = run(near, 40, ["density"])
  assert kinds(out) == ["crowd"] and out[0].details["via"] == "nearby_groups" and out[0].details["groups"] == 2
  out, _ = run(far, 40, ["density"])
  assert kinds(out) == ["group"]

def test_many_people_anywhere_in_view_are_a_crowd():
  out, _ = run(lambda i: crowd_of(row(25, gap=400)), 40, ["density"])
  assert kinds(out) == ["crowd"] and out[0].details["via"] == "total"

def test_a_group_that_grows_becomes_a_crowd_and_the_label_hands_over_cleanly():
  out, _ = run(lambda i: crowd_of(row(4 if i < 40 else 10)), 90, ["density"])
  assert [a.kind for a in sorted(out, key=lambda a: a.start_s)] == ["group", "crowd"]
  group, crowd = sorted(out, key=lambda a: a.start_s)
  assert group.end_s <= crowd.start_s + 1e-9 and 3.9 <= crowd.start_s <= 4.1

def test_crowd_status_survives_the_count_hovering_around_the_threshold():
  out, _ = run(lambda i: crowd_of(row(8 if (i // 3) % 2 == 0 else 7)), 60, ["density"])
  assert [a.kind for a in out] == ["crowd"]

def test_tiny_or_weak_detections_are_not_counted_as_people():
  assert not run(lambda i: crowd_of([(60 * k, 100, 4, 6) for k in range(10)]), 40, ["density"])[0]
  assert not run(lambda i: crowd_of(row(10), conf=0.2), 40, ["density"])[0]

def test_density_follows_raw_detections_even_if_the_tracker_misses_people():
  out, _ = run(lambda i: crowd_of(row(3)), 40, ["density"], dets=lambda i: crowd_of(row(10)))
  assert kinds(out) == ["crowd"]

def people(n): return [(k, "person", (100 * k, 100, 40, 100)) for k in range(1, n + 1)]

def test_steady_crowd_is_not_a_surge():
  assert not run(lambda i: people(10), 100, ["surge"], surge_count=8)[0]

def test_sudden_surge_is_flagged_from_its_onset():
  out, _ = run(lambda i: people(5 if i < 60 else 15), 100, ["surge"], surge_count=8)
  assert kinds(out) == ["crowd_surge"] and len(out) == 1
  assert 6.0 <= out[0].start_s <= 6.1 and out[0].details["count"] == 15 and abs(out[0].details["baseline"] - 5) < 1

def test_small_increase_is_not_a_surge():
  assert not run(lambda i: people(10 if i < 60 else 12), 100, ["surge"], surge_count=8)[0]

def test_surge_ends_when_baseline_adapts_to_the_new_normal():
  out, _ = run(lambda i: people(5 if i < 60 else 15), 300, ["surge"], surge_count=8, surge_baseline_seconds=2)
  assert kinds(out) == ["crowd_surge"] and out[0].end_s < 12

def test_sustained_surge_stays_flagged_inside_the_baseline_window():
  out, e = run(lambda i: people(5 if i < 60 else 15), 200, ["surge"], surge_count=8, finish=False)
  assert out == [] and len(e.active) == 1 and e.active[0].end_s >= 19.8

def test_count_hovering_near_the_surge_threshold_does_not_break_it():
  out, _ = run(lambda i: people(5 if i < 60 else 8 if (i - 60) % 8 == 7 else 12), 120, ["surge"], surge_count=6)
  assert kinds(out) == ["crowd_surge"] and len(out) == 1 and 6.0 <= out[0].start_s <= 6.1 and out[0].end_s >= 11.5

def test_surge_counts_raw_detections_and_ignores_weak_ones():
  out, _ = run(lambda i: people(5), 100, ["surge"], dets=lambda i: people(5 if i < 60 else 15), surge_count=8)
  assert kinds(out) == ["crowd_surge"] and out[0].details["count"] == 15
  weak = lambda i: [(k, "person", (100 * k, 100, 40, 100), 0.2) for k in range(1, 6 if i < 60 else 16)]
  assert not run(lambda i: people(5), 100, ["surge"], dets=weak, surge_count=8)[0]

def test_scene_change_pauses_open_events():
  out, e = run(car, 130, ["stationary"], flags=lambda i: 80 <= i < 100, finish=False)
  assert out == [] and len(e.active) == 1 and len(e.finish()) == 1

def test_event_closes_after_cooldown_when_object_leaves():
  out, e = run(lambda i: car(i) if i < 70 else [], 100, ["stationary"], finish=False)
  assert len(out) == 1 and out[0].end_s < 7.1 and not e.active