import numpy as np
from percival.config.schema import AnomalyCfg, TrackingCfg
from percival.contracts import Frame, Observation
from percival.tracking import TrackletBuilder
from percival.anomaly.features import label
from percival.behavior import BehaviorStates

def ob(i, tid, cls, box):
  return Observation(run_id="r", clip_id="c", source_index=i * 2, sample_index=i, timestamp_s=i * 0.1, box=box, class_id=0, class_name=cls, confidence=0.9, detector_version="v", track_id=tid, tracker_version="v")

def segments(scene, n, **over):
  cfg = AnomalyCfg(**{"stationary_seconds": 5, "loitering_seconds": 8, "owner_absent_seconds": 3, **over})
  b, bs, out = TrackletBuilder(TrackingCfg(max_lost_frames=30, min_track_length=3)), BehaviorStates(cfg, "r", "c"), []
  for i in range(n):
    obs = [ob(i, *s) for s in scene(i)]
    b.update(obs, i)
    out += bs.update(Frame("r", "c", i * 2, i, i * 0.1, np.zeros((1, 1, 3), np.uint8), 1.0), b.tracks)
  return out + bs.finish()

def stop_and_go(i):
  x = 15 * i if i < 30 else 450 if i < 80 else 450 + 15 * (i - 80)
  return [(1, "person", (x, 100, 40, 120))]

def test_a_person_who_walks_stops_and_walks_again_gets_a_timeline_of_states():
  segs = segments(stop_and_go, 140)
  assert [s.state for s in segs] == ["moving", "still", "moving"] and all(s.track_id == 1 and s.class_name == "person" for s in segs)
  assert segs[0].start_s <= 0.1 and 3.0 <= segs[1].start_s <= 4.5 and 8.0 <= segs[1].end_s <= 9.5
  assert segs[0].end_s == segs[1].start_s and segs[1].end_s == segs[2].start_s

def test_one_jolted_frame_does_not_change_the_state():
  segs = segments(lambda i: [(1, "person", (100 + (150 if i == 50 else 0), 100, 40, 120))], 100)
  assert [s.state for s in segs] == ["still"]

def test_a_track_that_disappears_closes_its_last_segment_at_the_last_sighting():
  segs = segments(lambda i: [(1, "person", (100, 100, 40, 120))] if i < 50 else [], 150)
  assert len(segs) == 1 and segs[0].state == "still" and 4.8 <= segs[0].end_s <= 5.0

def test_tiny_boxes_have_no_state():
  assert segments(lambda i: [(1, "person", (100, 100, 4, 8))], 80) == []

def test_people_and_vehicles_use_different_tolerances_for_the_same_sway():
  sway = lambda cls, box_h: (lambda i: [(1, cls, (100 + 30 * ((i // 10) % 2), 100, 80, box_h))])
  assert [s.state for s in segments(sway("person", 80), 100)] == ["still"]
  assert "moving" in [s.state for s in segments(sway("car", 80), 100)]

def test_the_class_label_follows_recent_sightings_not_the_whole_history():
  b = TrackletBuilder(TrackingCfg(label_window=10, min_track_length=3))
  for i in range(30): b.update([ob(i, 1, "backpack" if i < 20 else "handbag", (10, 10, 30, 30))], i)
  assert label(b.tracks[1]) == "handbag"
  tracklet = b.finish()[0]
  assert tracklet.class_name == "backpack" and [p.class_name for p in tracklet.points][:1] == ["backpack"] and tracklet.points[-1].class_name == "handbag"