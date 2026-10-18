import sys, types
import numpy as np
import pytest
from percival.config.schema import TrackerCfg
from percival.contracts import Detection
from percival.detection.tracker import Tracker, assign

def det(box, cls=0, conf=0.9):
  return Detection(run_id="r", clip_id="c", source_index=0, sample_index=0, timestamp_s=0.0, box=box, class_id=cls, class_name="person" if cls == 0 else "car", confidence=conf, detector_version="d")

class FakeByteTrack:
  rows = []
  def __init__(self, args, frame_rate): pass
  def update(self, results): return np.array(FakeByteTrack.rows, dtype=np.float32).reshape(len(FakeByteTrack.rows), 8)

@pytest.fixture
def tracker(monkeypatch):
  mod = types.ModuleType("ultralytics.trackers.byte_tracker")
  mod.BYTETracker = FakeByteTrack
  for name in ("ultralytics", "ultralytics.trackers"): monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
  monkeypatch.setitem(sys.modules, "ultralytics.trackers.byte_tracker", mod)
  FakeByteTrack.rows = []
  return Tracker(TrackerCfg(), "t", 10)

A, B, C = (100, 100, 40, 80), (300, 120, 50, 90), (500, 50, 30, 60)

def test_constant_index_column_cannot_collapse_boxes(tracker):
  FakeByteTrack.rows = [[102, 101, 141, 181, 5, 0.9, 0, 0], [301, 121, 351, 211, 6, 0.9, 0, 0], [499, 52, 531, 111, 7, 0.9, 0, 0]]
  obs = tracker.update(None, [det(A), det(B), det(C)])
  assert {o.track_id: o.box for o in obs} == {5: tuple(map(float, A)), 6: tuple(map(float, B)), 7: tuple(map(float, C))}
  assert len({o.box for o in obs}) == 3 and tracker.last_dropped == 0

def test_row_matching_no_detection_is_dropped(tracker):
  FakeByteTrack.rows = [[102, 101, 141, 181, 5, 0.9, 0, 0], [800, 800, 840, 880, 9, 0.9, 0, 1]]
  obs = tracker.update(None, [det(A), det(B)])
  assert [o.track_id for o in obs] == [5] and tracker.last_dropped == 1

def test_class_mismatch_is_not_matched(tracker):
  FakeByteTrack.rows = [[102, 101, 141, 181, 5, 0.9, 2, 0]]
  assert tracker.update(None, [det(A, cls=0)]) == [] and tracker.last_dropped == 1

def test_each_detection_is_used_once(tracker):
  FakeByteTrack.rows = [[100, 100, 140, 180, 1, 0.9, 0, 0], [110, 105, 150, 185, 2, 0.9, 0, 0]]
  obs = tracker.update(None, [det(A)])
  assert [o.track_id for o in obs] == [1] and tracker.last_dropped == 1

def test_empty_inputs(tracker):
  assert tracker.update(None, []) == [] and tracker.last_dropped == 0
  FakeByteTrack.rows = [[102, 101, 141, 181, 5, 0.9, 0, 0]]
  assert tracker.update(None, []) == []

def test_assign_prefers_best_overlap():
  rows = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
  dets = np.array([[0, 0, 10, 10]], dtype=np.float32)
  cls = np.zeros(2, dtype=np.int64)
  assert assign(rows, cls, dets, cls[:1]) == {0: 0}