import sys, types
import numpy as np
import pytest
from percival.config.schema import TrackerCfg
from percival.contracts import Detection
from percival.detection.tracker import Tracker, assign

def det(box, cls=0, conf=0.9):
  return Detection(run_id="r", clip_id="c", source_index=0, sample_index=0, timestamp_s=0.0, box=box, class_id=cls, class_name="person" if cls == 0 else "car", confidence=conf, detector_version="d")

class FakeNew:
  rows, seen = [], []
  def __init__(self, args): pass
  def update(self, results):
    mask = np.ones(len(results), dtype=bool)
    FakeNew.seen.append((len(results[mask]), results.conf.shape, results.xywh.shape, results.cls.shape))
    return np.array(FakeNew.rows, dtype=np.float32).reshape(len(FakeNew.rows), 8)

class FakeOld(FakeNew):
  def __init__(self, args, frame_rate=30): FakeNew.frame_rates.append(frame_rate)
FakeNew.frame_rates = []

def install(monkeypatch, cls):
  mod = types.ModuleType("ultralytics.trackers.byte_tracker")
  mod.BYTETracker = cls
  for name in ("ultralytics", "ultralytics.trackers"): monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
  monkeypatch.setitem(sys.modules, "ultralytics.trackers.byte_tracker", mod)
  FakeNew.rows, FakeNew.seen, FakeNew.frame_rates = [], [], []

@pytest.fixture(params=[FakeNew, FakeOld], ids=["new_signature", "old_signature"])
def tracker(request, monkeypatch):
  install(monkeypatch, request.param)
  return Tracker(TrackerCfg(), "t", 10)

A, B, C = (100, 100, 40, 80), (300, 120, 50, 90), (500, 50, 30, 60)

def test_old_signature_gets_neutral_frame_rate(monkeypatch):
  install(monkeypatch, FakeOld)
  Tracker(TrackerCfg(), "t", 12.5)
  assert FakeNew.frame_rates == [30]

def test_results_object_supports_boolean_indexing(tracker):
  tracker.update(None, [det(A), det(B)])
  assert FakeNew.seen == [(2, (2,), (2, 4), (2,))]

def test_constant_index_column_cannot_collapse_boxes(tracker):
  FakeNew.rows = [[102, 101, 141, 181, 5, 0.9, 0, 0], [301, 121, 351, 211, 6, 0.9, 0, 0], [499, 52, 531, 111, 7, 0.9, 0, 0]]
  obs = tracker.update(None, [det(A), det(B), det(C)])
  assert {o.track_id: o.box for o in obs} == {5: tuple(map(float, A)), 6: tuple(map(float, B)), 7: tuple(map(float, C))}
  assert len({o.box for o in obs}) == 3 and tracker.last_dropped == 0

def test_row_matching_no_detection_is_dropped(tracker):
  FakeNew.rows = [[102, 101, 141, 181, 5, 0.9, 0, 0], [800, 800, 840, 880, 9, 0.9, 0, 1]]
  obs = tracker.update(None, [det(A), det(B)])
  assert [o.track_id for o in obs] == [5] and tracker.last_dropped == 1

def test_class_mismatch_is_not_matched(tracker):
  FakeNew.rows = [[102, 101, 141, 181, 5, 0.9, 2, 0]]
  assert tracker.update(None, [det(A, cls=0)]) == [] and tracker.last_dropped == 1

def test_each_detection_is_used_once(tracker):
  FakeNew.rows = [[100, 100, 140, 180, 1, 0.9, 0, 0], [110, 105, 150, 185, 2, 0.9, 0, 0]]
  obs = tracker.update(None, [det(A)])
  assert [o.track_id for o in obs] == [1] and tracker.last_dropped == 1

def test_empty_inputs(tracker):
  assert tracker.update(None, []) == [] and tracker.last_dropped == 0
  FakeNew.rows = [[102, 101, 141, 181, 5, 0.9, 0, 0]]
  assert tracker.update(None, []) == []

def test_assign_prefers_best_overlap():
  rows = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
  dets = np.array([[0, 0, 10, 10]], dtype=np.float32)
  cls = np.zeros(2, dtype=np.int64)
  assert assign(rows, cls, dets, cls[:1]) == {0: 0}

try:
  import ultralytics.trackers.byte_tracker
  HAS_REAL = True
except ImportError:
  HAS_REAL = False

real = pytest.mark.skipif(not HAS_REAL, reason="ultralytics not installed")

def real_tracker():
  for name in [m for m in sys.modules if m.startswith("ultralytics")]: sys.modules.pop(name, None)
  return Tracker(TrackerCfg(), "t")

def scene(i, show_car=True):
  out = [det((50 + 12 * i, 100, 40, 80), conf=0.9), det((600 - 10 * i, 200, 40, 80), conf=0.85)]
  if show_car: out.append(det((300, 300, 90, 60), cls=2, conf=0.9))
  return out

@real
def test_real_bytetrack_keeps_three_identities_with_distinct_boxes():
  t, ids, per_frame = real_tracker(), set(), []
  for i in range(15):
    dets = scene(i)
    obs = t.update(None, dets)
    assert len(obs) == 3 and len({o.box for o in obs}) == 3 and t.last_dropped == 0
    assert {o.box for o in obs} == {d.box for d in dets}
    ids |= {o.track_id for o in obs}
  assert len(ids) == 3

def car_id(obs): return next(o.track_id for o in obs if o.class_name == "car")

@real
def test_real_bytetrack_recovers_identity_within_buffer_and_not_beyond():
  t = real_tracker()
  for i in range(5): first = car_id(t.update(None, scene(i)))
  for i in range(5, 25): t.update(None, scene(i, show_car=False))
  assert car_id(t.update(None, scene(25))) == first
  t = real_tracker()
  for i in range(5): first = car_id(t.update(None, scene(i)))
  for i in range(5, 50): t.update(None, scene(i, show_car=False))
  t.update(None, scene(50))
  assert car_id(t.update(None, scene(51))) != first