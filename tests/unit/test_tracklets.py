from percival.config.schema import TrackingCfg
from percival.contracts import Observation
from percival.tracking import TrackletBuilder

def ob(i, tid, cls="person", conf=0.8):
  return Observation(run_id="r", clip_id="c", source_index=i * 2, sample_index=i, timestamp_s=i * 0.2, box=(10.0, 20.0, 30.0, 40.0), class_id=0, class_name=cls, confidence=conf, detector_version="d", track_id=tid, tracker_version="t")

def test_long_track_is_emitted_when_it_goes_stale_and_short_ones_are_dropped():
  b, closed = TrackletBuilder(TrackingCfg(max_lost_frames=3, min_track_length=3)), []
  for i in range(10):
    seen = ([ob(i, 1, "backpack" if i == 2 else "person")] if i < 6 else []) + ([ob(i, 2)] if i < 2 else [])
    closed += b.update(seen, i)
  assert [t.track_id for t in closed] == [1]
  t = closed[0]
  assert t.class_name == "person" and len(t.points) == 6 and (t.start_index, t.end_index) == (0, 5) and abs(t.mean_confidence - 0.8) < 1e-9
  assert b.tracks == {} and b.finish() == []

def test_finish_flushes_tracks_that_are_still_open():
  b = TrackletBuilder(TrackingCfg(max_lost_frames=30, min_track_length=3))
  for i in range(5): b.update([ob(i, 9)], i)
  tail = b.finish()
  assert len(tail) == 1 and tail[0].track_id == 9 and not b.tracks
  tail[0].model_dump_json()

def test_gap_shorter_than_the_limit_keeps_one_tracklet():
  b = TrackletBuilder(TrackingCfg(max_lost_frames=10, min_track_length=3))
  for i in list(range(5)) + list(range(10, 15)): b.update([ob(i, 1)], i)
  assert len(b.finish()) == 1