from percival.config.schema import TrackingCfg
from percival.contracts import Observation
from percival.tracking import Stitcher

def ob(i, tid, box, cls="person", gap_t=0.1):
  return Observation(run_id="r", clip_id="c", source_index=i * 2, sample_index=i, timestamp_s=i * gap_t, box=box, class_id=0, class_name=cls, confidence=0.9, detector_version="d", track_id=tid, tracker_version="t")

def feed(frames, **over):
  st, out = Stitcher(TrackingCfg(**over)), []
  for i, obs in enumerate(frames): out.append(st.process([ob(i, *o) for o in obs]))
  return out, st

def ids(out): return [[o.track_id for o in frame] for frame in out]
box = lambda x, y=100: (x, y, 40.0, 100.0)

def test_a_new_id_that_appears_nearby_right_after_one_vanishes_is_the_same_person():
  frames = [[(1, box(100 + 2 * i))] for i in range(20)] + [[] for _ in range(8)] + [[(2, box(100 + 2 * i))] for i in range(28, 50)]
  out, st = feed(frames)
  assert {o.track_id for f in out for o in f} == {1}
  late = [o for f in out for o in f if o.sample_index >= 28]
  assert late and all(o.raw_track_id == 2 for o in late) and all(o.raw_track_id is None for f in out[:20] for o in f)

def test_different_classes_are_never_merged():
  out, _ = feed([[(1, box(100), "person")]] * 10 + [[]] * 3 + [[(2, box(100), "backpack")]] * 5)
  assert {o.track_id for f in out for o in f} == {1, 2}

def test_gaps_longer_than_the_limit_and_far_jumps_are_not_merged():
  late, _ = feed([[(1, box(100))]] * 10 + [[]] * 30 + [[(2, box(100))]] * 5)
  far, _ = feed([[(1, box(100))]] * 10 + [[]] * 3 + [[(2, box(1500))]] * 5)
  assert {o.track_id for f in late for o in f} == {1, 2} and {o.track_id for f in far for o in f} == {1, 2}

def test_a_track_that_is_still_alive_is_never_absorbed():
  out, _ = feed([[(1, box(100))]] * 5 + [[(1, box(100)), (2, box(110))]] * 10)
  assert {o.track_id for f in out for o in f} == {1, 2}

def test_two_lost_tracks_equally_close_to_a_newcomer_leave_it_unmerged():
  out, _ = feed([[(1, box(100)), (2, box(130))]] * 10 + [[]] * 3 + [[(3, box(115))]] * 5)
  assert ids(out)[-1] == [3]

def test_two_newcomers_for_one_lost_track_give_it_to_the_closer_one_only():
  out, _ = feed([[(1, box(100))]] * 10 + [[]] * 3 + [[(3, box(110)), (4, box(190))]] * 5)
  assert sorted(ids(out)[-1]) == [1, 4]

def test_identities_chain_across_several_fragments():
  out, st = feed([[(1, box(100))]] * 10 + [[]] * 3 + [[(2, box(100))]] * 10 + [[]] * 3 + [[(3, box(100))]] * 10)
  assert {o.track_id for f in out for o in f} == {1} and {o.raw_track_id for f in out for o in f} == {None, 2, 3}

def test_frames_without_any_observation_are_passed_through():
  st = Stitcher(TrackingCfg())
  assert st.process([]) == [] and st.merges == 0