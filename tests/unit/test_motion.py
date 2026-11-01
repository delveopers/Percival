import numpy as np
from percival.config.schema import MotionCfg
from percival.contracts import Frame
from percival.motion import MotionStage
from percival.motion.boxes import merge_boxes, overlap_ratio

def frame(img, i=0): return Frame("r", "c", i * 2, i, i * 0.1, img, 1.0)
def black(): return np.zeros((240, 320, 3), np.uint8)

def with_rect(x=100, y=80, w=60, h=60):
  img = black()
  img[y:y + h, x:x + w] = 255
  return img

def test_new_object_becomes_exactly_one_candidate_at_the_right_place():
  st = MotionStage(MotionCfg())
  st.process(frame(black()))
  r = st.process(frame(with_rect(), 1))
  assert len(r.candidates) == 1 and not r.scene_change
  x, y, w, h = r.candidates[0].box
  assert abs(x - 100) <= 6 and abs(y - 80) <= 6 and abs(w - 60) <= 8 and r.candidates[0].score > 0.8

def test_global_brightness_jump_is_a_scene_change_not_fifty_boxes():
  st = MotionStage(MotionCfg())
  st.process(frame(black()))
  r = st.process(frame(np.full((240, 320, 3), 120, np.uint8), 1))
  assert r.scene_change and not r.candidates and r.changed_fraction > 0.4

def test_low_level_noise_produces_nothing():
  st = MotionStage(MotionCfg())
  st.process(frame(black()))
  noise = np.random.default_rng(0).integers(0, 8, (240, 320, 3), dtype=np.uint8)
  r = st.process(frame(noise, 1))
  assert not r.candidates and not r.scene_change

def test_fixed_reference_never_moves_but_running_average_does():
  fixed, running = MotionStage(MotionCfg(reference_mode="fixed")), MotionStage(MotionCfg(reference_mode="running_average"))
  for st in (fixed, running):
    st.process(frame(black()))
    for i in range(1, 6): st.process(frame(with_rect(), i))
  assert fixed.ref.max() == 0 and running.ref.max() > 0

def test_overlapping_boxes_merge_into_their_union_and_distant_ones_stay_apart():
  assert merge_boxes([(0, 0, 50, 50), (20, 20, 50, 50)], 0.3) == [(0, 0, 70, 70)]
  assert len(merge_boxes([(0, 0, 50, 50), (200, 200, 50, 50)], 0.3)) == 2
  assert overlap_ratio((0, 0, 10, 10), (10, 0, 10, 10)) == 0.0 and overlap_ratio((0, 0, 100, 100), (10, 10, 20, 20)) == 1.0