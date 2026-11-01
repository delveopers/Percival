import math
import numpy as np
import pytest
from percival.config.schema import IngestionCfg
from percival.ingestion import scan_clips
from percival.preprocessing import iter_frames, resize_to_width

@pytest.fixture
def clip(project):
  project.video("v.mp4", frames=40)
  return scan_clips(project.cfg())[0]

@pytest.mark.parametrize("n", [1, 2, 3])
def test_sampling_keeps_every_nth_frame_with_deterministic_timestamps(clip, n):
  frames = list(iter_frames(clip, IngestionCfg(sample_every_n_frames=n, resize_width=400), "r"))
  assert len(frames) == math.ceil(40 / n) and [f.source_index for f in frames] == list(range(0, 40, n))
  assert [f.sample_index for f in frames] == list(range(len(frames)))
  assert all(abs(f.timestamp_s - f.source_index / 20) < 1e-9 for f in frames)
  assert frames[0].image.shape[1] == 400 and frames[0].scale == 0.625 and frames[0].run_id == "r" and frames[0].clip_id == "v"

def test_max_frames_counts_sampled_frames(clip):
  assert len(list(iter_frames(clip, IngestionCfg(sample_every_n_frames=2, max_frames=5), "r"))) == 5

def test_breaking_out_early_is_safe(clip):
  g = iter_frames(clip, IngestionCfg(sample_every_n_frames=1), "r")
  next(g)
  g.close()

def test_resize_never_upscales_and_keeps_aspect_ratio():
  small, scale = resize_to_width(np.zeros((100, 200, 3), np.uint8), 640)
  assert small.shape == (100, 200, 3) and scale == 1.0
  big, scale = resize_to_width(np.zeros((480, 800, 3), np.uint8), 400)
  assert big.shape[:2] == (240, 400) and scale == 0.5