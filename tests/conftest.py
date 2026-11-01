import copy
from pathlib import Path
import cv2
import numpy as np
import pytest
import yaml
from percival.config import load_config

BASE = {
  "app": {"run": {"name": "t", "seed": 1}, "paths": {"input_dir": "data/samples", "output_dir": "data/exports", "extensions": [".mp4"], "recursive": False}, "logging": {"level": "WARNING", "format": "json", "file": None}, "output": {"save_annotated_video": False, "save_observations": True, "observations_format": "jsonl", "show_window": False, "save_detections": True}},
  "processing": {"ingestion": {"sample_every_n_frames": 2, "resize_width": 320, "queue_size": 8, "max_frames": None}, "motion": {"enabled": True, "reference_mode": "running_average", "background_alpha": 0.02, "diff_threshold": 30, "morph_kernel": 5, "min_box": [10, 10], "min_contour_points": 5, "merge_overlap": 0.3, "scene_change_fraction": 0.4, "heatmap_alpha": 0.4}, "tracking": {"enabled": True, "max_lost_frames": 45, "min_track_length": 5}, "anomaly": {"stationary_seconds": 2, "loitering_seconds": 3, "owner_absent_seconds": 1, "warmup_seconds": 0, "cooldown_seconds": 1, "min_confidence": 0.3}},
  "models": {"detector": {"name": "yolo", "weights": "x.pt", "device": "auto", "imgsz": 640, "conf": 0.35, "iou": 0.5, "classes": ["car", "person"], "half": False}, "tracker": {"name": "bytetrack", "track_high_thresh": 0.5, "track_low_thresh": 0.1, "new_track_thresh": 0.6, "track_buffer": 30, "match_thresh": 0.8}, "version": {"detector": "dv", "tracker": "tv", "motion": "mv"}},
}

def merge(base, extra):
  for k, v in extra.items(): base[k] = merge(base.get(k, {}), v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
  return base

class Project:
  def __init__(self, root):
    self.root = Path(root)
    (self.root / "configs").mkdir()
    (self.root / "data" / "samples").mkdir(parents=True)
    self.write_config({})

  def write_config(self, overrides):
    docs = merge(copy.deepcopy(BASE), overrides)
    for name, doc in docs.items(): (self.root / "configs" / f"{name}.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")

  def cfg(self): return load_config(self.root / "configs")

  def video(self, name="clip.mp4", frames=120, fps=20, rect_from=20, size=(640, 480), folder=None, flash=None):
    path = (folder or self.root / "data" / "samples") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
      f = np.zeros((size[1], size[0], 3), np.uint8)
      if i >= rect_from: cv2.rectangle(f, (100, 100), (200, 180), (255, 255, 255), -1)
      if flash and flash[0] <= i < flash[1]: f[:] = 200
      w.write(f)
    w.release()
    return path

@pytest.fixture
def project(tmp_path): return Project(tmp_path)