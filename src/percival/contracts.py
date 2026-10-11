from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel, ConfigDict
import numpy as np

Box = tuple[float, float, float, float]

class ClipInfo(BaseModel):
  model_config = ConfigDict(frozen=True)
  clip_id: str
  path: Path
  fps: float; duration_s: float
  frame_count: int
  width: int; height: int

@dataclass(frozen=True, slots=True)
class Frame(BaseModel):
  run_id: str; clip_id: str
  source_index: int; sample_index: int
  timestamp_s: float
  image: np.ndarray
  scale: float

class ChangeCandidates(BaseModel):
  model_config = ConfigDict(frozen=True)
  run_id: str; clip_id: str
  source_index: int; sample_index: int
  timestamp_s: float
  box: Box
  score: float
  area: int

@dataclass(slote=True)
class MotionResult:
  candidates: list[ChangeCandidates]
  scene_change: bool
  diff_map: Optional[np.ndarray] = None

class Detection:
  model_config = ConfigDict(frozen=True)
  run_id: str; clip_id: str
  source_index: int; sample_index: int
  timestamp_s: float
  box: Box
  class_id: int; class_name: str
  confidence: float; detector_version: str

class Observation(Detection):
  track_id: int
  track_version: str

class TrackPoint(BaseModel):
  model_config = ConfigDict(frozen=True)
  source_index: int; sample_index: int
  timestamp_s: float
  box: Box
  confidence: float

class Tracklet(BaseModel):
  model_config = ConfigDict(frozen=True)
  run_id: str; clip_id: str
  track_id: int
  class_name: str
  start_s: float; end_s: float
  start_index: int; end_index: int
  mean_confidence: float
  points: list[TrackPoint]
  detector_version: str; tracker_version: str

@dataclass(slots=True)
class FrameContext:
  frame: Frame
  motion: Optional[MotionResult] = None
  observation: list = field(default_factory=list)
  anomalies: list = field(default_factory=list)