from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal, Any
from pydantic import BaseModel, ConfigDict
import numpy as np

Box = tuple[float, float, float, float]
AnomalyKind = Literal["stationary_object", "abandoned_object", "loitering", "speed_outlier", "crowd"]

class ClipInfo(BaseModel):
  model_config = ConfigDict(frozen=True)
  clip_id: str
  path: Path
  fps: float; duration_s: float
  frame_count: int
  width: int; height: int

@dataclass(frozen=True, slots=True)
class Frame:
  run_id: str; clip_id: str
  source_index: int; sample_index: int
  timestamp_s: float
  image: np.ndarray
  scale: float

class ChangeCandidate(BaseModel):
  model_config = ConfigDict(frozen=True)
  run_id: str; clip_id: str
  source_index: int; sample_index: int
  timestamp_s: float
  box: tuple[int, int, int, int]
  score: float
  area: int

@dataclass(slots=True)
class MotionResult:
  candidates: list[ChangeCandidate]
  scene_change: bool
  changed_fraction: float
  diff_map: Optional[np.ndarray] = None

class Detection(BaseModel):
  model_config = ConfigDict(frozen=True)
  run_id: str; clip_id: str
  source_index: int; sample_index: int
  timestamp_s: float
  box: Box
  class_id: int; class_name: str
  confidence: float; detector_version: str

class Observation(Detection):
  track_id: int
  tracker_version: str

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

class Anomaly(BaseModel):
  model_config = ConfigDict(frozen=True)
  run_id: str; clip_id: str
  kind: AnomalyKind
  confidence: float
  start_s: float; end_s: float
  start_index: int; end_index: int
  track_ids: list[int]
  class_names: list[str]
  box: Box
  details: dict[str, Any]

@dataclass(slots=True)
class FrameContext:
  frame: Frame
  motion: Optional[MotionResult] = None
  observation: list = field(default_factory=list)
  anomalies: list = field(default_factory=list)