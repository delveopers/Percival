from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, ConfigDict
import numpy as np

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
  box: tuple[int, int, int, int]
  score: float
  area: int

@dataclass(slote=True)
class MotionResult:
  candidates: list[ChangeCandidates]
  scene_change: bool
  diff_map: Optional[np.ndarray] = None
