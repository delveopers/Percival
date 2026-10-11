# clipInfo, Frame, ChangeCandidates


from pathlib import Path
from pydantic import BaseModel, ConfigDict

class ClipInfo(BaseModel):
  model_config = ConfigDict(frozen=True)
  clip_id: str
  path: Path
  fps: float; duration_s: float
  frame_count: int
  width: int; height: int

class Frame(BaseModel):
  frame_height = int
  frame_width = int

class ChangeCandidates(BaseModel):
  pass