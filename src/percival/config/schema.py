from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class Section(BaseModel):
  model_config = ConfigDict(extra="forbid", frozen=True)

class RunCfg(Section):
  name: str
  seed: int = 42

class PathsCfg(Section):
  input_dir: Path
  output_dir: Path
  extensions: tuple[str, ...]
  recursive: bool = False

  @field_validator("extensions")
  @classmethod
  def check_ext(cls, v):
    if not v or any(not e.startswith(".") for e in v): raise ValueError("extensions must be non-empty and start with '.'")
    return tuple(e.lower() for e in v)

class LoggingCfg(Section):
  level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
  format: Literal["json", "text"] = "json"
  file: Optional[Path] = None

class OutputCfg(Section):
  save_annotated_video: bool = True
  save_observations: bool = True
  observations_format: Literal["jsonl"] = "jsonl"
  show_window: bool = False

class IngestionCfg(Section):
  sample_every_n_frames: int = Field(2, ge=1)
  resize_width: int = Field(640, gt=0)
  queue_size: int = Field(32, ge=1)
  max_frames: Optional[int] = Field(None, ge=1)

class MotionCfg(Section):
  enabled: bool = True
  reference_mode: Literal["running_average", "fixed"] = "running_average"
  background_alpha: float = Field(0.02, gt=0, le=1)
  diff_threshold: int = Field(30, ge=0, le=255)
  morph_kernel: int = Field(5, ge=1)
  min_box: tuple[int, int] = (20, 20)
  min_contour_points: int = Field(5, ge=1)
  merge_overlap: float = Field(0.3, ge=0, le=1)
  scene_change_fraction: float = Field(0.4, gt=0, le=1)
  heatmap_alpha: float = Field(0.4, ge=0, le=1)

  @field_validator("morph_kernel")
  @classmethod
  def check_kernel(cls, v):
    if v % 2 == 0: raise ValueError("morph_kernel must be odd")
    return v

  @field_validator("min_box")
  @classmethod
  def check_box(cls, v):
    if min(v) < 1: raise ValueError("min_box values must be positive")
    return v

class TrackingCfg(Section):
  enabled: bool = True
  max_lost_frames: int = Field(30, ge=0)
  min_track_length: int = Field(5, ge=1)

class AnomalyCfg(Section):
  enabled: bool = True
  stationary_seconds: float = Field(20, gt=0)
  loitering_seconds: float = Field(30, gt=0)
  speed_zscore: float = Field(3.0, gt=0)
  crowd_count: int = Field(8, ge=1)
  warmup_frames: int = Field(150, ge=0)
  min_confidence: float = Field(0.5, ge=0, le=1)

  @model_validator(mode="after")
  def check_times(self):
    if self.loitering_seconds < self.stationary_seconds: raise ValueError("loitering_seconds must be >= stationary_seconds")
    return self

class DetectorCfg(Section):
  name: Literal["yolo"] = "yolo"
  weights: str
  device: str = "auto"
  imgsz: int = Field(640, gt=0)
  conf: float = Field(0.35, ge=0, le=1)
  iou: float = Field(0.5, ge=0, le=1)
  classes: tuple[str, ...]
  half: bool = False

class TrackerCfg(Section):
  name: Literal["bytetrack"] = "bytetrack"
  track_high_thresh: float = Field(0.5, ge=0, le=1)
  track_low_thresh: float = Field(0.1, ge=0, le=1)
  new_track_thresh: float = Field(0.6, ge=0, le=1)
  track_buffer: int = Field(30, ge=1)
  match_thresh: float = Field(0.8, ge=0, le=1)

  @model_validator(mode="after")
  def check_thresh(self):
    if self.track_low_thresh >= self.track_high_thresh: raise ValueError("track_low_thresh must be < track_high_thresh")
    return self

class VersionCfg(Section):
  detector: str; tracker: str; motion: str

class AppFile(Section):
  run: RunCfg
  paths: PathsCfg
  logging: LoggingCfg
  output: OutputCfg

class ProcessingFile(Section):
  ingestion: IngestionCfg
  motion: MotionCfg
  tracking: TrackingCfg
  anomaly: AnomalyCfg

class ModelsFile(Section):
  detector: DetectorCfg
  tracker: TrackerCfg
  version: VersionCfg

class AppConfig(Section):
  run: RunCfg
  paths: PathsCfg
  logging: LoggingCfg
  output: OutputCfg
  ingestion: IngestionCfg
  motion: MotionCfg
  tracking: TrackingCfg
  anomaly: AnomalyCfg
  detector: DetectorCfg
  tracker: TrackerCfg
  version: VersionCfg