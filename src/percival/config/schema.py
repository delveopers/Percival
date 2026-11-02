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
  save_detections: bool = True

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
  label_window: int = Field(15, ge=1)

RULE_NAMES = ("stationary", "dwell", "abandoned", "loitering", "speed", "density", "surge")

class AnomalyCfg(Section):
  enabled: bool = True
  rules: tuple[str, ...] = RULE_NAMES
  warmup_seconds: float = Field(12.0, ge=0)
  min_confidence: float = Field(0.5, ge=0, le=1)
  cooldown_seconds: float = Field(5.0, ge=0)
  rule_min_box_px: float = Field(10.0, ge=0)
  state_window_seconds: float = Field(2.0, gt=0)
  state_min_seconds: float = Field(1.0, ge=0)
  still_min_px: float = Field(4.0, ge=0)
  stationary_seconds: float = Field(20, gt=0)
  stationary_radius: float = Field(0.25, gt=0)
  stationary_classes: tuple[str, ...] = ("car", "bus", "truck", "motorcycle", "bicycle")
  dwell_seconds: float = Field(5.0, gt=0)
  dwell_radius: float = Field(0.5, gt=0)
  dwell_classes: tuple[str, ...] = ("person",)
  abandoned_classes: tuple[str, ...] = ("backpack", "handbag", "suitcase")
  owner_distance: float = Field(1.5, gt=0)
  owner_absent_seconds: float = Field(10, gt=0)
  loitering_seconds: float = Field(30, gt=0)
  loiter_radius: float = Field(3.0, gt=0)
  loiter_min_spread: float = Field(0.5, ge=0)
  loiter_min_px: float = Field(6.0, ge=0)
  loiter_classes: tuple[str, ...] = ("person",)
  speed_zscore: float = Field(3.0, gt=0)
  speed_min_samples: int = Field(50, ge=2)
  speed_min_seconds: float = Field(1.0, ge=0)
  speed_classes: tuple[str, ...] = ("person", "bicycle", "car", "motorcycle", "bus", "truck")
  crowd_classes: tuple[str, ...] = ("person",)
  crowd_min_confidence: float = Field(0.3, ge=0, le=1)
  group_min: int = Field(3, ge=2)
  group_link: float = Field(1.5, gt=0)
  crowd_min: int = Field(8, ge=2)
  crowd_link: float = Field(4.0, gt=0)
  crowd_total: int = Field(20, ge=2)
  density_min_seconds: float = Field(2.0, ge=0)
  surge_count: int = Field(8, ge=1)
  surge_zscore: float = Field(3.0, gt=0)
  surge_min_increase: int = Field(4, ge=1)
  surge_baseline_seconds: float = Field(30, gt=0)
  surge_min_seconds: float = Field(2.0, ge=0)

  @field_validator("rules")
  @classmethod
  def check_rules(cls, v):
    bad = [r for r in v if r not in RULE_NAMES]
    if bad: raise ValueError(f"unknown rules {bad}, valid: {list(RULE_NAMES)}")
    return v

  @model_validator(mode="after")
  def check_relations(self):
    if self.loitering_seconds < self.stationary_seconds: raise ValueError("loitering_seconds must be >= stationary_seconds")
    if self.owner_absent_seconds > self.stationary_seconds: raise ValueError("owner_absent_seconds must be <= stationary_seconds")
    if self.loiter_min_spread >= self.loiter_radius: raise ValueError("loiter_min_spread must be < loiter_radius")
    if self.crowd_min < self.group_min: raise ValueError("crowd_min must be >= group_min")
    if self.crowd_link < self.group_link: raise ValueError("crowd_link must be >= group_link")
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