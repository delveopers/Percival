import pytest
from percival.config import ConfigError, load_config

def test_valid_config_loads_and_paths_resolve_against_the_project_root(project):
  cfg = project.cfg()
  assert cfg.paths.input_dir == (project.root / "data" / "samples").resolve() and cfg.paths.output_dir.is_absolute()
  assert cfg.anomaly.warmup_seconds == 0 and cfg.output.save_detections

def test_unknown_key_names_the_file_and_the_field(project):
  project.write_config({"processing": {"motion": {"diff_treshold": 5}}})
  with pytest.raises(ConfigError, match=r"processing\.yaml.*diff_treshold"): project.cfg()

def test_out_of_range_value_is_rejected(project):
  project.write_config({"processing": {"motion": {"diff_threshold": 300}}})
  with pytest.raises(ConfigError, match="diff_threshold"): project.cfg()

def test_missing_file_is_reported(project):
  (project.root / "configs" / "models.yaml").unlink()
  with pytest.raises(ConfigError, match="missing config file"): project.cfg()

def test_broken_yaml_is_reported(project):
  (project.root / "configs" / "app.yaml").write_text("run: [unclosed", encoding="utf-8")
  with pytest.raises(ConfigError, match="invalid YAML"): project.cfg()

@pytest.mark.parametrize("section,over,fragment", [
  ("processing", {"anomaly": {"stationary_seconds": 20, "loitering_seconds": 10}}, "loitering_seconds"),
  ("processing", {"anomaly": {"stationary_seconds": 5, "loitering_seconds": 30, "owner_absent_seconds": 9}}, "owner_absent_seconds"),
  ("processing", {"anomaly": {"loiter_radius": 0.4, "loiter_min_spread": 0.5}}, "loiter_min_spread"),
  ("processing", {"anomaly": {"rules": ["stationary", "teleport"]}}, "unknown rules"),
  ("models", {"tracker": {"track_low_thresh": 0.6, "track_high_thresh": 0.5}}, "track_low_thresh"),
])
def test_cross_field_rules(project, section, over, fragment):
  project.write_config({section: over})
  with pytest.raises(ConfigError, match=fragment): project.cfg()

def test_config_is_read_only(project):
  with pytest.raises(Exception): project.cfg().motion.diff_threshold = 1