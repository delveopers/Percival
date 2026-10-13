from pathlib import Path
import yaml
from pydantic import ValidationError
from .schema import AppConfig, AppFile, ProcessingFile, ModelsFile

class ConfigError(Exception): pass

FILES = {"app": AppFile, "processing": ProcessingFile, "models": ModelsFile}

def read_yaml(path):
  if not path.is_file(): raise ConfigError(f"missing config file: {path}")
  try: data = yaml.safe_load(path.read_text(encoding="utf-8"))
  except yaml.YAMLError as e: raise ConfigError(f"invalid YAML in {path.name}: {e}") from e
  if not isinstance(data, dict): raise ConfigError(f"{path.name} must contain a top-level mapping")
  return data

def parse(path, model):
  try: return model.model_validate(read_yaml(path))
  except ValidationError as e:
    detail = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
    raise ConfigError(f"{path.name}: {detail}") from e

def load_config(config_dir="configs"):
  d = Path(config_dir).resolve()
  root = d.parent
  app, proc, models = (parse(d / f"{name}.yaml", model) for name, model in FILES.items())
  resolve = lambda p: p if p.is_absolute() else (root / p).resolve()

  paths = app.paths.model_copy(update={"input_dir": resolve(app.paths.input_dir), "output_dir": resolve(app.paths.output_dir)})
  logging_cfg = app.logging.model_copy(update={"file": resolve(app.logging.file) if app.logging.file else None})

  return AppConfig(run=app.run, paths=paths, logging=logging_cfg, output=app.output,
                   ingestion=proc.ingestion, motion=proc.motion, tracking=proc.tracking, anomaly=proc.anomaly,
                   detector=models.detector, tracker=models.tracker, version=models.version)