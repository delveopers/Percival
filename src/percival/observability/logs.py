import json, logging
from contextvars import ContextVar
from pathlib import Path

clip_var = ContextVar("clip_id", default="-")

def set_clip(clip_id): clip_var.set(clip_id)

class ClipFilter(logging.Filter):
  def filter(self, record):
    record.clip_id = clip_var.get()
    return True

class JsonFormatter(logging.Formatter):
  def format(self, record):
    d = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname, "logger": record.name, "clip_id": getattr(record, "clip_id", "-"), "msg": record.getMessage()}
    if record.exc_info: d["exc"] = self.formatException(record.exc_info)
    return json.dumps(d)

def setup_logging(cfg):
  root = logging.getLogger()
  root.setLevel(cfg.level)
  for h in list(root.handlers):
    if getattr(h, "_per", False):
      root.removeHandler(h)
      h.close()
  handlers = [logging.StreamHandler()]
  if cfg.file:
    Path(cfg.file).parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(cfg.file, encoding="utf-8"))
  fmt = JsonFormatter() if cfg.format == "json" else logging.Formatter("%(asctime)s %(levelname)s [%(clip_id)s] %(name)s: %(message)s")
  for h in handlers:
    h.setFormatter(fmt)
    h.addFilter(ClipFilter())
    h._per = True
    root.addHandler(h)