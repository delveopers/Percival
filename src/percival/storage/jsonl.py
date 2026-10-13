import json, os

class JsonlWriter:
  def __init__(self, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    self.f = path.open("w", encoding="utf-8")
    self.count = 0

  def write(self, record):
    self.f.write(record.model_dump_json() + "\n")
    self.count += 1

  def flush(self): self.f.flush()
  def close(self): self.f.close()
  def __enter__(self): return self
  def __exit__(self, *exc): self.close()

class NullWriter:
  count = 0
  def write(self, record): pass
  def flush(self): pass
  def close(self): pass
  def __enter__(self): return self
  def __exit__(self, *exc): pass

def write_manifest(path, data):
  tmp = path.with_suffix(".json.tmp")
  tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
  os.replace(tmp, path)