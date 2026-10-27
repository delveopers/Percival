import logging, math
import cv2
from percival.contracts import ClipInfo

log = logging.getLogger(__name__)
DEFAULT_FPS = 30.0

class NoClipsError(RuntimeError): pass

def discover(input_dir, extensions, recursive):
  if not input_dir.is_dir():
    raise NoClipsError(f"input directory not found: {input_dir}")

  it = input_dir.rglob("*") if recursive else input_dir.glob("*")
  files = (
    p for p in it
    if p.is_file()
    and not p.name.startswith(".")
    and p.suffix.lower() in extensions
    and p.stat().st_size > 0
  )

  return sorted(files, key=lambda p: (str(p).lower(), str(p)))

def make_clip_id(path, root):
  return "__".join(path.relative_to(root).with_suffix("").parts)

def probe(path, root):
  cap = cv2.VideoCapture(str(path))
  try:
    if not cap.isOpened(): return None
    ok, frame = cap.read()
    if not ok or frame is None: return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
      log.warning("invalid fps for %s, using %s", path.name, DEFAULT_FPS)
      fps = DEFAULT_FPS
    n = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    h, w = frame.shape[:2]
    return ClipInfo(clip_id=make_clip_id(path, root), path=path, fps=fps, frame_count=n, width=w, height=h, duration_s=n / fps)
  finally: cap.release()

def scan_clips(cfg):
  p = cfg.paths
  files = discover(p.input_dir, p.extensions, p.recursive)
  clips = []
  for f in files:
    info = probe(f, p.input_dir)
    if info is None:
      log.warning("skipping unreadable clip: %s", f)
      continue
    clips.append(info)
  if not clips: raise NoClipsError(f"no readable clips in {p.input_dir}")
  log.info("found %d clips, skipped %d", len(clips), len(files) - len(clips))
  return clips