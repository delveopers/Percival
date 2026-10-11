import cv2
from percival.contracts import Frame
import logging

log = logging.getLogger(__name__)
MAX_BAD_READS = 5

def resize_to_width(image, width):
  h, w = image.shape[:2]
  if w <= width: return image, 1.0

  scale = width / w
  return cv2.resize(image, (width, max(1, round(h * scale))), interpolation=cv2.INTER_AREA), scale

def iter_frames(clip, cfg, run_id):
  cap = cv2.VideoCapture(str(clip.path))
  if not cap.isOpened():
    raise RuntimeError(f"Cannot open {clip.path}")
  
  n = cfg.sample_every_n_frames
  src = sample = bad = 0
  ended = False

  try:
    while cfg.max_frames is None or sample < cfg.max_frames:
      if not cap.grab():
        ended = True
        break
      if src % n == 0:
        ok, img = cap.retrieve()
        if ok and img is not None:
          bad = 0
          img, scale = resize_to_width(img, cfg.resize_width)
          yield Frame(run_id, clip.clip_id, src, sample, src/clip.fps, img, scale)
          sample += 1
        else:
          bad += 1
          log.warning("Failed to read the frame at %d of %s", src, clip.clip_id)
          if bad >= MAX_BAD_READS: break
      src += 1
  finally:
    log.info("clip %s: read=%d yielded=%d expected=%d", clip.clip_id, src, sample, clip.frame_count)
    if ended and clip.frame_count and src < 0.9 * clip.frame_count:
      log.warning("clip %s ended early, possible truncation", clip.clip_id)
 
