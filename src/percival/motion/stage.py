import cv2
import numpy as np
from percival.contracts import ChangeCandidates, MotionResult
from .boxes import merge_boxes

BLUR = (5, 5)

class MotionStage:
  def __init__(self, cfg):
    self.cfg = cfg
    self.kernel = np.ones((cfg.morph_kernel, cfg.morph_kernel), np.uint8)
    self.ref = None

  def process(self, frame):
    c = self.cfg
    gray = cv2.GaussianBlur(cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY), BLUR, 0)
    if self.ref is None:
      self.ref = gray.astype(np.float32)
      return MotionResult([], False, 0.0, np.zeros_like(gray))

    diff = cv2.absdiff(cv2.convertScaleAbs(self.ref), gray)
    _, mask = cv2.threshold(diff, c.diff_threshold, 255, cv2.THRESH_BINARY)
    fraction = float(np.count_nonzero(mask)) / mask.size
    running = c.reference_mode == "running_average"

    if fraction > c.scene_change_fraction:
      if running: self.ref = gray.astype(np.float32)
      return MotionResult([], True, fraction, diff)

    boxes = merge_boxes(self._boxes(self._clean(mask)), c.merge_overlap)
    candidates = [self._candidate(frame, b, diff) for b in boxes]
    if running: cv2.accumulateWeighted(gray, self.ref, c.background_alpha)
    return MotionResult(candidates, False, fraction, diff)

  def _clean(self, mask):
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

  def _boxes(self, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_w, min_h = self.cfg.min_box
    boxes = []
    for cnt in contours:
      if len(cnt) < self.cfg.min_contour_points: continue
      x, y, w, h = cv2.boundingRect(cnt)
      if w >= min_w and h >= min_h: boxes.append((x, y, w, h))
    return boxes

  def _candidate(self, frame, box, diff):
    x, y, w, h = box
    score = float(diff[y:y + h, x:x + w].mean()) / 256
    return ChangeCandidates(run_id=frame.run_id, clip_id=frame.clip_id, source_index=frame.source_index, sample_index=frame.sample_index,
                            timestamp_s=frame.timestamp_s, box=box, score=score, area=w*h)