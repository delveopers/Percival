import cv2
import numpy as np
from percival.motion.render import annotate

FONT = cv2.FONT_HERSHEY_SIMPLEX

def track_color(track_id):
  b, g, r = cv2.cvtColor(np.uint8([[[(track_id * 37) % 180, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
  return int(b), int(g), int(r)

def draw_observations(img, observations):
  for o in observations:
    x, y, w, h = (int(v) for v in o.box)
    color = track_color(o.track_id)
    cv2.rectangle(img, (x, y), (x + w, y+ h), color, 2)
    cv2.putText(img, f"#{o.track_id} {o.class_name} {o.confidence:.2f}", (x, max(12, y - 6)), FONT, 0.5, color, 1, cv2.LINE_AA)

def draw_anomalies(img, anomalies):
  for a in anomalies:
    x, y, w, h = (int(v) for v in a.box)
    cv2.rectangle(img, (x-4, y-4), (x+w-4, y+h-4), (0, 0, 255), 2)
    cv2.putText(img, f"{a.kind} {a.confidence:.2f}", (x, min(img.shape[0] - 6, y + h + 20)), FONT, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

def compose(ctx, heatmap_alpha):
  out = annotate(ctx.frame, ctx.motion, heatmap_alpha) if ctx.motion is not None else ctx.frame.image.copy()
  draw_observations(out, ctx.observations)
  draw_anomalies(out, ctx.anomalies)
  cv2.putText(out, f"{ctx.frame.timestamp_s:6.1f}s  objects:{len(ctx.observations)}  anomalies:{len(ctx.anomalies)}", (10, out.shape[0] - 10), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
  return out