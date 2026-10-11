import cv2

def heatmap(image, diff_map, alpha, colormap=cv2.COLORMAP_JET):
  return cv2.addWeighted(image, 1 - alpha, cv2.applyColorMap(diff_map, colormap), alpha, 0)

def draw_candidates(image, candidates, color=(0, 255, 0)):
  out = image.copy()
  for c in candidates:
    x, y, w, h = c.box
    cv2.rectangle(out, (x, y), (x+w, y+h), color, 1)
    cv2.putText(out, f"{c.score:.2f}", (x, max(12, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
  return out

def annotate(frame, result, alpha):
  out = draw_candidates(heatmap(frame.image, result.diff_map, alpha), result.candidates)
  if result.scene_change: cv2.putText(out, "SCENE CHANGE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
  return out