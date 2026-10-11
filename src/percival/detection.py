import cv2
import numpy as np

def apply_global_heatmap(frame, diff_mask, colormap=cv2.COLORMAP_JET, alpha=0.4):
  heatmap = cv2.applyColorMap(diff_mask, colormap)
  heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2BGRA)
  frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
  blended = cv2.addWeighted(frame_bgra, 1-alpha, heatmap, alpha, 0)
  return cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)