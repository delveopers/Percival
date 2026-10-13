import os, sys
import cv2

class VideoSink:
  def __init__(self, path, fps):
    self.path, self.fps = path, fps
    self.writer = self.size = None

  def write(self, image):
    h, w = image.shape[:2]

    if self.writer is None:
      self.path.parent.mkdir(parents=True, exist_ok=True)
      self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
      
      if not self.writer.isOpened(): raise RuntimeError(f"Cannot open video writer for {self.path}")
      self.size = (w, h)
    
    if (w, h) != self.size: raise ValueError(f"frame size {(w, h)} differs from writer size {self.size}")
    self.writer.write(image)

  def close(self):
    if self.writer is not None: self.writer.release()
    self.writer = None

class Display:
  def __init__(self, name="percival"):
    self.name = name
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
      raise RuntimeError("show_window is enabled but no display is available")
    
    try: cv2.namedWindow(self.name)
    except cv2.error as e:
      raise RuntimeError("show_window needs an OpenCV build with GUI support (not opencv-python-headless)") from e

  def show(self, image):
    cv2.imshow(self.name, image)
    return (cv2.waitKey(1) & 0xFF) != 27

  def close(self): cv2.destroyAllWindows()