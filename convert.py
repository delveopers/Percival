
import cv2
from pathlib import Path
import re

INPUT_FOLDER = Path(r"D:\DEVS\PERCIVAL\model\test\MOT20-02\img1")
OUTPUT_VIDEO = Path(r"D:\DEVS\PERCIVAL\model\test\mot20-02.mp4")
FPS = 30

def natural_sort_key(path):
  return [
    int(part) if part.isdigit() else part.lower()
    for part in re.split(r"(\d+)", path.name)
  ]

def frames_to_video():
  images = sorted(
    (
      path for path in INPUT_FOLDER.iterdir()
      if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    ),
    key=natural_sort_key
  )
  if not images:
    raise FileNotFoundError(f"No JPG images found in {INPUT_FOLDER}")

  first_frame = cv2.imread(str(images[0]))

  if first_frame is None:
    raise ValueError(f"Cannot read image: {images[0]}")

  height, width = first_frame.shape[:2]

  OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

  writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (width, height)
  )

  if not writer.isOpened():
    raise RuntimeError("Could not create output video.")

  try:
    for index, image_path in enumerate(images):
      frame = cv2.imread(str(image_path))
      if frame is None:
        print(f"Skipping unreadable image: {image_path.name}")
        continue
      if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height))
      writer.write(frame)
      if (index + 1) % 100 == 0:
        print(f"Processed {index + 1}/{len(images)} frames")

  finally:
    writer.release()

  print(f"Video saved to: {OUTPUT_VIDEO}")
  print(f"Frames found: {len(images)}")
  print(f"FPS: {FPS}")
  print(f"Approximate duration: {len(images) / FPS:.2f} seconds")

if __name__ == "__main__":
  frames_to_video()