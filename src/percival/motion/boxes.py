
def overlap_ratio(a, b):
  ax, ay, aw, ah = a; bx, by, bw, bh = b
  iw = min(ax + aw, bx + bw) - max(ax, bx)
  ih = min(ay + ah, by + bh) - max(ay, by)
  if iw <= 0 or ih <= 0: return 0.0
  return iw * ih / min(aw * ah, bw * bh)

def union(a, b):
  x1, y1 = min(a[0], b[0]), min(a[1], b[1])
  x2, y2 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
  return (x1, y1, x2 - x1, y2 - y1)

def merge_boxes(boxes, threshold):
  boxes = [tuple(b) for b in boxes]
  merged = True

  while merged:
    merged = False
    out = []
    while boxes:
      cur = boxes.pop()
      i = 0
      while i < len(boxes):
        if overlap_ratio(cur, boxes[i]) > threshold:
          cur = union(cur, boxes.pop(i))
          merged, i = True, 0
        else: i += 1
      out.append(cur)
    boxes = out
  return boxes