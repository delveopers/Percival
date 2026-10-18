import argparse, json, math, statistics as st, sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

JUMP_PER_FRAME, DUP_IOU, MIN_COVERAGE, BIG_GAP, FRAG_GAP, FRAG_DIST = 1.0, 0.7, 0.6, 10, 30, 1.5
STAGES = ("motion", "detect", "track", "anomaly", "render")

def load(run_dir, name):
  for stem in (name, name.rstrip("s")):
    for ext in (".jsonl", ".json"):
      p = run_dir / f"{stem}{ext}"
      if p.is_file(): return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
  return []

def center(b): return (b[0] + b[2] / 2, b[1] + b[3] / 2)
def size(b): return max(b[2], b[3], 1e-6)

def iou(a, b):
  iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
  ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])

  if iw <= 0 or ih <= 0: return 0.0
  inter = iw * ih
  return inter / (a[2] * a[3] + b[2] * b[3] - inter)

def track_stats(rows):
  rows = sorted(rows, key=lambda r: r["sample_index"])
  idx = [r["sample_index"] for r in rows]
  gaps = [b - a - 1 for a, b in zip(idx, idx[1:]) if b - a > 1]
  hop = 0.0

  for p, q in zip(rows, rows[1:]):
    (px, py), (qx, qy) = center(p["box"]), center(q["box"])
    hop = max(hop, math.hypot(qx - px, qy - py) / (max(size(p["box"]), size(q["box"])) * (q["sample_index"] - p["sample_index"])))

  return {"track_id": rows[0]["track_id"], "class": Counter(r["class_name"] for r in rows).most_common(1)[0][0], "n": len(rows), "first": idx[0], "last": idx[-1], "coverage": len(rows) / (idx[-1] - idx[0] + 1), "max_gap": max(gaps, default=0), "gaps": len(gaps), "max_hop": hop, "first_box": rows[0]["box"], "last_box": rows[-1]["box"]}

def fragments(tracks):
  out = []

  for a, b in combinations(tracks, 2):
    for x, y in ((a, b), (b, a)):
      gap = y["first"] - x["last"]

      if not 1 <= gap <= FRAG_GAP or x["class"] != y["class"]: continue
      (ax, ay), (bx, by) = center(x["last_box"]), center(y["first_box"])

      if math.hypot(bx - ax, by - ay) <= FRAG_DIST * max(size(x["last_box"]), size(y["first_box"])): out.append([x["track_id"], y["track_id"], gap])

  return out

def duplicate_pairs(frames):
  pairs = defaultdict(int)
  for rows in frames.values():
    for a, b in combinations(rows, 2):
      if a["track_id"] != b["track_id"] and iou(a["box"], b["box"]) >= DUP_IOU: pairs[tuple(sorted((a["track_id"], b["track_id"])))] += 1
  return [[a, b, n] for (a, b), n in sorted(pairs.items())]

def unreachable(cfg, frames, fps):
  a, duration = cfg["anomaly"], frames / fps
  warm = a["warmup_frames"] / fps
  need = {"stationary": a["stationary_seconds"], "abandoned": a["stationary_seconds"], "loitering": a["loitering_seconds"], "crowd": a["crowd_min_seconds"], "speed": 0}
  return [f"{r}: needs {max(s, warm):.1f}s (rule {s}s, warm-up {warm:.1f}s) but the clip is {duration:.1f}s" for r, s in need.items() if r in a["rules"] and duration < max(s, warm)]

def clip_report(entry, cfg, obs, trk, ang, chg):
  counts, fps = entry["counts"], entry["effective_fps"]
  frames = counts["frames"]
  by_frame, by_track = defaultdict(list), defaultdict(list)

  for o in obs:
    by_frame[o["sample_index"]].append(o)
    by_track[o["track_id"]].append(o)

  per_frame = [len(v) for v in by_frame.values()]
  dup_frames = sum(1 for v in by_frame.values() if len(v) > 1 and len({tuple(round(x, 1) for x in r["box"]) for r in v}) < len(v))
  expected = counts.get("observations", counts.get("observation"))
  consistent = expected is None or expected == len(obs)
  geo = dup_frames == 0 and consistent
  tracks = [track_stats(v) for v in by_track.values()]
  chg_frames = {c["sample_index"] for c in chg}
  tracklet_points = sum(len(t["points"]) for t in trk)
  anomalies = [{"kind": a["kind"], "start_s": a["start_s"], "end_s": a["end_s"], "confidence": round(a["confidence"], 3), "tracks": len(a["track_ids"]), "at_warmup_end": a["start_index"] == cfg["anomaly"]["warmup_frames"], "at_clip_end": a["end_index"] >= frames - 1, "details": a["details"]} for a in ang]
  seconds = entry.get("seconds", {})
  total = sum(seconds.values()) or 1

  r = {"clip_id": entry["clip_id"], "frames": frames, "effective_fps": fps, "duration_s": round(frames / fps, 1), "observations": len(obs), "frames_with_observations": len(by_frame), "objects_per_frame_mean": round(st.mean(per_frame), 1) if per_frame else 0, "objects_per_frame_max": max(per_frame, default=0),
       "duplicate_box_frames": dup_frames, "tracks": len(tracks), "tracklets": len(trk), "orphan_observations": len(obs) - tracklet_points if consistent else None,
       "tracks_short": sum(t["n"] < cfg["tracking"]["min_track_length"] for t in tracks), "tracks_low_coverage": [t["track_id"] for t in tracks if t["coverage"] < MIN_COVERAGE], "tracks_big_gap": [[t["track_id"], t["max_gap"]] for t in tracks if t["max_gap"] > BIG_GAP], "tracks_hopping": [[t["track_id"], round(t["max_hop"], 2)] for t in tracks if t["max_hop"] > JUMP_PER_FRAME] if geo else [],
       "duplicate_track_pairs": duplicate_pairs(by_frame) if geo else [], "possible_fragments": fragments(tracks) if geo else [], "median_track_points": st.median([t["n"] for t in tracks]) if tracks else 0,
       "motion": {"candidates": len(chg), "frames_with_changes": len(chg_frames), "frames_motion_without_detection": len(chg_frames - set(by_frame))},
       "anomalies": anomalies, "unreachable_rules": unreachable(cfg, frames, fps), "tracker_dropped": counts.get("tracker_dropped"), "avg_ms_per_frame": entry.get("avg_ms_per_frame", {}), "time_share": {k: round(seconds.get(k, 0) / total, 2) for k in STAGES if k in seconds}}
  flags = []

  if dup_frames or counts.get("duplicate_box_frames"): flags.append(("FAIL", f"{dup_frames} frames have observations sharing the same box (tracker-to-detection mapping is broken)"))
  if not consistent: flags.append(("FAIL", f"manifest counted {expected} observations but {len(obs)} were loaded (observations file missing or renamed?)"))
  if not geo and consistent: flags.append(("INFO", "geometry checks (ID hopping, duplicate IDs, fragments) skipped because the boxes are invalid"))
  if not obs and chg and consistent: flags.append(("WARN", f"no detections at all, but motion found in {len(chg_frames)} frames"))
  if consistent and r["orphan_observations"]: flags.append(("WARN", f"{r['orphan_observations']} observations belong to no tracklet (short fragments or reused IDs)"))
  if r["tracks_hopping"]: flags.append(("WARN", f"{len(r['tracks_hopping'])} tracks jump farther than {JUMP_PER_FRAME} box sizes per frame (ID hopping): {r['tracks_hopping'][:5]}"))
  if r["tracks_low_coverage"]: flags.append(("WARN", f"{len(r['tracks_low_coverage'])} tracks are seen in under {int(MIN_COVERAGE * 100)}% of their frames: {r['tracks_low_coverage'][:8]}"))
  if r["tracks_big_gap"]: flags.append(("WARN", f"{len(r['tracks_big_gap'])} tracks have gaps over {BIG_GAP} frames: {r['tracks_big_gap'][:5]}"))
  if r["duplicate_track_pairs"] and not dup_frames: flags.append(("WARN", f"{len(r['duplicate_track_pairs'])} track pairs overlap on the same object (duplicate IDs): {r['duplicate_track_pairs'][:3]}"))
  if r["possible_fragments"]: flags.append(("WARN", f"{len(r['possible_fragments'])} possible ID fragments (one ends, another starts nearby): {r['possible_fragments'][:4]}"))

  flags += [("WARN", f"rule cannot fire on this clip, {u}") for u in r["unreachable_rules"]]

  for a in anomalies:
    if a["at_warmup_end"]: flags.append(("WARN", f"{a['kind']} starts exactly when warm-up ends, the real onset is unknown"))
    if a["at_clip_end"]: flags.append(("WARN", f"{a['kind']} is still active at clip end, its end is unknown"))

  r["flags"] = [{"level": l, "msg": m} for l, m in flags]
  r["status"] = "FAIL" if any(l == "FAIL" for l, _ in flags) else "WARN" if any(l == "WARN" for l, _ in flags) else "PASS"
  return r

def analyze(run_dir):
  run_dir = Path(run_dir)
  run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
  obs, trk, ang, chg = (load(run_dir, n) for n in ("observations", "tracklets", "anomalies", "changes"))

  pick = lambda rows, cid: [r for r in rows if r["clip_id"] == cid]
  clips = [clip_report(e, run["config"], pick(obs, e["clip_id"]), pick(trk, e["clip_id"]), pick(ang, e["clip_id"]), pick(chg, e["clip_id"])) for e in run["clips"] if e["status"] == "completed"]
  skipped = [{"clip_id": e["clip_id"], "status": e["status"], "error": e.get("error")} for e in run["clips"] if e["status"] != "completed"]
  status = "FAIL" if not clips or any(c["status"] == "FAIL" for c in clips) else "WARN" if any(c["status"] == "WARN" for c in clips) or skipped else "PASS"
  return {"run_id": run["run_id"], "run_status": run["status"], "status": status, "clips": clips, "not_completed": skipped}

def render(rep):
  out = [f"QA report for run {rep['run_id']} (run status: {rep['run_status']}, QA: {rep['status']})", ""]
  out.append(f"{'clip':40} {'frames':>6} {'obs':>6} {'tracks':>6} {'anoms':>5}  status")

  for c in rep["clips"]: out.append(f"{c['clip_id'][:40]:40} {c['frames']:>6} {c['observations']:>6} {c['tracks']:>6} {len(c['anomalies']):>5}  {c['status']}")
  for s in rep["not_completed"]: out.append(f"{s['clip_id'][:40]:40} not completed: {s['status']} {s['error'] or ''}")

  for c in rep["clips"]:
    out += ["", f"== {c['clip_id']}  [{c['status']}]", f"  {c['duration_s']}s at {c['effective_fps']} fps, {c['frames']} frames", f"  objects per frame: mean {c['objects_per_frame_mean']}, max {c['objects_per_frame_max']}; frames with objects: {c['frames_with_observations']}", f"  tracks {c['tracks']} (median {c['median_track_points']} points), tracklets {c['tracklets']}, orphan observations {'n/a' if c['orphan_observations'] is None else c['orphan_observations']}",
            f"  motion: {c['motion']['candidates']} candidates in {c['motion']['frames_with_changes']} frames, {c['motion']['frames_motion_without_detection']} of them with no detection"]
    if c["avg_ms_per_frame"]: out.append("  ms per frame: " + ", ".join(f"{k} {v}" for k, v in c["avg_ms_per_frame"].items()) + "  | time share: " + ", ".join(f"{k} {int(v * 100)}%" for k, v in c["time_share"].items()))

    for a in c["anomalies"]: out.append(f"  anomaly: {a['kind']} {a['start_s']}s to {a['end_s']}s, confidence {a['confidence']}, {a['tracks']} tracks")
    out += [f"  [{f['level']}] {f['msg']}" for f in c["flags"]]
  return "\n".join(out)

def main(argv=None):
  ap = argparse.ArgumentParser(description="QA checks for a Percival run")
  ap.add_argument("run_dir", nargs="?")
  ap.add_argument("--root", default="data/exports")
  args = ap.parse_args(argv)

  if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  if args.run_dir: run_dir = Path(args.run_dir)
  else:
    runs = sorted(p for p in Path(args.root).glob("*") if (p / "run.json").is_file())
    if not runs: sys.exit(f"no runs found under {args.root}")
    run_dir = runs[-1]

  rep = analyze(run_dir)
  text = render(rep)
  (run_dir / "qa_report.md").write_text(text, encoding="utf-8")
  (run_dir / "qa_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
  print(text)

  return 1 if rep["status"] == "FAIL" else 0