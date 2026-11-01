import json, logging, shutil
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from percival.contracts import Detection, Frame
from percival.detection import Tracker
from percival.observability import Metrics, setup_logging, set_clip
from percival.storage import JsonlWriter, NullWriter, write_manifest
from .analysis import Analysis
from .run import make_run_id, now_iso

log = logging.getLogger(__name__)
BLANK = np.zeros((1, 1, 3), np.uint8)
COPIED = ("frames", "detections", "changes")
WRITERS = ("observations", "tracklets", "anomalies")

def read_jsonl(path):
  return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.is_file() else []

def replay(cfg, source_dir, tracker_factory=None):
  setup_logging(cfg.logging)
  src_dir = Path(source_dir)
  source = json.loads((src_dir / "run.json").read_text(encoding="utf-8"))

  if not cfg.tracking.enabled: raise ValueError("replay needs tracking.enabled: true")
  frames, dets = read_jsonl(src_dir / "frames.jsonl"), read_jsonl(src_dir / "detections.jsonl")
  if not frames: raise ValueError(f"{src_dir} has no frames.jsonl, rerun the pipeline with output.save_detections: true")
  frames_by, dets_by = defaultdict(list), defaultdict(list)

  for f in frames: frames_by[f["clip_id"]].append(f)
  for d in dets: dets_by[(d["clip_id"], d["sample_index"])].append(d)

  run_id = make_run_id()
  out_dir = cfg.paths.output_dir / run_id
  out_dir.mkdir(parents=True, exist_ok=True)

  for n in COPIED:
    if (src_dir / f"{n}.jsonl").is_file(): shutil.copyfile(src_dir / f"{n}.jsonl", out_dir / f"{n}.jsonl")

  tracker_factory = tracker_factory or (lambda fps: Tracker(cfg.tracker, cfg.version.tracker, fps))
  entries, totals = [], Metrics()
  manifest = {"run_id": run_id, "mode": "replay", "source_run": source["run_id"], "status": "running", "started_at": now_iso(), "finished_at": None, "versions": cfg.version.model_dump(), "config": cfg.model_dump(mode="json"), "clips": entries, "totals": {}}
  path = out_dir / "run.json"
  write_manifest(path, manifest)
  log.info("replay %s of run %s started", run_id, source["run_id"])

  with ExitStack() as stack:
    on = cfg.output.save_observations
    writers = {n: stack.enter_context(JsonlWriter(out_dir / f"{n}.jsonl") if on else NullWriter()) for n in WRITERS}

    for src in source["clips"]:
      cid = src["clip_id"]
      if src["status"] != "completed" or cid not in frames_by: continue
      set_clip(cid)
      entry = {"clip_id": cid, "path": src.get("path"), "status": "running"}
      entries.append(entry)
      m = Metrics()
      try:
        analysis = Analysis(cfg, run_id, cid, tracker_factory(src["effective_fps"]), writers, m)
        for fm in sorted(frames_by[cid], key=lambda f: f["sample_index"]):
          frame = Frame(run_id, cid, fm["source_index"], fm["sample_index"], fm["timestamp_s"], BLANK, 1.0)
          found = [Detection.model_validate({**d, "run_id": run_id}) for d in dets_by[(cid, fm["sample_index"])] if d["confidence"] >= cfg.detector.conf and d["class_name"] in cfg.detector.classes]
          m.add("frames")
          m.add("scene_changes", fm["scene_change"])
          analysis.step(frame, found, fm["scene_change"])
        analysis.finish()
        m.add("changes", src.get("counts", {}).get("changes", 0))

        for w in writers.values(): w.flush()
        totals.merge(m)
        entry.update(status="completed", effective_fps=src["effective_fps"], **m.snapshot())
        log.info("clip replayed: %s", m.snapshot()["counts"])

      except Exception as e:
        log.exception("clip replay failed")
        entry.update(status="failed", error=f"{type(e).__name__}: {e}")
      write_manifest(path, manifest)
    set_clip("-")

  done = bool(entries) and all(e["status"] == "completed" for e in entries)
  manifest.update(status="completed" if done else "partial", finished_at=now_iso(), totals=totals.snapshot(), outputs={n: w.count for n, w in writers.items()})
  write_manifest(path, manifest)
  log.info("replay %s finished: %s", run_id, manifest["status"])

  return out_dir