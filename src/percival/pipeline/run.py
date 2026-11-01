import logging, random
from contextlib import ExitStack
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
from percival.contracts import FrameContext, FrameMeta
from percival.detection import Detector, Tracker
from percival.ingestion import scan_clips
from percival.motion import MotionStage
from percival.observability import Metrics, setup_logging, set_clip
from percival.preprocessing import iter_frames
from percival.storage import JsonlWriter, NullWriter, safe_name, write_manifest
from percival.visualization import VideoSink, Display, compose
from .analysis import Analysis

log = logging.getLogger(__name__)
WRITERS = ("observations", "tracklets", "anomalies", "changes")
CACHE_WRITERS = ("frames", "detections")

def make_run_id(): return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
def now_iso(): return datetime.now(timezone.utc).isoformat(timespec="seconds")

def process_clip(cfg, clip, run_id, out_dir, writers, detector, tracker_factory):
  ing = cfg.ingestion
  fps = clip.fps / ing.sample_every_n_frames

  m = Metrics()
  motion = MotionStage(cfg.motion) if cfg.motion.enabled else None
  analysis = Analysis(cfg, run_id, clip.clip_id, tracker_factory(fps), writers, m) if detector else None
  sink = VideoSink(out_dir / "annotated" / f"{safe_name(clip.clip_id)}.mp4", fps) if cfg.output.save_annotated_video else None
  display = Display() if cfg.output.show_window else None
  stopped = False

  try:
    for frame in iter_frames(clip, ing, run_id):
      m.add("frames")
      ctx = FrameContext(frame)

      if motion:
        with m.time("motion"): ctx.motion = motion.process(frame)
        for c in ctx.motion.candidates: writers["changes"].write(c)
        m.add("changes", len(ctx.motion.candidates))

      scene = bool(ctx.motion and ctx.motion.scene_change)
      m.add("scene_changes", scene)

      if analysis:
        with m.time("detect"): dets = detector.detect(frame)
        for d in dets: writers["detections"].write(d)
        writers["frames"].write(FrameMeta(run_id=run_id, clip_id=clip.clip_id, sample_index=frame.sample_index, source_index=frame.source_index, timestamp_s=frame.timestamp_s, scene_change=scene, detections=len(dets)))
        ctx.observations, ctx.anomalies = analysis.step(frame, dets, scene)

      if sink or display:
        with m.time("render"): img = compose(ctx, cfg.motion.heatmap_alpha)
        if sink: sink.write(img)
        if display and not display.show(img):
          stopped = True
          break
    if analysis: analysis.finish()
    for w in writers.values(): w.flush()

  finally:
    if sink: sink.close()
    if display: display.close()

  return m, stopped

def run(cfg, detector=None, tracker_factory=None):
  setup_logging(cfg.logging)
  clips = scan_clips(cfg)
  run_id = make_run_id()
  random.seed(cfg.run.seed)
  np.random.seed(cfg.run.seed)
  out_dir = cfg.paths.output_dir / run_id
  out_dir.mkdir(parents=True, exist_ok=True)

  if not cfg.tracking.enabled: detector = None
  elif detector is None: detector = Detector(cfg.detector, cfg.version.detector)
  if cfg.anomaly.enabled and not cfg.tracking.enabled: log.warning("anomaly rules need tracking, skipping them")

  tracker_factory = tracker_factory or (lambda fps: Tracker(cfg.tracker, cfg.version.tracker, fps))
  entries = []
  manifest = {"run_id": run_id, "status": "running", "mode": "live", "started_at": now_iso(), "finished_at": None, "versions": cfg.version.model_dump(), "config": cfg.model_dump(mode="json"), "clips": entries, "totals": {}}
  path, totals, interrupted = out_dir / "run.json", Metrics(), False
  write_manifest(path, manifest)

  log.info("run %s started: %d clips, output=%s", run_id, len(clips), out_dir)

  with ExitStack() as stack:
    enabled = {**{n: cfg.output.save_observations for n in WRITERS}, **{n: cfg.output.save_detections for n in CACHE_WRITERS}}
    writers = {n: stack.enter_context(JsonlWriter(out_dir / f"{n}.jsonl") if on else NullWriter()) for n, on in enabled.items()}
    try:
      for clip in clips:
        set_clip(clip.clip_id)
        entry = {"clip_id": clip.clip_id, "path": str(clip.path), "status": "running"}
        entries.append(entry)
        write_manifest(path, manifest)
        try:
          m, stopped = process_clip(cfg, clip, run_id, out_dir, writers, detector, tracker_factory)
          totals.merge(m)
          entry.update(status="interrupted" if stopped else "completed", effective_fps=clip.fps / cfg.ingestion.sample_every_n_frames, **m.snapshot())
          log.info("clip done: %s", m.snapshot()["counts"])

          if m.counts["duplicate_box_frames"]: log.error("%d frames have observations sharing a box, tracker mapping is broken", m.counts["duplicate_box_frames"])
          if stopped:
            interrupted = True
            break
        except Exception as e:
          log.exception("clip failed")
          entry.update(status="failed", error=f"{type(e).__name__}: {e}")
        write_manifest(path, manifest)
    except KeyboardInterrupt:
      interrupted = True
      if entries: entries[-1]["status"] = "interrupted"
      log.warning("interrupted by user")
    set_clip("-")

  done = all(e["status"] == "completed" for e in entries)
  manifest.update(status="interrupted" if interrupted else "completed" if done else "partial", finished_at=now_iso(), totals=totals.snapshot(), outputs={n: w.count for n, w in writers.items()})
  write_manifest(path, manifest)
  log.info("run %s finished: %s", run_id, manifest["status"])

  return out_dir