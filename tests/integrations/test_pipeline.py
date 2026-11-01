import json
import pytest
import yaml
from percival.contracts import Anomaly, ChangeCandidate, Detection, FrameMeta, Observation, Tracklet
from percival.evaluation import evaluate, load_labels
from percival.pipeline import replay, run
from percival.qa import analyze

class StubDetector:
  def __init__(self, scene): self.scene = scene

  def detect(self, frame):
    if frame.clip_id == "bad": raise RuntimeError("boom")
    return [Detection(run_id=frame.run_id, clip_id=frame.clip_id, source_index=frame.source_index, sample_index=frame.sample_index, timestamp_s=frame.timestamp_s, box=box, class_id=0, class_name=name, confidence=conf, detector_version="dv") for box, name, conf in self.scene(frame)]

class StubTracker:
  last_dropped = 0

  def __init__(self, limit=None): self.limit = limit

  def update(self, frame, dets):
    chosen = dets if self.limit is None else dets[:self.limit]
    return [Observation(**d.model_dump(), track_id=i + 1, tracker_version="tv") for i, d in enumerate(chosen)]

parked_car = lambda frame: [((50.0, 50.0, 40.0, 30.0), "car", 0.9)]

def rows(out, name): return [json.loads(l) for l in (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
def stripped(out, name): return [{k: v for k, v in r.items() if k != "run_id"} for r in rows(out, name)]

@pytest.fixture
def live(project):
  project.video("good.mp4")
  out = run(project.cfg(), StubDetector(parked_car), lambda fps: StubTracker())
  return project, out

def test_end_to_end_produces_valid_linked_outputs(live):
  _, out = live
  m = json.loads((out / "run.json").read_text())
  c = m["clips"][0]
  assert m["status"] == "completed" and m["mode"] == "live" and c["status"] == "completed" and c["effective_fps"] == 10
  assert c["counts"]["frames"] == 60 and c["counts"]["observations"] == 60 and c["counts"]["orphan_observations"] == 0 and c["counts"]["duplicate_box_frames"] == 0
  assert len(rows(out, "frames")) == 60 and len(rows(out, "detections")) == 60 and len(rows(out, "observations")) == 60
  assert [FrameMeta.model_validate(r) for r in rows(out, "frames")][5].sample_index == 5
  trk = [Tracklet.model_validate(r) for r in rows(out, "tracklets")]
  assert len(trk) == 1 and len(trk[0].points) == 60
  an = [Anomaly.model_validate(r) for r in rows(out, "anomalies")]
  assert [a.kind for a in an] == ["stationary_object"] and an[0].start_s >= 2.0
  assert all(ChangeCandidate.model_validate(r) for r in rows(out, "changes"))
  assert not list(out.glob("*.tmp"))

def test_replay_reproduces_the_live_results_exactly(live):
  project, out = live
  again = replay(project.cfg(), out, lambda fps: StubTracker())
  assert again != out and json.loads((again / "run.json").read_text())["mode"] == "replay" and json.loads((again / "run.json").read_text())["source_run"] == json.loads((out / "run.json").read_text())["run_id"]
  for name in ("observations", "tracklets", "anomalies"):
    assert stripped(again, name) == stripped(out, name) and rows(again, name)

def test_replay_with_a_different_config_changes_only_the_downstream_results(live):
  project, out = live
  project.write_config({"processing": {"anomaly": {"stationary_seconds": 4, "loitering_seconds": 10}}})
  again = replay(project.cfg(), out, lambda fps: StubTracker())
  assert stripped(again, "observations") == stripped(out, "observations")
  assert rows(out, "anomalies")[0]["start_s"] < 3 and rows(again, "anomalies")[0]["start_s"] >= 4.0

def test_replay_needs_the_detection_cache(project):
  project.video("good.mp4")
  project.write_config({"app": {"output": {"save_detections": False}}})
  out = run(project.cfg(), StubDetector(parked_car), lambda fps: StubTracker())
  with pytest.raises(ValueError, match="save_detections"): replay(project.cfg(), out, lambda fps: StubTracker())

def test_a_failing_clip_is_isolated_and_the_run_is_partial(project):
  project.video("good.mp4")
  project.video("bad.mp4")
  out = run(project.cfg(), StubDetector(parked_car), lambda fps: StubTracker())
  m = json.loads((out / "run.json").read_text())
  assert m["status"] == "partial" and {c["clip_id"]: c["status"] for c in m["clips"]} == {"bad": "failed", "good": "completed"}
  assert "RuntimeError" in next(c for c in m["clips"] if c["clip_id"] == "bad")["error"]
  assert {r["clip_id"] for r in rows(out, "anomalies")} == {"good"}
  assert [c["clip_id"] for c in json.loads((replay(project.cfg(), out, lambda fps: StubTracker()) / "run.json").read_text())["clips"]] == ["good"]

def test_crowd_rule_follows_raw_detections_even_when_the_tracker_misses_people(project):
  project.video("good.mp4")
  project.write_config({"processing": {"anomaly": {"rules": ["crowd"], "crowd_count": 8, "crowd_min_seconds": 1.0}}, "models": {"detector": {"classes": ["person"]}}})
  scene = lambda frame: [((100.0 * k, 100.0, 40.0, 100.0), "person", 0.9) for k in range(1, 6 if frame.sample_index < 30 else 16)]
  out = run(project.cfg(), StubDetector(scene), lambda fps: StubTracker(limit=5))
  an = rows(out, "anomalies")
  assert [a["kind"] for a in an] == ["crowd"] and an[0]["details"]["count"] == 15 and 2.9 <= an[0]["start_s"] <= 3.2
  assert max(len([o for o in rows(out, "observations") if o["sample_index"] == i]) for i in range(60)) == 5

def test_annotated_video_uses_a_safe_filename(project):
  project.video("my clip (1).mp4", frames=20)
  project.write_config({"app": {"output": {"save_annotated_video": True}}})
  out = run(project.cfg(), StubDetector(parked_car), lambda fps: StubTracker())
  names = [p.name for p in (out / "annotated").glob("*.mp4")]
  assert len(names) == 1 and all(ch.isalnum() or ch in "._-" for ch in names[0])

def test_qa_and_evaluation_work_on_a_live_run_and_on_its_replay(live, tmp_path):
  project, out = live
  again = replay(project.cfg(), out, lambda fps: StubTracker())
  labels = tmp_path / "labels.yaml"
  labels.write_text(yaml.safe_dump({"clips": {"good": {"events": [{"kind": "stationary_object", "start_s": 2.0, "end_s": None}]}}}))
  for d in (out, again):
    rep = analyze(d)
    assert rep["status"] != "FAIL" and rep["clips"][0]["observations"] == 60 and rep["clips"][0]["orphan_observations"] == 0
    ev = evaluate(d, load_labels(labels))
    assert ev["overall"]["precision"] == 1.0 and ev["overall"]["recall"] == 1.0

def test_scene_changes_pause_events_live_and_replay_keeps_those_flags(project):
  project.video("flashy.mp4", flash=(60, 70))
  absent = lambda frame: [] if 30 <= frame.sample_index < 45 else parked_car(frame)
  out = run(project.cfg(), StubDetector(absent), lambda fps: StubTracker())
  flags = [r["sample_index"] for r in rows(out, "frames") if r["scene_change"]]
  assert flags == [30, 35]
  assert len(rows(out, "anomalies")) == 1
  again = replay(project.cfg(), out, lambda fps: StubTracker())
  assert stripped(again, "anomalies") == stripped(out, "anomalies") and [r["scene_change"] for r in rows(again, "frames")] == [r["scene_change"] for r in rows(out, "frames")]

def test_replay_can_raise_the_confidence_floor_and_narrow_the_classes(project):
  project.video("good.mp4")
  mixed = lambda frame: [((50.0, 50.0, 40.0, 30.0), "car", 0.9), ((200.0, 50.0, 40.0, 30.0), "car", 0.4), ((300.0, 50.0, 20.0, 60.0), "person", 0.9)]
  out = run(project.cfg(), StubDetector(mixed), lambda fps: StubTracker())
  per_frame = lambda d: len([o for o in rows(d, "observations") if o["sample_index"] == 0])
  assert per_frame(out) == 3
  project.write_config({"models": {"detector": {"conf": 0.5}}})
  assert per_frame(replay(project.cfg(), out, lambda fps: StubTracker())) == 2
  project.write_config({"models": {"detector": {"conf": 0.5, "classes": ["car"]}}})
  assert per_frame(replay(project.cfg(), out, lambda fps: StubTracker())) == 1