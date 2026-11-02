import json
import pytest
import yaml
from percival.evaluation import evaluate, load_labels, render, template
from percival.evaluation.evaluate import family

def make_run(tmp_path, anomalies, clips=("a",)):
  (tmp_path / "run.json").write_text(json.dumps({"run_id": "r1", "clips": [{"clip_id": c, "status": "completed"} for c in clips]}))
  (tmp_path / "anomalies.jsonl").write_text("\n".join(json.dumps({"clip_id": c, "kind": k, "start_s": s, "end_s": e, "confidence": 0.7}) for c, k, s, e in anomalies))
  return tmp_path

def labels(tmp_path, body):
  p = tmp_path / "labels.yaml"
  p.write_text(yaml.safe_dump({"clips": body}))
  return load_labels(p)

def test_precision_and_recall_per_rule(tmp_path):
  run = make_run(tmp_path, [("a", "stationary_object", 8.0, 25.0), ("a", "loitering", 12.0, 14.0), ("a", "loitering", 30.0, 31.0)])
  rep = evaluate(run, labels(tmp_path, {"a": {"events": [{"kind": "stationary_object", "start_s": 8.5, "end_s": 24.0}, {"kind": "loitering", "start_s": 12.5, "end_s": 13.0}, {"kind": "crowd", "start_s": 3.0, "end_s": 9.0}]}}), tol=1.0)
  k = rep["kinds"]
  assert k["stationary_object"]["precision"] == 1.0 and k["stationary_object"]["recall"] == 1.0
  assert k["loitering"]["precision"] == 0.5 and k["loitering"]["recall"] == 1.0
  assert k["crowd"]["recall"] == 0.0 and k["crowd"]["precision"] is None
  assert rep["overall"]["precision"] == round(2 / 3, 3) and rep["overall"]["recall"] == round(2 / 3, 3)
  assert [f["start_s"] for f in rep["false_positives"]] == [30.0] and [m["kind"] for m in rep["missed"]] == ["crowd"]

def test_tolerance_decides_near_misses(tmp_path):
  run = make_run(tmp_path, [("a", "loitering", 20.0, 21.0)])
  lab = labels(tmp_path, {"a": {"events": [{"kind": "loitering", "start_s": 12.0, "end_s": 15.0}]}})
  assert evaluate(run, lab, tol=1.0)["overall"]["recall"] == 0.0
  assert evaluate(run, lab, tol=6.0)["overall"]["recall"] == 1.0

def test_open_ended_label_covers_everything_after_its_start(tmp_path):
  run = make_run(tmp_path, [("a", "stationary_object", 40.0, 50.0)])
  assert evaluate(run, labels(tmp_path, {"a": {"events": [{"kind": "stationary_object", "start_s": 8.0, "end_s": None}]}}), tol=0)["overall"]["recall"] == 1.0

def test_clip_labeled_as_empty_turns_every_prediction_into_a_false_positive(tmp_path):
  run = make_run(tmp_path, [("a", "crowd", 1.0, 5.0)])
  rep = evaluate(run, labels(tmp_path, {"a": {"events": []}}))
  assert rep["overall"]["precision"] == 0.0 and rep["overall"]["recall"] is None

def test_a_clip_not_in_the_run_is_reported_as_skipped(tmp_path):
  run = make_run(tmp_path, [("a", "crowd", 1.0, 5.0)], clips=("a", "b"))
  rep = evaluate(run, labels(tmp_path, {"zzz": {"events": []}}))
  assert rep["skipped_clips"] == ["zzz"] and rep["overall"]["predictions"] == 0

def test_the_template_is_unreviewed_so_it_cannot_score_itself(tmp_path):
  run = make_run(tmp_path, [("a", "crowd", 1.234, 5.678)])
  out = tmp_path / "t.yaml"
  out.write_text(yaml.safe_dump(template(run)))
  lab = load_labels(out)
  assert lab["a"]["reviewed"] is False and lab["a"]["events"] == [{"kind": "crowd", "start_s": 1.23, "end_s": 5.68}] and lab["a"]["ignore"] == []
  rep = evaluate(run, lab)
  assert rep["unreviewed_clips"] == ["a"] and rep["overall"]["predictions"] == 0 and rep["overall"]["precision"] is None
  lab["a"]["reviewed"] = True
  assert evaluate(run, lab)["overall"]["precision"] == 1.0

def test_ignored_events_are_neither_hits_nor_false_alarms(tmp_path):
  run = make_run(tmp_path, [("a", "loitering", 10.0, 12.0), ("a", "loitering", 40.0, 42.0), ("a", "crowd", 1.0, 3.0)])
  lab = labels(tmp_path, {"a": {"events": [{"kind": "crowd", "start_s": 1.0, "end_s": 3.0}], "ignore": [{"kind": "loitering", "start_s": 9.0, "end_s": 13.0}]}})
  rep = evaluate(run, lab, tol=0.5)
  assert rep["kinds"]["loitering"]["ignored"] == 1 and rep["kinds"]["loitering"]["precision"] == 0.0
  assert [f["start_s"] for f in rep["false_positives"]] == [40.0]
  assert rep["overall"]["precision"] == round(1 / 2, 3)

def test_a_real_label_beats_an_ignore_interval(tmp_path):
  run = make_run(tmp_path, [("a", "loitering", 10.0, 12.0)])
  lab = labels(tmp_path, {"a": {"events": [{"kind": "loitering", "start_s": 10.0, "end_s": 12.0}], "ignore": [{"kind": "loitering", "start_s": 0, "end_s": 100}]}})
  rep = evaluate(run, lab)
  assert rep["overall"]["precision"] == 1.0 and rep["overall"]["ignored"] == 0

def test_results_are_grouped_by_source_so_one_dataset_cannot_hide_the_others(tmp_path):
  run = make_run(tmp_path, [("mot20-01", "loitering", 1.0, 2.0), ("mot20-01", "loitering", 20.0, 21.0), ("00002", "crowd", 1.0, 2.0)], clips=("mot20-01", "00002"))
  lab = labels(tmp_path, {"mot20-01": {"events": [{"kind": "loitering", "start_s": 1.0, "end_s": 2.0}]}, "00002": {"events": [{"kind": "crowd", "start_s": 1.0, "end_s": 2.0}]}})
  rep = evaluate(run, lab)
  assert rep["sources"]["mot20"]["precision"] == 0.5 and rep["sources"]["numbered"]["precision"] == 1.0
  assert rep["by_source_and_kind"]["mot20/loitering"]["precision"] == 0.5
  assert "mot20" in render(rep) and "numbered" in render(rep)

def test_source_can_be_set_explicitly_and_is_derived_otherwise(tmp_path):
  assert family("00123") == "numbered" and family("mot20-02") == "mot20" and family("sample") == "sample"
  assert labels(tmp_path, {"x-1": {"source": "pets", "events": []}})["x-1"]["source"] == "pets"

@pytest.mark.parametrize("event", [{"kind": "teleport", "start_s": 1}, {"kind": "crowd"}, {"kind": "crowd", "start_s": 5, "end_s": 2}])
def test_bad_labels_are_rejected(tmp_path, event):
  p = tmp_path / "l.yaml"
  p.write_text(yaml.safe_dump({"clips": {"a": {"events": [event]}}}))
  with pytest.raises(ValueError): load_labels(p)

def test_new_event_kinds_are_accepted(tmp_path):
  lab = labels(tmp_path, {"a": {"events": [{"kind": k, "start_s": 1} for k in ("dwell", "group", "crowd", "crowd_surge")]}})
  assert [e["kind"] for e in lab["a"]["events"]] == ["dwell", "group", "crowd", "crowd_surge"]