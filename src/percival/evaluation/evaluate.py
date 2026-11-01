import argparse, json, sys
from collections import defaultdict
from pathlib import Path
from typing import get_args
import yaml
from percival.contracts import AnomalyKind
from percival.qa.report import load

KINDS = get_args(AnomalyKind)

def load_labels(path):
  data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
  if not isinstance(data, dict) or not isinstance(data.get("clips"), dict): raise ValueError("labels file must have a top-level 'clips' mapping")
  out = {}

  for cid, body in data["clips"].items():
    events = (body or {}).get("events") or []

    for e in events:
      if e.get("kind") not in KINDS: raise ValueError(f"{cid}: unknown kind {e.get('kind')!r}, valid: {list(KINDS)}")
      if not isinstance(e.get("start_s"), (int, float)): raise ValueError(f"{cid}: every event needs a numeric start_s")
      if e.get("end_s") is not None and not isinstance(e["end_s"], (int, float)): raise ValueError(f"{cid}: end_s must be a number or null")
      if e.get("end_s") is not None and e["end_s"] < e["start_s"]: raise ValueError(f"{cid}: end_s is before start_s")
    out[cid] = [{"kind": e["kind"], "start_s": float(e["start_s"]), "end_s": None if e.get("end_s") is None else float(e["end_s"])} for e in events]

  return out

def overlaps(pred, label, tol):
  end = float("inf") if label["end_s"] is None else label["end_s"]
  return pred["kind"] == label["kind"] and pred["start_s"] <= end + tol and label["start_s"] <= pred["end_s"] + tol

def ratio(a, b): return None if b == 0 else round(a / b, 3)

def evaluate(run_dir, labels, tol=2.0):
  run_dir = Path(run_dir)
  run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
  completed = {e["clip_id"] for e in run["clips"] if e["status"] == "completed"}
  anomalies = load(run_dir, "anomalies")
  counts = defaultdict(lambda: {"labels": 0, "predictions": 0, "matched_labels": 0, "matched_predictions": 0})
  false_pos, missed, skipped = [], [], []

  for cid, events in labels.items():
    if cid not in completed:
      skipped.append(cid)
      continue
    preds = [{"kind": a["kind"], "start_s": a["start_s"], "end_s": a["end_s"], "confidence": round(a["confidence"], 3)} for a in anomalies if a["clip_id"] == cid]

    for p in preds:
      hit = any(overlaps(p, l, tol) for l in events)
      counts[p["kind"]]["predictions"] += 1
      counts[p["kind"]]["matched_predictions"] += hit
      if not hit: false_pos.append({"clip_id": cid, **p})

    for l in events:
      hit = any(overlaps(p, l, tol) for p in preds)
      counts[l["kind"]]["labels"] += 1
      counts[l["kind"]]["matched_labels"] += hit
      if not hit: missed.append({"clip_id": cid, **l})

  kinds = {k: {**c, "precision": ratio(c["matched_predictions"], c["predictions"]), "recall": ratio(c["matched_labels"], c["labels"])} for k, c in sorted(counts.items())}
  tot = {f: sum(c[f] for c in counts.values()) for f in ("labels", "predictions", "matched_labels", "matched_predictions")}
  overall = {**tot, "precision": ratio(tot["matched_predictions"], tot["predictions"]), "recall": ratio(tot["matched_labels"], tot["labels"])}

  return {"run_id": run["run_id"], "tolerance_s": tol, "kinds": kinds, "overall": overall, "false_positives": false_pos, "missed": missed, "skipped_clips": skipped}

def template(run_dir):
  run_dir = Path(run_dir)
  run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
  anomalies = load(run_dir, "anomalies")
  clips = {e["clip_id"]: {"events": [{"kind": a["kind"], "start_s": round(a["start_s"], 2), "end_s": round(a["end_s"], 2)} for a in anomalies if a["clip_id"] == e["clip_id"]]} for e in run["clips"] if e["status"] == "completed"}

  return {"clips": clips}

def fmt(v): return "  n/a" if v is None else f"{v:5.2f}"

def render(rep):
  out = [f"Evaluation of run {rep['run_id']} (tolerance {rep['tolerance_s']}s)", "", f"{'rule':20} {'labels':>6} {'preds':>6} {'recall':>7} {'precision':>10}"]
  for k, c in rep["kinds"].items(): out.append(f"{k:20} {c['labels']:>6} {c['predictions']:>6} {fmt(c['recall']):>7} {fmt(c['precision']):>10}")
  o = rep["overall"]
  out.append(f"{'ALL':20} {o['labels']:>6} {o['predictions']:>6} {fmt(o['recall']):>7} {fmt(o['precision']):>10}")
  if rep["false_positives"]: out += ["", "False positives (predicted, not labeled):"] + [f"  {f['clip_id'][:40]}  {f['kind']} {f['start_s']}s-{f['end_s']}s conf {f['confidence']}" for f in rep["false_positives"]]
  if rep["missed"]: out += ["", "Missed (labeled, not predicted):"] + [f"  {m['clip_id'][:40]}  {m['kind']} {m['start_s']}s-{m['end_s']}s" for m in rep["missed"]]
  if rep["skipped_clips"]: out += ["", f"Labeled clips not completed in this run (skipped): {rep['skipped_clips']}"]

  return "\n".join(out)

def latest(root):
  runs = sorted(p for p in Path(root).glob("*") if (p / "run.json").is_file())
  if not runs: sys.exit(f"no runs found under {root}")
  return runs[-1]

def main(argv=None):
  ap = argparse.ArgumentParser(description="Score a run's anomalies against hand-made labels")
  ap.add_argument("run_dir", nargs="?")
  ap.add_argument("--labels", default="data/labels.yaml")
  ap.add_argument("--tolerance", type=float, default=2.0)
  ap.add_argument("--template", help="write a labels file pre-filled with this run's predictions, then edit it")
  ap.add_argument("--root", default="data/exports")

  args = ap.parse_args(argv)
  if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  run_dir = Path(args.run_dir) if args.run_dir else latest(args.root)

  if args.template:
    Path(args.template).parent.mkdir(parents=True, exist_ok=True)
    Path(args.template).write_text(yaml.safe_dump(template(run_dir), sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"template written to {args.template}: delete the events that are wrong, add the ones that were missed")
    return 0

  try: labels = load_labels(args.labels)
  except (OSError, ValueError) as e: sys.exit(f"labels error: {e}")
  rep = evaluate(run_dir, labels, args.tolerance)
  (run_dir / "eval_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
  print(render(rep))

  return 0