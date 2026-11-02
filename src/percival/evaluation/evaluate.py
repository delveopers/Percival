import argparse, json, sys
from collections import defaultdict
from pathlib import Path
from typing import get_args
import yaml
from percival.contracts import AnomalyKind
from percival.qa.report import load

KINDS = get_args(AnomalyKind)

def family(cid): return "numbered" if cid.isdigit() else cid.split("-")[0]

def parse_events(cid, events, what):
  out = []
  for e in events or []:
    if e.get("kind") not in KINDS: raise ValueError(f"{cid}: unknown kind {e.get('kind')!r} in {what}, valid: {list(KINDS)}")
    if not isinstance(e.get("start_s"), (int, float)): raise ValueError(f"{cid}: every event in {what} needs a numeric start_s")
    if e.get("end_s") is not None and not isinstance(e["end_s"], (int, float)): raise ValueError(f"{cid}: end_s must be a number or null")
    if e.get("end_s") is not None and e["end_s"] < e["start_s"]: raise ValueError(f"{cid}: end_s is before start_s")
    out.append({"kind": e["kind"], "start_s": float(e["start_s"]), "end_s": None if e.get("end_s") is None else float(e["end_s"])})
  return out

def load_labels(path):
  data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
  if not isinstance(data, dict) or not isinstance(data.get("clips"), dict): raise ValueError("labels file must have a top-level 'clips' mapping")
  return {cid: {"reviewed": bool((body or {}).get("reviewed", True)), "source": (body or {}).get("source") or family(cid), "events": parse_events(cid, (body or {}).get("events"), "events"), "ignore": parse_events(cid, (body or {}).get("ignore"), "ignore")} for cid, body in data["clips"].items()}

def overlaps(pred, label, tol):
  end = float("inf") if label["end_s"] is None else label["end_s"]
  return pred["kind"] == label["kind"] and pred["start_s"] <= end + tol and label["start_s"] <= pred["end_s"] + tol

def ratio(a, b): return None if b <= 0 else round(a / b, 3)
FIELDS = ("labels", "predictions", "matched_labels", "matched_predictions", "ignored")

def summarize(counters):
  tot = {f: sum(c[f] for c in counters) for f in FIELDS}
  return {**tot, "precision": ratio(tot["matched_predictions"], tot["predictions"] - tot["ignored"]), "recall": ratio(tot["matched_labels"], tot["labels"])}

def evaluate(run_dir, labels, tol=2.0):
  run_dir = Path(run_dir)
  run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
  completed = {e["clip_id"] for e in run["clips"] if e["status"] == "completed"}
  anomalies = load(run_dir, "anomalies")
  tally = defaultdict(lambda: defaultdict(int))
  false_pos, missed, skipped, unreviewed = [], [], [], []
  for cid, lab in labels.items():
    if cid not in completed:
      skipped.append(cid)
      continue
    if not lab["reviewed"]:
      unreviewed.append(cid)
      continue
    preds = [{"kind": a["kind"], "start_s": a["start_s"], "end_s": a["end_s"], "confidence": round(a["confidence"], 3)} for a in anomalies if a["clip_id"] == cid]

    for p in preds:
      hit = any(overlaps(p, l, tol) for l in lab["events"])
      ign = not hit and any(overlaps(p, l, tol) for l in lab["ignore"])
      t = tally[(lab["source"], p["kind"])]
      t["predictions"] += 1; t["matched_predictions"] += hit; t["ignored"] += ign
      if not hit and not ign: false_pos.append({"clip_id": cid, "source": lab["source"], **p})
    for l in lab["events"]:
      hit = any(overlaps(p, l, tol) for p in preds)
      t = tally[(lab["source"], l["kind"])]
      t["labels"] += 1; t["matched_labels"] += hit
      if not hit: missed.append({"clip_id": cid, "source": lab["source"], **l})
  full = {k: {f: t[f] for f in FIELDS} for k, t in tally.items()}
  group = lambda idx: {name: summarize([c for k, c in full.items() if k[idx] == name]) for name in sorted({k[idx] for k in full})}
  return {"run_id": run["run_id"], "tolerance_s": tol, "kinds": group(1), "sources": group(0), "by_source_and_kind": {f"{s}/{k}": summarize([c]) for (s, k), c in sorted(full.items())}, "overall": summarize(list(full.values())), "false_positives": false_pos, "missed": missed, "skipped_clips": skipped, "unreviewed_clips": unreviewed}

def template(run_dir):
  run_dir = Path(run_dir)
  run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
  anomalies = load(run_dir, "anomalies")
  clips = {e["clip_id"]: {"reviewed": False, "source": family(e["clip_id"]), "events": [{"kind": a["kind"], "start_s": round(a["start_s"], 2), "end_s": round(a["end_s"], 2)} for a in anomalies if a["clip_id"] == e["clip_id"]], "ignore": []} for e in run["clips"] if e["status"] == "completed"}
  return {"clips": clips}

def fmt(v): return "  n/a" if v is None else f"{v:5.2f}"

def table(title, rows):
  out = [f"{title:22} {'labels':>6} {'preds':>6} {'ignored':>7} {'recall':>7} {'precision':>10}"]
  for name, c in rows: out.append(f"{name:22} {c['labels']:>6} {c['predictions']:>6} {c['ignored']:>7} {fmt(c['recall']):>7} {fmt(c['precision']):>10}")
  return out

def render(rep, limit=40):
  out = [f"Evaluation of run {rep['run_id']} (tolerance {rep['tolerance_s']}s)", ""] + table("rule", list(rep["kinds"].items()) + [("ALL", rep["overall"])]) + [""] + table("source", list(rep["sources"].items()))
  for title, items, line in (("False positives (predicted, not labeled, not ignored)", rep["false_positives"], lambda f: f"  {f['clip_id'][:36]:36} {f['kind']} {f['start_s']:.1f}s-{f['end_s']:.1f}s conf {f['confidence']}"), ("Missed (labeled, not predicted)", rep["missed"], lambda m: f"  {m['clip_id'][:36]:36} {m['kind']} {m['start_s']:.1f}s-{'end' if m['end_s'] is None else format(m['end_s'], '.1f') + 's'}")):
    if items: out += ["", f"{title}: {len(items)}"] + [line(i) for i in items[:limit]] + ([f"  ... {len(items) - limit} more in eval_report.json"] if len(items) > limit else [])
  if rep["unreviewed_clips"]: out += ["", f"NOT scored, still reviewed: false ({len(rep['unreviewed_clips'])} clips): set reviewed: true after you have checked a clip"]
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