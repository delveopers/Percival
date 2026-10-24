import argparse, json, sys
from collections import Counter
from pathlib import Path

SCALARS = ("observations", "objects_per_frame_mean", "tracks", "tracklets", "orphan_observations")
LISTS = ("possible_fragments", "tracks_big_gap", "tracks_low_coverage", "tracks_hopping", "duplicate_track_pairs")

def load(run_dir):
  p = Path(run_dir) / "qa_report.json"
  if not p.is_file(): sys.exit(f"{p} not found, run qa_report.py on that run first")
  return json.loads(p.read_text(encoding="utf-8"))

def metrics(c):
  m = {k: c.get(k) for k in SCALARS}
  m.update({k: len(c.get(k, [])) for k in LISTS})
  m["anomalies"] = len(c["anomalies"])
  m["warnings"] = sum(f["level"] == "WARN" for f in c["flags"])
  return m

def fmt(v): return "n/a" if v is None else f"{v:.1f}" if isinstance(v, float) else str(v)

def main(argv=None):
  ap = argparse.ArgumentParser(description="Compare the QA reports of two Percival runs")
  ap.add_argument("before", nargs="?")
  ap.add_argument("after", nargs="?")
  ap.add_argument("--root", default="data/exports")
  args = ap.parse_args(argv)

  if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  if args.before and args.after: before, after = Path(args.before), Path(args.after)
  else:
    runs = sorted(p for p in Path(args.root).glob("*") if (p / "qa_report.json").is_file())
    if len(runs) < 2: sys.exit(f"need two runs with a qa_report.json under {args.root}")
    before, after = runs[-2], runs[-1]

  b, a = load(before), load(after)
  print(f"before: {b['run_id']}  [{b['status']}]\nafter : {a['run_id']}  [{a['status']}]")
  prev = {c["clip_id"]: c for c in b["clips"]}

  for c in a["clips"]:
    print(f"\n== {c['clip_id']}")
    if c["clip_id"] not in prev:
      print("  (not in the earlier run)")
      continue
    old, new = metrics(prev[c["clip_id"]]), metrics(c)

    for k in new:
      if old[k] is None or new[k] is None: continue
      d = new[k] - old[k]
      print(f"  {k:26} {fmt(old[k]):>8} -> {fmt(new[k]):>8}   {'' if abs(d) < 1e-9 else ('+' if d > 0 else '') + fmt(d)}")
    ko, kn = Counter(x["kind"] for x in prev[c["clip_id"]]["anomalies"]), Counter(x["kind"] for x in c["anomalies"])
    if ko != kn: print(f"  anomaly kinds: {dict(ko)} -> {dict(kn)}")

  return 0

if __name__ == "__main__":
  sys.exit(main())