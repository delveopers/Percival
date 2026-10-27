import argparse, sys
from pathlib import Path
from percival.config import load_config, ConfigError
from percival.pipeline import replay

def latest(root):
  runs = sorted(p for p in Path(root).glob("*") if (p / "frames.jsonl").is_file())
  if not runs: sys.exit(f"no run with a frames.jsonl under {root}, run the pipeline with output.save_detections: true first")
  return runs[-1]

def main():
  ap = argparse.ArgumentParser(description="Re-run tracking and anomaly rules from the cached detections of an earlier run")
  ap.add_argument("run_dir", nargs="?")
  ap.add_argument("--config", default="configs")
  args = ap.parse_args()
  try: cfg = load_config(args.config)
  except ConfigError as e: sys.exit(f"config error: {e}")
  try: out = replay(cfg, args.run_dir or latest(cfg.paths.output_dir))
  except ValueError as e: sys.exit(f"error: {e}")
  print(out)

if __name__ == "__main__":
  main()