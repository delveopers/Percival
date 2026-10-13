import argparse
import sys

from percival.config import load_config, ConfigError
from percival.ingestion import NoClipsError
from percival.pipeline import run


def main():
    p = argparse.ArgumentParser(
        description="Run the Percival pipeline over a folder of clips"
    )
    p.add_argument(
        "--config",
        default="configs",
        help="directory containing app.yaml, processing.yaml, models.yaml",
    )
    args = p.parse_args()

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        sys.exit(f"config error: {e}")

    try:
        out = run(cfg)
    except NoClipsError as e:
        sys.exit(f"error: {e}")

    print(out)


if __name__ == "__main__":
    main()