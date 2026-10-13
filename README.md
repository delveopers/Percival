# Percival

Percival is a modular computer-vision pipeline designed to analyze video footage, detect and track objects, identify potentially unusual activities, and generate structured, traceable results. It combines motion analysis, object detection, multi-object tracking, rule-based anomaly detection, and annotated video generation in a configurable processing workflow.

The project is designed around independent components, typed data contracts, reproducible processing runs, and explainable anomaly events.

> **Project status:** [PROTOTYPE] Synthetic-data testing has been performed for the tracklet builder, anomaly rules, event engine, output writers, visualization, logging, and pipeline runner. Real YOLO detector and ByteTrack integration, along with anomaly thresholds on real-world footage, still require validation.

## Project Structure

The following tree represents the architecture documented for Percival. Confirm it against the actual repository when adding or renaming files.

```text
Percival/
├── configs/
│   ├── app.yaml
│   ├── processing.yaml
│   └── models.yaml
│
├── src/
│   └── percival/
│       ├── __init__.py
│       ├── config/
│       │   ├── loader.py
│       │   └── schema.py
│       ├── contracts.py
│       ├── ingestion/
│       │   └── scanner.py
│       ├── preprocessing/
│       │   └── frames.py
│       ├── motion/
│       │   ├── detector.py
│       │   └── render.py
│       ├── detection/
│       │   ├── detector.py
│       │   └── tracker.py
│       ├── tracking/
│       │   └── tracklets.py
│       ├── anomaly/
│       │   ├── engine.py
│       │   └── rules.py
│       ├── storage/
│       │   └── jsonl.py
│       ├── visualization/
│       │   ├── overlay.py
│       │   └── video.py
│       ├── observability/
│       │   ├── logs.py
│       │   └── metrics.py
│       └── pipeline/
│           └── runner.py
│
├── scripts/
│   └── run_pipeline.py
│
├── tests/
├── data/
│   ├── input/
│   ├── samples/
│   └── exports/
│
├── README.md
├── requirements.txt
└── LICENSE
```

## Technology Stack

| Technology | Role |
|---|---|
| Python | Primary implementation language |
| OpenCV | Video decoding, frame processing, and image operations |
| NumPy | Numerical operations and image arrays |
| Pydantic | Typed data models and configuration validation |
| PyYAML | YAML configuration parsing |
| Ultralytics YOLO | Object detection |
| ByteTrack | Multi-object tracking |
| PyTorch | Model inference and device management |
| JSON Lines | Structured output persistence |

The exact compatible package versions should be pinned in the project's dependency file once the target environment has been validated.

## Installation

### Prerequisites

- Python with a version supported by the project's installed dependencies.
- `pip` and a virtual environment.
- OpenCV-compatible video codecs.
- Sufficient memory for video processing and model inference.
- Optional NVIDIA GPU and compatible CUDA/PyTorch installation for accelerated inference.

### 1. Clone the repository

```bash
git clone https://github.com/delveopers/Percival
cd Percival
```

### 2. Create a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If the repository contains a maintained `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The documented implementation uses packages including OpenCV, NumPy, Pydantic, PyYAML, PyTorch, and Ultralytics. If the dependency file is not yet available, install the required packages using compatible versions for your Python and hardware environment.

For GPU inference, install the PyTorch build compatible with your CUDA environment by following the official [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

### 4. Prepare the configuration

Ensure the `configs/` directory contains:

- `app.yaml`
- `processing.yaml`
- `models.yaml`

Review the input directory, output directory, frame sampling, model weights, detection classes, tracking parameters, and anomaly thresholds before running the pipeline.

### 5. Add input videos

Place supported video files in the directory configured by `input_dir`. The exact path and accepted extensions are controlled by the application configuration.

## Running Percival

The documented command-line entry point is `scripts/run_pipeline.py`.

### Run the full pipeline

From the repository root:

```bash
python scripts/run_pipeline.py
```

By default, the CLI looks for configuration files in `configs/`.

### Specify a configuration directory

```bash
python scripts/run_pipeline.py --config configs
```

Replace `configs` with the path to another directory containing the required YAML files.

## Current Limitations

Percival remains a prototype and has several known limitations.

- **Fixed-camera assumption:** Camera movement can invalidate spatial assumptions used by anomaly rules.
- **Model integration validation:** The documented YOLO detector and real ByteTrack integration were not validated in the original sandbox environment.
- **Rule calibration:** Default thresholds have not been established through systematic evaluation on representative real-world footage.
- **Detection errors:** Missed detections, occlusion, and unstable bounding boxes can affect tracking and anomaly results.
- **Track ID switches:** A single physical object may be represented by multiple tracklets if identity association fails.
- **Abandoned-object ambiguity:** A missed owner detection can lead to a false abandoned-object event.
- **Scene-dependent speed statistics:** Speed-outlier detection assumes that most observations represent normal behavior.
- **Sampling trade-offs:** Aggressive frame sampling reduces processing cost but can make object association more difficult.
- **Memory use:** Open tracks may retain extensive histories for long clips.
- **Heuristic confidence:** Anomaly confidence scores require labeled evaluation before they can be interpreted probabilistically.

Results should be treated as candidate events for review, not as definitive determinations of intent or wrongdoing.

## Contributing

Contributions are welcome, particularly those that improve reliability, reproducibility, test coverage, and model integration.

### How to contribute

1. Fork the repository.
2. Create a feature branch.
3. Implement a focused change.
4. Add or update tests for the changed behavior.
5. Run the relevant tests and verify that existing behavior remains intact.
6. Submit a pull request describing the motivation, implementation, and validation results.

### Development guidelines

- Keep processing stages modular and independently testable.
- Use the existing typed data contracts rather than passing loosely structured dictionaries between stages.
- Keep configuration validation centralized.
- Use timestamps in seconds for time-based anomaly logic.
- Preserve `run_id`, `clip_id`, frame provenance, and model version information in persisted records.
- Ensure video captures, writers, and output files are closed on errors and interruption.
- Add synthetic tests for behavioral logic whenever possible.
- Isolate dependencies on external detector and tracker APIs behind adapters.
- Document new configuration options, their defaults, and their expected effects.
- Avoid committing video datasets, model weights, credentials, generated exports, or local environment files.

## License

> This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

**Important:** Ultralytics and other third-party dependencies have their own licensing terms. The documented integration with Ultralytics may introduce AGPL licensing considerations, depending on the version, usage, and distribution model. Review applicable dependency licenses before distributing Percival or offering it as a service.
