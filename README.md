# telemetry-ad

End-to-end anomaly detection pipeline for time-series telemetry using NAB and SKAB.

## Current Status
- Offline training and evaluation are implemented for 4 models.
- Local end-to-end API-backed streaming has been validated.
- Raspberry Pi deployment scripts and API pull-mode support are included.
- Real hardware validation on the Raspberry Pi is still pending.

## Scope
- Datasets:
  - NAB `realAWSCloudwatch`
  - SKAB
- Offline stage:
  - training
  - evaluation
  - plots and metrics
- Online stage:
  - sliding-window inference
  - local replay or API-backed streaming
  - Raspberry Pi deployment path
- Models:
  - `zscore`
  - `iforest`
  - `lstm_ae`
  - `cnn_ae`

## Dataset Mapping
- NAB root: `Datasets/NAB/realAWSCloudwatch/realAWSCloudwatch`
- NAB labels: `Datasets/NAB/labels/combined_windows.json`
- SKAB train: `Datasets/SKAB/anomaly-free/anomaly-free.csv`
- SKAB test: `Datasets/SKAB/valve1/1.csv`

## Project Layout
```text
telemetry-ad/
  Datasets/
  configs/
  docs/
  scripts/
  src/telemetry_ad/
  artifacts/
  reports/
  logs/
```

## Setup
Install the main dependencies and the API extras:

```bash
pip install -r requirements.txt
pip install -r requirements-api.txt
```

## Quickstart

### SKAB: train and evaluate
```bash
python scripts/train_offline.py --dataset skab --split anomalyfree_vs_valve1_1
python scripts/evaluate_offline.py --dataset skab --split anomalyfree_vs_valve1_1
```

### SKAB: local streaming replay
```bash
python scripts/infer_stream_pi.py --dataset skab --split anomalyfree_vs_valve1_1 --model iforest --source local --log-file logs/stream_iforest_local.csv
```

### SKAB: API-backed streaming
In one terminal:

```bash
python scripts/serve_stream_api.py --dataset skab --split anomalyfree_vs_valve1_1 --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
python scripts/infer_stream_pi.py --dataset skab --split anomalyfree_vs_valve1_1 --model iforest --source api --api-base-url http://127.0.0.1:8000 --api-batch-size 32 --log-file logs/stream_iforest_api.csv
```

### NAB example
```bash
python scripts/train_offline.py --dataset nab --series ec2_cpu_utilization_5f5533
python scripts/evaluate_offline.py --dataset nab --series ec2_cpu_utilization_5f5533
```

## Streaming Modes
`scripts/infer_stream_pi.py` supports:
- `--source local`
  Replays the configured test split directly from local files.
- `--source api`
  Pulls telemetry batches from `GET /stream/next`.

Alert logs are written as CSV with:
- `timestamp`
- `model`
- `score`
- `threshold`

## FastAPI Endpoints
- `GET /health`
- `GET /stream/next?cursor=0&batch_size=1`

## Docker Examples
```bash
docker build -t telemetry-ad .

docker run --rm -v ${PWD}:/app telemetry-ad python scripts/train_offline.py --dataset skab --split anomalyfree_vs_valve1_1
docker run --rm -v ${PWD}:/app telemetry-ad python scripts/evaluate_offline.py --dataset skab --split anomalyfree_vs_valve1_1
docker run --rm -v ${PWD}:/app telemetry-ad python scripts/infer_stream_pi.py --dataset skab --split anomalyfree_vs_valve1_1 --model iforest --source local --log-file logs/stream_iforest_local.csv
```

## Raspberry Pi Quickstart
After cloning the repo on the Pi:

```bash
bash scripts/pi_setup.sh
source .venv/bin/activate
python scripts/pi_preflight.py --api-base-url http://<tailscale-host-or-ip>:8000
python scripts/infer_stream_pi.py --dataset skab --split anomalyfree_vs_valve1_1 --model iforest --source api --api-base-url http://<tailscale-host-or-ip>:8000 --api-batch-size 32 --log-file logs/pi_stream_iforest.csv
```

Detailed checklist:
- `docs/PI_DEPLOYMENT_PLAN.md`

## Configuration Notes

### Preprocessing
`configs/base.yaml` supports:
- `resample_rule`
- `forward_fill`
- `interpolate`
- `interpolate_limit_direction`
- `ewma_alpha`
- `standardize`

### Feature Engineering
Baseline feature engineering supports configurable lag embeddings through:
- `feature_engineering.lag_steps`

Current defaults:
- `1`
- `5`
- `10`

### Threshold Calibration
The pipeline supports:
- artifact thresholds learned offline
- warmup-percentile recalibration on the target stream

SKAB currently uses model-specific warmup calibration in `configs/skab.yaml` to reduce train-to-stream score shift.

## Important Notes
- SKAB CSV files are semicolon-delimited.
- Offline artifacts must be regenerated after changing preprocessing or feature-engineering settings.
- Keep training offline; copy only the required artifacts to the Raspberry Pi for inference.
- If NAB labels are unavailable locally, use weak or operational evaluation mode.
