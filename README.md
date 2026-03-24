# telemetry-ad

Project for Anomaly Detection on Time-Series telemetry

## Scope
- Datasets: NAB realAWSCloudwatch and SKAB
- Offline: training/evaluation in Docker or local machine
- Online: sliding-window streaming inference on Raspberry Pi 5
- Models: 2 baselines + 2 advanced

## Current dataset mapping
- NAB root: `Datasets/NAB/realAWSCloudwatch/realAWSCloudwatch`
- SKAB train: `Datasets/SKAB/anomaly-free/anomaly-free.csv`
- SKAB test: `Datasets/SKAB/valve1/1.csv`

## Project layout
```text
telemetry-ad/
  Datasets/
  configs/
  src/telemetry_ad/
  scripts/
  artifacts/
  reports/
  logs/
```

## Minimal run commands
```bash
python scripts/train_offline.py --dataset nab --series ec2_cpu_utilization_5f5533
python scripts/train_offline.py --dataset skab --split anomalyfree_vs_valve1_1
python scripts/evaluate_offline.py --dataset nab --series ec2_cpu_utilization_5f5533
python scripts/infer_stream_pi.py --dataset skab --split anomalyfree_vs_valve1_1 --model lstm_ae --source local
```

## Person B model outputs
- Offline training now exports advanced artifacts when `advanced.enabled=true` in config:
  - `lstm_ae.pt`
  - `cnn_ae.pt`
  - `seq_scaler.pkl`
  - thresholds for `lstm_ae` and `cnn_ae` in `thresholds.json`
- Offline evaluation now reports advanced model metrics and plots when artifacts are available.
- Streaming inference supports all models: `zscore`, `iforest`, `lstm_ae`, `cnn_ae`.
- SKAB can optionally recalibrate thresholds from a startup warmup window to reduce train-to-stream score shift, with per-model settings in `configs/skab.yaml`.

## Docker examples (Person B)
```bash
# Train + evaluate SKAB with advanced models
docker build -t telemetry-ad .
docker run --rm -v ${PWD}:/app telemetry-ad python scripts/train_offline.py --dataset skab --split anomalyfree_vs_valve1_1
docker run --rm -v ${PWD}:/app telemetry-ad python scripts/evaluate_offline.py --dataset skab --split anomalyfree_vs_valve1_1

# Simulate Raspberry Pi streaming inference
docker run --rm -v ${PWD}:/app telemetry-ad python scripts/infer_stream_pi.py --dataset skab --split anomalyfree_vs_valve1_1 --model lstm_ae --source local --log-file logs/stream_lstm.csv
```

## FastAPI stream server (for Pi pull mode)
```bash
# Install API dependencies once
pip install -r requirements-api.txt

# Serve telemetry points from test split
python scripts/serve_stream_api.py --dataset skab --split anomalyfree_vs_valve1_1 --host 0.0.0.0 --port 8000

# Pi/client consumes batches from the API instead of a local dataset file
python scripts/infer_stream_pi.py --dataset skab --split anomalyfree_vs_valve1_1 --model lstm_ae --source api --api-base-url http://127.0.0.1:8000 --api-batch-size 32 --log-file logs/stream_lstm_api.csv
```

Endpoints:
- `GET /health`
- `GET /stream/next?cursor=0&batch_size=1`

`infer_stream_pi.py` now supports two streaming modes:
- `--source local`: replay the configured test split locally on the Pi
- `--source api`: pull telemetry batches from the FastAPI stream server

## Raspberry Pi quickstart
```bash
# after cloning this repo on Pi
bash scripts/pi_setup.sh
source .venv/bin/activate

# verify local setup + API connectivity over Tailscale
python scripts/pi_preflight.py --api-base-url http://<tailscale-host-or-ip>:8000

# run API-backed inference from the Pi
python scripts/infer_stream_pi.py --dataset skab --split anomalyfree_vs_valve1_1 --model lstm_ae --source api --api-base-url http://<tailscale-host-or-ip>:8000 --api-batch-size 32 --log-file logs/pi_stream_lstm.csv
```

Detailed deployment checklist:
- `docs/PI_DEPLOYMENT_PLAN.md`

## Notes
- SKAB CSV files are semicolon-delimited (`sep=';'`).
- Preprocessing options in `configs/base.yaml` now support configurable forward fill and interpolation (`interpolate: false|true|linear|time`).
- Keep training offline; copy only required artifacts to Pi for inference.
- If NAB labels are unavailable locally, use weak/operational evaluation mode.
