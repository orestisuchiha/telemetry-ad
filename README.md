# telemetry-ad

End-to-end anomaly detection pipeline for time-series telemetry using NAB and SKAB.

## Current Status
- Offline training and evaluation are implemented for 4 models.
- Local end-to-end API-backed streaming has been validated.
- Raspberry Pi deployment scripts and API pull-mode support are included.
- Raspberry Pi inference has been validated in venv mode; Docker-on-Pi proof is still optional/pending.

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

Equivalent package install:

```bash
pip install .
pip install .[api]
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

## PC-to-Pi Runbook

This is the recommended end-to-end workflow when training on a PC and running inference on a Raspberry Pi.

### 1. On the PC: activate the environment
If you use the project Miniforge environment:

```bash
conda activate telemetry-ad
```

### 2. On the PC: train offline
SKAB example:

```bash
python scripts/train_offline.py --dataset skab --split anomalyfree_vs_valve1_1
python scripts/evaluate_offline.py --dataset skab --split anomalyfree_vs_valve1_1
```

NAB example:

```bash
python scripts/train_offline.py --dataset nab --series ec2_cpu_utilization_5f5533
python scripts/evaluate_offline.py --dataset nab --series ec2_cpu_utilization_5f5533
```

Use offline training and evaluation whenever you want:
- trained artifacts in `artifacts/<dataset>/<variant>/`
- metrics and plots in `reports/<dataset>/<variant>/`
- notebook-ready results before moving to streaming or Pi deployment

### 3. On the PC: start the telemetry API if the Pi will pull streamed data

```bash
python scripts/serve_stream_api.py --dataset skab --split anomalyfree_vs_valve1_1 --host 0.0.0.0 --port 8000
```

Check that it is alive:

```bash
curl http://127.0.0.1:8000/health
```

If you use Tailscale for Pi-to-PC communication, keep note of the PC Tailscale hostname or IP.

### 4. On the Pi: clone and set up the repo

```bash
git clone <repo-url>
cd telemetry-ad
bash scripts/pi_setup.sh
source .venv/bin/activate
```

### 5. On the Pi: run preflight

```bash
python scripts/pi_preflight.py --api-base-url http://<tailscale-host-or-ip>:8000
```

### 6. Copy offline artifacts from the PC to the Pi
At minimum, copy the required variant folder from:
- `artifacts/skab/anomalyfree_vs_valve1_1/`
- or `artifacts/nab/<series>/`

The Pi needs the trained artifact files before inference:
- `thresholds.json`
- `metadata.json`
- baseline model files such as `iforest.pkl` and `zscore_params.pkl`
- advanced model files such as `lstm_ae.pt`, `cnn_ae.pt`, `seq_scaler.pkl`

### 7. Choose local or API streaming on the Pi

`local` mode:
- replays the dataset test split directly from files on the Pi
- best for quick validation when you only want to confirm inference works

`api` mode:
- pulls telemetry batches from the PC over `GET /stream/next`
- best for the full deployment-style demo

### 8. On the Pi: local streaming inference
Example:

```bash
python scripts/infer_stream_pi.py --dataset skab --split anomalyfree_vs_valve1_1 --model iforest --source local --log-file logs/pi_iforest_local.csv
```

Repeat for:
- `zscore`
- `iforest`
- `lstm_ae`
- `cnn_ae`

### 9. On the Pi: API-backed streaming inference
Example:

```bash
python scripts/infer_stream_pi.py --dataset skab --split anomalyfree_vs_valve1_1 --model iforest --source api --api-base-url http://<tailscale-host-or-ip>:8000 --api-batch-size 32 --log-file logs/pi_iforest_api.csv
```

Repeat for:
- `zscore`
- `iforest`
- `lstm_ae`
- `cnn_ae`

### 10. Optional: live TUI during inference
For a richer terminal presentation on PC or Pi:

```bash
python scripts/infer_stream_pi.py --dataset skab --split anomalyfree_vs_valve1_1 --model iforest --source api --api-base-url http://<tailscale-host-or-ip>:8000 --api-batch-size 32 --tui --tui-refresh-every 100 --log-file logs/pi_iforest_api.csv
```

### 11. Offline vs streaming: when to use each

Offline commands:
- `train_offline.py`
- `evaluate_offline.py`

Use them when you want:
- model training
- threshold review
- metrics, plots, explainability, interpretation, and non-stationarity outputs

Streaming commands:
- `serve_stream_api.py`
- `infer_stream_pi.py`

Use them when you want:
- sliding-window inference
- live or replayed detection behavior
- Raspberry Pi validation
- local-vs-API deployment testing

### 12. What to verify after a successful Pi run
Confirm that:
- alert CSVs were written under `logs/`
- the model loaded the correct artifact folder
- local and API runs produce comparable thresholds and alert counts
- the PC API remains reachable during `--source api`

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
Validated path: Python virtual environment on the Pi. Docker-on-Pi remains optional if container proof is needed.

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
- `rolling_detrend_window`
- `ewma_alpha`
- `standardize`

NAB now uses an anomaly-aware split plus rolling detrending in `configs/nab.yaml` so the models are less sensitive to slow level shifts in the AWS CloudWatch series.

### Feature Engineering
Baseline feature engineering supports:
- `feature_engineering.lag_steps`
- `feature_engineering.fft.enabled`
- `feature_engineering.fft.include_dc`

Current defaults:
- `1`
- `5`
- `10`

This means the baseline models now use:
- rolling statistics
- deltas
- lag embeddings
- FFT window energy

### Seasonality Analysis
Offline evaluation can also run an STL-based seasonality decomposition through:
- `feature_engineering.seasonality_analysis.enabled`
- `feature_engineering.seasonality_analysis.period`
- `feature_engineering.seasonality_analysis.signal_column`
- `feature_engineering.seasonality_analysis.robust`

When `statsmodels` is installed, offline evaluation writes:
- `seasonality_summary.json`
- `seasonality_components.csv`
- `seasonality_stl_plot.png`

### Threshold Calibration
The pipeline supports:
- artifact thresholds learned offline
- warmup-percentile recalibration on the target stream

The recommended workflow is:
- learn artifact thresholds offline from the anomaly-score distribution on training data
- optionally recalibrate on the first `warmup_windows` of the target stream when those windows are expected to be mostly normal
- keep the warmup percentile dataset-specific and model-specific in config
- review the offline threshold sweep before finalizing deployment settings

Training-time artifact percentiles can be overridden per model through:
- `training.model_threshold_percentiles`

Streaming-time recalibration is controlled through:
- `inference.threshold_calibration.models.<model>.mode`
- `inference.threshold_calibration.models.<model>.percentile`
- `inference.threshold_calibration.warmup_windows`
- `inference.threshold_calibration.suppress_during_warmup`

Offline evaluation now writes threshold-review artifacts to the report directory:
- `metrics.json`
- `threshold_strategy.json`

These include:
- artifact threshold metadata from training
- effective deployment threshold after calibration
- score summaries
- a percentile sweep over candidate warmup thresholds
- the recommended percentile under any configured alert-rate cap

SKAB currently uses model-specific warmup calibration in `configs/skab.yaml`, with an offline sweep that favors the best F1 subject to a capped alert rate so the advanced models do not over-alert during demos.

NAB currently favors artifact thresholds in `configs/nab.yaml` because the anomaly-aware split and rolling detrending make the offline thresholds more stable than warmup recalibration on the tested AWS series.

### Model-Specific AE Settings
Autoencoder training defaults live under `advanced`, and can now be overridden per model through:
- `advanced.models.lstm_ae`
- `advanced.models.cnn_ae`

This is useful when one architecture benefits from longer training while another starts to overfit under the same dataset conditions.

## Important Notes
- SKAB CSV files are semicolon-delimited.
- Offline artifacts must be regenerated after changing preprocessing or feature-engineering settings.
- Keep training offline; copy only the required artifacts to the Raspberry Pi for inference.
- If NAB labels are unavailable locally, use weak or operational evaluation mode.

## Explainability Outputs
Offline evaluation now writes lightweight explainability artifacts per model:
- `explanations_zscore.json`
- `explanations_iforest.json`
- `explanations_lstm_ae.json`
- `explanations_cnn_ae.json`
- `model_explainability_summary.md`

The explanation style is intentionally simple:
- `zscore`: dominant triggering feature and robust z-score
- `iforest`: most extreme engineered features in the flagged window
- `lstm_ae` / `cnn_ae`: top channels and top timesteps by reconstruction error

## Operational Interpretation Outputs
Offline evaluation also writes heuristic operational interpretation artifacts:
- `interpreted_events_true.json`
- `interpreted_events_zscore.json`
- `interpreted_events_iforest.json`
- `interpreted_events_lstm_ae.json`
- `interpreted_events_cnn_ae.json`
- `operational_interpretation_summary.md`

These labels are intentionally heuristic and are meant to support report discussion, not causal diagnosis. The current mapping is:
- point event -> possible fault or short transient
- contextual event -> possible context-dependent abnormality
- collective event -> possible performance degradation or sustained fault
- long collective event persisting to the end of the horizon -> possible concept drift or setpoint shift

## Non-Stationarity Outputs
Offline evaluation also writes a lightweight non-stationarity review:
- `nonstationarity_summary.json`
- `nonstationarity_summary.md`

The current method is intentionally simple:
- compare fixed artifact thresholds against the deployed calibrated thresholds
- summarize how score distributions shift from training to warmup and full test windows
- report whether recalibration helped or hurt on the evaluated split
