# telemetry-ad

End-to-end anomaly detection for time-series telemetry, from offline model development on a laptop to streaming inference on a Raspberry Pi.

## Purpose
This project was built as a full machine-learning systems exercise rather than only a notebook experiment. The goal was to design, evaluate, and deploy an anomaly-detection pipeline that can:

- learn from historical telemetry offline
- compare simple and advanced anomaly detectors fairly
- evaluate both statistical performance and alert quality
- move trained artifacts to an edge device
- run streaming inference on a Raspberry Pi while the laptop serves telemetry over an API

The project uses two different anomaly-detection settings:

- `SKAB`, which is closer to a controlled industrial benchmark
- `NAB realAWSCloudwatch`, which is noisier and closer to cloud operations data

## What We Built
The repository implements a complete workflow with four models:

- `Z-score`
- `Isolation Forest`
- `LSTM Autoencoder`
- `CNN Autoencoder`

The pipeline includes:

- dataset loading for SKAB and NAB
- preprocessing and feature engineering
- offline training and evaluation
- plots, metrics, and report generation
- FastAPI-based telemetry streaming
- Raspberry Pi inference using saved artifacts
- private laptop-to-Pi communication through Tailscale

## How We Worked
The project was developed in three stages.

### 1. Offline model development
We first built the local pipeline on the laptop. This stage covered data loading, preprocessing, threshold calibration, training, and evaluation. The main objective was to make the experiments reproducible and to ensure the same assumptions used in training could also be used later during streaming inference.

### 2. Model comparison
We evaluated four models across SKAB and multiple NAB AWS telemetry series. The comparison was not based only on a single metric. We looked at:

- point-level detection quality
- event-level alert coherence
- PR-AUC for ranking quality

This matters because a model can score well statistically while still producing noisy or fragmented alerts in practice.

### 3. Edge deployment
After offline evaluation, the trained artifacts were copied to a Raspberry Pi. The laptop hosts a FastAPI service that streams telemetry batches, and the Pi pulls those batches over Tailscale and performs local inference. This creates a realistic split between development and deployment:

- laptop: train, evaluate, serve telemetry
- Raspberry Pi: load artifacts, infer, log alerts

## Results
The main outcome is that the best model depends on the dataset regime.

### Results Snapshot
| Dataset / Variant | Best Point-F1 Model | Point F1 | Best PR-AUC Model | PR-AUC | Best Event-F1 Model | Event F1 |
| --- | --- | --- | --- | --- | --- | --- |
| SKAB `valve1/1` | Z-score | 0.677 | LSTM-AE | 0.695 | Z-score | 1.000 |
| NAB `rds_cpu` | LSTM-AE | 0.571 | LSTM-AE | 0.493 | All models | 1.000 |
| NAB `ec2_cpu` | LSTM-AE | 0.302 | Z-score | 0.759 | Z-score / LSTM-AE / CNN-AE | 1.000 |
| NAB `ec2_network` | LSTM-AE | 0.613 | LSTM-AE | 0.950 | LSTM-AE / CNN-AE | 1.000 |

### Results Discussion
Two patterns stand out.

First, simple calibrated methods remain competitive. On SKAB, Z-score produced the strongest operational behavior because it turned the anomaly into one coherent alert. That is an important engineering result: the most deployable model is not always the most complex one.

Second, temporal autoencoders were stronger on the harder NAB telemetry. On the noisier AWS cloud series, anomalies were better captured as temporal patterns than as isolated outlier points. LSTM-AE was especially strong on `ec2_network`, where it achieved the best point-level F1 and PR-AUC while also producing one coherent event.

Isolation Forest was a useful nonlinear baseline, but it often fragmented alerts into many short events. This made it less reliable operationally even when its point-level score looked acceptable.

Overall, the project showed that model choice should depend on deployment goals:

- if stable alerting is the priority, a calibrated simple model may be enough
- if the anomaly is spread across time, sequence models are usually the better choice
- event-level behavior is just as important as point-level accuracy

## Deployment Summary
The deployment path is intentionally practical.

1. Train and evaluate models on the laptop.
2. Copy the saved artifacts to the Raspberry Pi.
3. Start the FastAPI telemetry service on the laptop.
4. Use Tailscale so the Pi can reach the laptop privately.
5. Run streaming inference on the Pi with the saved models.
6. Write anomaly alerts to CSV logs for inspection and reporting.

This matters because the project is not only a model comparison exercise. It demonstrates that the offline workflow can be turned into an edge inference system with a clear separation of responsibilities.

## Tech Stack
- Python
- pandas
- scikit-learn
- PyTorch
- FastAPI
- Tailscale
- Raspberry Pi

## Repository Layout
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

## Quick Start

### Install
```bash
pip install -r requirements.txt
pip install -r requirements-api.txt
```

### Train and evaluate on SKAB
```bash
python scripts/train_offline.py --dataset skab --split anomalyfree_vs_valve1_1
python scripts/evaluate_offline.py --dataset skab --split anomalyfree_vs_valve1_1
```

### Train and evaluate on NAB
```bash
python scripts/train_offline.py --dataset nab --series ec2_cpu_utilization_5f5533
python scripts/evaluate_offline.py --dataset nab --series ec2_cpu_utilization_5f5533
```

### Start the telemetry API on the laptop
```bash
python scripts/serve_stream_api.py --dataset skab --split anomalyfree_vs_valve1_1 --host 0.0.0.0 --port 8000
```

### Run streaming inference on the Raspberry Pi
```bash
bash scripts/pi_setup.sh
source .venv/bin/activate
python scripts/pi_preflight.py --api-base-url http://<tailscale-host-or-ip>:8000
python scripts/infer_stream_pi.py --dataset skab --split anomalyfree_vs_valve1_1 --model iforest --source api --api-base-url http://<tailscale-host-or-ip>:8000 --api-batch-size 32 --log-file logs/pi_stream_iforest.csv
```

## Project Notes
- SKAB uses a clean train/test split with anomaly-free training data and a fault-containing test file.
- NAB uses AWS cloud telemetry and is harder because it is noisier and less stationary.
- Offline artifacts must be regenerated if preprocessing or feature settings change.
- The Raspberry Pi does not train models; it only loads artifacts and performs inference.

## Documentation
- Final report: [docs/Final_Report.docx](docs/Final_Report.docx)
- Final report source: [docs/Final_Report.md](docs/Final_Report.md)
- Pi deployment checklist: [docs/PI_DEPLOYMENT_PLAN.md](docs/PI_DEPLOYMENT_PLAN.md)

## Why This Project Matters
From an engineering perspective, this project demonstrates more than model training. It shows the ability to:

- move from experimentation to a reproducible pipeline
- compare models with meaningful evaluation criteria
- reason about deployment tradeoffs rather than only benchmark scores
- connect local ML work to an edge device over a real communication path
- document the work clearly for technical and non-technical audiences
