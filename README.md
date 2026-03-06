# telemetry-ad

Minimal but complete organization for Project 2 (Telemetry Anomaly Detection).

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
python scripts/infer_stream_pi.py --dataset skab --split anomalyfree_vs_valve1_1 --model lstm_ae
```

## Notes
- SKAB CSV files are semicolon-delimited (`sep=';'`).
- Keep training offline; copy only required artifacts to Pi for inference.
- If NAB labels are unavailable locally, use weak/operational evaluation mode.
