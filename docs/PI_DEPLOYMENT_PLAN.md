# Raspberry Pi Deployment Plan (Remaining Steps)

This is the final roadmap phase after model implementation.

## 1) Backend/API Host (PC or Server)

Goal: expose telemetry points over HTTP so the Raspberry Pi can pull them.

1. Start FastAPI telemetry stream service:
   - `python scripts/serve_stream_api.py --dataset skab --split anomalyfree_vs_valve1_1 --host 0.0.0.0 --port 8000`
2. Verify local health:
   - `curl http://127.0.0.1:8000/health`
3. Note your Tailscale hostname/IP (example: `http://100.x.y.z:8000` or `http://my-pc-tailnet:8000`).

## 2) Raspberry Pi Setup

Goal: clone repo, create venv, install dependencies, run preflight.

1. Clone and setup:
   - `bash scripts/pi_setup.sh`
2. Run preflight + API reachability test:
   - `python scripts/pi_preflight.py --api-base-url http://<tailscale-host-or-ip>:8000`

## 3) Artifact Transfer to Pi

Goal: inference on Pi uses offline-trained artifacts.

1. Ensure these exist for your target dataset/variant:
   - `artifacts/<dataset>/<variant>/thresholds.json`
   - model artifacts (`iforest.pkl`, `zscore_params.pkl`, `lstm_ae.pt`, `cnn_ae.pt`, `seq_scaler.pkl`)
2. Copy repo/artifacts to Pi (git pull, rsync, or scp).

## 4) Pi Inference Validation

Goal: confirm local inference and alert logging work.

1. Local dataset streaming simulation:
   - `python scripts/infer_stream_pi.py --dataset skab --split anomalyfree_vs_valve1_1 --model lstm_ae --log-file logs/stream_lstm.csv`
2. Repeat for all models:
   - `zscore`, `iforest`, `lstm_ae`, `cnn_ae`
3. Confirm logs are written in `logs/`.

## 5) Communication Validation via Tailscale

Goal: Pi can reach API host over tailnet.

1. Check Tailscale up on both devices:
   - `tailscale status`
2. Ping API host from Pi:
   - `ping <tailscale-host>`
3. API check from Pi:
   - `curl http://<tailscale-host-or-ip>:8000/health`
   - `curl "http://<tailscale-host-or-ip>:8000/stream/next?batch_size=1"`

## 6) Final Demo Path

1. Offline train on PC.
2. Start API server on PC.
3. Pi pulls data over Tailscale.
4. Pi runs local model inference.
5. Alerts logged/shipped for presentation.
