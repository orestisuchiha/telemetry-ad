import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telemetry_ad.orchestration import infer_stream_pi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming inference on Raspberry Pi")
    parser.add_argument("--dataset", choices=["nab", "skab"], required=True)
    parser.add_argument("--model", choices=["zscore", "iforest", "lstm_ae", "cnn_ae"], required=True)
    parser.add_argument("--series", default=None, help="NAB series name")
    parser.add_argument("--split", default=None, help="SKAB split name")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--log-file", default="logs/stream_alerts.csv")
    return parser.parse_args()


if __name__ == "__main__":
    infer_stream_pi(parse_args())
