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
    parser.add_argument(
        "--source",
        choices=["local", "api"],
        default="local",
        help="Read telemetry from the local test split or pull it from the streaming API",
    )
    parser.add_argument("--series", default=None, help="NAB series name")
    parser.add_argument("--split", default=None, help="SKAB split name")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--log-file", default="logs/stream_alerts.csv")
    parser.add_argument("--api-base-url", default=None, help="Base URL for API mode, for example http://127.0.0.1:8000")
    parser.add_argument("--api-batch-size", type=int, default=32, help="Rows per /stream/next request in API mode")
    parser.add_argument("--api-start-cursor", type=int, default=0, help="Initial API cursor in API mode")
    parser.add_argument("--api-timeout", type=float, default=10.0, help="HTTP timeout in seconds for API mode")
    parser.add_argument("--tui", action="store_true", help="Show a lightweight live terminal dashboard during inference")
    parser.add_argument(
        "--tui-refresh-every",
        type=int,
        default=20,
        help="Refresh the live dashboard every N scored windows when --tui is enabled",
    )
    return parser.parse_args()


if __name__ == "__main__":
    infer_stream_pi(parse_args())
