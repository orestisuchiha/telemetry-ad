import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telemetry_ad.orchestration import train_offline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline model training")
    parser.add_argument("--dataset", choices=["nab", "skab"], required=True)
    parser.add_argument("--series", default=None, help="NAB series name")
    parser.add_argument("--split", default=None, help="SKAB split name")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-dir", default="artifacts")
    return parser.parse_args()


if __name__ == "__main__":
    train_offline(parse_args())
