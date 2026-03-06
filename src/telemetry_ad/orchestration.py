from pathlib import Path

from telemetry_ad.utils.seed import set_global_seed


def train_offline(args) -> None:
    set_global_seed(42)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"[train] dataset={args.dataset} series={args.series} split={args.split}")
    print("[train] Skeleton in place. Implement dataset loading, preprocessing, training, and artifact export.")


def evaluate_offline(args) -> None:
    Path(args.reports_dir).mkdir(parents=True, exist_ok=True)
    print(f"[eval] dataset={args.dataset} series={args.series} split={args.split}")
    print("[eval] Skeleton in place. Implement metrics, plots, and anomaly taxonomy reporting.")


def infer_stream_pi(args) -> None:
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    print(f"[infer] dataset={args.dataset} model={args.model} series={args.series} split={args.split}")
    print("[infer] Skeleton in place. Implement ring buffer streaming inference and alert logging.")
