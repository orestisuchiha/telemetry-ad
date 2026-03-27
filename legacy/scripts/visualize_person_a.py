import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and visualize Person A baseline results")
    parser.add_argument("--reports-dir", default="reports", help="Root reports directory")
    parser.add_argument("--output-dir", default="reports/person_a_summary", help="Output directory for summary files")
    return parser.parse_args()


def _collect_metrics(reports_dir: Path) -> pd.DataFrame:
    rows = []
    for dataset_dir in reports_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        if dataset_dir.name in {"experiments", "person_a_summary"}:
            continue
        for variant_dir in dataset_dir.iterdir():
            metrics_path = variant_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            with metrics_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            for model in ("zscore", "iforest"):
                block = payload.get(model, {})
                rows.append(
                    {
                        "dataset": payload.get("dataset", dataset_dir.name),
                        "variant": payload.get("variant", variant_dir.name),
                        "model": model,
                        "precision": block.get("precision"),
                        "recall": block.get("recall"),
                        "f1": block.get("f1"),
                        "pr_auc": block.get("pr_auc"),
                        "roc_auc": block.get("roc_auc"),
                        "event_precision": block.get("event_precision"),
                        "event_recall": block.get("event_recall"),
                        "event_f1": block.get("event_f1"),
                    }
                )
    return pd.DataFrame(rows)


def _plot_metric(df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    if df.empty:
        return
    pivot = df.pivot_table(index=["dataset", "variant"], columns="model", values=metric)
    pivot = pivot.sort_index()
    ax = pivot.plot(kind="bar", figsize=(11, 4), rot=25)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = _collect_metrics(reports_dir)
    if metrics_df.empty:
        raise ValueError(f"No metrics found under {reports_dir}")

    metrics_df = metrics_df.sort_values(["dataset", "variant", "model"]).reset_index(drop=True)
    metrics_df.to_csv(output_dir / "baseline_summary.csv", index=False)
    with (output_dir / "baseline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, indent=2)

    _plot_metric(
        metrics_df,
        metric="f1",
        output_path=output_dir / "f1_comparison.png",
        title="Point-level F1 Comparison by Dataset/Variant",
    )
    _plot_metric(
        metrics_df,
        metric="event_f1",
        output_path=output_dir / "event_f1_comparison.png",
        title="Event-level F1 Comparison by Dataset/Variant",
    )

    print(f"[viz] summary={output_dir / 'baseline_summary.csv'}")
    print(f"[viz] plots={output_dir}")


if __name__ == "__main__":
    main()
