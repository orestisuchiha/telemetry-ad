from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, Query

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telemetry_ad.dataset_io import load_nab_dataset, load_skab_dataset
from telemetry_ad.preprocessing.preprocess import basic_preprocess
from telemetry_ad.utils.io import load_yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_cfg(dataset: str, base_config: str) -> dict:
    base = load_yaml(base_config)
    ds = load_yaml(str(Path("configs") / f"{dataset}.yaml"))
    return _deep_merge(base, ds)


def _load_test_frame(dataset: str, series: str | None, split: str | None, config: str) -> tuple[pd.DataFrame, str, str]:
    cfg = _load_cfg(dataset, config)
    if dataset == "nab":
        selected_series = series or (cfg.get("series") or [None])[0]
        bundle = load_nab_dataset(
            dataset_root=cfg["dataset_root"],
            series=selected_series,
            labels_json=cfg.get("labels_json"),
            train_ratio=float(cfg.get("training", {}).get("train_ratio", 0.7)),
        )
    else:
        bundle = load_skab_dataset(
            train_file=cfg["train_file"],
            test_files=cfg["test_files"],
            separator=cfg.get("separator", ";"),
            split_name=split or cfg.get("split_name"),
        )

    pre_cfg = cfg.get("preprocessing", {})
    test_pre = basic_preprocess(
        df=bundle.test_df,
        timestamp_col=bundle.timestamp_col,
        resample_rule=pre_cfg.get("resample_rule"),
        forward_fill=bool(pre_cfg.get("forward_fill", True)),
        interpolate=pre_cfg.get("interpolate"),
        interpolate_limit_direction=str(pre_cfg.get("interpolate_limit_direction", "both")),
        ewma_alpha=pre_cfg.get("ewma_alpha"),
        exclude_cols=[bundle.label_col],
    )
    return test_pre, bundle.timestamp_col, bundle.label_col


def create_app(dataset: str, series: str | None, split: str | None, config: str) -> FastAPI:
    df, timestamp_col, label_col = _load_test_frame(dataset, series, split, config)
    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != label_col]

    app = FastAPI(title="Telemetry Stream API", version="1.0.0")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "dataset": dataset,
            "rows": len(df),
            "timestamp_col": timestamp_col,
            "label_col": label_col,
            "value_columns": numeric_cols,
        }

    @app.get("/stream/next")
    def stream_next(cursor: int = Query(0, ge=0), batch_size: int = Query(1, ge=1, le=512)):
        if cursor >= len(df):
            return {"rows": [], "next_cursor": cursor, "done": True}
        end = min(len(df), cursor + batch_size)
        sub = df.iloc[cursor:end]
        rows = []
        for _, row in sub.iterrows():
            rows.append(
                {
                    "timestamp": str(row[timestamp_col]),
                    "values": {col: float(row[col]) for col in numeric_cols},
                    "label": int(row[label_col]) if label_col in row else 0,
                }
            )
        return {"rows": rows, "next_cursor": end, "done": end >= len(df)}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve telemetry test split as FastAPI stream endpoints")
    parser.add_argument("--dataset", choices=["nab", "skab"], required=True)
    parser.add_argument("--series", default=None, help="NAB series name")
    parser.add_argument("--split", default=None, help="SKAB split name")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = create_app(args.dataset, args.series, args.split, args.config)
    uvicorn.run(app, host=args.host, port=args.port)
