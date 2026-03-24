from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    timestamp_col: str
    value_col: str | None
    label_col: str
    name: str
    variant: str


def _ensure_datetime(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce", utc=True)
    out = out.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)
    return out


def _find_nab_series_file(dataset_root: str, series: str) -> Path:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"NAB dataset_root does not exist: {root}")

    direct = root / f"{series}.csv"
    if direct.exists():
        return direct

    matches = list(root.rglob(f"{series}.csv"))
    if not matches:
        raise FileNotFoundError(f"NAB series not found: {series}.csv under {root}")
    return matches[0]


def _labels_from_windows(labels_json: str, series_file: Path, timestamps: pd.Series) -> pd.Series:
    if not labels_json:
        return pd.Series(0, index=timestamps.index, dtype=int)

    path = Path(labels_json)
    if not path.exists():
        return pd.Series(0, index=timestamps.index, dtype=int)

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    key_candidates = [
        str(series_file).replace("\\", "/"),
        str(series_file.name),
        f"realAWSCloudwatch/{series_file.name}",
    ]

    windows = None
    for key in key_candidates:
        if key in payload:
            windows = payload[key]
            break
    if windows is None:
        for key, value in payload.items():
            if str(key).replace("\\", "/").endswith(series_file.name):
                windows = value
                break
    if windows is None:
        return pd.Series(0, index=timestamps.index, dtype=int)

    labels = pd.Series(0, index=timestamps.index, dtype=int)
    ts = pd.to_datetime(timestamps, utc=True)
    for win in windows:
        if not isinstance(win, list) or len(win) != 2:
            continue
        start = pd.to_datetime(win[0], utc=True, errors="coerce")
        end = pd.to_datetime(win[1], utc=True, errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        labels[(ts >= start) & (ts <= end)] = 1
    return labels


def _resolve_nab_split_idx(
    labels: pd.Series,
    train_ratio: float,
    split_strategy: str = "ratio",
    test_normal_prefix_points: int = 0,
    min_train_points: int = 1,
) -> int:
    total = len(labels)
    split_idx = max(int(min_train_points), int(total * train_ratio))
    split_idx = max(1, min(split_idx, total - 1))
    if str(split_strategy) != "anomaly_aware":
        return split_idx

    anomaly_idx = np.flatnonzero(labels.to_numpy(dtype=int) > 0)
    if len(anomaly_idx) == 0:
        return split_idx

    first_anomaly = int(anomaly_idx[0])
    desired_split = first_anomaly - int(test_normal_prefix_points)
    if desired_split <= 0:
        return split_idx
    desired_split = max(int(min_train_points), desired_split)
    desired_split = min(desired_split, total - 1)
    return min(split_idx, desired_split)


def load_nab_dataset(
    dataset_root: str,
    series: str,
    labels_json: str | None,
    train_ratio: float = 0.7,
    split_strategy: str = "ratio",
    test_normal_prefix_points: int = 0,
    min_train_points: int = 1,
    drop_labeled_anomalies_from_train: bool = False,
) -> DatasetBundle:
    series_file = _find_nab_series_file(dataset_root=dataset_root, series=series)
    df = pd.read_csv(series_file)

    if "timestamp" not in df.columns:
        raise ValueError(f"NAB file {series_file} must contain a 'timestamp' column")

    value_cols = [c for c in df.columns if c != "timestamp"]
    if not value_cols:
        raise ValueError(f"NAB file {series_file} does not contain a value column")
    value_col = value_cols[0]

    df = _ensure_datetime(df, "timestamp")
    labels = _labels_from_windows(labels_json=labels_json or "", series_file=series_file, timestamps=df["timestamp"])
    df["label"] = labels

    split_idx = _resolve_nab_split_idx(
        labels=labels,
        train_ratio=train_ratio,
        split_strategy=split_strategy,
        test_normal_prefix_points=test_normal_prefix_points,
        min_train_points=min_train_points,
    )

    train_df = df.iloc[:split_idx].reset_index(drop=True)
    if drop_labeled_anomalies_from_train and "label" in train_df.columns:
        train_df = train_df.loc[train_df["label"].astype(int) == 0].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    return DatasetBundle(
        train_df=train_df,
        test_df=test_df,
        timestamp_col="timestamp",
        value_col=value_col,
        label_col="label",
        name="nab",
        variant=series,
    )


def _get_skab_test_file(test_files: dict, split_name: str | None) -> str:
    if split_name in (None, "", "anomalyfree_vs_valve1_1", "primary"):
        return test_files.get("primary")
    if split_name in ("valve1_2", "optional_secondary"):
        return test_files.get("optional_secondary")
    return test_files.get("primary")


def load_skab_dataset(
    train_file: str,
    test_files: dict,
    separator: str = ";",
    split_name: str | None = None,
) -> DatasetBundle:
    train_path = Path(train_file)
    selected_test = _get_skab_test_file(test_files=test_files, split_name=split_name)
    if not selected_test:
        raise ValueError("SKAB test file is not configured. Check configs/skab.yaml:test_files.")
    test_path = Path(selected_test)
    if not train_path.exists():
        raise FileNotFoundError(f"SKAB train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"SKAB test file not found: {test_path}")

    train_df = pd.read_csv(train_path, sep=separator)
    test_df = pd.read_csv(test_path, sep=separator)

    if "datetime" in train_df.columns:
        timestamp_col = "datetime"
    elif "timestamp" in train_df.columns:
        timestamp_col = "timestamp"
    else:
        # Fall back to synthetic time index if explicit timestamp is unavailable.
        timestamp_col = "timestamp"
        train_df[timestamp_col] = pd.RangeIndex(start=0, stop=len(train_df), step=1)
        test_df[timestamp_col] = pd.RangeIndex(start=0, stop=len(test_df), step=1)

    if timestamp_col in ("datetime", "timestamp"):
        train_df = _ensure_datetime(train_df, timestamp_col)
        test_df = _ensure_datetime(test_df, timestamp_col)

    for df in (train_df, test_df):
        if "anomaly" in df.columns:
            df["label"] = df["anomaly"].astype(int)
        elif "Label" in df.columns:
            df["label"] = df["Label"].astype(int)
        else:
            df["label"] = 0

    variant = split_name or "anomalyfree_vs_valve1_1"
    return DatasetBundle(
        train_df=train_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        timestamp_col=timestamp_col,
        value_col=None,
        label_col="label",
        name="skab",
        variant=variant,
    )
