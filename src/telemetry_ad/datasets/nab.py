from pathlib import Path

import pandas as pd

from telemetry_ad.datasets.base import BaseDatasetLoader, DatasetBundle


class NABLoader(BaseDatasetLoader):
    def load(self, series: str, labels_json: str | None = None) -> DatasetBundle:
        csv_path = self.root / f"{series}.csv"
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

        labels_df = None
        if labels_json and Path(labels_json).exists():
            # Optional: implement NAB window-label parsing when labels are available.
            labels_df = pd.DataFrame()

        # For anomaly detection, training split should be refined in preprocessing.
        return DatasetBundle(train_df=df.copy(), test_df=df.copy(), labels_df=labels_df)
