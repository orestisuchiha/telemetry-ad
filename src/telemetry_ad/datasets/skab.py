import pandas as pd

from telemetry_ad.datasets.base import BaseDatasetLoader, DatasetBundle


class SKABLoader(BaseDatasetLoader):
    def load(self, train_file: str, test_file: str, sep: str = ";") -> DatasetBundle:
        train_df = pd.read_csv(train_file, sep=sep)
        test_df = pd.read_csv(test_file, sep=sep)

        for frame in (train_df, test_df):
            frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
            frame.sort_values("datetime", inplace=True)
            frame.reset_index(drop=True, inplace=True)

        return DatasetBundle(train_df=train_df, test_df=test_df, labels_df=None)
