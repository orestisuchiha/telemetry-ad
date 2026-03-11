from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DatasetBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    labels_df: pd.DataFrame | None = None


class BaseDatasetLoader:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def load(self, **kwargs) -> DatasetBundle:
        raise NotImplementedError
