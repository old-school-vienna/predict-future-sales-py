import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class Trainset:
    x: np.array
    y: np.array
    y_min_max_scaler: MinMaxScaler


def pivot(df: pd.DataFrame, grp_vars: List[str], col: str, val: str) -> pd.DataFrame:
    grpd = df.groupby(grp_vars).first()
    return grpd.pivot_table(index=grp_vars, columns=col, values=val, fill_value=0.0)


def dd() -> Path:
    datadir = os.getenv("DATADIR")
    if datadir is None:
        raise RuntimeError("Environment variable DATADIR not defined")
    datadir_path = Path(datadir)
    if not datadir_path.exists():
        raise RuntimeError(f"Directory {datadir_path} does not exist")
    return datadir_path


_dt_start: datetime.date = datetime.strptime("1.1.2013", '%d.%m.%Y').date()


def to_ds(date: str) -> int:
    dt: datetime.date = datetime.strptime(date, '%d.%m.%Y').date()
    diff = dt - _dt_start
    return diff.days


def read_train_fillna() -> pd.DataFrame:
    file_name = dd() / "df_train.csv"
    df_train = pd.read_csv(file_name)
    return df_train.fillna(value=0.0)


# noinspection PyTypeChecker
def category_dict() -> Dict[int, int]:
    file_name = dd() / "items.csv"
    df = pd.read_csv(file_name)
    df = df[['item_id', 'item_category_id']]
    return pd.Series(df.item_category_id.values, index=df.item_id).to_dict()
