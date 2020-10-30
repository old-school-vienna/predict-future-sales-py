import os
import typing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import pandas as pd
import tensorflow.python.keras as keras
import tensorflow.python.keras.layers as kerasl
from sklearn.preprocessing import MinMaxScaler


@dataclass
class Trainset:
    id: str
    x: typing.Any  # Array like e.g np.array. pd.DataFrame. ...
    y: typing.Any  # Array like
    y_min_max_scaler: typing.Optional[MinMaxScaler]


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
    return _read_train_fillna('')


def read_train3_fillna() -> pd.DataFrame:
    return _read_train_fillna('3')


def read_train4_fillna() -> pd.DataFrame:
    return _read_train_fillna('4')


def read_trainx_fillna() -> pd.DataFrame:
    return _read_train_fillna('x')


def _read_train_fillna(qualifier: str) -> pd.DataFrame:
    file_name = dd() / 'in' / f"df_train{qualifier}.csv"
    df_train = pd.read_csv(file_name)
    return df_train.fillna(value=0.0)


# noinspection PyTypeChecker
def category_dict() -> Dict[int, int]:
    file_name = dd() / 'in' / "items.csv"
    df = pd.read_csv(file_name)
    df = df[['item_id', 'item_category_id']]
    return pd.Series(df.item_category_id.values, index=df.item_id).to_dict()


@dataclass
class LayerConfig:
    size_relative: float


@dataclass
class ModelConfig:
    activation: str
    optimizer: str
    loss: str
    layers: typing.List[LayerConfig]


def create_model(model_config: ModelConfig, input_size: int):
    model = keras.Sequential()
    model.add(kerasl.Dense(input_size, activation=model_config.activation))
    for layer in model_config.layers:
        model.add(kerasl.Dense(int(layer.size_relative * input_size), activation=model_config.activation))
    model.add(kerasl.Dense(1))
    model.compile(optimizer=model_config.optimizer, loss=model_config.loss)
    return model
