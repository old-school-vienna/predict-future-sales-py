from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from sklearn import preprocessing

import helpers as hlp


@dataclass
class NextConfig:
    id: str
    predictor_names: Optional[List[str]]


configs = {
    'all': NextConfig(id='all', predictor_names=None),
    'L': NextConfig(id='L', predictor_names=["cnt1", "cnt_6m", "cnt5", "cnt6", "cnt_3m", "cnt3", "cnt4"]),
    'M': NextConfig(id='M', predictor_names=["cnt1", "cnt_6m", "cnt5", "cnt6", "cnt_3m"]),
    'S': NextConfig(id='S', predictor_names=["cnt1", "cnt_6m", "cnt5", "cnt6"]),
}


def predictors(df_base: pd.DataFrame, cat_dict: dict) -> pd.DataFrame:
    df = df_base.copy()
    df = df[df['month_nr'] < 33]

    df['cat'] = df['shop_id'].map(cat_dict)
    cats = df['cat']
    dummy_cats = pd.get_dummies(cats, prefix="cat")
    df = dummy_cats.join(df)
    df = df.drop(['cat'], axis=1)

    month_nrs = df['month_nr']
    dummy_month_nrs = pd.get_dummies(month_nrs, prefix="mnr")
    df = dummy_month_nrs.join(df)
    df = df.drop(['month_nr'], axis=1)

    return df


def _read_train_data(cfg: NextConfig) -> hlp.Trainset:
    """
    :returns A trainset containing the data for training and cross validation
    """
    cat_dict = hlp.category_dict()
    df_base = hlp.read_train_fillna()

    df_p = predictors(df_base, cat_dict)

    df_x = df_p.drop(['cnt'], axis=1)
    df_y = df_p[['cnt']]

    x_min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = pd.DataFrame(x_min_max_scaler.fit_transform(df_x), columns=df_x.columns)
    if cfg.predictor_names is not None:
        x_scaled = x_scaled[cfg.predictor_names]

    y_min_max_scaler = preprocessing.MinMaxScaler()
    y_scaled = pd.DataFrame(y_min_max_scaler.fit_transform(df_y), columns=df_y.columns)

    return hlp.Trainset(f'next_{cfg.id}', x_scaled, y_scaled, y_min_max_scaler)


def read_train_data_all() -> hlp.Trainset:
    return _read_train_data(configs['all'])


def read_train_data_L() -> hlp.Trainset:
    return _read_train_data(configs['L'])


def read_train_data_M() -> hlp.Trainset:
    return _read_train_data(configs['M'])


def read_train_data_S() -> hlp.Trainset:
    return _read_train_data(configs['S'])
