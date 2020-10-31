from dataclasses import dataclass
from typing import List, Callable

import pandas as pd
from sklearn import preprocessing

import helpers as hlp


@dataclass
class NextConfig:
    id: str
    df_base: Callable[[], pd.DataFrame]
    numeric_predictors: List[str]
    categorical_predictors: List[str]
    normalized: bool = True


configs = {
    'L': NextConfig(id='L', df_base=hlp.read_train_fillna,
                    categorical_predictors=['cat', 'month_nr'],
                    numeric_predictors=["cnt1", "cnt_6m", "cnt5", "cnt6", "cnt_3m", "cnt3", "cnt4"]),
    'M': NextConfig(id='M', df_base=hlp.read_train_fillna,
                    categorical_predictors=['cat', 'month_nr'],
                    numeric_predictors=["cnt1", "cnt_6m", "cnt5", "cnt6", "cnt_3m"]),
    'S': NextConfig(id='S', df_base=hlp.read_train_fillna,
                    categorical_predictors=['cat', 'month_nr'],
                    numeric_predictors=["cnt1", "cnt_6m", "cnt5", "cnt6"]),
    'karl': NextConfig(id='karl', df_base=hlp.read_trainx_fillna,
                       categorical_predictors=['cat', 'month_nr', 'monthgroup', 'qtr'],
                       numeric_predictors=["cnt1", "cnt2", "cnt3", "cnt4", "cnt5", "cnt6", "cnt_3m", "cnt_6m",
                                           "cnt_shop1", "cnt_shop_3m", "cnt_item1", "cnt_item_3m", "year", "price", ]),
    'karl_not_norm': NextConfig(id='karl_not_norm', df_base=hlp.read_trainx_fillna,
                                categorical_predictors=['cat', 'month_nr', 'monthgroup', 'qtr', 'shop_id', 'year'],
                                numeric_predictors=["cnt1", "cnt2", "cnt3", "cnt4", "cnt5", "cnt6", "cnt_3m", "cnt_6m",
                                                    "cnt_shop1", "cnt_shop_3m", "cnt_item1", "cnt_item_3m", "price", ],
                                normalized=False)
}


def _read_train_data(cfg: NextConfig) -> hlp.Trainset:
    """
    :returns A trainset containing the data for training and cross validation
    """
    cat_dict = hlp.category_dict()
    price_dict = hlp.price_dict()
    df = cfg.df_base()

    df = df[df['month_nr'] < 33]

    df['cat'] = df['shop_id'].map(cat_dict)
    df['price'] = df[['shop_id', 'item_id']].apply(lambda row: price_dict[(row['shop_id'], row['item_id'])], axis=1)

    for one_hot in cfg.categorical_predictors:
        df = hlp.onehot(df, one_hot)

    df_x = df.drop(['cnt'], axis=1)

    predictor_names = hlp.filter_variables(list(df_x.keys()), cfg.numeric_predictors + cfg.categorical_predictors)
    df_x = df_x[predictor_names]
    df_y = df[['cnt']]

    if cfg.normalized:
        x_min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = pd.DataFrame(x_min_max_scaler.fit_transform(df_x), columns=df_x.columns)

        y_min_max_scaler = preprocessing.MinMaxScaler()
        y_scaled = pd.DataFrame(y_min_max_scaler.fit_transform(df_y), columns=df_y.columns)

        return hlp.Trainset(f'next_{cfg.id}', x_scaled, y_scaled, y_min_max_scaler)
    else:
        return hlp.Trainset(f'next_{cfg.id}', df_x, df_y, None)


def read_train_data_l() -> hlp.Trainset:
    return _read_train_data(configs['L'])


def read_train_data_m() -> hlp.Trainset:
    return _read_train_data(configs['M'])


def read_train_data_s() -> hlp.Trainset:
    return _read_train_data(configs['S'])


def read_train_data_karl() -> hlp.Trainset:
    return _read_train_data(configs['karl'])


def read_train_data_karl_not_norm() -> hlp.Trainset:
    return _read_train_data(configs['karl_not_norm'])
