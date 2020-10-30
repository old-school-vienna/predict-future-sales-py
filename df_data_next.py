from dataclasses import dataclass
from typing import List, Optional, Callable

import pandas as pd
from sklearn import preprocessing

import helpers as hlp


@dataclass
class NextConfig:
    id: str
    df_base: Callable[[], pd.DataFrame]
    predictor_names: Optional[List[str]]
    normalized: bool = True


configs = {
    'all': NextConfig(id='all', df_base=hlp.read_train_fillna,
                      predictor_names=None),
    'L': NextConfig(id='L', df_base=hlp.read_train_fillna,
                    predictor_names=["cnt1", "cnt_6m", "cnt5", "cnt6", "cnt_3m", "cnt3", "cnt4"]),
    'M': NextConfig(id='M', df_base=hlp.read_train_fillna,
                    predictor_names=["cnt1", "cnt_6m", "cnt5", "cnt6", "cnt_3m"]),
    'S': NextConfig(id='S', df_base=hlp.read_train_fillna,
                    predictor_names=["cnt1", "cnt_6m", "cnt5", "cnt6"]),
    'karl': NextConfig(id='karl', df_base=hlp.read_trainx_fillna,
                       predictor_names=["cnt1", "cnt2", "cnt3", "cnt4", "cnt5", "cnt6", "cnt_3m", "cnt_6m",
                                        "cnt_shop1", "cnt_shop_3m", "cnt_item1", "cnt_item_3m", "year",
                                        'qtr_Q1', 'qtr_Q2', 'qtr_Q3', 'qtr_Q4', 'cat_30', 'cat_37', 'cat_40',
                                        'cat_41', 'cat_43', 'cat_55', 'cat_57', 'cat_78', "shop_id",
                                        # "price",
                                        # "price_reduc",
                                        ]),
    'karl_not_norm': NextConfig(id='karl_not_norm', df_base=hlp.read_trainx_fillna,
                                predictor_names=["cnt1", "cnt2", "cnt3", "cnt4", "cnt5", "cnt6", "cnt_3m", "cnt_6m",
                                                 "cnt_shop1", "cnt_shop_3m", "cnt_item1", "cnt_item_3m", "year",
                                                 'qtr_Q1', 'qtr_Q2', 'qtr_Q3', 'qtr_Q4', 'cat_30', 'cat_37', 'cat_40',
                                                 'cat_41', 'cat_43', 'cat_55', 'cat_57', 'cat_78', "shop_id",
                                                 # "price",
                                                 # "price_reduc",
                                                 ],
                                normalized=False)
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

    month_group = df['monthgroup']
    dummy_month_group = pd.get_dummies(month_group, prefix="mnr")
    df = dummy_month_group.join(df)
    df = df.drop(['monthgroup'], axis=1)

    qtr_nrs = df['qtr']
    dummy_qtr_nrs = pd.get_dummies(qtr_nrs, prefix="qtr")
    df = dummy_qtr_nrs.join(df)
    df = df.drop(['qtr'], axis=1)

    return df


def _read_train_data(cfg: NextConfig) -> hlp.Trainset:
    """
    :returns A trainset containing the data for training and cross validation
    """
    cat_dict = hlp.category_dict()
    df_base = cfg.df_base()

    df_p = predictors(df_base, cat_dict)

    df_x = df_p.drop(['cnt'], axis=1)
    df_y = df_p[['cnt']]

    if cfg.normalized:
        x_min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = pd.DataFrame(x_min_max_scaler.fit_transform(df_x), columns=df_x.columns)
        if cfg.predictor_names is not None:
            x_scaled = x_scaled[cfg.predictor_names]

        y_min_max_scaler = preprocessing.MinMaxScaler()
        y_scaled = pd.DataFrame(y_min_max_scaler.fit_transform(df_y), columns=df_y.columns)

        return hlp.Trainset(f'next_{cfg.id}', x_scaled, y_scaled, y_min_max_scaler)
    else:
        return hlp.Trainset(f'next_{cfg.id}', df_x, df_y, None)


def read_train_data_all() -> hlp.Trainset:
    return _read_train_data(configs['all'])


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
