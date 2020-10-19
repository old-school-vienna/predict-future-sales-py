import itertools
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

import helpers as hlp

"""
TODO:
- Train keras models with the data in dataframe
- validate the model
- cross validate the model
- save normalisation ??
DONE
- create the dataset
- Normalize the data
"""

_vars = ['cat_30', 'cat_37', 'cat_40', 'cat_41', 'cat_43', 'cat_55', 'cat_57', 'cat_78',
         'cnt', 'cnt_shop', 'cnt_item', 'cnt1', 'cnt2', 'cnt3', 'cnt4', 'cnt5', 'cnt6',
         'cnt_3m', 'cnt_6m', 'cnt_shop1', 'cnt_shop2', 'cnt_shop3', 'cnt_shop_3m',
         'cnt_item1', 'cnt_item2', 'cnt_item3', 'cnt_item_3m']


@dataclass
class Trainset:
    x: np.array
    y: np.array
    y_min_max_scaler: MinMaxScaler


def read_train_data() -> Trainset:
    """
    :returns A trainset containing the data for training and cross validation
    """
    cat_dict = category_dict()

    file_name = hlp.dd() / "df_train.csv"
    df_train = pd.read_csv(file_name)
    df = df_train.fillna(value=0.0)

    df = df[df['month_nr'] < 33]

    df['cat'] = df['shop_id'].map(cat_dict)
    cats = df['cat']
    dummy_cats = pd.get_dummies(cats, prefix="cat")
    df = dummy_cats.join(df)

    df = df.drop(['cat'], axis=1)
    li = df[_vars].to_numpy().tolist()
    df = df.assign(data=li)
    df = df.drop(_vars, axis=1)

    df = df.groupby(['shop_id', 'item_id']).agg(list)
    df = df.drop(['month_nr'], axis=1)
    df['data'] = df['data'].apply(lambda l: list(itertools.chain.from_iterable(l)))

    y: np.array = read_truth()

    x = list_to_columns(df, 'data')

    x_min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = x_min_max_scaler.fit_transform(x)

    y_min_max_scaler = preprocessing.MinMaxScaler()
    y_scaled = y_min_max_scaler.fit_transform(y)

    return Trainset(x_scaled, y_scaled, y_min_max_scaler)


def read_truth() -> np.array:
    file_name = hlp.dd() / "df_train.csv"
    df_train = pd.read_csv(file_name)
    df = df_train.fillna(value=0.0)

    df = df[df['month_nr'] == 33]
    df = df.groupby(by=['shop_id', 'item_id']).sum()
    df = df.drop(['month_nr', 'cnt_shop', 'cnt_item', 'cnt1', 'cnt2', 'cnt3',
                  'cnt4', 'cnt5', 'cnt6', 'cnt_3m', 'cnt_6m', 'cnt_shop1', 'cnt_shop2',
                  'cnt_shop3', 'cnt_shop_3m', 'cnt_item1', 'cnt_item2', 'cnt_item3',
                  'cnt_item_3m'], axis=1)
    return df.values


# noinspection PyTypeChecker
def category_dict() -> Dict[int, int]:
    file_name = hlp.dd() / "items.csv"
    df = pd.read_csv(file_name)
    df = df[['item_id', 'item_category_id']]
    return pd.Series(df.item_category_id.values, index=df.item_id).to_dict()


def list_to_columns(df: pd.DataFrame, list_col_name: str) -> np.array:
    df2 = df.copy()
    df2 = df2[[list_col_name]]
    return df2[list_col_name].tolist()


if __name__ == '__main__':
    tset = read_train_data()
    print("-- x", type(tset.x), tset.x.shape)
    print("-- y", type(tset.y), tset.y.shape)
