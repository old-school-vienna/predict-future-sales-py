import itertools

import numpy as np
import pandas as pd
from sklearn import preprocessing

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


def read_train_data() -> hlp.Trainset:
    """
    :returns A trainset containing the data for training and cross validation
    """
    cat_dict = hlp.category_dict()
    df_base = hlp.read_train_fillna()

    def read_truth() -> np.array:
        df = df_base.copy()

        df = df[df['month_nr'] == 33]
        df = df.groupby(by=['shop_id', 'item_id']).sum()
        df = df.drop(['month_nr', 'cnt_shop', 'cnt_item', 'cnt1', 'cnt2', 'cnt3',
                      'cnt4', 'cnt5', 'cnt6', 'cnt_3m', 'cnt_6m', 'cnt_shop1', 'cnt_shop2',
                      'cnt_shop3', 'cnt_shop_3m', 'cnt_item1', 'cnt_item2', 'cnt_item3',
                      'cnt_item_3m'], axis=1)
        return df.values

    def columns_to_list(df_in: pd.DataFrame, list_col_name: str) -> np.array:
        df = df_in.copy()
        df = df[[list_col_name]]
        return df[list_col_name].tolist()

    def predictors() -> pd.DataFrame:
        df = df_base.copy()
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
        return df

    df1 = predictors()
    y: np.array = read_truth()
    x: np.array = columns_to_list(df1, 'data')

    x_min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = x_min_max_scaler.fit_transform(x)

    y_min_max_scaler = preprocessing.MinMaxScaler()
    y_scaled = y_min_max_scaler.fit_transform(y)

    return hlp.Trainset('flat', x_scaled, y_scaled, y_min_max_scaler)


if __name__ == '__main__':
    tset = read_train_data()
    print("-- x", type(tset.x), tset.x.shape)
    print("-- y", type(tset.y), tset.y.shape)
