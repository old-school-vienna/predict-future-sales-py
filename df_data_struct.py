import numpy as np
import pandas as pd
from sklearn import preprocessing

import helpers as hlp


def predictors(df_base: pd.DataFrame, cat_dict: dict) -> pd.DataFrame:
    df = df_base.copy()
    df = df[df['month_nr'] < 33]

    df['cat'] = df['item_id'].map(cat_dict)
    cats = df['cat']
    dummy_cats = pd.get_dummies(cats, prefix="cat")
    df = dummy_cats.join(df)
    df = df.drop(['cat'], axis=1)

    month_nrs = df['month_nr']
    dummy_month_nrs = pd.get_dummies(month_nrs, prefix="mnr")
    df = dummy_month_nrs.join(df)
    df = df.drop(['month_nr'], axis=1)

    return df


def truth(df_base: pd.DataFrame) -> pd.DataFrame:
    df = df_base.copy()
    df = df[df['month_nr'] == 33]
    df = df.groupby(by=['shop_id', 'item_id'], as_index=True).sum()
    df = df[['cnt']]
    df.rename(columns={'cnt': 'truth'}, inplace=True)
    return df


def read_train_data() -> hlp.Trainset:
    """
    :returns A trainset containing the data for training and cross validation
    """
    cat_dict = hlp.category_dict()
    df_base = hlp.read_train_fillna()

    df_p = predictors(df_base, cat_dict)
    df_t = truth(df_base)
    df_t = df_p.merge(df_t, how='outer', on=['shop_id', 'item_id'])

    x: np.array = df_p.values
    y: np.array = df_t[['truth']].values

    x_min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = x_min_max_scaler.fit_transform(x)

    y_min_max_scaler = preprocessing.MinMaxScaler()
    y_scaled = y_min_max_scaler.fit_transform(y)

    return hlp.Trainset('struct', x_scaled, y_scaled, y_min_max_scaler)


if __name__ == '__main__':
    tset = read_train_data()
    print("-- x", type(tset.x), tset.x.shape)
    print("-- y", type(tset.y), tset.y.shape)
