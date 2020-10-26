import pandas as pd
from sklearn import preprocessing

import helpers as hlp

"""
Info:
Columns used by karl
c("cnt1", "cnt5", "cnt_3m", "cnt_6m", "cnt2", "cnt_item_3m", "cnt_item1", "qtr", "cnt6", "cnt3", "item_category_id", "cnt_shop_3m")

"""


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


def read_train_data() -> hlp.Trainset1:
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

    y_min_max_scaler = preprocessing.MinMaxScaler()
    y_scaled = pd.DataFrame(y_min_max_scaler.fit_transform(df_y), columns=df_y.columns)

    return hlp.Trainset1('struct', x_scaled, y_scaled, y_min_max_scaler)
