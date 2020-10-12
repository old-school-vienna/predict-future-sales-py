import itertools
from typing import Dict

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


def read_predictors():
    """
    length of data is 891
    cnt contains the truth.
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

    df_truth = read_truth()
    df = pd.merge(df, df_truth, left_index=True, right_index=True)

    df_data = list_to_columns(df, 'data', 891, 'cnt')

    df = df.merge(df_data, left_index=True, right_index=True)
    df = df.drop(['data'], axis=1)

    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df.values)
    df = pd.DataFrame(x_scaled)

    print("-- keys", df.keys())
    print("-- data", df)

def read_truth() -> pd.DataFrame:
    file_name = hlp.dd() / "df_train.csv"
    df_train = pd.read_csv(file_name)
    df = df_train.fillna(value=0.0)

    df = df[df['month_nr'] == 33]
    df = df.groupby(by=['shop_id', 'item_id']).sum()
    df = df.drop(['month_nr', 'cnt_shop', 'cnt_item', 'cnt1', 'cnt2', 'cnt3',
                  'cnt4', 'cnt5', 'cnt6', 'cnt_3m', 'cnt_6m', 'cnt_shop1', 'cnt_shop2',
                  'cnt_shop3', 'cnt_shop_3m', 'cnt_item1', 'cnt_item2', 'cnt_item3',
                  'cnt_item_3m'], axis=1)
    return df


# noinspection PyTypeChecker
def category_dict() -> Dict[int, int]:
    file_name = hlp.dd() / "items.csv"
    df = pd.read_csv(file_name)
    df = df[['item_id', 'item_category_id']]
    return pd.Series(df.item_category_id.values, index=df.item_id).to_dict()


def list_to_columns(df: pd.DataFrame, list_col_name: str, count_list_items: int, prefix: str) -> pd.DataFrame:
    vns = [f"{prefix}_{i:03}" for i in range(count_list_items)]
    df2 = df.copy()
    df2 = df2[[list_col_name]]
    df2[vns] = pd.DataFrame(df2[list_col_name].tolist(), index= df2.index)
    df2 = df2.drop([list_col_name], axis=1)
    return df2


read_predictors()
# read_truth()
