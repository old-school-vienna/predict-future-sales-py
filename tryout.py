import helpers as hlp


def analyse():
    """
    read the struct dataset and visualize the existing months_nr
    :return: Nothing
    """
    import df_data_struct as sdf

    cat_dict = hlp.category_dict()
    df_base = hlp.read_train_fillna()

    df_p = sdf.predictors(df_base, cat_dict)

    print(df_p.keys())
    df_t = sdf.truth(df_base)

    p = df_p[['shop_id', 'item_id', 'cnt', 'mnr_0', 'mnr_1', 'mnr_2', 'mnr_3', 'mnr_4', 'mnr_30', 'mnr_31', 'mnr_32']]

    x = p.join(df_t, on=['shop_id', 'item_id'])
    x = x[x.truth > 10]

    x.sort_values(by=['shop_id', 'item_id'])

    fnam = hlp.dd() / 'train.csv'
    x.to_csv(fnam, index=False)
    print("wrote to", fnam)

    print(x)


def next_dat():
    """
    Create the next dataset
    :return: None
    """
    import df_data_next as ndat

    td = ndat.read_train_data()

    print(td.x.shape)
    print(td.y.shape)

    print("-----------------------------------")


if __name__ == '__main__':
    next_dat()
