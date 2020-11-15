import df_data_next as ndat
import helpers as hlp


def check_variables(ts_test: hlp.Trainset, ts_train: hlp.Trainset):
    teks = set(ts_test.x.keys())

    trks = sorted(ts_train.x.keys())
    te_ok = []
    te_missing = []
    for trk in trks:
        if trk in teks:
            te_ok.append(trk)
        else:
            te_missing.append(trk)

    te_ok = sorted(te_ok)
    te_missing = sorted(te_missing)

    print('OK:', te_ok)
    print('MISSING:', te_missing)


def basic_tests():
    test = hlp.read_test()
    print(test)

    trainx = hlp.read_trainx_fillna()
    month = trainx.groupby('month_nr').count()
    print(month)


def subm():
    ts_train = ndat.read_train_data_submission()

    print("train :", ts_train.id, ts_train.x.shape, ts_train.y.shape)

    mcfg = hlp.ModelConfig(activation='relu', optimizer='adam', loss='mean_squared_error', layers=[])
    model = hlp.create_model(mcfg, ts_train.x.shape[1])
    hist = model.fit(ts_train.x, ts_train.y, batch_size=10, epochs=6)
    print("=============== finished training ===============")
    print(hist.history['loss'])

    ts_test = ndat.read_test_data_submission()
    print("test  : ", ts_test.id, ts_test.x.shape, ts_test.y.shape)


if __name__ == '__main__':
    subm()
