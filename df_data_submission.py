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
    _ts_train = ndat.read_train_data_submission()
    _ts_test = ndat.read_test_data_submission()

    print("train:", _ts_train.id, _ts_train.x.shape, _ts_train.y.shape)
    print("test: ", _ts_test.id, _ts_test.x.shape, _ts_test.y.shape)


if __name__ == '__main__':
    subm()
