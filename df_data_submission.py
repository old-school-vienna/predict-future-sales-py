import df_data_next as ndat
import helpers as hlp

df_test = hlp.read_test()
print(df_test)

ts_train = ndat.read_train_data_submission()
ts_test = ndat.read_test_data_submission()

print("ts_train", ts_train.x.shape, ts_train.y.shape)
print("ts_test", ts_test.x.shape, ts_test.y.shape)

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
