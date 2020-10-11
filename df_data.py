import helpers as hlp
import pandas as pd

fn = hlp.dd() / "df_train.csv"

df = pd.read_csv(fn)
df = df.fillna(value=0.0) #
s = df['month_nr']
df = df.iloc[:,3:21]
ds = pd.get_dummies(s, prefix="month_nr")

df = ds.join(df)

print(df.shape)
print(df.head(10))
