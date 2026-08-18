import pandas as pd 
import sqlite3

conn = sqlite3.connect ("data/nikkei.db")
df = pd.read_sql("SELECT * FROM prices", conn)
print(df.head())
print(df.shape)

df["Return"] = df["Close"].pct_change()
print(df[["Date", "Close", "Return"]].head(10))

df["Momentum_20"] = df["Close"].pct_change(periods=20)
print(df[["Date", "Close", "Momentum_20"]].tail(10))