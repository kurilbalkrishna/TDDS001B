import pandas as pd 
import sqlite3

conn = sqlite3.connect ("data/nikkei.db")
df = pd.read_sql("SELECT * FROM prices", conn)
print(df.head())
print(df.shape)