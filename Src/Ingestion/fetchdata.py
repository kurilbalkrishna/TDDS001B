import yfinance as yf
import pandas as pd

print("start fetching Nikkei 225 data...")
df = yf.download("^N225", start="2010-01-01")
print("Data fetched successfully.")


print(df.head())
print(f"Total rows: {len(df)}")
print(f"Missing values:\n{df.isna().sum()}")

df.to_parquet("data/raw/nikkei225.parquet")
print("Saved successfully.")