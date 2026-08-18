import yfinance as yf
import pandas as pd
import yaml

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

print("start fetching Nikkei 225 data...")
df = yf.download(config["ticker"], start=config["start_date"])
print("Data fetched successfully.")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)


print(df.head())
print(f"Total rows: {len(df)}")
print(f"Missing values:\n{df.isna().sum()}")

df.to_parquet("data/raw/nikkei225.parquet")
print("Saved successfully.")