from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split


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

df["Volatility_20"] = df["Return"].rolling(window=20).std()
print(df[["Date", "Close", "Momentum_20", "Volatility_20"]].tail(10))

df["Future_Return_5"] = df["Close"].shift(-5) / df["Close"] - 1

df_clean = df.dropna()
print(df_clean.shape)

X = df_clean[["Momentum_20", "Volatility_20"]]
y = df_clean["Future_Return_5"]

model = LinearRegression()
model.fit(X, y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train R²: {train_score:.4f}")
print(f"Test R²: {test_score:.4f}")