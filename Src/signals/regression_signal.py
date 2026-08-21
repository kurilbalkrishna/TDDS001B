from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


import pandas as pd 
import sqlite3

# Connect to the SQLite database

conn = sqlite3.connect ("data/nikkei.db")
df = pd.read_sql("SELECT * FROM prices", conn)
print(df.head())
print(df.shape)

# Calculate returns and momentum features
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

#line regression model evaluation   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train R²: {train_score:.4f}")
print(f"Test R²: {test_score:.4f}")

#Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=4,
    min_samples_leaf=20,
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_train_score = rf_model.score(X_train, y_train)
rf_test_score = rf_model.score(X_test, y_test)

print(f"Random Forest Train R²: {rf_train_score:.4f}")
print(f"Random Forest Test R²: {rf_test_score:.4f}")

from sklearn.metrics import r2_score

window_size = 1000  # initial training window
step_size = 200      # how far to roll forward each time

results = []

for start in range(window_size, len(X) - step_size, step_size):
    X_train_wf = X.iloc[:start]
    y_train_wf = y.iloc[:start]
    X_test_wf = X.iloc[start:start + step_size]
    y_test_wf = y.iloc[start:start + step_size]

    # Regression
    reg_wf = LinearRegression()
    reg_wf.fit(X_train_wf, y_train_wf)
    reg_r2 = r2_score(y_test_wf, reg_wf.predict(X_test_wf))

    # Random Forest
    rf_wf = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=20, random_state=42)
    rf_wf.fit(X_train_wf, y_train_wf)
    rf_r2 = r2_score(y_test_wf, rf_wf.predict(X_test_wf))

    results.append({"train_size": start, "regression_r2": reg_r2, "rf_r2": rf_r2})
    print(f"Train size {start}: Regression R²={reg_r2:.4f}, RF R²={rf_r2:.4f}")
    
    import numpy as np

reg_scores = [r["regression_r2"] for r in results]
rf_scores = [r["rf_r2"] for r in results]

print("\n--- Walk-Forward Summary ---")
print(f"Regression: mean R² = {np.mean(reg_scores):.4f}, std = {np.std(reg_scores):.4f}")
print(f"Random Forest: mean R² = {np.mean(rf_scores):.4f}, std = {np.std(rf_scores):.4f}")

from scipy import stats
t_stat, p_value = stats.ttest_rel(reg_scores, rf_scores)
print(f"\nPaired t-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")