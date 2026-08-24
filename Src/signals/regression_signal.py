from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import pandas as pd 
import sqlite3
import numpy as np
import shap

# Connect to the SQLite database
conn = sqlite3.connect("data/nikkei.db")
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

# Create additional features
df["Momentum_5"] = df["Close"].pct_change(periods=5)
df["Momentum_60"] = df["Close"].pct_change(periods=60)
df["Volatility_60"] = df["Return"].rolling(window=60).std()

df["MA50"] = df["Close"].rolling(window=50).mean()
df["Price_to_MA50"] = df["Close"] / df["MA50"] - 1

# Define the target as the future 5-day return
df["Future_Return_5"] = df["Close"].shift(-5) / df["Close"] - 1

# Remove rows containing missing values
df_clean = df.dropna()
print(df_clean.shape)

# Select input features
X = df_clean[["Momentum_5", "Momentum_20", "Volatility_20"]]

# Check correlation between features
correlation_matrix = X.corr()
print("\n--- Factor Correlation Matrix ---")
print(correlation_matrix.round(3))

# Define target variable
y = df_clean["Future_Return_5"]

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Split data chronologically into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train and evaluate Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train R²: {train_score:.4f}")
print(f"Test R²: {test_score:.4f}")

# Generate predictions and create trading positions
predicted_returns = model.predict(X_test)
positions = (predicted_returns > 0).astype(int)

# Calculate strategy returns
y_test_actual = df_clean["Return"].iloc[-len(y_test):].values
strategy_returns = positions * y_test_actual

# Calculate transaction costs
position_changes = np.abs(np.diff(positions, prepend=0))
transaction_cost_bps = 10  # 10 basis points = 0.10% per trade
cost_per_trade = transaction_cost_bps / 10000

total_costs = position_changes.sum() * cost_per_trade
net_strategy_return = strategy_returns.sum() - total_costs

print(f"Number of trades: {position_changes.sum()}")
print(f"Total transaction costs: {total_costs:.4f}")
print(f"Strategy total return (after costs): {net_strategy_return:.4f}")

print(f"\nStrategy total return (no costs): {strategy_returns.sum():.4f}")
print(f"\nSample positions: {positions[:10]}")

# Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=4,
    min_samples_leaf=20,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Evaluate Random Forest
rf_train_score = rf_model.score(X_train, y_train)
rf_test_score = rf_model.score(X_test, y_test)

print(f"Random Forest Train R²: {rf_train_score:.4f}")
print(f"Random Forest Test R²: {rf_test_score:.4f}")

# Explain Random Forest predictions using SHAP
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

print("\n--- SHAP Mean Absolute Values (Feature Importance) ---")
for i, feature in enumerate(X_test.columns):
    print(f"{feature}: {abs(shap_values[:, i]).mean():.5f}")

# Import R² metric for walk-forward evaluation
from sklearn.metrics import r2_score

# Set walk-forward validation parameters
window_size = 1000
step_size = 200

results = []

# Walk-forward validation
for start in range(window_size, len(X) - step_size, step_size):
    X_train_wf = X.iloc[:start]
    y_train_wf = y.iloc[:start]
    X_test_wf = X.iloc[start:start + step_size]
    y_test_wf = y.iloc[start:start + step_size]

    # Linear Regression
    reg_wf = LinearRegression()
    reg_wf.fit(X_train_wf, y_train_wf)
    reg_r2 = r2_score(y_test_wf, reg_wf.predict(X_test_wf))

    # Random Forest
    rf_wf = RandomForestRegressor(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=20,
        random_state=42
    )
    rf_wf.fit(X_train_wf, y_train_wf)
    rf_r2 = r2_score(y_test_wf, rf_wf.predict(X_test_wf))

    # Store results
    results.append({
        "train_size": start,
        "regression_r2": reg_r2,
        "rf_r2": rf_r2
    })

    print(f"Train size {start}: Regression R²={reg_r2:.4f}, RF R²={rf_r2:.4f}")


# Calculate average and standard deviation of model scores
reg_scores = [r["regression_r2"] for r in results]
rf_scores = [r["rf_r2"] for r in results]

print("\n--- Walk-Forward Summary ---")
print(f"Regression: mean R² = {np.mean(reg_scores):.4f}, std = {np.std(reg_scores):.4f}")
print(f"Random Forest: mean R² = {np.mean(rf_scores):.4f}, std = {np.std(rf_scores):.4f}")

# Calculate Buy-and-Hold return for the test period
buy_hold_return = (df_clean["Close"].iloc[-1] / df_clean["Close"].iloc[len(X_train)] - 1)
print(f"\nBuy-and-Hold Return over Test Period: {buy_hold_return:.4f} ({buy_hold_return*100:.2f}%)")

# Calculate annualized Buy-and-Hold Sharpe Ratio
test_returns = df_clean["Return"].iloc[len(X_train):]
buy_hold_sharpe = (test_returns.mean() / test_returns.std()) * (252 ** 0.5)
print(f"Buy-and-Hold Sharpe Ratio (annualized): {buy_hold_sharpe:.4f}")

# Compare Linear Regression and Random Forest using a paired t-test
from scipy import stats

t_stat, p_value = stats.ttest_rel(reg_scores, rf_scores)
print(f"\nPaired t-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")