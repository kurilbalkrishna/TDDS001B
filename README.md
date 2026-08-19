NIKKEI SIGNAL LAB

What This Project Does

Systematic signal validation for the Nikkei 225 index — a comparative study testing whether machine learning outperforms traditional regression for generating equity signals on Japanese market data.

WHY

This project tests whether a machine learning approach performs better or worse than traditional regression for generating trading signals on Japanese equity data, with a focus on rigorous, out-of-sample validation rather than raw backtested performance alone.

TECH STACK

Python
yfinance (data)
pandas / SQLite (data handling/storage)
scikit-learn / XGBoost (modeling)

PROJECT STRUCTURE

config/ — settings such as ticker, date range, and file paths
data/raw/ — raw Nikkei 225 data fetched via yfinance
src/ingestion/ — Python code that fetches and saves the Nikkei data
src/database/ — sets up the SQLite database and loads the price data
src/signals/ — calculates factors (momentum, volatility) and builds the regression and machine learning signals

STATUS

In progress — data ingestion complete, factors built, regression and Random Forest models trained and compared on a single train/test split. Next: walk-forward validation.

AUTHOR

Balkrishna Kuril