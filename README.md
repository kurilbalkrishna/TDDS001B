Nikkei Signal Lab
What This Project Does

A signal-based stock selection model on the Nikkei 225 index.

Why

This project tests whether a machine learning approach performs better or worse than traditional regression for generating trading signals on Japanese equity data.

Tech Stack
Python
yfinance (data)
pandas / SQLite (data handling/storage)
scikit-learn / XGBoost (modeling)
Project Structure
config/ — settings such as ticker, date range, and file paths
data/raw/ — raw Nikkei 225 data fetched via yfinance
src/ingestion/ — Python code that fetches and saves the Nikkei data
src/database/ — sets up the SQLite database and loads the price data
src/signals/ — calculates factors (momentum, volatility) and builds the regression signal
Status

In progress — data ingestion complete, factors built, regression model trained. Next: model evaluation and ML comparison.

Author

Balkrishna Kuril