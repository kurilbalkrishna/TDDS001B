Nikkei Signal Lab
What this project does

A signal-based stock selection model on the Nikkei 225 index — comparing a traditional regression approach against machine learning to test which one generates more reliable trading signals.

Why

This project tests whether a machine learning model meaningfully outperforms a simpler regression approach for generating equity signals on Japanese market data, with a focus on rigorous, out-of-sample validation rather than raw backtested returns alone.

Tech Stack
Python
yfinance (data)
pandas (data handling)
SQLite (structured storage)
scikit-learn (regression modeling)
XGBoost (planned — ML signal comparison)

Project Structure
config/ — project settings (ticker, date range, file paths) in config.yaml
data/raw/ — raw Nikkei 225 data fetched via yfinance, cached as Parquet
data/nikkei.db — SQLite database holding structured price data
src/ingestion/ — fetches Nikkei 225 data and saves it locally
src/database/ — sets up the SQLite database and loads price data into it
src/signals/ — calculates factors (momentum, volatility) and builds the regression signal

Progress So Far
 Data ingestion — fetches full historical Nikkei 225 data (4,064+ rows) from Yahoo Finance
 Config-driven setup — ticker, date range, and file paths externalized to config.yaml
 SQLite database — structured storage with price data loaded and verified
 Factor engineering — daily returns, 20-day momentum, 20-day volatility
 Target variable — 5-day forward return, the value the model is trained to predict
 Regression model — LinearRegression trained on momentum and volatility factors (in progress — coefficients built, evaluation not yet done)
Status

In progress — regression signal built, model evaluation and ML comparison next.

Author

Balkrishna Kuril