# Nikkei Signal Lab

## What this project does
A signal-based stock selection model on the Nikkei 225 index.

## Why
This project tests whether a traditional regression approach performs 
better or worse than machine learning algorithms for generating trading 
signals.

## Tech Stack
- Python
- yfinance (data)
- pandas / SQLite (data handling/storage)
- scikit-learn / XGBoost (modeling)

## Project Structure
- `config/` — currently empty, reserved for future configuration settings
- `data/raw/` — raw Nikkei 225 data fetched via yfinance
- `src/ingestion/` — Python code that fetches and saves the Nikkei data

## Status
In progress — data ingestion complete, signal modeling next.

## Author
Balkrishna Kuril