from fastapi import FastAPI

app = FastAPI(title="Nikkei Signal Lab API")

import sqlite3
import pandas as pd

@app.get("/data/prices")
def get_prices():
    conn = sqlite3.connect("data/nikkei.db")
    df = pd.read_sql("SELECT Date, Close FROM prices ORDER BY Date DESC LIMIT 100", conn)
    return df.to_dict(orient="records")

@app.get("/")
def root():
    return {"message": "Nikkei Signal Lab API is running"}

@app.get("/results/single-split")
def single_split_results():
    return {
        "regression": {"train_r2": 0.0069, "test_r2": 0.0067},
        "random_forest": {"train_r2": 0.0596, "test_r2": 0.0216}
    }


@app.get("/results/walk-forward")
def walk_forward_results():
    window_sizes = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600]
    regression_scores = [-0.0341, -0.0076, -0.0071, 0.0077, -0.0303, -0.0023, -0.0036, 0.0081, -0.0212, -0.0214, 0.0143, -0.0974, 0.0303, 0.0127]
    rf_scores = [-0.0970, -0.0598, 0.0163, -0.0653, -0.0024, 0.0654, -0.0666, 0.0097, 0.0048, -0.0287, -0.0588, -0.1019, 0.0822, 0.0354]

    return {
        "window_sizes": window_sizes,
        "regression_scores": regression_scores,
        "random_forest_scores": rf_scores,
        "regression_mean": -0.0109,
        "regression_std": 0.0297,
        "rf_mean": -0.0190,
        "rf_std": 0.0560
    }


@app.get("/results/benchmark")
def benchmark_results():
    return {
        "buy_and_hold_return": 1.2527,
        "buy_and_hold_sharpe": 1.1548,
        "strategy_return_before_costs": 0.0436,
        "strategy_return_after_costs": -0.0194,
        "number_of_trades": 63
    }


@app.get("/results/significance")
def significance_results():
    return {"t_statistic": 0.6352, "p_value": 0.5363, "significant": False}