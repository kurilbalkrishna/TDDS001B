NIKKEI SIGNAL LAB

What This Project Does

Systematic signal validation for the Nikkei 225 index - a comparative study testing whether machine learning outperforms traditional regression for generating equity signals on Japanese market data.

WHY

This project tests whether a machine learning approach performs better or worse than traditional regression for generating trading signals on Japanese equity data, with a focus on rigorous, out-of-sample validation rather than raw backtested performance alone.

KEY FINDING

Neither model shows statistically significant predictive power once evaluated under walk-forward validation (paired t-test p = 0.46), and any small edge does not survive realistic transaction costs. An unconstrained Random Forest model is shown to overfit severely, underscoring the importance of regularization and rigorous multi-window validation over single-split evaluation.

TECH STACK

Python, pandas, NumPy, SQLite
scikit-learn, XGBoost, SciPy, SHAP
Streamlit, Plotly (dashboard)
FastAPI, Uvicorn (API)
pytest (testing)

PROJECT STRUCTURE

config/ - project settings (ticker, date range, file paths)
data/raw/ - raw Nikkei 225 data fetched via yfinance
data/nikkei.db - SQLite database with structured price data
src/ingestion/ - fetches and cleans Nikkei 225 data
src/database/ - sets up SQLite and loads price data
src/signals/ - factor engineering, regression and ML models, walk-forward validation, SHAP analysis
src/dashboard/ - Streamlit dashboard with interactive visualizations
src/api/ - FastAPI layer serving research results and live database queries
tests/ - unit tests for core factor calculations
paper/ - full LaTeX research paper with methodology and results

HOW TO RUN

1. Install dependencies: pip install -r requirements.txt
2. Fetch data: python src/ingestion/fetchdata.py
3. Load database: python src/database/db_setup.py
4. Run analysis: python src/signals/regression_signal.py
5. Launch dashboard: streamlit run src/dashboard/app.py
6. Launch API: uvicorn src.api.main:app --reload

STATUS

Complete - full pipeline from data ingestion through modeling, validation, explainability, dashboard, and API. Research paper included in /paper.

AUTHOR

Balkrishna Kuril
BSc Data Science and Analytics, KES Shroff College