import streamlit as st
import pandas as pd
import sqlite3

st.title("Nikkei Signal Lab")
st.write("Systematic signal validation for the Nikkei 225 index")

conn = sqlite3.connect("data/nikkei.db")
df = pd.read_sql("SELECT * FROM prices", conn)

st.write(f"Loaded {len(df)} rows of Nikkei 225 data")
st.dataframe(df.head(10))

import plotly.graph_objects as go

st.subheader("Nikkei 225 Price History")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close Price"))
fig.update_layout(xaxis_title="Date", yaxis_title="Price", height=400)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Walk-Forward Validation: Regression vs. Random Forest")

window_sizes = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600]
regression_scores = [-0.0341, -0.0076, -0.0071, 0.0077, -0.0303, -0.0023, -0.0036, 0.0081, -0.0212, -0.0214, 0.0143, -0.0974, 0.0303, 0.0127]
rf_scores = [-0.0970, -0.0598, 0.0163, -0.0653, -0.0024, 0.0654, -0.0666, 0.0097, 0.0048, -0.0287, -0.0588, -0.1019, 0.0822, 0.0354]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=window_sizes, y=regression_scores, mode="lines+markers", name="Regression"))
fig2.add_trace(go.Scatter(x=window_sizes, y=rf_scores, mode="lines+markers", name="Random Forest"))
fig2.update_layout(xaxis_title="Training Window Size", yaxis_title="R²", height=400)

st.plotly_chart(fig2, use_container_width=True)

st.subheader("Key Findings Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Regression Mean R²", "-0.0109", help="Walk-forward average")
    st.metric("Regression Std Dev", "0.0297")

with col2:
    st.metric("Random Forest Mean R²", "-0.0190", help="Walk-forward average")
    st.metric("Random Forest Std Dev", "0.0560")

with col3:
    st.metric("Buy-and-Hold Return", "125.27%")
    st.metric("Strategy Return (after costs)", "-1.94%", delta="-1.94%", delta_color="inverse")

st.write(f"**Statistical significance (paired t-test):** p = 0.5363 — the difference between models is not statistically significant.")