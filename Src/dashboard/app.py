import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go

st.set_page_config(
    page_title="Nikkei Signal Lab",
    page_icon="📈",
    layout="wide"
)

with st.sidebar:
    st.header("Nikkei Signal Lab")
    st.write("Systematic signal validation for the Nikkei 225 index - comparing regression and machine learning approaches.")

    st.markdown("---")
    st.subheader("Methodology")
    st.write("- Factors: Momentum (5, 20-day), Volatility (20-day)")
    st.write("- Target: 5-day forward return")
    st.write("- Models: Linear Regression, Random Forest")
    st.write("- Validation: Walk-forward across 14 rolling windows")

    st.markdown("---")
    st.subheader("Key Result")
    st.write("Neither model shows statistically significant predictive power once evaluated out-of-sample. Any small edge does not survive realistic transaction costs.")

    st.markdown("---")
    st.subheader("Tech Stack")
    st.write("Python, pandas, SQLite, scikit-learn, SHAP, Streamlit, Plotly")

    st.markdown("---")
    st.write("**Author:** Balkrishna Kuril")
    st.write("BSc Data Science and Analytics, KES Shroff College")
    st.write("[GitHub Repo](https://github.com/kurilbalkrishna/TDDS001B)")
    st.write("[LinkedIn](https://www.linkedin.com/in/balkrishna-kuril-18497531a)")


@st.cache_data
def load_data():
    conn = sqlite3.connect("data/nikkei.db")
    df = pd.read_sql("SELECT * FROM prices", conn)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


df = load_data()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close Price"))
fig.update_layout(xaxis_title="Date", yaxis_title="Price", height=400)

window_sizes = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600]
regression_scores = [-0.0341, -0.0076, -0.0071, 0.0077, -0.0303, -0.0023, -0.0036, 0.0081, -0.0212, -0.0214, 0.0143, -0.0974, 0.0303, 0.0127]
rf_scores = [-0.0970, -0.0598, 0.0163, -0.0653, -0.0024, 0.0654, -0.0666, 0.0097, 0.0048, -0.0287, -0.0588, -0.1019, 0.0822, 0.0354]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=window_sizes, y=regression_scores, mode="lines+markers", name="Regression"))
fig2.add_trace(go.Scatter(x=window_sizes, y=rf_scores, mode="lines+markers", name="Random Forest"))
fig2.update_layout(xaxis_title="Training Window Size", yaxis_title="R2", height=400)

st.title("Nikkei Signal Lab")
st.write("Systematic signal validation for the Nikkei 225 index")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Price Data", "Model Comparison", "Factor Analysis"])

with tab1:
    st.subheader("Key Findings Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Regression Mean R2", "-0.0109", help="Walk-forward average")
        st.metric("Regression Std Dev", "0.0297")
    with col2:
        st.metric("Random Forest Mean R2", "-0.0190", help="Walk-forward average")
        st.metric("Random Forest Std Dev", "0.0560")
    with col3:
        st.metric("Buy-and-Hold Return", "125.27%")
        st.metric("Strategy Return (after costs)", "-1.94%", delta="-1.94%", delta_color="inverse")
    st.write("Statistical significance (paired t-test): p = 0.5363 - the difference between models is not statistically significant.")

with tab2:
    st.subheader("Nikkei 225 Price History")

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    date_range = st.slider(
        "Select date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    filtered_df = df[(df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])]

    fig_filtered = go.Figure()
    fig_filtered.add_trace(go.Scatter(x=filtered_df["Date"], y=filtered_df["Close"], mode="lines", name="Close Price"))
    fig_filtered.update_layout(xaxis_title="Date", yaxis_title="Price", height=400)

    st.plotly_chart(fig_filtered, use_container_width=True)
    st.write(f"Showing {len(filtered_df)} rows")
    st.dataframe(filtered_df.head(10))

with tab3:
    st.subheader("Walk-Forward Validation: Regression vs. Random Forest")
    st.plotly_chart(fig2, use_container_width=True)

with tab4:
    st.subheader("Factor Correlation Matrix")
    correlation_data = {
        "Momentum_5": [1.000, 0.478, 0.006],
        "Momentum_20": [0.478, 1.000, -0.281],
        "Volatility_20": [0.006, -0.281, 1.000]
    }
    corr_df = pd.DataFrame(correlation_data, index=["Momentum_5", "Momentum_20", "Volatility_20"])

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_df.values, x=corr_df.columns, y=corr_df.index,
        colorscale="RdBu", zmid=0, text=corr_df.values, texttemplate="%{text:.3f}"
    ))
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("SHAP Feature Importance")
    shap_data = pd.DataFrame({
        "Factor": ["Volatility_20", "Momentum_5", "Momentum_20"],
        "Mean |SHAP value|": [0.00226, 0.00079, 0.00079]
    })
    fig_shap = go.Figure(go.Bar(x=shap_data["Mean |SHAP value|"], y=shap_data["Factor"], orientation="h"))
    fig_shap.update_layout(height=300, xaxis_title="Mean |SHAP value|")
    st.plotly_chart(fig_shap, use_container_width=True)
st.markdown("---")
st.caption("Built as part of ongoing research into signal validation for Japanese equity markets. Full methodology and findings available in the accompanying research paper.")