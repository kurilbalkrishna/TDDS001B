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