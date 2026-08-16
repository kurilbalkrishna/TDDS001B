import sqlite3

conn = sqlite3.connect("data/nikkei.db")
print("Connected to database.")

create_table_query = """
CREATE TABLE IF NOT EXISTS prices (
    Date TEXT,
    Open REAL,
    High REAL,
    Low REAL,
    Close REAL,
    Volume INTEGER
)
"""

conn.execute(create_table_query)
conn.commit()
print("Table created.")

import pandas as pd
import yaml

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

df = pd.read_parquet(config["raw_data_path"])
df.to_sql("prices", conn, if_exists="replace", index=True)
print("Data loaded into database.")

cursor = conn.execute("SELECT COUNT(*) FROM prices")
print(f"Rows in database: {cursor.fetchone()}")