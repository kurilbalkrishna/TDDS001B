import sqlite3

# Connect to SQLite database
conn = sqlite3.connect("data/nikkei.db")

print("Connected to database.")

# Create prices table if it does not already exist
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

# Load database configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load raw market data from Parquet file
df = pd.read_parquet(config["raw_data_path"])

# Store the data in the SQLite database
df.to_sql("prices", conn, if_exists="replace", index=True)

print("Data loaded into database.")

# Check the number of rows stored in the database
cursor = conn.execute("SELECT COUNT(*) FROM prices")
print(f"Rows in database: {cursor.fetchone()}")