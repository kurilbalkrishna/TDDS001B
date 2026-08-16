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