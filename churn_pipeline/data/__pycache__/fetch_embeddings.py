import psycopg2
import numpy as np
import pandas as pd

def fetch_embeddings():
    conn = psycopg2.connect(
        dbname="mydatabase",
        user="user",
        password="password",
        host="localhost",
        port="5433"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM langchain_pg_embedding;")
    rows = cursor.fetchall()
    embeddings = np.array([row[0] for row in rows])
    labels = np.array([row[1] for row in rows])
    return embeddings, labels