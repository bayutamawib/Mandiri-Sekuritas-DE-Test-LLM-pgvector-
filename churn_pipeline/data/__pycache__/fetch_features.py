import psycopg2
import numpy as np
from typing import Tuple, List

def fetch_features(
    host: str = "localhost"
    port: int = 5433,
    database: str = "mydatabase",
    user: str = "postgres",
    password: str = "admin",
    table: str = "crm_contact_features"
) -> Tuple[np.ndarray, List[str]]:
    """
    Fetch numeric features for each contact.
    Returns:
        features: np.ndarray of shape (N, D)
        contact_ids: list of contact_id strings
    """
    conn = psycopg2.connect(
    dbname=database,
    user=user,
    password=password,
    host=host,
    port=port
    )
    cursor = conn.cursor()
 
    try:
        cursor.execute(f"""
            SELECT contact_id, days_since_signup, num_logins, last_purchase_amount
            FROM {table})
        """)
        rows = cursor.fetchall()

        contact_ids = [row[0] for row in rows]
        features = np.array([row[1:] for row in rows], dtype="float32")

        return features, contact_ids

    finally:
        cursor.close()
        conn.close()