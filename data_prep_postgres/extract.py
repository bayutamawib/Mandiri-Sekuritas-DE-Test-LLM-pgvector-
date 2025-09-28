import psycopg2

DB_HOST = "localhost"
DB_NAME = "your_db"
DB_USER = "your_user"
DB_PASSWORD = "your_password"

# The SQL query is stored as a string in the script
sql_query = """
create or replace view llm_products as
select
	account_id,
	loan_id,
	amount,
	status,
	duration,
	payments,
	purpose,
	'redacted' as customer_feedback -- Masking PII
from completedloan c 
"""

try:
    # Connect to the database
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    cur = conn.cursor()

    # Execute the Query
    cur.execute(sql_query)

    # Fetch the results
    records = cur.fetchall()

    print(f"Successfully fetched {len(records)} records.")

except psycopg2.Error as e:
    print(f"Database error: {e}")

# Chunking
def fetch_data_from_db():
    chunks = []
    for record in records:
        chunk_text = f"account ID: {record[1]}. purpose: {record[2]}."
        metadata = {"loan_id": record[0]}
        chunks.append({"text": chunk_text, "metadata": metadata})
    return chunks

db_chunks = fetch_data_from_db()
print(f"Prepared {len(db_chunks)} chunks from the database")

finally:
    # Close the connection
    if conn:
        cur.close()
        conn.close()