import os
import psycopg2
from fastapi import FastAPI, Query
from dotenv import load_dotenv
from openai import OpenAI  # new style client

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Create FastAPI app
app = FastAPI()

# Database connection params
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_PORT = os.getenv("DB_PORT", 5433)

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT
    )

@app.get("/ask")
def ask_crm(query: str = Query(..., description="Natural language CRM question")):
    # 1. Embed the query using new client syntax
    embedding_response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    embedding_vector = embedding_response.data[0].embedding  # Python list of floats

    # 2. Search in Postgres (pgvector) for top matches
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, content, 1 - (embedding <=> %s::vector) AS similarity
        FROM crm_records
        ORDER BY embedding <=> %s::vector
        LIMIT 3;
        """,
        (embedding_vector, embedding_vector)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # 3. Build context string
    context = "\n".join(
        f"[{r[0]}] {r[1]} (score: {r[2]:.4f})" for r in rows
    )

    # 4. Generate an answer using new Chat Completions API
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful CRM assistant."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.2
    )

    answer = completion.choices[0].message.content
    return {"answer": answer, "matches": rows}
