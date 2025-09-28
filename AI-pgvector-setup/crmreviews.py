import os
import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
import psycopg2

# 0️⃣ Load environment variables (.env should have OPENAI_API_KEY and PG_DSN)
load_dotenv()

# ✅ Ensure API key is present
if not os.getenv("OPENAI_API_KEY"):
    sys.stderr.write(
        "\n🚨 ERROR: Missing OpenAI API key.\n"
        "Set it in your shell or .env file as:\n"
        "OPENAI_API_KEY=sk-...\n\n"
    )
    sys.exit(1)

# ✅ Ensure PG_DSN is present
pg_dsn = os.getenv("PG_DSN")
if not pg_dsn:
    sys.stderr.write(
        "\n🚨 ERROR: Missing Postgres DSN.\n"
        "Set it in your shell or .env file as:\n"
        "PG_DSN=postgresql+psycopg://user:password@localhost:5433/mydatabase\n\n"
    )
    sys.exit(1)

# 1️⃣ Set up embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 2️⃣ Connect to vector store (pgvector table)
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="embeddings_1536",  # Name of your pgvector collection/table
    connection=pg_dsn,
    use_jsonb=True
)

# 3️⃣ Fetch rows from source table
with psycopg2.connect(
    dbname="mydatabase",
    user="user",
    password="password",
    host="localhost",
    port=5433
) as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT complaint_id, consumer_complaint_narrative
            FROM crmevents
            WHERE consumer_complaint_narrative IS NOT NULL
        """)
        rows = cur.fetchall()

        cur.execute

# 4️⃣ Prepare documents
docs = [
    {
        "id": f"crmevents_{complaint_id}",
        "text": text,
        "metadata": {
            "entity_table": "crmevents",
            "entity_pk": complaint_id,
            "source_field": "consumer_complaint_narrative",
            "model_id": "text-embedding-3-large",
            "model_version": "2024-08-01"
        }
    }
    for complaint_id, text in rows
]


# 5️⃣ Insert/Update in vector store
vector_store.add_texts(
    texts=[d["text"] for d in docs],
    metadatas=[d["metadata"] for d in docs],
    ids=[d["id"] for d in docs]
)

print(f"✅ Inserted/updated {len(docs)} embeddings into pgvector via LangChain.")
