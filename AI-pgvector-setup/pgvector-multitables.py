import os
import sys
import torch
import psycopg2
import json
#from langchain_postgres import PGVector
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from more_itertools import chunked

# 0 Load environment variables
load_dotenv()

pg_dsn = os.getenv("PG_DSN")
if not pg_dsn:
    sys.stderr.write("\n ERROR: Missing Postgres DSN.\n")
    sys.exit(1)

# 0.5 Choose embedding / HuggingFace model
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModel.from_pretrained("yiyanghkust/finbert-tone")
device = torch.device("cpu")

# 1 Embedding model
class FinBERTEmbedder:
    def embed(self, text: str) -> list[float]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pooling over token embeddings
        pooled = outputs.last_hidden_state.mean(dim=1) # shape: [batch_size, hidden_size]
        
        if pooled.shape[1] != 768:
            raise ValueError(f"Unexpected embedding size: {pooled.shape}")
        
        return pooled.squeeze().tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

# 2 Vector store connection
#vector_store = PGVector(
#    embeddings=FinBERTEmbedder(),
#    collection_name="embeddings_768_finbert_v1", # ensure this table exists
#    connection=pg_dsn,
#    use_jsonb=True
#)

# 3 Helper to fetch and prepare docs
def fetch_docs(cur, table_name, id_col, text_col):
    cur.execute(f"""
        SELECT {id_col}, {text_col}
        FROM {table_name}
        WHERE {text_col} IS NOT NULL        
    """)
    rows = cur.fetchall()
    docs = [
        {
            "id": f"{table_name}_{row_id}",
            "text": text,
            "metadata": {
                "entity_table": table_name,
                "entity_pk": row_id,
                "source_field": text_col,
                "model_id": "yiyanghkust/finbert-tone",
                "model_version": "hf-v1"
            }
        }
        for row_id, text in rows
    ]
    return docs

# 4 Main ingestion
with psycopg2.connect(
    dbname="mydatabase",
    user="user",
    password="password",
    host="localhost",
    port=5433
) as conn:
    with conn.cursor() as cur:
        all_docs = []

        # Existing source
        all_docs.extend(fetch_docs(cur, "crmevents", "complaint_id", "consumer_complaint_narrative"))

        # New source
        all_docs.extend(fetch_docs(cur, "crmreviews", "review_id", "reviews"))

        for doc in all_docs:
            try:
                embedding = FinBERTEmbedder().embed(doc["text"])
                cur.execute("""
                    INSERT INTO embeddings_768_finbert_v1 (id, embedding, metadata) 
                    VALUES (%s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata
                        """,
                        (doc["id"], embedding, json.dumps(doc["metadata"])))
            except Exception as e:
                print(f"Error inserting/updating document {doc['id']}: {e}")
        conn.commit()

# 5 Insert/Update in vector store per batch
#for batch in chunked(all_docs, 100):
#    try:
#        vector_store.add_texts(
#            texts=[d["text"] for d in batch],
#            metadatas=[d["metadata"] for d in batch],
#            ids=[d["id"] for d in batch]
#        )
#    except Exception as e:
#        print(f"Error inserting/updating batch: {e}")

# 6 Save log to txt
with open("embedding_log.txt", "a") as log:
    for doc in all_docs:
        log.write(f"{doc['id']},\t{doc['metadata']['model_id']}, ")

print(f"Inserted/updated {len(all_docs)} embeddings into pgvector via LangChain")