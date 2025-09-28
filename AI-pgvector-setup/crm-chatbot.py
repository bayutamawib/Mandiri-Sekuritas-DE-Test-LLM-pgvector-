from fastapi import FastAPI, Query
from pydantic import BaseModel
import psycopg2
from openai
import os
import traceback

# --- Config ---
PG_CONN = "postgresql://user:password@localhost:5433/mydatabase" # this is the PG_CONN
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

app = FastAPI(title="CRM Q&A API", version="0.1.0")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_embedding(text: str):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding

def search_crm(vec, limit=5):
    print("Querying Postgres...")
    with psycopg2.connect(PG_CONN) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, document 
                FROM langchain_pg_embedding
                ORDER BY embedding <#> %s::vector
                LIMIT %s;
            """, (list(vec), limit))
            return cur.fetchall()
        if not results:
            return AskResponse(answer="No matching complaints found.")


def ask_llm(question: str, context: str):
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

class AskRequest(BaseModel):
    question: str
    limit: int = 5

class AskResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AskResponse)
def ask_endpoint(payload: AskRequest):
    try:
        print("Payload:", payload.dict())
        query_vec = get_embedding(payload.question)
        results = search_crm(query_vec, payload.limit)
        context = "\n\n".join(f"ID {rid}: {narrative}" for rid, narrative in results)
        answer = ask_llm(payload.question, context)
        return AskResponse(answer=answer)
    except Exception as e:
        print("Error in /ask:", e)
        traceback.print_exc()
        return AskResponse(answer=f"Error: {e}")
