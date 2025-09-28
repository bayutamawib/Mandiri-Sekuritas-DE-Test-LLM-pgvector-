from fastapi import FastAPI, Query
from pydantic import BaseModel
import psycopg2
import openai
import os
import traceback

# --- Config ---
PG_CONN = "postgresql://user:password@localhost:5433/mydatabase" # this is the PG_CONN
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"

app = FastAPI(title="CRM Q&A API", version="0.1.0")
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def search_crm(vec, limit=5):
    print("Querying Postgres...")
    with psycopg2.connect(PG_CONN) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, metadata 
                FROM embeddings_768_finbert_v1
                ORDER BY embedding <#> %s::vector
                LIMIT %s;
            """, (list(vec), limit))
            return cur.fetchall()

def ask_llm(question: str, context: str):
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a professional financial advisor who specializes in consumer complaints, budgeting, and fintech onboarding. You speak clearly, avoid jargon, and always provide actionable insights."},
                  {"role": "user", "content": prompt}]
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

        # Get embedding for the question
        res = client.embeddings.create(
        model=EMBED_MODEL,
        input=payload.question
        )
        query_vec = res.data[0].embedding

        # Search pgvector
        results = search_crm(query_vec, payload.limit)
        context = "\n\n".join(f"ID {rid}: {narrative}" for rid, narrative in results)
        
        # Ask LLM
        answer = ask_llm(payload.question, context)
        return AskResponse(answer=answer)
    
    except Exception as e:
        print("Error in /ask:", e)
        traceback.print_exc()
        return AskResponse(answer=f"Error: {e}")