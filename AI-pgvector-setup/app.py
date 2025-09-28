import os
import psycopg2
import openai
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "")

    # Embed query
    embedding_response = openai.Embedding.create(
        model="text-embedding-3-large",
        input=user_msg
    )
    embedding_vector = embedding_response["data"][0]["embedding"]

    # Search in Postgres
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, document, 1 - (embedding <=> %s::vector) AS similarity
        FROM langchain_pg_embedding
        ORDER BY embedding <=> %s::vector
        LIMIT 3;
        """,
        (embedding_vector, embedding_vector)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    context = "\n".join([f"[{r[0]}] {r[1]} (score: {r[2]:.4f})" for r in rows])

    # Get chat completion
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful CRM assistant."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_msg}"}
        ],
        temperature=0.2
    )

    answer = completion["choices"][0]["message"]["content"]
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)