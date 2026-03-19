from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import os

# ---------------------------
# ENV
# ---------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------
# APP
# ---------------------------

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}   # ✅ needed for Render

# ---------------------------
# REQUEST MODEL
# ---------------------------

class QueryRequest(BaseModel):
    query: str

# ---------------------------
# GLOBALS (lazy loading)
# ---------------------------

retriever = None

def load_store():
    global retriever

    if retriever is not None:
        return

    print("Loading FAISS...")

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "faiss_index",
        embedding,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    print("FAISS loaded")

# ---------------------------
# LLM
# ---------------------------

def generate_answer(query, context):

    if not context.strip():
        return "Not found in document"

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Answer only from context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ]
    )

    return response.choices[0].message.content

# ---------------------------
# API
# ---------------------------

@app.post("/ask")
def ask(request: QueryRequest):

    load_store()  # ✅ load after server starts

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([d.page_content for d in docs])

    answer = generate_answer(query, context)

    return {"answer": answer}
