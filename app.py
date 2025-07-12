import os
import streamlit as st
import pdfplumber
import numpy as np
import tempfile
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
import requests
import uuid
import json

# Load keys
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Connect to Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# App UI
st.set_page_config(page_title="NotebookLM with Supabase", layout="wide")
st.title("📘 NotebookLM (pgvector version)")
st.sidebar.title("📄 Upload PDF")

# Upload file
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

# Dummy user session (replace with login logic later)
user_id = "demo-user-001"

# Helper: extract and chunk
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Groq API call
def query_llm(system_prompt, user_prompt):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    body = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body)
    return response.json()['choices'][0]['message']['content']

# Main logic
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Extract
    raw_text = extract_text_from_pdf(tmp_path)
    chunks = chunk_text(raw_text)

    # Save doc metadata to Supabase
    filename = uploaded_file.name
    doc_insert = supabase.table("documents").insert({
        "user_id": user_id,
        "name": filename
    }).execute()
    doc_id = doc_insert.data[0]['id']

    # Store each chunk + embedding
    for chunk in chunks:
        embedding = model.encode([chunk])[0]
        supabase.table("vectors").insert({
            "document_id": doc_id,
            "chunk": chunk,
            "embedding": embedding.tolist()
        }).execute()

    # Ask question
    st.subheader("Ask a question about your PDF")
    user_input = st.text_input("❓ Your question")

    if user_input:
        query_embedding = model.encode([user_input])[0]
        query_str = f"""
        select chunk, embedding <#> '[{",".join(map(str, query_embedding))}]' as similarity
        from vectors
        where document_id = '{doc_id}'
        order by similarity asc
        limit 5;
        """
        response = supabase.rpc("execute_sql", {"query": query_str}).execute()
        retrieved_chunks = [row["chunk"] for row in response.data]

        context = "\n\n".join([f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])
        system_prompt = "You're a helpful assistant. Use only the provided context. Cite answers as (Source X)."
        user_prompt = f"Context:\n{context}\n\nQuestion:\n{user_input}"

        with st.spinner("Thinking..."):
            answer = query_llm(system_prompt, user_prompt)

        st.markdown("### 💡 Answer")
        st.write(answer)

        st.markdown("### 📎 Source Chunks")
        for i, chunk in enumerate(retrieved_chunks):
            st.markdown(f"**Source {i+1}:**")
            st.code(chunk)

else:
    st.warning("📤 Upload a PDF to get started.")
