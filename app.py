# app.py

import os
import tempfile
import json
import numpy as np
import streamlit as st
import pdfplumber
import faiss
import requests
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from auth_utils import login_user, signup_user, logout_user
from supabase_client import supabase

load_dotenv()

# 🔑 API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ⚙️ App Config
st.set_page_config(page_title="NotebookLM Clone", layout="wide")

# 🔄 Session State Init
if "user" not in st.session_state:
    st.session_state.user = None
if "current_doc_id" not in st.session_state:
    st.session_state.current_doc_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🔐 Auth UI
with st.sidebar:
    st.subheader("🔐 Login / Signup")
    auth_mode = st.radio("Choose", ["Login", "Sign Up"], horizontal=True)
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Submit"):
        if auth_mode == "Login":
            success, msg = login_user(email, password)
        else:
            success, msg = signup_user(email, password)
        st.toast(msg)
        if success:
            st.experimental_rerun()

    if st.session_state.user:
        st.success(f"Logged in as {st.session_state.user['email']}")
        if st.button("Logout"):
            logout_user()
            st.experimental_rerun()

# 🛑 Stop if not logged in
if not st.session_state.user:
    st.stop()

# 📄 Upload + Load PDFs
st.sidebar.title("📄 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

st.sidebar.title("⚙️ Settings")
show_sources = st.sidebar.checkbox("📎 Show Source Chunks", value=True)

# Load existing PDFs for dropdown
user_id = st.session_state.user["id"]
docs = supabase.table("documents").select("*").eq("user_id", user_id).order("uploaded_at", desc=True).execute().data

doc_options = {doc["filename"]: doc["id"] for doc in docs}
selected_doc = st.sidebar.selectbox("📂 Your PDFs", options=list(doc_options.keys()) if doc_options else [])

if selected_doc:
    st.session_state.current_doc_id = doc_options[selected_doc]

st.title("🧠 NotebookLM Clone")

# --- Helper Functions ---
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

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
    res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body)
    return res.json()["choices"][0]["message"]["content"]

def save_document(filename, user_id):
    response = supabase.table("documents").insert({
        "filename": filename,
        "user_id": user_id,
        "uploaded_at": datetime.utcnow().isoformat()
    }).execute()
    return response.data[0]["id"]

def save_chat(document_id, question, answer, chunks):
    supabase.table("chats").insert({
        "document_id": document_id,
        "question": question,
        "answer": answer,
        "chunks": json.dumps(chunks),
        "created_at": datetime.utcnow().isoformat()
    }).execute()

def load_chats(document_id):
    res = supabase.table("chats").select("*").eq("document_id", document_id).order("created_at").execute()
    return res.data

# --- If new PDF uploaded ---
if uploaded_file and GROQ_API_KEY:
    filename = uploaded_file.name
    document_id = save_document(filename, user_id)
    st.session_state.current_doc_id = document_id

    # Save file temporarily for this session
    os.makedirs("tmp", exist_ok=True)
    path = os.path.join("tmp", filename)
    with open(path, "wb") as f:
        f.write(uploaded_file.read())

    st.experimental_rerun()

# --- Main PDF view logic ---
if st.session_state.current_doc_id:
    doc_id = st.session_state.current_doc_id
    selected_doc = supabase.table("documents").select("*").eq("id", doc_id).single().execute().data
    filename = selected_doc["filename"]
    path = os.path.join("tmp", filename)

    if not os.path.exists(path):
        st.warning("PDF file not found locally. Re-upload it.")
        st.stop()

    raw_text = extract_text_from_pdf(path)
    chunks = chunk_text(raw_text)
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    user_input = st.chat_input("Ask a question about your PDF...")

    if user_input:
        query_embedding = model.encode([user_input])
        D, I = index.search(np.array(query_embedding), k=5)
        retrieved_chunks = [chunks[i] for i in I[0]]
        context = "\n\n".join([f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])

        system_prompt = (
            "You are a helpful assistant. Use only the context provided to answer the user's question.\n"
            "If the answer isn't found in the context, say \"I couldn't find the answer in the provided document.\"\n"
            "Always cite the source chunk number like (Source 2)."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion:\n{user_input}"

        with st.spinner("Thinking..."):
            answer = query_llm(system_prompt, user_prompt)

        save_chat(doc_id, user_input, answer, retrieved_chunks)
        st.experimental_rerun()

    # Load and display previous chat history
    chat_history = load_chats(doc_id)
    for chat in chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
            if show_sources:
                with st.expander("📎 Retrieved Chunks", expanded=False):
                    for i, chunk in enumerate(json.loads(chat["chunks"])):
                        st.markdown(f"**Source {i+1}:**")
                        st.code(chunk, language="markdown")
else:
    st.info("📎 Upload a PDF or select from the dropdown to get started.")
