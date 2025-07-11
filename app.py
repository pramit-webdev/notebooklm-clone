import os
import streamlit as st
import pdfplumber
import faiss
import numpy as np
import requests
import tempfile
import json
from sentence_transformers import SentenceTransformer

# Load Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit layout
st.set_page_config(page_title="NotebookLM Clone", layout="wide")
st.sidebar.title("📄 Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

st.sidebar.title("⚙️ Settings")
show_sources = st.sidebar.checkbox("📎 Show Source Chunks", value=True)

# File-based chat history helpers
def get_history_path(filename):
    safe_name = filename.replace(" ", "_").replace("/", "_")
    return f"chat_history_{safe_name}.json"

def load_chat_history(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return []

def save_chat_history(file_path, history):
    with open(file_path, "w") as f:
        json.dump(history, f, indent=2)

# Reset chat if user clears
if st.sidebar.button("🧹 Clear Chat History"):
    if "current_file" in st.session_state:
        hist_path = get_history_path(st.session_state.current_file.name)
        if os.path.exists(hist_path):
            os.remove(hist_path)
    st.session_state.chat_history = []
    st.session_state.pending_question = None

# Session setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if uploaded_files:
    file_names = [file.name for file in uploaded_files]
    selected_file_name = st.sidebar.selectbox("📂 Select a PDF to chat with", file_names)
    selected_file = next(file for file in uploaded_files if file.name == selected_file_name)

    st.session_state.current_file = selected_file

    # 🔁 When user selects a new file, reload its chat history
    if "active_filename" not in st.session_state:
        st.session_state.active_filename = selected_file_name
    elif st.session_state.active_filename != selected_file_name:
        st.session_state.chat_history = load_chat_history(get_history_path(selected_file_name))
        st.session_state.active_filename = selected_file_name

    st.title("🧠 NotebookLM Clone")
    st.markdown(f"You're chatting with: **{selected_file_name}**")

    # Extract PDF to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(selected_file.read())
        tmp_path = tmp.name

    # Extract and chunk text
    with pdfplumber.open(tmp_path) as pdf:
        raw_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    def chunk_text(text, chunk_size=500):
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    chunks = chunk_text(raw_text)
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # Show previous messages
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])

    # Process question if pending
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

        with st.chat_message("user"):
            st.markdown(question)

        query_embedding = model.encode([question])
        D, I = index.search(np.array(query_embedding), k=5)
        retrieved_chunks = [chunks[i] for i in I[0]]
        context = "\n\n".join([f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])

        system_prompt = (
            "You are a helpful assistant. Use only the context provided to answer the user's question.\n"
            "If the answer isn't found in the context, say \"I couldn't find the answer in the provided document.\"\n"
            "Always cite the source chunk number like (Source 2)."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={
                        "model": "llama3-8b-8192",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.3
                    }
                )
                answer = response.json()['choices'][0]['message']['content']
                st.markdown(answer)

        # Save Q&A
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })
        save_chat_history(get_history_path(selected_file_name), st.session_state.chat_history)

        if show_sources:
            st.markdown("---")
            st.markdown("### 📎 Retrieved Source Chunks")
            for i, chunk in enumerate(retrieved_chunks):
                st.markdown(f"**Source {i+1}:**")
                st.code(chunk, language="markdown")

    # Chat input
    user_input = st.chat_input("Ask a question about this PDF...")
    if user_input:
        st.session_state.pending_question = user_input
        st.rerun()

else:
    st.warning("📎 Please upload at least one PDF to begin.")
