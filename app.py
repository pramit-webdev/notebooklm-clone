import os
import streamlit as st
import pdfplumber
import faiss
import numpy as np
import requests
import tempfile
from sentence_transformers import SentenceTransformer

# Load Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# App setup
st.set_page_config(page_title="NotebookLM Clone", layout="wide")
st.sidebar.title("📄 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

st.sidebar.title("⚙️ Settings")
show_sources = st.sidebar.checkbox("📎 Show Source Chunks", value=True)

if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []

# Initialize session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🧠 NotebookLM Clone")
st.markdown("Upload a PDF and ask questions based on its content.")

# Helpers
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def query_llm(system_prompt, user_prompt):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

# Main logic
if uploaded_file and GROQ_API_KEY:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Extract and chunk text
    full_text = extract_text_from_pdf(tmp_path)
    chunks = chunk_text(full_text)

    # Embed chunks and build index
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])

    # Input area
    question = st.chat_input("Ask a question about your PDF...")

    if question:
        # 1. Add user message
        st.session_state.chat_history.append({"question": question, "answer": None})
        with st.chat_message("user"):
            st.markdown(question)

        # 2. Retrieve relevant chunks
        query_embedding = model.encode([question])
        D, I = index.search(np.array(query_embedding), k=5)
        retrieved_chunks = [chunks[i] for i in I[0]]
        context = "\n\n".join([f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])

        # 3. Create prompt
        system_prompt = (
            "You are a helpful assistant. Use only the context provided to answer the user's question.\n"
            "If the answer isn't found in the context, say \"I couldn't find the answer in the provided document.\"\n"
            "Always cite the source chunk number like (Source 2)."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"

        # 4. Show assistant response after loading
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = query_llm(system_prompt, user_prompt)
                st.markdown(answer)

        # 5. Save the assistant answer
        st.session_state.chat_history[-1]["answer"] = answer

        # 6. Optional: show source chunks
        if show_sources:
            st.markdown("---")
            st.markdown("### 📎 Retrieved Source Chunks")
            for i, chunk in enumerate(retrieved_chunks):
                st.markdown(f"**Source {i+1}:**")
                st.code(chunk, language="markdown")

else:
    if not uploaded_file:
        st.warning("📎 Please upload a PDF to begin.")
    if not GROQ_API_KEY:
        st.warning("🔐 Missing Groq API key in environment variables.")
