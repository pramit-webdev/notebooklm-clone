import os
import streamlit as st
import pdfplumber
import faiss
import numpy as np
import requests
import tempfile
from sentence_transformers import SentenceTransformer

# Load Groq API Key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# App configuration
st.set_page_config(page_title="NotebookLM Clone", layout="wide")
st.sidebar.title("📄 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

st.sidebar.title("⚙️ Settings")
show_sources = st.sidebar.checkbox("📎 Show Source Chunks", value=True)

if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🧠 NotebookLM Clone")
st.markdown("Upload a PDF and start chatting with it below.")

# Helper: chunk text
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Helper: extract text from PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

# Helper: call Groq API
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
if uploaded_file and GROQ_API_KEY:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Extract and chunk text
    raw_text = extract_text_from_pdf(tmp_path)
    chunks = chunk_text(raw_text)

    # Embed chunks and build FAISS index
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # Show existing chat history
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(message["question"])
        with st.chat_message("assistant"):
            st.markdown(message["answer"])

    # Capture new question
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    def on_submit():
        st.session_state.submitted_question = st.session_state.input_text
        st.session_state.input_text = ""  # Clear input after submission

    # Chat input box
    with st.chat_input("Ask a question about your PDF...", key="input_text", on_submit=on_submit):
        pass

    # Handle submitted question
    if "submitted_question" in st.session_state:
        question = st.session_state.submitted_question
        del st.session_state["submitted_question"]

        query_embedding = model.encode([question])
        D, I = index.search(np.array(query_embedding), k=5)
        retrieved_chunks = [chunks[i] for i in I[0]]
        context = "\n\n".join([f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])

        system_prompt = """You are a helpful assistant. Use only the context provided to answer the user's question.
If the answer isn't found in the context, say "I couldn't find the answer in the provided document."
Always cite the source chunk number like (Source 2)."""
        user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"

        with st.spinner("Thinking..."):
            answer = query_llm(system_prompt, user_prompt)

        # Store and display
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })

        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)

        if show_sources:
            st.markdown("---")
            st.markdown("### 📎 Retrieved Source Chunks")
            for i, chunk in enumerate(retrieved_chunks):
                st.markdown(f"**Source {i+1}:**")
                st.code(chunk, language="markdown")

else:
    if not uploaded_file:
        st.warning("📎 Please upload a PDF to get started.")
    if not GROQ_API_KEY:
        st.warning("🔐 Missing Groq API key in environment variables.")
