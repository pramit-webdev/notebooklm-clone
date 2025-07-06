# 🧠 NotebookLM Clone (RAG App)

A Streamlit-based app that allows you to upload a PDF and ask natural language questions grounded in its content — just like NotebookLM.

## 🔧 Features
- Upload any PDF
- Ask questions and get grounded answers
- LLaMA 3 (Groq) backend with context retrieval
- View source chunks supporting the answer
- Streamlit UI with optional citation toggle

## 🚀 How to Deploy (Streamlit Cloud)
1. Push this repo to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app → select `app.py`
4. Add a secret `GROQ_API_KEY` in settings

## 🧪 Local Dev
```bash
pip install -r requirements.txt
streamlit run app.py
