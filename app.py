
import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# load .env
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.error("No OPENAI_API_KEY found. Put it in a .env file.")
openai.api_key = OPENAI_KEY

# --- Helper: load docs ---
DOCS_DIR = Path("docs")

@st.cache_data
def load_documents():
    docs = []
    paths = list(DOCS_DIR.glob("*.txt"))
    for p in paths:
        text = p.read_text(encoding="utf-8")
        docs.append({"filename": p.name, "text": text})
    return pd.DataFrame(docs)

# --- Helper: simple TF-IDF retrieval ---
@st.cache_data
def build_vectorizer(corpus):
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9)
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X

def retrieve_top_k(query, vectorizer, X, docs, k=3):
    q_vec = vectorizer.transform([query])
    scores = (X @ q_vec.T).toarray().ravel()
    top_idx = scores.argsort()[::-1][:k]
    results = []
    for i in top_idx:
        results.append({"filename": docs.iloc[i]["filename"], "text": docs.iloc[i]["text"], "score": float(scores[i])})
    return results

# --- Chat completion with context ---
def ask_openai(question, context_texts):
    # Build the system and user prompt
    system_prompt = "You are an assistant that answers questions using provided context. If the answer is not in the context, say you don't know and offer general guidance."
    # join top contexts
    context_joined = "\n\n---\n\n".join([f"Document: {r['filename']}\n{r['text'][:2000]}" for r in context_texts])
    user_prompt = f"Context:\n{context_joined}\n\nQuestion: {question}\nAnswer concisely and cite which document you used by filename."
    # call OpenAI ChatCompletion (gpt-3.5-turbo)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=300,
        temperature=0.2,
    )
    return response["choices"][0]["message"]["content"]

# --- Streamlit UI ---
st.set_page_config(page_title="Simple RAG Chatbot", layout="wide")
st.title("Simple RAG Chatbot — Beginner friendly")

st.sidebar.header("Settings")
k = st.sidebar.number_input("Number of retrieved docs (k)", min_value=1, max_value=5, value=3)
st.sidebar.markdown("Put your `.txt` files into the `docs/` folder.")

docs_df = load_documents()
if docs_df.empty:
    st.warning("No documents found in docs/ folder. Add some .txt files and refresh.")
    st.stop()

# build vectorizer
vectorizer, X = build_vectorizer(docs_df["text"].tolist())

st.markdown("### Ask a question (the app will search your docs and use OpenAI to answer)")
question = st.text_input("Your question", "")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving relevant documents..."):
        retrieved = retrieve_top_k(question, vectorizer, X, docs_df, k=k)
    st.markdown("**Retrieved Documents (top k)**")
    for r in retrieved:
        st.markdown(f"- **{r['filename']}** (score: {r['score']:.4f})")

    with st.spinner("Asking the model..."):
        answer = ask_openai(question, retrieved)
    st.markdown("### Answer")
    st.write(answer)

    st.markdown("### Source snippets (for transparency)")
    for r in retrieved:
        st.write(f"**{r['filename']}** — first 800 chars:")
        st.code(r["text"][:800])
