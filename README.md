# RAG Chatbot

## Problem Statement
The goal of this project is to build a **Retrieval-Augmented Generation (RAG) chatbot** that can answer questions based on custom documents.  
Instead of relying solely on a large language model, the chatbot retrieves relevant information from user-provided documents (`docs` folder) and uses it as context to generate accurate answers.

**Use Case:**  
- Study notes assistant  
- Company or product FAQ bot  
- Personal knowledge base chatbot  

---

## Tech Stack Used
- **Python 3.12** – main programming language  
- **Streamlit** – to build a web interface for the chatbot  
- **OpenAI API** – for natural language processing and answer generation  
- **scikit-learn** – TF-IDF vectorization for document retrieval  
- **pandas** – for data handling  
- **python-dotenv** – to securely load environment variables like API keys  
- **faiss-cpu & tiktoken** – for document embeddings and efficient retrieval  

---

## Steps to Run the Project Locally

1. **Clone the repository**
```bash
git clone https://github.com/<your-username>/rag-chatbot.git
cd rag-chatbot
