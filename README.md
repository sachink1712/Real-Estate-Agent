# 🏡 Real Estate RAG Agent

Link - https://real-estate-agent-3zexndq6acnfkobuj8sc6s.streamlit.app/

A Retrieval-Augmented Generation (RAG) AI Agent built using **LangChain**, **ChromaDB**, and **Groq’s Llama 3.3 70B model**.  
The system scrapes real estate-related articles from the web, stores them in a vector database, and answers user questions **strictly from the available context** — with hallucination prevention and prompt-injection defense.

---

## 🚀 Features

✔ Scrapes and processes real estate articles from URLs  
✔ Text chunking for efficient embedding storage  
✔ Vector database powered by **ChromaDB**  
✔ Fast inference using **Groq**  
✔ Secure `.env` API key management  
✔ Prevents hallucination (answers only if context exists)  
✔ Ability to reset or update vector DB with new data  

---

## 🧩 Tech Stack

| Component | Tool / Library |
|----------|----------------|
| LLM | Groq – Llama-3.3-70B-Versatile |
| Embeddings | Alibaba-NLP/gte-base-en-v1.5 |
| Vector DB | ChromaDB |
| Framework | LangChain |
| Document Loader | WebBaseLoader |
| Language | Python 3.11 |

