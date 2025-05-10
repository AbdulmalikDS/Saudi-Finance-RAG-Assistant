# 📊 Saudi Stock Market Analysis Assistant

An AI-powered Arabic-first **RAG (Retrieval-Augmented Generation)** web app designed to analyze financial documents for Saudi companies — with a focus on *Elm (علم)*. It uses LangChain, ChromaDB, and Groq LLMs to deliver structured, explainable financial insights, backed by visualizations using Plotly.

---

## 🚀 Features

- 💬 **Arabic/English Q&A** for company financial analysis
- 🔎 **RAG-powered insights** using LangChain + Chroma
- 📈 **Dynamic charting** (bar, line, pie) via Plotly
- 🤖 **LLM Integration** using `ChatGroq` (Mixtral model)
- 🧠 Human-aligned financial summaries using custom prompts
- 📚 Embedded document search with OpenAI Embeddings

---

## 📂 Folder Structure

.
├── main.py # Main Gradio app with RAG + visualization
├── chroma_db/ # Local vector store with ChromaDB
└── README.md # Project documentation


---

## 🧠 Prompt Template (Arabic-Aware)

The assistant follows a strict financial report analysis format:

1. **Executive Summary**
2. **Key Financial Details**
3. **In-depth Analysis**
4. **Market/Competitor Context**

Prompt supports **bi-lingual** output (Arabic/English) based on user question language.

---

## 📦 Tech Stack

| Category       | Tools |
|----------------|-------|
| LLM Backend    | [Groq (Mixtral-8x7b)](https://groq.com) |
| RAG Framework  | [LangChain](https://www.langchain.com/) |
| Vector DB      | [ChromaDB](https://www.trychroma.com/) |
| Embeddings     | [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) |
| Visualization  | [Plotly](https://plotly.com/python/) |
| UI Framework   | [Gradio](https://www.gradio.app/) |
| Language       | Python 3.10+ |

---

## 🛠️ How to Run

1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/saudi-financial-rag.git
   cd saudi-financial-rag


