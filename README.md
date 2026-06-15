# 🧠 PaperBrain

PaperBrain is a production grade AI document intelligence system that lets you upload multiple PDFs and have a natural language conversation with them. Built with an advanced RAG pipeline featuring multi-query retrieval, cross-encoder reranking, and hallucination prevention.

## 🚀 Live Demo
[Try PaperBrain here](https://paper-brain.streamlit.app)

## ✨ Features
- 📄 Upload multiple PDFs simultaneously with per-document metadata tracking
- 💬 Conversational Q&A with persistent chat memory across questions
- 🔍 Multi-query retrieval — LLM generates multiple query variations for better coverage
- 🎯 Cross-encoder reranking — scores and filters chunks for maximum relevance
- 🛡️ Hallucination prevention — answers strictly grounded in uploaded documents
- 📌 Source citation — know exactly which document answered your question
- ⚡ Real-time streaming responses word by word
- 🔬 RAG evaluation script measuring faithfulness and relevance scores
- 🗑️ Session isolation per user with one-click clear

## 🏗️ Architecture
PDF Upload → Chunking → Embeddings → ChromaDB

↓

User Query → Multi-Query Generation → Vector Search → Reranking → LLM → Streaming Answer

## 🛠️ Tech Stack
- **LLM:** LLaMA 3.1 8B via Groq
- **Embeddings:** HuggingFace (all-MiniLM-L6-v2)
- **Reranker:** CrossEncoder (ms-marco-MiniLM-L-6-v2)
- **Vector Database:** ChromaDB (in-memory, session-scoped)
- **Framework:** LangChain
- **Frontend & Deployment:** Streamlit Cloud
- **Language:** Python

## ⚙️ Run Locally

1. Clone the repo
- git clone https://github.com/hakinam/PaperBrain.git
- cd PaperBrain

2. Install dependencies
- pip install -r requirements.txt

3. Create a `.env` file
- GROQ_API_KEY=your_api_key_here

4. Run the app
- streamlit run ui.py

5. To evaluate RAG quality
- python3 evaluate.py

## 📊 RAG Evaluation
PaperBrain includes a built-in evaluation script (`evaluate.py`) that measures:
- **Faithfulness** — are answers grounded in the document?
- **Answer Relevance** — does the answer address the question?

Uses LLM-as-Judge methodology for scoring.

## 📬 Contact
Built by Innam Ul Haq — [LinkedIn](https://linkedin.com/in/innam-ul-haq-801039280) | [GitHub](https://github.com/hakinam)

