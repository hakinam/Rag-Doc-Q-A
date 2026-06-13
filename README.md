# 🧠 PaperBrain

PaperBrain is an AI-powered document assistant that lets you upload multiple PDFs and have a conversation with them. Ask questions, get answers, and see exactly which document the answer came from.

## 🚀 Live Demo
[Try PaperBrain here](https://paper-brain.streamlit.app/)

## ✨ Features
- 📄 Upload multiple PDFs at once
- 💬 Chat with your documents conversationally
- 🔍 Source citation — know exactly which document answered your question
- ⚡ Streaming responses — answers appear word by word
- 🧠 Chat history — remembers previous questions in the conversation
- 🗑️ Clear session and start fresh anytime

## 🛠️ Tech Stack
- **Frontend & Deployment:** Streamlit
- **LLM:** LLaMA 3.1 8B via Groq
- **Embeddings:** HuggingFace (all-MiniLM-L6-v2)
- **Vector Database:** ChromaDB
- **Framework:** LangChain
- **Language:** Python

## ⚙️ Run Locally

1. Clone the repo
   - git clone https://github.com/hakinam/paperbrain.git
   - cd doc-qa-app
2. Install dependencies
   - pip install -r requirements.txt
3. Create a `.env` file
   - GROQ_API_KEY=your_api_key_here
4. Run the app
   - streamlit run ui.py

## 📬 Contact
Built by Innam Ul Haq — [LinkedIn](your_linkedin_url)
      
