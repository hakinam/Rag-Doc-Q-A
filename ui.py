import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)

# initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# initialise vectorstore
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

st.title("Document Q&A App")
st.write("Upload a PDF and ask questions about it!")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and st.session_state.vectorstore is None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        temp_path = f.name

    loader = PyPDFLoader(temp_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    st.session_state.vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    st.success("PDF processed! You can now ask questions.")

# display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# chat input
query = st.chat_input("Ask a question about your document")

if query and st.session_state.vectorstore:
    # add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.write(query)

    # build conversation history string
    history = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in st.session_state.messages[:-1]
    ])

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])

    response = llm.invoke(
        f"Conversation history:\n{history}\n\nContext:\n{context}\n\nQuestion: {query}"
    )

    # add assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.content
    })

    with st.chat_message("assistant"):
        st.write(response.content)