import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile
import uuid

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)

st.set_page_config(
    page_title="PaperBrain",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# initialise session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# sidebar
with st.sidebar:
    st.title("📚 PaperBrain")
    st.write("Upload your PDFs and start asking questions!")
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                        f.write(uploaded_file.read())
                        temp_path = f.name

                    loader = PyPDFLoader(temp_path)
                    pages = loader.load()

                    for page in pages:
                        page.metadata["source"] = uploaded_file.name

                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = splitter.split_documents(pages)

                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            collection_name=st.session_state.session_id
                        )
                    else:
                        st.session_state.vectorstore.add_documents(chunks)

                    st.session_state.processed_files.append(uploaded_file.name)
                    st.success(f"✅ {uploaded_file.name} ready!")

    if st.session_state.processed_files:
        st.divider()
        st.write("**Loaded Documents:**")
        for file in st.session_state.processed_files:
            st.write(f"📄 {file}")

    if st.session_state.processed_files:
        st.divider()
        if st.button("🗑️ Clear Everything"):
            st.session_state.messages = []
            st.session_state.vectorstore = None
            st.session_state.processed_files = []
            st.rerun()

# main chat area
st.title("📚 PaperBrain")
st.caption("Chat with your Documents")

if not st.session_state.processed_files:
    st.info("👈 Upload your PDFs from the sidebar to get started!")
else:
    # display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message:
                with st.expander("📄 Sources"):
                    for source in message["sources"]:
                        st.write(f"- {source}")

    # chat input
    query = st.chat_input("Ask anything about your documents...")

    if query and st.session_state.vectorstore:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        history = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in st.session_state.messages[:-1]
        ])

        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})
        docs = retriever.invoke(query)
        sources = list(set([doc.metadata["source"] for doc in docs]))
        context = "\n".join([
            f"[Source: {doc.metadata['source']}]\n{doc.page_content}"
            for doc in docs
        ])

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            for chunk in llm.stream(
                f"Conversation history:\n{history}\n\nContext:\n{context}\n\nQuestion: {query}"
            ):
                full_response += chunk.content
                response_placeholder.write(full_response)

            with st.expander("📄 Sources"):
                for source in sources:
                    st.write(f"- {source}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources
        })