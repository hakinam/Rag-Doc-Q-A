from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key
)

loader = PyPDFLoader("document.pdf")
pages = loader.load()

### print(f"Total pages loaded: {len(pages)}")
### print(pages[0].page_content[:500])

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(pages)
### print(f"Total chunks created: {len(chunks)}")
### print(chunks[0].page_content)

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

query = "is innam good for the role of an AI engineer?"
docs = retriever.invoke(query)

context = "\n".join([doc.page_content for doc in docs])

response = llm.invoke(f"Answer this: {query}\n\nContext: {context}")

print(response.content)