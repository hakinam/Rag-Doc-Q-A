
# RAG Evaluation Script
# Measures Faithfulness and Answer Relevance of the RAG pipeline
# Uses LLM-as-Judge approach to score responses


from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import re

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)

# load document
loader = PyPDFLoader("document.pdf")
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# test questions with expected answers
test_cases = [
    {
        "question": "What is Innam's CGPA?",
        "expected": "8.89"
    },
    {
        "question": "Where does Innam live?",
        "expected": "Noida"
    },
    {
        "question": "What programming languages does Innam know?",
        "expected": "Python"
    },
]

print("\n========= RAG EVALUATION =========\n")

faithfulness_scores = []
relevance_scores = []

for test in test_cases:
    question = test["question"]
    expected = test["expected"]

    # retrieve context
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])

    # get answer
    response = llm.invoke(
        f"""Answer ONLY from the context provided.
        Context: {context}
        Question: {question}"""
    )
    answer = response.content

    # evaluate faithfulness
    faithfulness_prompt = f"""Rate from 0 to 1 how faithful this answer is to the context.
    1 means answer is completely based on context.
    0 means answer contains information not in context.
    Return ONLY a number between 0 and 1.
    
    Context: {context}
    Answer: {answer}"""
    
    
    faith_response = llm.invoke(faithfulness_prompt).content.strip()
    faithfulness_score = float(re.search(r'\d+\.?\d*', faith_response).group())

    # evaluate relevance
    relevance_prompt = f"""Rate from 0 to 1 how relevant this answer is to the question.
    1 means answer directly addresses the question.
    0 means answer is completely irrelevant.
    Return ONLY a number between 0 and 1.
    
    Question: {question}
    Answer: {answer}"""
    
    relevance_response = llm.invoke(relevance_prompt).content.strip()
    relevance_score = float(re.search(r'\d+\.?\d*', relevance_response).group())

    faithfulness_scores.append(faithfulness_score)
    relevance_scores.append(relevance_score)

    print(f"Question: {question}")
    print(f"Expected: {expected}")
    print(f"Got: {answer[:100]}...")
    print(f"Faithfulness: {faithfulness_score}")
    print(f"Relevance: {relevance_score}")
    print("-" * 40)

avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
avg_relevance = sum(relevance_scores) / len(relevance_scores)

print(f"\n✅ Average Faithfulness: {avg_faithfulness:.2f}")
print(f"✅ Average Relevance: {avg_relevance:.2f}")
print(f"✅ Overall Score: {((avg_faithfulness + avg_relevance) / 2):.2f}")