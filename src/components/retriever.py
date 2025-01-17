# src/components/retriever.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from src.components.embeddings import retriever

# Load and split documents
loader = TextLoader("/content/DERMNET (1).md")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
doc_splits = text_splitter.split_documents(loader.load())

# Function to retrieve documents
def retrieve_documents(state):
    print("---RETRIEVE DOCUMENTS---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
