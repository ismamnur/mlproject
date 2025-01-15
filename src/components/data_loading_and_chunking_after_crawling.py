from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader


def load_and_split_data(file_path, chunk_size=500, chunk_overlap=0):
    """Load and split data from a file."""
    loader = TextLoader(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(loader.load())
