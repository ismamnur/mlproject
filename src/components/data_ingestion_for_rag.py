import cassio
from langchain.vectorstores.cassandra import Cassandra
from langchain_huggingface import HuggingFaceEmbeddings


def init_astra_connection(token, database_id):
    """Initialize AstraDB connection."""
    cassio.init(token=token, database_id=database_id)


def create_vector_store(model_name, table_name):
    """Initialize vector store with embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return Cassandra(embedding=embeddings, table_name=table_name)


def ingest_documents(vector_store, documents):
    """Ingest documents into the vector store."""
    vector_store.add_documents(documents)
