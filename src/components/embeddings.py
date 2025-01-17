# src/components/embeddings.py
import cassio
from langchain.vectorstores.cassandra import Cassandra
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize AstraDB connection
ASTRA_DB_APPLICATION_TOKEN = "xx"
ASTRA_DB_ID = "xxx"

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Vector Store
astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="dermnet",
    session=None,
    keyspace=None
)

retriever = astra_vector_store.as_retriever()
