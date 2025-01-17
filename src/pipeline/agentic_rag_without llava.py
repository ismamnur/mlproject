import os
import cassio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.cassandra import Cassandra
from langchain.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langgraph.graph import END, StateGraph, START
from typing import Literal, List
from typing_extensions import TypedDict
from langchain.schema import Document
from pprint import pprint
from langchain_groq import ChatGroq

# Set up environment variables
ASTRA_DB_APPLICATION_TOKEN = "your_astra_token_here"
ASTRA_DB_ID = "your_astra_db_id_here"
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

groq_api_key = "your_groq_api_key_here"
os.environ["GROQ_API_KEY"] = groq_api_key

# Load text data
loader = TextLoader("DERMNET.md")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
doc_splits = text_splitter.split_documents(loader.load())

# Initialize Vector Store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="dermnet",
    session=None,
    keyspace=None
)

retriever = astra_vector_store.as_retriever()

# Wikipedia API Wrapper
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=10, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

# Data model for routing
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search"] = Field(..., description="Route user query.")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama-3.1-70b-Versatile")
structured_llm_router = llm.with_structured_output(RouteQuery)

# Routing Logic
system_prompt = """
You are an expert at routing questions to a vectorstore or Wikipedia.
Use the vectorstore for skin disease-related questions. Otherwise, use Wikipedia.
"""
route_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])

question_router = route_prompt | structured_llm_router

class GraphState(TypedDict):
    question: str
    summarized_answer: str
    documents: List[str]

def route_question(state):
    question = state["question"]
    source = question_router.invoke({"question": question})
    return "wiki_search" if source.datasource == "wiki_search" else "vectorstore"

def summarize_documents_with_chatgroq(documents):
    content = "\n\n".join([doc.page_content for doc in documents])
    summarization_prompt = f"""
    Summarize the following information:
    {content}
    """
    return llm.invoke(summarization_prompt)

def retrieve_documents(state):
    documents = retriever.invoke(state["question"])
    return {"documents": documents, "question": state["question"]}

def wiki_search(state):
    docs = wiki.invoke({"query": state["question"]})
    return {"documents": [Document(page_content=docs)] if docs else [], "question": state["question"]}

def summarize_state(state):
    documents = state.get("documents", [])
    return {"summarized_answer": summarize_documents_with_chatgroq(documents) if documents else "No relevant documents found.", "documents": documents}

def summarize_wiki_state(state):
    documents = state.get("documents", [])
    return {"summarized_answer": summarize_documents_with_chatgroq(documents) if documents else "No relevant Wikipedia results found.", "documents": documents}

# Define workflow
graph = StateGraph(GraphState)
graph.add_node("wiki_search", wiki_search)
graph.add_node("retrieve_documents", retrieve_documents)
graph.add_node("summarize_state", summarize_state)
graph.add_node("summarize_wiki_state", summarize_wiki_state)
graph.add_conditional_edges(START, route_question, {"wiki_search": "wiki_search", "vectorstore": "retrieve_documents"})
graph.add_edge("retrieve_documents", "summarize_state")
graph.add_edge("summarize_state", END)
graph.add_edge("wiki_search", "summarize_wiki_state")
graph.add_edge("summarize_wiki_state", END)
app = graph.compile()

# Example query execution
if __name__ == "__main__":
    inputs = {"question": "What causes rosacea"}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Node '{key}':")
            pprint(value)
        pprint("---")
    print(value.get('summarized_answer', 'No summary generated.'))
