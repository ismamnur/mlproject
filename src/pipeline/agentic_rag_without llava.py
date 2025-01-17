# 


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
from typing import Literal, List, Dict
from typing_extensions import TypedDict
from langchain.schema import Document
from pprint import pprint
from langchain_groq import ChatGroq

# Set up environment variables
ASTRA_DB_APPLICATION_TOKEN="xxx"
ASTRA_DB_ID="xxx"
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

groq_api_key = "xxx"
os.environ["GROQ_API_KEY"] = groq_api_key

# Load text data
loader = TextLoader("xxx")
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
retriever = astra_vector_store.as_retriever(search_kwargs={"return_similarities": True})

# Wikipedia API Wrapper
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=10, doc_content_chars_max=100)
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
    documents: List[Document]

def route_question(state: Dict) -> str:
    question = state["question"]
    source = question_router.invoke({"question": question})
    return "wiki_search" if source.datasource == "wiki_search" else "vectorstore"

def retrieve_documents(state: Dict) -> Dict:
    results = retriever.invoke(state["question"])
    
    # Extract documents and similarity scores
    documents = [doc for doc, score in results]
    similarity_scores = [score for doc, score in results]
    
    # Check if any document has a similarity score above the threshold (0.5)
    max_similarity = max(similarity_scores) if similarity_scores else 0
    print(f"max_similarity : {max_similarity}")
    if max_similarity < 0.5:
        return {"documents": [], "question": state["question"], "retrieval_attempt": "vectorstore"}
    
    return {"documents": documents, "question": state["question"]}

def wiki_search(state: Dict) -> Dict:
    docs = wiki.invoke({"query": state["question"]})
    return {"documents": [Document(page_content=docs)] if docs else [], "question": state["question"], "retrieval_attempt": "wiki"}

def check_relevance(state: Dict) -> Dict:
    documents = state.get("documents", [])
    
    # If no relevant documents found in vector store, fallback to Wikipedia search
    if not documents:
        return {"next_step": "wiki_search" if state.get("retrieval_attempt", "vectorstore") == "vectorstore" else "rewrite_query"}
    
    return {"next_step": "generate_answer"}

def rewrite_query(state: Dict) -> Dict:
    rewritten_prompt = f"Rewrite the following query for better search results: {state['question']}"
    new_question = llm.invoke(rewritten_prompt)
    return {"question": new_question}

def generate_answer(state: Dict) -> Dict:
    documents = state.get("documents", [])
    
    if not documents:
        return {"summarized_answer": "No relevant documents found.", "documents": []}
    
    content = "\n\n".join([doc.page_content for doc in documents])
    summarization_prompt = f"""
    Summarize the following information:
    {content}
    """
    return {"summarized_answer": llm.invoke(summarization_prompt), "documents": documents}

# Define workflow
graph = StateGraph(GraphState)
graph.add_node("wiki_search", wiki_search)
graph.add_node("retrieve_documents", retrieve_documents)
graph.add_node("check_relevance", check_relevance)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("generate_answer", generate_answer)

graph.add_conditional_edges(START, route_question, {"wiki_search": "wiki_search", "vectorstore": "retrieve_documents"})
graph.add_edge("retrieve_documents", "check_relevance")

graph.add_edge("wiki_search", "check_relevance")
graph.add_conditional_edges("check_relevance", lambda state: state["next_step"], {"rewrite_query": "rewrite_query", "wiki_search": "wiki_search", "generate_answer": "generate_answer"})
graph.add_edge("rewrite_query", "retrieve_documents")
graph.add_edge("generate_answer", END)

app = graph.compile()

from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass


# Example query execution
if __name__ == "__main__":
    inputs = {"question": "What causes rosacea"}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Node '{key}':")
            pprint(value)
        pprint("---")
    print(value.get('summarized_answer', 'No summary generated.'))
