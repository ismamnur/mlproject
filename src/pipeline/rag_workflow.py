# src/pipeline/rag_workflow.py
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal, List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from src.components.retriever import retrieve_documents
from src.components.summarizer import summarize_state
from src.components.embeddings import retriever
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

# Wikipedia setup
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=10, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

class GraphState(TypedDict):
    question: str
    summarized_answer: str
    documents: List[str]

def route_question(state):
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = retriever.invoke({"question": question})
    return "wiki_search" if source.datasource == "wiki_search" else "vectorstore"

def wiki_search(state):
    print("---WIKIPEDIA SEARCH---")
    question = state["question"]
    docs = wiki.invoke({"query": question})
    wiki_results = [Document(page_content=docs)] if docs else []
    return {"documents": wiki_results, "question": question}

def summarize_wiki_state(state):
    print("---SUMMARIZE WIKI---")
    documents = state.get("documents", [])
    if not documents:
        return {"summarized_answer": "No relevant Wikipedia results found.", "documents": documents}
    summarized_answer = summarize_documents_with_chatgroq(documents)
    return {"summarized_answer": summarized_answer, "documents": documents}

# Workflow Definition
workflow = StateGraph(GraphState)
workflow.add_node("wiki_search", wiki_search)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("summarize_state", summarize_state)
workflow.add_node("summarize_wiki_state", summarize_wiki_state)

workflow.add_conditional_edges(
    START,
    route_question,
    {"wiki_search": "wiki_search", "vectorstore": "retrieve_documents"},
)

workflow.add_edge("retrieve_documents", "summarize_state")
workflow.add_edge("summarize_state", END)
workflow.add_edge("wiki_search", "summarize_wiki_state")
workflow.add_edge("summarize_wiki_state", END)

app = workflow.compile()
