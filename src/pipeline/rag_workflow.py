from langgraph.graph import END, StateGraph, START
from src.components.data_retrieval import retrieve_documents
from src.components.data_summarization import summarize_documents_with_llm
from src.components.wikipedia_search import wiki_search
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal, List, TypedDict

# Define the state schema using TypedDict
class GraphState(TypedDict):
    """Represents the state of the RAG workflow."""
    question: str
    summarized_answer: str
    documents: List[str]  # Adjust type based on your document structure


class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ..., description="Route a user query to either the vectorstore or Wikipedia search."
    )


def init_workflow(retriever, llm, wikipedia_tool):
    """Initialize the workflow."""
    workflow = StateGraph(GraphState)  # ✅ Fix: Using TypedDict instead of dict

    def route_question(state: GraphState):
        question = state["question"]
        source = route_prompt | structured_llm_router.invoke({"question": question})
        return "wiki_search" if source.datasource == "wiki_search" else "retrieve_documents"

    def summarize_state(state: GraphState):
        documents = state.get("documents", [])
        if not documents:
            return {"summarized_answer": "No relevant documents found.", "documents": documents}
        return {"summarized_answer": summarize_documents_with_llm(llm, documents), "documents": documents}

    workflow.add_node("wiki_search", lambda state: wiki_search(wikipedia_tool, state["question"]))
    workflow.add_node("retrieve_documents", lambda state: retrieve_documents(retriever, state["question"]))
    workflow.add_node("summarize_state", summarize_state)

    workflow.add_edge(START, "retrieve_documents", condition=lambda state: route_question(state) == "retrieve_documents")
    workflow.add_edge(START, "wiki_search", condition=lambda state: route_question(state) == "wiki_search")
    workflow.add_edge("retrieve_documents", "summarize_state")
    workflow.add_edge("wiki_search", "summarize_state")
    workflow.add_edge("summarize_state", END)

    return workflow
