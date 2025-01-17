# src/components/summarizer.py
from langchain_groq import ChatGroq
from langchain.schema import Document
import os

# Ensure API key is set
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama-3.1-70b-Versatile")

def summarize_documents_with_chatgroq(documents):
    content = "\n\n".join([doc.page_content for doc in documents])
    summarization_prompt = f"Summarize the following:\n\n{content}\n\nBrief summary:"
    return llm.invoke(summarization_prompt)

def summarize_state(state):
    print("---SUMMARIZE---")
    documents = state.get("documents", [])
    if not documents:
        return {"summarized_answer": "No relevant documents found.", "documents": documents}
    summarized_answer = summarize_documents_with_chatgroq(documents)
    return {"summarized_answer": summarized_answer, "documents": documents}
