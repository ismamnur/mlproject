import os
from langchain_groq import ChatGroq


def init_groq_llm(api_key, model_name="Llama-3.1-70b-Versatile"):
    """Initialize ChatGroq LLM."""
    os.environ["GROQ_API_KEY"] = api_key
    return ChatGroq(groq_api_key=api_key, model_name=model_name)
