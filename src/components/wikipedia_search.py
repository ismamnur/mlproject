from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun


def init_wikipedia_search(top_k=10, doc_content_chars_max=200):
    """Initialize Wikipedia search."""
    api_wrapper = WikipediaAPIWrapper(top_k_results=top_k, doc_content_chars_max=doc_content_chars_max)
    return WikipediaQueryRun(api_wrapper=api_wrapper)


def wiki_search(wikipedia_tool, query):
    """Search Wikipedia and retrieve results."""
    docs = wikipedia_tool.invoke({"query": query})
    return docs if docs else []
