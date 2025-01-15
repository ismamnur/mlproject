def retrieve_documents(retriever, query):
    """Retrieve documents from the vector store."""
    return retriever.invoke(query)
