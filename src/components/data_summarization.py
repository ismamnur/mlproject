def summarize_documents_with_llm(llm, documents):
    """Summarize retrieved documents."""
    content = "\n\n".join([doc.page_content for doc in documents])
    summarization_prompt = f"""
    Summarize the following information in a concise and clear manner:

    {content}

    Provide a brief summary:
    """
    return llm.invoke(summarization_prompt)
