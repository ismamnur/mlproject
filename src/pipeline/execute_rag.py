import sys
import os

# Add the root directory of your project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))



from src.pipeline.rag_workflow import init_workflow
from src.components.data_ingestion_for_rag import init_astra_connection, create_vector_store
from src.utils import init_groq_llm
from src.components.wikipedia_search import init_wikipedia_search
from src.components.data_loading_and_chunking_after_crawling import load_and_split_data


if __name__ == "__main__":
    # Initialize components
    astra_token = "enter token"
    database_id = "enter token"
    init_astra_connection(astra_token, database_id)

    vector_store = create_vector_store("all-MiniLM-L6-v2", "dermnet")
    llm = init_groq_llm("enter token")
    wikipedia_tool = init_wikipedia_search()

    # Load and split data
    documents = load_and_split_data("E:\\mlproject\\DERMNET (1).md")

    # Ingest data
    vector_store.add_documents(documents)

    # Initialize workflow
    workflow = init_workflow(vector_store.as_retriever(), llm, wikipedia_tool)

    # Run query
    inputs = {"question": "What causes rosacea?"}
    for output in workflow.stream(inputs):
        print(output.get("summarized_answer", "No summary generated."))
