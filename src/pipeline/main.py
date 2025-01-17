# src/pipeline/main.py
from src.pipeline.rag_workflow import app
from pprint import pprint

# Example query execution
inputs = {"question": "What causes rosacea"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")
        pprint(value)
    pprint("---")

print(value.get('summarized_answer', 'No summary generated.'))
