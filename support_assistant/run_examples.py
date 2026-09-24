"""Run the two required mock-mode examples and save their raw JSON."""
import os
import json
from pathlib import Path

os.environ.setdefault("MOCK_LLM", "1")
from .rag import answer_query

HERE = Path(__file__).resolve().parent
examples = {
    "policy": "What is the delivery fee for an order below INR 149?",
    "general": "What is the capital of France?",
}
outputs = {}
for name, query in examples.items():
    outputs[name] = {"query": query, "response": answer_query(query).model_dump()}

(HERE / "example_responses.json").write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(outputs, indent=2, ensure_ascii=False))
