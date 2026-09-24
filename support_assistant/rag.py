"""Offline-first Zepto RAG assistant using Sentence Transformers, ChromaDB and LangGraph."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypedDict

import chromadb
import requests
from langgraph.graph import END, StateGraph
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer

from .models import AskResponse
from .prompt import PROMPT_TEMPLATE

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "docs"
CHROMA_DIR = HERE / "chroma_db"
COLLECTION_NAME = "zepto_policies"
MODEL_NAME = "all-MiniLM-L6-v2"

KEYWORDS = ["delivery", "return", "refund", "membership", "tracking", "track", "cancel", "gift card", "support hours"]


class State(TypedDict, total=False):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float


_embedding_model: SentenceTransformer | None = None
_collection = None
_graph = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(MODEL_NAME)
    return _embedding_model


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        if _collection.count() < 8:
            docs, ids, metas = [], [], []
            for path in sorted(DOCS_DIR.glob("doc_*.txt")):
                text = path.read_text(encoding="utf-8").strip()
                docs.append(text)
                ids.append(path.stem)
                metas.append({"document_id": path.stem})
            vectors = get_embedding_model().encode(docs, normalize_embeddings=True).tolist()
            _collection.upsert(ids=ids, documents=docs, embeddings=vectors, metadatas=metas)
    return _collection


def classify_intent(state: State) -> State:
    query = state["query"]
    lowered = query.lower()
    if any(keyword in lowered for keyword in KEYWORDS):
        return {"intent": "policy_question"}
    return {"intent": "general_question"}


def retrieve(query: str, n: int = 3):
    collection = get_collection()
    vector = get_embedding_model().encode([query], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=vector, n_results=n, include=["documents", "metadatas", "distances"])
    docs = result.get("documents", [[]])[0]
    ids = result.get("ids", [[]])[0]
    return list(zip(ids, docs))


def mock_policy_answer(query: str, retrieved: list[tuple[str, str]]) -> AskResponse:
    top_id, top_doc = retrieved[0]
    snippet = top_doc.replace("\n", " ").strip()
    return AskResponse(answer=f"Based on the retrieved context: {snippet}", sources=[x[0] for x in retrieved], confidence=1.0)


def mock_general_answer() -> AskResponse:
    return AskResponse(answer="I can only answer questions about Zepto policies right now.", sources=[], confidence=1.0)


def call_groq(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required when MOCK_LLM=0.")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def real_structured_answer(query: str, context: str, sources: list[str]) -> AskResponse:
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)
    last_error = None
    for attempt in range(3):
        corrective = "" if attempt == 0 else "\nYour previous output failed schema validation. Return ONLY valid JSON matching the exact schema."
        try:
            raw = call_groq(prompt + corrective)
            payload = json.loads(raw)
            validated = AskResponse.model_validate(payload)
            return validated
        except (json.JSONDecodeError, ValidationError, RuntimeError, requests.RequestException) as exc:
            last_error = exc
    return AskResponse(answer=f"ERROR: structured response validation failed after 3 attempts: {last_error}", sources=sources, confidence=0.0)


def retrieve_and_answer(state: State) -> State:
    retrieved = retrieve(state["query"], 3)
    sources = [x[0] for x in retrieved]
    if os.getenv("MOCK_LLM", "1") != "0":
        response = mock_policy_answer(state["query"], retrieved)
    else:
        context = "\n\n".join(f"[{doc_id}] {doc}" for doc_id, doc in retrieved)
        response = real_structured_answer(state["query"], context, sources)
    return {"answer": response.answer, "sources": response.sources, "confidence": response.confidence}


def direct_answer(state: State) -> State:
    if os.getenv("MOCK_LLM", "1") != "0":
        response = mock_general_answer()
    else:
        response = real_structured_answer(state["query"], "No retrieval context was supplied.", [])
    return {"answer": response.answer, "sources": response.sources, "confidence": response.confidence}


def route_after_classify(state: State) -> str:
    return state["intent"]


def build_graph():
    graph = StateGraph(State)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)
    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {"policy_question": "retrieve_and_answer", "general_question": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def answer_query(query: str) -> AskResponse:
    result = get_graph().invoke({"query": query})
    return AskResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=float(result.get("confidence", 0.0)),
    )
