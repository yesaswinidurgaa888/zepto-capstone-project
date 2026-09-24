from fastapi import FastAPI

from .models import AskRequest, AskResponse
from .rag import answer_query

app = FastAPI(title="Zepto Policy Support Assistant", version="1.0.0")


@app.get("/")
def health():
    return {"status": "ok", "mock_llm": True}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return answer_query(request.query)
