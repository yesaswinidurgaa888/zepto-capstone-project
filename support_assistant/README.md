# Zepto Policy Support Assistant

## Project Overview

The Zepto Policy Support Assistant is a Retrieval-Augmented Generation (RAG) application that answers customer questions using a set of Zepto policy documents.

The application uses:

* Python
* Sentence Transformers
* `all-MiniLM-L6-v2` embeddings
* ChromaDB for vector retrieval
* LangGraph for the question-processing workflow
* Pydantic for request/response validation
* FastAPI for the REST API
* Docker for containerized deployment

The required graded baseline works without an external LLM API key by using `MOCK_LLM=1`.

---

## Graded Baseline

The required path is fully offline with respect to LLM providers.

Leave `MOCK_LLM` unset or set:

```text
MOCK_LLM=1
```

From the project root:

```powershell
cd support_assistant
python run_examples.py
```

To run the FastAPI application outside Docker:

```powershell
python -m uvicorn main:app --reload --port 7860
```

The API is then available at:

```text
http://127.0.0.1:7860
```

Swagger API documentation:

```text
http://127.0.0.1:7860/docs
```

Example request:

```json
{
  "query": "What is the delivery fee for an order below INR 149?"
}
```

The first run downloads the open-source `all-MiniLM-L6-v2` model if it is not already available locally. No LLM API key is required for the graded mock path.

---

## Architecture

```text
8 policy TXT files
        |
        v
Ingestion / chunking
(one document = one chunk)
        |
        v
SentenceTransformer
(all-MiniLM-L6-v2)
        |
        v
ChromaDB
collection: zepto_policies
        |
        v
LangGraph
classify_intent
        |
        |------------------------------|
        |                              |
        v                              v
policy_question                 general_question
        |                              |
        v                              v
retrieve_and_answer              direct_answer
        |                              |
        | top-3 cosine retrieval       | fixed mock answer
        v                              |
Mock generation OR                   |
optional real LLM generation         |
        |                              |
        |------------------------------|
                       |
                       v
          Pydantic AskResponse
                       |
                       v
                 FastAPI /ask
```

### Pipeline Details

Ingestion is handled by `get_collection()` in `rag.py`.

The application loads all eight policy documents and stores one embedded chunk per document in ChromaDB.

Embeddings are generated using Sentence Transformers with:

```text
all-MiniLM-L6-v2
```

Retrieval is handled by the `retrieve()` function inside the `retrieve_and_answer` LangGraph node.

The application uses cosine similarity retrieval and returns the top three relevant documents.

Final answer generation occurs in:

* `retrieve_and_answer` for Zepto policy questions
* `direct_answer` for general questions

Only the generation step branches on `MOCK_LLM`.

Classification remains deterministic using the keyword-based intent heuristic, and retrieval remains real in both mock and real-LLM modes.

---

## Mock LLM Mode

The default graded configuration uses:

```text
MOCK_LLM=1
```

In this mode:

* Policy questions use the retrieved document context.
* Answers follow the required `Based on the retrieved context: ...` format.
* General questions receive a fixed out-of-scope response.
* No external LLM API key is required.

Example out-of-scope response:

```text
I can only answer questions about Zepto policies right now.
```

---

## Prompt Design

The actual role, context, task, format, length constraint, negative constraint, and few-shot example are defined in:

```text
prompt.py
```

The main prompt template is:

```text
PROMPT_TEMPLATE
```

---

## FastAPI API

The application exposes:

### Health Check

```text
GET /
```

Example response:

```json
{
  "status": "ok",
  "mock_llm": true
}
```

### Ask a Question

```text
POST /ask
```

Example request:

```json
{
  "query": "How can I track my order?"
}
```

Example response structure:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": [
    "doc_04",
    "doc_06",
    "doc_01"
  ],
  "confidence": 1
}
```

The API also validates incoming requests. An empty query returns HTTP `422 Unprocessable Entity`.

---

## Docker Deployment

Docker is used to package and run the Support Assistant as a containerized FastAPI application.

### Build the Docker Image

From the project root:

```powershell
docker build -t zepto-support-asistant -f .\support_assistant\Dockerfile .
```

### Run the Container

```powershell
docker run -d --name zepto-support-container -p 7860:7860 zepto-support-asistant:latest
```

The FastAPI service runs inside the container on port `7860`.

Open the Swagger documentation:

```text
http://localhost:7860/docs
```

### Check the Running Container

```powershell
docker ps
```

### View Container Logs

```powershell
docker logs zepto-support-container
```

### Stop the Container

```powershell
docker stop zepto-support-container
```

### Start the Existing Container Again

```powershell
docker start zepto-support-container
```

### Docker Image

The locally built image is:

```text
zepto-support-asistant:latest
```

The application exposes:

```text
7860/tcp
```

The Docker container uses:

```text
MOCK_LLM=1
```

for the required offline baseline.

---

## Testing Performed

The Dockerized FastAPI application was tested through Swagger UI.

### Health Test

Endpoint:

```text
GET /
```

Result:

```text
HTTP 200 OK
```

### Order Tracking Test

Query:

```text
How can I track my order?
```

Result:

```text
HTTP 200 OK
```

Sources returned:

```text
doc_04
doc_06
doc_01
```

### Delivery Fee Test

Query:

```text
What is the delivery fee for small orders?
```

Result:

```text
HTTP 200 OK
```

Sources returned:

```text
doc_01
doc_05
doc_02
```

### Cancellation Test

Query:

```text
How can I cancel my order?
```

Result:

```text
HTTP 200 OK
```

Sources returned:

```text
doc_05
doc_06
doc_02
```

### Out-of-Scope Test

Query:

```text
What is the capital of France?
```

Result:

```text
HTTP 200 OK
```

The assistant correctly returned:

```text
I can only answer questions about Zepto policies right now.
```

No sources were returned for the out-of-scope question.

### Validation Test

Query:

```text
""
```

Result:

```text
HTTP 422 Unprocessable Entity
```

The API correctly rejected an empty query because the minimum query length is one character.

---

## Optional Real LLM

An optional real-LLM extension is available.

Set:

```text
MOCK_LLM=0
```

and provide:

```text
GROQ_API_KEY
```

when testing the optional Groq-compatible real-LLM path.

The real-LLM path uses the structured prompt and retries up to two additional times after schema or JSON validation failure.

Never commit an API key to GitHub or other public repositories.

---

## Project Structure

```text
zepto_project/
|
+-- data_pipeline/
|   +-- scraping scripts
|   +-- database
|   +-- SQL analysis
|   +-- pandas analysis
|
+-- analytics/
|   +-- EDA
|   +-- modeling
|   +-- plots
|   +-- reports
|   +-- best_pipeline.joblib
|
+-- support_assistant/
    +-- data/
    +-- docs/
    |   +-- doc_01.txt
    |   +-- doc_02.txt
    |   +-- ...
    |   +-- doc_08.txt
    |
    +-- chroma_db/
    +-- Dockerfile
    +-- main.py
    +-- models.py
    +-- prompt.py
    +-- rag.py
    +-- run_examples.py
    +-- example_responses.json
    +-- README.md
    +-- __init__.py
```

---

## Key Files

| File                     | Purpose                                                         |
| ------------------------ | --------------------------------------------------------------- |
| `main.py`                | FastAPI application and API endpoints                           |
| `models.py`              | Pydantic request and response models                            |
| `rag.py`                 | RAG ingestion, retrieval, classification, and answer generation |
| `prompt.py`              | Structured prompt template                                      |
| `run_examples.py`        | Example RAG queries                                             |
| `example_responses.json` | Example outputs                                                 |
| `Dockerfile`             | Docker image configuration                                      |
| `docs/`                  | Zepto policy documents                                          |
| `chroma_db/`             | ChromaDB vector store                                           |

---

## Conclusion

The Zepto Policy Support Assistant provides a complete RAG-based support workflow:

```text
Policy Documents
       |
       v
Embeddings
       |
       v
ChromaDB Retrieval
       |
       v
LangGraph Workflow
       |
       v
Answer Generation
       |
       v
FastAPI REST API
       |
       v
Docker Container
```

The application has been tested successfully through the Dockerized FastAPI service using Swagger UI.
