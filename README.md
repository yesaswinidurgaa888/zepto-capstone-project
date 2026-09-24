# Zepto Data & AI Platform

A single repository containing the three required capstone modules:
* data_pipeline/ - web scraping -> cleaning -> fixed-rate currency conversion -> normalized SQLite -> SQL/pandas queries.
* nalytics/ - one Titanic load -> cleaning -> leakage-safe classification -> imbalance comparison -> tuning -> regression -> saved complete pipeline.
* support_assistant/ - policy corpus -> local embeddings -> ChromaDB retrieval -> LangGraph routing -> deterministic mock generation -> Pydantic -> FastAPI -> Docker.

The repository intentionally keeps the required baseline free of paid services and API keys. The optional real-LLM path in the support assistant is disabled by default.

## 1. Environment setup

Use Python **3.11** for the smoothest compatibility with the specified libraries.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Run the data pipeline

```bash
python data_pipeline/scrape_and_load.py

python data_pipeline/queries.py
```

This scrapes the first five catalogue pages, giving at least 60 books, creates `data_pipeline/zepto_books.db`, and writes the SQL outputs to `data_pipeline/query_outputs.txt`.

The required currency conversion is the assignment's fixed baseline:

**1 GBP = 105.50 INR**

No currency API is used.

## 3. Run the analytics pipeline

Run in this order:

```bash
python analytics/01_eda.py

python analytics/02_modeling.py
```

The raw Titanic dataset is loaded from Seaborn exactly once and saved as `analytics/titanic.csv`. The EDA script creates `analytics/cleaned_titanic.csv`, which is the dataset used by the modeling script. Generated written results are stored in `EDA_REPORT.md` and `MODELING_REPORT.md`; supporting plots are under `analytics/plots/`.

## 4. Run the support assistant

The graded mode is the default mock mode. No LLM API key is required.

```bash
python -m support_assistant.run_examples

python -m uvicorn support_assistant.main:app --reload --port 7860
```

The API is available at:

```text
http://127.0.0.1:7860
```

Swagger documentation:

```text
http://127.0.0.1:7860/docs
```

Example request:

```text
POST /ask
```

Example JSON body:

```json
{
  "query": "What is the delivery fee for an order below INR 149?"
}
```

The first support-assistant run may download `all-MiniLM-L6-v2` from the model repository. After that, embeddings and retrieval run locally.

## 5. Docker

Docker is used to package and run the Support Assistant as a containerized FastAPI application.

From the project root:

```powershell
docker build -t zepto-support-assistant -f .\support_assistant\Dockerfile .
```

Run the container:

```powershell
docker run -d --name zepto-support-container -p 7860:7860 zepto-support-assistant:latest
```

The FastAPI service is available at:

```text
http://localhost:7860/docs
```

Useful Docker commands:

```powershell
docker ps
```

```powershell
docker logs zepto-support-container
```

```powershell
docker stop zepto-support-container
```

```powershell
docker start zepto-support-container
```

The Dockerized application uses `MOCK_LLM=1` for the required offline baseline.

The containerized service was verified locally using direct HTTP requests, and the health endpoint returned HTTP 200 OK.

## 6. Git workflow required by the assignment

The repository uses a feature branch for development, followed by a merge into `main`.

The required visible history is:

```text
main
  |
  +-- feature/zepto-platform
        |
        +-- Feature commit 1
        |
        +-- Feature commit 2
        |
        +-- merge back into main
```

The final repository should show at least two commits on the feature branch before it is merged back into `main`.

## 7. Final submission checklist

* [x] One public GitHub repository only.
* [x] Root `README.md` present.
* [x] `/data_pipeline` has the scraper, cleaning, SQLite schema, five SQL queries, outputs, and SQL/pandas JOIN comparison.
* [x] `/analytics/titanic.csv` exists after running `01_eda.py`.
* [x] Analytics report contains all required missing-value percentages, outlier counts, skewness, survival rates, exact six-column correlation matrix, two strongest correlations, four+ interpreted charts, and standardization check.
* [x] Three classifiers use the same stratified split and leakage-safe preprocessing.
* [x] Classification metrics, ROC/AUC, imbalance comparison, GridSearchCV and OOB score are recorded.
* [x] Fare regression includes MAE, RMSE, R2, adjusted R2 and residual interpretation.
* [x] `best_pipeline.joblib` contains preprocessing + estimator together and reloads successfully.
* [x] All eight support documents are present.
* [x] ChromaDB embeddings and LangGraph three-node flow work in mock mode.
* [x] Pydantic output has `answer`, `sources`, and `confidence`.
* [x] FastAPI `/ask` works.
* [x] Two mock example responses are recorded by `run_examples.py`.
* [x] Dockerfile builds and serves the API locally.
* [x] Git history visibly contains the required feature branch and merge.
