# Zepto Data & AI Platform

A single repository containing the three required capstone modules:

- `data_pipeline/` — web scraping → cleaning → fixed-rate currency conversion → normalized SQLite → SQL/pandas queries.
- `analytics/` — one Titanic load → cleaning → EDA → leakage-safe classification → imbalance comparison → tuning → regression → saved complete pipeline.
- `support_assistant/` — policy corpus → local embeddings → ChromaDB retrieval → LangGraph routing → deterministic mock generation → Pydantic → FastAPI → Docker.

The repository intentionally keeps the required baseline free of paid services and API keys. The optional real-LLM path in the support assistant is disabled by default.

## 1. Environment setup

Use Python **3.11** for the smoothest compatibility with the specified libraries.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\\Scripts\\activate
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

The raw Titanic dataset is loaded from Seaborn exactly once, immediately saved as `analytics/titanic.csv`, and then all modeling work reads that committed CSV. Generated written results are stored in `EDA_REPORT.md` and `MODELING_REPORT.md`; supporting plots are under `analytics/plots/`.

## 4. Run the support assistant

The graded mode is the default mock mode. No LLM API key is required.

```bash
python -m support_assistant.run_examples
uvicorn support_assistant.main:app --reload --port 7860
```

Example request:

```bash
curl -X POST http://127.0.0.1:7860/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is the delivery fee for an order below INR 149?\"}"
```

On macOS/Linux, use `\\` line continuations or place the curl command on one line.

The first support-assistant run may download `all-MiniLM-L6-v2` from the model repository. After that, embeddings and retrieval run locally.

## 5. Docker

```bash
cd support_assistant
docker build -t zepto-support .
docker run --rm -p 7860:7860 zepto-support
```

## 6. Git workflow required by the assignment

The repository needs a feature branch with at least two commits, followed by a merge into `main`. Example:

```bash
git init
git add .
git commit -m "Initial project structure"
git branch -M main
git checkout -b feature/zepto-platform

git add .
git commit -m "Build capstone modules"
# make another meaningful change, then:
git add .
git commit -m "Complete validation and documentation"
git checkout main
git merge --no-ff feature/zepto-platform -m "Merge Zepto platform feature"
```

The important part is the visible history: feature branch → at least two commits → merge back to `main`.

## 7. Final submission checklist

- [ ] One public GitHub repository only.
- [ ] Root `README.md` present.
- [ ] `/data_pipeline` has the scraper, cleaning, SQLite schema, five SQL queries, outputs, and SQL/pandas JOIN comparison.
- [ ] `/analytics/titanic.csv` exists after running `01_eda.py`.
- [ ] Analytics report contains all required missing-value percentages, outlier counts, skewness, survival rates, exact six-column correlation matrix, two strongest correlations, four+ interpreted charts, and standardization check.
- [ ] Three classifiers use the same stratified split and leakage-safe preprocessing.
- [ ] Classification metrics, ROC/AUC, imbalance comparison, GridSearchCV and OOB score are recorded.
- [ ] Fare regression includes MAE, RMSE, R², adjusted R² and residual interpretation.
- [ ] `best_pipeline.joblib` contains preprocessing + estimator together and reloads successfully.
- [ ] All eight support documents are present.
- [ ] ChromaDB embeddings and LangGraph three-node flow work in mock mode.
- [ ] Pydantic output has `answer`, `sources`, and `confidence`.
- [ ] FastAPI `/ask` works.
- [ ] Two mock example responses are recorded by `run_examples.py`.
- [ ] Dockerfile builds and serves the API locally.
- [ ] Git history visibly contains the required feature branch and merge.

## Important

The assignment states that the code, analysis and written interpretations must be authored by the student. Use this repository as a working implementation and understand/test each component before submitting it.
