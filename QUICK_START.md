# Absolute beginner quick start

1. Install Python 3.11.
2. Extract/open this folder in VS Code.
3. Open VS Code Terminal.
4. Create the virtual environment:
   `python -m venv .venv`
5. Activate it on Windows PowerShell:
   `.venv\\Scripts\\Activate.ps1`
6. Install everything:
   `pip install -r requirements.txt`
7. Run:
   `python data_pipeline/scrape_and_load.py`
8. Then:
   `python data_pipeline/queries.py`
9. Then:
   `python analytics/01_eda.py`
10. Then:
   `python analytics/02_modeling.py`
11. Then:
   `python -m support_assistant.run_examples`
12. Start the API:
   `uvicorn support_assistant.main:app --reload --port 7860`
13. Open `http://127.0.0.1:7860/docs` in your browser to use the FastAPI Swagger interface.

If a command gives an error, copy the complete error message and send it back rather than changing random lines. The error can then be diagnosed specifically.
