# Retail AI Platform

A Streamlit frontend and FastAPI backend for retail forecasting, anomaly detection, product and sales management, and RAG-based document search.

See [streamlit_app.py](streamlit_app.py) for the UI and [backend/main.py](backend/main.py) for the API.

To run locally:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r streamlit_requirements.txt
uvicorn backend.main:app --reload
streamlit run streamlit_app.py
```
