# Docker Setup

This project runs as two application containers:

- `backend`: FastAPI APIs, agents, ML models, RAG, Azure integrations
- `streamlit`: Streamlit web app

MongoDB is included as a local fallback container. If `MONGO_URI` is set in `.env`, the backend uses that external MongoDB connection.

## Build and Run

```powershell
docker compose up --build
```

Open:

- Streamlit UI: `http://localhost:8501`
- FastAPI docs: `http://localhost:8000/docs`

## Environment

Create `.env` from `.env.example` and fill your secrets:

```powershell
copy .env.example .env
```

Required for Azure GenAI:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_DEPLOYMENT_NAME`

Required for Azure AI Search:

- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_API_KEY`
- `AZURE_SEARCH_INDEX_NAME`

Do not commit `.env`.

## Useful Commands

```powershell
docker compose ps
docker compose logs -f backend
docker compose logs -f streamlit
docker compose down
```

Rebuild only one service:

```powershell
docker compose build backend
docker compose build streamlit
```
