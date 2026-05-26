# Azure AI Search Setup

This project can use Azure AI Search as the cloud vector store for the Document Assistant Agent. If Azure AI Search is not configured, RAG falls back to the local ChromaDB store.

## Required Azure Component

Create an Azure AI Search service and an index named `retail-documents`.

Recommended index fields:

- `id`: string, key
- `content`: string, searchable
- `source`: string, filterable
- `content_vector`: vector field with 384 dimensions

The 384 dimensions match the local embedding model: `all-MiniLM-L6-v2`.

## Environment Variables

Add these values locally and in Azure Web App configuration:

```env
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_API_KEY=your-search-admin-or-query-key
AZURE_SEARCH_INDEX_NAME=retail-documents
```

`AZURE_SEARCH_INDEX_NAME` is the index name used by the application. Use `retail-documents` unless you intentionally choose another name.

## Python Package

Install the Azure Search SDK:

```powershell
venv\Scripts\pip.exe install azure-search-documents azure-core
```

## Create the Index

After setting the environment variables, run:

```powershell
venv\Scripts\python.exe -B rag\create_azure_search_index.py
```

The script creates the `retail-documents` index with these fields:

- `id`
- `content`
- `source`
- `content_vector`, 384 dimensions

## Upload Project Documents

After the index exists, upload PDF chunks and embeddings:

```powershell
venv\Scripts\python.exe -B rag\upload_documents_to_azure_search.py
```

The script reads PDFs from `rag/documents`, chunks them, creates embeddings with `all-MiniLM-L6-v2`, and uploads them to Azure AI Search.

## Security

For local development, use environment variables. For production, store `AZURE_SEARCH_API_KEY`, `AZURE_OPENAI_API_KEY`, and `MONGO_URI` in Azure Key Vault and load them through managed identity.
