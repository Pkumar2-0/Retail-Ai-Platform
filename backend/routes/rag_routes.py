from fastapi import APIRouter

from backend.services.rag_service import (
    search_documents
)

router = APIRouter()

@router.get("/search-documents")
async def search_docs(query: str):

    results = search_documents(query)

    return {
        "query": query,
        "results": results
    }