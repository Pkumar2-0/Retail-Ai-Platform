import os

from dotenv import load_dotenv
from openai import OpenAI

from backend.services.azure_search_service import (
    is_azure_search_configured,
    search_azure_documents,
)

AZURE_EMBEDDING_DIMENSIONS = 384
LOCAL_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_azure_query_embedding(query):
    load_dotenv()
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
        "text-embedding-3-large",
    )

    if not all([endpoint, api_key, deployment]):
        return None

    client = OpenAI(
        api_key=api_key,
        base_url=endpoint.rstrip("/"),
    )
    response = client.embeddings.create(
        model=deployment,
        input=query,
        dimensions=AZURE_EMBEDDING_DIMENSIONS,
    )
    return response.data[0].embedding


def _get_local_query_embedding(query):
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL_NAME)
    return embedding_model.encode(query).tolist()

def _search_chroma_documents(query_embedding, n_results=3):
    import chromadb

    client = chromadb.PersistentClient(
        path="rag/vector_store/chroma_db"
    )
    collection = client.get_collection(
        name="retail_documents"
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return {
        "documents": results["documents"][0],
        "sources": ["ChromaDB"] * len(results["documents"][0]),
        "backend": "ChromaDB",
    }


def search_documents(query):

    search_backend = None
    try:
        if is_azure_search_configured():
            query_embedding = _get_azure_query_embedding(query)
            if query_embedding is None:
                query_embedding = _get_local_query_embedding(query)
            search_backend = search_azure_documents(query, query_embedding)
    except Exception as exc:
        try:
            query_embedding = _get_local_query_embedding(query)
            search_backend = _search_chroma_documents(query_embedding)
            search_backend["fallback_reason"] = str(exc)
        except Exception as fallback_exc:
            return {
                "answer": "Document search is unavailable because Azure AI Search and local Chroma fallback both failed.",
                "documents": [],
                "sources": [],
                "backend": "Unavailable",
                "fallback_reason": f"Azure error: {exc}; local fallback error: {fallback_exc}",
            }

    if search_backend is None:
        query_embedding = _get_local_query_embedding(query)
        search_backend = _search_chroma_documents(query_embedding)

    documents = search_backend["documents"]

    # Simple AI-style response
    answer = f"""
    Based on retail business documents,
    the following insights were found
    related to your query:

    {documents[0]}
    """

    return {
        "answer": answer,
        "documents": documents,
        "sources": search_backend.get("sources", []),
        "backend": search_backend.get("backend", "ChromaDB"),
        "fallback_reason": search_backend.get("fallback_reason"),
    }
