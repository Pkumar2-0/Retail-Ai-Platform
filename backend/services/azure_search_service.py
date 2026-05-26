import os
from dataclasses import dataclass

from dotenv import load_dotenv


AZURE_SEARCH_VECTOR_FIELD = "content_vector"


@dataclass
class AzureSearchConfig:
    endpoint: str | None
    key: str | None
    index_name: str | None

    @property
    def enabled(self):
        return all([self.endpoint, self.key, self.index_name])


def get_azure_search_config():
    load_dotenv()
    return AzureSearchConfig(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        key=os.getenv("AZURE_SEARCH_API_KEY"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", "retail-documents"),
    )


def is_azure_search_configured():
    return get_azure_search_config().enabled


def _get_search_client(config):
    try:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
    except ImportError as exc:
        raise RuntimeError(
            "Azure AI Search SDK is not installed. Install azure-search-documents and azure-core."
        ) from exc

    return SearchClient(
        endpoint=config.endpoint,
        index_name=config.index_name,
        credential=AzureKeyCredential(config.key),
    )


def search_azure_documents(query, query_embedding, n_results=3):
    try:
        from azure.search.documents.models import VectorizedQuery
    except ImportError as exc:
        raise RuntimeError(
            "Azure AI Search SDK is not installed. Install azure-search-documents and azure-core."
        ) from exc

    config = get_azure_search_config()
    if not config.enabled:
        raise RuntimeError(
            "Azure AI Search is not configured. Set AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, and AZURE_SEARCH_INDEX_NAME."
        )

    client = _get_search_client(config)
    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=n_results,
        fields=AZURE_SEARCH_VECTOR_FIELD,
    )

    results = client.search(
        search_text=query,
        vector_queries=[vector_query],
        select=["id", "content", "source"],
        top=n_results,
    )

    documents = []
    sources = []
    for result in results:
        documents.append(result.get("content", ""))
        sources.append(result.get("source", "Azure AI Search"))

    return {
        "documents": documents,
        "sources": sources,
        "backend": "Azure AI Search",
    }
