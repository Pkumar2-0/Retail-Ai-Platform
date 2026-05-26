import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.azure_search_service import get_azure_search_config


def create_index():
    try:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents.indexes import SearchIndexClient
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SimpleField,
            VectorSearch,
            VectorSearchProfile,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install Azure Search SDK first: venv\\Scripts\\pip.exe install azure-search-documents azure-core"
        ) from exc

    load_dotenv()
    config = get_azure_search_config()
    if not config.enabled:
        raise RuntimeError(
            "Set AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, and AZURE_SEARCH_INDEX_NAME in .env first."
        )

    client = SearchIndexClient(
        endpoint=config.endpoint,
        credential=AzureKeyCredential(config.key),
    )

    index = SearchIndex(
        name=config.index_name,
        fields=[
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            SimpleField(
                name="source",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=384,
                vector_search_profile_name="retail-vector-profile",
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(name="retail-hnsw"),
            ],
            profiles=[
                VectorSearchProfile(
                    name="retail-vector-profile",
                    algorithm_configuration_name="retail-hnsw",
                ),
            ],
        ),
    )

    client.create_or_update_index(index)
    print(f"Azure AI Search index ready: {config.index_name}")


if __name__ == "__main__":
    create_index()
