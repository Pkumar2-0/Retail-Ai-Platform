import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.azure_search_service import get_azure_search_config
from rag.chunking.text_chunker import split_text


DOCUMENTS_PATH = PROJECT_ROOT / "rag" / "documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 50


def _get_search_client(config):
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient

    return SearchClient(
        endpoint=config.endpoint,
        index_name=config.index_name,
        credential=AzureKeyCredential(config.key),
    )


def upload_documents():
    load_dotenv()
    config = get_azure_search_config()
    if not config.enabled:
        raise RuntimeError(
            "Set AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, and AZURE_SEARCH_INDEX_NAME in .env first."
        )

    client = _get_search_client(config)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    batch = []
    total_uploaded = 0

    pdf_files = sorted(DOCUMENTS_PATH.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"No PDF files found in {DOCUMENTS_PATH}")

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()
        full_text = "\n".join(page.page_content for page in pages)

        for chunk_index, chunk in enumerate(split_text(full_text)):
            vector = embedding_model.encode(chunk).tolist()
            batch.append(
                {
                    "id": f"{pdf_file.stem}-{chunk_index}",
                    "content": chunk,
                    "source": pdf_file.name,
                    "content_vector": vector,
                }
            )

            if len(batch) >= BATCH_SIZE:
                result = client.upload_documents(batch)
                total_uploaded += sum(1 for item in result if item.succeeded)
                batch = []

    if batch:
        result = client.upload_documents(batch)
        total_uploaded += sum(1 for item in result if item.succeeded)

    print(f"Uploaded {total_uploaded} chunks to Azure AI Search index: {config.index_name}")


if __name__ == "__main__":
    upload_documents()
