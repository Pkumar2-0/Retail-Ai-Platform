import os
import sys
from pathlib import Path

import chromadb

from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import PyPDFLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.chunking.text_chunker import split_text

# Load embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Initialize ChromaDB
client = chromadb.PersistentClient(
    path="rag/vector_store/chroma_db"
)

collection = client.get_or_create_collection(
    name="retail_documents"
)

# Documents folder
documents_path = "rag/documents"

# Read all PDFs
pdf_files = [
    file for file in os.listdir(documents_path)
    if file.endswith(".pdf")
]

document_id = 0

for pdf_file in pdf_files:

    pdf_path = os.path.join(
        documents_path,
        pdf_file
    )

    print(f"Processing: {pdf_file}")

    # Load PDF
    loader = PyPDFLoader(pdf_path)

    pages = loader.load()

    # Extract full text
    full_text = ""

    for page in pages:
        full_text += page.page_content

    # Split into chunks
    chunks = split_text(full_text)

    # Generate embeddings
    for chunk in chunks:

        embedding = embedding_model.encode(
            chunk
        ).tolist()

        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(document_id)]
        )

        document_id += 1

print("Embeddings generated successfully!")
