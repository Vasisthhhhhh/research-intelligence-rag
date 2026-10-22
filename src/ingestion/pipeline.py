from pathlib import Path
import json

from src.ingestion.pdf_loader import extract_text_from_pdf
from src.ingestion.chunker import chunk_pages
from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore


PDF_PATH = Path(
    "data/raw/Predicting_drug_response_from_single-cell_expressi.pdf"
)

CHUNKS_PATH = Path(
    "data/processed/chunks.json"
)


def save_chunks(chunks, path=CHUNKS_PATH):
    """Save processed chunks so other retrieval methods can reuse them."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved {len(chunks)} chunks to {path}")


def run_ingestion():

    print("Loading PDF...")

    pages = extract_text_from_pdf(str(PDF_PATH))

    print(f"Pages extracted: {len(pages)}")

    chunks = chunk_pages(pages)

    print(f"Chunks created: {len(chunks)}")

    save_chunks(chunks)

    print("\nCreating embeddings...")

    embedder = Embedder()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedder.embed_texts(texts)

    print(f"Embeddings created: {len(embeddings)}")

    print("\nStoring vectors in Qdrant...")

    vector_store = VectorStore()

    vector_store.create_collection()

    vector_store.add_chunks(
        chunks,
        embeddings,
    )

    print("\nIngestion complete!")


if __name__ == "__main__":
    run_ingestion()