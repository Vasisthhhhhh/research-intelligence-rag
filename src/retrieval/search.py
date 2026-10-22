import json
from pathlib import Path

from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.generation.generator import Generator


CHUNKS_PATH = Path(
    "data/processed/chunks.json"
)


class Search:

    def __init__(self, embedder=None):

        self.chunks = json.loads(
            CHUNKS_PATH.read_text(
                encoding="utf-8"
            )
        )

        self.embedder = (
            embedder
            if embedder is not None
            else Embedder()
        )

        self.vector_store = VectorStore()

    def search(
        self,
        query: str,
        limit: int = 5,
    ):

        query_vector = (
            self.embedder.embed_text(
                query
            )
        )

        dense_results = (
            self.vector_store.search(
                query_vector,
                limit=limit,
            )
        )

        results = []

        for result in dense_results:

            chunk_id = result.payload[
                "chunk_id"
            ]

            results.append(
                {
                    "chunk": self.chunks[
                        chunk_id
                    ],
                    "score": float(
                        result.score
                    ),
                }
            )

        return results

    def close(self):
        self.vector_store.close()


def search_paper(
    query: str,
    limit: int = 5,
):

    print(f"\nQuery: {query}\n")

    searcher = Search()

    results = searcher.search(
        query,
        limit=limit,
    )

    context_parts = []
    sources = []

    for rank, result in enumerate(
        results,
        start=1,
    ):

        chunk = result["chunk"]

        print("=" * 70)
        print(f"RESULT {rank}")
        print(
            f"Dense Score: "
            f"{result['score']:.4f}"
        )
        print(
            f"Page: "
            f"{chunk['page_number']}"
        )
        print(
            f"Chunk: "
            f"{chunk['chunk_id']}"
        )

        context_parts.append(
            f"[Page {chunk['page_number']}, "
            f"Chunk {chunk['chunk_id']}]\n"
            f"{chunk['text']}"
        )

        sources.append(
            {
                "page": chunk["page_number"],
                "chunk": chunk["chunk_id"],
            }
        )

    context = "\n\n".join(
        context_parts
    )

    print("\nGenerating answer...\n")

    generator = Generator()

    answer = generator.generate(
        query=query,
        context=context,
    )

    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(answer)

    print("\nSOURCES")

    for source in sources:
        print(
            f"[Page {source['page']}, "
            f"Chunk {source['chunk']}]"
        )

    searcher.close()


if __name__ == "__main__":

    query = input(
        "Ask a question about the research paper:\n\n"
    )

    search_paper(query)