from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


COLLECTION_NAME = "research_papers"
VECTOR_SIZE = 768


class VectorStore:
    def __init__(self, path:str = "data/qdrant"):
        """Initialize the Qdrant client."""
        self.client = QdrantClient(path=path)


    def close(self):
        self.client.close()


    """Creates vector collection if it already doesn't exist."""
    def create_collection(self):   
        collections = self.client.get_collections().collections

        collection_names = [collection.name for collection in collections]

        if COLLECTION_NAME not in collection_names:
            collections = self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

            print(f"Created collection:{COLLECTION_NAME}")


        else:
            print(f"Collection already exists: {COLLECTION_NAME}")


    def add_chunks(self, chunks, embeddings):
        """Store chunks and their embeddings in qdrant."""

        points = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            point = PointStruct(
            id=index,
            vector=embedding,
            payload={
                "text": chunk["text"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
            },
        )


            points.append(point)

        self.client.upload_points(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        print(f"Stored {len(points)} chunks in Qdrant.")


    def search(self, query_vector, limit=5):
        """Find the most relevant chunks for a query vector"""

        results = self.client.query_points(
            collection_name = COLLECTION_NAME,
            query = query_vector,
            limit = limit,
        )

        return results.points

    

if __name__ == "__main__":
    store = VectorStore()

    store.create_collection()

    print("Qdrant vector store is ready.")