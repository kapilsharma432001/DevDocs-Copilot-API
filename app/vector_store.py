import os

from langchain_core.documents import Document


from app.chunking import DocumentChunk
from app.embeddings import get_embedding_model
from dotenv import load_dotenv
from langchain_postgres import PGVector

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://devdocs:devdocs@localhost:5433/devdocs",
)

PGVECTOR_COLLECTION_NAME = os.getenv(
    "PGVECTOR_COLLECTION_NAME",
    "devdocs_chunks",
)

def _create_vector_store() -> PGVector:
    return PGVector(
        embeddings=get_embedding_model(),
        collection_name=PGVECTOR_COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )


_vector_store = _create_vector_store()

def add_chunks_to_vector_store(chunks: list[DocumentChunk]) -> list[str]:
    """
    Add document chunks to LangChain's vector store.

    Why?
    - LangChain handles embedding generation.
    - LangChain handles vector storage.
    - LangChain handles similarity search.
    - Later, we can replace InMemoryVectorStore with PGVector.
    """

    documents = [
        Document(
            page_content=chunk.content,
            metadata={
                **chunk.metadata,
                "chunk_id": chunk.id,
                "source_name": chunk.source_name,
                "chunk_index": chunk.chunk_index,
            },
        )
        for chunk in chunks
    ]

    ids = [chunk.id for chunk in chunks]

    ids = _vector_store.add_documents(
        documents=documents,
        ids=ids,
    )

    return ids


def similarity_search(query: str, top_k: int = 3) -> list[tuple[Document, float]]:
    """
    Search for chunks that are semantically similar to the query.
    """

    return _vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
    )


def vector_store_size() -> int:
    """
    PGVector stores data in Postgres, so we do not directly inspect an in-memory dict.

    We will implement an exact DB count later when we add our own document/chunk table.
    """
    return -1
