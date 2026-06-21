from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from app.chunking import DocumentChunk
from app.embeddings import get_embedding_model

VECTOR_STORE_PATH = Path(__file__).resolve().parents[1] / ".data" / "vector_store.json"


def _create_vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore(
        embedding=get_embedding_model(),
    )


def _load_vector_store() -> InMemoryVectorStore:
    if VECTOR_STORE_PATH.exists():
        try:
            return InMemoryVectorStore.load(
                path=str(VECTOR_STORE_PATH),
                embedding=get_embedding_model(),
            )
        except Exception:
            return _create_vector_store()

    return _create_vector_store()


def _persist_vector_store() -> None:
    _vector_store.dump(str(VECTOR_STORE_PATH))


_vector_store = _load_vector_store()


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

    _persist_vector_store()
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
    return len(_vector_store.store)
