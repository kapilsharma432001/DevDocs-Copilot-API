from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from app.chunking import DocumentChunk
from app.embeddings import get_embedding_model

_vector_store = InMemoryVectorStore(
    embedding=get_embedding_model(),
)


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

    return _vector_store.add_documents(
        documents=documents,
        ids=ids,
    )


def similarity_search(query: str, top_k: int = 3) -> list[tuple[Document, float]]:
    """
    Search for chunks that are semantically similar to the query.
    """

    return _vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
    )