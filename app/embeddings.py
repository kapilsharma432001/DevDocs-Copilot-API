import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from app.chunking import DocumentChunk


load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

@dataclass(frozen=True)
class EmbeddedDocumentChunk:
    id: str
    source_name: str
    chunk_index: int
    content: str
    metadata: dict[str, Any]
    embedding: list[float]

def get_embedding_model() -> OpenAIEmbeddings:
    """
    Create the embedding model client.

    Why this function exists:
    - Keeps model creation in one place.
    - Makes it easy to switch providers later.
    - Keeps FastAPI route code clean.
    """

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
    )

def embed_chunks(chunks: list[DocumentChunk]) -> list[EmbeddedDocumentChunk]:
    """
    Convert document chunks into embedding vectors.

    Each chunk gets one vector.

    Later:
    - these vectors will be stored in pgvector
    - user query will also be embedded
    - vector similarity search will find the most relevant chunks
    """

    if not chunks:
        return []

    embedding_model = get_embedding_model()

    texts = [chunk.content for chunk in chunks]

    vectors = embedding_model.embed_documents(texts)
    embedded_chunks: list[EmbeddedDocumentChunk] = []


    for chunk, vector in zip(chunks, vectors):
        embedded_chunks.append(
            EmbeddedDocumentChunk(
                id=chunk.id,
                source_name=chunk.source_name,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                metadata={
                    **chunk.metadata,
                    "embedding_model": EMBEDDING_MODEL,
                    "embedding_dimensions": EMBEDDING_DIMENSIONS,
                },
                embedding=vector,
            )
        )

    print(f"Embedded {len(embedded_chunks)} - chunks:- \n{embedded_chunks} -  chunks using model {EMBEDDING_MODEL} with {EMBEDDING_DIMENSIONS} dimensions.")
    return embedded_chunks


def embed_query_text(text: str) -> list[float]:
    """
    Convert a user question/search query into an embedding vector.

    Why?
    - Document chunks and user questions must be embedded into the same vector space.
    - Once both are vectors, we can compare them using cosine similarity.
    """
    normalized_text = text.strip()

    if not normalized_text:
        return []

    embedding_model = get_embedding_model()
    return embedding_model.embed_query(normalized_text)


