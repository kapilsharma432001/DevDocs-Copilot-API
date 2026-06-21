from dataclasses import dataclass
from typing import Any
from uuid import uuid4


from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    source_name: str
    chunk_index: int
    content: str
    metadata: dict[str, Any]


def chunk_text(source_name: str, text: str) -> list[DocumentChunk]:
    """
    Split a raw document/note into smaller chunks.

    Why?
    - LLMs should not receive huge documents directly.
    - Retrieval systems work better when they search smaller meaningful chunks.
    - Later, each chunk will get its own embedding and will be stored in pgvector.
    """
    normalized_text = text.strip()

    if not normalized_text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    documents = splitter.create_documents(
        texts=[normalized_text],
        metadatas=[
            {
                "source_name": source_name,
            }
        ],
    )
    chunks: list[DocumentChunk] = []

    for index, document in enumerate(documents):
        chunk = DocumentChunk(
            id=str(uuid4()),
            source_name=source_name,
            chunk_index=index,
            content=document.page_content,
            metadata={
                **document.metadata,
                "chunk_index": index,
                "char_count": len(document.page_content),
            },
        )
        chunks.append(chunk)

    return chunks
