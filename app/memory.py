from app.embeddings import EmbeddedDocumentChunk

_DOCUMENT_CHUNKS: list[EmbeddedDocumentChunk] = []


def add_chunks(chunks: list[EmbeddedDocumentChunk]) -> None:
    _DOCUMENT_CHUNKS.extend(chunks)


def list_chunks() -> list[EmbeddedDocumentChunk]:
    return list(_DOCUMENT_CHUNKS)


def clear_chunks() -> None:
    _DOCUMENT_CHUNKS.clear()
