from app.chunking import DocumentChunk


_DOCUMENT_CHUNKS: list[DocumentChunk] = []


def add_chunks(chunks: list[DocumentChunk]) -> None:
    _DOCUMENT_CHUNKS.extend(chunks)


def list_chunks() -> list[DocumentChunk]:
    return list(_DOCUMENT_CHUNKS)


def clear_chunks() -> None:
    _DOCUMENT_CHUNKS.clear()