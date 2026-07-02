from langchain_core.documents import Document

from app.keyword_search import keyword_search
from app.vector_store import similarity_search


DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_MULTIPLIER = 3
MIN_CANDIDATE_LIMIT = 10


def _document_key(document: Document) -> str:
    """
    Create a stable key so the same chunk from semantic and keyword search
    can be merged into one final result.
    """

    chunk_id = document.metadata.get("chunk_id")

    if chunk_id:
        return str(chunk_id)

    source_name = document.metadata.get("source_name", "unknown")
    chunk_index = document.metadata.get("chunk_index", "unknown")

    return f"{source_name}:{chunk_index}:{hash(document.page_content)}"


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    top_k: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[tuple[Document, float]]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.

    RRF does not compare raw semantic scores and keyword scores.
    It only uses rank positions.
    """

    fused_scores: dict[str, float] = {}
    documents_by_key: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, document in enumerate(ranked_list, start=1):
            key = _document_key(document)

            documents_by_key[key] = document
            fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

    sorted_keys = sorted(
        fused_scores,
        key=lambda key: fused_scores[key],
        reverse=True,
    )

    return [
        (
            documents_by_key[key],
            round(fused_scores[key], 6),
        )
        for key in sorted_keys[:top_k]
    ]


def hybrid_search(query: str, top_k: int = 3) -> list[tuple[Document, float]]:
    """
    Run semantic search and keyword search, then combine them using RRF.

    Flow:
    1. Get semantic candidates from PGVector.
    2. Get keyword candidates from Postgres full-text search.
    3. Fuse rankings using RRF.
    4. Return final top_k chunks.
    """

    candidate_limit = max(top_k * DEFAULT_CANDIDATE_MULTIPLIER, MIN_CANDIDATE_LIMIT)

    semantic_results = similarity_search(
        query=query,
        top_k=candidate_limit,
    )

    keyword_results = keyword_search(
        query=query,
        limit=candidate_limit,
    )

    semantic_documents = [document for document, _score in semantic_results]
    keyword_documents = [document for document, _score in keyword_results]

    return reciprocal_rank_fusion(
        ranked_lists=[
            semantic_documents,
            keyword_documents,
        ],
        top_k=top_k,
    )