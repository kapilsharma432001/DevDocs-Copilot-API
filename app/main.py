from fastapi import FastAPI, HTTPException, status

from app.chunking import chunk_text
from app.schemas import (
    AgentAskRequest,
    AgentAskResponse,
    AgentStep,
    ChunkPreview,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SourceChunk,
)
from app.embeddings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from app.vector_store import add_chunks_to_vector_store, similarity_search, vector_store_size
from app.rag import generate_rag_answer

app = FastAPI(
    title="DevDocs Copilot API",
    description="A production-shaped FastAPI covering concepts - RAG, embeddings, LangChain, LangGraph, and MCP.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="DevDocs Copilot API",
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest_document(request: IngestRequest) -> IngestResponse:

    chunks = chunk_text(
        source_name=request.source_name,
        text=request.text,
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty.",
        )

    try:
        add_chunks_to_vector_store(chunks)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add chunks to vector store: {str(exc)}",
        ) from exc

    return IngestResponse(
        message="Document ingested successfully into PGVector.",
        source_name=request.source_name,
        chunks_created=len(chunks),
        embedding_model=EMBEDDING_MODEL,
        embedding_dimensions=EMBEDDING_DIMENSIONS,
        chunks=[
            ChunkPreview(
                chunk_id=chunk.id,
                chunk_index=chunk.chunk_index,
                char_count=chunk.metadata.get("char_count", len(chunk.content)),
                preview=chunk.content[:160],
            )
            for chunk in chunks
        ],
    )


@app.post("/ask", response_model=AgentAskResponse)
def ask_agent(request: AgentAskRequest) -> AgentAskResponse:
    try:
        results = similarity_search(
            query=request.question,
            top_k=request.top_k,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform semantic search: {str(exc)}",
        ) from exc

    try:
        rag_answer = generate_rag_answer(
            question=request.question,
            results=results,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate RAG answer: {str(exc)}",
        ) from exc

    generation_detail = "Generated the final answer using the retrieved document context."
    if not results:
        generation_detail = "Skipped LLM generation because no indexed context was available."

    store_size = vector_store_size()
    store_detail = (
        "Vector store size is not calculated for PGVector yet."
        if store_size == -1
        else f"The store currently contains {store_size} chunk(s)."
)
    return AgentAskResponse(
        answer=rag_answer.answer,
        steps=[
            AgentStep(
                step_name="retrieve",
                detail=(
                    f"Retrieved {len(results)} relevant chunk(s) from PGVector. {store_detail}"
                ),
            ),
            AgentStep(
                step_name="generate",
                detail=generation_detail,
            ),
        ],
        sources=[
            SourceChunk(
                source_name=source.source_name,
                chunk_text=source.chunk_text,
                score=source.score,
            )
            for source in rag_answer.sources
        ],
    )
