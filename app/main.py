from fastapi import FastAPI, HTTPException, status

from app.chunking import chunk_text
from app.memory import add_chunks
from app.schemas import (
    AgentAskRequest,
    AgentAskResponse,
    AgentStep,
    AskRequest,
    AskResponse,
    ChunkPreview,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SourceChunk,
)
from app.embeddings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from app.vector_store import add_chunks_to_vector_store, similarity_search

app = FastAPI(
    title="DevDocs Copilot API",
    description="A production-shaped FastAPI covering concepts - RAG, embeddings, LangChain, LangGraph, and MCP.",
    version="0.1.0",
)

@app.get("/health", response_model = HealthResponse)
def health_check() -> HealthResponse:
        return HealthResponse(
        status="ok",
        service="DevDocs Copilot API",
    )

@app.post("/ingest", response_model = IngestResponse)
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


@app.post("/ask", response_model = AgentAskResponse)
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
    
    return AskResponse(
        answer=(
            "Semantic search completed using LangChain vector store. "
            "RAG answer generation is not implemented yet."
        ),
        sources=[
            SourceChunk(
                source_name=document.metadata.get("source_name", "unknown"),
                chunk_text=document.page_content,
                score=round(score, 4),
            )
            for document, score in results
        ],
    )   