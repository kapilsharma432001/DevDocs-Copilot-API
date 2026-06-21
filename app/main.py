from fastapi import FastAPI, HTTPException, status
from app.embeddings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, embed_chunks

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
            embedded_chunks = embed_chunks(chunks)
        except Exception as exc:
            raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to create embeddings: {str(exc)}",
                ) from exc

        add_chunks(embedded_chunks)

        return IngestResponse(
                message="Document chunked successfully.",
                source_name=request.source_name,
                chunks_created=len(chunks),
                embedding_model=EMBEDDING_MODEL,
                embedding_dimensions=EMBEDDING_DIMENSIONS,
                chunks=[
                ChunkPreview(
                        chunk_id=chunk.id,
                        chunk_index=chunk.chunk_index,
                        char_count=len(chunk.content),
                        preview=chunk.content[:160],
                )
                for chunk in embedded_chunks
                ],
        )


@app.post("/agent/ask", response_model = AgentAskResponse)
def ask_agent(request: AgentAskRequest) -> AgentAskResponse:
        # Dummy implementation for now
        return AgentAskResponse(
                answer = "Agent answer generation is not implemented yet. You asked: {request.question}",
                steps = [
                        AgentStep(
                                step_name = "received_question",
                                detail = f"Received the question: {request.question}"
                        )
                ],
                sources = [],
        )