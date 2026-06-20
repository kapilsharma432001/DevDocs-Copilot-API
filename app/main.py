from fastapi import FastAPI

from app.schemas import (
    AgentAskRequest,
    AgentAskResponse,
    AgentStep,
    AskRequest,
    AskResponse,
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
        # For now it's just a dummy implementation
        return IngestResponse(
        message="Document received successfully. Chunking is not implemented yet.",
        source_name=request.source_name,
        chunks_created=0,
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