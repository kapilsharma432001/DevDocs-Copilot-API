from typing import List, Optional, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    service: str = Field(..., examples=["DevDocs Copilot API"])


class IngestRequest(BaseModel):
    source_name: str = Field(
        ...,
        examples=["system-design-notes.md"],
        description="Name of the document or note being ingested.",
    )
    text: str = Field(
        ...,
        examples=["Consistent hashing is used to distribute keys across nodes..."],
        description="Raw text content that should be chunked, embedded, and stored.",
    )


class ChunkPreview(BaseModel):
    chunk_id: str
    chunk_index: int
    char_count: int
    preview: str


class IngestResponse(BaseModel):
    message: str
    source_name: str
    chunks_created: int
    embedding_model: str
    embedding_dimensions: int
    chunks: List[ChunkPreview] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        examples=["What is consistent hashing?"],
        description="Question that should be answered using the indexed documents.",
    )


class SourceChunk(BaseModel):
    source_name: str
    chunk_text: str
    score: Optional[float] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


class AgentAskRequest(BaseModel):
    question: str = Field(
        ...,
        examples=["Explain RAG and create a revision task for me."],
        description="Question or instruction that may require retrieval and tool usage.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of relevant chunks to retrieve before generating the answer.",
    )
    search_mode: Literal["semantic", "keyword", "hybrid"] = Field(
        default="hybrid",
        description="Retrieval strategy to use before generating the answer.",
    )


class AgentStep(BaseModel):
    step_name: str
    detail: str


class AgentAskResponse(BaseModel):
    answer: str
    steps: List[AgentStep]
    sources: List[SourceChunk]
