from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_chat_model

RAG_SYSTEM_PROMPT = """
You are DevDocs Copilot, a careful technical assistant.

Answer the user's question using only the provided context.

Rules:
1. If the context does not contain enough information, say:
   "I do not have enough information in the indexed documents to answer this."
2. Do not invent facts.
3. Keep the answer clear and practical.
4. If useful, explain the idea in simple language.
5. Mention source names used at the end.
""".strip()


@dataclass(frozen=True)
class RagSource:
    source_name: str
    chunk_text: str
    score: float | None = None


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[RagSource]


def format_context(results: list[tuple[Document, float]]) -> str:
    """
    Convert retrieved chunks into a clean context block for the LLM.

    Why?
    - The LLM does not directly know about our vector store.
    - We must explicitly place retrieved content inside the prompt.
    """

    context_parts: list[str] = []

    for index, (document, score) in enumerate(results, start=1):
        source_name = document.metadata.get("source_name", "unknown")
        chunk_index = document.metadata.get("chunk_index", "unknown")

        context_parts.append(f"""
                [Source {index}]
                source_name: {source_name}
                chunk_index: {chunk_index}
                retrieval_score: {score}

                content:
                {document.page_content}
                """.strip())

    return "\n\n---\n\n".join(context_parts)

def generate_rag_answer(
    question: str,
    results: list[tuple[Document, float]],
) -> RagAnswer:
    """
    Generate a grounded answer using retrieved chunks.

    Flow:
    - format retrieved chunks as context
    - build a prompt
    - call the chat model
    - return answer + sources
    """

    if not results:
        return RagAnswer(
            answer="No relevant chunks were found. Please ingest documents first.",
            sources=[],
        )

    context = format_context(results)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            (
                "human",
                """
                Question:
                {question}

                Context:
                {context}
                """.strip(),
                            ),
                        ]
                    )

    model = get_chat_model()

    chain = prompt | model # langchain expression language style 

    response = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    sources = [
        RagSource(
            source_name=document.metadata.get("source_name", "unknown"),
            chunk_text=document.page_content,
            score=round(score, 4),
        )
        for document, score in results
    ]

    return RagAnswer(
        answer=response.content,
        sources=sources,
    )
