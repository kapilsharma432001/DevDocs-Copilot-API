import os

import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from psycopg.rows import dict_row


load_dotenv()


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://devdocs:devdocs@localhost:5433/devdocs",
)

PGVECTOR_COLLECTION_NAME = os.getenv(
    "PGVECTOR_COLLECTION_NAME",
    "devdocs_chunks",
)


def _to_psycopg_url(database_url: str) -> str:
    """
    psycopg.connect expects postgresql:// style URLs.

    SQLAlchemy/LangChain can use:
    postgresql+psycopg://...

    psycopg directly uses:
    postgresql://...
    """

    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)

# keyword search directly queries the tables created by LangChain PGVector
def keyword_search(query: str, limit: int = 10) -> list[tuple[Document, float]]:
    """
    Run keyword-based full-text search over the PGVector document table.

    Why?
    - Semantic search is good for meaning.
    - Keyword search is good for exact terms like function names, IDs, APIs, acronyms.
    - Hybrid search combines this with vector search.
    """

    normalized_query = query.strip()

    if not normalized_query:
        return []

    sql = """
        WITH search_query AS (
            SELECT websearch_to_tsquery('english', %(query)s) AS ts_query
        )
        SELECT
            e.id,
            e.document,
            e.cmetadata,
            ts_rank_cd(
                to_tsvector('english', coalesce(e.document, '')),
                search_query.ts_query
            ) AS keyword_score
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c
            ON e.collection_id = c.uuid
        CROSS JOIN search_query
        WHERE c.name = %(collection_name)s
          AND search_query.ts_query @@ to_tsvector('english', coalesce(e.document, ''))
        ORDER BY keyword_score DESC
        LIMIT %(limit)s;
    """

    connection_url = _to_psycopg_url(DATABASE_URL)

    with psycopg.connect(connection_url, row_factory=dict_row) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                sql,
                {
                    "query": normalized_query,
                    "collection_name": PGVECTOR_COLLECTION_NAME,
                    "limit": limit,
                },
            )
            rows = cursor.fetchall()

    results: list[tuple[Document, float]] = []

    for row in rows:
        metadata = row["cmetadata"] or {}
        metadata = {
            **metadata,
            "pgvector_id": str(row["id"]),
            "keyword_score": float(row["keyword_score"]),
            "retrieval_source": "keyword",
        }

        document = Document(
            page_content=row["document"],
            metadata=metadata,
        )

        results.append(
            (
                document,
                float(row["keyword_score"]),
            )
        )

    return results