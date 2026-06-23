"""FastAPI dependency providers. Real wiring happens once retrieval/generation are implemented."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, settings
from app.db.session import SessionLocal, get_session
from app.generation.answerer import Answerer
from app.generation.llm_client import LLMClient
from app.ingest.embedder import OpenAIEmbedder
from app.retrieval.dense import DenseRetriever

# Lazy-initialized singleton retriever
_dense_retriever: DenseRetriever | None = None

# Lazy-initialized singleton answerer
_answerer: Answerer | None = None


async def get_db() -> AsyncIterator[AsyncSession]:
    """Yield a database session per request; rollback on error."""
    async for session in get_session():
        yield session


def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    return settings


def get_dense_retriever() -> DenseRetriever:
    """Return the singleton DenseRetriever instance (lazy initialization)."""
    global _dense_retriever
    if _dense_retriever is None:
        embedder = OpenAIEmbedder(
            api_key=settings.OPENAI_API_KEY,
            model=settings.EMBEDDING_MODEL,
        )
        _dense_retriever = DenseRetriever(
            db_session_maker=SessionLocal,
            embedder=embedder,
        )
    return _dense_retriever


def get_answerer() -> Answerer:
    """Return the singleton Answerer instance (lazy initialization).

    Builds the full RAG pipeline (dense + BM25 retrieval, reranker, LLM client)
    from application settings on first call.
    """
    global _answerer
    if _answerer is None:
        embedder = OpenAIEmbedder(
            api_key=settings.OPENAI_API_KEY,
            model=settings.EMBEDDING_MODEL,
        )
        llm_client = LLMClient(
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        _answerer = Answerer(
            db_session_maker=SessionLocal,
            embedder=embedder,
            llm_client=llm_client,
        )
    return _answerer


DbSession = Annotated[AsyncSession, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
DenseRetrieverDep = Annotated[DenseRetriever, Depends(get_dense_retriever)]
AnswererDep = Annotated[Answerer, Depends(get_answerer)]

