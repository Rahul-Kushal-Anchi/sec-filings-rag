"""Tests for the Answerer orchestrator."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.schemas import Answer
from app.generation.answerer import Answerer
from app.retrieval.dense import ChunkResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(n: int, score: float = 0.9) -> ChunkResult:
    return ChunkResult(
        chunk_id=uuid.uuid4(),
        filing_id=uuid.uuid4(),
        text=f"Apple reported revenue growth in Q{n}. Supply chain risks remain.",
        page_number=n,
        section="Risk Factors",
        score=score,
    )


@pytest.fixture
def mock_answerer():
    db = MagicMock()
    embedder = MagicMock()
    llm = MagicMock()
    llm.complete_with_fallback = AsyncMock(
        return_value=("anthropic", "Apple faces supply chain risks.")
    )
    answerer = Answerer(
        db_session_maker=db,
        embedder=embedder,
        llm_client=llm,
    )
    answerer._bm25_built = True  # skip index build in tests
    return answerer


# ---------------------------------------------------------------------------
# Original stubs (kept so CI doesn't lose them)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="answerer not yet implemented")
def test_answerer_placeholder_1() -> None:
    pass


@pytest.mark.skip(reason="answerer not yet implemented")
def test_answerer_placeholder_2() -> None:
    pass


# ---------------------------------------------------------------------------
# Real tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_answer_returns_answer_object(mock_answerer: Answerer) -> None:
    chunk = make_chunk(1)
    mock_answerer.dense.retrieve = AsyncMock(return_value=[chunk])
    mock_answerer.bm25.retrieve = MagicMock(return_value=[chunk])
    mock_answerer.reranker.rerank = MagicMock(return_value=[chunk])
    with patch("app.generation.answerer.reciprocal_rank_fusion", return_value=[chunk]):
        result = await mock_answerer.answer("What are Apple's risks?")
    assert isinstance(result, Answer)
    assert result.text == "Apple faces supply chain risks."


@pytest.mark.asyncio
async def test_answer_empty_chunks_returns_no_info(mock_answerer: Answerer) -> None:
    mock_answerer.dense.retrieve = AsyncMock(return_value=[])
    mock_answerer.bm25.retrieve = MagicMock(return_value=[])
    mock_answerer.reranker.rerank = MagicMock(return_value=[])
    with patch("app.generation.answerer.reciprocal_rank_fusion", return_value=[]):
        result = await mock_answerer.answer("test question")
    assert result.text == "No relevant information found."
    assert result.citations == []
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_answer_citations_match_chunks(mock_answerer: Answerer) -> None:
    chunks = [make_chunk(1), make_chunk(2)]
    mock_answerer.dense.retrieve = AsyncMock(return_value=chunks)
    mock_answerer.bm25.retrieve = MagicMock(return_value=chunks)
    mock_answerer.reranker.rerank = MagicMock(return_value=chunks)
    with patch("app.generation.answerer.reciprocal_rank_fusion", return_value=chunks):
        result = await mock_answerer.answer("What are Apple's risks?")
    assert len(result.citations) == 2
    assert result.citations[0].page_number == 1


@pytest.mark.asyncio
async def test_answer_confidence_is_top_chunk_score(mock_answerer: Answerer) -> None:
    chunk = make_chunk(1, score=0.87)
    mock_answerer.dense.retrieve = AsyncMock(return_value=[chunk])
    mock_answerer.bm25.retrieve = MagicMock(return_value=[chunk])
    mock_answerer.reranker.rerank = MagicMock(return_value=[chunk])
    with patch("app.generation.answerer.reciprocal_rank_fusion", return_value=[chunk]):
        result = await mock_answerer.answer("What are Apple's risks?")
    assert result.confidence == pytest.approx(0.87, abs=0.01)