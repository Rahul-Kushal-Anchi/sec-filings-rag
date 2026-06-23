"""Tests for cross-encoder reranker."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

from app.retrieval.dense import ChunkResult
from app.retrieval.reranker import CrossEncoderReranker


def make_chunk(chunk_id: str, text: str, score: float = 1.0) -> ChunkResult:
    """Helper to create synthetic ChunkResult for testing."""
    return ChunkResult(
        chunk_id=uuid.UUID(chunk_id),
        filing_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        text=text,
        page_number=1,
        section="Risk Factors",
        score=score,
    )


def test_reranker_empty_chunks_returns_empty() -> None:
    """Reranker should return empty list when given empty chunks."""
    reranker = CrossEncoderReranker()
    result = reranker.rerank("test query", [], top_k=5)
    assert result == []


@patch("app.retrieval.reranker.CrossEncoder")
def test_reranker_top_k_respected(mock_cross_encoder_class: MagicMock) -> None:
    """Reranker should return at most top_k results."""
    # Mock the CrossEncoder model
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9, 0.5, 0.3]
    mock_cross_encoder_class.return_value = mock_model

    # Create 3 chunks
    chunks = [
        make_chunk("00000000-0000-0000-0000-000000000001", "Chunk 1"),
        make_chunk("00000000-0000-0000-0000-000000000002", "Chunk 2"),
        make_chunk("00000000-0000-0000-0000-000000000003", "Chunk 3"),
    ]

    reranker = CrossEncoderReranker()
    result = reranker.rerank("test query", chunks, top_k=2)

    # Should return only top 2
    assert len(result) == 2
    assert all(isinstance(chunk, ChunkResult) for chunk in result)


@patch("app.retrieval.reranker.CrossEncoder")
def test_reranker_scores_normalized_0_to_1(mock_cross_encoder_class: MagicMock) -> None:
    """Reranker should normalize scores to [0, 1] range using sigmoid."""
    # Mock the CrossEncoder model with various scores
    mock_model = MagicMock()
    # Raw cross-encoder scores can be negative or > 1
    mock_model.predict.return_value = [-2.0, 0.0, 5.0]
    mock_cross_encoder_class.return_value = mock_model

    chunks = [
        make_chunk("00000000-0000-0000-0000-000000000001", "Chunk 1"),
        make_chunk("00000000-0000-0000-0000-000000000002", "Chunk 2"),
        make_chunk("00000000-0000-0000-0000-000000000003", "Chunk 3"),
    ]

    reranker = CrossEncoderReranker()
    result = reranker.rerank("test query", chunks, top_k=3)

    # All scores should be normalized to [0, 1]
    for chunk in result:
        assert 0.0 <= chunk.score <= 1.0


@patch("app.retrieval.reranker.CrossEncoder")
def test_reranker_orders_by_score_descending(mock_cross_encoder_class: MagicMock) -> None:
    """Reranker should order chunks by cross-encoder score (descending)."""
    # Mock the CrossEncoder model
    mock_model = MagicMock()
    # Return scores in non-descending order: 0.3, 0.9, 0.5
    mock_model.predict.return_value = [0.3, 0.9, 0.5]
    mock_cross_encoder_class.return_value = mock_model

    chunks = [
        make_chunk("00000000-0000-0000-0000-00000000000a", "Chunk A"),
        make_chunk("00000000-0000-0000-0000-00000000000b", "Chunk B"),
        make_chunk("00000000-0000-0000-0000-00000000000c", "Chunk C"),
    ]

    reranker = CrossEncoderReranker()
    result = reranker.rerank("test query", chunks, top_k=3)

    # Result should be reordered by score: B (0.9), C (0.5), A (0.3)
    assert result[0].chunk_id == uuid.UUID("00000000-0000-0000-0000-00000000000b")
    assert result[1].chunk_id == uuid.UUID("00000000-0000-0000-0000-00000000000c")
    assert result[2].chunk_id == uuid.UUID("00000000-0000-0000-0000-00000000000a")

    # Scores should be in descending order
    assert result[0].score > result[1].score > result[2].score
