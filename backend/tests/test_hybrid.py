"""Tests for hybrid retrieval scoring (author will fill in once hybrid.py lands)."""

from __future__ import annotations

import uuid

import pytest

from app.retrieval.dense import ChunkResult
from app.retrieval.hybrid import reciprocal_rank_fusion


@pytest.mark.skip(reason="hybrid retrieval not yet implemented")
def test_hybrid_rrf_combines_two_rankings() -> None:
    """Reciprocal-rank-fusion must merge two rankings into a single sorted list."""
    # TODO: instantiate scorer, feed two ranked lists, assert fused order.
    assert False


@pytest.mark.skip(reason="hybrid retrieval not yet implemented")
def test_hybrid_handles_empty_sparse_results() -> None:
    """Hybrid scorer must not crash if BM25 returns no hits."""
    # TODO: assert graceful fallback to dense-only ranking.
    assert False


# Helper for creating synthetic ChunkResult objects
def make_chunk(chunk_id: str, text: str, score: float = 1.0) -> ChunkResult:
    return ChunkResult(
        chunk_id=uuid.UUID(chunk_id),
        filing_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        text=text,
        page_number=1,
        section="Risk Factors",
        score=score,
    )


def test_rrf_empty_inputs_returns_empty() -> None:
    """Empty dense and BM25 results should return empty list."""
    result = reciprocal_rank_fusion([], [], top_k=5)
    assert result == []


def test_rrf_top_k_respected() -> None:
    """RRF should return at most top_k results."""
    # Create 5 chunks
    chunks = [
        make_chunk(f"00000000-0000-0000-0000-00000000000{i}", f"Chunk {i}")
        for i in range(1, 6)
    ]
    
    # All 5 in dense, first 3 in BM25
    dense_results = chunks
    bm25_results = chunks[:3]
    
    result = reciprocal_rank_fusion(dense_results, bm25_results, top_k=3)
    
    assert len(result) == 3
    assert all(isinstance(item, ChunkResult) for item in result)


def test_rrf_chunk_appearing_in_both_lists_scores_higher() -> None:
    """Chunk appearing in both dense and BM25 lists should score highest."""
    # Create 3 chunks
    chunk_a = make_chunk("00000000-0000-0000-0000-00000000000a", "Chunk A")
    chunk_b = make_chunk("00000000-0000-0000-0000-00000000000b", "Chunk B")
    chunk_c = make_chunk("00000000-0000-0000-0000-00000000000c", "Chunk C")
    
    # A ranks 1st in dense, B 2nd, C 3rd
    dense_results = [chunk_a, chunk_b, chunk_c]
    
    # A ranks 1st in BM25, C 2nd (B absent)
    bm25_results = [chunk_a, chunk_c]
    
    result = reciprocal_rank_fusion(dense_results, bm25_results, top_k=3)
    
    # Chunk A appears in both lists, so should have highest score
    assert result[0].chunk_id == chunk_a.chunk_id
    assert result[0].score >= result[1].score
    assert result[0].score >= result[2].score

