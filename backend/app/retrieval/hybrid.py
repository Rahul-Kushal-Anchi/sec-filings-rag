"""Hybrid retrieval: Reciprocal Rank Fusion over dense + BM25 results.

Design decisions documented in DECISIONS.md D3:
- RRF(k=60) combines dense and BM25 rankings without score normalization
- k=60 from Cormack et al. 2009, prevents rank-1 domination
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import replace

from app.retrieval.dense import ChunkResult


def reciprocal_rank_fusion(
    dense_results: list[ChunkResult],
    bm25_results: list[ChunkResult],
    top_k: int = 10,
    k: int = 60,
) -> list[ChunkResult]:
    """Combine dense and BM25 results using Reciprocal Rank Fusion.

    RRF score for a chunk = sum of 1/(k + rank) across all lists
    where rank is 1-indexed position in each result list.

    Args:
        dense_results: Chunks ranked by cosine similarity (index 0 = rank 1)
        bm25_results: Chunks ranked by BM25 score (index 0 = rank 1)
        top_k: Number of results to return
        k: RRF constant (default 60, from original RRF paper)

    Returns:
        Re-ranked list of ChunkResult with score = RRF score, length <= top_k
    """
    all_chunks: dict[uuid.UUID, ChunkResult] = {
        chunk.chunk_id: chunk
        for chunk in dense_results + bm25_results
    }

    rrf_scores: defaultdict[uuid.UUID, float] = defaultdict(float)

    for rank, chunk in enumerate(dense_results):
        rrf_scores[chunk.chunk_id] += 1 / (k + rank + 1)

    for rank, chunk in enumerate(bm25_results):
        rrf_scores[chunk.chunk_id] += 1 / (k + rank + 1)

    sorted_ids = sorted(
        rrf_scores,
        key=lambda chunk_id: rrf_scores[chunk_id],
        reverse=True,
    )

    fused_results: list[ChunkResult] = []
    for chunk_id in sorted_ids[:top_k]:
        chunk = all_chunks[chunk_id]
        fused_results.append(
            replace(
                chunk,
                score=rrf_scores[chunk_id],
            )
        )

    return fused_results