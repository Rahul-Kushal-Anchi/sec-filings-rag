"""Cross-encoder reranker (ms-marco-MiniLM-L-6-v2) for two-stage retrieval."""

from __future__ import annotations

import math

import structlog
from sentence_transformers import CrossEncoder

from app.retrieval.dense import ChunkResult

logger = structlog.get_logger().bind(retriever="reranker")


class CrossEncoderReranker:
    """Rerank retrieved chunks using a cross-encoder model for more accurate scoring.
    
    The cross-encoder scores query-chunk pairs directly, providing more accurate
    relevance scores than embedding similarity alone. Model is lazy-loaded on first use.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """Initialize reranker with model name (model loaded lazily on first use).
        
        Args:
            model_name: HuggingFace model identifier for cross-encoder.
        """
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def rerank(
        self, query: str, chunks: list[ChunkResult], top_k: int = 5
    ) -> list[ChunkResult]:
        """Rerank chunks using cross-encoder scores.
        
        Args:
            query: User query string.
            chunks: List of chunks to rerank.
            top_k: Number of top results to return.
            
        Returns:
            Top-k chunks reranked by cross-encoder score (descending).
        """
        if not chunks:
            logger.debug("rerank_called_with_empty_chunks")
            return []

        # Lazy-load model on first use
        if self._model is None:
            logger.info("loading_cross_encoder_model", model=self.model_name)
            self._model = CrossEncoder(self.model_name)

        # Create query-chunk pairs for scoring
        pairs = [[query, chunk.text] for chunk in chunks]

        # Get raw scores from cross-encoder
        raw_scores = self._model.predict(pairs)  # type: ignore[arg-type]
        logger.debug(
            "reranking_complete",
            num_chunks=len(chunks),
            top_k=top_k,
        )

        # Normalize scores to [0, 1] using sigmoid and sort descending
        scored_chunks = []
        for chunk, raw_score in zip(chunks, raw_scores, strict=True):
            # Sigmoid normalization: 1 / (1 + e^(-score))
            normalized_score = 1.0 / (1.0 + math.exp(-float(raw_score)))

            # Create new ChunkResult with updated score
            reranked_chunk = ChunkResult(
                chunk_id=chunk.chunk_id,
                filing_id=chunk.filing_id,
                text=chunk.text,
                page_number=chunk.page_number,
                section=chunk.section,
                ticker=chunk.ticker,
                form_type=chunk.form_type,
                filing_date=chunk.filing_date,
                score=normalized_score,
            )
            scored_chunks.append(reranked_chunk)

        # Sort by score descending and return top_k
        scored_chunks.sort(key=lambda c: c.score, reverse=True)
        return scored_chunks[:top_k]
