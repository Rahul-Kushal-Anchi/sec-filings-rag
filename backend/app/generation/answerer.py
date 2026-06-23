"""Answer orchestrator: wires retrieval into LLM generation.

Pipeline:
    query
    → DenseRetriever (top 20)  + BM25Retriever (top 20)
    → reciprocal_rank_fusion   (top 20 merged)
    → CrossEncoderReranker     (top 5 final)
    → format_rag_prompt        (system + user prompts)
    → LLMClient.complete       (answer text)
    → Answer with citations
"""
from __future__ import annotations

import time
from datetime import date, datetime

import structlog
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.schemas import Answer, Citation
from app.generation.llm_client import LLMClient
from app.generation.prompts import format_rag_prompt
from app.ingest.embedder import OpenAIEmbedder
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.reranker import CrossEncoderReranker

logger = structlog.get_logger().bind(component="answerer")


class Answerer:
    """Orchestrates the full RAG pipeline from query to cited answer."""

    def __init__(
        self,
        db_session_maker: async_sessionmaker,
        embedder: OpenAIEmbedder,
        llm_client: LLMClient,
    ) -> None:
        self.dense = DenseRetriever(db_session_maker, embedder)
        self.bm25 = BM25Retriever(db_session_maker)
        self.reranker = CrossEncoderReranker()
        self.llm = llm_client
        self._bm25_built = False

    async def _ensure_bm25_index(self) -> None:
        """Build BM25 index on first call using lazy initialization."""
        if not self._bm25_built:
            count = await self.bm25.build_index()
            logger.info("bm25_index_ready", chunks_indexed=count)
            self._bm25_built = True

    async def answer(self, question: str, max_results: int = 5) -> Answer:
        """Run the full RAG pipeline and return a cited answer.

        Args:
            question: User's natural language question.
            max_results: Number of final chunks to pass to the LLM.

        Returns:
            Answer with generated text, citations, confidence, and latency.
        """
        t0 = time.time()

        await self._ensure_bm25_index()

        dense_results = await self.dense.retrieve(question, top_k=20)
        bm25_results = self.bm25.retrieve(question, top_k=20)

        fused_results = reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            top_k=20,
        )

        reranked = self.reranker.rerank(
            question,
            fused_results,
            top_k=max_results,
        )

        latency_ms = int((time.time() - t0) * 1000)

        if not reranked:
            logger.info(
                "answer_no_relevant_chunks",
                question_len=len(question),
                latency_ms=latency_ms,
            )
            return Answer(
                text="No relevant information found.",
                citations=[],
                confidence=0.0,
                latency_ms=latency_ms,
            )

        chunk_dicts = [
            {
                "text": c.text,
                "section": c.section or "Unknown",
                "page_number": c.page_number,
                "ticker": c.ticker,
                "form_type": c.form_type,
            }
            for c in reranked
        ]

        system_prompt, user_prompt = format_rag_prompt(question, chunk_dicts)

        provider, answer_text = await self.llm.complete_with_fallback(
            system_prompt,
            user_prompt,
        )

        citations = [
            Citation(
                filing_ticker=chunk.ticker,
                filing_form_type=chunk.form_type,
                filing_date=datetime.strptime(chunk.filing_date, "%Y-%m-%d").date() if chunk.filing_date != "1970-01-01" else date(1970, 1, 1),
                page_number=chunk.page_number,
                snippet=chunk.text[:200],
                score=max(0.0, min(float(chunk.score), 1.0)),
            )
            for chunk in reranked
        ]

        latency_ms = int((time.time() - t0) * 1000)
        confidence = max(0.0, min(float(reranked[0].score), 1.0))

        logger.info(
            "answer_complete",
            provider=provider,
            chunks_used=len(reranked),
            confidence=confidence,
            latency_ms=latency_ms,
        )

        return Answer(
            text=answer_text,
            citations=citations,
            confidence=confidence,
            latency_ms=latency_ms,
        )