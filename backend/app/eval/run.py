"""Offline evaluation runner: executes the golden set against the RAG pipeline."""

from __future__ import annotations

import structlog

from app.eval.scorer import EvalResult, QuestionResult, compute_scores
from app.generation.answerer import Answerer

logger = structlog.get_logger().bind(component="eval_runner")


class EvalRunner:
    """Runs a golden question set through the Answerer and computes scores."""

    def __init__(self, answerer: Answerer, golden_set: list[dict]) -> None:
        """Initialize the runner.

        Args:
            answerer: The RAG pipeline orchestrator to evaluate.
            golden_set: List of dicts, each with keys ``question``,
                ``expected_keywords``, ``ticker``, and ``section``.
        """
        self.answerer = answerer
        self.golden_set = golden_set

    async def run(self) -> EvalResult:
        """Evaluate every question in the golden set and aggregate the scores.

        Returns:
            EvalResult with per-question outcomes and aggregate metrics.
        """
        results: list[QuestionResult] = []

        for item in self.golden_set:
            question = item["question"]
            expected_keywords = item.get("expected_keywords", [])
            ticker = item.get("ticker", "UNKNOWN")
            section = item.get("section", "Unknown")

            answer = await self.answerer.answer(question)

            answer_lower = answer.text.lower()
            keyword_hit = any(
                keyword.lower() in answer_lower for keyword in expected_keywords
            )

            question_result = QuestionResult(
                question=question,
                answer_text=answer.text,
                citations_count=len(answer.citations),
                confidence=answer.confidence,
                latency_ms=answer.latency_ms,
                keyword_hit=keyword_hit,
                ticker=ticker,
                section=section,
            )
            results.append(question_result)

            logger.info(
                "eval_question_complete",
                ticker=ticker,
                section=section,
                keyword_hit=keyword_hit,
                confidence=answer.confidence,
                citations_count=len(answer.citations),
                latency_ms=answer.latency_ms,
            )

        return compute_scores(results)
