"""Scoring functions and result dataclasses for the RAG evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QuestionResult:
    """Outcome of evaluating a single golden-set question."""

    question: str
    answer_text: str
    citations_count: int
    confidence: float
    latency_ms: int
    keyword_hit: bool
    ticker: str
    section: str


@dataclass
class EvalResult:
    """Aggregate metrics across an entire evaluation run."""

    total_questions: int
    keyword_hit_rate: float
    avg_confidence: float
    avg_latency_ms: float
    avg_citations: float
    results: list[QuestionResult] = field(default_factory=list)


def compute_scores(results: list[QuestionResult]) -> EvalResult:
    """Compute aggregate metrics from per-question results.

    Args:
        results: Per-question evaluation outcomes.

    Returns:
        EvalResult with aggregate metrics. All averages are 0.0 when ``results``
        is empty (avoids division by zero).
    """
    total = len(results)
    if total == 0:
        return EvalResult(
            total_questions=0,
            keyword_hit_rate=0.0,
            avg_confidence=0.0,
            avg_latency_ms=0.0,
            avg_citations=0.0,
            results=[],
        )

    keyword_hits = sum(1 for r in results if r.keyword_hit)

    return EvalResult(
        total_questions=total,
        keyword_hit_rate=keyword_hits / total,
        avg_confidence=sum(r.confidence for r in results) / total,
        avg_latency_ms=sum(r.latency_ms for r in results) / total,
        avg_citations=sum(r.citations_count for r in results) / total,
        results=results,
    )


def format_report(eval_result: EvalResult) -> str:
    """Render an EvalResult as a human-readable markdown report.

    Args:
        eval_result: Aggregate evaluation metrics.

    Returns:
        Markdown-formatted report string.
    """
    lines = [
        "# RAG Evaluation Report",
        "",
        "## Summary",
        "",
        f"- **Total questions:** {eval_result.total_questions}",
        f"- **Keyword hit rate:** {eval_result.keyword_hit_rate:.1%}",
        f"- **Average confidence:** {eval_result.avg_confidence:.3f}",
        f"- **Average latency (ms):** {eval_result.avg_latency_ms:.1f}",
        f"- **Average citations:** {eval_result.avg_citations:.2f}",
        "",
        "## Per-question results",
        "",
        "| # | Ticker | Section | Hit | Confidence | Citations | Latency (ms) | Question |",
        "|---|--------|---------|-----|------------|-----------|--------------|----------|",
    ]

    for idx, result in enumerate(eval_result.results, start=1):
        hit_mark = "✅" if result.keyword_hit else "❌"
        question_preview = (
            result.question if len(result.question) <= 60 else f"{result.question[:60]}…"
        )
        lines.append(
            f"| {idx} | {result.ticker} | {result.section} | {hit_mark} | "
            f"{result.confidence:.3f} | {result.citations_count} | "
            f"{result.latency_ms} | {question_preview} |"
        )

    return "\n".join(lines)
