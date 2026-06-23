"""Tests for the evaluation harness scorer and result dataclasses."""

from __future__ import annotations

from app.eval.scorer import (
    EvalResult,
    QuestionResult,
    compute_scores,
    format_report,
)


def make_question_result(
    question: str = "What are the risks?",
    keyword_hit: bool = True,
    confidence: float = 0.9,
    citations_count: int = 3,
    latency_ms: int = 100,
    ticker: str = "AAPL",
    section: str = "Risk Factors",
) -> QuestionResult:
    """Build a synthetic QuestionResult for testing."""
    return QuestionResult(
        question=question,
        answer_text="A synthetic answer.",
        citations_count=citations_count,
        confidence=confidence,
        latency_ms=latency_ms,
        keyword_hit=keyword_hit,
        ticker=ticker,
        section=section,
    )


def test_eval_result_dataclass() -> None:
    """EvalResult should expose the expected aggregate fields."""
    result = EvalResult(
        total_questions=5,
        keyword_hit_rate=0.8,
        avg_confidence=0.75,
        avg_latency_ms=120.5,
        avg_citations=2.5,
        results=[],
    )

    assert result.total_questions == 5
    assert result.keyword_hit_rate == 0.8
    assert result.avg_confidence == 0.75
    assert result.avg_latency_ms == 120.5
    assert result.avg_citations == 2.5
    assert result.results == []


def test_question_result_dataclass() -> None:
    """QuestionResult should expose the expected per-question fields."""
    qr = make_question_result()

    assert qr.question == "What are the risks?"
    assert qr.answer_text == "A synthetic answer."
    assert qr.citations_count == 3
    assert qr.confidence == 0.9
    assert qr.latency_ms == 100
    assert qr.keyword_hit is True
    assert qr.ticker == "AAPL"
    assert qr.section == "Risk Factors"


def test_compute_scores_empty() -> None:
    """compute_scores should return all-zero metrics for empty input."""
    result = compute_scores([])

    assert result.total_questions == 0
    assert result.keyword_hit_rate == 0.0
    assert result.avg_confidence == 0.0
    assert result.avg_latency_ms == 0.0
    assert result.avg_citations == 0.0
    assert result.results == []


def test_compute_scores_calculates_correctly() -> None:
    """compute_scores should compute hit rate and averages over results."""
    results = [
        make_question_result(
            keyword_hit=True, confidence=0.9, citations_count=4, latency_ms=100
        ),
        make_question_result(
            keyword_hit=True, confidence=0.6, citations_count=2, latency_ms=200
        ),
        make_question_result(
            keyword_hit=False, confidence=0.3, citations_count=0, latency_ms=300
        ),
    ]

    eval_result = compute_scores(results)

    assert eval_result.total_questions == 3
    # 2 of 3 hits
    assert eval_result.keyword_hit_rate == 2 / 3
    # (0.9 + 0.6 + 0.3) / 3 = 0.6
    assert eval_result.avg_confidence == 0.6
    # (100 + 200 + 300) / 3 = 200
    assert eval_result.avg_latency_ms == 200.0
    # (4 + 2 + 0) / 3 = 2.0
    assert eval_result.avg_citations == 2.0
    assert len(eval_result.results) == 3


def test_format_report_contains_metrics() -> None:
    """format_report should produce markdown containing the key metrics."""
    results = [
        make_question_result(keyword_hit=True, confidence=0.9),
        make_question_result(keyword_hit=False, confidence=0.4),
    ]
    eval_result = compute_scores(results)
    report = format_report(eval_result)

    assert isinstance(report, str)
    assert "# RAG Evaluation Report" in report
    assert "Total questions" in report
    assert "Keyword hit rate" in report
    assert "Average confidence" in report
    assert "Average latency" in report
    assert "Average citations" in report
    # Per-question table header present
    assert "Ticker" in report
    assert "Section" in report
