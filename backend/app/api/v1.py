"""Version 1 HTTP routes."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Query, status
from langsmith import traceable
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import (
    Answer,
    EvalReport,
    FilingResponse,
    QueryRequest,
)
from app.deps import AnswererDep
from app.generation.answerer import Answerer
from app.logging_config import get_logger

router = APIRouter(tags=["v1"])
logger = get_logger("app.api.v1")


@traceable(name="rag_query")
async def _generate_answer(
    answerer: Answerer, question: str, max_results: int
) -> Answer:
    """Run the RAG pipeline, wrapped in a LangSmith trace for observability."""
    return await answerer.answer(question, max_results=max_results)


async def _stream_answer(
    answerer: Answerer, payload: QueryRequest
) -> AsyncIterator[dict[str, str]]:
    """Yield SSE events: per-token chunks, a terminal done event, or an error event."""
    try:
        answer = await _generate_answer(
            answerer, payload.question, payload.max_results
        )

        # Stream the answer text token-by-token (whitespace-delimited).
        for token in answer.text.split(" "):
            yield {"data": json.dumps({"type": "chunk", "text": token + " "})}

        yield {
            "data": json.dumps(
                {
                    "type": "done",
                    "citations": [
                        citation.model_dump(mode="json")
                        for citation in answer.citations
                    ],
                    "confidence": answer.confidence,
                    "latency_ms": answer.latency_ms,
                }
            )
        }
    except Exception as exc:  # noqa: BLE001 - surface any failure as an SSE error event
        logger.error("query_stream_failed", error=str(exc))
        yield {"data": json.dumps({"type": "error", "message": str(exc)})}


@router.post(
    "/query",
    response_model=None,
    summary="Ask a question against the SEC filings corpus.",
    description=(
        "Returns a cited answer from the RAG pipeline. Pass ?stream=true to "
        "receive token-by-token Server-Sent Events instead of a single JSON body."
    ),
)
async def query(
    payload: QueryRequest,
    answerer: AnswererDep,
    stream: bool = Query(default=False),
) -> Answer | EventSourceResponse:
    """Retrieve, rerank, and generate an answer with citations.

    When ``stream`` is false, returns a single :class:`Answer` JSON body. When
    true, returns an SSE stream of ``chunk`` events followed by a ``done`` event.
    """
    if stream:
        return EventSourceResponse(_stream_answer(answerer, payload))

    return await _generate_answer(answerer, payload.question, payload.max_results)


@router.get(
    "/filings",
    response_model=list[FilingResponse],
    summary="List ingested filings.",
)
async def list_filings(
    ticker: str | None = Query(default=None),
    form_type: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[FilingResponse]:
    """Return ingested filings filtered by ticker/form_type with pagination."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="filings listing not implemented yet",
    )


@router.get(
    "/eval/runs",
    response_model=list[EvalReport],
    summary="List recent evaluation runs.",
)
async def list_eval_runs() -> list[EvalReport]:
    """Return summaries of previous offline evaluation runs."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="eval reporting not implemented yet",
    )
