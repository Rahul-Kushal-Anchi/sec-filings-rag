"""Tests for the /v1/query endpoint: synchronous JSON and SSE streaming."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.api.schemas import Answer, Citation
from app.deps import get_answerer
from app.main import create_app


def make_answer(
    text: str = "Apple faces supply chain risks.",
    citations: list[Citation] | None = None,
    confidence: float = 0.9,
) -> Answer:
    """Build a synthetic Answer for mocking the Answerer."""
    return Answer(
        text=text,
        citations=citations if citations is not None else [],
        confidence=confidence,
        latency_ms=42,
    )


@pytest.fixture
def mock_answerer() -> MagicMock:
    """Mock Answerer whose answer() returns a deterministic Answer."""
    answerer = MagicMock()
    answerer.answer = AsyncMock(return_value=make_answer())
    return answerer


@pytest_asyncio.fixture
async def app_client(mock_answerer: MagicMock) -> AsyncIterator[AsyncClient]:
    """HTTP client backed by the app with the Answerer dependency overridden."""
    # Disable prometheus middleware in tests to avoid _IncludedRouter conflicts
    # on newer prometheus-fastapi-instrumentator versions (CI).
    with (
        patch(
            "prometheus_fastapi_instrumentator.Instrumentator.instrument",
            return_value=MagicMock(),
        ),
        patch(
            "prometheus_fastapi_instrumentator.Instrumentator.expose",
            return_value=MagicMock(),
        ),
    ):
        app = create_app()

    app.dependency_overrides[get_answerer] = lambda: mock_answerer
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_query_endpoint_returns_json(app_client: AsyncClient) -> None:
    """POST /v1/query with stream=False should return a 200 JSON Answer."""
    response = await app_client.post(
        "/v1/query",
        json={"question": "What are Apple's risks?"},
    )

    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]

    data = response.json()
    assert data["text"] == "Apple faces supply chain risks."
    assert data["confidence"] == 0.9
    assert "citations" in data
    assert "latency_ms" in data


@pytest.mark.asyncio
async def test_query_endpoint_streams_events(
    app_client: AsyncClient, mock_answerer: MagicMock
) -> None:
    """POST /v1/query with stream=True should return SSE chunk + done events."""
    citation = Citation(
        filing_ticker="AAPL",
        filing_form_type="10-K",
        filing_date=date(2024, 1, 1),
        page_number=12,
        snippet="Supply chain risks remain.",
        score=0.88,
    )
    mock_answerer.answer = AsyncMock(
        return_value=make_answer(citations=[citation])
    )

    response = await app_client.post(
        "/v1/query?stream=true",
        json={"question": "What are Apple's risks?"},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    body = response.text
    # Should contain per-token chunk events and a terminal done event.
    assert '"type": "chunk"' in body
    assert '"type": "done"' in body

    # Parse the SSE data lines and confirm the done event carries metadata.
    data_payloads = [
        json.loads(line[len("data:"):].strip())
        for line in body.splitlines()
        if line.startswith("data:")
    ]
    event_types = [payload["type"] for payload in data_payloads]
    assert "chunk" in event_types
    assert "done" in event_types

    done_event = next(p for p in data_payloads if p["type"] == "done")
    assert done_event["confidence"] == 0.9
    assert done_event["latency_ms"] == 42
    assert len(done_event["citations"]) == 1


@pytest.mark.asyncio
async def test_query_endpoint_handles_empty_corpus(
    app_client: AsyncClient, mock_answerer: MagicMock
) -> None:
    """Endpoint should gracefully return a no-info answer when no chunks match."""
    mock_answerer.answer = AsyncMock(
        return_value=make_answer(
            text="No relevant information found.",
            confidence=0.0,
        )
    )

    response = await app_client.post(
        "/v1/query",
        json={"question": "Some utterly obscure question"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "No relevant information found."
    assert data["confidence"] == 0.0
    assert data["citations"] == []
