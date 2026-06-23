"""Tests for LLM client and prompt formatting."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest
from anthropic.types import TextBlock

from app.generation.llm_client import LLMClient
from app.generation.prompts import format_rag_prompt


@pytest.mark.asyncio
@patch("app.generation.llm_client.anthropic.AsyncAnthropic")
async def test_llm_client_returns_string(mock_anthropic_class: MagicMock) -> None:
    """LLMClient.complete() should return a string response."""
    # Mock Anthropic client and response
    mock_client = AsyncMock()
    mock_anthropic_class.return_value = mock_client
    
    # Create mock response matching Anthropic's structure
    mock_response = MagicMock()
    mock_response.content = [TextBlock(type="text", text="This is a test response")]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    mock_client.messages.create.return_value = mock_response
    
    # Create LLM client and call complete
    client = LLMClient(
        anthropic_api_key="test-anthropic-key",
        openai_api_key="test-openai-key",
    )
    
    result = await client.complete(
        system_prompt="You are a helpful assistant",
        user_prompt="Hello",
    )
    
    assert isinstance(result, str)
    assert result == "This is a test response"
    assert mock_client.messages.create.called


@pytest.mark.asyncio
@patch("app.generation.llm_client.openai.AsyncOpenAI")
@patch("app.generation.llm_client.anthropic.AsyncAnthropic")
async def test_llm_client_falls_back_to_openai(
    mock_anthropic_class: MagicMock, mock_openai_class: MagicMock
) -> None:
    """LLMClient should fall back to OpenAI when Anthropic raises APIError."""
    # Mock Anthropic client to raise APIError
    mock_anthropic_client = AsyncMock()
    mock_anthropic_class.return_value = mock_anthropic_client
    
    # Create a mock request for the APIError
    mock_request = MagicMock()
    mock_anthropic_client.messages.create.side_effect = anthropic.APIError(
        "API failed", request=mock_request, body=None
    )
    
    # Mock OpenAI client to succeed
    mock_openai_client = AsyncMock()
    mock_openai_class.return_value = mock_openai_client
    
    mock_openai_response = MagicMock()
    mock_openai_response.choices = [
        MagicMock(message=MagicMock(content="OpenAI fallback response"))
    ]
    mock_openai_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    mock_openai_client.chat.completions.create.return_value = mock_openai_response
    
    # Create LLM client and call complete
    client = LLMClient(
        anthropic_api_key="test-anthropic-key",
        openai_api_key="test-openai-key",
    )
    
    result = await client.complete(
        system_prompt="You are a helpful assistant",
        user_prompt="Hello",
    )
    
    # Should return OpenAI's response after Anthropic failed
    assert result == "OpenAI fallback response"
    assert mock_openai_client.chat.completions.create.called


@pytest.mark.asyncio
@patch("app.generation.llm_client.anthropic.AsyncAnthropic")
async def test_llm_client_complete_with_fallback_returns_provider(
    mock_anthropic_class: MagicMock,
) -> None:
    """LLMClient.complete_with_fallback() should return (provider, response) tuple."""
    # Mock Anthropic client
    mock_client = AsyncMock()
    mock_anthropic_class.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.content = [TextBlock(type="text", text="Anthropic response")]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    mock_client.messages.create.return_value = mock_response
    
    # Create LLM client and call complete_with_fallback
    client = LLMClient(
        anthropic_api_key="test-anthropic-key",
        openai_api_key="test-openai-key",
    )
    
    provider, result = await client.complete_with_fallback(
        system_prompt="You are a helpful assistant",
        user_prompt="Hello",
    )
    
    # Should return tuple with provider name
    assert provider == "anthropic"
    assert result == "Anthropic response"
    assert isinstance(provider, str)
    assert isinstance(result, str)


def test_format_rag_prompt_includes_context() -> None:
    """format_rag_prompt should include chunk text in the formatted prompt."""
    chunks = [
        {
            "text": "Apple Inc. reported revenue of $123.4 billion.",
            "ticker": "AAPL",
            "form_type": "10-K",
            "section": "Financial Results",
            "page_number": 42,
        },
        {
            "text": "Operating expenses increased by 15%.",
            "ticker": "AAPL",
            "form_type": "10-K",
            "section": "Management Discussion",
            "page_number": 58,
        },
    ]
    
    question = "What was Apple's revenue?"
    system_prompt, user_prompt = format_rag_prompt(question, chunks)
    
    # Check that chunk text appears in user prompt
    assert "Apple Inc. reported revenue of $123.4 billion." in user_prompt
    assert "Operating expenses increased by 15%." in user_prompt
    
    # Check that metadata appears
    assert "AAPL" in user_prompt
    assert "10-K" in user_prompt
    assert "Financial Results" in user_prompt
    assert "Page 42" in user_prompt


def test_format_rag_prompt_includes_question() -> None:
    """format_rag_prompt should include the question in the user prompt."""
    chunks = [
        {
            "text": "Revenue was $100M.",
            "ticker": "TEST",
            "form_type": "10-Q",
            "section": "Results",
            "page_number": 1,
        }
    ]
    
    question = "What was the company's revenue in Q3?"
    system_prompt, user_prompt = format_rag_prompt(question, chunks)
    
    # Question should appear in user prompt
    assert question in user_prompt
    
    # System prompt should define the assistant's role
    assert "financial analyst" in system_prompt.lower()
    assert "SEC filings" in system_prompt
