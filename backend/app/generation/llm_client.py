"""Unified async LLM client wrapping Anthropic (default) and OpenAI (fallback)."""

from __future__ import annotations

import time

import anthropic
import openai
import structlog
from anthropic.types import TextBlock
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger().bind(component="llm_client")


class LLMClient:
    """Async LLM client with Anthropic primary and OpenAI fallback.
    
    Provides unified interface for chat completion across providers, with automatic
    retry logic and fallback behavior. All calls are logged for observability.
    """

    def __init__(
        self,
        anthropic_api_key: str,
        openai_api_key: str,
        anthropic_model: str = "claude-sonnet-4-6",
        openai_model: str = "gpt-4o-mini",
    ) -> None:
        """Initialize LLM client with API keys and model names.
        
        Args:
            anthropic_api_key: Anthropic API key.
            openai_api_key: OpenAI API key.
            anthropic_model: Anthropic model identifier (default: claude-sonnet-4-6).
            openai_model: OpenAI model identifier (default: gpt-4o-mini).
        """
        self.anthropic_client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.anthropic_model = anthropic_model
        self.openai_model = openai_model

    @retry(
        retry=retry_if_exception_type((anthropic.APIError, anthropic.RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
    )
    async def _call_anthropic(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> str:
        """Call Anthropic API with retry logic.
        
        Args:
            system_prompt: System prompt/instructions.
            user_prompt: User message.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Response text content.
            
        Raises:
            anthropic.APIError: On API failures after retries.
            anthropic.RateLimitError: On rate limit after retries.
        """
        start_time = time.time()
        
        response = await self.anthropic_client.messages.create(
            model=self.anthropic_model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            "llm_completion_success",
            provider="anthropic",
            model=self.anthropic_model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
        )
        
        # Extract text from response
        block = response.content[0]
        text_content = block.text if isinstance(block, TextBlock) else ""
        return text_content

    @retry(
        retry=retry_if_exception_type((openai.APIError, openai.RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
    )
    async def _call_openai(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> str:
        """Call OpenAI API with retry logic.
        
        Args:
            system_prompt: System prompt/instructions.
            user_prompt: User message.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Response text content.
            
        Raises:
            openai.APIError: On API failures after retries.
            openai.RateLimitError: On rate limit after retries.
        """
        start_time = time.time()
        
        response = await self.openai_client.chat.completions.create(
            model=self.openai_model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            "llm_completion_success",
            provider="openai",
            model=self.openai_model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
            latency_ms=latency_ms,
        )
        
        # Extract text from response
        text_content = response.choices[0].message.content or ""
        return text_content

    async def complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 1024
    ) -> str:
        """Complete a prompt using Anthropic (primary) with OpenAI fallback.
        
        Args:
            system_prompt: System prompt/instructions.
            user_prompt: User message.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Response text content.
        """
        try:
            return await self._call_anthropic(system_prompt, user_prompt, max_tokens)
        except (anthropic.APIError, anthropic.RateLimitError, RetryError) as e:
            logger.warning(
                "anthropic_failed_falling_back_to_openai",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            return await self._call_openai(system_prompt, user_prompt, max_tokens)

    async def complete_with_fallback(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 1024
    ) -> tuple[str, str]:
        """Complete a prompt and return which provider was used.
        
        Args:
            system_prompt: System prompt/instructions.
            user_prompt: User message.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Tuple of (provider_name, response_text) where provider_name is
            "anthropic" or "openai".
        """
        try:
            response = await self._call_anthropic(system_prompt, user_prompt, max_tokens)
            return ("anthropic", response)
        except (anthropic.APIError, anthropic.RateLimitError, RetryError) as e:
            logger.warning(
                "anthropic_failed_falling_back_to_openai",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            response = await self._call_openai(system_prompt, user_prompt, max_tokens)
            return ("openai", response)
