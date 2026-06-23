"""System prompt, user template, and answer schema. See DECISIONS.md D5."""

from __future__ import annotations

# RAG system prompt defining the assistant's role and constraints
RAG_SYSTEM_PROMPT = """You are a financial analyst assistant specializing in SEC filings.
Answer questions using ONLY the provided context from SEC 10-K and 10-Q filings.
If the context does not contain enough information to answer the question, say so explicitly.
Always cite the specific filing section and page number when making claims.
Be precise with numbers, dates, and financial figures."""

# User prompt template for RAG queries
RAG_USER_PROMPT_TEMPLATE = """Context from SEC filings:
{context}

Question: {question}

Provide a clear, accurate answer based only on the context above.
Include specific citations (ticker, form type, section, page) for key claims."""

# Prompt for extracting structured citations from generated answers
CITATION_EXTRACTION_PROMPT = """Extract citations from this answer as a JSON array.
Each citation should have: ticker, form_type, section, page_number, quote (max 50 chars).
Return ONLY valid JSON, no explanation."""


def format_rag_prompt(question: str, chunks: list[dict]) -> tuple[str, str]:
    """Format system + user prompts for RAG answering.
    
    Args:
        question: User's question.
        chunks: List of dicts with keys: text, section, page_number, ticker, form_type.
    
    Returns:
        (system_prompt, user_prompt) tuple ready for LLMClient.complete().
    """
    # Format each chunk with source metadata
    context_blocks = []
    for chunk in chunks:
        ticker = chunk.get("ticker", "UNKNOWN")
        form_type = chunk.get("form_type", "UNKNOWN")
        section = chunk.get("section", "Unknown Section")
        page_number = chunk.get("page_number", 0)
        text = chunk.get("text", "")
        
        context_block = (
            f"[Source: {ticker} {form_type}, Section: {section}, Page {page_number}]\n"
            f"{text}"
        )
        context_blocks.append(context_block)
    
    # Join all context blocks with double newlines
    context = "\n\n".join(context_blocks)
    
    # Format user prompt with context and question
    user_prompt = RAG_USER_PROMPT_TEMPLATE.format(context=context, question=question)
    
    return (RAG_SYSTEM_PROMPT, user_prompt)
