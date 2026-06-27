# Backlog

Known limitations and future improvements. These inform what to improve next during technical review.

---

## Retrieval Improvements

- HyDE (Hypothetical Document Embeddings) — generate a hypothetical answer, embed it, and retrieve against that vector
- Query rewriting and multi-query retrieval — decompose complex questions into sub-queries
- A/B test different embedding models — compare text-embedding-3-small against text-embedding-3-large and open-source alternatives
- Tune RRF parameters or evaluate weighted RRF using a dev set
- Add bge-reranker-base as an alternative cross-encoder
- Pre-filter by metadata such as ticker, filing date and form type before vector search

---

## Generation Improvements

- Self-consistency — generate multiple answers and vote on the most supported response
- Chain-of-verification — check whether each claim is supported by the retrieved passages
- Cost-based model routing — use a smaller mple factual queries and a stronger model for complex multi-part questions
- Context compression — summarize retrieved passages before sending them to the generation model to reduce token usage

---

## Evaluation Improvements

- Expand the golden question set beyond 20 questions across more tickers
- Add per-section accuracy breakdown
- Add faithfulness and comprehensiveness metrics
- Sample a controlled percentage of real queries for offline review
- Separate retrieval failure rate from generation failure rate

---

## Production and Scale

- Cache embeddings for repeated queries
- Add rate limiting per user or API key
- Move BM25 to OpenSearch or Elasticsearch for durable distributed lexical search
- Add a background re-indexing pipeline for new filings
- Add an async ingestion queue using Celery and Redis
- Evaluate sharded pgvector for corpora exceeding tens of millions of chunks

---

## User Experience

- Multi-turn conversation with session memory
- Cross-company comparison queries
- Citation hovepreview showing full passage context
- Export answers as PDF with numbered endnotes

---

## Safety and Compliance

- PII detection on user queries before retrieval
- Audit logging for queries and answers
- Output safety filtering
- Source attribution display in the UI

---

## Known Parser Limitations

Section-name extraction on iXBRL filings can occasionally capture trailing words or truncate long titles.

Examples:

- "Risk Factors The" instead of "Risk Factors"
- "Market for Registrant" instead of the full section title

This is acceptable for the current corpus because retrieval is driven by chunk content rather than section labels. Section labels are used only for citation metadata.

A more robust solution would involve corpus-wide regex refinement or LLM-based section-header extraction on a per-filing basis.
