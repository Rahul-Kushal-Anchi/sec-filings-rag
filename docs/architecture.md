# Architecture

This document describes the architecture of `sec-filings-rag`, a retrieval-augmented generation system for querying SEC 10-K and 10-Q filings with source-grounded answers.

## System Overview

The application consists of five primary layers:

1. Filing ingestion and normalization
2. Chunking and embedding generation
3. Hybrid document retrieval
4. Answer generation with structured citations
5. Evaluation and observability

## Ingestion Pipeline

### 1. Filing acquisition

SEC 10-K and 10-Q filings are downloaded from EDGAR and stored with identifying metadata including ticker, company name, form type, filing date, accession number, and source URL.

### 2. Filing parsing

The parser handles SEC submission files that may contain multiple documents in a single SGML container. The parsing process separates embedded filing documents, selects the primary document, excludes XBRL exhibits, normalizes HTML content, and preserves section metadata for citation generation.

### 3. Chunking

| Setting | Value |
|---|---|
| Target chunk size | 800 characters |
| Chunk overlap | 150 characters |
| Strategy | Recursive separator hierarchy |
| Metadata | Filing, section, page, ticker and date |

### 4. Embedding and storage

Each chunk is embedded using `text-embedding-3-small`. Embeddings and metadata are stored in PostgreSQL with the pgvector extension. An HNSW index supports approximate nearest-neighbor search using cosine distance.

## Query Pipeline

### 1. Dense retrieval

Searches the pgvector HNSW index for semantically similar filing chunks (top_k = 20).

### 2. Sparse retrieval

BM25 retrieval searches for lexical overlap (top_k = 20). Useful for exact financial terminology and numeric queries.

### 3. Reciprocal Rank Fusion

RRF combines dense and BM25 rankings using rank position rather than raw scores, where k = 60.

### 4. Cross-encoder reranking

Fused candidates are reranked using cross-encoder/ms-marco-MiniLM-L-6-v2. The top five passages are passed to the generation layer.

### 5. Answer generation

The primary provider is Anthropic Claude Sonnet 4.6, with OpenAI GPT-4o-mini configured as a fallback.

## Citation Model

Each source includes the ticker, form type, filing date, section, page and supporting snippet.

## Design Trade-offs

**Hybrid retrieval over dense-only:** SEC filings contain both semantic prose and exact accounting terminology.

**RRF over raw-score merging:** Dense similarity and BM25 scores are not directly comparable.

**PostgreSQL and pgvector:** Metadata and embeddings remain in the same persistence layer.

**Structured citations:** Returned as structured API data rather than embedded only in generated prose.
