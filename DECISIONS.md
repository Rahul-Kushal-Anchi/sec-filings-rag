# Design Decisions

This document records the main technical decisions behind `sec-filings-rag`, including their rationale, trade-offs, measured results, and current limitations.

---

## D1: Why PostgreSQL and pgvector instead of Pinecone, Weaviate, or FAISS?

**Decision:** Use PostgreSQL with the pgvector extension.

**Reasoning:**

* Filing metadata and embedding vectors remain in one datastore, avoiding synchronization between a relational database and a separate vector database.
* PostgreSQL supports transactional ingestion, relational filtering, migrations, and vector search within the same operational system.
* pgvector provides HNSW indexing with cosine-distance search, which is appropriate for the current corpus of 81,864 chunks from 125 filings across 25 tickers.
* SQL filters can restrict retrieval by ticker, filing type, filing date, or other metadata before or alongside vector search.
* The project only requires one persistence layer to operate and back up.

**Trade-off:** A dedicated vector search system may provide better horizontal scaling, distributed indexing, and operational tooling at substantially larger corpus sizes.

**At much larger scale:** Evaluate systems such as Qdrant, Vespa, Weaviate, or a managed vector service after benchmarking PostgreSQL under the expected workload.

---

## D2: Chunk Size, Overlap, and Recursive Splitting

**Decision:** Use an 800-character target chunk size with 150 characters of overlap and recursive separators.

SEC filings contain dense financial prose, legal disclosures, converted tables, and long narrative sections. An 800-character target is large enough to retain useful financial context while remaining focused enough for retrieval and reranking.

The 150-character overlap carries context across adjacent chunks without excessively duplicating content in the vector store.

Instead of splitting at fixed character positions, the chunker attempts progressively smaller semantic boundaries:

1. Section or paragraph boundaries
2. Line breaks
3. Sentence boundaries
4. Word boundaries
5. Raw character boundaries as a final fallback

**Trade-off:** Smaller chunks can improve retrieval precision but may lose surrounding financial context. Larger chunks preserve context but increase embedding, reranking, and generation costs while potentially reducing retrieval precision.

**Current corpus:**

| Metric            |          Value |
| ----------------- | -------------: |
| Chunks            |         81,864 |
| Filings           |            125 |
| Tickers           |             25 |
| Target chunk size | 800 characters |
| Chunk overlap     | 150 characters |

---

## D3: Hybrid Retrieval Scoring

**Decision:** Combine dense and BM25 rankings using Reciprocal Rank Fusion with `k = 60`.

The query pipeline retrieves:

* Top 20 candidates from dense pgvector search
* Top 20 candidates from BM25 search
* Top 20 combined candidates after Reciprocal Rank Fusion

For a document (d), its fused score is:

```text
RRF(d) = Σ 1 / (k + rank(d))
```

The implementation uses:

```text
k = 60
```

**Why RRF instead of directly combining scores:**

Dense cosine similarity and BM25 use different scoring scales. A weighted sum would first require normalization or calibration, and those normalization choices could behave differently across query types.

RRF uses rank position rather than raw score, which:

* Avoids score-scale mismatch
* Requires little parameter tuning
* Allows strong results from either retrieval method to remain competitive
* Produces deterministic and explainable fusion behavior

Dense retrieval helps when the question is semantically related but worded differently from the filing. BM25 helps when the question contains exact accounting terms, product names, numerical labels, or company-specific language.

**Trade-off:** Standard RRF gives the ranked lists equal structural importance. A future evaluation could compare standard RRF with weighted RRF or a learned fusion model using the golden evaluation set.

---

## D4: Why Use a Cross-Encoder Reranker?

**Decision:** Apply `cross-encoder/ms-marco-MiniLM-L-6-v2` to the fused candidate set and pass the top five passages to answer generation.

**Reasoning:**

Dense embedding models encode the question and document independently. This makes large-scale retrieval efficient, but it can miss fine-grained relationships between a specific question and a candidate passage.

A cross-encoder evaluates the question and passage together, allowing it to model token-level interactions and produce a more precise relevance score.

The two-stage design balances efficiency and relevance:

1. Dense and BM25 retrieval rapidly reduce the corpus to a small candidate set.
2. RRF combines both rankings.
3. The cross-encoder reranks the fused candidates.
4. Only the top five passages enter the LLM context.

**Measured system result:**

* Current end-to-end query latency is approximately 15–18 seconds.

This is the complete request latency and includes query embedding, retrieval, fusion, reranking, generation, and network time.

**Measurement limitation:** Reranker-only latency and Recall@10 with and without reranking have not yet been isolated in a controlled benchmark. The project should not claim a reranking improvement percentage until that experiment is run.

**Next evaluation:**

* Measure reranker latency separately
* Compare retrieval Recall@5 and Recall@10 before and after reranking
* Record relevance changes on the same golden questions
* Profile each pipeline stage independently

---

## D5: Why Structured Outputs Instead of Free-Form Generation?

**Decision:** Represent answers and citations through typed Pydantic response schemas.

The response contract includes answer text, structured citations, confidence information, and source metadata.

**Reasoning:**

* The frontend needs stable citation fields to render clickable and inspectable source cards.
* Typed responses make API behavior easier to test.
* Validation catches malformed output before it reaches the client.
* Filing metadata can be returned independently from generated prose.
* Downstream code does not need to parse informal citation markers such as `[1]` or `[2]`.

**Trade-off:** Structured generation adds schema constraints and error-handling complexity, but it provides a more reliable application interface than unrestricted text.

---

## D6: How Is Grounding and Hallucination Risk Evaluated?

**Decision:** Use a filing-derived golden question set, structured citations, expected-answer keywords, confidence measurements, and planned manual validation.

The current evaluation harness contains 20 golden questions across AAPL and JPM filings. Expected keywords are derived from the actual filing content rather than generated independently.

The evaluation records:

* Expected keyword coverage
* Answer confidence
* End-to-end latency
* Citation presence
* Provider usage
* Whether the answer is supported by retrieved filing passages

**Current observed results:**

| Query category      | Observed confidence |
| ------------------- | ------------------: |
| Financial queries   |               0.997 |
| Risk-factor queries |               0.862 |

These confidence values are system outputs and are not equivalent to a measured hallucination rate.

**Grounding controls:**

* The generation prompt instructs the model to answer only from retrieved filing context.
* Answers include structured filing citations.
* Retrieved passages retain ticker, form type, filing date, section, page, and source snippet metadata.
* The model is instructed to state when the supplied context is insufficient.
* Golden answers are tied to content present in the underlying filings.

**Honest limitation:** The project does not currently have enough evidence to claim that hallucination decreased from a specific percentage to another specific percentage. A defensible hallucination-rate claim would require a larger labeled dataset, a written annotation rubric, blinded human review, and inter-rater agreement or a separately validated judge model.

**Future evaluation:**

* Expand the golden set beyond 20 questions
* Add answer-level faithfulness scoring
* Manually review a statistically meaningful sample
* Separate retrieval failures from generation failures
* Report confidence calibration rather than treating confidence as correctness

---

## D7: Why Stream Responses?

**Decision:** Stream answer tokens from FastAPI to the React client using Server-Sent Events.

**Reasoning:** The current pipeline has an end-to-end latency of approximately 15–18 seconds. Streaming reduces perceived waiting time by allowing the user to begin reading before the complete answer is available.

**Trade-off:** Streaming improves perceived latency but does not reduce total computation time. Errors, citations, and final metadata must also be handled correctly when the response is delivered incrementally.

---

## D8: What Did Not Work, and What Would Be Improved Next?

### Oversized Docker build context

The initial backend Docker build sent approximately 952.92 MB of build context because the local virtual environment and development caches were not excluded.

The primary contributors were:

* `backend/.venv`
* `backend/.mypy_cache`
* Python and testing caches

Adding `backend/.dockerignore` reduced the transferred build context to approximately 2 KB during the verified build.

**Lesson:** Docker build context must be treated as part of the production build design. Local development artifacts should never be copied into the image or uploaded to a remote builder.

### Documentation duplicated repository overview content

The original architecture document largely duplicated the README and retained stale repository references.

It was replaced with a dedicated architecture document describing ingestion, retrieval, fusion, reranking, generation, citations, and system trade-offs.

**Lesson:** The README should explain how to understand and run the project, while architecture documentation should explain how and why the system works.

### End-to-end latency remains high

The measured 15–18 second response time is acceptable for a portfolio demonstration but is too slow for a polished production chat experience.

**Next steps:**

* Measure query embedding, retrieval, reranking, and generation separately
* Cache repeated query embeddings where appropriate
* Benchmark faster generation models
* Reduce unnecessary context tokens
* Evaluate reranker batch size and candidate count
* Retain streaming for perceived responsiveness

### BM25 is currently maintained in memory

The in-memory BM25 index is simple and suitable for the current project, but it introduces rebuild and synchronization concerns as the corpus grows.

**At larger scale:** Persist the sparse index or move lexical retrieval to a system designed for durable distributed search.

---

## D9: Production Concerns

### Cost

The estimated cost of embedding the complete current corpus is approximately **$3–8**.

This is an engineering estimate based on corpus size and embedding usage, not a value reconciled against a provider billing statement.

Per-query embedding and generation costs have not yet been measured separately. Production reporting should capture:

* Embedding tokens
* Input and output generation tokens
* Provider and model used
* Retry attempts
* Cost per successful query
* Cost per failed query

### Corpus scale

The current indexed corpus contains:

* 81,864 chunks
* 125 SEC filings
* 25 tickers

These values should be reported separately from future target scale.

### Privacy and safety

SEC filings are public documents, so the current application does not process private internal company documents.

A deployment over private documents would require:

* Access control
* Tenant isolation
* Encryption and key management
* Audit logging
* Data-retention policies
* Protection against sensitive data appearing in traces or prompts

### Online evaluation

A production system should sample a controlled portion of real queries for offline evaluation while excluding sensitive content.

Evaluation should monitor:

* Retrieval relevance
* Unsupported-answer rate
* Citation validity
* Empty-result frequency
* Latency percentiles
* Provider fallback frequency
* User feedback

### Index updates

The ingestion design supports adding or updating filings without replacing the entire database. However, complete index rebuild duration has not yet been benchmarked and should not be reported as a measured value.

Production ingestion should be idempotent and track filing accession numbers, chunk versions, ingestion status, and update timestamps.

### Reliability

Production deployment should include:

* Provider timeouts and retry limits
* LLM fallback behavior
* Database connection-pool monitoring
* Health and readiness checks
* Rate limiting
* Structured logs
* Prometheus metrics
* Alerting on latency, error rate, and failed ingestion jobs
