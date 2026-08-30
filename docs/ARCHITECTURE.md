# TraceRAG Architecture

## Public Request Chain

```text
Client (any network)
  │  HTTPS
  ▼
Cloudflare Edge → Worker smillick-org (smillick.org/*)
  │  run_worker_first: Worker intercepts /api/v1/rag/* before static assets
  │
  ├─ GET/HEAD /api/v1/rag/health        public, no auth, proxies origin
  ├─ GET     /api/v1/rag/collections    Bearer auth + API rate limit (120/60s)
  ├─ GET     /api/v1/rag/stats          Bearer auth + API rate limit (120/60s)
  └─ POST    /api/v1/rag/ask            Bearer auth + Ask rate limit (10/60s) + 65 536-byte body cap
  │
  │  Worker rebuilds Authorization: Bearer $RAG_ORIGIN_API_KEY (never forwards client credentials)
  ▼
https://rag.smillick.org  (origin — clients must not connect directly)
  │
  │  Cloudflare Tunnel (cloudflared, outbound QUIC, no inbound ports)
  ▼
127.0.0.1:8000  FastAPI + Uvicorn
  │
  ▼
TraceRAG Pipeline (bge-m3 + Chroma + BM25 + bge-reranker + DeepSeek)
```

**Key design rule**: `rag.smillick.org` is an origin-only hostname. The Worker is the sole public entry point. Clients that bypass the Worker still hit FastAPI's `RAG_API_KEY` authentication, but lose the gateway-layer Bearer auth, rate limiting, unified error sanitization, request-id tracing, and security headers.

## Trust Boundaries

| Boundary | Mechanism |
|---|---|
| Client → Worker | Bearer token (`RAG_GATEWAY_API_KEY`) checked via constant-time comparison. Failed auth returns 401 before rate limiting or origin call. |
| Worker → Origin | Worker discards the client token and reconstructs `Authorization: Bearer $RAG_ORIGIN_API_KEY`. This value **must match** the FastAPI `.env` `RAG_API_KEY` exactly — otherwise the origin returns 401. The origin never sees client credentials. |
| Worker → Origin transport | Cloudflare Tunnel (outbound QUIC). No inbound firewall ports on the origin host. Uvicorn binds `127.0.0.1` only. |
| Origin app layer | `RAG_API_KEY` middleware on FastAPI validates the incoming origin key. `RAG_ORIGIN_API_KEY` (Wrangler secret) and `RAG_API_KEY` (.env) are always a matched pair. `RAG_GATEWAY_API_KEY` should be separate from both and independently rotatable. |

## Worker Gateway Behavior

### Authentication and Rate Limiting

1. **Authenticate first**: `proxyProtected` calls `authenticateGateway` before any rate-limiter call.
2. **Stable rate-limit key**: derived from `SHA-256(token:route)`, truncated to 32 hex chars. Non-secret — safe to log. Binds rate limits to the authenticated identity + route, not to IP alone.
3. **Two-tier limits**:
   - `RAG_API_RATE_LIMITER`: 120 requests / 60 s (collections, stats).
   - `RAG_ASK_RATE_LIMITER`: 10 requests / 60 s (ask).
4. **Fail closed**: if a limiter binding is missing or throws, the Worker returns 503 JSON `{"error":"rate_limiter_unavailable"}` with `Retry-After: 60`. It never falls through to the origin.

### Request Handling

- **X-Request-ID**: validated against `^[A-Za-z0-9_-]{1,128}$`. Malformed IDs are replaced with a fresh UUID. Every response (including errors) carries the validated ID.
- **Cache-Control: no-store** on every RAG response, including 405, 401, 413, 415, 429, 502, 503, and streamed 2xx.
- **Security headers**: CSP, X-Frame-Options DENY, X-Content-Type-Options nosniff, Referrer-Policy, COOP — applied to all responses including static assets.
- **Body limit**: `/ask` enforces 65 536-byte cap via `Content-Length` pre-check + streamed body metering. Oversized requests return 413 without calling the origin.
- **Content-Type**: `/ask` requires `application/json`; other media types return 415.
- **Method enforcement**: `/collections` and `/stats` accept GET only; `/ask` accepts POST only. Wrong methods return 405 with `Allow` header.
- **Upstream error sanitization**: non-2xx origin responses are discarded (body cancelled). The client receives a safe 502 JSON `{"error":"upstream_error"}` with the request ID — never the upstream body or status.
- **Streaming**: only successful (2xx) origin responses are streamed back to the client.

### Structured Logging

`logEvent` emits JSON lines with timestamp, level, event name, request ID, path, and relevant fields. No secrets, tokens, or API keys appear in logs.

## Pipeline

```text
User query
  -> code query detection (expand top_k)
  -> HyDE: generate hypothetical answer document
  -> vector search (hypothetical doc) + BM25 (raw query)
  -> weighted RRF fusion
  -> bge-reranker-v2-m3 rerank (raw query)
  -> code-aware neighbor chunk expansion
  -> DeepSeek generate grounded answer with citations
  -> JSONL call-chain log
```

## API Service

`server.py` wraps the pipeline with FastAPI. The service loads `RAGPipeline` once during application startup (via `lifespan`) and reuses the in-memory PDF chunks, BM25 index, vector store handle, reranker, and model client for each request.

Endpoints (origin paths, proxied by the Worker under `/api/v1/rag/`):

- `GET /health`: readiness check.
- `GET /stats`: loaded document, page, chunk, and model statistics.
- `GET /collections`: available PDF collections and source paths.
- `POST /ask`: question answering with citations, sources, and trace.
- `GET /project/progress`: milestones and loading status.
- `GET /project/code-files`: key source file list for code browser.
- `GET /project/code/{key}`: full source code of a key file.

The `/ask` response includes:

- `answer`: the generated answer string.
- `citations`: chunk number, source, page, and chunk_id for each cited passage.
- `sources`: full metadata (source_path, collection, page, chunk_id, chunk_index, char_start, char_end, preview).
- `trace`: HyDE status, RRF fusion scores, rerank scores, timing breakdown (hyde / retrieval / rerank / generation / total).

## Failure and Degradation

| Failure | Behavior |
|---|---|
| Missing/invalid client token | 401 JSON, no origin call, no rate-limiter call. |
| Rate limiter binding missing or throws | 503 JSON `rate_limiter_unavailable`, `Retry-After: 60`, no origin call. |
| Rate limit exceeded | 429 JSON `rate_limited`, `Retry-After: 60`. |
| Origin fetch throws (network) | 502 JSON `upstream_unavailable`. |
| Origin returns non-2xx | 502 JSON `upstream_error`; upstream body discarded and cancelled. |
| Vector search throws (origin-side) | BM25-only retrieval (pipeline-level, within FastAPI). |
| LLM API unavailable | Extractive fallback answer from retrieved passages. |
| Worker unhandled exception | 500 JSON `internal_error` with request ID; structured error log. |

Every error response includes `X-Request-ID`, `Cache-Control: no-store`, and the full security header set.

## Index Data Flow

```text
data/**/*.pdf
  │
  ▼
_pdf_inventory()                    sha256 + mtime fingerprint per file
  │
  ▼
load_or_update_chunks()             compare cached manifest fingerprints
  ├─ cache hit    → load chunks.jsonl as-is
  ├─ incompatible → full re-parse, re-chunk, re-cache
  └─ incremental  → re-parse changed sources, drop deleted, merge, re-cache
  │
  ▼
IndexUpdateResult
  chunks, changed_sources, deleted_sources, deleted_chunk_ids, full_rebuild
  │
  ▼
sync_vectorstore()
  ├─ full_rebuild=True        → wipe persist_dir, rebuild Chroma, write .embedding_meta.json
  ├─ no persist dir           → build fresh
  └─ incremental               → delete stale chunk_ids, add only changed-source chunks
  │
  ▼
_embedding_meta.json               model + dimension fingerprint for drift detection
  │
  ▼
Chroma persist_directory             batched via lazy generator (_batched), batch_size=128
```

**Lazy batching**: `_batched` is a generator that yields one batch at a time (`Iterator[list[Document]]`), so memory usage stays flat regardless of corpus size. `validate_batch_size` enforces positive-integer validation in one reusable place, called by both `_batched` and `_add_documents_in_batches`.

**Embedding drift detection**: if `.embedding_meta.json` records a different model or dimension, `sync_vectorstore` forces a full rebuild to prevent vector-dimension corruption.

## Design Choices

### Retrieval

- **bge-m3** (1024-dim, multilingual) replaces MiniLM for better Chinese semantic understanding. Embeddings are L2-normalized for cosine similarity.
- **Embedding fingerprint** (`.embedding_meta.json`) auto-detects model changes and forces vector store rebuild, preventing dimension-mismatch corruption.
- **Weighted RRF** fuses vector and BM25 results by rank, avoiding score normalization issues between distance-based and frequency-based retrievers. Overlapping documents accumulate score, naturally boosting consensus hits.
- **BM25** indexes are built lazily per collection on first query, reducing startup memory for large corpora.
- **Vector search failures** gracefully degrade to BM25-only retrieval.

### Query Understanding

- **HyDE** generates a hypothetical answer passage via LLM before retrieval. The passage embedding lands closer to real answer chunks than the short question would. BM25 and reranker still use the raw query to preserve exact-term matching. LLM failures fall back to the raw query without breaking the pipeline.
- **Code query detection** expands top_k (12 retrieve, 6 rerank) and pulls neighbor chunks (2 before, 3 after) by `chunk_index`, keeping split code blocks and comments together. Neighbor expansion respects source and collection boundaries.

### Ranking

- **bge-reranker-v2-m3** handles multilingual semantic relevance directly, reducing dependence on lexical hacks.
- A **lightweight lexical bonus** (0.15 × coverage) breaks near-ties on exact technical-term matches (acronyms like SM4/AES, function names).

### Generation

- Prompt constrains the model to answer only from retrieved chunks and cite chunk numbers.
- **Extractive fallback**: when the LLM API is unavailable, the system builds a deterministic answer from retrieved passages using query-term overlap scoring, so the pipeline never returns empty.

### Data Pipeline

- `PyMuPDF` extracts text per page, preserving page-level iteration.
- `RecursiveCharacterTextSplitter` respects Chinese punctuation (。？！；，) and paragraph boundaries.
- Chunk metadata includes global `chunk_index`, per-page `chunk_in_page`, `char_start`/`char_end`, `content_hash`, and stable `chunk_id`, making retrieval results traceable and debuggable.
- **Incremental indexing**: sha256 + mtime file fingerprints compare only stable fields (not runtime path strings), so `./data` and absolute-path invocations produce identical change detection. Changed PDFs are re-parsed and synced into Chroma by stable chunk IDs.

### Evaluation

- **Pure-stdlib IR metrics**: Recall@K, Precision@K, MRR, NDCG@K with multi-grade relevance (source match = 1, source + page match = 2).
- **Five-config ablation**: vector-only, BM25-only, hybrid-RRF, +rerank, full+HyDE. Output includes a comparison table and optional JSON report.

## Extension Points

- `src/query_rewrite.py`: multi-query expansion or query decomposition.
- `src/evaluate.py`: add answer faithfulness metrics (RAGAS-style LLM judge) or expand eval case coverage.
- `src/chain.py`: add streaming generation, multi-turn context, or function-calling for agentic retrieval.
- `worker/index.js`: add new protected routes or adjust rate-limit thresholds in `wrangler.jsonc`.
- `server.py`: expose pipeline as MCP tools for IDE/Agent integration.
