# Architecture

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

Endpoints:

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
- `server.py`: expose pipeline as MCP tools for IDE/Agent integration.
