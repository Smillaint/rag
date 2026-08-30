# Large Corpus Notes

This project can handle small and medium local PDF knowledge bases directly. For a book-sized corpus, use the following settings and expectations.

## What Changed for Larger Inputs

- **Lazy batched indexing**: `_batched` is now a generator (`Iterator[list[Document]]`) that yields one batch at a time, so memory stays flat regardless of corpus size. `validate_batch_size` enforces positive-integer validation in one reusable place.
- **argparse rebuild script**: `scripts/rebuild_vectorstore.py` supports `--full-rebuild` / `--incremental` modes, environment-variable defaults, and CLI overrides.
- **Incremental sync**: in `--incremental` mode, the script passes the cache's `changed_sources` / `deleted_chunk_ids` / `full_rebuild` into `sync_vectorstore`, so only affected chunks are re-embedded and written.
- BM25 indexes are built lazily on first search and cached per `collection`, instead of being built for all chunks during service startup.
- `collection` filtering is applied to both vector search and BM25 search, so large unrelated document groups do not compete in one retrieval pool.

## Recommended Layout

```text
data/
  algorithms_book/
    chapter01.pdf
    chapter02.pdf
  crypto_book/
    sm4.pdf
```

Run queries with a collection filter whenever possible:

```powershell
.\.venv\Scripts\python.exe main.py --collection algorithms_book "Dijkstra算法适合解决什么问题？"
```

## Vector Store Rebuild Script

### Full rebuild (default)

When neither `--full-rebuild` nor `--incremental` is given, the script defaults to full rebuild. This preserves the historical behavior.

```powershell
.\.venv\Scripts\python.exe scripts\rebuild_vectorstore.py --full-rebuild
```

Summary output（数值为占位符，实际取决于语料和硬件）：

```
mode=full chunks=<chunk_count> changed=0 deleted=0 batch_size=32 elapsed=<elapsed_seconds>s
DONE
```

### Incremental sync

Only re-embed and write chunks from changed/deleted sources detected by the chunk cache:

```powershell
.\.venv\Scripts\python.exe scripts\rebuild_vectorstore.py --incremental
```

Summary output（数值为占位符，实际取决于语料和硬件）：

```
mode=incremental chunks=<chunk_count> changed=<changed_count> deleted=<deleted_count> batch_size=32 elapsed=<elapsed_seconds>s
DONE
```

### Environment variables and CLI overrides

All path and size defaults respect the corresponding `RAG_*` environment variables. CLI flags override the environment.

| Option | Env var | Default | CLI flag |
|---|---|---|---|
| Data directory | `RAG_DATA_DIR` | `./data` | `--data-dir` |
| Cache directory | `RAG_CACHE_DIR` | `./.rag_cache` | `--cache-dir` |
| Vector store directory | `RAG_VECTORSTORE_DIR` | `./vectorstore` | `--persist-dir` |
| Chunk size | `RAG_CHUNK_SIZE` | `900` | `--chunk-size` |
| Chunk overlap | `RAG_CHUNK_OVERLAP` | `120` | `--chunk-overlap` |
| Batch size | `RAG_VECTORSTORE_BATCH_SIZE` | `32` | `--batch-size` |

Example with overrides:

```powershell
$env:RAG_VECTORSTORE_BATCH_SIZE = "64"
.\.venv\Scripts\python.exe scripts\rebuild_vectorstore.py --incremental --chunk-size 1200 --chunk-overlap 150
```

### Mode behavior

| Mode | `force_rebuild` (cache) | `full_rebuild` (sync) | What happens |
|---|---|---|---|
| `--full-rebuild` (or default) | `True` | `True` | Wipe `vectorstore/`, re-embed all chunks, rewrite `.embedding_meta.json`. |
| `--incremental` | `False` | From cache result | Delete stale `deleted_chunk_ids`, add only `changed_sources` chunks. If the cache itself reports `full_rebuild=True`, sync receives `full_rebuild=True` too. |

> When the embedding model changes (detected via `.embedding_meta.json`), `sync_vectorstore` forces a full rebuild regardless of the CLI mode.

### Lazy batched writes

`_batched` yields batches lazily — it does not materialize a list of all batch lists. This means a 100 000-chunk corpus does not create 100 000 / batch_size slice objects upfront; only one batch exists in memory at a time. `validate_batch_size` is called once at the start of `_batched` and `_add_documents_in_batches`.

## Current Single-Machine Boundary

A hundred-megabyte text corpus is feasible as a local RAG demo if:

- the embedding and reranker models are already cached locally;
- indexing is allowed to take time（具体耗时取决于 CPU/GPU、chunk 数量和语料规模，需在目标机器上 benchmark）;
- queries usually specify `collection`;
- the machine has enough memory for the loaded chunks and the active BM25 collection;
- concurrent users are low (the Worker rate limit is 120 API requests / 60 s and 10 ask requests / 60 s per identity).

This is still not an enterprise-scale RAG backend. The Chroma vector store is a single-process embedded database; BM25 indexes live in process memory; embedding and reranking happen on CPU. There is no horizontal scaling, no cross-machine index sharing, and no background indexing queue. Actual indexing and query latency depend on hardware (CPU vs GPU, core count, memory), model size, and corpus size — run `scripts/rebuild_vectorstore.py` on the target machine to benchmark.

## Evolution Roadmap (Planned, Not Implemented)

The following are directions for scaling beyond the single-machine boundary. None of these are implemented today.

### Background indexing

- **Cloudflare Queues**: decouple PDF parsing and embedding from the request path. A rebuild trigger enqueues a job; a Worker or external consumer writes embeddings without blocking user queries.
- **Current state**: indexing is synchronous — `scripts/rebuild_vectorstore.py` runs in the foreground and the API server must be stopped if `vectorstore/` files are locked.

### Object storage for corpus

- **Cloudflare R2**: store PDFs and chunk cache (`chunks.jsonl`, `manifest.json`) in R2 instead of local disk. Enables multi-machine access and backup.
- **Current state**: PDFs and cache live on the local filesystem (`./data`, `./.rag_cache`).

### Managed vector search

- **Cloudflare Vectorize**: replace Chroma with Cloudflare's managed vector index. Embeddings are written via the Workers API; retrieval happens at the edge without a tunnel to the origin.
- **External Qdrant / Milvus / Weaviate**: self-hosted or managed vector databases with horizontal scaling, sharding, and filtered ANN search.
- **Current state**: Chroma embedded in the FastAPI process, single-machine persistence.

### Managed keyword search

- **Elasticsearch / OpenSearch**: move BM25 to a dedicated search engine for million-level corpora with multi-shard support.
- **Current state**: `rank_bm25` with in-memory indexes, lazily built per collection.

### What stays the same

The RRF fusion, HyDE query rewrite, reranker, and DeepSeek generation pipeline are retriever-agnostic. Swapping Chroma for Vectorize or BM25 for OpenSearch would only change the `vectorstore` and `bm25_index` layers in `src/retriever.py` and `src/chain.py` — the fusion and ranking logic remains unchanged.
