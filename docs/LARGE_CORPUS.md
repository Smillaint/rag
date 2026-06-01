# Large Corpus Notes

This project can handle small and medium local PDF knowledge bases directly. For a book-sized corpus, use the following settings and expectations.

## What Changed for Larger Inputs

- Chroma indexing is batched with `--vectorstore-batch-size` instead of inserting every chunk in one call.
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

## Suggested Indexing Command

For a large PDF or a full book, use a larger chunk size to reduce chunk count:

```powershell
.\.venv\Scripts\python.exe main.py --rebuild `
  --chunk-size 900 `
  --chunk-overlap 120 `
  --vectorstore-batch-size 128 `
  --collection algorithms_book `
  "这本书主要讲了什么？"
```

If Windows reports that `vectorstore/` files are in use, stop the running API server first, then rebuild.

## Current Boundary

A hundred-megabyte text corpus is feasible as a local RAG demo if:

- the embedding and reranker models are already cached locally;
- indexing is allowed to take time;
- queries usually specify `collection`;
- the machine has enough memory for the loaded chunks and the active BM25 collection.

This is still not an enterprise-scale RAG backend. For million-level chunks or multi-user concurrency, move keyword retrieval to Elasticsearch/OpenSearch and vector retrieval to Qdrant, Milvus, or a managed vector service.
