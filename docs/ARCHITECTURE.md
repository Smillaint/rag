# Architecture

## Pipeline

```text
Local PDFs
  -> load_pdfs()
  -> split_documents() with page/offset metadata
  -> build_vectorstore() + BM25Index
  -> hybrid_search()
  -> rerank()
  -> code-aware neighbor expansion
  -> generate_answer()
```

## Design Choices

- `PyMuPDF` is used for PDF extraction because it is lightweight and preserves page-level iteration.
- `RecursiveCharacterTextSplitter` keeps chunks around a fixed size while respecting paragraph and sentence boundaries when possible.
- Chunk metadata includes global chunk index, per-page chunk index, character offsets, and chunk length, making retrieval results easier to debug and cite.
- `Chroma` provides persistent local vector search, so embeddings do not need to be rebuilt for every run.
- `BM25` complements vector retrieval for exact keywords, names, acronyms, and technical terms.
- `CrossEncoder` reranking scores `(query, passage)` pairs directly, improving precision before LLM generation.
- Code-related questions automatically use a larger retrieval window and include adjacent chunks, which keeps split code blocks and comments together.
- Source metadata is preserved through the full pipeline to support citations and debugging.

## Main Extension Points

- `src/query_rewrite.py`: add query rewriting or multi-query expansion.
- `src/evaluate.py`: add Recall@K, MRR, nDCG, or answer faithfulness metrics.
- `main.py`: add service mode or route the pipeline behind FastAPI.
