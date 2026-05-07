# -*- coding: utf-8 -*-
from pathlib import Path

from src.loader import load_pdfs, split_documents
from src.reranker import load_reranker, rerank
from src.retriever import BM25Index, build_vectorstore, hybrid_search, load_vectorstore


docs = load_pdfs("./data")
chunks = split_documents(docs)

if Path("./vectorstore").exists():
    vectorstore = load_vectorstore("./vectorstore")
else:
    vectorstore = build_vectorstore(chunks)

query = "How does the document describe the main workflow?"
candidates = hybrid_search(
    vectorstore,
    chunks,
    query,
    top_k=4,
    bm25_index=BM25Index(chunks),
)

reranker = load_reranker()
final_results = rerank(reranker, query, candidates, top_k=3)

print("\n--- Reranked results ---")
for i, doc in enumerate(final_results, start=1):
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", "unknown")
    print(f"\n[{i}] source: {source}, page: {page}")
    print(doc.page_content[:200])
