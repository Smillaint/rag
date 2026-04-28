# -*- coding: utf-8 -*-
from src.loader import load_pdfs, split_documents
from src.retriever import BM25Index, build_vectorstore, hybrid_search


docs = load_pdfs("./data")
chunks = split_documents(docs)
vectorstore = build_vectorstore(chunks)
bm25_index = BM25Index(chunks)

query = "联邦学习"
results = hybrid_search(vectorstore, chunks, query, top_k=3, bm25_index=bm25_index)

print("\n--- 检索结果 ---")
for i, doc in enumerate(results, start=1):
    source = doc.metadata.get("source", "未知")
    page = doc.metadata.get("page", "未知页")
    print(f"\n[{i}] 来源: {source}, page: {page}")
    print(doc.page_content[:150])
