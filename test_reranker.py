# -*- coding: utf-8 -*-
from src.loader import load_pdfs, split_documents
from src.retriever import build_vectorstore, load_vectorstore, hybrid_search
from src.reranker import load_reranker, rerank
import os

# 加载数据
docs = load_pdfs('./data')
chunks = split_documents(docs)

# 向量库已存在就直接加载，不用重复构建
if os.path.exists('./vectorstore'):
    vectorstore = load_vectorstore('./vectorstore')
else:
    vectorstore = build_vectorstore(chunks)

# 混合检索
query = "联邦学习如何保护数据隐私"
candidates = hybrid_search(vectorstore, chunks, query, top_k=4)

# Rerank 精排
reranker = load_reranker()
final_results = rerank(reranker, query, candidates, top_k=3)

print("\n--- Rerank 后最终结果 ---")
for i, doc in enumerate(final_results):
    print(f"\n[{i+1}] 来源: {doc.metadata.get('source', '未知')}")
    print(doc.page_content[:200])
