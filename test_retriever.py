# -*- coding: utf-8 -*-
from src.loader import load_pdfs, split_documents
from src.retriever import build_vectorstore, hybrid_search

# 加载数据
docs = load_pdfs('./data')
chunks = split_documents(docs)

# 构建向量库
vectorstore = build_vectorstore(chunks)

# 测试混合检索
query = "联邦学习"
results = hybrid_search(vectorstore, chunks, query, top_k=3)

print("\n--- 检索结果 ---")
for i, doc in enumerate(results):
    print(f"\n[{i+1}] 来源: {doc.metadata.get('source', '未知')}")
    print(doc.page_content[:150])
