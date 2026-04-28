# -*- coding: utf-8 -*-
import os

from src.generator import generate_answer, load_generator
from src.loader import load_pdfs, split_documents
from src.reranker import load_reranker, rerank
from src.retriever import hybrid_search, load_vectorstore


API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set DEEPSEEK_API_KEY before running this test.")


docs = load_pdfs("./data")
chunks = split_documents(docs)

vectorstore = load_vectorstore("./vectorstore")
reranker = load_reranker()
client, model = load_generator(api_key=API_KEY)

queries = [
    "联邦学习如何保护数据隐私？",
    "项目中使用了什么硬件加速方案？",
    "数据选择算法的核心指标是什么？",
]

for query in queries:
    print(f"\n{'=' * 50}")
    print(f"问题：{query}")

    candidates = hybrid_search(vectorstore, chunks, query, top_k=5)
    final_docs = rerank(reranker, query, candidates, top_k=3)
    answer = generate_answer(client, model, query, final_docs)

    print(f"回答：\n{answer}")
