# -*- coding: utf-8 -*-
# src/reranker.py

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


def load_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """加载 CrossEncoder rerank 模型"""
    model = CrossEncoder(model_name, max_length=512)
    print("✅ Reranker 模型加载完成")
    return model


def rerank(model, query: str, docs: list, top_k: int = 3) -> list:
    """
    对候选文档重新精排
    :param model: CrossEncoder 模型
    :param query: 用户问题
    :param docs: 候选 Document 列表
    :param top_k: 返回最相关的前 k 个
    :return: 精排后的 Document 列表
    """
    if not docs:
        return []

    # 构建 (query, passage) 对
    pairs = [(query, doc.page_content) for doc in docs]

    # 打分
    scores = model.predict(pairs)

    # 按分数降序排列
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    top_docs = [doc for _, doc in ranked[:top_k]]
    print(f"✅ Rerank 完成，从 {len(docs)} 个候选中选出 top {top_k}")
    return top_docs
