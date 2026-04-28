# -*- coding: utf-8 -*-
# src/retriever.py

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置国内镜像

from langchain_huggingface import HuggingFaceEmbeddings  # 新版写法
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
import jieba


def get_embedding_model():
    """加载向量模型（使用国内镜像）"""
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}
    )
    return model


def build_vectorstore(chunks: list, persist_dir: str = "./vectorstore"):
    """构建并持久化向量数据库"""
    embedding = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_dir
    )
    print(f"向量库构建完成，共 {len(chunks)} 条记录，保存至 {persist_dir}")
    return vectorstore


def load_vectorstore(persist_dir: str = "./vectorstore"):
    """加载已有向量数据库"""
    embedding = get_embedding_model()
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding
    )
    print(f"向量库加载完成")
    return vectorstore


def bm25_search(chunks: list, query: str, top_k: int = 5) -> list:
    """BM25 关键词检索"""
    tokenized_corpus = [list(jieba.cut(doc.page_content)) for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i] for i in top_indices]


def hybrid_search(vectorstore, chunks: list, query: str, top_k: int = 5) -> list:
    """混合检索：向量检索 + BM25，结果合并去重"""
    vector_results = vectorstore.similarity_search(query, k=top_k)
    bm25_results = bm25_search(chunks, query, top_k=top_k)

    seen = set()
    combined = []
    for doc in vector_results + bm25_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)

    print(f"混合检索完成，返回 {len(combined)} 个候选片段")
    return combined[:top_k * 2]
