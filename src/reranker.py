# -*- coding: utf-8 -*-
import os
import re

import jieba
from sentence_transformers import CrossEncoder


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_reranker(model_name: str = "BAAI/bge-reranker-v2-m3"):
    """Load the multilingual CrossEncoder reranker model."""
    try:
        model = CrossEncoder(model_name, max_length=512)
    except Exception as exc:
        print(f"Reranker remote check failed, retrying local cache: {exc}")
        model = CrossEncoder(model_name, max_length=512, local_files_only=True)
    print("Reranker model loaded")
    return model


def rerank(model, query: str, docs: list, top_k: int = 3) -> list:
    """Rerank candidate documents and return the top_k most relevant chunks."""
    return [doc for doc, _ in rerank_with_scores(model, query, docs, top_k=top_k)]


def _query_terms(query: str) -> set[str]:
    ascii_terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9_./#-]{2,}", query)
        if len(term.strip("-_./#")) >= 2
    }
    chinese_terms = {
        term.strip().lower()
        for term in jieba.cut(query)
        if len(term.strip()) >= 2
    }
    return ascii_terms | chinese_terms


def _lexical_bonus(query: str, text: str) -> float:
    """Gentle lexical tiebreaker on top of the multilingual reranker.

    bge-reranker-v2-m3 already handles Chinese semantics well, so this only
    adds a small coverage-weighted bonus for exact technical-term matches
    (e.g. acronyms like SM4/AES) to break near-ties.
    """
    terms = _query_terms(query)
    if not terms:
        return 0.0

    normalized_text = text.lower()
    matched = [term for term in terms if term in normalized_text]
    if not matched:
        return 0.0

    coverage = len(matched) / len(terms)
    return 0.15 * coverage


def rerank_with_scores(model, query: str, docs: list, top_k: int = 3) -> list[tuple]:
    """Rerank candidates with CrossEncoder plus lexical term boosting."""
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    cross_scores = model.predict(pairs)
    scored = []
    for cross_score, doc in zip(cross_scores, docs):
        bonus = _lexical_bonus(query, doc.page_content)
        final_score = float(cross_score) + bonus
        scored.append((final_score, doc))

    ranked = sorted(scored, key=lambda x: x[0], reverse=True)

    print(f"Rerank selected top {top_k} from {len(docs)} candidates")
    return [(doc, float(score)) for score, doc in ranked[:top_k]]
