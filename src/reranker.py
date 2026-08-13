# -*- coding: utf-8 -*-
import logging
import os
import re
from typing import TYPE_CHECKING

import jieba
from sentence_transformers import CrossEncoder


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_MAX_LENGTH = 512
LEXICAL_BONUS_WEIGHT = 0.15
MIN_TERM_LENGTH = 2


def load_reranker(model_name: str = RERANKER_MODEL_NAME) -> CrossEncoder:
    """Load the multilingual CrossEncoder reranker model."""
    try:
        model = CrossEncoder(model_name, max_length=RERANKER_MAX_LENGTH)
    except Exception as exc:
        logger.warning("Reranker remote check failed, retrying local cache: %s", exc)
        model = CrossEncoder(
            model_name,
            max_length=RERANKER_MAX_LENGTH,
            local_files_only=True,
        )
    logger.info("Reranker model loaded")
    return model


def rerank(
    model: CrossEncoder,
    query: str,
    docs: list["Document"],
    top_k: int = 3,
) -> list["Document"]:
    """Rerank candidate documents and return the top_k most relevant chunks."""
    return [doc for doc, _ in rerank_with_scores(model, query, docs, top_k=top_k)]


def _query_terms(query: str) -> set[str]:
    """Extract ASCII tokens and jieba Chinese terms for lexical matching."""
    ascii_terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9_./#-]{2,}", query)
        if len(term.strip("-_./#")) >= MIN_TERM_LENGTH
    }
    chinese_terms = {
        term.strip().lower()
        for term in jieba.cut(query)
        if len(term.strip()) >= MIN_TERM_LENGTH
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
    return LEXICAL_BONUS_WEIGHT * coverage


def rerank_with_scores(
    model: CrossEncoder,
    query: str,
    docs: list["Document"],
    top_k: int = 3,
) -> list[tuple["Document", float]]:
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

    ranked = sorted(scored, key=lambda item: item[0], reverse=True)

    logger.info("Rerank selected top %d from %d candidates", top_k, len(docs))
    return [(doc, float(score)) for score, doc in ranked[:top_k]]
