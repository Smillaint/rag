# -*- coding: utf-8 -*-
"""Information-retrieval metrics for RAG evaluation (pure standard library).

Relevance is graded per retrieved chunk:
  grade 2 = source AND page both match the labelled ground truth
  grade 1 = source matches but page was not specifically labelled
  grade 0 = irrelevant

This lets NDCG reward page-level precision while Recall/MRR work at the
source level, which is the practical labelling granularity for book-sized
corpora where chunk-level ground truth is infeasible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class RelevantPage:
    source: str
    pages: list[int] = field(default_factory=list)


@dataclass
class EvalCase:
    question: str
    collection: str | None = None
    relevant_sources: list[str] = field(default_factory=list)
    relevant_source_pages: list[RelevantPage] = field(default_factory=list)
    any_keywords: list[str] = field(default_factory=list)
    all_keywords: list[str] = field(default_factory=list)


def chunk_relevance_grade(metadata: dict, case: EvalCase) -> int:
    """Return relevance grade 0/1/2 for a chunk given the eval case."""
    source = str(metadata.get("source", "")).lower()
    page = metadata.get("page")

    for entry in case.relevant_source_pages:
        if entry.source.lower() == source and (not entry.pages or page in entry.pages):
            return 2

    for expected in case.relevant_sources:
        if expected.lower() in source:
            return 1

    return 0


def _grades_for_ranked_chunks(
    ranked_metadata: list[dict],
    case: EvalCase,
) -> list[int]:
    return [chunk_relevance_grade(meta, case) for meta in ranked_metadata]


def recall_at_k(grades: list[int], k: int) -> float:
    """Fraction of relevant items found in top-K (binary relevance)."""
    if not grades:
        return 0.0
    top_k = grades[:k]
    relevant_found = sum(1 for g in top_k if g > 0)
    total_relevant = sum(1 for g in grades if g > 0)
    if total_relevant == 0:
        return 0.0
    return relevant_found / total_relevant


def precision_at_k(grades: list[int], k: int) -> float:
    """Fraction of top-K items that are relevant (binary relevance)."""
    if not grades or k <= 0:
        return 0.0
    top_k = grades[:k]
    relevant_found = sum(1 for g in top_k if g > 0)
    return relevant_found / k


def mrr(grades: list[int]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant item."""
    for rank, grade in enumerate(grades, start=1):
        if grade > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(grades: list[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain with multi-grade relevance.

    DCG@K  = sum( grade_i / log2(i + 1) ) for i in 1..K
    IDCG@K = DCG@K of the ideal (sorted descending) ordering
    NDCG@K = DCG@K / IDCG@K
    """
    if not grades:
        return 0.0

    top_k = grades[:k]

    dcg = 0.0
    for i, grade in enumerate(top_k, start=1):
        dcg += (2 ** grade - 1) / math.log2(i + 1)

    ideal = sorted(grades, reverse=True)[:k]
    idcg = 0.0
    for i, grade in enumerate(ideal, start=1):
        idcg += (2 ** grade - 1) / math.log2(i + 1)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def aggregate_metrics(
    per_case_grades: list[list[int]],
    k_values: list[int],
) -> dict:
    """Compute mean Recall@K / Precision@K / MRR / NDCG@K across all cases."""
    results: dict[str, float] = {}
    n = len(per_case_grades)

    for k in k_values:
        recalls = [recall_at_k(grades, k) for grades in per_case_grades]
        precisions = [precision_at_k(grades, k) for grades in per_case_grades]
        ndcgs = [ndcg_at_k(grades, k) for grades in per_case_grades]
        results[f"recall@{k}"] = round(sum(recalls) / n, 4) if n else 0.0
        results[f"precision@{k}"] = round(sum(precisions) / n, 4) if n else 0.0
        results[f"ndcg@{k}"] = round(sum(ndcgs) / n, 4) if n else 0.0

    mrrs = [mrr(grades) for grades in per_case_grades]
    results["mrr"] = round(sum(mrrs) / n, 4) if n else 0.0
    results["case_count"] = n
    return results
