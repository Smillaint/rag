# -*- coding: utf-8 -*-
from src.metrics import (
    EvalCase,
    RelevantPage,
    aggregate_metrics,
    chunk_relevance_grade,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def _case(sources=None, pages=None):
    return EvalCase(
        question="q",
        relevant_sources=sources or [],
        relevant_source_pages=pages or [],
    )


def test_chunk_relevance_grade_source_only() -> None:
    case = _case(sources=["graph.pdf"])
    assert chunk_relevance_grade({"source": "graph.pdf", "page": 1}, case) == 1
    assert chunk_relevance_grade({"source": "other.pdf", "page": 1}, case) == 0


def test_chunk_relevance_grade_source_and_page() -> None:
    case = _case(
        sources=["graph.pdf"],
        pages=[RelevantPage(source="graph.pdf", pages=[5, 6])],
    )
    assert chunk_relevance_grade({"source": "graph.pdf", "page": 5}, case) == 2
    assert chunk_relevance_grade({"source": "graph.pdf", "page": 7}, case) == 1
    assert chunk_relevance_grade({"source": "other.pdf", "page": 5}, case) == 0


def test_chunk_relevance_grade_case_insensitive() -> None:
    case = _case(sources=["Graph.PDF"])
    assert chunk_relevance_grade({"source": "graph.pdf", "page": 1}, case) == 1


def test_recall_at_k() -> None:
    grades = [1, 0, 0, 1, 0]
    assert recall_at_k(grades, 3) == 0.5
    assert recall_at_k(grades, 5) == 1.0
    assert recall_at_k(grades, 2) == 0.5


def test_recall_at_k_no_relevant() -> None:
    assert recall_at_k([0, 0, 0], 3) == 0.0


def test_precision_at_k() -> None:
    grades = [1, 0, 1, 0, 0]
    assert precision_at_k(grades, 5) == 0.4
    assert precision_at_k(grades, 2) == 0.5
    assert precision_at_k(grades, 1) == 1.0


def test_mrr() -> None:
    assert mrr([1, 0, 0]) == 1.0
    assert mrr([0, 1, 0]) == 0.5
    assert mrr([0, 0, 1]) == 1 / 3
    assert mrr([0, 0, 0]) == 0.0


def test_ndcg_at_k_ideal_ordering() -> None:
    grades = [2, 1, 0]
    assert abs(ndcg_at_k(grades, 3) - 1.0) < 1e-6


def test_ndcg_at_k_suboptimal_ordering() -> None:
    grades = [0, 1, 2]
    result = ndcg_at_k(grades, 3)
    assert 0.0 < result < 1.0


def test_ndcg_at_k_all_irrelevant() -> None:
    assert ndcg_at_k([0, 0, 0], 3) == 0.0


def test_ndcg_at_k_multi_grade_rewards_higher() -> None:
    grades_first = [2, 0, 0]
    grades_second = [0, 2, 0]
    assert ndcg_at_k(grades_first, 3) > ndcg_at_k(grades_second, 3)


def test_aggregate_metrics() -> None:
    per_case = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ]
    result = aggregate_metrics(per_case, k_values=[3, 5])
    assert "recall@3" in result
    assert "precision@3" in result
    assert "ndcg@3" in result
    assert "mrr" in result
    assert result["case_count"] == 2
    assert result["recall@5"] == 1.0
    assert result["mrr"] == 0.75


def test_aggregate_metrics_empty() -> None:
    result = aggregate_metrics([], k_values=[5])
    assert result["recall@5"] == 0.0
    assert result["case_count"] == 0
