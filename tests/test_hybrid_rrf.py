# -*- coding: utf-8 -*-
from langchain_core.documents import Document

from src.retriever import (
    BM25Index,
    _doc_key,
    _embedding_model_changed,
    _embedding_meta_path,
    _rrf_score,
    hybrid_search,
    hybrid_search_with_scores,
)


def _doc(cid: str, text: str, source: str = "a.pdf") -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "source_path": source,
            "page": 1,
            "chunk_id": cid,
        },
    )


class _FakeVectorStore:
    """Mimics Chroma similarity_search_with_score (smaller distance = better)."""

    def __init__(self, hits: list[tuple[Document, float]]):
        self._hits = hits

    def similarity_search_with_score(self, query, k=None, filter=None):
        return self._hits[:k]


def test_rrf_formula() -> None:
    # rank 1 with weight 1.0 and k=60 -> 1/61
    assert _rrf_score(1, 1.0, 60) == 1 / 61
    # rank 2 with weight 2.0 -> 2/62
    assert _rrf_score(2, 2.0, 60) == 2 / 62


def test_doc_key_prefers_chunk_id() -> None:
    doc = _doc("a.pdf:p1:h0:0", "hello")
    assert _doc_key(doc) == ("chunk_id", "a.pdf:p1:h0:0")


def test_overlap_document_accumulates_score_and_ranks_first() -> None:
    only_vector = _doc("c1", "vector only hit")
    overlap = _doc("c2", "agreed by both retrievers")
    only_bm25 = _doc("c3", "bm25 only hit")

    vectorstore = _FakeVectorStore([(overlap, 0.1), (only_vector, 0.5)])
    bm25_index = BM25Index([overlap, only_bm25])

    fused = hybrid_search_with_scores(
        vectorstore,
        [overlap, only_vector, only_bm25],
        "agreed",
        top_k=2,
        bm25_index=bm25_index,
    )
    keys = [_doc_key(doc) for doc, _ in fused]
    assert keys[0] == ("chunk_id", "c2")
    top_score = fused[0][1]
    single_scores = [score for _, score in fused]
    assert top_score == max(single_scores)
    # overlap should have strictly more score than each single-source hit
    assert top_score > fused[-1][1]


def test_hybrid_search_returns_documents_only() -> None:
    doc = _doc("c1", "hello world")
    vectorstore = _FakeVectorStore([(doc, 0.2)])
    result = hybrid_search(
        vectorstore,
        [doc],
        "hello",
        top_k=1,
        bm25_index=BM25Index([doc]),
    )
    assert all(isinstance(item, Document) for item in result)
    assert result[0].metadata["chunk_id"] == "c1"


def test_vector_search_failure_falls_back_to_bm25_only() -> None:
    doc = _doc("c1", "fallback doc")

    class _BrokenVectorStore:
        def similarity_search_with_score(self, query, k=None, filter=None):
            raise RuntimeError("chroma down")

    result = hybrid_search(
        _BrokenVectorStore(),
        [doc],
        "fallback",
        top_k=1,
        bm25_index=BM25Index([doc]),
    )
    assert len(result) == 1
    assert result[0].metadata["chunk_id"] == "c1"


def test_embedding_model_changed_detects_mismatch(tmp_path) -> None:
    import json

    from src.retriever import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME

    meta_path = tmp_path / ".embedding_meta.json"
    meta_path.write_text(
        json.dumps({"model": "old-model", "dimension": 384}),
        encoding="utf-8",
    )
    assert _embedding_model_changed(meta_path) is True

    meta_path.write_text(
        json.dumps({"model": EMBEDDING_MODEL_NAME, "dimension": EMBEDDING_DIMENSION}),
        encoding="utf-8",
    )
    assert _embedding_model_changed(meta_path) is False


def test_embedding_model_changed_when_meta_missing(tmp_path) -> None:
    meta_path = tmp_path / ".embedding_meta.json"
    assert _embedding_model_changed(meta_path) is True
