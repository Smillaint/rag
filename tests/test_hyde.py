# -*- coding: utf-8 -*-
from types import SimpleNamespace

from langchain_core.documents import Document

from src.query_rewrite import generate_hypothetical_document
from src.retriever import BM25Index, hybrid_search_with_scores


class _FakeCompletions:
    def __init__(self, content=None, raises=None):
        self._content = content
        self._raises = raises
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


class _FakeClient:
    """Mimics the OpenAI client.chat.completions.create surface."""

    def __init__(self, content=None, raises=None):
        self._completions = _FakeCompletions(content=content, raises=raises)
        self.chat = SimpleNamespace(completions=self._completions)

    @property
    def calls(self):
        return self._completions.calls


def test_hyde_generates_document_and_marks_used() -> None:
    client = _FakeClient(content="这是一段假想的答案文档，描述了如何启动实验平台。")
    result = generate_hypothetical_document(client, "test-model", "如何启动实验平台？")

    assert result.used is True
    assert result.error is None
    assert result.document is not None
    assert "实验" in result.document
    assert result.elapsed_ms >= 0
    assert len(client.calls) == 1
    messages = client.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "如何启动实验平台" in messages[1]["content"]


def test_hyde_falls_back_on_exception() -> None:
    client = _FakeClient(raises=RuntimeError("network down"))
    result = generate_hypothetical_document(client, "test-model", "任意问题")

    assert result.used is False
    assert result.document is None
    assert "network down" in (result.error or "")


def test_hyde_falls_back_on_empty_response() -> None:
    client = _FakeClient(content="   ")
    result = generate_hypothetical_document(client, "test-model", "任意问题")

    assert result.used is False
    assert result.document is None
    assert result.error == "empty_response"


class _FakeVectorStore:
    """Records the query it was called with; returns no hits."""

    def __init__(self):
        self.last_query = None

    def similarity_search_with_score(self, query, k=None, filter=None):
        self.last_query = query
        return []


def test_vector_query_used_for_vector_search_but_not_bm25() -> None:
    doc = _doc("c1", "密码学实验包含sm4算法的分组加密实现")
    vectorstore = _FakeVectorStore()
    bm25_index = BM25Index([doc])

    hyde_doc = "假想文档：本节介绍sm4分组密码的实验步骤。"
    hybrid_search_with_scores(
        vectorstore,
        [doc],
        "sm4算法怎么实现？",
        top_k=3,
        bm25_index=bm25_index,
        vector_query=hyde_doc,
    )

    assert vectorstore.last_query == hyde_doc


def test_vector_query_none_falls_back_to_raw_query() -> None:
    doc = _doc("c1", "内容")
    vectorstore = _FakeVectorStore()
    bm25_index = BM25Index([doc])

    hybrid_search_with_scores(
        vectorstore,
        [doc],
        "原始问题",
        top_k=3,
        bm25_index=bm25_index,
        vector_query=None,
    )

    assert vectorstore.last_query == "原始问题"


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
