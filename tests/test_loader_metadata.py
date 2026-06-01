# -*- coding: utf-8 -*-
from langchain_core.documents import Document

from src.loader import split_documents


def test_split_documents_preserves_trace_metadata() -> None:
    docs = [
        Document(
            page_content="第一段介绍图论基础。第二段介绍最短路径算法。",
            metadata={"source": "chapter.pdf", "page": 3},
        )
    ]

    chunks = split_documents(docs, chunk_size=20, chunk_overlap=4)

    assert chunks
    first = chunks[0]
    assert first.metadata["source"] == "chapter.pdf"
    assert first.metadata["page"] == 3
    assert "chunk_id" in first.metadata
    assert "char_start" in first.metadata
    assert "char_end" in first.metadata
