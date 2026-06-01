# -*- coding: utf-8 -*-
from langchain_core.documents import Document

from src.generator import generate_extractive_answer


def test_extractive_answer_contains_chunk_citation() -> None:
    docs = [
        Document(
            page_content="Dijkstra算法用于求解非负权图上的单源最短路径问题。",
            metadata={"source": "graph.pdf", "page": 1, "chunk_id": "graph.pdf:p1:c0"},
        )
    ]

    answer = generate_extractive_answer("Dijkstra算法适合解决什么问题？", docs)

    assert "Dijkstra" in answer
    assert "[Chunk 1]" in answer
