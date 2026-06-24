# -*- coding: utf-8 -*-
from langchain_core.documents import Document

from src.chain import RAGPipeline


def _doc(source_path: str, index: int) -> Document:
    return Document(
        page_content=f"chunk {index} from {source_path}",
        metadata={
            "source": source_path.split("/")[-1],
            "source_path": source_path,
            "collection": source_path.split("/")[0],
            "page": 1,
            "chunk_index": index,
            "chunk_in_page": index,
            "chunk_id": f"{source_path}:p1:c{index}",
        },
    )


def test_code_context_expansion_does_not_cross_source_boundary() -> None:
    chunks = [
        _doc("algorithm/a.pdf", 0),
        _doc("algorithm/a.pdf", 1),
        _doc("algorithm/b.pdf", 2),
    ]
    pipeline = RAGPipeline(
        chunks=chunks,
        vectorstore=None,
        bm25_index=None,
        reranker=None,
        client=None,
        model="test",
        log_dir="",
    )

    expanded = pipeline.expand_with_neighbor_chunks([chunks[1]], before=1, after=2)

    assert expanded
    assert {doc.metadata["source_path"] for doc in expanded} == {"algorithm/a.pdf"}
