# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document


@dataclass
class EvalCase:
    question: str
    expected_sources: list[str]
    expected_keywords: list[str]


def load_eval_cases(path: str) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            question = payload.get("question")
            if not question:
                raise ValueError(f"Missing question at {path}:{line_number}")
            cases.append(
                EvalCase(
                    question=question,
                    expected_sources=payload.get("expected_sources", []),
                    expected_keywords=payload.get("expected_keywords", []),
                )
            )
    return cases


def _contains_expected_source(docs: list[Document], expected_sources: list[str]) -> bool:
    if not expected_sources:
        return True

    sources = [str(doc.metadata.get("source", "")).lower() for doc in docs]
    return any(
        expected.lower() in source
        for expected in expected_sources
        for source in sources
    )


def _contains_expected_keyword(docs: list[Document], expected_keywords: list[str]) -> bool:
    if not expected_keywords:
        return True

    combined_text = "\n".join(doc.page_content for doc in docs).lower()
    return any(keyword.lower() in combined_text for keyword in expected_keywords)


def evaluate_retrieval(
    cases: list[EvalCase],
    chunks: list[Document],
    vectorstore,
    retrieve_top_k: int,
    rerank_top_k: int,
    use_reranker: bool,
) -> dict:
    from src.reranker import load_reranker, rerank
    from src.retriever import BM25Index, hybrid_search

    bm25_index = BM25Index(chunks)
    reranker = load_reranker() if use_reranker else None
    results = []

    for case in cases:
        candidates = hybrid_search(
            vectorstore,
            chunks,
            case.question,
            top_k=retrieve_top_k,
            bm25_index=bm25_index,
        )
        docs = rerank(reranker, case.question, candidates, top_k=rerank_top_k) if reranker else candidates[:rerank_top_k]
        source_hit = _contains_expected_source(docs, case.expected_sources)
        keyword_hit = _contains_expected_keyword(docs, case.expected_keywords)
        passed = source_hit and keyword_hit
        results.append(
            {
                "question": case.question,
                "passed": passed,
                "source_hit": source_hit,
                "keyword_hit": keyword_hit,
                "returned_sources": [
                    {
                        "source": doc.metadata.get("source"),
                        "page": doc.metadata.get("page"),
                        "chunk_id": doc.metadata.get("chunk_id"),
                    }
                    for doc in docs
                ],
            }
        )

    passed_count = sum(1 for result in results if result["passed"])
    total = len(results)
    return {
        "total": total,
        "passed": passed_count,
        "pass_rate": passed_count / total if total else 0.0,
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality with JSONL cases.")
    parser.add_argument("--eval-file", default="./examples/eval_questions.jsonl")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--vectorstore-dir", default="./vectorstore")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--retrieve-top-k", type=int, default=5)
    parser.add_argument("--rerank-top-k", type=int, default=3)
    parser.add_argument("--skip-rerank", action="store_true")
    return parser.parse_args()


def main():
    from src.loader import load_pdfs, split_documents
    from src.retriever import build_vectorstore, load_vectorstore

    args = parse_args()
    cases = load_eval_cases(args.eval_file)
    docs = load_pdfs(args.data_dir)
    chunks = split_documents(docs)

    vectorstore_path = Path(args.vectorstore_dir)
    if args.rebuild or not vectorstore_path.exists():
        vectorstore = build_vectorstore(chunks, persist_dir=args.vectorstore_dir)
    else:
        vectorstore = load_vectorstore(args.vectorstore_dir)

    report = evaluate_retrieval(
        cases=cases,
        chunks=chunks,
        vectorstore=vectorstore,
        retrieve_top_k=args.retrieve_top_k,
        rerank_top_k=args.rerank_top_k,
        use_reranker=not args.skip_rerank,
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
