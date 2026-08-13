# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from src.metrics import (
    EvalCase,
    RelevantPage,
    aggregate_metrics,
    chunk_relevance_grade,
)

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from sentence_transformers import CrossEncoder
    from openai import OpenAI
    from src.retriever import BM25Index, Chroma

logger = logging.getLogger(__name__)


def load_eval_cases(path: str) -> list[EvalCase]:
    """Load eval cases from JSONL, supporting both old and new formats."""
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

            relevant_sources = payload.get("relevant_sources", payload.get("expected_sources", []))
            raw_pages = payload.get("relevant_source_pages", [])
            relevant_source_pages = [
                RelevantPage(
                    source=entry.get("source", ""),
                    pages=entry.get("pages", []),
                )
                for entry in raw_pages
            ]

            cases.append(
                EvalCase(
                    question=question,
                    collection=payload.get("collection"),
                    relevant_sources=relevant_sources,
                    relevant_source_pages=relevant_source_pages,
                    any_keywords=payload.get("any_keywords", payload.get("expected_keywords", [])),
                    all_keywords=payload.get("all_keywords", []),
                )
            )
    return cases


def _metadata_list(docs: list["Document"]) -> list[dict]:
    return [doc.metadata for doc in docs]


def _run_config(
    config_name: str,
    cases: list[EvalCase],
    chunks: list["Document"],
    vectorstore: "Chroma",
    bm25_index: "BM25Index",
    reranker: "CrossEncoder",
    client: "OpenAI",
    model: str,
    retrieve_top_k: int,
    rerank_top_k: int,
    k_values: list[int],
) -> dict:
    """Run one retrieval configuration across all eval cases and return metrics."""
    from src.retriever import hybrid_search_with_scores
    from src.reranker import rerank_with_scores
    from src.query_rewrite import generate_hypothetical_document

    per_case_grades: list[list[int]] = []
    per_case_details: list[dict] = []
    total_started = time.perf_counter()

    for case in cases:
        docs: list[Document] = []

        if config_name == "vector_only":
            search_kwargs: dict[str, object] = {"k": retrieve_top_k}
            if case.collection:
                search_kwargs["filter"] = {"collection": case.collection}
            docs = [doc for doc, _ in vectorstore.similarity_search_with_score(case.question, **search_kwargs)]

        elif config_name == "bm25_only":
            docs = bm25_index.search(case.question, top_k=retrieve_top_k, collection=case.collection)

        elif config_name == "hybrid_rrf":
            candidates = hybrid_search_with_scores(
                vectorstore, chunks, case.question,
                top_k=retrieve_top_k, bm25_index=bm25_index, collection=case.collection,
            )
            docs = [doc for doc, _ in candidates[:rerank_top_k]]

        elif config_name == "hybrid_rrf_rerank":
            candidates = hybrid_search_with_scores(
                vectorstore, chunks, case.question,
                top_k=retrieve_top_k, bm25_index=bm25_index, collection=case.collection,
            )
            ranked = rerank_with_scores(reranker, case.question, [doc for doc, _ in candidates], top_k=rerank_top_k)
            docs = [doc for doc, _ in ranked]

        elif config_name == "full_pipeline":
            hyde = generate_hypothetical_document(client, model, case.question)
            candidates = hybrid_search_with_scores(
                vectorstore, chunks, case.question,
                top_k=retrieve_top_k, bm25_index=bm25_index, collection=case.collection,
                vector_query=hyde.document,
            )
            ranked = rerank_with_scores(reranker, case.question, [doc for doc, _ in candidates], top_k=rerank_top_k)
            docs = [doc for doc, _ in ranked]
        else:
            raise ValueError(f"Unknown config: {config_name}")

        grades = [chunk_relevance_grade(meta, case) for meta in _metadata_list(docs)]
        per_case_grades.append(grades)

        per_case_details.append({
            "question": case.question,
            "grades": grades,
            "returned_sources": [
                {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "grade": chunk_relevance_grade(doc.metadata, case),
                }
                for doc in docs
            ],
        })

    elapsed_ms = (time.perf_counter() - total_started) * 1000
    metrics = aggregate_metrics(per_case_grades, k_values)
    metrics["elapsed_ms"] = round(elapsed_ms, 1)
    return {"config": config_name, "metrics": metrics, "details": per_case_details}


ABLATION_CONFIGS = ["vector_only", "bm25_only", "hybrid_rrf", "hybrid_rrf_rerank", "full_pipeline"]


def run_ablation(
    cases: list[EvalCase],
    chunks: list["Document"],
    vectorstore: "Chroma",
    bm25_index: "BM25Index",
    reranker: "CrossEncoder",
    client: "OpenAI",
    model: str,
    retrieve_top_k: int = 5,
    rerank_top_k: int = 3,
    k_values: list[int] | None = None,
    configs: list[str] | None = None,
) -> list[dict]:
    """Run multiple retrieval configs and return per-config metrics."""
    k_values = k_values or [3, 5]
    configs = configs or ABLATION_CONFIGS
    results = []
    for config in configs:
        logger.info("=== Running config: %s ===", config)
        result = _run_config(
            config, cases, chunks, vectorstore, bm25_index, reranker,
            client, model, retrieve_top_k, rerank_top_k, k_values,
        )
        logger.info("  %s", json.dumps(result["metrics"], ensure_ascii=False))
        results.append(result)
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation tool."""
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality with IR metrics.")
    parser.add_argument("--eval-file", default="./examples/eval_questions.jsonl")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--vectorstore-dir", default="./vectorstore")
    parser.add_argument("--cache-dir", default="./.rag_cache")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--retrieve-top-k", type=int, default=5)
    parser.add_argument("--rerank-top-k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--k-values", type=int, nargs="*", default=[3, 5])
    parser.add_argument("--configs", type=str, nargs="*", default=None,
                        help="Configs to run (default: all). Choices: vector_only bm25_only hybrid_rrf hybrid_rrf_rerank full_pipeline")
    parser.add_argument("--output", default=None, help="Write full report JSON to this path.")
    parser.add_argument("--summary-only", action="store_true", help="Print only the metrics table, skip per-case details.")
    return parser.parse_args()


def main() -> None:
    """Run the ablation evaluation from the command line."""
    from src.index_cache import load_or_update_chunks
    from src.retriever import BM25Index, sync_vectorstore
    from src.reranker import load_reranker
    from src.generator import load_generator

    args = parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Please set DEEPSEEK_API_KEY before running evaluate.")

    cases = load_eval_cases(args.eval_file)
    logger.info("Loaded %d eval cases from %s", len(cases), args.eval_file)

    index_result = load_or_update_chunks(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        cache_dir=args.cache_dir,
        force_rebuild=args.rebuild,
    )
    chunks = index_result.chunks
    vectorstore = sync_vectorstore(
        chunks=chunks,
        persist_dir=args.vectorstore_dir,
        changed_sources=index_result.changed_sources,
        deleted_chunk_ids=index_result.deleted_chunk_ids,
        full_rebuild=args.rebuild or index_result.full_rebuild,
    )
    bm25_index = BM25Index(chunks)
    reranker = load_reranker()
    client, model = load_generator(api_key=api_key)

    results = run_ablation(
        cases=cases,
        chunks=chunks,
        vectorstore=vectorstore,
        bm25_index=bm25_index,
        reranker=reranker,
        client=client,
        model=model,
        retrieve_top_k=args.retrieve_top_k,
        rerank_top_k=args.rerank_top_k,
        k_values=args.k_values,
        configs=args.configs,
    )

    summary_separator = "=" * 70
    print()
    print(summary_separator)
    print("ABLATION SUMMARY")
    print(summary_separator)
    header_k_values = args.k_values
    metric_names: list[str] = []
    for k in header_k_values:
        metric_names += [f"R@{k}", f"P@{k}", f"N@{k}"]
    metric_names += ["MRR", "ms"]
    headers = ["Config"] + metric_names
    col_widths = [max(len(header), 18) for header in headers]
    print("  ".join(header.ljust(width) for header, width in zip(headers, col_widths)))
    print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for result in results:
        metrics = result["metrics"]
        row = [result["config"]]
        for k in header_k_values:
            row.append(f"{metrics.get(f'recall@{k}', 0):.3f}")
            row.append(f"{metrics.get(f'precision@{k}', 0):.3f}")
            row.append(f"{metrics.get(f'ndcg@{k}', 0):.3f}")
        row.append(f"{metrics.get('mrr', 0):.3f}")
        row.append(f"{metrics.get('elapsed_ms', 0):.0f}")
        print("  ".join(str(value).ljust(width) for value, width in zip(row, col_widths)))

    if args.output:
        report = {
            "eval_file": args.eval_file,
            "retrieve_top_k": args.retrieve_top_k,
            "rerank_top_k": args.rerank_top_k,
            "k_values": args.k_values,
            "results": results if not args.summary_only else [
                {key: value for key, value in result.items() if key != "details"}
                for result in results
            ],
        }
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nFull report written to {args.output}")


if __name__ == "__main__":
    main()
