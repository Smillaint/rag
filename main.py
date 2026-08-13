# -*- coding: utf-8 -*-
import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def load_local_env(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file without overriding the shell."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the CLI."""
    parser = argparse.ArgumentParser(description="Run PDF RAG question answering.")
    parser.add_argument("query", nargs="?", help="Question to ask. If omitted, starts interactive mode.")
    parser.add_argument("--data-dir", default="./data", help="Directory containing local PDF files.")
    parser.add_argument("--vectorstore-dir", default="./vectorstore", help="Chroma persistence directory.")
    parser.add_argument("--cache-dir", default="./.rag_cache", help="Chunk cache directory.")
    parser.add_argument("--log-dir", default="./logs", help="Daily JSONL query log directory.")
    parser.add_argument("--collection", help="Limit retrieval to one data subdirectory collection.")
    parser.add_argument("--list-collections", action="store_true", help="List available PDF collections and exit.")
    parser.add_argument("--show-collection-files", action="store_true", help="Show PDF source paths when listing collections.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the vector store from PDFs.")
    parser.add_argument("--base-url", default="https://api.deepseek.com", help="OpenAI-compatible API base URL.")
    parser.add_argument("--model", default="deepseek-chat", help="Chat model name.")
    parser.add_argument("--retrieve-top-k", type=int, default=5, help="Top K for vector and BM25 retrieval.")
    parser.add_argument("--rerank-top-k", type=int, default=3, help="Top K after reranking.")
    parser.add_argument("--chunk-size", type=int, default=900, help="Maximum characters per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Overlapping characters between chunks.")
    parser.add_argument("--vectorstore-batch-size", type=int, default=128, help="Chroma indexing batch size.")
    return parser.parse_args()


def print_collections(data_dir: str, show_files: bool = False) -> dict[str, list[str]]:
    """List available PDF collections to stdout and return them as a dict."""
    from src.loader import list_pdf_collections

    collections = list_pdf_collections(data_dir)
    if not collections:
        print(f"No PDF collections found under {data_dir}.")
        return collections

    print("Available collections:")
    for name, sources in sorted(collections.items()):
        print(f"  - {name}: {len(sources)} PDF(s)")
        if show_files:
            for source in sources[:5]:
                print(f"      {source}")
            if len(sources) > 5:
                print(f"      ... {len(sources) - 5} more")
    print("Use --collection <name> to ask within a specific collection.")
    return collections


def main() -> None:
    """Run the CLI: parse args, build the pipeline, and answer questions."""
    load_local_env()
    args = parse_args()

    collections = print_collections(args.data_dir, show_files=args.show_collection_files)
    if args.list_collections:
        return
    if args.collection and args.collection not in collections:
        available = ", ".join(sorted(collections)) or "none"
        raise ValueError(
            f"Unknown collection '{args.collection}'. Available collections: {available}"
        )
    if not args.collection:
        print("No --collection specified; searching all collections.")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Please set DEEPSEEK_API_KEY before running main.py.")

    from src.chain import RAGPipeline

    pipeline = RAGPipeline.from_paths(
        api_key=api_key,
        data_dir=args.data_dir,
        vectorstore_dir=args.vectorstore_dir,
        cache_dir=args.cache_dir,
        rebuild=args.rebuild,
        base_url=args.base_url,
        model=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        log_dir=args.log_dir,
        vectorstore_batch_size=args.vectorstore_batch_size,
    )

    if args.query:
        print(pipeline.ask(args.query, args.retrieve_top_k, args.rerank_top_k, args.collection))
        return

    print("Enter a question, or type exit to quit.")
    while True:
        query = input("> ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            break
        if not query:
            continue
        print(pipeline.ask(query, args.retrieve_top_k, args.rerank_top_k, args.collection))


if __name__ == "__main__":
    main()
