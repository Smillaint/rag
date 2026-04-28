# -*- coding: utf-8 -*-
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run PDF RAG question answering.")
    parser.add_argument("query", nargs="?", help="Question to ask. If omitted, starts interactive mode.")
    parser.add_argument("--data-dir", default="./data", help="Directory containing local PDF files.")
    parser.add_argument("--vectorstore-dir", default="./vectorstore", help="Chroma persistence directory.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the vector store from PDFs.")
    parser.add_argument("--base-url", default="https://api.deepseek.com", help="OpenAI-compatible API base URL.")
    parser.add_argument("--model", default="deepseek-chat", help="Chat model name.")
    parser.add_argument("--retrieve-top-k", type=int, default=5, help="Top K for vector and BM25 retrieval.")
    parser.add_argument("--rerank-top-k", type=int, default=3, help="Top K after reranking.")
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Please set DEEPSEEK_API_KEY before running main.py.")

    from src.chain import RAGPipeline

    pipeline = RAGPipeline.from_paths(
        api_key=api_key,
        data_dir=args.data_dir,
        vectorstore_dir=args.vectorstore_dir,
        rebuild=args.rebuild,
        base_url=args.base_url,
        model=args.model,
    )

    if args.query:
        print(pipeline.ask(args.query, args.retrieve_top_k, args.rerank_top_k))
        return

    print("Enter a question, or type exit to quit.")
    while True:
        query = input("> ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            break
        if not query:
            continue
        print(pipeline.ask(query, args.retrieve_top_k, args.rerank_top_k))


if __name__ == "__main__":
    main()
