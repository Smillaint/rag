# -*- coding: utf-8 -*-
import os

from src.chain import RAGPipeline
from main import load_local_env


def main() -> None:
    load_local_env()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Please set DEEPSEEK_API_KEY before running this script.")

    pipeline = RAGPipeline.from_paths(api_key=api_key)

    queries = [
        "What is the core topic of the document?",
        "Which workflow or method does the document describe?",
        "What are the key metrics or criteria mentioned in the document?",
    ]

    for query in queries:
        print(f"\n{'=' * 50}")
        print(f"Question: {query}")
        result = pipeline.ask_with_sources(query)
        print(f"Generation mode: {result['trace']['generation']['mode']}")
        print(f"Answer:\n{result['answer']}")


if __name__ == "__main__":
    main()
