# -*- coding: utf-8 -*-
import os

from src.chain import RAGPipeline


API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set DEEPSEEK_API_KEY before running this test.")


pipeline = RAGPipeline.from_paths(api_key=API_KEY)

queries = [
    "What is the core topic of the document?",
    "Which workflow or method does the document describe?",
    "What are the key metrics or criteria mentioned in the document?",
]

for query in queries:
    print(f"\n{'=' * 50}")
    print(f"Question: {query}")
    print(f"Answer:\n{pipeline.ask(query)}")
