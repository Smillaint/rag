# -*- coding: utf-8 -*-
import os

from sentence_transformers import CrossEncoder


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Load the CrossEncoder reranker model."""
    try:
        model = CrossEncoder(model_name, max_length=512)
    except Exception as exc:
        print(f"Reranker remote check failed, retrying local cache: {exc}")
        model = CrossEncoder(model_name, max_length=512, local_files_only=True)
    print("Reranker model loaded")
    return model


def rerank(model, query: str, docs: list, top_k: int = 3) -> list:
    """Rerank candidate documents and return the top_k most relevant chunks."""
    return [doc for doc, _ in rerank_with_scores(model, query, docs, top_k=top_k)]


def rerank_with_scores(model, query: str, docs: list, top_k: int = 3) -> list[tuple]:
    """Rerank candidate documents and return documents with CrossEncoder scores."""
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    print(f"Rerank selected top {top_k} from {len(docs)} candidates")
    return [(doc, float(score)) for score, doc in ranked[:top_k]]
