# -*- coding: utf-8 -*-
import os

from sentence_transformers import CrossEncoder


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Load the CrossEncoder reranker model."""
    model = CrossEncoder(model_name, max_length=512)
    print("Reranker model loaded")
    return model


def rerank(model, query: str, docs: list, top_k: int = 3) -> list:
    """Rerank candidate documents and return the top_k most relevant chunks."""
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    top_docs = [doc for _, doc in ranked[:top_k]]
    print(f"Rerank selected top {top_k} from {len(docs)} candidates")
    return top_docs
