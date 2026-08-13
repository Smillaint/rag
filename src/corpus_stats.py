# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fitz

from src.retriever import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _pdf_file_stats(data_dir: str) -> dict:
    """Count PDFs, pages, sizes, and collections under a data directory."""
    root = Path(data_dir)
    pdf_paths = sorted(root.rglob("*.pdf"))
    total_size_bytes = 0
    total_pages = 0
    page_read_errors: list[str] = []
    collections: dict[str, int] = {}

    for path in pdf_paths:
        total_size_bytes += path.stat().st_size
        relative_path = path.relative_to(root)
        collection = relative_path.parts[0] if len(relative_path.parts) > 1 else "default"
        collections[collection] = collections.get(collection, 0) + 1
        try:
            with fitz.open(str(path)) as pdf:
                total_pages += pdf.page_count
        except Exception as exc:
            error_msg = f"{relative_path.as_posix()}: {type(exc).__name__}: {exc}"
            page_read_errors.append(error_msg)
            logger.warning("Failed to read PDF page count: %s", error_msg)

    return {
        "document_count": len(pdf_paths),
        "total_pages": total_pages,
        "source_size_bytes": total_size_bytes,
        "source_size_mb": round(total_size_bytes / (1024 * 1024), 2),
        "collection_pdf_counts": collections,
        "page_read_errors": page_read_errors,
    }


def _chunk_stats(chunks: list["Document"]) -> dict:
    """Compute chunk-level statistics (count, length distribution, sources)."""
    lengths = [len(chunk.page_content) for chunk in chunks]
    chunk_pages = {
        (chunk.metadata.get("source_path", chunk.metadata.get("source")), chunk.metadata.get("page"))
        for chunk in chunks
    }
    sources = {
        str(chunk.metadata.get("source_path", chunk.metadata.get("source", "")))
        for chunk in chunks
    }
    return {
        "chunk_count": len(chunks),
        "chunk_page_count": len(chunk_pages),
        "chunk_source_count": len(sources),
        "avg_chunk_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        "min_chunk_chars": min(lengths) if lengths else 0,
        "max_chunk_chars": max(lengths) if lengths else 0,
    }


def build_corpus_stats(data_dir: str, chunks: list["Document"], model: str) -> dict:
    """Return corpus and index statistics for monitoring and debugging."""
    pdf_stats = _pdf_file_stats(data_dir)
    chunk_stats = _chunk_stats(chunks)
    return {
        **pdf_stats,
        **chunk_stats,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "generation_model": model,
    }
