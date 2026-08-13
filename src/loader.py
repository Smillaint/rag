# -*- coding: utf-8 -*-
import hashlib
import logging
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_HASH_LENGTH = 12

CHUNK_SEPARATORS = [
    "\n\n",
    "\n",
    "\u3002",
    "\uff1f",
    "\uff01",
    "\uff1b",
    "\uff0c",
    " ",
    "",
]


def _relative_pdf_metadata(pdf_path: Path, data_root: Path | None = None) -> dict:
    """Compute source, source_path, and collection from a PDF path relative to data root."""
    root = data_root or pdf_path.parent
    try:
        relative_path = pdf_path.relative_to(root)
    except ValueError:
        relative_path = Path(pdf_path.name)

    source_path = relative_path.as_posix()
    collection = relative_path.parts[0] if len(relative_path.parts) > 1 else "default"
    return {
        "source": pdf_path.name,
        "source_path": source_path,
        "collection": collection,
    }


def load_pdfs(pdf_dir: str) -> list[Document]:
    """Load PDFs recursively as page-level Documents with trace metadata."""
    docs: list[Document] = []
    data_root = Path(pdf_dir)
    pdf_paths = sorted(data_root.rglob("*.pdf"))

    for pdf_path in pdf_paths:
        docs.extend(load_pdf_file(pdf_path, data_root=data_root))

    logger.info("Loaded %d PDF pages from %s", len(docs), pdf_dir)
    return docs


def list_pdf_collections(pdf_dir: str) -> dict[str, list[str]]:
    """Return available collections and PDF source paths under a data directory."""
    data_root = Path(pdf_dir)
    collections: dict[str, list[str]] = {}
    for pdf_path in sorted(data_root.rglob("*.pdf")):
        metadata = _relative_pdf_metadata(pdf_path, data_root)
        collection = str(metadata["collection"])
        collections.setdefault(collection, []).append(str(metadata["source_path"]))
    return collections


def load_pdf_file(pdf_path: str | Path, data_root: str | Path | None = None) -> list[Document]:
    """Load a single PDF as page-level Documents."""
    path = Path(pdf_path)
    root = Path(data_root) if data_root is not None else None
    base_metadata = _relative_pdf_metadata(path, root)
    docs: list[Document] = []
    with fitz.open(str(path)) as pdf:
        for page_index, page in enumerate(pdf, start=1):
            text = page.get_text().strip()
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        **base_metadata,
                        "page": page_index,
                    },
                )
            )
    return docs


def split_documents(
    docs: list[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Split documents into chunks while preserving source metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHUNK_SEPARATORS,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    assign_chunk_metadata(chunks)

    stats = summarize_chunks(chunks)
    logger.info(
        "Split into %d chunks (chunk_size=%d, overlap=%d, avg_len=%.1f, max_len=%d)",
        stats["total_chunks"],
        chunk_size,
        chunk_overlap,
        stats["avg_length"],
        stats["max_length"],
    )
    return chunks


def assign_chunk_metadata(chunks: list[Document]) -> None:
    """Assign stable source/page chunk metadata after splitting or merging chunks."""
    per_page_counts: dict[tuple[str, int | str], int] = {}
    for index, chunk in enumerate(chunks):
        source = str(chunk.metadata.get("source", "unknown"))
        source_path = str(chunk.metadata.get("source_path", source))
        page = chunk.metadata.get("page", "unknown")
        page_key = (source_path, page)
        chunk_in_page = per_page_counts.get(page_key, 0)
        per_page_counts[page_key] = chunk_in_page + 1

        char_start = chunk.metadata.get("start_index")
        if isinstance(char_start, int):
            chunk.metadata["char_start"] = char_start
            chunk.metadata["char_end"] = char_start + len(chunk.page_content)
        content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_in_page"] = chunk_in_page
        chunk.metadata["content_hash"] = content_hash
        chunk.metadata["logical_chunk_id"] = f"{source_path}:p{page}:c{chunk_in_page}"
        chunk.metadata["chunk_id"] = f"{source_path}:p{page}:h{content_hash[:CHUNK_HASH_LENGTH]}:c{chunk_in_page}"
        chunk.metadata["chunk_length"] = len(chunk.page_content)


def summarize_chunks(chunks: list[Document]) -> dict:
    """Return lightweight chunk statistics for debugging and tuning."""
    lengths = [len(chunk.page_content) for chunk in chunks]
    sources = {chunk.metadata.get("source_path", chunk.metadata.get("source")) for chunk in chunks}
    collections = {chunk.metadata.get("collection", "default") for chunk in chunks}
    pages = {
        (chunk.metadata.get("source"), chunk.metadata.get("page"))
        for chunk in chunks
    }

    return {
        "total_chunks": len(chunks),
        "source_count": len(sources),
        "collection_count": len(collections),
        "collections": sorted(str(collection) for collection in collections),
        "page_count": len(pages),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "avg_length": sum(lengths) / len(lengths) if lengths else 0.0,
    }
