# -*- coding: utf-8 -*-
import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from src.loader import assign_chunk_metadata, load_pdf_file, split_documents


CACHE_VERSION = 1


@dataclass
class IndexUpdateResult:
    chunks: list[Document]
    changed_sources: list[str]
    deleted_sources: list[str]
    deleted_chunk_ids: list[str]
    cache_hit: bool
    full_rebuild: bool


def _cache_paths(cache_dir: str) -> tuple[Path, Path]:
    root = Path(cache_dir)
    return root / "manifest.json", root / "chunks.jsonl"


def _pdf_inventory(data_dir: str) -> dict[str, dict]:
    inventory: dict[str, dict] = {}
    for path in sorted(Path(data_dir).glob("*.pdf")):
        stat = path.stat()
        inventory[path.name] = {
            "name": path.name,
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return inventory


def _document_to_record(doc: Document) -> dict:
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata,
    }


def _record_to_document(record: dict) -> Document:
    return Document(
        page_content=record["page_content"],
        metadata=record.get("metadata", {}),
    )


def _load_cache(cache_dir: str) -> tuple[dict | None, list[Document]]:
    manifest_path, chunks_path = _cache_paths(cache_dir)
    if not manifest_path.exists() or not chunks_path.exists():
        return None, []

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks: list[Document] = []
    with chunks_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            chunks.append(_record_to_document(json.loads(line)))
    return manifest, chunks


def _save_cache(
    cache_dir: str,
    manifest: dict,
    chunks: list[Document],
) -> None:
    manifest_path, chunks_path = _cache_paths(cache_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with chunks_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(_document_to_record(chunk), ensure_ascii=False) + "\n")


def _cache_is_compatible(
    manifest: dict | None,
    chunk_size: int,
    chunk_overlap: int,
) -> bool:
    if not manifest:
        return False
    return (
        manifest.get("version") == CACHE_VERSION
        and manifest.get("chunk_size") == chunk_size
        and manifest.get("chunk_overlap") == chunk_overlap
    )


def load_or_update_chunks(
    data_dir: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    cache_dir: str = "./.rag_cache",
    force_rebuild: bool = False,
) -> IndexUpdateResult:
    """Load cached chunks and incrementally parse PDFs that changed."""
    current_files = _pdf_inventory(data_dir)
    manifest, cached_chunks = _load_cache(cache_dir)
    compatible = _cache_is_compatible(manifest, chunk_size, chunk_overlap)

    if force_rebuild or not compatible:
        changed_sources = sorted(current_files)
        chunks: list[Document] = []
        for source in changed_sources:
            docs = load_pdf_file(current_files[source]["path"])
            chunks.extend(split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
        assign_chunk_metadata(chunks)
        _save_cache(
            cache_dir,
            {
                "version": CACHE_VERSION,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "files": current_files,
            },
            chunks,
        )
        print(f"Built chunk cache with {len(chunks)} chunks at {cache_dir}")
        return IndexUpdateResult(
            chunks=chunks,
            changed_sources=changed_sources,
            deleted_sources=[],
            deleted_chunk_ids=[],
            cache_hit=False,
            full_rebuild=True,
        )

    cached_files = manifest.get("files", {})
    current_sources = set(current_files)
    cached_sources = set(cached_files)
    deleted_sources = sorted(cached_sources - current_sources)
    changed_sources = sorted(
        source
        for source in current_sources
        if source not in cached_files or current_files[source] != cached_files[source]
    )

    affected_sources = set(deleted_sources + changed_sources)
    deleted_chunk_ids = [
        str(chunk.metadata.get("chunk_id"))
        for chunk in cached_chunks
        if chunk.metadata.get("source") in affected_sources
    ]
    chunks = [
        chunk
        for chunk in cached_chunks
        if chunk.metadata.get("source") not in affected_sources
    ]

    for source in changed_sources:
        docs = load_pdf_file(current_files[source]["path"])
        chunks.extend(split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    assign_chunk_metadata(chunks)
    cache_hit = not changed_sources and not deleted_sources

    _save_cache(
        cache_dir,
        {
            "version": CACHE_VERSION,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "files": current_files,
        },
        chunks,
    )

    if cache_hit:
        print(f"Loaded {len(chunks)} chunks from cache at {cache_dir}")
    else:
        print(
            "Updated chunk cache: "
            f"{len(changed_sources)} changed, {len(deleted_sources)} deleted, "
            f"{len(chunks)} total chunks"
        )

    return IndexUpdateResult(
        chunks=chunks,
        changed_sources=changed_sources,
        deleted_sources=deleted_sources,
        deleted_chunk_ids=deleted_chunk_ids,
        cache_hit=cache_hit,
        full_rebuild=False,
    )
