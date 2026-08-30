# -*- coding: utf-8 -*-
"""Rebuild or incrementally synchronise the Chroma vector store.

Usage examples
--------------
    # Full rebuild (default when neither flag is given)
    python scripts/rebuild_vectorstore.py --full-rebuild

    # Incremental sync — only changed/deleted sources are touched
    python scripts/rebuild_vectorstore.py --incremental

All defaults respect the corresponding RAG_* environment variables; CLI
flags override the environment.  This module is import-safe: no work runs on
import.  Entry point is ``main(argv=None)``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure the project root is importable when run as a standalone script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Keep Hugging Face offline so the script never phones home.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_BATCH_SIZE = 32
ENV_VARS = (
    "RAG_DATA_DIR",
    "RAG_CACHE_DIR",
    "RAG_VECTORSTORE_DIR",
    "RAG_CHUNK_SIZE",
    "RAG_CHUNK_OVERLAP",
    "RAG_VECTORSTORE_BATCH_SIZE",
)


def load_local_env(path: str = ".env") -> None:
    """Load a minimal .env file into ``os.environ`` (does not overwrite)."""
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


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{name} must be an integer, got {value!r}.") from exc


def default_options() -> dict:
    """Return option defaults derived from the environment."""
    return {
        "data_dir": _env_str("RAG_DATA_DIR", "./data"),
        "cache_dir": _env_str("RAG_CACHE_DIR", "./.rag_cache"),
        "persist_dir": _env_str("RAG_VECTORSTORE_DIR", "./vectorstore"),
        "chunk_size": _env_int("RAG_CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
        "chunk_overlap": _env_int("RAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP),
        "batch_size": _env_int("RAG_VECTORSTORE_BATCH_SIZE", DEFAULT_BATCH_SIZE),
    }


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser. Exposed for unit testing."""
    defaults = default_options()
    parser = argparse.ArgumentParser(
        prog="rebuild_vectorstore",
        description="Rebuild or incrementally synchronise the RAG vector store.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--full-rebuild",
        action="store_true",
        help="Force a full vector store rebuild (default when neither flag is given).",
    )
    mode.add_argument(
        "--incremental",
        action="store_true",
        help="Only synchronise changed/deleted sources detected by the chunk cache.",
    )
    parser.add_argument(
        "--data-dir",
        default=defaults["data_dir"],
        help=f"PDF data directory (default: {defaults['data_dir']}).",
    )
    parser.add_argument(
        "--cache-dir",
        default=defaults["cache_dir"],
        help=f"Chunk cache directory (default: {defaults['cache_dir']}).",
    )
    parser.add_argument(
        "--persist-dir",
        default=defaults["persist_dir"],
        help=f"Vector store directory (default: {defaults['persist_dir']}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=defaults["chunk_size"],
        help=f"Chunk size in characters (default: {defaults['chunk_size']}).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=defaults["chunk_overlap"],
        help=f"Chunk overlap in characters (default: {defaults['chunk_overlap']}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults["batch_size"],
        help=f"Vector store ingestion batch size (default: {defaults['batch_size']}).",
    )
    return parser


def resolve_mode(args: argparse.Namespace) -> str:
    """Return ``"full"`` or ``"incremental"`` based on parsed args.

    When neither --full-rebuild nor --incremental is given, default to
    "full" to preserve the historical behaviour of this script.
    """
    if args.incremental:
        return "incremental"
    return "full"


def print_summary(
    mode: str,
    chunk_count: int,
    changed_count: int,
    deleted_count: int,
    batch_size: int,
    elapsed: float,
) -> None:
    """Print a concise human-readable summary line."""
    print(
        f"mode={mode} chunks={chunk_count} changed={changed_count} "
        f"deleted={deleted_count} batch_size={batch_size} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )


def run(args: argparse.Namespace) -> dict:
    """Execute the rebuild/sync and return a summary dict.

    Heavy imports (index_cache, retriever, embeddings) are deferred to this
    function so that ``import scripts.rebuild_vectorstore`` is cheap and
    does not require the ML stack.
    """
    from src.index_cache import load_or_update_chunks
    from src.retriever import sync_vectorstore, validate_batch_size

    mode = resolve_mode(args)
    batch_size = validate_batch_size(args.batch_size)

    start = time.perf_counter()
    idx = load_or_update_chunks(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        cache_dir=args.cache_dir,
        force_rebuild=(mode == "full"),
    )

    if mode == "incremental":
        full_rebuild = idx.full_rebuild
        changed_sources = idx.changed_sources
        deleted_chunk_ids = idx.deleted_chunk_ids
    else:
        full_rebuild = True
        changed_sources = []
        deleted_chunk_ids = []

    sync_vectorstore(
        chunks=idx.chunks,
        persist_dir=args.persist_dir,
        changed_sources=changed_sources,
        deleted_chunk_ids=deleted_chunk_ids,
        full_rebuild=full_rebuild,
        batch_size=batch_size,
    )
    elapsed = time.perf_counter() - start

    changed_count = len(changed_sources) if mode == "incremental" else len(idx.changed_sources)
    deleted_count = len(deleted_chunk_ids) if mode == "incremental" else 0
    print_summary(
        mode=mode,
        chunk_count=len(idx.chunks),
        changed_count=changed_count,
        deleted_count=deleted_count,
        batch_size=batch_size,
        elapsed=elapsed,
    )
    return {
        "mode": mode,
        "chunk_count": len(idx.chunks),
        "changed_count": changed_count,
        "deleted_count": deleted_count,
        "batch_size": batch_size,
        "elapsed": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    load_local_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
