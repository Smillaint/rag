# -*- coding: utf-8 -*-
import os
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_local_env(path: str = ".env") -> None:
    from pathlib import Path
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


load_local_env()
from src.index_cache import load_or_update_chunks
from src.retriever import sync_vectorstore

print("Loading chunks...", flush=True)
t = time.perf_counter()
idx = load_or_update_chunks(
    data_dir="./data", chunk_size=900, chunk_overlap=120,
    cache_dir="./.rag_cache", force_rebuild=False,
)
print(f"Loaded {len(idx.chunks)} chunks in {time.perf_counter()-t:.1f}s", flush=True)

print("Rebuilding vectorstore with bge-m3 (1024-dim)...", flush=True)
t = time.perf_counter()
vs = sync_vectorstore(
    chunks=idx.chunks,
    persist_dir="./vectorstore",
    changed_sources=idx.changed_sources,
    deleted_chunk_ids=idx.deleted_chunk_ids,
    full_rebuild=True,
    batch_size=32,
)
print(f"Vectorstore rebuilt in {time.perf_counter()-t:.1f}s", flush=True)
print("DONE", flush=True)
