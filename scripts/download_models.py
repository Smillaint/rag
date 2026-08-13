# -*- coding: utf-8 -*-
"""Manual large-file download for HF models behind hf-mirror.com.

huggingface_hub's HEAD request to hf-mirror.com fails for LFS files in
this environment, so we download them directly with requests and place
them into the HF cache snapshot directory.
"""
import os
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

LARGE_FILES = [
    ("BAAI/bge-m3", "pytorch_model.bin",
     "https://hf-mirror.com/BAAI/bge-m3/resolve/main/pytorch_model.bin"),
    ("BAAI/bge-reranker-v2-m3", "model.safetensors",
     "https://hf-mirror.com/BAAI/bge-reranker-v2-m3/resolve/main/model.safetensors"),
]


def hf_cache_snapshot_dir(repo_id: str) -> Path:
    """Return the HF hub cache snapshot dir for a repo."""
    cache_root = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
    repo_dir = cache_root / ("models--" + repo_id.replace("/", "--"))
    snapshots_dir = repo_dir / "snapshots"
    return snapshots_dir


def ensure_small_files(repo_id: str) -> Path:
    """Download small files via hf_hub_download and return snapshot dir."""
    small_files = [
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "sentence_bert_config.json",
        "modules.json", "config_sentence_transformers.json",
        "sentencepiece.bpe.model", "vocab.txt",
    ]
    snapshot_path = None
    for f in small_files:
        try:
            p = hf_hub_download(repo_id, f)
            if snapshot_path is None:
                snapshot_path = Path(p).parent
        except Exception:
            pass
    if snapshot_path is None:
        snapshots = hf_cache_snapshot_dir(repo_id)
        dirs = list(snapshots.iterdir()) if snapshots.exists() else []
        if dirs:
            snapshot_path = dirs[0]
    return snapshot_path


def download_large_file(repo_id: str, filename: str, url: str) -> Path:
    """Download a large file directly into the HF cache snapshot dir."""
    snapshot_dir = ensure_small_files(repo_id)
    if snapshot_dir is None:
        raise RuntimeError(f"Could not locate snapshot dir for {repo_id}")

    target_path = snapshot_dir / filename

    if target_path.exists():
        size_mb = target_path.stat().st_size / (1024 * 1024)
        print(f"  Already exists: {target_path} ({size_mb:.1f} MB)")
        return target_path

    print(f"  Downloading {filename}")
    print(f"  Target: {target_path}")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024

    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 / total
                    print(f"\r  {pct:.1f}% ({downloaded / (1024*1024):.0f}/{total / (1024*1024):.0f} MB)",
                          end="", flush=True)
    print()
    print(f"  Done: {target_path}")
    return target_path


def main():
    for repo_id, filename, url in LARGE_FILES:
        print(f"\n=== {repo_id} / {filename} ===")
        try:
            download_large_file(repo_id, filename, url)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n=== Verification ===")
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("BAAI/bge-m3", device="cpu")
        emb = m.encode(["test"])
        print(f"  bge-m3 OK, dim={emb.shape[1]}")
    except Exception as e:
        print(f"  bge-m3 FAIL: {type(e).__name__}: {e}")

    try:
        from sentence_transformers import CrossEncoder
        r = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
        score = r.predict([("query", "passage")])
        print(f"  bge-reranker-v2-m3 OK, score={score}")
    except Exception as e:
        print(f"  bge-reranker-v2-m3 FAIL: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
