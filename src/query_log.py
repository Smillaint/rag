# -*- coding: utf-8 -*-
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def write_query_log(
    log_dir: str,
    *,
    query: str,
    collection: str | None,
    retrieve_top_k: int,
    rerank_top_k: int,
    result: dict[str, Any],
) -> Path:
    """Append one JSONL call-chain record to logs/YYYY-MM-DD.jsonl."""
    now = datetime.now().astimezone()
    root = Path(log_dir)
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / f"{now.date().isoformat()}.jsonl"

    record = {
        "timestamp": now.isoformat(timespec="seconds"),
        "query": query,
        "collection": collection,
        "retrieve_top_k": retrieve_top_k,
        "rerank_top_k": rerank_top_k,
        "answer_preview": result.get("answer", "")[:240],
        "citations": result.get("citations", []),
        "trace": result.get("trace", {}),
    }
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return log_path
