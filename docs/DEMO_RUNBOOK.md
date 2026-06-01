# Demo Runbook

## 1. Collections

Subdirectories under `data/` are retrieval collections:

```text
data/
  algorithm/
  netsec_lab/
  pcb/
  tcpip/
```

Each chunk records `collection`, `source_path`, `source`, `page`, `chunk_id`, `char_start`, and `char_end`.

List available collections:

```powershell
.\.venv\Scripts\python.exe main.py --list-collections
```

Expected output:

```text
Available collections:
  - algorithm: 8 PDF(s)
  - netsec_lab: 6 PDF(s)
  - pcb: 1 PDF(s)
  - tcpip: 3 PDF(s)
Use --collection <name> to ask within a specific collection.
```

## 2. Build The Index

After moving PDFs between folders, rebuild once:

```powershell
.\.venv\Scripts\python.exe main.py --rebuild `
  --chunk-size 900 `
  --chunk-overlap 120 `
  --vectorstore-batch-size 128 `
  --collection algorithm `
  "Dijkstra算法适合解决什么问题？"
```

Do not use `--rebuild` during a live demo unless PDFs or chunk parameters changed.

## 3. Command-Line QA

Ask within a specific collection:

```powershell
.\.venv\Scripts\python.exe main.py --collection algorithm "Dijkstra算法适合解决什么问题？"
.\.venv\Scripts\python.exe main.py --collection tcpip "TCP连接建立的三次握手过程是什么？"
.\.venv\Scripts\python.exe main.py --collection netsec_lab "ARP缓存中毒实验的目的是什么？"
.\.venv\Scripts\python.exe main.py --collection pcb "PCB设计中布线有什么注意事项？"
```

If `--collection` is omitted, the system searches all collections and prints a warning.

## 4. FastAPI QA

Start the service:

```powershell
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Useful endpoints:

```text
GET  /health
GET  /stats
GET  /collections
POST /ask
```

Example `/ask` body:

```json
{
  "query": "Dijkstra算法适合解决什么问题？",
  "collection": "algorithm",
  "retrieve_top_k": 5,
  "rerank_top_k": 3,
  "include_sources": false
}
```

The response focuses on `answer`, `citations`, and `trace`. Set `include_sources` to `true` only when debugging retrieved chunk previews.

## 5. Query Logs

Every CLI/API question appends one JSONL record to:

```text
logs/YYYY-MM-DD.jsonl
```

The log contains query, collection, top-k settings, answer preview, citations, rerank scores, generation mode, and timing. It intentionally does not store API keys.

## 6. Smoke Scripts

Manual smoke checks live under `scripts/`; automated tests live under `tests/`.

```powershell
.\.venv\Scripts\python.exe scripts\smoke_loader.py
.\.venv\Scripts\python.exe scripts\smoke_generator.py
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```
