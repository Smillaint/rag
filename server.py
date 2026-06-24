# -*- coding: utf-8 -*-
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from main import load_local_env
from src.chain import RAGPipeline
from src.loader import list_pdf_collections


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: str | None = Field(None, description="Limit retrieval to one data subdirectory collection.")
    retrieve_top_k: int = Field(5, ge=1, le=30)
    rerank_top_k: int = Field(3, ge=1, le=20)
    include_sources: bool = Field(False, description="Return retrieved chunk previews for debugging.")


class AskResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    trace: dict[str, Any]
    sources: list[dict[str, Any]]


pipeline: RAGPipeline | None = None
PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

KEY_CODE_FILES = {
    "chain": {
        "path": PROJECT_ROOT / "src" / "chain.py",
        "title": "RAGPipeline 编排",
        "group": "Pipeline",
        "summary": "串联 PDF chunk、混合检索、精排、生成、引用和日志。",
    },
    "loader": {
        "path": PROJECT_ROOT / "src" / "loader.py",
        "title": "PDF 解析与元数据",
        "group": "Ingestion",
        "summary": "递归读取 PDF，按 collection/source/page 保留可追溯元数据。",
    },
    "retriever": {
        "path": PROJECT_ROOT / "src" / "retriever.py",
        "title": "Chroma + BM25 检索",
        "group": "Retrieval",
        "summary": "向量检索、BM25 懒加载、collection 过滤和分批建库。",
    },
    "reranker": {
        "path": PROJECT_ROOT / "src" / "reranker.py",
        "title": "CrossEncoder 精排",
        "group": "Ranking",
        "summary": "CrossEncoder 打分并结合词面匹配加权，提升中文技术词稳定性。",
    },
    "generator": {
        "path": PROJECT_ROOT / "src" / "generator.py",
        "title": "答案生成与降级",
        "group": "Generation",
        "summary": "OpenAI 兼容接口生成答案，API 不可用时返回本地抽取式回答。",
    },
    "query_log": {
        "path": PROJECT_ROOT / "src" / "query_log.py",
        "title": "JSONL 调用链日志",
        "group": "Observability",
        "summary": "按日期记录 query、collection、引用、trace 和耗时。",
    },
    "server": {
        "path": PROJECT_ROOT / "server.py",
        "title": "FastAPI 服务",
        "group": "API",
        "summary": "暴露问答、collection、统计、项目进度和代码浏览接口。",
    },
    "main": {
        "path": PROJECT_ROOT / "main.py",
        "title": "CLI 入口",
        "group": "CLI",
        "summary": "支持列出 collection、指定 collection 问答和重建索引。",
    },
}


def build_pipeline() -> RAGPipeline:
    load_local_env()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Please set DEEPSEEK_API_KEY in .env or the shell.")

    return RAGPipeline.from_paths(
        api_key=api_key,
        data_dir=os.getenv("RAG_DATA_DIR", "./data"),
        vectorstore_dir=os.getenv("RAG_VECTORSTORE_DIR", "./vectorstore"),
        cache_dir=os.getenv("RAG_CACHE_DIR", "./.rag_cache"),
        base_url=os.getenv("RAG_BASE_URL", "https://api.deepseek.com"),
        model=os.getenv("RAG_MODEL", "deepseek-chat"),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "120")),
        log_dir=os.getenv("RAG_LOG_DIR", "./logs"),
        vectorstore_batch_size=int(os.getenv("RAG_VECTORSTORE_BATCH_SIZE", "128")),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = build_pipeline()
    yield
    pipeline = None


app = FastAPI(
    title="Local PDF RAG API",
    description="A FastAPI service for local PDF RAG question answering.",
    version="0.1.0",
    lifespan=lifespan,
)

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")


def get_pipeline() -> RAGPipeline:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not ready.")
    return pipeline


@app.get("/health")
def health() -> dict:
    ready = pipeline is not None
    return {"status": "ok" if ready else "loading", "ready": ready}


@app.get("/stats")
def stats() -> dict:
    rag = get_pipeline()
    sources = sorted({str(chunk.metadata.get("source")) for chunk in rag.chunks})
    collections = sorted({str(chunk.metadata.get("collection", "default")) for chunk in rag.chunks})
    pages = {
        (chunk.metadata.get("source"), chunk.metadata.get("page"))
        for chunk in rag.chunks
    }
    return {
        "chunk_count": len(rag.chunks),
        "page_count": len(pages),
        "source_count": len(sources),
        "collections": collections,
        "sources": sources,
        "model": rag.model,
    }


@app.get("/collections")
def collections() -> dict:
    data_dir = os.getenv("RAG_DATA_DIR", "./data")
    return {"collections": list_pdf_collections(data_dir)}


@app.get("/")
def frontend() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_path)


@app.get("/project/progress")
def project_progress() -> dict:
    data_dir = os.getenv("RAG_DATA_DIR", "./data")
    available_collections = list_pdf_collections(data_dir)
    rag_ready = pipeline is not None
    chunk_count = len(pipeline.chunks) if pipeline is not None else 0
    loaded_collections = (
        sorted({str(chunk.metadata.get("collection", "default")) for chunk in pipeline.chunks})
        if pipeline is not None
        else []
    )
    return {
        "name": "Local PDF RAG QA",
        "branch": "codex-rag-collections-large-corpus",
        "status": "ready" if rag_ready else "loading",
        "summary": "本地 PDF RAG 问答系统，支持 collection 隔离、混合检索、CrossEncoder 精排、引用回答和调用链日志。",
        "metrics": {
            "chunks": chunk_count,
            "collections": len(available_collections),
            "code_files": len(KEY_CODE_FILES),
            "vectorstore": "ready" if Path(os.getenv("RAG_VECTORSTORE_DIR", "./vectorstore")).exists() else "missing",
        },
        "collections": [
            {"name": name, "pdf_count": len(files), "loaded": name in loaded_collections}
            for name, files in sorted(available_collections.items())
        ],
        "milestones": [
            {
                "name": "PDF 解析与可追溯 chunk",
                "status": "done",
                "detail": "按页解析 PDF，保留 source_path、collection、page、chunk_id、char_start、char_end。",
            },
            {
                "name": "混合检索",
                "status": "done",
                "detail": "Chroma 向量检索 + BM25 关键词检索，双方都支持 collection 过滤。",
            },
            {
                "name": "CrossEncoder 精排",
                "status": "done",
                "detail": "CrossEncoder 成对打分，并加入词面匹配加权修正中文技术词排序。",
            },
            {
                "name": "大文本适配",
                "status": "done",
                "detail": "Chroma 分批写入，BM25 按 collection 懒加载，降低本地索引压力。",
            },
            {
                "name": "可观测性",
                "status": "done",
                "detail": "每轮问答写入 logs/YYYY-MM-DD.jsonl，返回 trace、timing、rerank_scores。",
            },
            {
                "name": "评测增强",
                "status": "planned",
                "detail": "后续可补 Recall@K、MRR、hybrid/rerank ablation 自动报告。",
            },
        ],
    }


@app.get("/project/code-files")
def project_code_files() -> dict:
    return {
        "files": [
            {
                "key": key,
                "title": meta["title"],
                "group": meta["group"],
                "summary": meta["summary"],
                "path": str(meta["path"].relative_to(PROJECT_ROOT)).replace("\\", "/"),
            }
            for key, meta in KEY_CODE_FILES.items()
        ]
    }


@app.get("/project/code/{file_key}")
def project_code(file_key: str) -> dict:
    meta = KEY_CODE_FILES.get(file_key)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown code file key.")

    path = meta["path"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Code file not found.")

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    return {
        "key": file_key,
        "title": meta["title"],
        "group": meta["group"],
        "summary": meta["summary"],
        "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "line_count": len(lines),
        "content": content,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict:
    rag = get_pipeline()
    result = rag.ask_with_sources(
        request.query,
        retrieve_top_k=request.retrieve_top_k,
        rerank_top_k=request.rerank_top_k,
        collection=request.collection,
    )
    if not request.include_sources:
        result["sources"] = []
    return result
