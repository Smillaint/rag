# -*- coding: utf-8 -*-
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
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
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "64")),
        log_dir=os.getenv("RAG_LOG_DIR", "./logs"),
        vectorstore_batch_size=int(os.getenv("RAG_VECTORSTORE_BATCH_SIZE", "256")),
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


@app.get("/collections")
def collections() -> dict:
    data_dir = os.getenv("RAG_DATA_DIR", "./data")
    return {"collections": list_pdf_collections(data_dir)}
    return {
        "chunk_count": len(rag.chunks),
        "page_count": len(pages),
        "source_count": len(sources),
        "collections": collections,
        "sources": sources,
        "model": rag.model,
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
