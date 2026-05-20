# -*- coding: utf-8 -*-
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from main import load_local_env
from src.chain import RAGPipeline


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    retrieve_top_k: int = Field(5, ge=1, le=30)
    rerank_top_k: int = Field(3, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
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
        base_url=os.getenv("RAG_BASE_URL", "https://api.deepseek.com"),
        model=os.getenv("RAG_MODEL", "deepseek-chat"),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "64")),
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
    pages = {
        (chunk.metadata.get("source"), chunk.metadata.get("page"))
        for chunk in rag.chunks
    }
    return {
        "chunk_count": len(rag.chunks),
        "page_count": len(pages),
        "source_count": len(sources),
        "sources": sources,
        "model": rag.model,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict:
    rag = get_pipeline()
    return rag.ask_with_sources(
        request.query,
        retrieve_top_k=request.retrieve_top_k,
        rerank_top_k=request.rerank_top_k,
    )
