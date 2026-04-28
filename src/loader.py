# -*- coding: utf-8 -*-
# src/loader.py

import fitz  # pymupdf
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_pdfs(pdf_dir: str) -> list:
    """加载目录下所有 PDF"""
    docs = []
    for pdf_path in Path(pdf_dir).glob("*.pdf"):
        pdf = fitz.open(str(pdf_path))
        text = ""
        for page in pdf:
            text += page.get_text()
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": pdf_path.name}
            ))
    print(f"共加载 {len(docs)} 个文档")
    return docs


def split_documents(docs: list, chunk_size=512, chunk_overlap=64) -> list:
    """切分文档为 chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"切分后共 {len(chunks)} 个 chunk（chunk_size={chunk_size}）")
    return chunks
