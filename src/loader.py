# -*- coding: utf-8 -*-
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdfs(pdf_dir: str) -> list[Document]:
    """Load PDFs as page-level Documents with source and page metadata."""
    docs: list[Document] = []
    pdf_paths = sorted(Path(pdf_dir).glob("*.pdf"))

    for pdf_path in pdf_paths:
        with fitz.open(str(pdf_path)) as pdf:
            for page_index, page in enumerate(pdf, start=1):
                text = page.get_text().strip()
                if not text:
                    continue
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_index,
                        },
                    )
                )

    print(f"Loaded {len(docs)} PDF pages from {pdf_dir}")
    return docs


def split_documents(
    docs: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """Split documents into chunks while preserving source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            "\u3002",
            "\uff1f",
            "\uff01",
            "\uff1b",
            "\uff0c",
            " ",
            "",
        ],
    )
    chunks = splitter.split_documents(docs)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    print(f"Split into {len(chunks)} chunks (chunk_size={chunk_size})")
    return chunks
