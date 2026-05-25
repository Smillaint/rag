# -*- coding: utf-8 -*-
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SEPARATORS = [
    "\n\n",
    "\n",
    "\u3002",
    "\uff1f",
    "\uff01",
    "\uff1b",
    "\uff0c",
    " ",
    "",
]


def load_pdfs(pdf_dir: str) -> list[Document]:
    """Load PDFs as page-level Documents with source and page metadata."""
    docs: list[Document] = []
    pdf_paths = sorted(Path(pdf_dir).glob("*.pdf"))

    for pdf_path in pdf_paths:
        docs.extend(load_pdf_file(pdf_path))

    print(f"Loaded {len(docs)} PDF pages from {pdf_dir}")
    return docs


def load_pdf_file(pdf_path: str | Path) -> list[Document]:
    """Load a single PDF as page-level Documents."""
    path = Path(pdf_path)
    docs: list[Document] = []
    with fitz.open(str(path)) as pdf:
        for page_index, page in enumerate(pdf, start=1):
            text = page.get_text().strip()
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "page": page_index,
                    },
                )
            )
    return docs


def split_documents(
    docs: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """Split documents into chunks while preserving source metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHUNK_SEPARATORS,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    assign_chunk_metadata(chunks)

    stats = summarize_chunks(chunks)
    print(
        "Split into "
        f"{stats['total_chunks']} chunks "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap}, "
        f"avg_len={stats['avg_length']:.1f}, max_len={stats['max_length']})"
    )
    return chunks


def assign_chunk_metadata(chunks: list[Document]) -> None:
    """Assign stable source/page chunk metadata after splitting or merging chunks."""
    per_page_counts: dict[tuple[str, int | str], int] = {}
    for index, chunk in enumerate(chunks):
        source = str(chunk.metadata.get("source", "unknown"))
        page = chunk.metadata.get("page", "unknown")
        page_key = (source, page)
        chunk_in_page = per_page_counts.get(page_key, 0)
        per_page_counts[page_key] = chunk_in_page + 1

        char_start = chunk.metadata.get("start_index")
        if isinstance(char_start, int):
            chunk.metadata["char_start"] = char_start
            chunk.metadata["char_end"] = char_start + len(chunk.page_content)
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_in_page"] = chunk_in_page
        chunk.metadata["chunk_id"] = f"{source}:p{page}:c{chunk_in_page}"
        chunk.metadata["chunk_length"] = len(chunk.page_content)


def summarize_chunks(chunks: list[Document]) -> dict:
    """Return lightweight chunk statistics for debugging and tuning."""
    lengths = [len(chunk.page_content) for chunk in chunks]
    sources = {chunk.metadata.get("source") for chunk in chunks}
    pages = {
        (chunk.metadata.get("source"), chunk.metadata.get("page"))
        for chunk in chunks
    }

    return {
        "total_chunks": len(chunks),
        "source_count": len(sources),
        "page_count": len(pages),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "avg_length": sum(lengths) / len(lengths) if lengths else 0.0,
    }
