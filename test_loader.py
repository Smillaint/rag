# -*- coding: utf-8 -*-
from src.loader import load_pdfs, split_documents, summarize_chunks


docs = load_pdfs("./data")
chunks = split_documents(docs)

if chunks:
    first = chunks[0]
    source = first.metadata.get("source", "unknown")
    page = first.metadata.get("page", "unknown")
    print(f"Chunk stats: {summarize_chunks(chunks)}")
    print(f"First chunk source: {source}, page: {page}")
    print(f"First chunk metadata: {first.metadata}")
    print(first.page_content[:100])
else:
    print("No chunks found. Put PDF files in the data directory first.")
