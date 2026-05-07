# -*- coding: utf-8 -*-
from src.loader import load_pdfs, split_documents


docs = load_pdfs("./data")
chunks = split_documents(docs)

if chunks:
    first = chunks[0]
    source = first.metadata.get("source", "unknown")
    page = first.metadata.get("page", "unknown")
    print(f"First chunk source: {source}, page: {page}")
    print(first.page_content[:100])
else:
    print("No chunks found. Put PDF files in the data directory first.")
