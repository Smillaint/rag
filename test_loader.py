# -*- coding: utf-8 -*-
from src.loader import load_pdfs, split_documents


docs = load_pdfs("./data")
chunks = split_documents(docs)

if chunks:
    first = chunks[0]
    source = first.metadata.get("source", "未知")
    page = first.metadata.get("page", "未知页")
    print(f"第一个 chunk 来源: {source}, page: {page}")
    print(first.page_content[:100])
else:
    print("没有找到任何 chunk，请确认 data 目录下有 PDF 文件")
