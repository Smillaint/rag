# test_loader.py
from src.loader import load_pdfs, split_documents

docs = load_pdfs('./data')
chunks = split_documents(docs)

if chunks:
    print(f"第一个chunk内容预览：{chunks[0].page_content[:100]}")
else:
    print("?? 没有找到任何chunk，请确认data目录下有PDF文件")
