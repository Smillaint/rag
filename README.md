# Local PDF RAG Assistant

A local Retrieval-Augmented Generation (RAG) project for question answering over PDF documents. It supports PDF parsing, chunking, hybrid retrieval, CrossEncoder reranking, and answer generation through an OpenAI-compatible chat API such as DeepSeek.

## Features

- Loads local PDFs page by page with `source`, `page`, and `chunk_id` metadata.
- Splits long documents into overlapping chunks for stable retrieval.
- Builds a persistent Chroma vector store with multilingual sentence embeddings.
- Combines dense vector search with BM25 keyword retrieval for better recall.
- Uses a CrossEncoder reranker to improve final context precision.
- Generates grounded answers with chunk citations and anti-hallucination prompting.
- Provides a lightweight retrieval evaluation script for measurable project results.

## Architecture

```text
PDF files
  -> PyMuPDF page extraction
  -> Recursive chunking
  -> Chroma vector index + BM25 keyword index
  -> Hybrid candidate retrieval
  -> CrossEncoder reranking
  -> OpenAI-compatible LLM answer generation
```

Core modules:

```text
src/
  loader.py       PDF loading and chunking
  retriever.py    Chroma, BM25, and hybrid retrieval
  reranker.py     CrossEncoder reranking
  generator.py    OpenAI-compatible answer generation
  chain.py        End-to-end RAGPipeline orchestration
  evaluate.py     Retrieval evaluation CLI
main.py           Command-line entry point
examples/         Example evaluation cases
```

## Setup

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put PDF files in the local `data/` directory:

```text
data/
  your-document.pdf
```

`data/` and `vectorstore/` are ignored by Git because they contain local documents and generated indexes.

## API Key

Set a local API key before running generation:

```powershell
$env:DEEPSEEK_API_KEY="your_api_key_here"
```

The code uses the OpenAI Python SDK with an OpenAI-compatible endpoint. By default it targets DeepSeek:

```text
base_url = https://api.deepseek.com
model    = deepseek-chat
```

## Run

Rebuild the vector store after adding or changing PDFs:

```powershell
python main.py --rebuild "What is this document about?"
```

Reuse the existing local vector store:

```powershell
python main.py "How does the document describe privacy protection?"
```

Start interactive mode:

```powershell
python main.py
```

Optional parameters:

```powershell
python main.py --data-dir ./data --vectorstore-dir ./vectorstore --model deepseek-chat "Your question"
```

## Evaluate Retrieval

Create JSONL evaluation cases in `examples/eval_questions.jsonl`:

```jsonl
{"question":"What is the core topic?","expected_sources":["paper.pdf"],"expected_keywords":["retrieval"]}
```

Run retrieval evaluation:

```powershell
python -m src.evaluate --eval-file ./examples/eval_questions.jsonl --skip-rerank
```

Use full reranking when the CrossEncoder model is available locally or can be downloaded:

```powershell
python -m src.evaluate --eval-file ./examples/eval_questions.jsonl
```

The report includes pass rate, source hit, keyword hit, and returned source metadata.

## Notes

- First-time embedding or reranker model loading may require network access.
- The default Hugging Face endpoint is set to `https://hf-mirror.com` for easier model downloads in China.
- Do not commit API keys, local PDFs, or generated vector stores.
