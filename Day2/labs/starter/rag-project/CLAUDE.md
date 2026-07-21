# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A **Retrieval-Augmented Generation (RAG)** app for question-answering over PDF documents.

**Pipeline**: PDF → chunked → embedded (`nomic-embed-text` via Ollama) → ChromaDB → retrieved (top-4) → answered by `llama3.2:1b` via Ollama

**No AWS credentials or API keys required — runs fully local via Ollama.** `llama3.2:1b` is used (not a larger model) specifically because this runs on classroom laptops with modest specs — see the Samsung GenAI program's established pattern in `day2/prompt_playground` and `day3/interrogation_room`.

## Pre-requisites

Ollama must be running with both models pulled:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2:1b
ollama serve              # if not already running
```

> **Important:** If you have an existing `chroma_db/` folder built with a different embedding model, delete it (or use the app's **Clear Index** button) before first run — different embedding models produce different vector dimensions, and mixing them in one collection will error.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_indexer.py -v

# Run a single test by name
pytest tests/test_indexer.py::test_chunk_document_returns_list -v
```

## Architecture

Two modules in `rag/`:

### `rag/indexer.py` — Document ingestion
- `chunk_document(pdf_path, chunk_size=1000, chunk_overlap=200)` — loads PDF via PyPDFLoader, splits with RecursiveCharacterTextSplitter
- `index_exists(persist_dir)` — checks if the ChromaDB collection at `persist_dir` actually has vectors (not just whether the directory exists)
- `get_chunk_count(persist_dir)` — returns the collection's vector count, for UI status display
- `clear_index(persist_dir)` — deletes the ChromaDB collection via its own API (never touches the SQLite file directly)
- `build_index(pdf_paths, persist_dir="chroma_db", mode="add")` — `mode="replace"` clears first then rebuilds; `mode="add"` appends to an existing collection

### `rag/retriever.py` — QA chain
- `get_qa_chain(persist_dir="chroma_db")` — wires a Chroma retriever (k=4) + `ChatOllama` into a RetrievalQA chain
- `get_answer(question, persist_dir="chroma_db")` — returns `{"answer": str, "sources": list[str]}`

### `chroma_db/` — Runtime vector store
Generated on first `build_index()` call, persists to disk (survives app restarts, unlike an in-memory store). Use the app's **Clear Index** button (or `clear_index()`) to reset — never delete the directory directly while the app is running.

## Configuration

No `.env`, no API key — `rag/indexer.py`'s `EMBEDDING_MODEL` and `rag/retriever.py`'s `GENERATION_MODEL` constants point at local Ollama model names. Change either constant to swap models; both must already be pulled (`ollama pull <name>`) before use.

## Test Design

Tests in `tests/` are unit tests. `test_indexer.py` exercises real chunking logic against `Corporate_HR_Policy_Document.pdf` (included in repo), and mocks `OllamaEmbeddings`/`Chroma` for the index-management functions so they don't require a live Ollama server. `test_retriever.py` mocks `get_qa_chain` entirely to avoid live model calls.
