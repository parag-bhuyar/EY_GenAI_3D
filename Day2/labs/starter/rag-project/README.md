# RAG PDF Q&A

A local Retrieval-Augmented Generation (RAG) app for question-answering over PDF documents. Upload PDFs, build a persistent vector index, and ask natural-language questions — answers are grounded in the actual content of your documents.

**100% local — no API key, no cost.** Both the embedding model and the answer-generation model run on your own laptop via [Ollama](https://ollama.com).

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│  INDEXING                                               │
│  PDF Files → Chunking → nomic-embed-text (Ollama) → ChromaDB │
├─────────────────────────────────────────────────────────┤
│  QUERYING                                               │
│  Question → Embed → ChromaDB Search → Top-4 Chunks     │
│          → llama3.2:1b (Ollama) → Grounded Answer        │
└─────────────────────────────────────────────────────────┘
```

**Indexing:** Each PDF is split into 1000-character chunks with 200-character overlap, embedded using Ollama's `nomic-embed-text`, and stored in a local ChromaDB vector database (persists to disk — survives app restarts).

**Querying:** The question is embedded with the same model, ChromaDB finds the 4 most similar chunks, and Ollama's `llama3.2:1b` generates an answer grounded in those chunks.

## Project Structure

```
rag-project/
├── app.py                        # Streamlit UI
├── rag/
│   ├── indexer.py                # PDF loading, chunking, ChromaDB index management
│   └── retriever.py              # RetrievalQA chain (ChromaDB + Ollama)
├── tests/
│   ├── test_indexer.py
│   └── test_retriever.py
├── chroma_db/                    # Vector store (auto-created, gitignored)
└── Corporate_HR_Policy_Document.pdf   # Sample PDF for testing
```

## Setup

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com) installed and running, with both models pulled:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:1b
ollama serve   # if not already running
```

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the app — no .env, no API key needed
streamlit run app.py
```

The app opens at **http://localhost:8501**.

## Usage

1. **Upload PDFs** — drag one or more PDFs into the sidebar uploader.
2. **Choose index mode:**
   - *Add to index* — appends new documents to the existing store.
   - *Replace index* — wipes the current store and rebuilds from scratch.
3. **Click Build Index** — chunks and embeds the PDFs into ChromaDB (runs locally, no internet needed after Ollama models are pulled).
4. **Ask a question** — type a question in the main area and click Ask.
5. **View sources** — expand the Sources section to see the exact chunks retrieved from ChromaDB.

## Configuration

| Setting | Default | Where |
|---------|---------|-------|
| Embedding model | `nomic-embed-text` (Ollama) | `rag/indexer.py` (`EMBEDDING_MODEL`) |
| LLM | `llama3.2:1b` (Ollama) | `rag/retriever.py` (`GENERATION_MODEL`) |
| Chunk size | 1000 chars | `rag/indexer.py` |
| Chunk overlap | 200 chars | `rag/indexer.py` |
| Top-k retrieval | 4 chunks | `rag/retriever.py` |
| Vector store path | `chroma_db/` | `app.py` (`PERSIST_DIR`) |

## Running Tests

```bash
pytest tests/ -v
```

`test_indexer.py` runs real chunking against the included sample PDF (mocks `OllamaEmbeddings`/`Chroma` for the index-management tests). `test_retriever.py` mocks the QA chain to avoid needing a live Ollama server for unit tests.

## Key Design Notes

- **ChromaDB is never deleted** — `clear_index` uses ChromaDB's own collection API (`delete_collection()`) rather than deleting the SQLite file on disk. Deleting the file while the chromadb process singleton holds it open causes `SQLITE_READONLY_DBMOVED` errors.
- **`index_exists` checks collection state** — it queries the actual ChromaDB collection count, not just whether the `chroma_db/` directory exists, so it correctly returns `False` after a clear.
- **If you have an existing `chroma_db/` folder built with a different embedding model**, delete it (or use the app's **Clear Index** button) before first run — `nomic-embed-text` uses a different vector dimension than other embedding models, and mixing them in one collection will error.
