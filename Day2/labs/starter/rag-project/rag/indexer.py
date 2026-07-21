from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

EMBEDDING_MODEL = "nomic-embed-text"


def chunk_document(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def _get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def index_exists(persist_dir: str) -> bool:
    """True only if persist_dir exists AND its Chroma collection actually has vectors."""
    if not Path(persist_dir).is_dir():
        return False
    try:
        vs = Chroma(persist_directory=persist_dir, embedding_function=_get_embeddings())
        return vs._collection.count() > 0
    except Exception:
        return False


def get_chunk_count(persist_dir: str) -> int:
    if not Path(persist_dir).is_dir():
        return 0
    try:
        vs = Chroma(persist_directory=persist_dir, embedding_function=_get_embeddings())
        return vs._collection.count()
    except Exception:
        return 0


def clear_index(persist_dir: str) -> None:
    """Deletes the ChromaDB collection via its own API — never deletes the SQLite
    file on disk directly, since the chromadb process singleton holding it open
    causes SQLITE_READONLY_DBMOVED errors if the file disappears underneath it."""
    try:
        vs = Chroma(persist_directory=persist_dir, embedding_function=_get_embeddings())
        vs.delete_collection()
    except Exception:
        pass


def build_index(pdf_paths: "str | list[str]", persist_dir: str = "chroma_db", mode: str = "add") -> Chroma:
    """Build or update a persistent Chroma index from PDF paths.
    mode='replace' clears any existing collection first; mode='add' appends."""
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    all_chunks = []
    for path in pdf_paths:
        all_chunks.extend(chunk_document(path))

    embeddings = _get_embeddings()

    if mode == "replace":
        clear_index(persist_dir)
        return Chroma.from_documents(all_chunks, embeddings, persist_directory=persist_dir)

    if index_exists(persist_dir):
        vs = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        vs.add_documents(all_chunks)
        return vs

    return Chroma.from_documents(all_chunks, embeddings, persist_directory=persist_dir)
