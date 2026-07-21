import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.indexer import chunk_document, index_exists, clear_index, build_index

PDF_PATH = "Corporate_HR_Policy_Document.pdf"


def test_chunk_document_returns_list():
    chunks = chunk_document(PDF_PATH)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunk_document_items_have_page_content():
    chunks = chunk_document(PDF_PATH)
    assert all(hasattr(chunk, "page_content") for chunk in chunks)


def test_chunk_document_respects_max_size():
    chunks = chunk_document(PDF_PATH, chunk_size=500)
    assert all(len(chunk.page_content) <= 500 for chunk in chunks)


def test_index_exists_returns_false_for_missing_dir():
    assert index_exists("/nonexistent/path/xyz") is False


def test_index_exists_returns_false_for_empty_dir():
    # ignore_cleanup_errors=True: ChromaDB creates a SQLite file in the temp dir
    # which Windows holds open; the assertion still runs before cleanup.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        assert index_exists(tmpdir) is False


def test_index_exists_returns_false_when_no_collection():
    # A non-empty directory with no ChromaDB collection → False
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        Path(tmpdir, "unrelated.txt").write_text("noise")
        assert index_exists(tmpdir) is False


def test_chunk_document_raises_for_missing_pdf():
    with pytest.raises(FileNotFoundError):
        chunk_document("/nonexistent/file.pdf")


def test_clear_index_does_not_raise_for_missing_dir():
    # clear_index deletes the ChromaDB collection; it must never raise,
    # even when the directory does not exist.
    clear_index("/nonexistent/path/xyz_does_not_exist_abc")


def test_clear_index_leaves_dir_empty_of_collection():
    # After clear_index, index_exists must return False for the same path.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        clear_index(tmpdir)
        assert index_exists(tmpdir) is False


def test_build_index_accepts_list_of_paths():
    """build_index() should accept a list of PDF paths."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        persist = Path(tmpdir) / "db"
        with patch("rag.indexer.OllamaEmbeddings"), \
             patch("rag.indexer.Chroma") as mock_chroma:
            mock_chroma.from_documents.return_value = MagicMock()
            build_index([PDF_PATH], persist_dir=str(persist), mode="replace")
            assert mock_chroma.from_documents.called


def test_build_index_replace_mode_calls_clear_and_from_documents():
    """mode='replace' must call clear_index then Chroma.from_documents."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        persist = Path(tmpdir) / "db"
        with patch("rag.indexer.OllamaEmbeddings"), \
             patch("rag.indexer.Chroma") as mock_chroma, \
             patch("rag.indexer.clear_index") as mock_clear:
            mock_chroma.from_documents.return_value = MagicMock()
            build_index([PDF_PATH], persist_dir=str(persist), mode="replace")
            mock_clear.assert_called_once_with(str(persist))
            assert mock_chroma.from_documents.called


def test_build_index_add_mode_calls_add_documents():
    """mode='add' must call add_documents() when an index already exists."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        persist = Path(tmpdir) / "db"
        mock_vectorstore = MagicMock()
        with patch("rag.indexer.OllamaEmbeddings"), \
             patch("rag.indexer.index_exists", return_value=True), \
             patch("rag.indexer.Chroma") as mock_chroma:
            mock_chroma.return_value = mock_vectorstore
            build_index([PDF_PATH], persist_dir=str(persist), mode="add")
            mock_vectorstore.add_documents.assert_called_once()
