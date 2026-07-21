from unittest.mock import MagicMock, patch

from rag.retriever import get_answer


def test_get_answer_returns_answer_key():
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "result": "Employees get 20 days of leave.",
        "source_documents": [],
    }
    with patch("rag.retriever.get_qa_chain", return_value=mock_chain):
        result = get_answer("What is the leave policy?")

    assert "answer" in result
    assert result["answer"] == "Employees get 20 days of leave."


def test_get_answer_returns_sources_key():
    mock_chain = MagicMock()
    doc = MagicMock()
    doc.page_content = "chunk text"
    mock_chain.invoke.return_value = {
        "result": "Some answer.",
        "source_documents": [doc],
    }
    with patch("rag.retriever.get_qa_chain", return_value=mock_chain):
        result = get_answer("Any question?")

    assert "sources" in result
    assert len(result["sources"]) == 1
    assert result["sources"][0] == "chunk text"
