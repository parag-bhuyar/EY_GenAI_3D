from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA

from rag.indexer import _get_embeddings

GENERATION_MODEL = "llama3.2:1b"


def get_qa_chain(persist_dir: str = "chroma_db"):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=_get_embeddings())
    llm = ChatOllama(model=GENERATION_MODEL, temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
    )


def get_answer(question: str, persist_dir: str = "chroma_db") -> dict:
    chain = get_qa_chain(persist_dir)
    result = chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.page_content for doc in result["source_documents"]],
    }
