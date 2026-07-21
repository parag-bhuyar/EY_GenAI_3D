import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from rag.indexer import build_index, index_exists, get_chunk_count, clear_index
from rag.retriever import get_answer

PERSIST_DIR = "chroma_db"

st.set_page_config(page_title="RAG PDF Q&A", layout="wide")
st.title("PDF Question & Answer")
st.caption("🖥️ 100% local — Ollama `nomic-embed-text` (embeddings) + `llama3.2:1b` (answers). No API key, no cost.")


def _rag_diagram(query_active: bool = False) -> str:
    idx_arrow = "#60a5fa"
    q_arrow   = "#a78bfa" if not query_active else "#7c3aed"
    q_glow    = "box-shadow:0 0 12px 3px rgba(124,58,237,0.45);" if query_active else ""
    q_border  = "border:2px solid #7c3aed;" if query_active else ""
    pulse_css = """
      @keyframes pulse {
        0%,100%{opacity:1;transform:scale(1)}
        50%{opacity:.7;transform:scale(1.08)}
      }
    """ if query_active else ""
    pulse_style = "animation:pulse 1.4s ease-in-out infinite;" if query_active else ""

    def node(bg, icon, label, sublabel, extra_style=""):
        return f"""
          <div style="text-align:center;flex-shrink:0;">
            <div style="background:{bg};border-radius:12px;width:58px;height:58px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:26px;margin:0 auto 6px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.10);{extra_style}">
              {icon}
            </div>
            <div style="font-size:11px;font-weight:700;color:#1f2937;line-height:1.3;">{label}</div>
            <div style="font-size:9px;color:#6b7280;margin-top:2px;line-height:1.3;">{sublabel}</div>
          </div>"""

    def arrow(color):
        return f'<div style="color:{color};font-size:22px;font-weight:300;flex-shrink:0;padding:0 2px;margin-bottom:14px;">→</div>'

    idx_nodes = (
        node("#dbeafe", "📄", "PDF Files",  "Uploaded docs") +
        arrow(idx_arrow) +
        node("#fef3c7", "✂️", "Chunking",   "1000-char splits") +
        arrow(idx_arrow) +
        node("#ede9fe", "🔢", "Embed",      "nomic-embed-text (Ollama)") +
        arrow(idx_arrow) +
        node("#d1fae5", "⚡", "ChromaDB",   "Persistent vectors",
             "border:2px solid #34d399;")
    )

    q_node_style = q_glow + q_border + pulse_style
    q_nodes = (
        node("#fce7f3", "❓", "Your Query",    "Natural language",  q_node_style) +
        arrow(q_arrow) +
        node("#ede9fe", "🔢", "Embed Query",   "nomic-embed-text",  q_node_style) +
        arrow(q_arrow) +
        node("#d1fae5", "🔍", "ChromaDB",      "Similarity search", "border:2px solid #34d399;" + q_glow + pulse_style) +
        arrow(q_arrow) +
        node("#fef9c3", "📋", "Top-4 Chunks",  "Closest matches",   q_node_style) +
        arrow(q_arrow) +
        node("#fee2e2", "🤖", "llama3.2:1b",   "Local (Ollama)",    q_node_style) +
        arrow(q_arrow) +
        node("#dcfce7", "💬", "Answer",         "Final response",   q_node_style)
    )

    shared_note = """
      <div style="display:flex;align-items:center;gap:6px;
                  font-size:10px;color:#065f46;margin:0 0 0 134px;padding-left:14px;">
        <span style="font-size:13px;">↕</span>
        <span>ChromaDB is the same persistent store — built during indexing, queried at search time, survives app restarts</span>
      </div>"""

    return f"""
    <style>
      {pulse_css}
      .rag-wrap * {{box-sizing:border-box;}}
    </style>
    <div class="rag-wrap" style="
      font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
      background:linear-gradient(135deg,#f0f4ff 0%,#fdf4ff 100%);
      border:1px solid #e5e7eb;border-radius:16px;padding:22px 20px 18px;
    ">
      <div style="text-align:center;font-size:15px;font-weight:700;color:#111827;
                  margin-bottom:20px;letter-spacing:-0.2px;">
        How RAG Works — Retrieval-Augmented Generation
      </div>

      <!-- Indexing row -->
      <div style="display:flex;align-items:center;margin-bottom:10px;">
        <div style="min-width:120px;flex-shrink:0;font-size:10px;font-weight:700;
                    color:#2563eb;text-transform:uppercase;letter-spacing:1px;
                    padding-right:14px;border-right:2px solid #bfdbfe;text-align:right;
                    line-height:1.5;">
          📥 Build<br/>Index
        </div>
        <div style="flex:1;display:flex;align-items:center;
                    justify-content:space-evenly;padding-left:14px;">
          {idx_nodes}
        </div>
      </div>

      {shared_note}

      <!-- Query row -->
      <div style="display:flex;align-items:center;margin-top:10px;">
        <div style="min-width:120px;flex-shrink:0;font-size:10px;font-weight:700;
                    color:#7c3aed;text-transform:uppercase;letter-spacing:1px;
                    padding-right:14px;border-right:2px solid #ddd6fe;text-align:right;
                    line-height:1.5;">
          🔍 Answer<br/>Query
        </div>
        <div style="flex:1;display:flex;align-items:center;
                    justify-content:space-evenly;padding-left:14px;">
          {q_nodes}
        </div>
      </div>
    </div>"""


def _langchain_explainer() -> str:
    def card(icon, title, body, bg="#f8fafc", border="#e2e8f0"):
        return f"""
        <div style="background:{bg};border:1px solid {border};border-radius:12px;
                    padding:16px 18px;flex:1;min-width:160px;">
          <div style="font-size:22px;margin-bottom:8px;">{icon}</div>
          <div style="font-size:12px;font-weight:700;color:#111827;margin-bottom:6px;">{title}</div>
          <div style="font-size:11px;color:#4b5563;line-height:1.6;">{body}</div>
        </div>"""

    def layer(icon, label, desc, bg, border):
        return f"""
        <div style="display:flex;align-items:flex-start;gap:14px;
                    background:{bg};border-left:4px solid {border};
                    border-radius:0 10px 10px 0;padding:12px 16px;margin-bottom:8px;">
          <div style="font-size:20px;flex-shrink:0;margin-top:1px;">{icon}</div>
          <div>
            <div style="font-size:12px;font-weight:700;color:#111827;">{label}</div>
            <div style="font-size:11px;color:#4b5563;margin-top:3px;line-height:1.6;">{desc}</div>
          </div>
        </div>"""

    def this_app_row(component, lc_class, role):
        return f"""
        <tr>
          <td style="padding:8px 12px;font-size:11px;font-weight:600;color:#1f2937;
                     border-bottom:1px solid #f3f4f6;">{component}</td>
          <td style="padding:8px 12px;font-size:11px;font-family:monospace;color:#6d28d9;
                     border-bottom:1px solid #f3f4f6;">{lc_class}</td>
          <td style="padding:8px 12px;font-size:11px;color:#4b5563;
                     border-bottom:1px solid #f3f4f6;">{role}</td>
        </tr>"""

    return """
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
                background:linear-gradient(135deg,#fafffe 0%,#f0f9ff 100%);
                border:1px solid #e5e7eb;border-radius:16px;padding:28px 24px;">

      <!-- Header -->
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:24px;">
        <div style="background:#fef3c7;border-radius:14px;width:52px;height:52px;
                    display:flex;align-items:center;justify-content:center;font-size:26px;
                    flex-shrink:0;box-shadow:0 2px 8px rgba(0,0,0,0.08);">🦜</div>
        <div>
          <div style="font-size:18px;font-weight:800;color:#111827;letter-spacing:-0.4px;">
            What is LangChain?
          </div>
          <div style="font-size:12px;color:#6b7280;margin-top:3px;">
            The framework that wires together LLMs, embeddings, vector stores, and retrieval chains
          </div>
        </div>
      </div>

      <!-- One-liner -->
      <div style="background:#fffbeb;border:1px solid #fde68a;border-radius:10px;
                  padding:14px 18px;margin-bottom:22px;font-size:12px;color:#92400e;
                  line-height:1.7;">
        <strong>LangChain</strong> is an open-source Python (and JS) framework that provides
        <strong>composable building blocks</strong> for LLM-powered applications — loaders,
        splitters, embeddings, vector stores, chains, and agents — so you don't have to
        wire each piece together from scratch.
      </div>

      <!-- Core concepts cards -->
      <div style="font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;
                  letter-spacing:1px;margin-bottom:12px;">Core Concepts</div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:22px;">
""" + card("📄", "Document Loaders",
           "Read data from PDFs, web pages, databases, etc. and return a list of Document objects with text + metadata.",
           "#eff6ff", "#bfdbfe") + """
""" + card("✂️", "Text Splitters",
           "Break large documents into smaller overlapping chunks so they fit within the LLM's context window.",
           "#fefce8", "#fde68a") + """
""" + card("🔢", "Embeddings",
           "Convert text into high-dimensional numeric vectors. Semantically similar text lands close together in vector space.",
           "#f5f3ff", "#ddd6fe") + """
""" + card("⚡", "Vector Stores",
           "Databases that store and index embedding vectors. Similarity search returns the nearest neighbours to a query vector.",
           "#f0fdf4", "#bbf7d0") + """
""" + card("⛓️", "Chains",
           "Pre-built sequences of steps — e.g. RetrievalQA: retrieve relevant chunks → format a prompt → call the LLM → parse output.",
           "#fff1f2", "#fecdd3") + """
      </div>

      <!-- Layer diagram -->
      <div style="font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;
                  letter-spacing:1px;margin-bottom:12px;">LangChain Layer by Layer</div>
""" + layer("🦜", "langchain-core",
            "Foundation — base classes for Runnables, Documents, Messages, and the LCEL pipe syntax (chain1 | chain2).",
            "#f0f9ff", "#38bdf8") + \
      layer("📦", "langchain (orchestration)",
            "Pre-built chains and agents: RetrievalQA, ConversationalRetrievalChain, SQL Agent, etc.",
            "#fdf4ff", "#c084fc") + \
      layer("🔌", "langchain-community / partner packages",
            "Integrations: OpenAI, Anthropic, FAISS, Pinecone, HuggingFace, AWS, and 300+ others — each a separate pip package.",
            "#f0fdf4", "#4ade80") + \
      layer("🛠️", "LangSmith (optional)",
            "Observability platform for tracing, evaluating, and debugging chains in production.",
            "#fffbeb", "#fbbf24") + """

      <!-- How this app uses LangChain -->
      <div style="font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;
                  letter-spacing:1px;margin:22px 0 12px;">How This App Uses LangChain</div>
      <table style="width:100%;border-collapse:collapse;background:#fff;border-radius:10px;
                    overflow:hidden;border:1px solid #e5e7eb;font-size:11px;">
        <thead>
          <tr style="background:#f9fafb;">
            <th style="text-align:left;padding:10px 12px;font-size:10px;font-weight:700;
                       color:#6b7280;text-transform:uppercase;letter-spacing:0.8px;
                       border-bottom:1px solid #e5e7eb;">What</th>
            <th style="text-align:left;padding:10px 12px;font-size:10px;font-weight:700;
                       color:#6b7280;text-transform:uppercase;letter-spacing:0.8px;
                       border-bottom:1px solid #e5e7eb;">LangChain Class</th>
            <th style="text-align:left;padding:10px 12px;font-size:10px;font-weight:700;
                       color:#6b7280;text-transform:uppercase;letter-spacing:0.8px;
                       border-bottom:1px solid #e5e7eb;">Role</th>
          </tr>
        </thead>
        <tbody>
""" + this_app_row("Load PDF", "PyPDFLoader", "Reads each PDF page into a Document object") + \
      this_app_row("Split text", "RecursiveCharacterTextSplitter", "Splits into 1000-char chunks, 200-char overlap") + \
      this_app_row("Embed chunks", "OllamaEmbeddings (nomic-embed-text)", "Converts each chunk to a vector — runs 100% locally via Ollama") + \
      this_app_row("Store vectors", "Chroma (langchain-chroma)", "Persists vectors to disk in chroma_db/ — survives app restarts") + \
      this_app_row("Retrieve", "Chroma retriever (k=4)", "Similarity search, returns top-4 chunks") + \
      this_app_row("Answer", "RetrievalQA chain + llama3.2:1b", "Stuffs chunks into prompt, calls Ollama locally — no API key, no cost") + """
        </tbody>
      </table>

      <div style="margin-top:16px;font-size:10px;color:#9ca3af;text-align:center;">
        LangChain v0.3 · langchain-chroma · langchain-ollama · 100% local
      </div>
    </div>"""


# ── Tabs ──────────────────────────────────────────────────────────────────
tab_rag, tab_lc = st.tabs(["📊 How RAG Works", "🦜 What is LangChain?"])

with tab_rag:
    query_active = st.session_state.get("query_running", False)
    components.html(_rag_diagram(query_active), height=300)

with tab_lc:
    components.html(_langchain_explainer(), height=680)

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.success("🖥️ Local Ollama — no API key needed")

    st.divider()
    st.header("Index Management")

    uploaded_files = st.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True
    )

    mode_label = st.radio("Index mode", options=["Add to index", "Replace index"])
    mode = "replace" if mode_label == "Replace index" else "add"

    if st.button("Build Index", disabled=not uploaded_files):
        with st.spinner("Indexing… (embedding runs locally via Ollama)"):
            with tempfile.TemporaryDirectory() as tmpdir:
                all_paths = []
                for f in uploaded_files:
                    tmp_path = Path(tmpdir) / f.name
                    tmp_path.write_bytes(f.read())
                    all_paths.append(str(tmp_path))

                try:
                    build_index(all_paths, persist_dir=PERSIST_DIR, mode=mode)
                    st.session_state.setdefault("session_files", []).extend(
                        Path(p).name for p in all_paths
                    )
                    if mode == "replace":
                        st.session_state["session_files"] = [Path(p).name for p in all_paths]
                    st.success(f"Indexed {len(all_paths)} file(s).")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")

    # Index status
    st.divider()
    if index_exists(PERSIST_DIR):
        chunk_count = get_chunk_count(PERSIST_DIR)
        st.success(f"Index ready — {chunk_count} chunks")
        files = st.session_state.get("session_files", [])
        if files:
            st.caption("This session: " + ", ".join(files))
    else:
        st.info("No index yet")

    # Clear index
    st.divider()
    if st.button("Clear Index"):
        st.session_state["confirm_clear"] = True

    if st.session_state.get("confirm_clear"):
        st.warning("Clear the saved index? This cannot be undone.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Confirm", key="confirm_yes"):
                clear_index(PERSIST_DIR)
                st.session_state.pop("session_files", None)
                st.session_state.pop("confirm_clear", None)
                st.rerun()
        with c2:
            if st.button("Cancel", key="confirm_no"):
                st.session_state.pop("confirm_clear", None)
                st.rerun()

# ── Main area: Q&A ────────────────────────────────────────────────────────
if not index_exists(PERSIST_DIR):
    st.info("Upload and index PDFs using the sidebar to get started.")
else:
    question = st.text_input("Ask a question about your documents")

    if st.button("Ask", disabled=not question.strip()):
        st.session_state["query_running"] = True
        st.session_state["last_question"] = question
        st.rerun()

    if st.session_state.get("query_running"):
        st.session_state["query_running"] = False
        with st.spinner("Searching ChromaDB and generating answer locally…"):
            try:
                result = get_answer(
                    st.session_state["last_question"],
                    persist_dir=PERSIST_DIR,
                )
                st.markdown("### Answer")
                st.markdown(result["answer"])
                with st.expander("Sources — chunks retrieved from ChromaDB"):
                    for i, chunk in enumerate(result["sources"], 1):
                        st.markdown(f"**Chunk {i}**")
                        st.text(chunk)
            except Exception as exc:
                st.error(f"Error: {exc}")
