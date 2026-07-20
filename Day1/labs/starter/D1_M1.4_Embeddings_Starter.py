"""
D1 · M1.4 — Embeddings & Semantic Search Foundations
In-memory semantic search over a small Meridian Retail Bank knowledge base: embed a
set of banking passages, then retrieve the most relevant ones for a customer query
by comparing meaning, not keywords.

Deliverable (per SYLLABUS_TRACE.md): a mini semantic-search prototype. This file IS
that prototype — deliberately in-memory, no vector database yet (that's Day 2's
M2.2) — the goal today is understanding what a vector DB will later do FOR you.

**This file contains fully working code, not TODOs** — same read-and-understand
format as every module so far. Read every block before running it.

Run directly for a quick manual check:
    python D1_M1.4_Embeddings_Starter.py
"""
import json
import math
import os
from pathlib import Path

from openai import OpenAI

assert os.environ.get("OPENAI_API_KEY"), (
    "OPENAI_API_KEY is not set. Set it before continuing — never hard-code a key in this file."
)

client = OpenAI()
MODEL_EMBED = "text-embedding-3-small"  # cheap, fast — this programme's default embedding model

_DAY1_DIR = Path(__file__).resolve().parents[2]
PASSAGES_PATH = _DAY1_DIR / "data" / "D1_M1.4_banking_passages.json"
QUERIES_PATH = _DAY1_DIR / "data" / "D1_M1.4_fixed_queries.json"


def get_embedding(text: str) -> list:
    """
    Calls the OpenAI embeddings endpoint and returns the embedding as a plain list
    of floats. This is the "meaning becomes geometry" step — every call turns text
    into a fixed-length vector (1536 numbers for text-embedding-3-small) positioned
    in a space where semantically similar text ends up geometrically close.
    """
    response = client.embeddings.create(model=MODEL_EMBED, input=text)
    return response.data[0].embedding


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """
    Cosine similarity, implemented by hand (no numpy) so the actual math is visible:
    the dot product of two vectors, divided by the product of their magnitudes.
    Result ranges from -1 (opposite meaning) to 1 (identical meaning); in practice,
    real embeddings for unrelated text usually land somewhere around 0.0-0.3, and
    genuinely related text often lands well above 0.5 — there's no universal
    "similar enough" cutoff, it's always relative to your other candidates.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


def load_passages() -> list:
    with open(PASSAGES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_fixed_queries() -> list:
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Task 1 — Warm-up: three sentences, one query, watch the ranking make sense
#
# What this code does: embeds three unrelated-ish sentences and one banking
# query, then ranks the three sentences by cosine similarity to the query.
#
# Why we do this: this is the smallest possible version of "meaning becomes
# geometry" — before searching 10 real passages, watch it work on 3 sentences
# where the right answer is obvious, so you trust the mechanism before relying
# on it at scale.
# ---------------------------------------------------------------------------
WARMUP_SENTENCES = [
    "The cat sat quietly on the windowsill all afternoon.",
    "A dog ran happily around the park chasing a ball.",
    "Meridian Retail Bank increased its savings account interest rate this quarter.",
]
WARMUP_QUERY = "What is the current interest rate on savings accounts?"


def run_warmup_demo():
    print("=" * 72)
    print("TASK 1 — WARM-UP: ranking 3 sentences against 1 query")
    print("=" * 72)
    query_vec = get_embedding(WARMUP_QUERY)
    ranked = []
    for sentence in WARMUP_SENTENCES:
        vec = get_embedding(sentence)
        score = cosine_similarity(query_vec, vec)
        ranked.append({"sentence": sentence, "score": score})

    ranked.sort(key=lambda r: r["score"], reverse=True)
    for r in ranked:
        print(f"  {r['score']:.4f}  |  {r['sentence']}")
    return ranked


# ---------------------------------------------------------------------------
# Task 2 — Embed the real knowledge base
#
# What this code does: embeds all 10 Meridian Retail Bank passages once and
# caches the vectors in memory (a plain Python list of dicts).
#
# Why we do this: this is exactly what a vector database does at scale
# (embed once, store, reuse for many searches) — except here it's a Python
# list instead of Chroma/Qdrant, and it lives only as long as this script
# runs. Day 2's M2.2 replaces this in-memory cache with a real persistent
# vector store; the underlying idea (embed once, compare many times) doesn't
# change, only the storage layer does.
# ---------------------------------------------------------------------------
def build_passage_index(passages: list) -> list:
    index = []
    for p in passages:
        vec = get_embedding(p["text"])
        index.append({"id": p["id"], "text": p["text"], "embedding": vec})
    return index


# ---------------------------------------------------------------------------
# Task 3 — Semantic search: the actual retrieval function
#
# What this code does: embeds the incoming query, compares it against every
# cached passage embedding with cosine_similarity, and returns the top_k
# highest-scoring passages.
#
# Why we do this: this IS the module's Deliverable — "a mini semantic-search
# prototype." Brute-force comparison against every passage is completely fine
# at 10 passages; it would NOT be fine at 10 million, which is exactly why
# production systems use an indexed vector store (HNSW, in Day 2) instead of
# comparing against every single stored vector one at a time.
# ---------------------------------------------------------------------------
def semantic_search(query: str, index: list, top_k: int = 3) -> list:
    query_vec = get_embedding(query)
    scored = [
        {"id": entry["id"], "text": entry["text"], "score": cosine_similarity(query_vec, entry["embedding"])}
        for entry in index
    ]
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Task 4 — Semantic vs. keyword search
#
# What this code does: a deliberately naive keyword search (counts shared
# words between the query and each passage) run against the same fixed
# queries as semantic_search, so you can compare them directly.
#
# Why we do this: "semantic search understands meaning, keyword search
# doesn't" is a claim. Running both against the same real queries turns it
# into something you watched happen — especially query #4 ("someone used my
# account without my permission"), which shares almost no words with the
# fraud passage it should match.
# ---------------------------------------------------------------------------
def keyword_search(query: str, passages: list, top_k: int = 3) -> list:
    query_words = set(query.lower().split())
    scored = []
    for p in passages:
        passage_words = set(p["text"].lower().split())
        overlap = len(query_words & passage_words)
        scored.append({"id": p["id"], "text": p["text"], "score": overlap})
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_k]


def run_semantic_vs_keyword_comparison(passages: list, index: list, fixed_queries: list):
    print("\n" + "=" * 72)
    print("TASK 4 — SEMANTIC vs. KEYWORD search, same fixed queries")
    print("=" * 72)
    comparison = []
    for q in fixed_queries:
        sem_top = semantic_search(q["query"], index, top_k=1)[0]
        kw_top = keyword_search(q["query"], passages, top_k=1)[0]
        sem_correct = sem_top["id"] == q["expected_top_id"]
        kw_correct = kw_top["id"] == q["expected_top_id"]
        comparison.append({
            "query": q["query"], "expected_top_id": q["expected_top_id"],
            "semantic_top_id": sem_top["id"], "semantic_correct": sem_correct,
            "keyword_top_id": kw_top["id"], "keyword_correct": kw_correct,
        })
        print(f"\n\"{q['query']}\"")
        print(f"  semantic -> id {sem_top['id']}  ({'correct' if sem_correct else 'WRONG'})")
        print(f"  keyword  -> id {kw_top['id']}  ({'correct' if kw_correct else 'WRONG'})")
    return comparison


# ---------------------------------------------------------------------------
# Task 5 — Run the fixed query set through semantic search (the graded part)
#
# What this code does: runs all 5 fixed queries through semantic_search and
# checks whether the #1 result matches each query's expected_top_id.
#
# Why we do this: this IS the module's Expected Output per the programme
# syllabus — "a working top-k semantic search over a small document set" —
# graded on retrieval correctness for a fixed query set, exactly as named in
# the session-plan's Assessment wording.
# ---------------------------------------------------------------------------
def run_fixed_query_evaluation(index: list, fixed_queries: list) -> list:
    print("\n" + "=" * 72)
    print("TASK 5 — FIXED QUERY EVALUATION (graded)")
    print("=" * 72)
    report = []
    for q in fixed_queries:
        results = semantic_search(q["query"], index, top_k=3)
        top_id = results[0]["id"]
        correct = top_id == q["expected_top_id"]
        report.append({
            "query": q["query"], "expected_top_id": q["expected_top_id"],
            "retrieved_top_id": top_id, "top_3_ids": [r["id"] for r in results], "pass": correct,
        })
        marker = "PASS" if correct else "FAIL"
        print(f"\n\"{q['query']}\"")
        print(f"  top-3 ids: {[r['id'] for r in results]}  (expected #1: {q['expected_top_id']})  [{marker}]")
    return report


# ---------------------------------------------------------------------------
# Task 6 — Stretch (Diamond tier): chunking's effect on retrieval quality
#
# What this code does: embeds ONE combined passage that mixes two unrelated
# topics (savings rate + loan eligibility) as a single block, then embeds the
# same content split into two separate chunks. Compares how well a
# loan-specific query matches the combined version vs. the best matching
# chunk.
#
# Why we do this: "chunking affects retrieval quality" is an abstract claim
# until you see a diluted, multi-topic embedding score lower on a specific
# query than a focused, single-topic chunk does.
#
# Only do this if you've finished Tasks 1-5 with time to spare.
# ---------------------------------------------------------------------------
COMBINED_UNCHUNKED_PASSAGE = (
    "Meridian Retail Bank's standard savings account currently earns 3.2% annual "
    "interest, credited monthly, reviewed quarterly by the treasury desk. Separately, "
    "personal loan applications require proof of income, a valid ID, and a minimum "
    "credit score of 650, with approved loans disbursed within 2 business days of signing."
)
CHUNK_SAVINGS = (
    "Meridian Retail Bank's standard savings account currently earns 3.2% annual "
    "interest, credited monthly, reviewed quarterly by the treasury desk."
)
CHUNK_LOAN = (
    "Personal loan applications at Meridian Retail Bank require proof of income, a "
    "valid ID, and a minimum credit score of 650, with approved loans disbursed "
    "within 2 business days of signing."
)
CHUNKING_QUERY = "What credit score do I need for a personal loan?"


def run_chunking_experiment():
    print("\n" + "=" * 72)
    print("TASK 6 (STRETCH) — chunking's effect on retrieval quality")
    print("=" * 72)
    query_vec = get_embedding(CHUNKING_QUERY)
    combined_vec = get_embedding(COMBINED_UNCHUNKED_PASSAGE)
    savings_vec = get_embedding(CHUNK_SAVINGS)
    loan_vec = get_embedding(CHUNK_LOAN)

    combined_score = cosine_similarity(query_vec, combined_vec)
    savings_score = cosine_similarity(query_vec, savings_vec)
    loan_score = cosine_similarity(query_vec, loan_vec)
    best_chunk_score = max(savings_score, loan_score)

    print(f"  Unchunked (both topics combined):  {combined_score:.4f}")
    print(f"  Chunked   (savings-only chunk):     {savings_score:.4f}")
    print(f"  Chunked   (loan-only chunk):        {loan_score:.4f}   <- should be the highest")
    print(f"\n  Best chunk score ({best_chunk_score:.4f}) vs. unchunked score ({combined_score:.4f}): "
          f"chunking {'helped' if best_chunk_score > combined_score else 'did not help'} here.")

    return {
        "query": CHUNKING_QUERY,
        "unchunked_score": combined_score,
        "chunk_savings_score": savings_score,
        "chunk_loan_score": loan_score,
        "chunking_helped": best_chunk_score > combined_score,
    }


if __name__ == "__main__":
    passages = load_passages()
    fixed_queries = load_fixed_queries()

    warmup_ranking = run_warmup_demo()

    print("\nBuilding the in-memory passage index (10 embedding calls)...")
    passage_index = build_passage_index(passages)
    print(f"Indexed {len(passage_index)} passages.")

    semantic_vs_keyword = run_semantic_vs_keyword_comparison(passages, passage_index, fixed_queries)
    evaluation_report = run_fixed_query_evaluation(passage_index, fixed_queries)

    # ---------------------------------------------------------------------
    # Task 7 — Reflect (short, guided — not an essay). Answer in your own
    # words, referencing your ACTUAL output above, in 1-2 sentences each.
    # ---------------------------------------------------------------------
    reflection = """
    REPLACE THIS PLACEHOLDER with your own answers to:
    1. Looking at Task 4, which fixed query showed the biggest gap between semantic
       and keyword search, and why do you think that happened?
    2. Looking at Task 5's report, did every query retrieve the expected passage at
       #1? If not, look at the actual top-3 ids — was the right passage still in
       there, just not ranked first?
    3. (If you ran Task 6) What did the chunking experiment show about combining two
       unrelated topics into one embedded passage?
    """

    # Only run this if you have time left after Tasks 1-5 (optional, Diamond tier).
    chunking_results = {}  # leave empty if skipping the stretch task
    # chunking_results = run_chunking_experiment()

    results = {
        "warmup_ranking": warmup_ranking,
        "semantic_vs_keyword": semantic_vs_keyword,
        "evaluation_report": evaluation_report,
        "reflection": reflection,
        "chunking_results": chunking_results,
    }

    with open("m1_4_search_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved m1_4_search_results.json —", len(json.dumps(results)), "bytes")
