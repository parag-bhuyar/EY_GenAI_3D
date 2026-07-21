"""
Autograder for D2 · M2.3 + M2.4 — Hybrid Search, Retrieval, Re-ranking & Evaluation.

Checks the two deliverables SYLLABUS_TRACE.md names for these modules:
  M2.3 — "End-to-end RAG pipeline with citations"
  M2.4 — "RAG evaluation report" (the traced-run half is covered by the timing figures
         the notebook records for each pipeline stage)

Reads the results file the notebook saves rather than re-running the notebook live —
same pattern as every other module's autograder.

Run this from anywhere inside the repo, after running EVERY cell of
Day2/labs/starter/D2_M2.3_M2.4_Hybrid_Rerank_Starter.ipynb in order (the last cell
saves m2_3_m2_4_results.json):

    python Day2/autograder/D2_M2.3_M2.4_Hybrid_Rerank_Autograder.py

Pass criteria:
  1. Keyword-only search FAILED to put the correct article first — the bug is reproduced,
     not skipped. (Deterministic: BM25 over the shipped corpus always does this.)
  2. Metadata filtering removed exactly the articles that are genuinely out of segment or
     out of region, checked against the reference data — not just "the pool got smaller".
  3. Retrieval fetched k=20 candidates and re-ranking returned 5, and every re-ranked id
     is actually one of the retrieved candidates — a real check that re-ranking operated
     on the retrieved set instead of some other list.
  4. Fusion + re-ranking recovered the correct article: id 1 is in the final 5 documents.
  5. All five phrasings of the missing-salary question were answered WITH citations
     (non-empty source ids) — the M2.3 deliverable.
  6. The pipeline did not invent an answer to an out-of-scope question.
  7. An evaluation was run over all 10 eval queries for BOTH pipelines, and hybrid +
     re-ranking is not worse than semantic-only on Recall@1 — the M2.4 deliverable.

On pass, writes Day2/submissions/D2_M2.3.json.
Stretch/Diamond = Recall@1 of 100% AND a clean refusal on the out-of-scope question.
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_shared"))
from autograder_utils import fail, write_completion  # noqa: E402

DAY = 2
MODULE_ID = "M2.3"
RESULTS_FILENAME = "m2_3_m2_4_results.json"
ARTICLES_FILENAME = "D2_M2.3_meridian_articles.json"
EVAL_FILENAME = "D2_M2.3_eval_queries.json"

GOLD_ID = 1                 # "Salary credit delayed - employer NEFT batch not settled"
EXPECTED_K_RETRIEVE = 20
EXPECTED_K_FINAL = 5
EXPECTED_EVAL_QUERIES = 10
CUSTOMER = {"segment": "retail", "region": "IN"}


def _repo_root():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def find_file(filename):
    matches = glob.glob(os.path.join(_repo_root(), "**", filename), recursive=True)
    return matches[0] if matches else None


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_reference_articles():
    path = os.path.join(_repo_root(), "Day2", "data", ARTICLES_FILENAME)
    return load_json(path)


def check_filter(block, articles, errors):
    """The filter must remove exactly the articles that genuinely fail the customer's
    segment/region — computed from the reference data, not taken on trust."""
    if not isinstance(block, dict):
        errors.append("results file has no 'filter' block — re-run the metadata filter cell in Part 3.")
        return

    should_remove = sorted(
        a["id"] for a in articles
        if a["segment"] != CUSTOMER["segment"] or a["region"] != CUSTOMER["region"]
    )
    removed = sorted(int(i) for i in block.get("removed_ids", []))

    if removed != should_remove:
        missed = sorted(set(should_remove) - set(removed))
        extra = sorted(set(removed) - set(should_remove))
        detail = []
        if missed:
            detail.append(f"left in the pool but should have been filtered out: {missed}")
        if extra:
            detail.append(f"filtered out but should have stayed: {extra}")
        errors.append(
            "the metadata filter is not restricting results correctly — " + "; ".join(detail)
        )

    if block.get("pool_after") != len(articles) - len(should_remove):
        errors.append(
            f"filtered pool size is {block.get('pool_after')!r}, "
            f"expected {len(articles) - len(should_remove)}."
        )


def check_retrieval(block, errors):
    if not isinstance(block, dict):
        errors.append("results file has no 'retrieval' block — re-run Part 3.")
        return

    candidates = [int(i) for i in block.get("candidate_ids", [])]
    reranked = [int(i) for i in block.get("reranked_ids", [])]

    if len(candidates) != EXPECTED_K_RETRIEVE:
        errors.append(
            f"retrieval returned {len(candidates)} candidates, expected {EXPECTED_K_RETRIEVE}. "
            "The whole point of this module is retrieving wider than you show the model."
        )
    if len(reranked) != EXPECTED_K_FINAL:
        errors.append(f"re-ranking returned {len(reranked)} documents, expected {EXPECTED_K_FINAL}.")

    stray = [i for i in reranked if i not in set(candidates)]
    if stray:
        errors.append(
            f"re-ranked ids {stray} were never in the retrieved candidate list — "
            "the re-ranker is not operating on the retrieved set."
        )

    if reranked and GOLD_ID not in reranked:
        errors.append(
            f"the correct article (id {GOLD_ID}) is not in the final {EXPECTED_K_FINAL} documents. "
            "Hybrid retrieval plus re-ranking is supposed to recover it — check your fusion cell."
        )


def check_citations(answers, errors):
    """M2.3's deliverable is a pipeline with CITATIONS, so empty sources is a fail."""
    if not isinstance(answers, list) or len(answers) < 5:
        errors.append(
            f"'answers' has {len(answers) if isinstance(answers, list) else 0} records, expected 5 — "
            "re-run the end-to-end cell in Part 4."
        )
        return

    for record in answers:
        query = str(record.get("query", "?"))[:40]
        if not record.get("answer", "").strip():
            errors.append(f"no answer text recorded for {query!r}.")
        sources = record.get("source_ids") or []
        if not sources:
            errors.append(f"no citations recorded for {query!r} — the deliverable requires sources.")

    grounded = sum(1 for r in answers if GOLD_ID in (r.get("source_ids") or []))
    if grounded < 4:
        errors.append(
            f"only {grounded} of {len(answers)} phrasings cited the correct article (id {GOLD_ID}). "
            "All five describe the same problem and should resolve to the same article."
        )


def check_grounding(block, errors, warnings):
    """Hard-fail only if the model actually invented a figure. A soft 'didn't clearly
    decline' is a warning, because refusal wording varies run to run."""
    if not isinstance(block, dict):
        errors.append("results file has no 'grounding' block — re-run the out-of-scope cell in Part 4.")
        return False

    answer = str(block.get("answer", ""))
    declined = bool(block.get("declined"))
    invented_number = bool(re.search(r"[₹$]\s?\d|\b\d+\.\d{2}\b", answer))

    if not declined and invented_number:
        errors.append(
            "the pipeline invented a figure for a question the knowledge base cannot answer:\n"
            f"      {answer[:160]!r}\n"
            "      Check the 'answer ONLY from the context' line is still in your prompt."
        )
    elif not declined:
        warnings.append(
            "the out-of-scope answer did not clearly decline. Read it yourself and confirm it "
            "did not state a share price:\n"
            f"      {answer[:160]!r}"
        )
    return declined


def check_evaluation(block, errors):
    """M2.4's deliverable is the evaluation report."""
    if not isinstance(block, dict):
        errors.append("results file has no 'evaluation' block — re-run the evaluation cell in Part 4.")
        return None, None

    if block.get("n_queries") != EXPECTED_EVAL_QUERIES:
        errors.append(
            f"evaluation ran over {block.get('n_queries')!r} queries, expected {EXPECTED_EVAL_QUERIES}."
        )

    base = block.get("semantic_only") or {}
    new = block.get("hybrid_reranked") or {}
    for label, d in (("semantic_only", base), ("hybrid_reranked", new)):
        for metric in ("recall_at_1", "recall_at_5"):
            if not isinstance(d.get(metric), (int, float)):
                errors.append(f"evaluation block is missing a numeric {label}.{metric}.")

    base1, new1 = base.get("recall_at_1"), new.get("recall_at_1")
    if isinstance(base1, (int, float)) and isinstance(new1, (int, float)) and new1 < base1:
        errors.append(
            f"hybrid + re-ranking scored WORSE than semantic-only on Recall@1 "
            f"({new1:.0%} vs {base1:.0%}). Something is wired wrong — most likely the filter is "
            "being applied to one engine but not the other in hybrid_retrieve()."
        )
    return base1, new1


def main():
    results_path = find_file(RESULTS_FILENAME)
    if results_path is None:
        fail([
            f"{RESULTS_FILENAME} not found. Run every cell of "
            "D2_M2.3_M2.4_Hybrid_Rerank_Starter.ipynb in order — the last cell saves it."
        ])
        return

    data = load_json(results_path)
    articles = load_reference_articles()
    errors, warnings = [], []

    # 1. The bug must have been reproduced, not skipped.
    keyword_top5 = [int(i) for i in data.get("keyword_top5", [])]
    if not keyword_top5:
        errors.append("no 'keyword_top5' recorded — re-run the BM25 cell in Part 1.")
    elif keyword_top5[0] == GOLD_ID:
        errors.append(
            "keyword search put the correct article first, which the shipped corpus should never do. "
            "Did you edit the articles file? The point of Part 1 is watching BM25 fail."
        )

    # 2-4. Filter, retrieval, re-ranking.
    check_filter(data.get("filter"), articles, errors)
    check_retrieval(data.get("retrieval"), errors)

    # 5-6. M2.3 deliverable: end-to-end pipeline with citations, and no invention.
    check_citations(data.get("answers"), errors)
    declined = check_grounding(data.get("grounding"), errors, warnings)

    # 7. M2.4 deliverable: the evaluation report.
    base1, new1 = check_evaluation(data.get("evaluation"), errors)

    # --- what the trainer sees ---
    rrf_top5 = [int(i) for i in data.get("rrf_top5", [])]
    retrieval = data.get("retrieval") or {}
    print(f"Results file : {results_path}")
    print(f"Keyword rank 1: id {keyword_top5[0] if keyword_top5 else '?'}  (expected: not {GOLD_ID})")
    print(f"RRF rank 1    : id {rrf_top5[0] if rrf_top5 else '?'}")
    print(f"Retrieved     : {len(retrieval.get('candidate_ids', []))} candidates "
          f"-> re-ranked to {len(retrieval.get('reranked_ids', []))}")
    print(f"Final 5       : {retrieval.get('reranked_ids')}")
    if isinstance(base1, (int, float)) and isinstance(new1, (int, float)):
        print(f"Recall@1      : {base1:.0%} (semantic only) -> {new1:.0%} (hybrid + re-rank)")
    print(f"Declined out-of-scope question: {declined}")
    print()

    for w in warnings:
        print(f"  WARNING: {w}")
    if warnings:
        print()

    if errors:
        fail(errors)
        return

    # Stretch/Diamond: a perfect Recall@1 AND a clean refusal — the pipeline is both
    # accurate and honest, which is the standard M2.4 actually asks for.
    stretch = bool(isinstance(new1, (int, float)) and new1 >= 1.0 and declined)
    write_completion(DAY, MODULE_ID, stretch_completed=stretch)


if __name__ == "__main__":
    main()
