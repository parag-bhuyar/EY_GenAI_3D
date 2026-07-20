"""
Autograder for D1 · M1.4 — Embeddings & Semantic Search Foundations.

Imports the student's semantic-search module, verifies the cosine_similarity math
directly (no API calls needed for that part), then builds a fresh passage index and
runs the 5 fixed queries against it — this makes real OpenAI embedding calls (10
passages + 5 queries = 15 calls), which is very cheap (embeddings pricing is far
below chat-completion pricing) but not free.

Usage (from the repo root):
    python Day1/autograder/D1_M1.4_Embeddings_Autograder.py

Pass criteria (SYLLABUS_TRACE.md M1.4 — graded on retrieval correctness for a fixed
query set):
  1. cosine_similarity must be mathematically correct: identical vectors -> ~1.0,
     orthogonal vectors -> ~0.0, opposite vectors -> ~-1.0. Checked with plain
     Python lists, no API calls.
  2. The passage set and fixed query set data files must be unmodified (10 passages,
     5 queries, each with the expected fields).
  3. At least 4 of the 5 fixed queries must retrieve their expected_top_id as the
     #1 semantic_search result (80% threshold, matching this programme's other
     modules' slack for occasional model/embedding variance).
On pass, writes Day1/submissions/D1_M1.4.json. Stretch/Diamond = a perfect 5/5.
"""
import glob
import importlib.util
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_shared"))
from autograder_utils import fail, write_completion  # noqa: E402

DAY = 1
MODULE_ID = "M1.4"
PASS_THRESHOLD = 4  # out of 5 fixed queries
STARTER_FILENAME = "D1_M1.4_Embeddings_Starter.py"


def find_student_module():
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    matches = glob.glob(os.path.join(repo_root, "**", STARTER_FILENAME), recursive=True)
    matches = [m for m in matches if os.sep + "solution" + os.sep not in m]
    return matches[0] if matches else None


def load_module(path):
    spec = importlib.util.spec_from_file_location("student_m1_4", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_cosine_similarity_math(mod):
    """
    Verifies the hand-written cosine_similarity function is actually correct,
    independent of any API call — this is pure linear algebra and should be
    exactly right, not just "close enough on real embeddings."
    """
    if not hasattr(mod, "cosine_similarity"):
        return False, "cosine_similarity function not found."

    cases = [
        ("identical vectors", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1.0),
        ("orthogonal vectors", [1.0, 0.0], [0.0, 1.0], 0.0),
        ("opposite vectors", [1.0, 0.0], [-1.0, 0.0], -1.0),
    ]
    for label, a, b, expected in cases:
        try:
            result = mod.cosine_similarity(a, b)
        except Exception as e:
            return False, f"cosine_similarity raised {type(e).__name__} on {label}: {e}"
        if not isinstance(result, (int, float)) or abs(result - expected) > 1e-6:
            return False, f"cosine_similarity({label}) expected {expected}, got {result!r}"

    return True, None


def load_reference_data():
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    with open(os.path.join(repo_root, "Day1", "data", "D1_M1.4_banking_passages.json"), "r", encoding="utf-8") as f:
        passages = json.load(f)
    with open(os.path.join(repo_root, "Day1", "data", "D1_M1.4_fixed_queries.json"), "r", encoding="utf-8") as f:
        queries = json.load(f)
    return passages, queries


def main():
    student_path = find_student_module()
    if student_path is None:
        fail([f"{STARTER_FILENAME} not found anywhere in the repo (excluding labs/solution/)."])
        return

    try:
        mod = load_module(student_path)
    except Exception as e:
        fail([f"Student module failed to import: {type(e).__name__}: {e}"])
        return

    for required in ("get_embedding", "cosine_similarity", "semantic_search", "build_passage_index"):
        if not hasattr(mod, required):
            fail([f"{required} not found in the student module."])
            return

    math_ok, math_error = check_cosine_similarity_math(mod)
    print(f"cosine_similarity math check: {'PASS' if math_ok else 'FAIL'}")
    if not math_ok:
        fail([f"cosine_similarity is not mathematically correct: {math_error}"])
        return

    passages, queries = load_reference_data()
    if len(passages) != 10:
        fail([f"Expected exactly 10 passages, found {len(passages)}. Did the data file get edited?"])
        return
    if len(queries) != 5:
        fail([f"Expected exactly 5 fixed queries, found {len(queries)}. Did the data file get edited?"])
        return

    print("Building a fresh passage index from the student's build_passage_index (10 embedding calls)...")
    try:
        index = mod.build_passage_index(passages)
    except Exception as e:
        fail([f"build_passage_index raised {type(e).__name__}: {e}"])
        return

    passed = 0
    report = []
    for q in queries:
        try:
            results = mod.semantic_search(q["query"], index, top_k=3)
            top_id = results[0]["id"]
        except Exception as e:
            top_id = f"error:{type(e).__name__}"

        correct = top_id == q["expected_top_id"]
        if correct:
            passed += 1
        report.append({"query": q["query"], "expected": q["expected_top_id"], "retrieved": top_id, "pass": correct})

    print("Fixed query evaluation:")
    for row in report:
        print(f"  \"{row['query']}\" -> expected {row['expected']}, retrieved {row['retrieved']} "
              f"[{'PASS' if row['pass'] else 'FAIL'}]")

    print(f"\n{passed}/{len(queries)} fixed queries retrieved their expected passage at #1.")

    if passed < PASS_THRESHOLD:
        fail([f"Only {passed}/{len(queries)} fixed queries passed; need at least "
              f"{PASS_THRESHOLD}/{len(queries)} to pass."])
        return

    # Stretch: a perfect 5/5 counts as this module's stretch/Diamond signal.
    write_completion(DAY, MODULE_ID, stretch_completed=(passed == len(queries)))


if __name__ == "__main__":
    main()
