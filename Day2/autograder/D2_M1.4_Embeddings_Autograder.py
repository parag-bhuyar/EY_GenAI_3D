"""
Autograder for D1 · M1.4 — Embeddings & Semantic Search Foundations.

Lives under Day2/autograder because M1.4 is now delivered combined with M2.2 (Part 1 —
see D2_M1.4_M2.2_VectorStores_Starter.ipynb) — but this checks M1.4's own module, on its
own original Assessment Method, independent of M2.2's Chroma/Qdrant deliverable.

Run this from anywhere inside the repo, after running every cell of the combined notebook
in order (it saves m1_4_m2_2_part1_results.json next to itself):

    python Day2/autograder/D2_M1.4_Embeddings_Autograder.py

Pass criteria (SYLLABUS_TRACE.md M1.4 — graded on retrieval correctness for a fixed query set):
  1. cosine_similarity must be mathematically correct: identical vectors -> ~1.0, orthogonal
     -> ~0.0, opposite -> ~-1.0. Extracted from the notebook's source via AST and executed
     standalone (no API calls, no other notebook code runs) — same principle as importing a
     module, but notebooks aren't importable.
  2. The passage set and fixed query set data files must be unmodified (10 passages, 5
     queries, each with the expected fields).
  3. At least 4 of the 5 fixed queries must have retrieved their expected_top_id as the #1
     semantic_search result, read from the saved results file (not re-run live).
On pass, writes Day1/submissions/D1_M1.4.json. Stretch/Diamond = a perfect 5/5.
"""
import ast
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_shared"))
from autograder_utils import fail, write_completion  # noqa: E402

DAY = 1
MODULE_ID = "M1.4"
PASS_THRESHOLD = 4  # out of 5 fixed queries
STARTER_FILENAME = "D2_M1.4_M2.2_VectorStores_Starter.ipynb"
RESULTS_FILENAME = "m1_4_m2_2_part1_results.json"


def find_student_notebook():
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    matches = glob.glob(os.path.join(repo_root, "**", STARTER_FILENAME), recursive=True)
    matches = [m for m in matches if os.sep + "solution" + os.sep not in m]
    return matches[0] if matches else None


def find_results_file():
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    matches = glob.glob(os.path.join(repo_root, "**", RESULTS_FILENAME), recursive=True)
    return matches[0] if matches else None


def extract_notebook_source(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def check_cosine_similarity_math(notebook_source):
    """
    Extracts just the cosine_similarity function definition via AST and runs it standalone —
    no other notebook code (including API calls) executes. Same intent as M1.4's original
    module-import check, adapted for a notebook that can't be imported directly.
    """
    try:
        tree = ast.parse(notebook_source)
    except SyntaxError as e:
        return False, f"Notebook source failed to parse: {e}"

    func_node = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cosine_similarity"),
        None,
    )
    if func_node is None:
        return False, "cosine_similarity function not found anywhere in the notebook."

    import math
    namespace = {"math": math}  # cosine_similarity calls math.sqrt(); it isn't extracted with the function
    try:
        exec(compile(ast.Module(body=[func_node], type_ignores=[]), "<cosine_similarity>", "exec"), namespace)
    except Exception as e:
        return False, f"cosine_similarity failed to compile/execute standalone: {type(e).__name__}: {e}"

    cosine_similarity = namespace.get("cosine_similarity")
    if cosine_similarity is None:
        return False, "cosine_similarity did not define itself correctly."

    cases = [
        ("identical vectors", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1.0),
        ("orthogonal vectors", [1.0, 0.0], [0.0, 1.0], 0.0),
        ("opposite vectors", [1.0, 0.0], [-1.0, 0.0], -1.0),
    ]
    for label, a, b, expected in cases:
        try:
            result = cosine_similarity(a, b)
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
    notebook_path = find_student_notebook()
    if notebook_path is None:
        fail([f"{STARTER_FILENAME} not found anywhere in the repo (excluding labs/solution/)."])
        return

    notebook_source = extract_notebook_source(notebook_path)

    math_ok, math_error = check_cosine_similarity_math(notebook_source)
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

    results_path = find_results_file()
    if results_path is None:
        fail([f"{RESULTS_FILENAME} not found anywhere in the repo. Run every cell of the "
              "combined notebook in order, ending with the final save cell, then try again."])
        return

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    evaluation = data.get("evaluation_report")
    if not isinstance(evaluation, list) or not evaluation:
        fail([f"{RESULTS_FILENAME} is missing 'evaluation_report' — re-run the notebook's "
              "Task 5 cell and final save cell."])
        return

    print("Fixed query evaluation:")
    passed = 0
    for row in evaluation:
        ok = bool(row.get("pass"))
        if ok:
            passed += 1
        print(f"  \"{row.get('query')}\" -> expected {row.get('expected_top_id')}, "
              f"retrieved {row.get('retrieved_top_id')} [{'PASS' if ok else 'FAIL'}]")

    total = len(evaluation)
    print(f"\n{passed}/{total} fixed queries retrieved their expected passage at #1.")

    if passed < PASS_THRESHOLD:
        fail([f"Only {passed}/{total} fixed queries passed; need at least "
              f"{PASS_THRESHOLD}/{total} to pass."])
        return

    write_completion(DAY, MODULE_ID, stretch_completed=(passed == total))


if __name__ == "__main__":
    main()
