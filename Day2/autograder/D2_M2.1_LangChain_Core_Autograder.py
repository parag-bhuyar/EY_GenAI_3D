"""
Autograder for D2 · M2.1 — LangChain 1.0 Core.

Run this from anywhere inside the repo, after you've run every cell in your
D2_M2.1_LangChain_Core_Starter.ipynb (or your renamed working copy) and it has
saved m2_1_extraction_results.json next to itself:

    python Day2/autograder/D2_M2.1_LangChain_Core_Autograder.py

Reads the notebook's saved results file rather than importing the notebook
directly (notebooks aren't importable as Python modules) — same pattern as
D1_M1.1's autograder.

Pass criteria (SYLLABUS_TRACE.md M2.1 — graded on correct use of 1.x patterns,
no deprecated APIs):
  1. No deprecated LangChain APIs (AgentExecutor, LLMChain) anywhere in the
     student's starter notebook source — current 1.x patterns only.
  2. At least 8 of the 10 sample messages must have extracted into a valid
     CustomerRequest (checked from the saved results, not re-run live).
On pass, writes Day2/submissions/D2_M2.1.json.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_shared"))
from autograder_utils import fail, write_completion  # noqa: E402

DAY = 2
MODULE_ID = "M2.1"
PASS_THRESHOLD = 8  # out of 10 sample messages
STARTER_FILENAME = "D2_M2.1_LangChain_Core_Starter.ipynb"
RESULTS_FILENAME = "m2_1_extraction_results.json"
DEPRECATED_PATTERNS = ["AgentExecutor", "LLMChain"]


def find_student_notebook():
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    matches = glob.glob(os.path.join(repo_root, "**", STARTER_FILENAME), recursive=True)
    matches = [m for m in matches if os.sep + "solution" + os.sep not in m]
    return matches[0] if matches else None


def find_results_file():
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    matches = glob.glob(os.path.join(repo_root, "**", RESULTS_FILENAME), recursive=True)
    return matches[0] if matches else None


def check_no_deprecated_apis(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    found = [p for p in DEPRECATED_PATTERNS if p in source]
    return (len(found) == 0), found


def main():
    notebook_path = find_student_notebook()
    if notebook_path is None:
        fail([f"{STARTER_FILENAME} not found anywhere in the repo (excluding labs/solution/)."])
        return

    deprecated_ok, found = check_no_deprecated_apis(notebook_path)
    print(f"No deprecated LangChain APIs check: {'PASS' if deprecated_ok else 'FAIL'}")
    if not deprecated_ok:
        fail([f"Found deprecated pattern(s) in your notebook: {found}. Use current LangChain 1.x "
              "patterns only (init_chat_model, create_agent, middleware) — no AgentExecutor/LLMChain."])
        return

    results_path = find_results_file()
    if results_path is None:
        fail([f"{RESULTS_FILENAME} not found anywhere in the repo. Run every cell of the "
              "notebook in order, ending with the final save cell, then try again."])
        return

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    evaluation = data.get("sample_evaluation")
    if not isinstance(evaluation, list) or not evaluation:
        fail([f"{RESULTS_FILENAME} is missing 'sample_evaluation' — re-run the notebook's "
              "Task 5 cell and final save cell."])
        return

    passed = sum(1 for r in evaluation if isinstance(r, dict) and r.get("ok"))
    total = len(evaluation)
    errors = [
        f"id {r.get('id')}: {r.get('error')}"
        for r in evaluation if isinstance(r, dict) and not r.get("ok")
    ]

    print(f"{passed}/{total} sample messages extracted successfully.")
    if errors:
        print("Failures:")
        for e in errors:
            print(f"  - {e}")

    if passed < PASS_THRESHOLD:
        fail([f"Only {passed}/{total} sample messages extracted successfully; "
              f"need at least {PASS_THRESHOLD}/{total} to pass."])
        return

    write_completion(DAY, MODULE_ID, stretch_completed=(passed == total))


if __name__ == "__main__":
    main()
