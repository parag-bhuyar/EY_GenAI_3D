"""
Autograder for D2 · M2.2 — Vector Databases (Chroma & Qdrant).

Checks the full M2.2 deliverable per SYLLABUS_TRACE.md: "Populated Chroma collection;
equivalent Qdrant collection with metadata filters." Reads both Part 1's and Part 2's
saved results files rather than re-running either notebook live.

Run this from anywhere inside the repo, after running every cell of BOTH notebooks in
order (Part 1 saves m1_4_m2_2_part1_results.json, Part 2 saves m2_2_part2_results.json):

    python Day2/autograder/D2_M2.2_VectorStores_Autograder.py

Pass criteria:
  1. Part 1's results file exists with a Chroma collection of exactly 10 passages, and a
     metadata-filtered query (category='fraud') that returned only passage(s) actually
     tagged 'fraud' in the reference data — not just a non-empty result.
  2. Part 2's results file exists with an equivalent Qdrant collection (10 passages) and
     the same metadata-filter correctness check.
  3. Part 2's comparison_table exists, showing both stores queried against the same input.
On pass, writes Day2/submissions/D2_M2.2.json. Stretch/Diamond = Qdrant actually connected
via self-hosted Docker (the syllabus's primary path), not the in-memory fallback.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_shared"))
from autograder_utils import fail, write_completion  # noqa: E402

DAY = 2
MODULE_ID = "M2.2"
PART1_RESULTS_FILENAME = "m1_4_m2_2_part1_results.json"
PART2_RESULTS_FILENAME = "m2_2_part2_results.json"
EXPECTED_PASSAGE_COUNT = 10


def find_file(filename):
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    matches = glob.glob(os.path.join(repo_root, "**", filename), recursive=True)
    return matches[0] if matches else None


def load_reference_categories():
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    path = os.path.join(repo_root, "Day1", "data", "D1_M1.4_banking_passages.json")
    with open(path, "r", encoding="utf-8") as f:
        passages = json.load(f)
    return {p["id"]: p["category"] for p in passages}


def check_filter_correctness(ids, expected_category, categories_by_id, label, errors):
    """Every id returned by a category-filtered query must actually carry that category
    in the reference data — a real correctness check, not just 'got something back'."""
    if not ids:
        errors.append(f"{label}: filtered query returned no results at all.")
        return
    for pid in ids:
        pid_int = int(pid)
        actual = categories_by_id.get(pid_int)
        if actual != expected_category:
            errors.append(
                f"{label}: id {pid} has category '{actual}', expected '{expected_category}' — "
                "the metadata filter isn't actually restricting results correctly."
            )


def main():
    categories_by_id = load_reference_categories()
    errors = []

    # --- Part 1 / Chroma ---
    part1_path = find_file(PART1_RESULTS_FILENAME)
    if part1_path is None:
        fail([f"{PART1_RESULTS_FILENAME} not found. Run Part 1's notebook (all cells, in order) first."])
        return
    with open(part1_path, "r", encoding="utf-8") as f:
        part1_data = json.load(f)

    chroma = part1_data.get("chroma_query_results")
    if not isinstance(chroma, dict):
        fail([f"{PART1_RESULTS_FILENAME} is missing 'chroma_query_results' — re-run Part 1's Chroma cell."])
        return
    if chroma.get("collection_count") != EXPECTED_PASSAGE_COUNT:
        errors.append(
            f"Chroma collection_count is {chroma.get('collection_count')!r}, expected {EXPECTED_PASSAGE_COUNT}."
        )
    check_filter_correctness(chroma.get("filtered_ids", []), "fraud", categories_by_id, "Chroma", errors)

    # --- Part 2 / Qdrant ---
    part2_path = find_file(PART2_RESULTS_FILENAME)
    if part2_path is None:
        fail([f"{PART2_RESULTS_FILENAME} not found. Run Part 2's notebook (all cells, in order)."])
        return
    with open(part2_path, "r", encoding="utf-8") as f:
        part2_data = json.load(f)

    qdrant = part2_data.get("qdrant_query_results")
    if not isinstance(qdrant, dict):
        fail([f"{PART2_RESULTS_FILENAME} is missing 'qdrant_query_results' — re-run Part 2's Qdrant cell."])
        return
    if qdrant.get("collection_count") != EXPECTED_PASSAGE_COUNT:
        errors.append(
            f"Qdrant collection_count is {qdrant.get('collection_count')!r}, expected {EXPECTED_PASSAGE_COUNT}."
        )
    check_filter_correctness(qdrant.get("filtered_ids", []), "fraud", categories_by_id, "Qdrant", errors)

    comparison = part2_data.get("comparison_table")
    if not isinstance(comparison, dict) or "chroma" not in comparison or "qdrant" not in comparison:
        errors.append(f"{PART2_RESULTS_FILENAME} is missing a valid 'comparison_table' with both stores.")

    print(f"Chroma: {chroma.get('collection_count')} passages, filtered_ids={chroma.get('filtered_ids')}")
    print(f"Qdrant: {qdrant.get('collection_count')} passages, filtered_ids={qdrant.get('filtered_ids')} "
          f"(mode: {qdrant.get('mode')})")

    if errors:
        fail(errors)
        return

    # Stretch/Diamond: Qdrant actually ran self-hosted (Docker), the syllabus's primary/production
    # path — not just the in-memory fallback.
    stretch = str(qdrant.get("mode", "")).startswith("docker")
    write_completion(DAY, MODULE_ID, stretch_completed=stretch)


if __name__ == "__main__":
    main()
