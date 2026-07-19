"""
Autograder for D1 · M1.1 — LLM Mechanics for Practitioners.

Run this from anywhere inside the repo, after you've run every cell in your
D1_M1.1_LLM_Mechanics_Starter.ipynb (or your renamed working copy) and it has
saved m1_1_comparison_results.json next to itself:

    python Day1/autograder/D1_M1.1_LLM_Mechanics_Autograder.py

On pass, writes Day1/submissions/D1_M1.1.json — push that file to your fork.
On fail, prints exactly what's missing and writes nothing.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_shared"))
from autograder_utils import fail, write_completion  # noqa: E402

DAY = 1
MODULE_ID = "M1.1"
RESULTS_FILENAME = "m1_1_comparison_results.json"


def find_results_file():
    # Search the whole repo for the results file, in case the student ran the
    # notebook from a renamed copy or a different working directory.
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    matches = glob.glob(os.path.join(repo_root, "**", RESULTS_FILENAME), recursive=True)
    return matches[0] if matches else None


def check_run_list(runs, required_keys, min_entries, label, errors):
    if not isinstance(runs, list) or len(runs) < min_entries:
        errors.append(f"{label}: expected a list of at least {min_entries} entries, found {runs!r}")
        return
    for i, r in enumerate(runs):
        if not isinstance(r, dict):
            errors.append(f"{label}[{i}]: expected a dict, found {type(r).__name__}")
            continue
        for key in required_keys:
            if key not in r:
                errors.append(f"{label}[{i}]: missing required key '{key}'")
                continue
            if key in ("latency_seconds", "prompt_tokens", "completion_tokens") and not (
                isinstance(r[key], (int, float)) and r[key] > 0
            ):
                errors.append(f"{label}[{i}]: '{key}' must be a positive number, found {r[key]!r}")
            if key == "response_text" and not (isinstance(r[key], str) and r[key].strip()):
                errors.append(f"{label}[{i}]: 'response_text' must be a non-empty string")


def main():
    path = find_results_file()
    errors = []

    if path is None:
        fail([f"{RESULTS_FILENAME} not found anywhere in the repo. Run the notebook's final "
              f"save cell first."])
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    check_run_list(
        data.get("temperature_runs"),
        required_keys=["temperature", "latency_seconds", "prompt_tokens", "completion_tokens", "response_text"],
        min_entries=6,
        label="temperature_runs",
        errors=errors,
    )
    distinct_temps = {r.get("temperature") for r in data.get("temperature_runs", []) if isinstance(r, dict)}
    if len(distinct_temps) < 3:
        errors.append(f"temperature_runs: expected 3 distinct temperature values, found {distinct_temps}")

    check_run_list(
        data.get("sampling_runs"),
        required_keys=["variant", "latency_seconds", "prompt_tokens", "completion_tokens", "response_text"],
        min_entries=4,
        label="sampling_runs",
        errors=errors,
    )
    expected_variants = {"top_p_0.1", "top_p_1.0", "max_tokens_20", "stop_period"}
    found_variants = {r.get("variant") for r in data.get("sampling_runs", []) if isinstance(r, dict)}
    if not expected_variants.issubset(found_variants):
        errors.append(f"sampling_runs: missing variants {expected_variants - found_variants}")

    check_run_list(
        data.get("model_comparison_runs"),
        required_keys=["model", "latency_seconds", "prompt_tokens", "completion_tokens", "response_text"],
        min_entries=2,
        label="model_comparison_runs",
        errors=errors,
    )
    distinct_models = {r.get("model") for r in data.get("model_comparison_runs", []) if isinstance(r, dict)}
    if len(distinct_models) < 2:
        errors.append(f"model_comparison_runs: expected 2 distinct models, found {distinct_models}")

    annotation = data.get("annotation", "")
    if not isinstance(annotation, str) or len(annotation.strip()) < 200 or "REPLACE THIS PLACEHOLDER" in annotation:
        errors.append(
            "annotation: must be your own genuine reflection (200+ characters), not the "
            "starter placeholder text."
        )

    if errors:
        fail(errors)
        return

    stretch = data.get("reproducibility_variance") or {}
    stretch_ok = (
        isinstance(stretch, dict)
        and "temp_0_avg_similarity" in stretch
        and "temp_1_avg_similarity" in stretch
    )

    write_completion(DAY, MODULE_ID, stretch_completed=stretch_ok)


if __name__ == "__main__":
    main()
