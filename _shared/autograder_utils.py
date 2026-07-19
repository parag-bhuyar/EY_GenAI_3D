"""
Shared helpers for every module's autograder.

Reused across all 12 modules so the completion-JSON contract (session.md Section 5)
stays identical everywhere instead of being reimplemented — and possibly drifting —
in every autograder script. Do not duplicate this logic inside a module's autograder;
import it instead.

Student setup (one-time, Day 1 morning): copy `student_config.example.json` at the
repo root to `student_config.json` and fill in your real name and GitHub username.
Every autograder reads that file via `load_student_config()`.
"""
import json
import os
import sys
from datetime import datetime, timezone

# Repo root = two levels up from Day{n}/autograder/ where each autograder script lives.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))

STUDENT_CONFIG_PATH = os.path.join(REPO_ROOT, "student_config.json")


class AutograderError(Exception):
    """Raised when the student config or lab output can't be validated."""


def load_student_config():
    """Read student_config.json from the repo root. Fails loudly and clearly if missing."""
    if not os.path.exists(STUDENT_CONFIG_PATH):
        raise AutograderError(
            "student_config.json not found at the repo root.\n"
            "Copy student_config.example.json to student_config.json and fill in your "
            "real name and GitHub username before running any lab autograder."
        )
    with open(STUDENT_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("student_name", "student_github"):
        if not cfg.get(key) or not str(cfg[key]).strip():
            raise AutograderError(f"student_config.json is missing a non-empty '{key}' value.")
    return cfg


def write_completion(day: int, module_id: str, stretch_completed: bool = False):
    """
    Write this module's completion JSON to Day{day}/submissions/D{day}_{module_id}.json,
    per the schema in session.md Section 5. Only call this after all pass checks succeed —
    the autograder must never call this on a failing run.
    """
    cfg = load_student_config()
    record = {
        "student_name": cfg["student_name"],
        "student_github": cfg["student_github"],
        "day": day,
        "module_id": module_id,
        "status": "pass",
        "stretch_completed": bool(stretch_completed),
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    out_dir = os.path.join(REPO_ROOT, f"Day{day}", "submissions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"D{day}_{module_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"PASS — wrote {out_path}")
    print(json.dumps(record, indent=2))
    return out_path


def fail(messages):
    """Print all failed checks clearly and exit non-zero. Never writes a completion JSON."""
    print("FAIL — lab did not pass the autograder. Fix the following and re-run:\n")
    for m in messages:
        print(f"  - {m}")
    sys.exit(1)
