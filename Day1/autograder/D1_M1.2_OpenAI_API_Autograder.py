"""
Autograder for D1 · M1.2 — Building on the OpenAI API.

Imports the student's structured-extraction module and runs it against
Day1/data/D1_M1.2_sample_customer_messages.json — this makes real OpenAI API
calls (10 of them), so it costs a small amount on your API key each run.

Usage (from the repo root):
    python Day1/autograder/D1_M1.2_OpenAI_API_Autograder.py

Pass criteria (SYLLABUS_TRACE.md M1.2 — schema-conformance AND error-handling rubric):
  1. The retry loop must recover from one simulated malformed model response
     (checked via monkeypatching, no real API call for this part).
  2. At least 8 of the 10 sample messages must extract into a CustomerRequest
     that validates cleanly (allows some slack for occasional model flakiness
     rather than requiring a perfect 10/10).
On pass, writes Day1/submissions/D1_M1.2.json.
"""
import glob
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_shared"))
from autograder_utils import fail, write_completion  # noqa: E402

DAY = 1
MODULE_ID = "M1.2"
PASS_THRESHOLD = 8  # out of 10 sample messages
STARTER_FILENAME = "D1_M1.2_OpenAI_API_Starter.py"


def find_student_module():
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    matches = glob.glob(os.path.join(repo_root, "**", STARTER_FILENAME), recursive=True)
    # Prefer a match that isn't inside labs/solution/ (that's the trainer's copy)
    matches = [m for m in matches if os.sep + "solution" + os.sep not in m]
    return matches[0] if matches else None


def load_module(path):
    spec = importlib.util.spec_from_file_location("student_m1_2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_sample_messages():
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    path = os.path.join(repo_root, "Day1", "data", "D1_M1.2_sample_customer_messages.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_error_handling(mod):
    """
    Per SYLLABUS_TRACE.md M1.2, this module is graded on error-handling, not just
    schema-conformance. Monkeypatch the student's own `client.chat.completions.create`
    to return one malformed response (bad enum value + too-short summary) followed by
    a valid one, and confirm their retry logic actually recovers instead of crashing
    on the first bad response. Restores the real client afterwards regardless of outcome.
    """
    import types

    if not hasattr(mod, "client"):
        return False

    good_args = {
        "name": "Test Customer", "account_type": "debit_card", "intent": "card_issue",
        "urgency": "high", "customer_summary": "Customer's debit card was declined and needs urgent help.",
    }
    bad_args = {
        "name": "Test Customer", "account_type": "not_a_real_account_type", "intent": "card_issue",
        "urgency": "high", "customer_summary": "short",
    }

    class FakeToolCall:
        def __init__(self, args): self.function = types.SimpleNamespace(arguments=json.dumps(args))

    class FakeMessage:
        def __init__(self, args): self.tool_calls = [FakeToolCall(args)]

    class FakeChoice:
        def __init__(self, args): self.message = FakeMessage(args)

    class FakeResponse:
        def __init__(self, args): self.choices = [FakeChoice(args)]

    call_count = {"n": 0}

    def fake_create(*args, **kwargs):
        call_count["n"] += 1
        return FakeResponse(bad_args if call_count["n"] == 1 else good_args)

    original_create = mod.client.chat.completions.create
    mod.client.chat.completions.create = fake_create
    try:
        result = mod.extract_customer_request("irrelevant — response is mocked for this check", max_retries=2)
        recovered = isinstance(result, mod.CustomerRequest) and call_count["n"] >= 2
    except Exception:
        recovered = False
    finally:
        mod.client.chat.completions.create = original_create

    return recovered


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

    if not hasattr(mod, "extract_customer_request"):
        fail(["extract_customer_request function not found in the student module."])
        return
    if not hasattr(mod, "CustomerRequest"):
        fail(["CustomerRequest schema class not found in the student module."])
        return

    error_handling_ok = check_error_handling(mod)
    print(f"Error-handling check (retry-on-bad-response): {'PASS' if error_handling_ok else 'FAIL'}")
    if not error_handling_ok:
        fail(["extract_customer_request did not recover from a simulated bad model response. "
              "Per SYLLABUS_TRACE.md M1.2, this module is graded on a schema-conformance AND "
              "error-handling rubric — the retry loop must catch a ValidationError/JSONDecodeError "
              "on the first bad response and try again instead of raising immediately."])
        return

    messages = load_sample_messages()
    passed, errors = 0, []

    for item in messages:
        try:
            result = mod.extract_customer_request(item["message"])
            # result must actually be validated against the module's own schema
            if not isinstance(result, mod.CustomerRequest):
                errors.append(f"id {item['id']}: did not return a CustomerRequest instance")
                continue
            passed += 1
        except Exception as e:
            errors.append(f"id {item['id']}: {type(e).__name__}: {e}")

    print(f"{passed}/{len(messages)} sample messages extracted successfully.")
    if errors:
        print("Failures:")
        for e in errors:
            print(f"  - {e}")

    if passed < PASS_THRESHOLD:
        fail([f"Only {passed}/{len(messages)} sample messages extracted successfully; "
              f"need at least {PASS_THRESHOLD}/{len(messages)} to pass."])
        return

    # Stretch: treat a perfect 10/10 as this module's stretch/Diamond signal.
    write_completion(DAY, MODULE_ID, stretch_completed=(passed == len(messages)))


if __name__ == "__main__":
    main()
