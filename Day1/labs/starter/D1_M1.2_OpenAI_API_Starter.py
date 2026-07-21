"""
D1 · M1.2 — Building on the OpenAI API
Structured-extraction module: converts a free-text Meridian Retail Bank customer
message into a validated, structured record using tool calling.

Deliverable (per SYLLABUS_TRACE.md): a working structured-extraction module with
schema validation. This file IS that module — and it already works end to end.

**This file contains fully working code, not TODOs.** Your job is to read every
block below before you run it, understand what it does and why, then run it and
look at the real output. Same format as M1.1 — you're not writing this code from
scratch (the real skill being tested here is whether you understand tool calling
and validation well enough to use, extend, and debug it, not whether you can
type it out cold in a 2-hour window). Comprehension is checked live afterward,
not by whether you typed the code yourself.

Run directly for a quick manual check:
    python D1_M1.2_OpenAI_API_Starter.py
"""
import json
import os
import time
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# Loads OPENAI_API_KEY from the .env file sitting next to this script — you
# never need to type or export the key in the terminal.
load_dotenv(Path(__file__).resolve().parent / ".env")

# Same pattern as M1.1: fail loudly and immediately if the key is missing,
# rather than proceeding and getting a confusing error 30 seconds later on
# the actual API call.
assert os.environ.get("OPENAI_API_KEY"), (
    "OPENAI_API_KEY is not set. Set it before continuing — never hard-code a key in this file."
)

client = OpenAI()
MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Schema — the contract every extraction must satisfy.
#
# These fields match the programme syllabus's session-plan wording exactly:
# name, account type and request intent (SYLLABUS_TRACE.md M1.2). urgency and
# customer_summary are additional fields kept because a real triage system
# needs them, and because this same module becomes Capstone Phase 1 — not a
# substitute for the three named fields.
#
# Pydantic does two jobs here: (1) it's a plain data container, and (2) calling
# CustomerRequest(**some_dict) actively validates every field — wrong type,
# missing field, or a value outside an allowed `Literal` list all raise a
# ValidationError immediately. That's what makes this a real safety net
# instead of just documentation.
# ---------------------------------------------------------------------------
class CustomerRequest(BaseModel):
    """Structured record extracted from a free-text Meridian Retail Bank customer message."""

    name: str = Field(
        ..., min_length=1,
        description="Customer's name if stated or clearly implied in the message; 'Unknown' if not given.",
    )
    account_type: Literal[
        "debit_card", "credit_card", "savings_account",
        "personal_loan", "mortgage", "unknown",
    ] = Field(..., description="The Meridian Retail Bank account/product type the message is about.")
    intent: Literal[
        "card_issue", "account_access", "loan_inquiry",
        "complaint", "general_question", "fraud_report",
    ] = Field(..., description="The primary reason the customer is reaching out.")
    urgency: Literal["low", "medium", "high"] = Field(
        ..., description="How time-sensitive this request is, based on the message's own wording."
    )
    customer_summary: str = Field(
        ..., min_length=10, description="One-sentence, human-readable summary of what the customer needs."
    )


# ---------------------------------------------------------------------------
# Tool schema — this is what actually gets sent to the model. It describes
# CustomerRequest's fields as JSON schema so the model knows exactly what
# shape of arguments it's allowed to return.
#
# Note this is written by hand rather than auto-generated from
# CustomerRequest.model_json_schema() — that method gets you close but
# includes extra Pydantic metadata keys the tool-calling API doesn't expect.
# Writing it once by hand also makes the mapping from schema -> tool spec
# concrete instead of "magic."
# ---------------------------------------------------------------------------
EXTRACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_customer_request",
        "description": "Extract a structured record from a Meridian Retail Bank customer message.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Customer's name if stated or clearly implied; 'Unknown' if not given.",
                },
                "account_type": {
                    "type": "string",
                    "enum": ["debit_card", "credit_card", "savings_account",
                              "personal_loan", "mortgage", "unknown"],
                    "description": "The Meridian Retail Bank account/product type the message is about.",
                },
                "intent": {
                    "type": "string",
                    "enum": ["card_issue", "account_access", "loan_inquiry",
                              "complaint", "general_question", "fraud_report"],
                    "description": "The primary reason the customer is reaching out.",
                },
                "urgency": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "How time-sensitive this request is, based on the message's own wording.",
                },
                "customer_summary": {
                    "type": "string",
                    "description": "One-sentence, human-readable summary of what the customer needs.",
                },
            },
            # Every field is required — the model must always return a complete record,
            # never a partial one your code would have to guess how to fill in.
            "required": ["name", "account_type", "intent", "urgency", "customer_summary"],
        },
    },
}

# Read this system prompt carefully — it's what tells the model to call the
# tool at all rather than just answering the customer directly in plain text.
SYSTEM_PROMPT = (
    "You are a request-triage assistant for Meridian Retail Bank. Extract structured "
    "information from the customer's message by calling the extract_customer_request "
    "tool. Do not answer the customer directly — only call the tool with your best "
    "reading of their name (or 'Unknown' if not stated), account type, intent, urgency, "
    "and a one-sentence summary."
)


def extract_customer_request(message: str, max_retries: int = 2) -> CustomerRequest:
    """
    Call the model with tool calling forced to EXTRACTION_TOOL, parse the returned
    arguments, and validate them against CustomerRequest. On a validation or parsing
    failure, retries (up to max_retries) instead of crashing immediately — this is the
    "robustness" half of this module's topics: a model occasionally returns a value
    outside the allowed Literal options, and production code has to recover from that,
    not just hope it never happens.
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            # Step 1 — call the model. tool_choice FORCES this specific tool to be
            # called (as opposed to "auto", which lets the model decide whether to
            # call a tool at all). Forcing it is deliberate here: an extraction task
            # should never "decide" not to extract. temperature=0 because we want the
            # most consistent, deterministic reading of the message, not creative variety.
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message},
                ],
                tools=[EXTRACTION_TOOL],
                tool_choice={"type": "function", "function": {"name": "extract_customer_request"}},
                temperature=0,
            )

            # Step 2 — pull the tool call out of the response. If tool_choice is set
            # correctly this should never be empty, but we check anyway rather than
            # assuming — trusting the API blindly is exactly the habit this module
            # is trying to break.
            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                raise ValueError("Model did not return a tool call.")

            # Step 3 — the arguments come back as a JSON *string*, not a parsed dict.
            # json.loads() can itself fail if the model ever returns malformed JSON
            # (rare, but possible) — that's why this whole block is inside the retry's
            # try/except, not just the validation step.
            raw_args = tool_calls[0].function.arguments
            parsed = json.loads(raw_args)

            # Step 4 — this is the actual validation. If `parsed` is missing a field,
            # has the wrong type, or has a value outside an allowed Literal (e.g. the
            # model invents an account_type not in our enum), this line raises
            # ValidationError right here — it does not silently pass through.
            return CustomerRequest(**parsed)

        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            # Step 5 — this is the retry logic. We remember the error, wait a short,
            # increasing amount of time (simple backoff — not production-grade, but
            # enough to demonstrate the pattern), and try again instead of raising
            # immediately. Only after max_retries failures do we actually give up.
            last_error = e
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
            continue

    raise RuntimeError(f"Extraction failed after {max_retries + 1} attempts: {last_error}")


if __name__ == "__main__":
    # Manual smoke test — run this file directly to see one real extraction before
    # you run the autograder against all 10 sample messages.
    sample = "My debit card was declined twice today, this is urgent, I need to pay rent."
    result = extract_customer_request(sample)
    print(result.model_dump_json(indent=2))
