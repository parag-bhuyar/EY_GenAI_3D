"""
D1 · M1.2 — Building on the OpenAI API
Structured-extraction module: converts a free-text Meridian Retail Bank customer
message into a validated, structured record using tool calling.

Deliverable (per SYLLABUS_TRACE.md): a working structured-extraction module with
schema validation. This file IS that module — fill in the TODOs so that
`extract_customer_request(message)` returns a validated CustomerRequest.

Run directly for a quick manual check:
    python Day1_M1.2_OpenAI_API_Starter.py
"""
import json
import os
import time
from typing import Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

assert os.environ.get("OPENAI_API_KEY"), (
    "OPENAI_API_KEY is not set. Set it before continuing — never hard-code a key in this file."
)

client = OpenAI()
MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Schema — the contract every extraction must satisfy.
# ---------------------------------------------------------------------------
class CustomerRequest(BaseModel):
    """
    Structured record extracted from a free-text Meridian Retail Bank customer message.
    Fields match the programme syllabus's session-plan wording exactly: name, account
    type and request intent. urgency and customer_summary are additional fields kept
    because they're genuinely useful for triage — don't change any field names, the
    autograder depends on this exact shape.
    """

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
# Tool schema — TODO: this must describe CustomerRequest's fields to the model.
# ---------------------------------------------------------------------------
EXTRACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_customer_request",
        "description": "Extract a structured record from a Meridian Retail Bank customer message.",
        # TODO: fill in "parameters" as a JSON-schema object matching CustomerRequest's fields
        # (name, account_type, intent, urgency, customer_summary). Hint: CustomerRequest.model_json_schema()
        # gets you most of the way there, but double-check it's valid for the API's tool format
        # (an object with "type": "object", "properties": {...}, "required": [...]).
        "parameters": {},  # <-- TODO
    },
}


def extract_customer_request(message: str, max_retries: int = 2) -> CustomerRequest:
    """
    Call the model with tool calling forced to EXTRACTION_TOOL, parse the returned
    arguments, and validate them against CustomerRequest. Retry on a validation
    failure (the model occasionally returns a value outside the allowed Literal
    options) up to `max_retries` times before raising.
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        # TODO:
        #   1. Call client.chat.completions.create(...) with:
        #      - model=MODEL
        #      - messages=[a system message giving context, a user message with `message`]
        #      - tools=[EXTRACTION_TOOL]
        #      - tool_choice forcing EXTRACTION_TOOL to be called
        #   2. Pull the tool call's arguments string out of the response and json.loads() it
        #   3. Validate with CustomerRequest(**parsed_args) — this raises ValidationError
        #      if a field is missing or doesn't match the schema (e.g. bad Literal value)
        #   4. Return the validated CustomerRequest on success
        #   5. On ValidationError or JSON decode error, save it as last_error and retry
        #      (with a short delay) instead of raising immediately
        raise NotImplementedError("TODO: implement extract_customer_request")

    raise RuntimeError(f"Extraction failed after {max_retries + 1} attempts: {last_error}")


if __name__ == "__main__":
    sample = "My debit card was declined twice today, this is urgent, I need to pay rent."
    result = extract_customer_request(sample)
    print(result.model_dump_json(indent=2))
