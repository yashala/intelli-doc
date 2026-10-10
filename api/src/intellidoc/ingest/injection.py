"""Flag text that is addressed to a model rather than to a human reader.

Detection does not remove content. Flagged chunks are still retrievable; the answer prompt
marks them so the model treats them as quoted data, and the evaluation measures whether the
model followed them anyway. Pattern matching is easy to evade; see docs/limitations.md.
"""

from __future__ import annotations

import re

PATTERNS: dict[str, re.Pattern[str]] = {
    "override_instructions": re.compile(
        r"\b(ignore|disregard|forget|override)\b[^.]{0,40}"
        r"\b(previous|prior|above|earlier|all|system)\b[^.]{0,20}\b(instructions?|rules|prompts?)\b",
        re.IGNORECASE,
    ),
    "addressed_to_model": re.compile(
        r"\b(note|instructions?|message)s?\s+(to|for)\s+(the\s+)?(ai|llm|language model|chatbot)?"
        r"\s*(assistants?|models?|bots?)\b|\bassistant instructions?\b|\bsystem (note|prompt)\b",
        re.IGNORECASE,
    ),
    "output_control": re.compile(
        r"\b(reply|respond|answer)\s+only\s+with\b|\bdo not cite\b|\bwithout citing\b",
        re.IGNORECASE,
    ),
    "credential_request": re.compile(
        r"\b(email|send|submit|share)\b[^.]{0,60}\b(passwords?|credentials|api keys?|passcodes?)\b",
        re.IGNORECASE,
    ),
    "role_change": re.compile(r"\byou are now\b|\bfrom now on,? you\b", re.IGNORECASE),
}


def scan(text: str) -> list[str]:
    return [name for name, pattern in PATTERNS.items() if pattern.search(text)]
