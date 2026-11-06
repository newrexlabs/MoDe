"""Request format: ``{"state": <any JSON>, "questions": {question_id: question}}``.

A question has a ``type`` (``noul`` = yes/no, ``choice``, ``score``), optional ``instructions``, and ``criteria``:
a mapping of option id to description (or null) for ``choice``, an ordered list for ``score``, optional descriptions
of ``true`` / ``false`` for ``noul``. MoDe returns a probability for every option of every question.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

QUESTION_TYPES = ("noul", "choice", "score")
NOUL_OPTIONS = ("true", "false")


class InvalidRequest(ValueError):
    pass


class InputTooLong(ValueError):
    """The encoded request is longer than the model's context. MoDe never truncates input."""

    def __init__(self, tokens: int, limit: int):
        super().__init__(f"request needs {tokens} tokens; the limit is {limit}")
        self.tokens, self.limit = tokens, limit


def option_ids(question: Mapping[str, Any]) -> tuple[str, ...]:
    """Option ids in request order."""
    if question["type"] == "noul":
        return NOUL_OPTIONS
    if question["type"] == "choice":
        return tuple(str(key) for key in question["criteria"])
    return tuple(str(index) for index in range(len(question["criteria"])))


def validate(state: Any, questions: Mapping[str, Any]) -> None:
    if not isinstance(questions, Mapping) or not questions:
        raise InvalidRequest("a request needs at least one question")
    for qid, question in questions.items():
        kind = question.get("type")
        if kind not in QUESTION_TYPES:
            raise InvalidRequest(f"{qid}: type must be one of {QUESTION_TYPES}, got {kind!r}")
        if kind == "choice" and not isinstance(question.get("criteria"), Mapping):
            raise InvalidRequest(f"{qid}: choice criteria must be a mapping of option id to description")
        if kind == "score" and not isinstance(question.get("criteria"), list):
            raise InvalidRequest(f"{qid}: score criteria must be a list")
        if kind != "noul" and len(question["criteria"]) < 2:
            raise InvalidRequest(f"{qid}: needs at least two options")
