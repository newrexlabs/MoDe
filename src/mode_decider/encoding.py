"""Turning a request into one token sequence, and a batch into the index tensors the head reads.

    <bos>STATE: {state}
    [Q1|noul] {instructions}      <- question vector: last token of this line
    (1) true: {description}       <- option vectors: last token of each option line
    (2) false: {description}
    [Q2|choice] ...

Each piece is tokenized separately and the ids are concatenated, so positions are exact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch

from mode_decider.schema import InputTooLong, option_ids

QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
NOUL_DEFAULT = {"true": "yes, the statement holds", "false": "no, the statement does not hold"}


def render(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def options(question: dict[str, Any]) -> list[tuple[str, Any]]:
    if question["type"] == "noul":
        criteria = dict(NOUL_DEFAULT) | dict(question.get("criteria") or {})
        return [("true", criteria["true"]), ("false", criteria["false"])]
    if question["type"] == "choice":
        return [(str(k), v) for k, v in question["criteria"].items()]
    return [(str(i), v) for i, v in enumerate(question["criteria"])]


@dataclass(frozen=True)
class Encoded:
    input_ids: tuple[int, ...]
    question_ids: tuple[str, ...]
    question_types: tuple[int, ...]
    question_pos: tuple[int, ...]
    option_pos: tuple[tuple[int, ...], ...]
    option_ids: tuple[tuple[str, ...], ...]


def encode(tokenizer, state: Any, questions: dict[str, Any], max_length: int) -> Encoded:
    def tok(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False).input_ids

    ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    ids += tok("STATE: " + render(state) + "\n")
    qids, qtypes, qpos, opos, oids = [], [], [], [], []
    for index, (qid, question) in enumerate(questions.items()):
        ids += tok(f"[Q{index + 1}|{question['type']}] {render(question.get('instructions') or qid)}\n")
        qpos.append(len(ids) - 1)
        positions, keys = [], []
        for j, (option_id, description) in enumerate(options(question)):
            ids += tok(f"({j + 1}) {option_id}" + (f": {render(description)}" if description is not None else "") + "\n")
            positions.append(len(ids) - 1)
            keys.append(option_id)
        assert tuple(keys) == option_ids(question)
        qids.append(str(qid))
        qtypes.append(QUESTION_TYPES[question["type"]])
        opos.append(tuple(positions))
        oids.append(tuple(keys))
    if len(ids) > max_length:
        raise InputTooLong(len(ids), max_length)
    return Encoded(tuple(ids), tuple(qids), tuple(qtypes), tuple(qpos), tuple(opos), tuple(oids))


def collate(records: list[Encoded], pad_id: int, device) -> dict[str, Any]:
    """Right-padded ids plus flat gather indices for every question and option in the batch."""
    width = max(len(r.input_ids) for r in records)
    input_ids = torch.full((len(records), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(records), width), dtype=torch.long)
    q_flat, q_type, slot_flat, slot_q, slot_k = [], [], [], [], []
    k_max = max(len(p) for r in records for p in r.option_pos)
    q_index = 0
    for b, r in enumerate(records):
        input_ids[b, : len(r.input_ids)] = torch.tensor(r.input_ids)
        mask[b, : len(r.input_ids)] = 1
        for qt, qp, positions in zip(r.question_types, r.question_pos, r.option_pos):
            q_flat.append(b * width + qp)
            q_type.append(qt)
            for k, p in enumerate(positions):
                slot_flat.append(b * width + p)
                slot_q.append(q_index)
                slot_k.append(k)
            q_index += 1

    def t(x):
        return torch.tensor(x, dtype=torch.long, device=device)

    option_mask = torch.zeros((q_index, k_max), dtype=torch.bool, device=device)
    option_mask[t(slot_q), t(slot_k)] = True
    return {"input_ids": input_ids.to(device), "attention_mask": mask.to(device), "records": records,
            "q_flat": t(q_flat), "q_type": t(q_type), "slot_flat": t(slot_flat), "slot_q": t(slot_q),
            "slot_k": t(slot_k), "option_mask": option_mask}


def unpad(logits: torch.Tensor, batch: dict[str, Any]) -> list[dict[str, dict[str, float]]]:
    """Padded logits [questions, max options] -> per request {question_id: {option_id: logit}}."""
    rows = logits.detach().float().cpu().tolist()
    out, q = [], 0
    for r in batch["records"]:
        decision = {}
        for qid, keys in zip(r.question_ids, r.option_ids):
            decision[qid] = dict(zip(keys, rows[q][: len(keys)]))
            q += 1
        out.append(decision)
    return out
