import math
from types import SimpleNamespace

import pytest
import torch

from mode_decider.encoding import collate, encode
from mode_decider.head import PointerHead
from mode_decider.model import MoDe, Unsupported
from mode_decider.schema import InputTooLong, InvalidRequest, validate


class CharTokenizer:
    """One token per character, so positions are easy to check."""

    bos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[2 + ord(c) % 250 for c in text])


QUESTIONS = {
    "ok": {"type": "noul", "instructions": "Is it ok?"},
    "pick": {"type": "choice", "criteria": {"a": "first", "b": None, "c": "third"}},
}


def test_encode_reads_the_last_token_of_each_line():
    tokenizer = CharTokenizer()
    e = encode(tokenizer, {"x": 1}, QUESTIONS, max_length=4096)
    text = "STATE: " + '{"x":1}' + "\n"
    q1 = "[Q1|noul] Is it ok?\n"
    assert e.question_pos[0] == 1 + len(text) + len(q1) - 1
    assert e.input_ids[e.question_pos[0]] == tokenizer("\n").input_ids[0]
    assert e.option_ids == (("true", "false"), ("a", "b", "c"))
    assert all(e.input_ids[p] == tokenizer("\n").input_ids[0] for ps in e.option_pos for p in ps)


def test_encode_never_truncates():
    with pytest.raises(InputTooLong):
        encode(CharTokenizer(), "x" * 100, QUESTIONS, max_length=50)


def test_head_masks_missing_options():
    tokenizer = CharTokenizer()
    records = [encode(tokenizer, "s", QUESTIONS, 4096), encode(tokenizer, "longer state", {"q": QUESTIONS["ok"]}, 4096)]
    batch = collate(records, tokenizer.pad_token_id, "cpu")
    hidden = torch.randn(len(records), batch["input_ids"].shape[1], 16)
    logits = PointerHead(16, width=32, heads=4).eval()(hidden, batch)
    assert logits.shape == (3, 3)
    assert torch.isinf(logits[0, 2]) and torch.isinf(logits[2, 2]) and torch.isfinite(logits[1]).all()


def test_validate_rejects_bad_requests():
    with pytest.raises(InvalidRequest):
        validate({}, {})
    with pytest.raises(InvalidRequest):
        validate({}, {"q": {"type": "choice", "criteria": {"only": None}}})
    with pytest.raises(InvalidRequest):
        validate({}, {"q": {"type": "free_text"}})


def fake_model(max_length):
    """A MoDe whose forward is replaced by uniform logits; enough to test splitting and calibration."""
    model = MoDe.__new__(MoDe)
    model.tokenizer, model.max_length, model.temperatures = CharTokenizer(), max_length, {"*": 1.0}

    def run(encoded):
        out = []
        for e in encoded:
            logits = {qid: {o: 0.0 for o in keys} for qid, keys in zip(e.question_ids, e.option_ids)}
            out.append(model._softmax(e, logits))
        return out

    model._run = run
    return model


def test_long_requests_are_split_by_question_and_keep_order():
    questions = {f"q{i}": {"type": "noul", "instructions": "x" * 40} for i in range(6)}
    full = len(encode(CharTokenizer(), "state", questions, 10**6).input_ids)
    model = fake_model(max_length=full // 2)
    decision = model.decide("state", questions)
    assert list(decision) == list(questions)
    assert all(math.isclose(sum(p.values()), 1.0) for p in decision.values())


def test_a_question_too_long_on_its_own_is_unsupported():
    model = fake_model(max_length=30)
    with pytest.raises(Unsupported):
        model.decide("a long state " * 10, {"q": QUESTIONS["ok"]})
