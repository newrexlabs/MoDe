"""Loading a MoDe model and asking it questions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from mode_decider.encoding import Encoded, collate, encode, unpad
from mode_decider.head import PointerHead
from mode_decider.schema import InputTooLong, validate

TYPE_NAMES = {0: "noul", 1: "choice", 2: "score"}


class Unsupported(ValueError):
    """A single question that does not fit in the context on its own. Nothing is ever truncated."""


class MoDe:
    def __init__(self, backbone, head: PointerHead, tokenizer, config: dict[str, Any], device: torch.device,
                 graphed=None):
        self.backbone, self.head, self.tokenizer, self.config = backbone, head, tokenizer, config
        self.device, self.graphed = device, graphed
        self.max_length = config["max_length"]
        self.temperatures = config.get("temperatures") or {}

    @classmethod
    def from_pretrained(cls, name_or_path: str, revision: str | None = None, device: str | None = None,
                        cuda_graphs: bool | None = None, token: str | None = None) -> MoDe:
        """Load from a Hugging Face repo (e.g. ``newrexlabs/MoDe-M``) or a local folder.

        ``cuda_graphs`` (default: on when a GPU is used) captures the backbone at load for low-latency serving.
        """
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        from mode_decider.moe import use_fast_moe

        path = Path(name_or_path)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(name_or_path, revision=revision, token=token))
        config = json.loads((path / "mode_config.json").read_text())
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 10:
            torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN attention gives wrong masked outputs on Blackwell

        # The models were trained with the base tokenizer exactly as shipped; fix_mistral_regex=False keeps that and silences
        # transformers' suggestion to change it. Length is checked by encode(), so the tokenizer's own limit is lifted.
        tokenizer = AutoTokenizer.from_pretrained(path, fix_mistral_regex=False)
        tokenizer.model_max_length = 10**9
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        backbone = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=dtype).to(device).eval()
        if device.type == "cuda":
            use_fast_moe(backbone)
        head = PointerHead(backbone.config.hidden_size, **config["head"])
        head.load_state_dict(load_file(path / "head.safetensors"))
        head.to(device).eval()

        graphed = None
        if cuda_graphs if cuda_graphs is not None else device.type == "cuda":
            from mode_decider.graphs import GraphedBackbone

            graphed = GraphedBackbone(backbone, tokenizer.pad_token_id, device)
            graphed.precapture()
        return cls(backbone, head, tokenizer, config, device, graphed)

    def decide(self, state: Any, questions: dict[str, Any]) -> dict[str, dict[str, float]]:
        """Probabilities for every option of every question: ``{question_id: {option_id: probability}}``.

        If the request is longer than the context, the questions are answered in smaller groups, each with the
        full state. A question that does not fit on its own raises ``Unsupported``.
        """
        validate(state, questions)
        return self._decide(state, questions)

    def decide_batch(self, requests: list[dict[str, Any]], batch_size: int = 32) -> list[dict[str, dict[str, float]]]:
        """``decide`` for many ``{"state", "questions"}`` requests. Requests of similar length are batched together."""
        encoded: list[Encoded | None] = []
        for r in requests:
            validate(r["state"], r["questions"])
            try:
                encoded.append(encode(self.tokenizer, r["state"], r["questions"], self.max_length))
            except InputTooLong:
                encoded.append(None)
        out: list[Any] = [None] * len(requests)
        ready = sorted((i for i, e in enumerate(encoded) if e is not None), key=lambda i: len(encoded[i].input_ids))
        for start in range(0, len(ready), batch_size):
            chunk = ready[start:start + batch_size]
            for i, decision in zip(chunk, self._run([encoded[i] for i in chunk])):
                out[i] = decision
        for i, e in enumerate(encoded):
            if e is None:
                out[i] = self._decide(requests[i]["state"], requests[i]["questions"])
        return out

    def _decide(self, state: Any, questions: dict[str, Any]) -> dict[str, dict[str, float]]:
        try:
            (decision,) = self._run([encode(self.tokenizer, state, questions, self.max_length)])
            return decision
        except InputTooLong as error:
            if len(questions) == 1:
                raise Unsupported(f"one question needs {error.tokens} tokens; the limit is {error.limit}") from error
            keys = list(questions)
            mid = len(keys) // 2
            out: dict[str, dict[str, float]] = {}
            for part in (keys[:mid], keys[mid:]):
                out.update(self._decide(state, {k: questions[k] for k in part}))
            return {k: out[k] for k in keys}

    @torch.inference_mode()
    def _run(self, encoded: list[Encoded]) -> list[dict[str, dict[str, float]]]:
        batch = collate(encoded, self.tokenizer.pad_token_id, self.device)
        if self.graphed is not None:
            hidden = self.graphed(batch["input_ids"], batch["attention_mask"])
        else:
            hidden = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                   use_cache=False).last_hidden_state
        logits = unpad(self.head(hidden, batch), batch)
        return [self._softmax(e, d) for e, d in zip(encoded, logits)]

    def _softmax(self, encoded: Encoded, logits: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        """Per-type temperature scaling (fitted on held-out data), then softmax."""
        types = dict(zip(encoded.question_ids, encoded.question_types))
        out = {}
        for qid, option_logits in logits.items():
            t = self.temperatures.get(TYPE_NAMES[types[qid]], self.temperatures.get("*", 1.0))
            top = max(option_logits.values())
            exp = {o: math.exp((v - top) / t) for o, v in option_logits.items()}
            total = sum(exp.values())
            out[qid] = {o: v / total for o, v in exp.items()}
        return out
