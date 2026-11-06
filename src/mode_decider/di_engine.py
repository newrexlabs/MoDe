"""MoDe as a Decision Index engine, so the kit's own runner and scorers can evaluate it:

    python -m decision_index pipeline --engine mode_decider.di_engine:MoDeEngine --option model=newrexlabs/MoDe-M --out runs/mode-m

Options: ``model`` (Hub repo or local folder), ``revision``, ``cuda_graphs`` (``true``/``false``, default on with a GPU).
A question too long for the context on its own is reported as unsupported; nothing is truncated.
"""

from __future__ import annotations

from typing import Any

from decision_index.engines import Engine
from decision_index.engines import Unsupported as DIUnsupported

from mode_decider.model import MoDe, Unsupported


class MoDeEngine(Engine):
    name = "mode"
    latency = "In-process request wall time: tokenize + forward + response formatting."

    def __init__(self, model: str, revision: str | None = None, cuda_graphs: str | bool | None = None, **options: Any):
        super().__init__(model=model, **options)
        if isinstance(cuda_graphs, str):
            cuda_graphs = cuda_graphs.lower() in ("1", "true", "yes")
        self.mode = MoDe.from_pretrained(model, revision=revision, cuda_graphs=cuda_graphs)
        self.model_name = model
        self.info = {"engine": "mode", "model": model, "revision": revision, "max_length": self.mode.max_length,
                     "cuda_graphs": self.mode.graphed is not None, "temperatures": self.mode.temperatures}

    def __call__(self, state: Any, questions: dict[str, Any], **_: Any):
        try:
            decision = self.mode.decide(state, questions)
        except Unsupported as error:
            raise DIUnsupported(str(error)) from error
        answers = {}
        for qid, question in questions.items():
            probs = decision[qid]
            if question["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": probs["true"]}
            elif question["type"] == "choice":
                answers[qid] = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}
            else:
                raise DIUnsupported(f"question type {question['type']}")
        return {"model": self.model_name, "answers": answers}, None

    def synchronize(self) -> None:
        if self.mode.device.type == "cuda":
            import torch

            torch.cuda.synchronize(self.mode.device)

    def runtime(self) -> dict[str, Any]:
        return self.info
