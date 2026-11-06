# MoDe

MoDe (Mixture of Deciders) is a family of small mixture-of-experts models that answer decision questions. You give it a
state (any JSON) and a set of typed questions; it returns a probability for every option of every question, in one forward
pass, with no text generation.

| Model | Base | Weights |
|---|---|---|
| [MoDe-S](https://huggingface.co/newrexlabs/MoDe-S) | MobileMoE-S | 2.4 GB |
| [MoDe-M](https://huggingface.co/newrexlabs/MoDe-M) | MobileMoE-M | 5.3 GB |
| [MoDe-L](https://huggingface.co/newrexlabs/MoDe-L) | MobileMoE-L | 10 GB |

## Install

```bash
pip install "MoDe @ git+https://github.com/newrexlabs/MoDe"
```

The weights are gated: accept the licence on the model page, then `huggingface-cli login`.

## Use

```python
from mode_decider import MoDe

model = MoDe.from_pretrained("newrexlabs/MoDe-M")

state = {"ticket": "My card was charged twice for the same order.", "customer_tier": "gold"}
questions = {
    "refund": {"type": "noul", "instructions": "Should we refund the duplicate charge?"},
    "team": {"type": "choice", "instructions": "Which team handles this?",
             "criteria": {"billing": "payments and refunds", "shipping": None, "tech": "app and login problems"}},
    "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["low", "medium", "high"]},
}

model.decide(state, questions)
# {"refund": {"true": ..., "false": ...}, "team": {"billing": ..., "shipping": ..., "tech": ...},
#  "urgency": {"0": ..., "1": ..., "2": ...}}
```

Question types:

- `noul`: yes/no. Options `true` and `false`.
- `choice`: pick one. `criteria` maps each option id to a description (or `null`).
- `score`: an ordered scale. `criteria` is a list; option ids are the positions `"0"`, `"1"`, ...

`model.decide_batch([{"state": ..., "questions": ...}, ...])` answers many requests at once.

The context is 8,192 tokens. A request that is too long is answered in smaller groups of questions, each with the full
state; a single question that does not fit raises `Unsupported`. Input is never truncated.

On a GPU, CUDA graphs are captured when the model loads (pass `cuda_graphs=False` to skip). It also runs on CPU.

## Decision Index

MoDe ships an engine for the [Decision Index](https://github.com/apolinario/decision-index) kit:

```bash
pip install "MoDe[eval] @ git+https://github.com/newrexlabs/MoDe"
python -m decision_index pipeline --engine mode_decider.di_engine:MoDeEngine --option model=newrexlabs/MoDe-M --out runs/mode-m
```

## Licence

The code in this repository is Apache-2.0. The model weights are derived from Meta's MobileMoE and are released under the
FAIR Noncommercial Research License (noncommercial research use only); see each model page.

MoDe was built with help from Claude Opus 5.5 (Anthropic).
