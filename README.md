![newrexlabs](docs/figures/newrexlabs-banner.png)

# MoDe: Mixture of Deciders

**MoDe** is a family of small mixture-of-experts decision models. Give a model a state (text or any JSON) and the questions
you need answered: pick one of several options, say yes or no, or rate on a scale. It answers all of them in one forward
pass and returns a probability for every option, without generating text. It is called from software, not chatted with.

This repository is the inference package: load a model, ask questions, batch, serve with CUDA graphs, and evaluate on the
Decision Index.

| Model | Parameters (active) | Decision Index | Median latency | Use it for |
| --- | ---: | ---: | ---: | --- |
| [MoDe-S](https://huggingface.co/newrexlabs/MoDe-S) | 1.27B (0.28B) | 42.02 | 12.1 ms | the smallest and fastest |
| [MoDe-M](https://huggingface.co/newrexlabs/MoDe-M) | 2.82B (0.53B) | 47.10 | 19.3 ms | the balance of speed and quality |
| [MoDe-L](https://huggingface.co/newrexlabs/MoDe-L) | 5.33B (0.93B) | 51.36 | 27.5 ms | the most accurate |

## Highlights

- **Few active parameters:** 60 experts, 4 routed per token, so MoDe-L runs under 1B parameters per token.
- **Decision Index:** MoDe-M scores 47.10, +4.9 over [d3-nano](https://huggingface.co/vllm-sr/d3-nano) (2.21B dense); MoDe-S is within 0.2 of it with 0.28B
  parameters active. The [Decision Index](https://github.com/apolinario/decision-index) is an open benchmark of 37 decision
  tasks in five areas; scores are on its full public test set.
- **Fast:** 12-28 ms median per request, one request at a time, on an NVIDIA RTX PRO 6000 Blackwell GPU.
- **Many questions, one pass:** every question about a request is answered from the same forward pass. Long requests are
  split by question automatically; input is never cut.
- **Calibrated:** per-type temperatures, fitted on held-out data, are applied to every answer.

## Install

```bash
pip install "MoDe @ git+https://github.com/newrexlabs/MoDe"
huggingface-cli login   # the weights are gated: accept the licence on the model page first
```

Python 3.10+, PyTorch 2.8+, Transformers 4.57.6.

## Quickstart

```python
from mode_decider import MoDe

model = MoDe.from_pretrained("newrexlabs/MoDe-M")

decision = model.decide(
    state={
        "ticket": "I was charged twice for order #4417. I need the duplicate refunded before my rent is due on Friday.",
        "customer": {"plan": "premium", "orders_last_year": 23},
    },
    questions={
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "Payments, refunds and invoices",
                "shipping": "Deliveries and returns",
                "tech": "App, login and account problems",
            },
        },
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["Routine", "Soon", "Today"]},
    },
)
```

MoDe-M returns, in 16.1 ms on one RTX PRO 6000 Blackwell:

```json
{"team":    {"billing": 0.999, "shipping": 0.001, "tech": 0.0},
 "refund":  {"true": 0.997, "false": 0.003},
 "urgency": {"0": 0.02, "1": 0.765, "2": 0.215}}
```

## Request format

A request is a `state` (any JSON) and `questions`, a mapping of question id to question. Each question has a `type`, optional
`instructions`, and `criteria`:

| Type | Answers | `criteria` | Option ids |
| --- | --- | --- | --- |
| `noul` | yes / no | optional descriptions of `true` and `false` | `true`, `false` |
| `choice` | one of several options | option id → description (or `null`) | the keys, in order |
| `score` | a point on an ordered scale | a list, lowest first | `"0"`, `"1"`, ... |

The result is `{question_id: {option_id: probability}}`, and each question's probabilities sum to 1.

- `model.decide_batch([{"state": ..., "questions": ...}, ...], batch_size=32)` answers many requests, batching those of
  similar length.
- The context is 8,192 tokens. A request that is too long is answered in smaller groups of questions, each with the full
  state; a single question that does not fit on its own raises `Unsupported`. Input is never truncated.
- On a GPU, CUDA graphs are captured when the model loads (`cuda_graphs=False` skips this). The model also runs on CPU.

## Evaluation

Scores on the full public test set of the [Decision Index](https://github.com/apolinario/decision-index) (version 0.3,
140,178 requests), an open benchmark of 37 decision tasks; 0 is random guessing and 100 is perfect.

| Model | Parameters | Active per token | Decision Index ↑ |
| --- | ---: | ---: | ---: |
| [d3-edge](https://huggingface.co/vllm-sr/d3-edge) | 0.59B | 0.59B | 28.82 |
| [d3-lite](https://huggingface.co/vllm-sr/d3-lite) | 0.85B | 0.85B | 36.93 |
| **MoDe-S** | 1.27B | 0.28B | **42.02** |
| [d3-nano](https://huggingface.co/vllm-sr/d3-nano) | 2.21B | 2.21B | 42.22 |
| **MoDe-M** | 2.82B | 0.53B | **47.10** |
| **MoDe-L** | 5.33B | 0.93B | **51.36** |
| [d3-mini](https://huggingface.co/vllm-sr/d3-mini) | 4.54B | 4.54B | 54.90 |

By area:

| Model | Knowledge | Language | Retrieval | Tools | Arts |
| --- | ---: | ---: | ---: | ---: | ---: |
| MoDe-S | 18.4 | 40.9 | 54.5 | 61.6 | 45.1 |
| MoDe-M | 20.6 | 46.6 | 62.5 | 68.4 | 47.0 |
| MoDe-L | 24.8 | 51.3 | 65.3 | 75.2 | 48.8 |

<sub>MoDe: all 140,178 requests of the public test set, run by us with the benchmark's own evaluation tool (version 0.3)
and this package's engine on the released weights; 4 requests whose single question is longer than 8,192 tokens are refused
and scored as wrong. Not yet verified by the benchmark's maintainers. Other models: scores from their own model cards, on the
same test set with the same tool; their parameter counts include a vision encoder. Latency: one request at a time over 1,500
requests from the test set, on half of an NVIDIA RTX PRO 6000 Blackwell GPU (split with MIG), CUDA graphs on.</sub>

To run the benchmark yourself with its evaluation tool:

```bash
pip install "MoDe[eval] @ git+https://github.com/newrexlabs/MoDe"
python -m decision_index pipeline --engine mode_decider.di_engine:MoDeEngine --option model=newrexlabs/MoDe-M --out runs/mode-m
```

## Limitations

- **Text only**, up to 8,192 tokens per request.
- **One pass:** knowledge stays near the base model's level (the weakest area above), and multi-step reasoning is not solved
  by reading harder.
- **Tested in English.**

## License

The code in this repository is Apache-2.0 ([LICENSE](LICENSE)). The model weights are derived from Meta's MobileMoE and are
released under the FAIR Noncommercial Research License: noncommercial research use only, subject to the FAIR Acceptable Use
Policy; see each model page. Please acknowledge MobileMoE in any publication that uses MoDe.

## Citation

```bibtex
@misc{mode_2026,
  title        = {{MoDe}: Mixture of Deciders},
  author       = {{newrexlabs}},
  year         = {2026},
  howpublished = {\url{https://github.com/newrexlabs/MoDe}}
}
```

MoDe was built with help from Claude Opus 5.5 (Anthropic).
