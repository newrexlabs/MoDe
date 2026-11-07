<div align="center">
  <img src="docs/figures/newrexlabs-banner.png" alt="newrexlabs" style="width: 100%; max-width: 100%; height: auto; display: block; margin: 0;" />
  <div align="center" style="display: flex; justify-content: center; align-items: center; gap: 1.25em; margin: 8px 0 1.7rem; padding: 0; line-height: 1.3;">
    <a href="https://github.com/newrexlabs/MoDe" style="display: inline-flex; align-items: center; gap: 6px;"><img src="data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%2216%22 height=%2216%22 viewBox=%220 0 24 24%22 fill=%22none%22 stroke-width=%222.2%22 stroke-linecap=%22round%22 stroke-linejoin=%22round%22%3E%3Cstyle%3Epath%7Bstroke:%23000%7D%40media (prefers-color-scheme:dark)%7Bpath%7Bstroke:%23d1d5dc%7D%7D%3C/style%3E%3Cpath d=%22m8 6-6 6 6 6m8-12 6 6-6 6m-2-14-4 16%22/%3E%3C/svg%3E" width="16" height="16" alt="" aria-hidden="true" style="display: block; flex: 0 0 16px; width: 16px; height: 16px; margin: 0;" /><strong>Code</strong></a>
    <a href="https://huggingface.co/collections/newrexlabs/mode" style="display: inline-flex; align-items: center; gap: 6px;"><img src="data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%2216%22 height=%2216%22 viewBox=%220 0 24 24%22 fill=%22none%22 stroke-width=%222.2%22 stroke-linecap=%22round%22 stroke-linejoin=%22round%22%3E%3Cstyle%3Epath%7Bstroke:%23000%7D%40media (prefers-color-scheme:dark)%7Bpath%7Bstroke:%23d1d5dc%7D%7D%3C/style%3E%3Cpath d=%22m12 2 10 5-10 5L2 7l10-5Zm-10 10 10 5 10-5M2 17l10 5 10-5%22/%3E%3C/svg%3E" width="16" height="16" alt="" aria-hidden="true" style="display: block; flex: 0 0 16px; width: 16px; height: 16px; margin: 0;" /><strong>Models</strong></a>
    <a href="https://huggingface.co/datasets/newrexlabs/decision-index-results" style="display: inline-flex; align-items: center; gap: 6px;"><img src="data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%2216%22 height=%2216%22 viewBox=%220 0 24 24%22 fill=%22none%22 stroke-width=%222.2%22 stroke-linecap=%22round%22 stroke-linejoin=%22round%22%3E%3Cstyle%3Epath%7Bstroke:%23000%7D%40media (prefers-color-scheme:dark)%7Bpath%7Bstroke:%23d1d5dc%7D%7D%3C/style%3E%3Cpath d=%22M3 3v18h18M7 15v2m4-6v6m4-9v9m4-12v12%22/%3E%3C/svg%3E" width="16" height="16" alt="" aria-hidden="true" style="display: block; flex: 0 0 16px; width: 16px; height: 16px; margin: 0;" /><strong>Results</strong></a>
    <a href="https://huggingface.co/spaces/multimodalart/jev-decision-index" style="display: inline-flex; align-items: center; gap: 6px;"><img src="data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%2216%22 height=%2216%22 viewBox=%220 0 24 24%22 fill=%22none%22 stroke-width=%222.2%22 stroke-linecap=%22round%22 stroke-linejoin=%22round%22%3E%3Cstyle%3Epath%7Bstroke:%23000%7D%40media (prefers-color-scheme:dark)%7Bpath%7Bstroke:%23d1d5dc%7D%7D%3C/style%3E%3Cpath d=%22M8 21h8m-4-4v4M7 4h10v5a5 5 0 0 1-10 0V4Zm0 2H4v1a3 3 0 0 0 3 3m10-4h3v1a3 3 0 0 1-3 3%22/%3E%3C/svg%3E" width="16" height="16" alt="" aria-hidden="true" style="display: block; flex: 0 0 16px; width: 16px; height: 16px; margin: 0;" /><strong>Decision Index</strong></a>
  </div>
</div>

# MoDe: Mixture of Deciders

MoDe is a family of **sparse mixture-of-experts decision models for text and JSON**, built on Meta's
[MobileMoE](https://huggingface.co/facebook/MobileMoE-M-Base). They return calibrated answers in one forward pass, without
generating tokens. This repository is the inference package; training code and data are not part of the release.

- **Fast:** MoDe-S has the lowest median, mean and p95 latency of 17 open decision models we measured
  (11.0 ms median). MoDe-M is faster and more accurate than [vLLM-SR's d3-nano](https://huggingface.co/vllm-sr/d3-nano), and MoDe-L matches [Liquid AI's d1-2B](https://huggingface.co/LiquidAI/d1-2B) on the
  [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) with a far steadier latency. All three run 1.7x to 5x faster than [GLiDE](https://huggingface.co/fastino/GLiDE-2B) and
  [Intern-Decision](https://huggingface.co/internlm/Intern-Decision-4B).
- **Sparse:** each layer has 60 routed experts and one shared expert; a token uses the shared expert and 4 routed ones, so even
  MoDe-L runs under 1B parameters per token, with any number of questions about one state answered in one pass.
- **Calibrated:** probabilities are temperature-scaled per question type, so they work as confidence scores.

## Models
<hr style="margin: -0.7em 0 1.2em 0; border: 0; border-top: 1px solid rgba(127, 127, 127, 0.3);" />

<table style="width: 100%; display: table; border: 0; border-radius: 0; border-collapse: collapse;">
<thead style="border: 0;">
<tr style="border: 0;">
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Model</th>
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Parameters (active)</th>
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Layers</th>
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Decision Index</th>
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Median latency</th>
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Use it for</th>
</tr>
</thead>
<tbody style="border: 0;">
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://huggingface.co/newrexlabs/MoDe-S"><strong>MoDe-S</strong></a></td>
<td style="border: 0; padding: 0.55em 0.75em;">1.27B (0.28B)</td>
<td style="border: 0; padding: 0.55em 0.75em;">20</td>
<td style="border: 0; padding: 0.55em 0.75em;">42.02</td>
<td style="border: 0; padding: 0.55em 0.75em;">11.0 ms</td>
<td style="border: 0; padding: 0.55em 0.75em;">the fastest</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://huggingface.co/newrexlabs/MoDe-M"><strong>MoDe-M</strong></a></td>
<td style="border: 0; padding: 0.55em 0.75em;">2.82B (0.53B)</td>
<td style="border: 0; padding: 0.55em 0.75em;">26</td>
<td style="border: 0; padding: 0.55em 0.75em;">47.10</td>
<td style="border: 0; padding: 0.55em 0.75em;">17.7 ms</td>
<td style="border: 0; padding: 0.55em 0.75em;">the balance of speed and accuracy</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://huggingface.co/newrexlabs/MoDe-L"><strong>MoDe-L</strong></a></td>
<td style="border: 0; padding: 0.55em 0.75em;">5.33B (0.93B)</td>
<td style="border: 0; padding: 0.55em 0.75em;">32</td>
<td style="border: 0; padding: 0.55em 0.75em;">51.36</td>
<td style="border: 0; padding: 0.55em 0.75em;">25.0 ms</td>
<td style="border: 0; padding: 0.55em 0.75em;">the most accurate</td>
</tr>
</tbody>
</table>

All three share the request format, this package and the calibration method. Each is a decision fine-tune of the
MobileMoE model of the same size with an option-scoring head: a context of 8,192 tokens per forward pass (longer requests are
split by question, never truncated), a 128,256-token vocabulary, English text in, a calibrated probability for every option
of every question out.

MoDe is designed for any pipeline step that needs a decision: routing and triage, intent and topic classification,
moderation, tool selection, and judging. It is not a chat model. Knowledge-heavy questions and multi-step reasoning favor
larger models (see the knowledge scores below).

## Inference
<hr style="margin: -0.7em 0 1.2em 0; border: 0; border-top: 1px solid rgba(127, 127, 127, 0.3);" />

Every size runs with this package, which loads the weights, prepares the GPU, and reads the answers from one
forward pass, so there are no sampling parameters to set.

**Quick start:**

```bash
pip install "MoDe @ git+https://github.com/newrexlabs/MoDe"
huggingface-cli login   # the weights are gated: accept the licence on the model page first
```

```python
from mode_decider import MoDe

model = MoDe.from_pretrained("newrexlabs/MoDe-M")

# Several named questions over one state, answered in one pass
decision = model.decide(
    state={"ticket": "I was charged twice for order #4417. I need the duplicate refunded before my rent is due on Friday.",
           "customer": {"plan": "premium", "orders_last_year": 23}},
    questions={
        "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Payments, refunds and invoices", "shipping": "Deliveries and returns",
                              "tech": "App, login and account problems"}},
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},  # yes/no
        "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["Routine", "Soon", "Today"]},
    },
)
```

```text
team     billing    ████████████████████████ 0.999  ◀ decision
         shipping   ░░░░░░░░░░░░░░░░░░░░░░░░ 0.001
         tech       ░░░░░░░░░░░░░░░░░░░░░░░░ 0.000

refund   true       ████████████████████████ 0.997  ◀ decision
         false      ░░░░░░░░░░░░░░░░░░░░░░░░ 0.003

urgency  0 Routine  ░░░░░░░░░░░░░░░░░░░░░░░░ 0.020
         1 Soon     ██████████████████░░░░░░ 0.765  ◀ decision
         2 Today    █████░░░░░░░░░░░░░░░░░░░ 0.215
```

<details>
<summary><strong>Many requests in one call</strong></summary>

```python
tickets = ["Where is my parcel? It was due Monday.", "The app crashes when I open settings."]
team = {"team": {"type": "choice", "criteria": {"billing": "Payments", "shipping": "Deliveries", "tech": "App problems"}}}
for decision in model.decide_batch([{"state": t, "questions": team} for t in tickets]):
    print(max(decision["team"], key=decision["team"].get))
```

</details>

<details>
<summary><strong>How inference works</strong></summary>

1. The request becomes one token sequence: `STATE: <state>` (a non-string state as compact JSON with sorted keys), then for
   each question a line `[Q<i>|<type>] <instructions>` followed by one line per option, `(<j>) <option id>: <description>`.
2. One causal forward pass through the backbone. On a GPU, a single request of up to 8,192 tokens is replayed from a CUDA
   graph captured at load.
3. The hidden state is read at the end of each question line and each option line. A small pointer head (one transformer
   layer) lets a question's options attend to each other and to the question, then scores every option.
4. Each question's scores are divided by its type's temperature, and a softmax is taken over that question's options.

Nothing is generated or sampled, and no answer is inserted into the input. A request longer than the context is answered in
smaller groups of questions, each with the full state; a single question that does not fit on its own is rejected.

</details>

<details>
<summary><strong>Calls</strong></summary>

<table style="width: 100%; display: table; border: 0; border-radius: 0; border-collapse: collapse;">
<thead style="border: 0;">
<tr style="border: 0;">
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Call</th>
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">What it does</th>
</tr>
</thead>
<tbody style="border: 0;">
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><code>MoDe.from_pretrained(name, revision=None, device=None, cuda_graphs=None)</code></td>
<td style="border: 0; padding: 0.55em 0.75em;">Loads a model from the Hub or a folder. On a GPU it captures CUDA graphs at load (<code>cuda_graphs=False</code> skips this).</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><code>decide(state, questions)</code></td>
<td style="border: 0; padding: 0.55em 0.75em;">Named questions over one state, in one pass.</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><code>decide_batch([{"state": ..., "questions": ...}, ...], batch_size=32)</code></td>
<td style="border: 0; padding: 0.55em 0.75em;">Many requests, batched by length.</td>
</tr>
</tbody>
</table>

`decide` returns `{question_id: {option_id: probability}}`, in request order; each question's probabilities sum to 1.

</details>

<details>
<summary><strong>Question types</strong></summary>

<table style="width: 100%; display: table; border: 0; border-radius: 0; border-collapse: collapse;">
<thead style="border: 0;">
<tr style="border: 0;">
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);"><code>type</code></th>
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Question</th>
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);"><code>criteria</code></th>
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Answer keys</th>
</tr>
</thead>
<tbody style="border: 0;">
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><code>noul</code></td>
<td style="border: 0; padding: 0.55em 0.75em;">Yes or no</td>
<td style="border: 0; padding: 0.55em 0.75em;">optional descriptions of <code>true</code> and <code>false</code></td>
<td style="border: 0; padding: 0.55em 0.75em;"><code>true</code>, <code>false</code></td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><code>choice</code></td>
<td style="border: 0; padding: 0.55em 0.75em;">One of several options</td>
<td style="border: 0; padding: 0.55em 0.75em;">ordered <code>{option id: description}</code> (or <code>null</code>)</td>
<td style="border: 0; padding: 0.55em 0.75em;">the option ids, in order</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><code>score</code></td>
<td style="border: 0; padding: 0.55em 0.75em;">A point on an ordered scale</td>
<td style="border: 0; padding: 0.55em 0.75em;">a list, lowest first</td>
<td style="border: 0; padding: 0.55em 0.75em;"><code>"0"</code>, <code>"1"</code>, …</td>
</tr>
</tbody>
</table>

Supply at least one question; `choice` and `score` need at least two options. Calibrate any decision threshold on data from
your own use.

</details>

<details>
<summary><strong>Calibration</strong></summary>

`p = softmax(scores / T)`, with one temperature per question type, fitted for this checkpoint by minimising log-loss on
held-out calibration records (never a benchmark test set) and clamped to T ≥ 1. It changes confidence, never which option
is most likely. Stored in `mode_config.json`:

<table style="width: 100%; display: table; border: 0; border-radius: 0; border-collapse: collapse;">
<thead style="border: 0;">
<tr style="border: 0;">
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);"><code>noul</code></th>
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);"><code>choice</code></th>
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);"><code>score</code></th>
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">other</th>
</tr>
</thead>
<tbody style="border: 0;">
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;">1.156</td>
<td style="border: 0; padding: 0.55em 0.75em;">1.046</td>
<td style="border: 0; padding: 0.55em 0.75em;">1.000</td>
<td style="border: 0; padding: 0.55em 0.75em;">1.036</td>
</tr>
</tbody>
</table>

</details>

## Performance
<hr style="margin: -0.7em 0 1.2em 0; border: 0; border-top: 1px solid rgba(127, 127, 127, 0.3);" />

### Benchmarks

Accuracy rises with size on the Decision Index and on all five of its areas. On the Intern-Decision suites, MoDe-M and
MoDe-L are level (77.26 and 76.98 on average), each ahead on three of the seven.

<div role="region" tabindex="0" style="width: 100%; max-width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch;">
<table style="width: 100%; display: table; border: 0; border-radius: 0; border-collapse: collapse;">
<thead style="border: 0;">
<tr style="border: 0;">
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Benchmark</th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">MoDe<br>S</th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">MoDe<br>M</th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">MoDe<br>L</th>
</tr>
</thead>
<tbody style="border: 0;">
<tr style="border: 0;"><th scope="rowgroup" style="text-align: left; background-color: transparent; padding: 0.55em 0.75em; border: 0; font-weight: 600; padding-top: 1em; padding-bottom: 0.5em;">Decision Index</th><td aria-hidden="true" style="border: 0; padding: 0.55em 0.75em;"></td><td aria-hidden="true" style="border: 0; padding: 0.55em 0.75em;"></td><td aria-hidden="true" style="border: 0; padding: 0.55em 0.75em;"></td></tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://github.com/apolinario/decision-index">Decision Index</a> public suite (v0.3)</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">42.02</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>47.10</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>51.36</strong></td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;">&nbsp;&nbsp;Knowledge</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">18.40</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>20.60</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>24.80</strong></td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;">&nbsp;&nbsp;Language</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">40.90</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>46.60</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>51.30</strong></td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;">&nbsp;&nbsp;Retrieval</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">54.50</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>62.50</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>65.30</strong></td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;">&nbsp;&nbsp;Tools</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">61.60</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>68.40</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>75.20</strong></td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;">&nbsp;&nbsp;Arts</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">45.10</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>47.00</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>48.80</strong></td>
</tr>
</tbody>
<tbody style="border: 0;">
<tr style="border: 0;"><th scope="rowgroup" style="text-align: left; background-color: transparent; padding: 0.55em 0.75em; border: 0; font-weight: 600; padding-top: 1em; padding-bottom: 0.5em; border-top: 1px solid rgba(127, 127, 127, 0.25);"><a href="https://huggingface.co/internlm/Intern-Decision-4B">Intern-Decision suites</a></th><td aria-hidden="true" style="border: 0; padding: 0.55em 0.75em; border-top: 1px solid rgba(127, 127, 127, 0.25);"></td><td aria-hidden="true" style="border: 0; padding: 0.55em 0.75em; border-top: 1px solid rgba(127, 127, 127, 0.25);"></td><td aria-hidden="true" style="border: 0; padding: 0.55em 0.75em; border-top: 1px solid rgba(127, 127, 127, 0.25);"></td></tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;">Jevbench Easy</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>93.75</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>100.00</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>100.00</strong></td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;">Jevbench Original</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">84.72</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>87.50</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>90.28</strong></td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;">Jevbench Hard</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">36.94</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>47.75</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>44.14</u></td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;">Typed Decision</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">64.80</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>65.50</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>66.25</strong></td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;">ToolACE</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">76.45</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>78.71</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>78.06</u></td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;">AG News</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">89.28</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>89.51</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>89.91</strong></td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;">WildJailBreak</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">64.52</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>71.86</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>70.18</u></td>
</tr>
</tbody>
<tbody style="border: 0;">
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><strong>Seven-suite average</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">72.92</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>77.26</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>76.98</u></td>
</tr>
</tbody>
</table>
</div>

<details>
<summary><small>Evaluation details</small></summary>

<p><small>Decision Index: all 140,178 requests of the public test set (v0.3), run with the benchmark's own evaluation tool on
the released weights; 4 requests whose single question is longer than 8,192 tokens are refused and scored as wrong.
Per-request results are in <a href="https://huggingface.co/datasets/newrexlabs/decision-index-results">newrexlabs/decision-index-results</a>; not yet verified by the benchmark's
maintainers. Intern-Decision suites: the seven accuracy suites of the Intern-Decision cards, scored by their own harness and
scorer unchanged (internlm/Intern-Decision @ 5653502, jevbench @ 7ce310c). None of their items appears verbatim in MoDe's
training data; scored only on items without heavy training overlap, each size's average moves by at most 0.1.</small></p>

</details>

### Speed

All models below ran on the same card, a split (MIG) NVIDIA RTX PRO 6000, one request at a time, each with its authors' own
inference engine. Sorted by median latency per request.

<div role="region" tabindex="0" style="width: 100%; max-width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch;">
<table style="width: 100%; display: table; border: 0; border-radius: 0; border-collapse: collapse;">
<thead style="border: 0;">
<tr style="border: 0;">
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Model</th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Active params</th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Decision Index ↑</th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Median ↓</th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Mean ↓</th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">p95 ↓</th>
</tr>
</thead>
<tbody style="border: 0;">
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em; background-color: rgba(70, 90, 200, 0.09);"><strong><a href="https://huggingface.co/newrexlabs/MoDe-S">MoDe-S</a></strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em; background-color: rgba(70, 90, 200, 0.09);">0.28B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em; background-color: rgba(70, 90, 200, 0.09);">42.02</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em; background-color: rgba(70, 90, 200, 0.09);"><strong>11.0 ms</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em; background-color: rgba(70, 90, 200, 0.09);"><strong>19.1 ms</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em; background-color: rgba(70, 90, 200, 0.09);"><strong>42.0 ms</strong></td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://huggingface.co/LiquidAI/d1-2B">Liquid AI d1-2B</a><sup>c</sup></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">1.92B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">51.10</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>16.2 ms</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">159.1 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">60.1 ms</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://huggingface.co/LiquidAI/d1-Omni-2B">Liquid AI d1-Omni-2B</a><sup>c</sup></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">2.17B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">49.02</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>16.2 ms</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">160.3 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">60.3 ms</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em; background-color: rgba(70, 90, 200, 0.09);"><strong><a href="https://huggingface.co/newrexlabs/MoDe-M">MoDe-M</a></strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em; background-color: rgba(70, 90, 200, 0.09);">0.53B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em; background-color: rgba(70, 90, 200, 0.09);">47.10</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em; background-color: rgba(70, 90, 200, 0.09);">17.7 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em; background-color: rgba(70, 90, 200, 0.09);"><u>29.8 ms</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em; background-color: rgba(70, 90, 200, 0.09);">71.8 ms</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://huggingface.co/vllm-sr/d3-edge">vLLM-SR d3-edge</a></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">0.59B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">28.82</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">19.3 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">39.4 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">63.1 ms</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em; background-color: rgba(70, 90, 200, 0.09);"><strong><a href="https://huggingface.co/newrexlabs/MoDe-L">MoDe-L</a></strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em; background-color: rgba(70, 90, 200, 0.09);">0.93B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em; background-color: rgba(70, 90, 200, 0.09);">51.36</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em; background-color: rgba(70, 90, 200, 0.09);">25.0 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em; background-color: rgba(70, 90, 200, 0.09);">45.2 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em; background-color: rgba(70, 90, 200, 0.09);">107.6 ms</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://huggingface.co/OmniJev/OneJev1.6-4B">OneJev1.6-4B</a></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">4.54B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>61.95</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">35.3 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">117.8 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">290.3 ms</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://huggingface.co/vllm-sr/d3-nano">vLLM-SR d3-nano</a></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">2.21B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">42.22</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">40.1 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">86.2 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">178.5 ms</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://huggingface.co/chiranko/PKU-KCL-Meteor-0.8B">PKU-KCL-Meteor-0.8B</a></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">0.85B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">58.77</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">40.3 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">55.0 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">107.7 ms</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://huggingface.co/vllm-sr/d3-lite">vLLM-SR d3-lite</a></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">0.85B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">36.93</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">40.6 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">70.1 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">107.7 ms</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://huggingface.co/chiranko/PKU-KCL-Meteor-2B">PKU-KCL-Meteor-2B</a></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">2.21B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>60.95</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">40.8 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">69.4 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">178.3 ms</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://huggingface.co/internlm/Intern-Decision-2B">Intern-Decision-2B</a><sup>r</sup></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">2.27B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">19.26</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">41.4 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">44.2 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">75.6 ms</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://huggingface.co/internlm/Intern-Decision-0.8B">Intern-Decision-0.8B</a><sup>r</sup></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">0.87B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">11.81</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">41.4 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">40.9 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>47.7 ms</u></td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://huggingface.co/fastino/GLiDE-2B">GLiDE 2B</a><sup>h</sup></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">1.88B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">60.64</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">41.9 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">67.4 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">176.2 ms</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://huggingface.co/fastino/GLiDE-0.8B">GLiDE 0.8B</a><sup>h</sup></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">0.75B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">54.32</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">42.2 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">54.5 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">107.8 ms</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://huggingface.co/vllm-sr/d3-mini">vLLM-SR d3-mini</a></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">4.54B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">54.90</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">54.2 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">178.0 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">442.4 ms</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://huggingface.co/internlm/Intern-Decision-4B">Intern-Decision-4B</a><sup>r</sup></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">4.66B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">38.00</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">55.2 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">71.0 ms</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">191.1 ms</td>
</tr>
</tbody>
</table>
</div>

<sub><strong>Bold</strong>: best in the column; <u>underline</u>: second best. MoDe rows are shaded.
<sup>c</sup> compiled with <code>model.compile()</code>, the d1 card's fast path; the mean includes recompiles on new input shapes.
<sup>h</sup> served over HTTP by the authors' own server, as submitted to the Decision Index.
<sup>r</sup> refused 70 of the 700 requests (over 16 questions or 8,192 tokens); timed on the rest, which are the shorter ones.</sub>

<details>
<summary><small>Measurement details</small></summary>

<p><small>Five independent runs per model of the Decision Index board's latency design: 700 timed requests sampled by
benchmark and length (seed 20260926), after 10 untimed warm-up requests, timed by the benchmark's own runner, bf16. Each
model used the engine and settings its authors submitted or document (MoDe: CUDA graphs). The median shown is the median of
the five run medians; MoDe's runs differ by under 0.6 ms. The card is split with MIG (2g.48gb, half the GPU), so these are
not the board's full-card numbers. Decision Index scores are the board's public 0.3 scores; MoDe's are our full runs.</small></p>

</details>

## License and acknowledgment
<hr style="margin: -0.7em 0 1.2em 0; border: 0; border-top: 1px solid rgba(127, 127, 127, 0.3);" />

The code in this repository is Apache-2.0 ([LICENSE](LICENSE)). The model weights are derived from Meta's MobileMoE and are
released under the FAIR Noncommercial Research License: noncommercial research use only, subject to the FAIR Acceptable Use
Policy; see each model page. We thank Meta for the MobileMoE models; please acknowledge MobileMoE in any publication that uses
MoDe. MoDe was built with help from Claude Opus 5.5 (Anthropic).

## Citation
<hr style="margin: -0.7em 0 1.2em 0; border: 0; border-top: 1px solid rgba(127, 127, 127, 0.3);" />

```bibtex
@misc{mode_2026,
  title        = {{MoDe}: Mixture of Deciders},
  author       = {{newrexlabs}},
  year         = {2026},
  howpublished = {\url{https://github.com/newrexlabs/MoDe}}
}
```

For questions and feedback, open an issue on [GitHub](https://github.com/newrexlabs/MoDe/issues).
