# LLM Eval Harness

A small, dependency-free harness for evaluating LLM and RAG systems on **quality, hallucination, refusal behaviour, latency and cost** before they ship.

Each test case has a question, the retrieved context, and either a reference answer or `expect_refusal: true`. The harness runs a system over all cases and reports:

| Metric | What it catches |
| --- | --- |
| `token_f1`, `exact_match` | Answer correctness against a reference (SQuAD-style) |
| `keyword_recall` | Missing required facts (`must_include`) |
| `groundedness` | Share of answer content words supported by the context, a cheap, deterministic hallucination signal |
| `refusal_accuracy` | Whether the system says "I don't know" when the context has no answer |
| `latency_p50_ms`, `latency_max_ms` | Response time |
| `total_cost_usd` | Token usage × price per 1k tokens |

A case **passes** if all required keywords are present, groundedness is ≥ 0.6, and the system didn't refuse an answerable question. For unanswerable cases it passes only if the system refuses.

## Example run

Two built-in systems show what the harness catches:

- `extractive-baseline`: answers with the most relevant context sentence and refuses when nothing matches
- `hallucinating-stub`: fluent but adds invented details ("ISO 9001", "safety board") and never refuses

```
| system | pass_rate | token_f1 | keyword_recall | groundedness | refusal_accuracy | latency_p50_ms | total_cost_usd |
|---|---|---|---|---|---|---|---|
| extractive-baseline | 0.8 | 0.4455 | 0.75 | 0.75 | 1 | 0.0867 | 6.96e-05 |
| hallucinating-stub | 0 | 0.3714 | 1 | 0.4895 | 0 | 0.012 | 0.00198 |
```

The hallucinating system reaches 100% keyword recall but fails every case on groundedness and refusal. Keyword or reference matching alone would have rated it well. The baseline's one failure is a **false refusal** on a paraphrased question, which is a useful regression to track too.

## Evaluate a real model

```bash
pip install openai
export OPENAI_API_KEY=...
python -m src.harness --openai gpt-4o-mini
```

To evaluate any other system (your RAG pipeline, an agent, a local model), implement one method:

```python
class MySystem:
    name = "my-rag"
    price_per_1k_in = 0.0
    price_per_1k_out = 0.0

    def answer(self, question: str, context: str) -> Response:
        ...
```

## Project layout

```
src/
  metrics.py     # exact match, token F1, keyword recall, groundedness, refusal detection
  harness.py     # runner, pass/fail rules, latency + cost accounting, systems, report table
data/
  cases.json     # test cases (incl. one unanswerable case)
tests/
  test_metrics.py
  test_harness.py
```

## Run it

```bash
pip install -r requirements.txt
python -m pytest        # 13 tests
python -m src.harness   # prints the comparison table and writes report.json
```

## Design notes

- The metrics are deterministic and need no API calls, so they can run in CI on every prompt or retriever change. An LLM-as-judge scorer can be added as another metric for nuanced answers.
- The clock is injectable, so latency accounting is tested exactly.
- Refusal behaviour is scored separately from answer quality. Refusing an answerable question and inventing an answer to an unanswerable one are different failures and need different fixes.

## License

MIT
