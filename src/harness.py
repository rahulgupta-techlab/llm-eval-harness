"""Run an LLM / RAG system over test cases and score quality, latency and cost.

    python -m src.harness                       # compare the two built-in stub systems
    python -m src.harness --openai gpt-4o-mini  # evaluate a real model (needs OPENAI_API_KEY)
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from src import metrics

DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Response:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class System(Protocol):
    name: str
    price_per_1k_in: float
    price_per_1k_out: float

    def answer(self, question: str, context: str) -> Response: ...


@dataclass
class CaseResult:
    id: str
    answer: str
    latency_ms: float
    cost_usd: float
    scores: dict[str, float] = field(default_factory=dict)
    passed: bool = False


def score_case(case: dict, answer: str) -> tuple[dict[str, float], bool]:
    """Return metric scores and a pass/fail decision for one case."""
    if case.get("expect_refusal"):
        refused = metrics.is_refusal(answer)
        return {"refusal_correct": float(refused)}, refused

    scores = {
        "exact_match": metrics.exact_match(answer, case["reference"]),
        "token_f1": metrics.token_f1(answer, case["reference"]),
        "keyword_recall": metrics.keyword_recall(answer, case.get("must_include", [])),
        "groundedness": metrics.groundedness(answer, case["context"]),
    }
    passed = (scores["keyword_recall"] == 1.0 and scores["groundedness"] >= 0.6
              and not metrics.is_refusal(answer))
    return scores, passed


def run(system: System, cases: list[dict], clock: Callable[[], float] = time.perf_counter) -> dict:
    results = []
    for case in cases:
        start = clock()
        resp = system.answer(case["question"], case["context"])
        latency_ms = (clock() - start) * 1000
        cost = (resp.input_tokens * system.price_per_1k_in + resp.output_tokens * system.price_per_1k_out) / 1000
        scores, passed = score_case(case, resp.text)
        results.append(CaseResult(case["id"], resp.text, latency_ms, cost, scores, passed))

    answerable = [r for r, c in zip(results, cases) if not c.get("expect_refusal")]
    unanswerable = [r for r, c in zip(results, cases) if c.get("expect_refusal")]
    mean = lambda key: statistics.fmean(r.scores[key] for r in answerable) if answerable else 0.0
    latencies = sorted(r.latency_ms for r in results)
    summary = {
        "system": system.name,
        "cases": len(results),
        "pass_rate": sum(r.passed for r in results) / len(results),
        "token_f1": mean("token_f1"),
        "keyword_recall": mean("keyword_recall"),
        "groundedness": mean("groundedness"),
        "refusal_accuracy": (sum(r.passed for r in unanswerable) / len(unanswerable)) if unanswerable else None,
        "latency_p50_ms": statistics.median(latencies),
        "latency_max_ms": latencies[-1],
        "total_cost_usd": sum(r.cost_usd for r in results),
    }
    return {"summary": summary, "results": [asdict(r) for r in results]}


# ---------------------------------------------------------------- systems under test

def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ExtractiveSystem:
    """Grounded baseline: answers with the context sentence closest to the question, refuses otherwise."""
    name = "extractive-baseline"
    price_per_1k_in = 0.00015
    price_per_1k_out = 0.0006

    def answer(self, question: str, context: str) -> Response:
        q = set(metrics.normalize(question))
        sentences = [s.strip() for s in context.split(".") if s.strip()]
        best = max(sentences, key=lambda s: len(q & set(metrics.normalize(s))))
        overlap = len(q & set(metrics.normalize(best)))
        text = best if overlap >= 2 else "I don't know - this is not mentioned in the context."
        return Response(text, _approx_tokens(question + context), _approx_tokens(text))


class HallucinatingSystem:
    """Deliberately bad system: confident answers with invented details, never refuses."""
    name = "hallucinating-stub"
    price_per_1k_in = 0.0025
    price_per_1k_out = 0.01

    def answer(self, question: str, context: str) -> Response:
        text = f"According to ISO 9001 guidance, {context.split('.')[0].lower()}, reviewed annually by the safety board."
        if "budget" in question.lower():
            text = "The annual maintenance budget is 2.4 million euros."
        return Response(text, _approx_tokens(question + context), _approx_tokens(text))


class OpenAISystem:
    """Real model via the OpenAI SDK (pip install openai, set OPENAI_API_KEY)."""

    def __init__(self, model: str, price_in: float = 0.00015, price_out: float = 0.0006):
        from openai import OpenAI
        self.client, self.model = OpenAI(), model
        self.name, self.price_per_1k_in, self.price_per_1k_out = model, price_in, price_out

    def answer(self, question: str, context: str) -> Response:
        out = self.client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[
                {"role": "system", "content": "Answer only from the context. If the answer is not in it, say \"I don't know\"."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
            ],
        )
        return Response(out.choices[0].message.content or "", out.usage.prompt_tokens, out.usage.completion_tokens)


def format_table(summaries: list[dict]) -> str:
    cols = ["system", "pass_rate", "token_f1", "keyword_recall", "groundedness",
            "refusal_accuracy", "latency_p50_ms", "total_cost_usd"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for s in summaries:
        cells = [s[c] if isinstance(s[c], str) else ("n/a" if s[c] is None else f"{s[c]:.4g}") for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai", metavar="MODEL", help="also evaluate an OpenAI model")
    parser.add_argument("--out", default="report.json")
    args = parser.parse_args()

    cases = json.loads((DATA / "cases.json").read_text(encoding="utf-8"))
    systems: list = [ExtractiveSystem(), HallucinatingSystem()]
    if args.openai:
        systems.append(OpenAISystem(args.openai))

    reports = [run(s, cases) for s in systems]
    Path(args.out).write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(format_table([r["summary"] for r in reports]))
    print(f"\nPer-case results written to {args.out}")


if __name__ == "__main__":
    main()
