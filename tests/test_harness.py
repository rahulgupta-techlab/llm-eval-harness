import itertools

from src.harness import (DATA, ExtractiveSystem, HallucinatingSystem, Response,
                         format_table, run, score_case)
import json

CASES = json.loads((DATA / "cases.json").read_text(encoding="utf-8"))


def fake_clock(step=0.05):
    counter = itertools.count()
    return lambda: next(counter) * step


class EchoSystem:
    name = "echo"
    price_per_1k_in = 1.0
    price_per_1k_out = 2.0

    def answer(self, question, context):
        return Response(context, input_tokens=1000, output_tokens=500)


def test_refusal_case_scoring():
    case = {"expect_refusal": True}
    assert score_case(case, "I don't know")[1] is True
    assert score_case(case, "It is 5 million")[1] is False


def test_answerable_case_requires_keywords_and_grounding():
    case = CASES[0]
    assert score_case(case, "at least every 90 days")[1] is True
    assert score_case(case, "every 90 days per ISO certified external auditors")[1] is False
    assert score_case(case, "I don't know")[1] is False


def test_latency_and_cost_accounting():
    report = run(EchoSystem(), CASES[:2], clock=fake_clock(0.05))
    s = report["summary"]
    assert abs(s["latency_p50_ms"] - 50.0) < 1e-6
    assert s["total_cost_usd"] == 2 * (1000 * 1.0 + 500 * 2.0) / 1000


def test_baseline_beats_hallucinating_system():
    good = run(ExtractiveSystem(), CASES)["summary"]
    bad = run(HallucinatingSystem(), CASES)["summary"]
    assert good["pass_rate"] > bad["pass_rate"]
    assert good["groundedness"] > bad["groundedness"]
    assert good["refusal_accuracy"] == 1.0 and bad["refusal_accuracy"] == 0.0


def test_report_table_renders():
    table = format_table([run(ExtractiveSystem(), CASES)["summary"]])
    assert table.startswith("| system |") and "extractive-baseline" in table
