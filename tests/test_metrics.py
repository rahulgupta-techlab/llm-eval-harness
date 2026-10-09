import pytest

from src import metrics


def test_exact_match_ignores_case_punctuation_and_stopwords():
    assert metrics.exact_match("The Lockout/Tagout.", "lockout tagout") == 1.0
    assert metrics.exact_match("lockout", "tagout") == 0.0


def test_token_f1_partial_overlap():
    assert metrics.token_f1("plant manager", "plant manager and on-call engineer") == pytest.approx(2 / 3)  # "on" is a stopword: 2 of 4 reference tokens
    assert metrics.token_f1("nothing shared", "completely different") == 0.0


def test_keyword_recall():
    assert metrics.keyword_recall("Page the plant manager", ["plant manager", "on-call"]) == 0.5
    assert metrics.keyword_recall("anything", []) == 1.0


def test_groundedness_penalises_invented_facts():
    ctx = "Every machine receives preventive maintenance at least every 90 days."
    assert metrics.groundedness("maintenance every 90 days", ctx) == 1.0
    assert metrics.groundedness("maintenance costs 2 million euros", ctx) < 0.5


@pytest.mark.parametrize("text,expected", [
    ("I don't know.", True),
    ("This is not mentioned in the context.", True),
    ("I cannot answer that from the document.", True),
    ("Every 90 days.", False),
])
def test_refusal_detection(text, expected):
    assert metrics.is_refusal(text) is expected
