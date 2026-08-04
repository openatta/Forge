from verify.math_answer import MathAnswerVerifier, _extract_answer_text, _find_boxed_matches
from tests.factories import make_attempt, make_question


def _verify(content: str, gold, verify_method: str = "numeric"):
    q = make_question(id="q-1", verify_method=verify_method, gold=gold)
    a = make_attempt(id="att-1", question_id="q-1", content=content)
    return MathAnswerVerifier().verify(a, q)


def test_numeric_correct_match():
    assert _verify("Let's compute...\nFinal Answer: 4", gold="4").passed is True


def test_numeric_incorrect_match():
    assert _verify("Final Answer: 5", gold="4").passed is False


def test_numeric_algebraic_equivalence():
    assert _verify("Final Answer: 2*2", gold="4").passed is True


def test_numeric_set_order_independent():
    result = _verify("Final Answer: 3, -2", gold="-2, 3", verify_method="numeric_set")
    assert result.passed is True


def test_numeric_set_cardinality_mismatch():
    result = _verify("Final Answer: 3", gold="-2, 3", verify_method="numeric_set")
    assert result.passed is False


def test_boxed_answer_used_over_trailing_prose():
    assert _verify(r"work work \boxed{7} trailing prose", gold="7").passed is True


def test_boxed_last_occurrence_wins():
    assert _verify(r"first try \boxed{3} no wait \boxed{9}", gold="9").passed is True


def test_unit_suffix_is_stripped():
    assert _verify("Final Answer: 20 km", gold="20").passed is True


def test_leading_qualifier_is_stripped():
    assert _verify("Final Answer: about 5", gold="5").passed is True
    assert _verify("Final Answer: ~5", gold="5").passed is True
    assert _verify("Final Answer: ≈5", gold="5").passed is True  # "≈5"


def test_find_boxed_matches_handles_nested_braces():
    """Regression test for the extraction fix: a flat regex like \\boxed\\{([^}]*)\\} stops at the
    first '}' and would return "2^{10" instead of the full "2^{10}"."""
    content = r"so the value is \boxed{2^{10}} which is the final answer"
    assert _find_boxed_matches(content) == ["2^{10}"]


def test_extract_answer_text_prefers_boxed_over_final_answer_line():
    content = "Final Answer: wrong\nactually \\boxed{42}"
    assert _extract_answer_text(content) == "42"


def test_extract_answer_text_falls_back_to_last_nonempty_line():
    content = "some reasoning\n\nthe answer is 8"
    assert _extract_answer_text(content) == "the answer is 8"
