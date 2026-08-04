"""math-verify-style answer checking, sympy-based (not the `math-verify` PyPI package — our seed
questions are plain numeric/algebraic text with no LaTeX, so a lightweight sympify+simplify equality
check is sufficient and avoids the heavier latex2sympy2/antlr4 dependency chain).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import sympy

from core.ledger import now, verify_id
from core.schemas import Attempt, Question, VerifyResult
from verify.base import Verifier

# Anchored to line start (not a bare substring search) so prose that merely *mentions* "final answer"
# after the real answer line -- e.g. a model second-guessing itself -- doesn't get picked up instead
# of the actual "Final Answer: ..." line the system prompt asks for.
_FINAL_ANSWER_RE = re.compile(r"^\s*final answer\s*:?\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_BOXED_START_RE = re.compile(r"\\boxed\{")
_LEADING_QUALIFIER_RE = re.compile(r"^(?:≈|~|about|approx(?:imately)?|roughly)\s*", re.IGNORECASE)

# Content hash of this file, used as the verifier's version so a code change to _check/_extract_*
# automatically invalidates any VerifyResult cached under the old (possibly buggy) logic.
_VERIFIER_VERSION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:8]


def _find_boxed_matches(content: str) -> list[str]:
    """Finds every \\boxed{...} body, respecting nested braces (e.g. \\boxed{\\frac{1}{2}}) -- a flat
    regex like \\boxed\\{([^}]*)\\} stops at the first '}' and would truncate that to "\\frac{1"."""
    matches = []
    for start_match in _BOXED_START_RE.finditer(content):
        i = start_match.end()
        depth = 1
        while i < len(content) and depth > 0:
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            matches.append(content[start_match.end() : i - 1])
    return matches


def _extract_answer_text(content: str) -> str:
    boxed_matches = _find_boxed_matches(content)
    if boxed_matches:
        return boxed_matches[-1]
    final_matches = _FINAL_ANSWER_RE.findall(content)
    if final_matches:
        return final_matches[-1].strip()
    for line in reversed(content.strip().splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _normalize_term(term: str) -> str:
    term = term.strip().strip(".").strip("$").strip()
    term = re.sub(r"\\[()\[\]]", "", term)  # strip LaTeX \( \) \[ \] delimiters, e.g. "\(x = -2\)"
    term = term.strip()
    # Strip a leading approximation qualifier ("about 5", "~5", "≈5") before it reaches sympify --
    # otherwise these fail to parse entirely and fall back to a literal string compare against gold,
    # false-negativing an answer that's numerically correct.
    term = _LEADING_QUALIFIER_RE.sub("", term)
    term = re.sub(r"^[xX]\s*=\s*", "", term)
    return term.strip()


def _parse_term(term: str) -> sympy.Expr | None:
    """Tries the term as-is first; if that fails, progressively strips trailing words (e.g. units --
    "20 km", "8 cups of flour") and retries, so a correct answer with a unit attached isn't scored as
    wrong just because "20 km" doesn't sympify to a bare number. Never invents a numeric value that
    wasn't in the original text -- it only ever narrows to a leading prefix of the actual term."""
    try:
        return sympy.sympify(term)
    except Exception:  # noqa: BLE001
        pass
    words = term.split()
    for i in range(len(words) - 1, 0, -1):
        try:
            return sympy.sympify(" ".join(words[:i]))
        except Exception:  # noqa: BLE001
            continue
    return None


def _numeric_equal(a: sympy.Expr, b: sympy.Expr) -> bool:
    try:
        return bool(sympy.simplify(a - b) == 0)
    except Exception:  # noqa: BLE001
        return False


class MathAnswerVerifier(Verifier):
    id = "math_answer_v1"
    version = _VERIFIER_VERSION

    def verify(self, attempt: Attempt, question: Question) -> VerifyResult:
        raw = _extract_answer_text(attempt.content)
        terms = [_normalize_term(t) for t in raw.split(",") if t.strip()]

        if question.verify_method == "numeric":
            gold_terms = [_normalize_term(str(question.gold))]
        elif question.verify_method == "numeric_set":
            gold_terms = [_normalize_term(t) for t in str(question.gold).split(",") if t.strip()]
        else:
            return VerifyResult(
                id=verify_id(attempt.id, self.id, self.version),
                attempt_id=attempt.id,
                passed=False,
                detail=f"unsupported verify_method: {question.verify_method!r}",
                verifier_id=self.id,
                ts=now(),
            )

        passed, detail = self._check(terms, gold_terms, question.verify_method)
        return VerifyResult(
            id=verify_id(attempt.id, self.id, self.version), attempt_id=attempt.id, passed=passed,
            detail=detail, verifier_id=self.id, ts=now(),
        )

    def _check(self, terms: list[str], gold_terms: list[str], method: str) -> tuple[bool, str]:
        detail_prefix = f"extracted={terms!r}, gold={gold_terms!r}, method={method}"

        if method == "numeric":
            if len(terms) != 1:
                return False, f"{detail_prefix}, match=False (expected exactly 1 term)"
            a, b = _parse_term(terms[0]), _parse_term(gold_terms[0])
            if a is None or b is None:
                match = terms[0].strip().lower() == gold_terms[0].strip().lower()
            else:
                match = _numeric_equal(a, b)
            return match, f"{detail_prefix}, match={match}"

        # numeric_set: order-independent, cardinality must match
        if len(terms) != len(gold_terms):
            return False, f"{detail_prefix}, match=False (cardinality {len(terms)} != {len(gold_terms)})"
        remaining = list(gold_terms)
        for t in terms:
            parsed_t = _parse_term(t)
            found_idx = None
            for i, g in enumerate(remaining):
                parsed_g = _parse_term(g)
                if parsed_t is not None and parsed_g is not None:
                    same = _numeric_equal(parsed_t, parsed_g)
                else:
                    same = t.strip().lower() == g.strip().lower()
                if same:
                    found_idx = i
                    break
            if found_idx is None:
                return False, f"{detail_prefix}, match=False (no gold match for term {t!r})"
            remaining.pop(found_idx)
        return True, f"{detail_prefix}, match=True"
