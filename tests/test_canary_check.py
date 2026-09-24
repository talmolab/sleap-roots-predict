"""Offline tests for scripts/canary_check.py's selection and pass/fail logic."""

import importlib.util
from pathlib import Path

from card_builders import make_card


def _canary():
    spec = importlib.util.spec_from_file_location(
        "canary_check", Path(__file__).parents[1] / "scripts" / "canary_check.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_RICE3 = {"species": "rice", "mode": "cylinder", "age": 3}


def test_ambiguous_selection_is_a_fail_line_not_a_traceback():
    """Two matching cards produce a FAIL message instead of an uncaught raise."""
    canary = _canary()
    cards = [make_card("primary", "reg/a"), make_card("primary", "reg/b")]
    results = canary.select_all(cards, [_RICE3])
    failures = canary.check(results, [], None, 0, None)
    assert len(failures) == 1 and "selection raised" in failures[0]


def test_expect_is_checked_per_root_type():
    """A suffix matched by another root type does not satisfy the named one."""
    canary = _canary()
    cards = [make_card("crown", "reg/rice-younger-primary")]
    results = canary.select_all(cards, [_RICE3])
    failures = canary.check(
        results, [("primary", "rice-younger-primary")], None, 0, None
    )
    assert failures and "expected primary" in failures[0]


def test_expect_roots_requires_the_exact_set():
    """An extra or missing root type fails --expect-roots."""
    canary = _canary()
    cards = [make_card("primary", "reg/p"), make_card("crown", "reg/c")]
    results = canary.select_all(cards, [_RICE3])
    assert canary.check(results, [], ["primary"], 0, None)
    assert canary.check(results, [], ["primary", "crown"], 0, None) == []


def test_every_context_is_checked():
    """A context that resolves nothing fails even when another context passes."""
    canary = _canary()
    cards = [make_card("primary", "reg/rice-p")]
    results = canary.select_all(
        cards, [_RICE3, {"species": "rice", "mode": "cylinder", "age": 9}]
    )
    failures = canary.check(results, [("primary", "rice-p")], None, 0, None)
    assert len(failures) == 1 and "rice|cylinder|9" in failures[0]


def test_skip_count_mismatch_fails():
    """An unexpected skip-warning count is a failure."""
    canary = _canary()
    assert canary.check({}, [], None, 12, 13) == ["expected 13 skips, saw 12"]
