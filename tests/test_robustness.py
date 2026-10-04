from __future__ import annotations

import numpy as np
import pytest

from intent_router.robustness import (
    ABBREVIATIONS,
    IDSWAPS,
    abbreviate,
    apply_noise,
    noise_types,
    paired_drop_ci,
    perturb_chars,
    row_rng,
    swap_ids,
)


def test_swap_ids_po_to_ld_keeps_digits_and_other_text() -> None:
    assert swap_ids("PO-71045 lines are duplicated", "PO", "LD") == (
        "LD-71045 lines are duplicated",
        1,
    )


def test_swap_ids_no_id_untouched() -> None:
    text = "where is my shipment? PO 123 or po-123 or XPO"
    assert swap_ids(text, "PO", "LD") == (text, 0)
    assert swap_ids(text, "other", "REF") == (text, 0)


def test_swap_ids_multiple_ids_only_source_replaced() -> None:
    out, n = swap_ids("PO-1 and PO-22 on LD-333, see APT-5077", "PO", "REF")
    assert out == "REF-1 and REF-22 on LD-333, see APT-5077"
    assert n == 2


def test_swap_ids_other_excludes_po_ld_ref() -> None:
    out, n = swap_ids("PO-1 LD-2 REF-3 APT-5077 TR-9 DOC-12", "other", "REF")
    assert out == "PO-1 LD-2 REF-3 REF-5077 REF-9 REF-12"
    assert n == 3


def test_swap_ids_prefix_boundary() -> None:
    # XPO-12 is a 3-letter "other" prefix, not a PO id; ABCDE-1 has a 5-letter prefix (no match)
    assert swap_ids("XPO-12", "PO", "LD") == ("XPO-12", 0)
    assert swap_ids("ABCDE-1", "other", "REF") == ("ABCDE-1", 0)
    assert swap_ids("LD-5", "LD", "PO")[0] == "PO-5"


def test_idswap_table_is_the_preregistered_five() -> None:
    assert set(IDSWAPS) == {"PO->LD", "PO->REF", "LD->PO", "LD->REF", "other->REF"}


def test_noise_deterministic_and_seed_sensitive() -> None:
    t = "where is my shipment, the appointment window is wrong"
    a = apply_noise(t, "char_mixed_10", "syn-1", 42)
    assert a == apply_noise(t, "char_mixed_10", "syn-1", 42)
    assert a != t
    assert (
        apply_noise(t, "char_mixed_10", "syn-1", 7) != a
        or apply_noise(t, "char_mixed_10", "syn-2", 42) != a
    )


@pytest.mark.parametrize("rate", [0.05, 0.10])
def test_drop_and_insert_hit_the_requested_rate(rate: float) -> None:
    text = "the quick brown fox jumps over the lazy dog " * 5  # 175 alphabetic chars
    n_alpha = sum(c.isalpha() for c in text)
    want = int(rate * n_alpha + 0.5)
    dropped, k = perturb_chars(text, rate, "drop", row_rng(42, "x", "r"))
    assert k == want
    assert len(text) - len(dropped) == want
    inserted, k = perturb_chars(text, rate, "insert", row_rng(42, "x", "r"))
    assert k == want
    assert len(inserted) - len(text) == want


def test_swap_preserves_multiset_and_length() -> None:
    text = "abcdefghij klmnopqrst"
    out, _ = perturb_chars(text, 0.2, "swap", np.random.default_rng(0))
    assert sorted(out) == sorted(text)
    assert out != text


def test_perturb_chars_edge_cases() -> None:
    assert perturb_chars("12345 !!", 0.1, "drop", np.random.default_rng(0)) == ("12345 !!", 0)
    assert perturb_chars("", 0.1, "swap", np.random.default_rng(0)) == ("", 0)
    with pytest.raises(ValueError):
        perturb_chars("abc", 0.1, "bogus", np.random.default_rng(0))


def test_lowercase_noise() -> None:
    assert apply_noise("Where Is PO-1?", "lowercase", "r", 42) == "where is po-1?"


def test_abbreviate_word_boundaries_and_case() -> None:
    assert abbreviate("Please send YOUR hours") == "pls send ur hrs"
    assert abbreviate("What is the Appointment for tomorrow, thanks") == (
        "whats the appt for tmrw, thx"
    )
    # substrings of longer words must not be touched
    assert abbreviate("youth yourself shipments hourly") == "youth yourself shipments hourly"
    assert abbreviate("what   is it") == "whats it"


def test_abbreviation_dictionary_has_the_preregistered_entries() -> None:
    for k, v in {"please": "pls", "what is": "whats", "you": "u", "your": "ur", "thanks": "thx",
                 "hours": "hrs", "tomorrow": "tmrw", "shipment": "shpmt",
                 "appointment": "appt"}.items():  # fmt: skip
        assert ABBREVIATIONS[k] == v


def test_noise_types_names() -> None:
    names = noise_types([0.05, 0.10])
    assert names[:4] == ["char_swap_5", "char_drop_5", "char_insert_5", "char_mixed_5"]
    assert "char_mixed_10" in names and names[-2:] == ["lowercase", "abbrev"]
    with pytest.raises(ValueError):
        apply_noise("x", "nope", "r", 1)


def test_paired_drop_ci_identical_predictions_zero_drop() -> None:
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    out = paired_drop_ci(y, y, y, 3, n_resamples=200, seed=42)
    assert out["accuracy_drop"] == {"point": 0.0, "lo": 0.0, "hi": 0.0}
    assert out["macro_f1_drop"]["point"] == 0.0


def test_paired_drop_ci_detects_a_drop_and_is_seeded() -> None:
    y = np.tile(np.arange(3), 20)
    clean = y.copy()
    pert = y.copy()
    pert[:30] = (pert[:30] + 1) % 3
    a = paired_drop_ci(y, clean, pert, 3, n_resamples=500, seed=42)
    assert a["accuracy_drop"]["point"] == pytest.approx(0.5)
    assert a["accuracy_drop"]["lo"] > 0
    assert a == paired_drop_ci(y, clean, pert, 3, n_resamples=500, seed=42)
