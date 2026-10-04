from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from intent_router import llm_audit
from intent_router import phase4c_data as p4


def test_lang_for_row_deterministic_and_balanced() -> None:
    ids = [f"syn-{i:04d}" for i in range(1, 501)]
    a = [p4.lang_for_row(i) for i in ids]
    assert a == [p4.lang_for_row(i) for i in ids]
    assert set(a) == set(p4.A3_LANGS)
    # 500 draws over 4 languages: expected 125 each; a hash far from uniform would be a bug.
    assert all(80 <= n <= 170 for n in Counter(a).values())
    # the seed is part of the key
    assert a != [p4.lang_for_row(i, seed=43) for i in ids]
    # order independence
    assert p4.lang_for_row("syn-0007") == a[6]


def test_parse_batch_valid_and_fenced() -> None:
    assert p4.parse_batch('["a  b", "c"]', 2) == ["a b", "c"]
    assert p4.parse_batch('```json\n["x", "y"]\n```', 2) == ["x", "y"]
    # extras beyond the request are truncated
    assert p4.parse_batch('["a", "b", "c"]', 2) == ["a", "b"]


@pytest.mark.parametrize(
    "raw",
    ["not json", '{"a": 1}', '["a", 3]', '["a", "  "]', "[]", '["only one"]', '["a", "b"'],
)
def test_parse_batch_malformed(raw: str) -> None:
    with pytest.raises(p4.BatchParseError):
        p4.parse_batch(raw, 10)


def test_dedupe_exact_and_cosine_on_toy_embeddings() -> None:
    emb = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.99, 0.141], [1.0, 0.0], [0.6, 0.8]], dtype=np.float32
    )
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    texts = ["Hello there", "other", "hello  there!", "HELLO THERE", "mid"]
    # idx3 is an exact (case-insensitive) duplicate of idx0; idx2 has cosine ~0.99 to idx0
    kept, drops = p4.dedupe(texts, emb, 0.95)
    assert kept == [0, 1, 4]
    assert drops == {"exact": 1, "cosine": 1}
    # at a threshold of 1.0 nothing is a cosine duplicate
    kept2, drops2 = p4.dedupe(texts, emb, 1.0)
    assert kept2 == [0, 1, 2, 4] and drops2 == {"exact": 1, "cosine": 0}


def test_max_cos_and_drop_threshold_is_strict() -> None:
    syn = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    held = np.array([[1.0, 0.0]], dtype=np.float32)
    mx, drop = p4.max_cos_and_drop(syn, held, 1.0)
    assert mx.tolist() == [1.0, 0.0] and drop.tolist() == [False, False]  # > thr, not >=
    _, drop2 = p4.max_cos_and_drop(syn, held, 0.85)
    assert drop2.tolist() == [True, False]


def test_assert_no_test_ids() -> None:
    p4.assert_no_test_ids(["a", "b"], {"c"})
    with pytest.raises(AssertionError):
        p4.assert_no_test_ids(["a", "c"], {"c"})


def test_eval_vs_train_mt_system_inequality() -> None:
    p4.assert_distinct_mt("facebook/m2m100_418M", "facebook/nllb-200-distilled-600M")
    with pytest.raises(AssertionError):
        p4.assert_distinct_mt("x", "x")


def test_loopback_guard_reused_from_llm_audit() -> None:
    assert p4.validate_loopback_url is llm_audit.validate_loopback_url
    assert p4._http_json is llm_audit._http_json
    assert p4.validate_loopback_url("http://127.0.0.1:11434/") == "http://127.0.0.1:11434"
    with pytest.raises(ValueError):
        p4.validate_loopback_url("http://example.com:11434")


def test_plan_batches_quotas_and_seeds() -> None:
    scfg = {
        "n_target": 640,
        "batch_size": 20,
        "batch_seed_base": 42000,
        "lang_quota": {"en": 0.75, "es": 0.0625, "fr": 0.0625, "de": 0.0625, "zh": 0.0625},
        "kind_quota": {"logistics_novel": 0.75, "generic_oos": 0.25},
    }
    plan = p4.plan_batches(scfg)
    assert sum(b["n"] for b in plan) == 640
    assert all(1 <= b["n"] <= 20 for b in plan)
    by_lang = Counter()
    for b in plan:
        by_lang[b["lang"]] += b["n"]
    assert by_lang["en"] == 480 and by_lang["zh"] == 40
    assert [b["seed"] for b in plan] == list(range(42000, 42000 + len(plan)))
    assert plan == p4.plan_batches(scfg)


def test_build_prompt_lists_all_twelve_exclusions() -> None:
    msgs = p4.build_prompt("English", "generic_oos", 20)
    text = json.dumps(msgs)
    assert all(d in text for d in llm_audit.LABEL_DESCRIPTIONS.values())
    assert len(llm_audit.LABEL_DESCRIPTIONS) == 12


def test_topic_hint_deterministic_cyclic_and_kind_specific() -> None:
    a = p4.topic_hint("logistics_novel", 3)
    assert a == p4.topic_hint("logistics_novel", 3) and len(set(a)) == 4
    assert all(t in p4.NOVEL_TOPICS for t in a)
    assert all(t in p4.OOS_TOPICS for t in p4.topic_hint("generic_oos", 40))
    assert (
        "Spread the messages across"
        in p4.build_prompt("English", "generic_oos", 5, a)[1]["content"]
    )
