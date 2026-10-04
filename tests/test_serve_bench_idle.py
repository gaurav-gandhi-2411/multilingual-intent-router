"""Model-free test of the latency idle gate's verdict logic."""

from __future__ import annotations

import pytest

from intent_router.serve_bench import idle_summary


def test_idle_summary_verdicts() -> None:
    quiet = [2.0] * 58 + [40.0, 10.0]  # one spike: 59/60 below 15, mean ~2.6
    s = idle_summary(quiet)
    assert s["idle"] and s["max_pct"] == 40.0 and s["n_samples"] == 60
    assert not idle_summary([30.0] * 60)["idle"]
    assert not idle_summary([2.0] * 50 + [60.0] * 10)["idle"]  # 83% below: not sustained
    with pytest.raises(ValueError):
        idle_summary([])
