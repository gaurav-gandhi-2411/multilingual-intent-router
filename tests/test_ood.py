from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from intent_router import ood


# ------------------------------------------------------------------ metrics, hand-checked
def test_auroc_aupr_fpr_hand_checked() -> None:
    # known {0.9, 0.7, 0.5}, unknown {0.8, 0.2}; higher = known.
    known, unknown = np.array([0.9, 0.7, 0.5]), np.array([0.8, 0.2])
    # known > unknown in 4 of 6 pairs (0.7 and 0.5 lose to 0.8).
    assert ood.auroc_unknown_positive(known, unknown) == pytest.approx(4 / 6)
    # ranking by -score: U(0.2) K K U(0.8) K -> AP = 0.5 * 1/1 + 0.5 * 2/4 = 0.75.
    assert ood.aupr_unknown_positive(known, unknown) == pytest.approx(0.75)
    # TPR 0.95 of 3 knowns needs all 3 => t = 0.5; unknown >= 0.5 is {0.8} => FPR 1/2.
    assert ood.fpr_at_known_tpr(known, unknown, 0.95) == pytest.approx(0.5)


def test_auroc_perfect_inverted_and_ties() -> None:
    assert ood.auroc_unknown_positive(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert ood.auroc_unknown_positive(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0
    assert ood.auroc_unknown_positive(np.array([1.0, 1.0]), np.array([1.0])) == 0.5
    assert ood.aupr_unknown_positive(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0


def test_threshold_at_95_retention() -> None:
    s = np.arange(1.0, 21.0)  # n = 20: keep ceil(19.0) = 19 rows => t = 2 (rows 2..20)
    t = ood.threshold_at_retention(s, 0.95)
    assert t == 2.0 and np.mean(s >= t) == pytest.approx(0.95)
    s = np.arange(1.0, 101.0)  # n = 100: keep 95 rows => t = 6
    t = ood.threshold_at_retention(s, 0.95)
    assert t == 6.0 and np.mean(s >= t) == 0.95
    s = np.arange(1.0, 8.0)  # n = 7: ceil(6.65) = 7 rows => the minimum is the threshold
    assert ood.threshold_at_retention(s, 0.95) == 1.0
    # shuffled input gives the same threshold
    rng = np.random.default_rng(0)
    assert ood.threshold_at_retention(rng.permutation(np.arange(1.0, 101.0)), 0.95) == 6.0


def test_threshold_ties_keep_at_least_target() -> None:
    s = np.array([5.0] * 10)
    t = ood.threshold_at_retention(s, 0.95)
    assert t == 5.0 and np.mean(s >= t) == 1.0  # ties accepted together: retention above target
    s = np.array([1.0, 2.0, 2.0, 2.0, 3.0] * 4)  # n = 20, keep 19 => t = 1.0 would keep 20
    t = ood.threshold_at_retention(s, 0.95)
    assert np.mean(s >= t) >= 0.95


# ---------------------------------------------------------------------------- scores
def test_logit_scores_hand_checked() -> None:
    lg = np.array([[0.0, 0.0], [2.0, 0.0]])
    assert ood.score_msp(lg) == pytest.approx([0.5, 1 / (1 + np.exp(-2.0))], rel=1e-6)
    assert ood.score_max_logit(lg).tolist() == [0.0, 2.0]
    assert ood.score_neg_energy(lg) == pytest.approx([np.log(2.0), np.log(np.exp(2.0) + 1)])
    # T > 1 flattens: temp-scaled MSP of [2, 0] at T = 2 is sigmoid(1)
    assert ood.score_msp_temp(lg, 2.0)[1] == pytest.approx(1 / (1 + np.exp(-1.0)), rel=1e-6)


def test_knn_kth_neighbour_hand_example() -> None:
    bank = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])  # normalised internally
    query = np.array([[5.0, 0.0]])
    r = np.sqrt(0.5)
    assert ood.score_knn_cosine(query, bank, 1)[0] == pytest.approx(1.0)
    assert ood.score_knn_cosine(query, bank, 2)[0] == pytest.approx(r)  # sims: 1, 0.707, 0
    assert ood.score_knn_cosine(query, bank, 3)[0] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        ood.score_knn_cosine(query, bank, 4)


def _gaussians(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    d, centers = 20, np.zeros((3, 20))
    centers[0, 0], centers[1, 1], centers[2, 2] = 8.0, 8.0, 8.0
    x = np.concatenate([rng.normal(c, 1.0, size=(60, d)) for c in centers])
    y = np.repeat(np.arange(3), 60)
    x_known = np.concatenate([rng.normal(c, 1.0, size=(30, d)) for c in centers])
    unk_center = np.zeros(d)
    unk_center[3:8] = 6.0
    return x, y, x_known, rng.normal(unk_center, 1.0, size=(40, d))


def test_mahalanobis_separates_synthetic_gaussians() -> None:
    x, y, known, unknown = _gaussians()
    g = ood.fit_gaussian_lw(x, y, 3)
    ks, us = ood.score_mahalanobis(known, g), ood.score_mahalanobis(unknown, g)
    assert (ks <= 0).all() and (us <= 0).all()
    assert ood.auroc_unknown_positive(ks, us) > 0.99
    assert 0.0 <= g.shrinkage <= 1.0


def test_mahalanobis_distance_value_with_identity_like_cov() -> None:
    # unit-variance residuals: Mahalanobis ~ Euclidean distance to the nearest class mean
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 1, (400, 2)), rng.normal(10, 1, (400, 2))])
    y = np.repeat([0, 1], 400)
    g = ood.fit_gaussian_lw(x, y, 2)
    q = g.means[0] + np.array([[3.0, 0.0]])
    assert -ood.score_mahalanobis(q, g)[0] == pytest.approx(3.0, abs=0.35)


def test_gaussian_requires_every_class() -> None:
    with pytest.raises(ValueError):
        ood.fit_gaussian_lw(np.zeros((4, 2)), np.array([0, 0, 1, 1]), 3)


def test_fit_scorer_all_methods_and_no_frozen() -> None:
    x, y, known, _ = _gaussians()
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(len(y), 3)) + 3 * np.eye(3)[y]
    sc = ood.fit_scorer(logits, y, x, y, 3, train_frozen=x[:, :5], knn_ks=(1, 5))
    out = sc.score(logits[:10], x[:10], x[:10, :5])
    assert list(out) == ood.method_names((1, 5))
    assert sc.temperature > 0
    assert list(sc.score(logits[:10], x[:10], None)) == ood.method_names((1, 5), frozen=False)


# ------------------------------------------------------------- operating point / modes
def test_strict_vs_lenient_counting() -> None:
    labels = ["a", "other", "chitchat", "b"]  # local index: a=0 other=1 chitchat=2 b=3
    thr = 0.5
    known_scores = np.array([0.9, 0.8, 0.4, 0.7])
    known_gold = np.array([0, 3, 0, 1])
    known_pred = np.array([0, 1, 0, 1])  # row 1: in-domain known sent to 'other'; row 3 gold other
    unk_scores = np.array([0.1, 0.9, 0.9, 0.6, 0.2])  # rejected: rows 0 and 4
    unk_pred = np.array([0, 1, 2, 3, 3])  # accepted rows: 1 -> other, 2 -> chitchat, 3 -> b
    r = ood.operating_point(thr, known_scores, known_pred, known_gold, unk_scores, unk_pred, labels)
    assert r["retention_known"] == 0.75  # 0.9, 0.8, 0.7 kept
    assert r["strict_rejection_recall"] == pytest.approx(2 / 5)
    assert r["lenient_rejection_recall"] == pytest.approx(4 / 5)  # + rows 1 and 2
    assert r["unknown_not_rejected_by_pred_label"] == {"b": 1, "chitchat": 1, "other": 1}
    assert r["known_accepted_routed_to_safe"] == 2  # accepted rows 1 and 3 predicted 'other'
    assert r["known_accepted_routed_to_safe_gold_not_safe"] == 1  # row 1 (gold b)
    assert r["known_accepted_routed_to_safe_frac_of_known"] == 0.5
    assert r["n_unknown_not_rejected"] == 3
    assert r["known_accepted_accuracy"] == pytest.approx(2 / 3)


def _toy_table() -> pd.DataFrame:
    labels = ["a", "other", "b"]
    rng = np.random.default_rng(3)
    rows = []
    for st, n, unk in (("cal", 40, False), ("eval", 40, False), ("eval", 30, True)):
        gold = rng.choice(["a", "b"], n) if not unk else np.array(["z"] * n)
        pred = [g if not unk and rng.random() > 0.1 else rng.choice(labels) for g in gold]
        base = 2.0 if not unk else 0.0
        t = pd.DataFrame(
            {
                "set": st,
                "is_unknown": unk,
                "gold": gold,
                "pred": pred,
                "msp": rng.normal(base, 1.0, n),
            }
        )
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def test_run_metrics_recomputes_from_table_only() -> None:
    t = _toy_table()
    m = ood.run_metrics(t, ["msp"], ["a", "other", "b"], 0.95)
    r = m["methods"]["msp"]
    cal = t[t["set"] == "cal"]["msp"].to_numpy()
    thr = ood.threshold_at_retention(cal, 0.95)
    ev = t[t["set"] == "eval"]
    kn, un = ev[~ev["is_unknown"]], ev[ev["is_unknown"]]
    assert r["threshold"] == thr
    assert r["retention_known"] == pytest.approx(np.mean(kn["msp"] >= thr))
    assert r["strict_rejection_recall"] == pytest.approx(np.mean(un["msp"] < thr))
    assert r["lenient_rejection_recall"] >= r["strict_rejection_recall"]
    assert r["auroc"] == pytest.approx(
        np.mean([k > u for k in kn["msp"] for u in un["msp"]]), abs=1e-12
    )
    assert m["n_eval_known"] == 40 and m["n_eval_unknown"] == 30 and m["n_cal"] == 40
    assert 0.0 <= r["aurc_open_set"] <= 1.0


def test_run_metrics_rejects_unknowns_in_calibration() -> None:
    t = _toy_table()
    t.loc[t["set"] == "cal", "is_unknown"] = True
    with pytest.raises(AssertionError):
        ood.run_metrics(t, ["msp"], ["a", "other", "b"])


def test_mean_std_sample_std() -> None:
    r = ood.mean_std([1.0, 2.0, 3.0])
    assert r["mean"] == 2.0 and r["std"] == pytest.approx(1.0) and r["n"] == 3
    assert ood.mean_std([4.0])["std"] is None


# ------------------------------------------------------------- holdout data builder
def _frame() -> pd.DataFrame:
    labels = ["a", "b", "c", "d", "other", "chitchat"]
    rows = []
    n = 0
    for lab in labels:
        for split, cnt in (("train", 7), ("val", 3), ("test", 3)):
            for _ in range(cnt):
                rows.append({"id": f"r{n:04d}", "text": f"t{n}", "label": lab, "split": split})
                n += 1
    return pd.DataFrame(rows)


def test_holdout_builder_never_leaks_heldout_rows() -> None:
    df = _frame()
    s = ood.build_holdout_sets(df, ["a", "c"])
    assert s.labels == ["b", "chitchat", "d", "other"]  # sorted remaining classes
    for part in (s.train, s.cal, s.eval_known):
        assert not part["label"].isin(["a", "c"]).any()
    assert set(s.train["split"]) == {"train"} and set(s.cal["split"]) == {"val"}
    assert set(s.eval_known["split"]) == {"test"}
    # unknown side: ALL rows of the held-out classes from every split
    assert len(s.eval_unknown) == 2 * 13 and set(s.eval_unknown["split"]) == {
        "train",
        "val",
        "test",
    }
    assert set(s.eval_unknown["label"]) == {"a", "c"}
    ids = [set(x["id"]) for x in (s.train, s.cal, s.eval_known, s.eval_unknown)]
    assert all(not (ids[i] & ids[j]) for i in range(4) for j in range(i + 1, 4))
    assert s.audit["n_train"] == 4 * 7 and s.audit["n_cal"] == 4 * 3 == s.audit["n_eval_known"]
    assert all(s.audit["assertions"].values())
    # held-out rows are exactly those of the held-out classes (nothing lost, nothing extra)
    assert len(s.train) + len(s.cal) + len(s.eval_known) + len(s.eval_unknown) == len(df)


def test_holdout_builder_smoke_drops_test_rows() -> None:
    s = ood.build_holdout_sets(_frame(), ["a"], smoke=True)
    for part in (s.train, s.cal, s.eval_known, s.eval_unknown):
        assert "test" not in set(part["split"])
    assert s.eval_known["id"].tolist() == s.cal["id"].tolist()  # known side = calibration rows
    assert s.audit["smoke"] is True


def test_holdout_builder_input_validation() -> None:
    with pytest.raises(ValueError):
        ood.build_holdout_sets(_frame(), ["nope"])
    with pytest.raises(ValueError):
        ood.build_holdout_sets(_frame(), [])
    with pytest.raises(ValueError):
        ood.build_holdout_sets(_frame(), ["a", "a"])


def test_holdout_builder_detects_a_class_missing_from_train() -> None:
    df = _frame()
    df = df[~((df["label"] == "b") & (df["split"] == "train"))]  # b has no train rows
    with pytest.raises(AssertionError):
        ood.build_holdout_sets(df, ["a"])
