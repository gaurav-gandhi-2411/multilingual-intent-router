from __future__ import annotations

import numpy as np
import pytest

from intent_router.cv import RunSpec, fold_split, load_cv_frame
from intent_router.data import load_splits
from intent_router.train import class_weights, parent_index


def test_cv_frame_excludes_test_ids() -> None:
    frame = load_cv_frame()
    test_ids = set(load_splits().query("split == 'test'")["id"])
    assert test_ids
    assert not test_ids & set(frame["id"])
    assert set(frame["split"]) <= {"train", "val"}


def test_fold_split_disjoint_and_covering() -> None:
    frame = load_cv_frame()
    seen: list[str] = []
    for fold in range(5):
        tr, ev = fold_split(frame, 0, fold)
        assert not set(tr["id"]) & set(ev["id"])
        assert len(tr) + len(ev) == len(frame)
        seen += list(ev["id"])
    assert sorted(seen) == sorted(frame["id"])


def test_fold_split_rejects_frame_with_test_rows() -> None:
    from intent_router.data import get_frame

    with pytest.raises(AssertionError):
        fold_split(get_frame(["train", "val", "test"]), 0, 0)


def test_run_id_format() -> None:
    s = RunSpec("FacebookAI/xlm-roberta-base", 3e-5, 1, 2, 4)
    assert s.run_id == "xlmr_lr3e-05_fs1_ms2_f4"
    assert RunSpec("intfloat/multilingual-e5-base", 2e-5, 0, 0, 0, "lld").run_id.endswith("_lld")


def test_parent_index_collapses_siblings_to_ten_parents() -> None:
    labels = [
        "a",
        "shipment_information.x",
        "shipment_information.y",
        "b",
        "shipment_information.z",
    ]
    assert parent_index(labels) == [0, 1, 1, 2, 1]


def test_class_weights_inverse_frequency_mean_one() -> None:
    w = class_weights(np.array([0, 0, 0, 1]), 2)
    assert w.mean() == pytest.approx(1.0)
    assert w[1] == pytest.approx(3 * w[0])
