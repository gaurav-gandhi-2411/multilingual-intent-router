from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from intent_router.models import predict, state_dict_sha256  # noqa: E402


def _word_id(w: str) -> int:
    return 2 + sum(ord(c) * (i + 1) for i, c in enumerate(w)) % 90


class _StubTokenizer:
    """Whitespace tokenizer with a tiny vocab; mimics the HF call signature used by predict."""

    def __call__(
        self,
        texts: list[str],
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> transformers.BatchEncoding:
        ids = [[_word_id(w) for w in t.split()][:max_length] or [2] for t in texts]
        width = max(len(i) for i in ids)
        return transformers.BatchEncoding(
            {
                "input_ids": torch.tensor([i + [1] * (width - len(i)) for i in ids]),
                "attention_mask": torch.tensor(
                    [[1] * len(i) + [0] * (width - len(i)) for i in ids]
                ),
            }
        )


def _tiny_model() -> transformers.XLMRobertaForSequenceClassification:
    cfg = transformers.XLMRobertaConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=70,
        num_labels=5,
        pad_token_id=1,
    )
    torch.manual_seed(0)
    return transformers.XLMRobertaForSequenceClassification(cfg)


def test_features_are_input_to_out_proj() -> None:
    model = _tiny_model()
    texts = ["alpha beta", "gamma", "delta epsilon zeta eta"]
    logits, feats = predict(model, _StubTokenizer(), texts, "tiny", max_len=16, batch_size=2)
    assert logits.shape == (3, 5) and feats.shape == (3, 32)
    assert logits.dtype == np.float32 and feats.dtype == np.float32

    # Independent route: run the head pieces by hand on the encoder's <s> state.
    model.eval()
    tok = _StubTokenizer()
    with torch.no_grad():
        for i, t in enumerate(texts):
            enc = tok([t], padding=True, truncation=True, max_length=16, return_tensors="pt")
            h = model.roberta(**enc).last_hidden_state[:, 0, :]
            expect = torch.tanh(model.classifier.dense(h))[0].numpy()
            np.testing.assert_allclose(feats[i], expect, atol=1e-5)
            from_feats = model.classifier.out_proj(torch.from_numpy(feats[i])).numpy()
            np.testing.assert_allclose(logits[i], from_feats, atol=1e-5)


def test_predict_restores_train_mode_and_is_repeatable() -> None:
    model = _tiny_model()
    model.train()
    a = predict(model, _StubTokenizer(), ["x y", "z"], "tiny", 16, 8)
    assert model.training
    b = predict(model, _StubTokenizer(), ["x y", "z"], "tiny", 16, 8)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_state_dict_sha256_sensitive_to_weights() -> None:
    m = _tiny_model()
    h1 = state_dict_sha256(m)
    assert h1 == state_dict_sha256(m)
    with torch.no_grad():
        m.classifier.out_proj.bias[0] += 1e-3
    assert state_dict_sha256(m) != h1
