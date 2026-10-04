from __future__ import annotations

from intent_router.models import mask_entities, prepare_text


def test_prefix_only_for_e5() -> None:
    assert prepare_text("hi", "intfloat/multilingual-e5-base") == "query: hi"
    assert prepare_text("hi", "FacebookAI/xlm-roberta-base") == "hi"


def test_entity_masking() -> None:
    s = "where is LD-11208 and PO-77 also ABC-9 ok"
    assert mask_entities(s) == "where is [LOAD_ID] and [PO_ID] also [REF_ID] ok"
    assert prepare_text(s, "microsoft/mdeberta-v3-base", mask=True).startswith("where is [LOAD_ID]")
