"""The evidence ledger stays internal; the person sees the note. `explain=True` is the only way back to it."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knots import contract, notes, notewriter, presenter  # noqa: E402
from knots.catalogs import ContentItem  # noqa: E402
from knots.extraction import Interest  # noqa: E402
from knots.ranking import render_card  # noqa: E402


def _film():
    return ContentItem(title="Margin Call", content_type="movie", source_url="https://x/Margin-Call", topic="memecoins",
                       topic_overlap_score=0.8, matched_keywords=["memecoins"], rating=7.1, popularity=4200,
                       year="2011", abstract="A banker watches a crash unfold overnight. It is slow and talky.")


def _plan():
    it = Interest(topic="memecoins", depth="deep", sentiment="curious", mentions=1, concepts=["memecoins"], keywords=["memecoins"])
    extras = {"receipts": [{"kind": "you_said", "text": "I keep seeing memecoin threads"}, {"kind": "tags", "text": "memecoins"}],
              "fit_features": {"sim_session": 0.4}, "fit_score": 0.7, "sim": 0.4, "confidence": None}
    card = render_card(_film(), it, extras)
    plan = {"picks": {"watch": [card], "read": [], "explore": []}, "pairs": [], "themes": [], "ask": "More?", "spoken": "x"}
    plan["response"] = contract.build_response(plan, [_film()], [it],
                                               writer=lambda es: [("Start from the thread you keep seeing: this is the night it all unravels.", "A night it all unravels.")])
    return plan


INTERNAL = {"receipts", "fit_features", "fit_score", "sim", "evidence", "evidence_types", "relevance", "medium_rank", "reasoning"}


def test_default_payload_has_no_ledger_and_carries_the_written_note():
    out = presenter.public_plan(_plan())
    card = out["picks"]["watch"][0]
    assert not INTERNAL & set(card)
    assert card["note"].startswith("Start from the thread") and card["title"] == "Margin Call"
    assert all("why" not in p for p in out["response"]["picks"])
    assert out["response"]["picks"][0]["facts"]                 # the sourced facts still travel


def test_explain_returns_the_full_cards_and_the_input_is_never_mutated():
    plan = _plan()
    assert presenter.public_plan(plan, explain=True)["picks"]["watch"][0]["receipts"]
    presenter.public_plan(plan)
    assert plan["picks"]["watch"][0]["receipts"] and "why" in plan["response"]["picks"][0]


def test_ranked_cards_keep_their_one_line_reason_but_lose_the_ledger():
    it = Interest(topic="memecoins", depth="deep", sentiment="curious", mentions=1, concepts=["memecoins"], keywords=["memecoins"])
    res = {"extractor": "bedrock", "interests": [], "cards": [render_card(_film(), it, {"receipts": [{"kind": "tags", "text": "x"}], "fit_score": 1.0})]}
    card = presenter.public_cards(res)["cards"][0]
    assert "reasoning" in card and not (INTERNAL - {"reasoning"}) & set(card)
    assert "receipts" in presenter.public_cards(res, explain=True)["cards"][0]


def test_an_unknown_new_card_field_stays_internal_by_default():
    plan = _plan()
    plan["picks"]["watch"][0]["debug_blob"] = {"secret": 1}
    assert "debug_blob" not in presenter.public_plan(plan)["picks"]["watch"][0]


def test_note_writer_accepts_a_verbatim_quote_of_what_the_person_said_and_rejects_an_invented_one():
    said = "I keep seeing memecoin threads and I cannot stop scrolling"
    ok = "You said “I keep seeing memecoin threads”, so watch the night it all unravels."
    bad = "You said “crypto is a scam”, so watch the night it all unravels."
    assert notewriter.valid(ok, 160, _film(), None, said=said)
    assert not notewriter.valid(bad, 160, _film(), None, said=said)
    assert not notewriter.valid(ok, 160, _film(), None)           # with no quote available, a quote is invented


def test_templated_note_leads_with_the_person_when_only_their_sentence_is_known():
    n = notes.build_note(_film(), None, None, "I keep seeing memecoin threads ...")
    assert n.startswith("You said “I keep seeing memecoin threads”") and "...”" not in n
    assert notes.build_note(_film()).startswith("A banker watches")      # no link to the person: description first


def test_the_brief_asks_for_why_you_would_love_it_and_forbids_explaining_the_choice():
    s = notewriter._SYSTEM
    assert "make this ONE person want to press play" in s and "Never explain how you chose it" in s
