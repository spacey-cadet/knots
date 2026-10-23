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


def test_ranked_cards_show_the_note_not_the_ledger_or_the_templated_reason():
    it = Interest(topic="memecoins", depth="deep", sentiment="curious", mentions=1, concepts=["memecoins"], keywords=["memecoins"])
    card = render_card(_film(), it, {"receipts": [{"kind": "tags", "text": "x"}], "fit_score": 1.0})
    card["note"], card["spoken_note"] = "A night it all unravels.", "A night."
    res = {"extractor": "bedrock", "interests": [], "cards": [card]}
    shown = presenter.public_cards(res)["cards"][0]
    assert shown["note"] == "A night it all unravels." and not INTERNAL & set(shown)       # reasoning is in INTERNAL
    full = presenter.public_cards(res, explain=True)["cards"][0]
    assert "receipts" in full and "reasoning" in full


def test_note_cards_gives_ranked_cards_the_written_note_or_the_templated_one():
    it = Interest(topic="memecoins", depth="deep", sentiment="curious", mentions=1, concepts=["memecoins"], keywords=["memecoins"])
    mk = lambda cleared=True: [render_card(_film(), it, {"receipts": [{"kind": "you_said", "text": "I keep seeing memecoin threads"}],
                                                          "cleared_floor": cleared})]
    calls = []
    def writer(entries):
        calls.append(entries); return [("One night, one crash, and the thread you keep seeing.", "One night, one crash.")]
    cards = contract.note_cards(mk(), [_film()], [it], writer=writer)
    assert cards[0]["note"].startswith("One night") and cards[0]["written_by"] == "model" and len(calls) == 1
    assert calls[0][0]["said"] == "I keep seeing memecoin threads"                   # the person's own sentence goes to the writer
    fallback = contract.note_cards(mk(), [_film()], [it], writer=lambda es: [None])[0]
    assert "written_by" not in fallback and fallback["note"].startswith("You said “I keep seeing memecoin threads”")
    assert fallback["note"] != fallback["reasoning"] and fallback["spoken_note"]
    plain = contract.note_cards(mk(cleared=False), [_film()], [it], writer=lambda es: [None])[0]
    assert plain["note"].startswith("A banker watches a crash") and "You said" not in plain["note"]   # no certified fit: no claim
    assert contract.note_cards(mk(), [_film()], [it], writer=lambda es: [None])[0]["note"].startswith("You said")
    assert contract.note_cards([], [_film()], [it], writer=lambda es: 1 / 0) == []      # nothing to write: no call at all


def test_a_card_whose_item_is_unknown_keeps_its_templated_reason_as_the_note():
    it = Interest(topic="memecoins", depth="deep", sentiment="curious", mentions=1, concepts=["memecoins"], keywords=["memecoins"])
    card = render_card(_film(), it, {})
    out = contract.note_cards([card], [], [it], writer=lambda es: [])[0]
    assert out["note"] == card["reasoning"] and out["spoken_note"] == ""


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


def test_cleared_floor_needs_the_full_floor_not_just_a_pass(tmp_path, monkeypatch):
    import json
    from knots import fitmodel
    f = tmp_path / "w.json"
    f.write_text(json.dumps({"weights": fitmodel.DEFAULT_WEIGHTS, "floor": {"sim": 0.5, "score": 0.0, "embedder": "cohere"}}))
    monkeypatch.setattr(fitmodel, "WEIGHTS_FILE", f); fitmodel.reload()
    sc = lambda sims, scores, emb="cohere", q=True: fitmodel.Scored(items=[None] * len(sims), features=[{}] * len(sims), scores=scores,
                                                                  sims=sims, vecs=[[]] * len(sims), embedder=emb, query_used=q)
    assert fitmodel.cleared_floor(sc([0.9], [1.0]), 0) is True
    assert fitmodel.cleared_floor(sc([0.2], [1.0]), 0) is False                       # below the similarity floor
    other = sc([0.1], [1.0], emb="hashed")
    assert fitmodel.passes_floor(other, 0) is True and fitmodel.cleared_floor(other, 0) is False   # shown, but not certified
    assert fitmodel.cleared_floor(sc([0.9], [1.0], q=False), 0) is False
    monkeypatch.setenv("KNOTS_FLOOR", "off")
    assert fitmodel.cleared_floor(sc([0.9], [1.0]), 0) is False
    fitmodel.reload()


def test_cards_record_whether_they_cleared_the_floor_and_the_presenter_keeps_it_internal():
    assert "cleared_floor" not in presenter.PUBLIC_CARD_KEYS and "cleared_floor" not in presenter.PUBLIC_RANKED_KEYS
    it = Interest(topic="memecoins", depth="deep", sentiment="curious", mentions=1, concepts=["memecoins"], keywords=["memecoins"])
    card = render_card(_film(), it, {"cleared_floor": True, "receipts": [{"kind": "tags", "text": "x"}]})
    assert "cleared_floor" not in presenter.public_cards({"cards": [card]})["cards"][0]
    assert presenter.public_cards({"cards": [card]}, explain=True)["cards"][0]["cleared_floor"] is True


def test_spoken_brief_is_for_the_ear_and_keeps_the_hard_rules():
    s = notewriter._SYSTEM
    assert "SAYS aloud" in s and "never repeat the title" in s and "never name another item" in s   # new voice guidance
    assert "Use ONLY the facts" in s and "Never explain how you chose it" in s                      # hard rules untouched


def test_spoken_line_may_run_to_two_short_sentences_but_not_further():
    two = "That crash you keep circling is the whole film, told overnight. It is slow and talky, and better for it."
    assert len(two) <= notewriter.MAX_SPOKEN and notewriter.valid(two, notewriter.MAX_SPOKEN, _film(), None, spoken=True)
    assert not notewriter.valid(two + " " + two + " " + two, notewriter.MAX_SPOKEN, _film(), None, spoken=True)   # too long
    assert not notewriter.valid("It runs 94 minutes and is slow and talky overall.", notewriter.MAX_SPOKEN, _film(), None, spoken=True)  # digits stay out of speech
