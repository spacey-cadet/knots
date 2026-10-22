"""Beyond-accuracy metrics (knots/beyond.py, eval/eval_beyond.py) and online metrics with retrain triggers (knots/online.py)."""
import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_script(name):
    sys.path.insert(0, str(ROOT / "eval"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "eval" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


from knots import beyond as B  # noqa: E402
from knots import online as O  # noqa: E402
from knots import retrain as R  # noqa: E402


def card(url, type_="movie", votes=1000, overlap=0.0, sim=0.0, topic=None, citations=None):
    return {"url": url, "type": type_, "votes": votes, "citations": citations, "matched_topic": topic or url,
            "fit_features": {"overlap": overlap, "sim_session": sim}}


def film_pool(n=20):
    """n films with votes 1..n * 1000 and overlap/sim rising together, so percentiles are easy to reason about."""
    return [card(f"f{i}", votes=(i + 1) * 1000, overlap=float(i), sim=float(i)) for i in range(n)]


# ------------------------------------------------------------------------------------------ beyond.py
def test_novelty_is_one_minus_popularity_percentile_within_the_source_family():
    pool = B.Pool(film_pool())
    assert pool.novelty(card("x", votes=1)) > 0.95                      # rarer than every film in the pool
    assert pool.novelty(card("y", votes=10**7)) < 0.05                  # more famous than all of them
    # a video's views are not compared with film votes: no video reference, so unknown rather than guessed
    assert pool.novelty(card("v", "video", votes=5)) is None
    assert pool.novelty(card("p", "paper", votes=None, citations=None)) is None     # no citation count: unknown
    assert B.Pool(film_pool(3)).novelty(card("z", votes=1)) is None                 # pool too small to rank against


def test_unexpectedness_is_low_for_the_obvious_pick_and_high_for_the_far_one():
    pool = B.Pool(film_pool())
    assert pool.unexpectedness(card("obvious", overlap=19, sim=19)) < 0.05
    assert pool.unexpectedness(card("far", overlap=-5, sim=-5)) > 0.95
    assert pool.unexpectedness({"url": "bare", "type": "movie"}) is None            # nothing to compare: unknown


def test_ild_format_spread_and_topic_spread():
    assert B.ild([[1, 0], [1, 0]]) == 0.0 and B.ild([[1, 0], [0, 1]]) == 1.0
    assert B.ild([[1, 0]]) is None and B.ild([]) is None
    assert B.format_spread([card("a"), card("b")]) == 0.0                           # same format twice
    assert B.format_spread([card("a"), card("b", "video"), card("c", "paper")]) == pytest.approx(1.0)
    assert B.format_spread([card("a")]) is None
    assert B.topic_spread([card("a", topic="t"), card("b", topic="t")]) == 0.5
    assert B.interest_coverage([card("a", topic="memes")], ["memes", "coins"]) == 0.5
    assert B.interest_coverage([card("a")], []) is None


def test_serendipity_needs_a_good_pick_that_is_unexpected_and_creativity_is_gone():
    pool = B.Pool(film_pool())
    rare_far = card("rf", votes=1, overlap=-5, sim=-5)          # rare AND far from the person's words
    famous_far = card("ff", votes=10**7, overlap=-5, sim=-5)    # far, but everyone has seen it
    rare_near = card("rn", votes=1, overlap=19, sim=19)         # rare, but exactly what they said
    m = B.list_metrics([rare_far, famous_far, rare_near], [1, 1, 1], pool)
    assert m["serendipity"] == pytest.approx((1 + 1 + 0) / 3, abs=0.1)          # only the two far picks are unexpected
    assert "creativity" not in m and "creative_hit" not in m                     # distance is not rewarded
    assert m["serendipitous_share"] == pytest.approx(2 / 3, abs=0.01)
    bad = B.list_metrics([rare_far, famous_far, rare_near], [0, 0, 0], pool)     # nothing good: nothing serendipitous
    assert bad["serendipity"] == 0
    assert math.isnan(bad["serendipitous_share"])                                # no good picks to take a share of
    assert "serendipity" not in B.list_metrics([rare_far], None, pool)           # label-free call omits them


def test_unknown_values_are_left_out_not_zeroed():
    pool = B.Pool(film_pool())
    unknown_pop = card("u", "paper", votes=None, citations=None, overlap=-5, sim=-5)
    m = B.list_metrics([unknown_pop], [1], pool)
    assert math.isnan(m["novelty"])
    assert m["serendipity"] > 0.9                    # its unexpectedness IS known, so serendipity still counts it


def test_coverage_and_gini():
    lists = [[card("a"), card("b")], [card("a"), card("c")], [card("a", "video"), card("d", "paper")]]
    c = B.coverage(lists)
    assert c["slots"] == 6 and c["distinct_items"] == 4 and c["unique_share"] == pytest.approx(4 / 6, abs=1e-3)
    assert c["top_item_share"] == pytest.approx(0.5) and c["formats_seen"] == 3
    assert B.gini([1, 1, 1, 1]) == 0.0 and B.gini([1, 1, 1, 9]) > 0.4 and B.gini([5]) == 0.0
    assert math.isnan(B.coverage([])["unique_share"])


def test_pool_round_trips_through_json_so_a_reference_can_be_frozen():
    import json
    pool = B.Pool(film_pool())
    again = B.Pool.from_dict(json.loads(json.dumps(pool.to_dict())))
    probe = card("p", votes=7000, overlap=7, sim=3)
    assert again.novelty(probe) == pool.novelty(probe) and again.unexpectedness(probe) == pool.unexpectedness(probe)


# ------------------------------------------------------------------------------------------ eval_beyond.py
def _entry(sid, ext, cards, topics=("t",)):
    return {"ok": True, "session_id": sid, "extractor": ext, "interests": [{"topic": t, "in_current": True} for t in topics],
            "plan": {"picks": {"watch": cards, "read": [], "explore": []}}}


def test_eval_beyond_reports_labelled_and_label_free_blocks_and_never_gives_coverage_an_interval():
    EB = _load_script("eval_beyond")
    cards = {s: [card(f"{s}{i}", votes=(i + 1) * 500, overlap=float(i), sim=float(i), topic=f"t{i % 2}") | {"kind": "watch"}
                 for i in range(3)] for s in ("s1", "s2", "s3", "s4")}
    results = [_entry(s, "bedrock", cs, topics=("t0", "t1")) for s, cs in cards.items()]
    labels = [{"session_id": s, "extractor": "bedrock", "url": c["url"], "good": "y" if i == 0 else "n"}
              for s, cs in cards.items() if s != "s3" for i, c in enumerate(cs)]          # s3 unlabelled
    splits = {"s1": "train", "s2": "train", "s3": "train", "s4": "holdout"}
    built = EB.ER.build_lists(results, labels, splits, None)
    pool = EB.B.Pool([c for cs in cards.values() for c in cs], min_pool=3)
    labelled = EB.evaluate(built["train"], built["lab"], pool, None)
    assert "serendipity" in labelled["as read aloud"] and "creativity" not in labelled["as read aloud"]
    assert labelled["as read aloud"]["p@3"]["mean"] == pytest.approx(1 / 3, abs=1e-3)
    assert labelled["as read aloud"]["interest_coverage"]["mean"] == 1.0                  # both interests answered
    shown = EB.shown_lists(results, splits, None)
    assert [e["session_id"] for e, _ in shown] == ["s1", "s2", "s3"]                      # holdout out, unlabelled in
    free = EB.evaluate(shown, None, pool, None)
    assert "serendipity" not in free["as read aloud"] and "novelty" in free["as read aloud"]
    cov = free["as read aloud"]["coverage"]
    assert cov["slots"] == 9 and cov["distinct_items"] == 9 and "ci90" not in str(cov)    # a plain number, no interval
    assert EB.RANDOM in free and "coverage" in free[EB.RANDOM] and EB.ALL_SHOWN in free
    assert EB.evaluate([], {}, pool, None) is None


def test_summarize_rankings_accepts_metric_dicts_without_a_map_column():
    sys.path.insert(0, str(ROOT / "eval"))
    import metrics as M
    s = M.summarize_rankings([{"session": "a", "metrics": {"x": 0.2}}, {"session": "b", "metrics": {"x": 0.4}}], n_boot=50)
    assert s["x"]["mean"] == pytest.approx(0.3) and s["lists_without_a_good_pick"] == 0


# ------------------------------------------------------------------------------------------ online.py
def fb(user, sid, url, good, ts=1000.0, source="user"):
    return {"type": "feedback", "user": user, "session_id": sid, "url": url, "good": good, "source": source, "ts": ts}


def imp(user, sid, cards, ts=1000.0, status="ok", exploration=()):
    return O.impression_event(user, sid, cards, status, "v1", set(exploration), ts)


def population(n_sessions, good_per_ten, prefix, votes=3000, overlap=2.0, start=1000.0, status="ok", users=25):
    """n_sessions sessions of three picks each, every pick answered; `good_per_ten` of each ten answers are good."""
    ev, k = [], 0
    for i in range(n_sessions):
        cards = [card(f"{prefix}{i}-{j}", votes=votes + 37 * ((i * 3 + j) % 50), overlap=overlap + ((i + j) % 5) * 0.3,
                      sim=overlap) | {"fit_score": 1.0 + ((i * 3 + j) % 7) * 0.1} for j in range(3)]
        user, sid, ts = f"u{i % users}", f"{prefix}s{i}", start + i
        ev.append(imp(user, sid, cards, ts, status))
        for c in cards:
            ev.append(fb(user, sid, c["url"], "y" if (k % 10) < good_per_ten else "n", ts + 0.5))
            k += 1
    return ev


def test_event_log_validates_appends_skips_damaged_lines_and_forgets_a_person(tmp_path):
    log = O.EventLog(tmp_path / "ev" / "events.jsonl")
    log.append(imp("u1", "s1", [card("a")]))
    log.append(fb("u1", "s1", "a", "y"))
    log.append(fb("u2", "s2", "b", "n"))
    with open(log.path, "a") as f:
        f.write("{not json\n")
    rows = log.read()
    assert len(rows) == 3 and log.skipped == 1 and all("ts" in r for r in rows)
    for bad in ({"type": "nope", "session_id": "s"}, {"type": "feedback", "session_id": "s", "user": "u", "url": "a", "good": "maybe"},
                {"type": "feedback", "session_id": "s", "good": "y", "url": "a"},            # no verified user
                {"type": "impression", "session_id": "s", "picks": [], "status": "weird"}, {"type": "engage", "session_id": "s"},
                {"type": "feedback", "good": "y", "user": "u", "url": "a"}):                  # no session
        with pytest.raises(ValueError):
            log.append(bad)
    assert log.forget("u1") == 2
    left = log.read()
    assert [r.get("user") for r in left] == ["u2"]
    assert log.read(since=10**12) == []


def test_only_explicit_listener_feedback_ever_becomes_a_training_label():
    events = [fb("u1", "s", "a", "y"), fb("u1", "s", "b", "n", source="expert"),
              {"type": "engage", "session_id": "s", "url": "a", "user": "u1", "action": "more"},
              {"type": "reformulate", "session_id": "s", "user": "u1"}, imp("u1", "s", [card("a")])]
    labels = O.labels_from_events(events)
    assert [(x["url"], x["good"], x["source"]) for x in labels] == [("a", "y", "user")]
    assert R.eligible(labels, R.Policy())[0]["user"] == "u1"            # and they are exactly what retrain accepts


def test_wilson_and_psi():
    lo, hi = O.wilson(40, 100)
    assert lo < 0.4 < hi and O.wilson(0, 0) is None and 0 <= O.wilson(0, 10)[0] <= O.wilson(0, 10)[1] < 0.3
    ref = [i / 100 for i in range(200)]
    assert O.psi(ref, ref) == pytest.approx(0, abs=0.02)
    assert O.psi(ref, [x + 5 for x in ref]) > 0.25                       # everything moved above the old range
    assert O.psi([1, 1, 1], [1, 2]) is None and O.psi(ref, []) is None


def test_window_metrics_cover_feedback_position_exploration_repeats_and_status():
    a, b, c, d = (card(u) for u in "abcd")
    events = [imp("u1", "s1", [a, b, c, d], 1.0, exploration={"d"}), imp("u1", "s2", [a, d], 2.0, status="relaxed"),
              imp("u2", "s3", [], 3.0, status="empty"),
              fb("u1", "s1", "a", "y"), fb("u1", "s1", "b", "n", ts=1.0), fb("u1", "s1", "d", "y"), fb("u2", "s1", "a", "y"),
              fb("u1", "s1", "b", "y", ts=1.5),                                      # a later answer replaces the earlier one
              {"type": "engage", "session_id": "s1", "url": "a", "action": "more"},
              {"type": "reformulate", "session_id": "s2", "user": "u1"}]
    m = O.window_metrics(events)
    assert m["lists"] == 3 and m["sessions"] == 3 and m["empty_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["relaxed_rate"] == pytest.approx(1 / 3, abs=1e-3)
    f = m["feedback"]
    assert f["n"] == 4 and f["people"] == 2 and f["positive_rate"] == 1.0                # b became y; the first n is replaced
    assert f["positive_rate_top3"] == 1.0 and f["n_top3"] == 3                           # d sat at position 4
    assert m["exploration"]["n_explore"] == 1 and m["exploration"]["positive_rate_explore"] == 1.0
    assert m["exploration"]["n_ranked"] == 3
    assert m["engagement_rate"] == pytest.approx(1 / 3, abs=1e-3) and m["reformulation_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["repeat_rate"] == 1.0                                       # u1's second list repeated a and d
    assert m["beyond"]["novelty"] is None and m["beyond"]["unexpectedness"] is None     # no reference pool: unknown
    assert m["beyond"]["coverage"]["distinct_items"] == 4
    assert O.window_metrics([])["feedback"]["positive_rate"] is None


def test_percentile_metrics_need_a_frozen_pool_because_against_their_own_window_they_are_always_one_half():
    ev = population(30, 7, "w")
    own = O.window_metrics(ev, B.Pool([p for e in ev if e["type"] == "impression" for p in e["picks"]]))
    assert own["beyond"]["novelty"] == pytest.approx(0.5, abs=0.05)                     # tautology
    assert O.window_metrics(ev)["beyond"]["novelty"] is None                            # so without a reference: None


def test_quality_drop_alert_fires_only_on_a_real_drop_and_is_retrain_relevant():
    base_ev = population(40, 8, "b")                                                    # 80% good
    baseline = O.make_baseline(base_ev)
    assert baseline["metrics"]["feedback"]["positive_rate"] == pytest.approx(0.8) and len(baseline["scores"]) > 50
    same = population(30, 8, "n", start=5000.0)
    assert not [a for a in O.alerts(same, baseline) if a["kind"] == "quality_drop"]
    worse = population(30, 4, "w", start=5000.0)                                        # 40% good over 90 answers
    found = [a for a in O.alerts(worse, baseline) if a["kind"] == "quality_drop"]
    assert found and found[0]["retrain_relevant"] and "40%" in found[0]["detail"] and "80%" in found[0]["detail"]
    few = population(5, 0, "f", start=5000.0)                                           # 15 answers, all bad: too few to alarm
    assert not [a for a in O.alerts(few, baseline) if a["kind"] == "quality_drop"]
    assert O.alerts(worse, None) == []                                                  # nothing to compare with


def test_score_drift_retrieval_and_filter_bubble_alerts():
    baseline = O.make_baseline(population(40, 8, "b"))
    shifted = population(40, 8, "d", start=5000.0)
    for e in shifted:
        for p in e.get("picks", []):
            p["fit_score"] = (p["fit_score"] or 0) + 3.0
    drift = [a for a in O.alerts(shifted, baseline) if a["kind"] == "score_drift"]
    assert drift and drift[0]["retrain_relevant"]
    empty = population(40, 8, "e", start=5000.0, status="empty")
    ret = [a for a in O.alerts(empty, baseline) if a["kind"] == "retrieval"]
    assert ret and not ret[0]["retrain_relevant"]                                       # a catalog problem, not a ranker one
    famous = population(40, 8, "m", votes=10**7, overlap=50.0, start=5000.0)             # only famous, obvious picks now
    bub = [a for a in O.alerts(famous, baseline) if a["kind"] == "filter_bubble"]
    assert bub and not bub[0]["retrain_relevant"] and "novelty" in bub[0]["detail"]


def test_an_alert_lowers_only_the_new_label_requirement_never_the_totals_or_the_number_of_people():
    pol = R.Policy(min_labels=200, min_users=20, per_user_cap=25, min_new=100, alert_min_new=50)
    labels = [{"user": f"u{i % 25}", "session_id": f"s{i}", "url": f"x{i}", "good": "y", "source": "user", "ts": i}
              for i in range(200)]
    state = {"trained_on": 120}                                                          # 80 new: short of 100, over 50
    assert not R.should_retrain(labels, state, pol)[0]
    ok, why = R.should_retrain(labels, state, pol, alerts=[{"kind": "quality_drop", "retrain_relevant": True}])
    assert ok and "online alert: quality_drop" in why
    state_far = {"trained_on": 180}                                                      # 20 new: under 50 either way
    assert not R.should_retrain(labels, state_far, pol, alerts=[{"kind": "quality_drop"}])[0]
    few_labels = labels[:60]
    assert not R.should_retrain(few_labels, {"trained_on": 0}, pol, alerts=[{"kind": "quality_drop"}])[0]
    one_person = [dict(x, user="solo") for x in labels]                                  # one enthusiast: the cap leaves 25
    assert not R.should_retrain(one_person, {"trained_on": 0}, pol, alerts=[{"kind": "quality_drop"}])[0]


def test_retrain_decision_combines_volume_and_online_alerts_and_ignores_unfixable_ones():
    baseline = O.make_baseline(population(40, 8, "b"))
    worse = population(30, 4, "w", start=5000.0)
    labels = O.labels_from_events(worse)
    pol = R.Policy(min_labels=60, min_users=10, per_user_cap=25, min_new=100, alert_min_new=20)
    d = O.retrain_decision(labels, worse, {"trained_on": 0, "baseline": baseline}, pol, now=5100.0)
    assert d["retrain"] and any(a["kind"] == "quality_drop" for a in d["alerts"]) and "online alert" in d["reason"]
    assert d["metrics"]["feedback"]["positive_rate"] == pytest.approx(0.4)
    quiet = O.retrain_decision(labels, worse, {"trained_on": 0, "baseline": None}, pol, now=5100.0)
    assert not quiet["retrain"] and quiet["alerts"] == []                                # no baseline, not enough new labels
    empty = population(40, 8, "e", start=5000.0, status="empty")
    r = O.retrain_decision(O.labels_from_events(empty)[:50], empty, {"trained_on": 0, "baseline": baseline}, pol, now=5100.0)
    assert any(a["kind"] == "retrieval" for a in r["alerts"]) and "online alert" not in r["reason"]
    assert O.retrain_decision(labels, [], {"trained_on": 0, "baseline": baseline}, pol, now=5100.0)["alerts"] == []
    old = [dict(e, ts=e["ts"] - 90 * O.DAY) for e in worse]                              # outside the 14-day window
    assert O.retrain_decision(labels, old, {"trained_on": 0, "baseline": baseline}, pol, now=5100.0)["alerts"] == []


def test_registry_keeps_a_baseline_per_champion_and_rollback_restores_it(tmp_path):
    reg = R.Registry(tmp_path)
    v1 = reg.register({"w": 1}, {"p@3": 0.5}, 100)
    reg.promote(v1, 100, baseline={"metrics": {"tag": "one"}})
    v2 = reg.register({"w": 2}, {"p@3": 0.6}, 200)
    reg.promote(v2, 200, baseline={"metrics": {"tag": "two"}})
    assert reg.state()["baseline"]["metrics"]["tag"] == "two"
    assert reg.rollback() == v1 and reg.state()["baseline"]["metrics"]["tag"] == "one"
    assert reg.rollback() is None
    old = tmp_path / "old"
    old.mkdir()
    (old / "state.json").write_text('{"champion": "v2", "history": ["v1"], "trained_on": 5}')     # written before baselines
    r2 = R.Registry(old)
    assert r2.rollback() == "v1" and r2.state()["baseline"] is None


# ------------------------------------------------------------------------------------------ model card
def test_model_card_has_a_beyond_accuracy_section_with_numbers_only_from_the_file(tmp_path, monkeypatch):
    import json
    MC = _load_script("make_model_card")
    monkeypatch.setattr(MC, "ROOT", tmp_path)
    monkeypatch.setattr(MC, "RES", tmp_path)
    MC.main()
    assert "Not yet measured: run `python eval/eval_beyond.py`" in (tmp_path / "MODEL_CARD.md").read_text()
    cell = lambda m: {"mean": m, "ci90": [round(m - 0.05, 3), round(m + 0.05, 3)]}  # noqa: E731
    nan = {"mean": float("nan"), "ci90": [float("nan")] * 2}
    row = {"p@3": cell(0.61), "ild": nan, "novelty": cell(0.47), "unexpectedness": cell(0.52),
           "serendipity": cell(0.33), "grounding": cell(0.29), "overreach": cell(0.08), "theme_coverage": nan,
           "coverage": {"slots": 90, "distinct_items": 81, "unique_share": 0.9, "gini": 0.07, "top_item_share": 0.02}}
    (tmp_path / "metrics_beyond.json").write_text(json.dumps({
        "k": 3, "ild_embedder": "hashed",
        "bedrock": {"labelled_lists": 12, "labelled": {"as read aloud": row, "random (chance)": row},
                    "all_shown_lists": 30, "all_shown": {"as read aloud": row}}}))
    MC.main()
    sect = (tmp_path / "MODEL_CARD.md").read_text().split("## Beyond accuracy")[1].split("## Watch-and-learn")[0]
    assert "0.47 [0.42, 0.52]" in sect and "0.33 [0.28, 0.38]" in sect         # copied from the file
    assert "| as read aloud |" in sect and "| random (chance) |" in sect and "| n/a |" in sect   # unknown ILD is n/a, not 0
    assert "hashed embedder" in sect and "12 complete lists" in sect           # the lexical-embedder warning is shown
    assert "81 distinct items in 90 slots" in sect and "not the fraction of a catalog" in sect


# ---- Grounding -----------------------------------------------------------------------------------------------

def test_grounding_pays_for_evidence_not_distance():
    from knots import grounding as G
    assert G.evidence_types("movie", ["memecoin"], 0.4, True) == ["keyword", "theme_quote", "overview_sim"]
    assert G.evidence_types("video", ["x"], 0.1, True) == []                     # a video's "match" is not evidence
    assert G.evidence_types("paper", None, 0.5, True) == ["overview_sim"]
    assert G.evidence_types("movie", None, None, False) == []                    # missing facts never fire a type
    assert G.pick_grounding(1, 1, 3) == 1.0 and G.pick_grounding(1, 1, 1) == pytest.approx(1 / 3)
    assert G.pick_grounding(0, 1, 3) == 0 and G.pick_grounding(1, 0, 3) == 0
    assert G.pick_grounding(1, 1, 9) == 1.0                                      # capped
    near = G.evidence_types("movie", ["a"], 0.9, False)
    far = G.evidence_types("movie", ["a"], 0.31, False)
    assert near == far                                                           # nearer scores the same


def test_overreach_proxy_and_gate():
    EB = _load_script("eval_beyond")
    ctx = EB.Context(reasons={("s", "e", "u1"): "relevance", ("s", "e", "u2"): "taste"})
    assert ctx.is_supported(("s", "e", "u1"), 0) == 0                            # bad because the link was wrong
    assert ctx.is_supported(("s", "e", "u2"), 0) == 1                            # bad for taste: still supported
    assert ctx.is_supported(("s", "e", "u1"), 1) == 1
    ctx2 = EB.Context(supported={("s", "e", "u2"): "n"})
    assert ctx2.is_supported(("s", "e", "u2"), 1) == 0                           # an explicit label wins
    from knots import retrain
    pol = retrain.Policy()
    assert pol.overreach_tolerance == 0.02


def test_eval_reports_grounding_overreach_and_theme_coverage():
    EB = _load_script("eval_beyond")
    cs = [card(f"g{i}", votes=1000, overlap=1.0, sim=0.5 - 0.1 * i, topic=f"t{i % 2}") | {"evidence": ["k"]} for i in range(3)]
    e = _entry("s1", "bedrock", cs, topics=("t0", "t1"))
    labels = [{"session_id": "s1", "extractor": "bedrock", "url": c["url"], "good": g, "reason": r}
              for c, g, r in zip(cs, "yny", ("", "relevance", ""))]
    built = EB.ER.build_lists([e], labels, {"s1": "train"}, None)
    ctx = EB.load_context(labels)
    ctx.themes[("s1", "bedrock")] = [{"topic": "t0", "theme": "a"}, {"topic": "t1", "theme": "b"}]
    res = EB.evaluate(built["train"], built["lab"], EB.B.Pool([], min_pool=1), None, ctx=ctx)["as read aloud"]
    assert res["overreach"]["mean"] == pytest.approx(1 / 3, abs=1e-3)            # one pick labelled relevance-bad
    assert 0 < res["grounding"]["mean"] <= 1
    assert res["theme_coverage"]["mean"] == pytest.approx(0.5, abs=1e-3)         # t0 served by a good pick; t1 only by a bad one
    rep = EB.evidence_report(built["train"], built["lab"], ctx)
    assert rep["picks"] == 3 and set(rep["grounding_by_sim_threshold"]) == {"0.25", "0.30", "0.35"}
