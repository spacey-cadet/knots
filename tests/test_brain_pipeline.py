"""Tests for the learned and trust layers: judge, fit model, receipts, pairing, paper signals,
challenge slot, calibration, and the eval tooling. Offline: no network, no AWS, no keys."""
import asyncio
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from knots import calibration, fitmodel, pairing, receipts, signals  # noqa: E402
from knots import catalogs as C  # noqa: E402
from knots import engine  # noqa: E402
from knots import extraction as I  # noqa: E402
from knots.brain import judge  # noqa: E402
from knots.brain.neighbors import Neighbors  # noqa: E402
from knots.catalogs import ContentItem  # noqa: E402
from knots.plan import build_session_plan  # noqa: E402
from knots.ranking import medium_weight, rank_and_render  # noqa: E402


@pytest.fixture(autouse=True)
def offline_hashed(monkeypatch):
    monkeypatch.setenv("EMBEDDER", "hashed")
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
    monkeypatch.setenv("KNOTS_OFFLINE", "1")
    yield
    fitmodel.reload()
    judge.reload()


def interest(topic="free will", concepts=None, **kw):
    return I.Interest(topic=topic, depth="deep", sentiment="curious", mentions=1,
                      concepts=concepts or [topic], keywords=[topic], **kw)


def item(title, ctype, topic="free will", abstract="", tags=(), rating=0.0, score=0.8, **kw):
    return ContentItem(title=title, content_type=ctype, source_url=f"https://x/{title.replace(' ', '-')}",
                       topic=topic, topic_overlap_score=score, blurb=abstract[:180], abstract=abstract,
                       matched_keywords=list(tags), rating=rating, popularity=500 if rating else 0, **kw)


# ---------------------------------------------------------------- calibration + metrics
def test_calibration_apply_and_meta():
    assert calibration.apply(0.0, 1.0, 0.0) == pytest.approx(0.5)
    assert calibration.apply(10.0, 1.0, 0.0) > 0.99
    assert calibration.from_meta({"calibration": {"a": 2, "b": -1}}) == (2.0, -1.0)
    assert calibration.from_meta({}) is None and calibration.from_meta(None) is None


def test_reliability_and_ece_detect_overconfidence():
    import metrics as M
    p = [0.9] * 10
    assert M.ece(p, [1] * 9 + [0]) < 0.01                  # right about 90% of the time
    assert M.ece(p, [1] * 5 + [0] * 5) == pytest.approx(0.4)   # says 90%, right half the time
    assert M.reliability(p, [1] * 10)[0]["n"] == 10


# ---------------------------------------------------------------- signals
def test_survey_and_reading_level_and_authority():
    assert signals.is_survey("A Survey of Free Will Research") and signals.is_survey("x", "We review the field.")
    assert not signals.is_survey("Bond pricing under stochastic rates")
    plain = "People make choices every day. We think about why. It is a simple question about us."
    dense = ("Notwithstanding methodological heterogeneity, intertemporal discounting parameterizations "
             "demonstrate considerable neurophysiological underdetermination across experimental paradigms.")
    assert signals.reading_level(plain) < signals.reading_level(dense)
    famous = item("P", "paper", citations=4000, is_survey=True)
    obscure = item("Q", "paper", citations=3)
    unknown = item("R", "paper", citations=None)
    assert signals.authority(famous) > signals.authority(obscure) > signals.authority(unknown) == 0.0
    assert signals.authority(item("F", "movie", rating=8)) == 0.0


def test_level_ladder_drama_below_explainer_below_dense_paper():
    drama = item("D", "movie")
    explainer = item("E: explained", "video", duration_s=600)
    lecture = item("Advanced lecture", "video", duration_s=3000)
    assert signals.level_of(drama) < signals.level_of(explainer) < signals.level_of(lecture)


# ---------------------------------------------------------------- arXiv + OpenAlex
ATOM = """<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/1</id><title>A Survey of Free Will</title><summary>We review decades of work on choice.</summary>
<arxiv:journal_ref>Nature 12 (2020)</arxiv:journal_ref></entry>
<entry><id>http://arxiv.org/abs/2</id><title>Narrow Result</title><summary>A very specific lemma.</summary></entry></feed>"""


def test_arxiv_parse_keeps_abstract_status_and_survey():
    a, b = C._parse_arxiv(ATOM, "free will")
    assert a.is_survey and a.status == "published" and a.venue == "Nature 12 (2020)" and "decades" in a.abstract
    assert b.status == "preprint" and not b.is_survey and b.venue == ""


def _openalex_client(payload=None, status=200):
    def handler(request):
        assert "openalex.org" in str(request.url)
        return httpx.Response(status, json=payload or {})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_openalex_attaches_citations_only_for_a_matching_title():
    good = item("A Survey of Free Will", "paper")
    other = item("Totally Different Paper Name Here", "paper")

    async def run(payload):
        client = _openalex_client(payload)
        await C._openalex_enrich([good, other], client)
        await client.aclose()

    asyncio.run(run({"results": [{"title": "A survey of free will", "cited_by_count": 321}]}))
    assert good.citations == 321 and other.citations is None   # near-identical title only


def test_openalex_failure_leaves_citations_unknown_and_is_recorded():
    p = item("Some Paper", "paper")
    C.FAILURES.clear()

    async def run():
        client = _openalex_client(status=500)
        await C._openalex_enrich([p], client)
        await client.aclose()

    asyncio.run(run())
    assert p.citations is None and any(f.startswith("OpenAlex") for f in C.FAILURES)


# ---------------------------------------------------------------- receipts
def test_your_words_returns_the_sentence_or_nothing():
    q = "Is free will an illusion? I also like pizza. Tell me about memecoins."
    assert receipts.your_words(q, ["free will"]) == "Is free will an illusion?"
    assert receipts.your_words(q, ["quantum chromodynamics"]) == ""
    long = "free will " + "word " * 60
    assert receipts.your_words(long, ["free will"]).endswith("...")


def test_receipts_distinguish_preprints_from_published_and_require_a_source():
    it = interest()
    pre = item("Narrow Result", "paper", abstract="A very specific lemma about choice.", status="preprint", citations=12)
    pub = item("Big Review", "paper", abstract="We review choice.", status="published", venue="Nature 1 (2020)")
    kinds = lambda r: {x["kind"]: x["text"] for x in r}  # noqa: E731
    assert kinds(receipts.build(pre, it, ""))["status"] == "Preprint, not peer reviewed"
    assert kinds(receipts.build(pre, it, ""))["citations"] == "12 citations (OpenAlex)"
    assert kinds(receipts.build(pub, it, ""))["status"].startswith("Published: Nature")
    bare = item("Mystery", "movie")                           # no tags, no overview
    assert not receipts.sourced(receipts.build(bare, it, "free will"))
    assert receipts.sourced(receipts.build(item("X", "movie", tags=["free will"]), it, ""))


# ---------------------------------------------------------------- fit model + ranking
def test_zscore_flat_for_constant_input():
    assert fitmodel.zscore([0.3, 0.3, 0.3]) == [0.0, 0.0, 0.0]
    z = fitmodel.zscore([1.0, 2.0, 3.0])
    assert z[0] < 0 < z[2] and sum(z) == pytest.approx(0.0)


QUERY = "Is free will an illusion? Neuroscience says decisions are made before we are aware of them."


def test_session_similarity_beats_a_constant_overlap_score():
    it = interest()
    on = item("The Illusion of Free Will", "paper", abstract="Neuroscience of decisions made before awareness", tags=["free will"])
    off = item("Bond pricing under stochastic rates", "paper", abstract="Interest rate term structure models", tags=["free will"])
    sc = fitmodel.score_items(QUERY, [off, on], [it], medium_weight)
    assert sc.scores[1] > sc.scores[0] and sc.sims[1] > sc.sims[0]


def test_confidence_is_none_without_calibration_and_a_number_with_it(monkeypatch, tmp_path):
    it = interest()
    items = [item("A", "paper", abstract="free will neuroscience", tags=["t"]), item("B", "paper", abstract="bond rates", tags=["t"])]
    assert fitmodel.score_items(QUERY, items, [it], medium_weight).confidence(0) is None
    f = tmp_path / "w.json"
    f.write_text(json.dumps({"weights": fitmodel.DEFAULT_WEIGHTS, "calibration": {"a": 1.0, "b": 0.0}}))
    monkeypatch.setattr(fitmodel, "WEIGHTS_FILE", f)
    fitmodel.reload()
    sc = fitmodel.score_items(QUERY, items, [it], medium_weight)
    assert 0.0 < sc.confidence(0) < 1.0 and fitmodel.trained()
    assert fitmodel.score_items("", items, [it], medium_weight).confidence(0) is None   # no query, no claim


def test_rank_with_query_adds_receipts_and_drops_unsourced_cards():
    it = interest()
    good = item("Free Will Film", "movie", abstract="A man questions whether his choices are his.", tags=["free will"], rating=7.5)
    bare = item("Mystery Film", "movie", rating=8.0)                       # no tags, no overview
    cards = rank_and_render([bare, good], [it], limit=5, query=QUERY)
    assert [c["title"] for c in cards] == ["Free Will Film"]
    c = cards[0]
    assert {"you_said", "tags", "summary"} <= {r["kind"] for r in c["receipts"]}
    assert set(c["fit_features"]) == set(fitmodel.FEATURES) and c["confidence"] is None and c["level"] is not None


def test_rank_without_query_is_the_original_order_and_unchanged_cards():
    it = interest()
    a = item("A", "movie", rating=6.6, score=0.9)
    b = item("B", "movie", topic="free will", rating=9.0, score=0.5)
    cards = rank_and_render([b, a], [it], limit=5)
    assert cards[0]["title"] == "A" and "receipts" not in cards[0]


def test_paper_authority_moves_a_well_cited_survey_up():
    it = interest()
    kw = dict(abstract="free will and the neuroscience of decisions", tags=["free will"])
    plain = item("Paper Plain", "paper", **kw)
    famous = item("Paper Famous", "paper", citations=4000, is_survey=True, **kw)
    sc = fitmodel.score_items(QUERY, [plain, famous], [it], medium_weight)
    assert sc.scores[1] > sc.scores[0]


# ---------------------------------------------------------------- pairing
def test_level_match_is_a_ladder():
    assert pairing.level_match(0.2, 0.4) == 1.0 and pairing.level_match(0.2, 0.2) == 1.0
    assert pairing.level_match(0.5, 0.4) == 0.7 and pairing.level_match(0.9, 0.1) == 0.5


def _pair_set():
    it = interest()
    film = item("The Free Will Film", "movie", abstract="A neuroscientist discovers decisions are made before awareness", tags=["free will", "neuroscience"], rating=7.8)
    paper = item("Decisions Before Awareness", "paper", abstract="Neuroscience evidence that decisions are made before awareness", tags=[], status="preprint")
    return it, film, paper


def test_pair_built_with_a_factual_bridge():
    it, film, paper = _pair_set()
    pairs = pairing.build_pairs([film, paper], [it], QUERY)
    assert len(pairs) == 1
    p = pairs[0]
    assert p["watch"]["title"] == film.title and p["learn"]["title"] == paper.title
    assert "Watch The Free Will Film" in p["bridge"] and "neuroscience" in p["bridge"]   # shared term is literal
    assert 0 < p["score"] <= 1 and p["components"]["level_match"] in (0.5, 0.7, 1.0)


def test_no_pair_across_topics_or_when_unrelated_or_unsourced():
    it, film, paper = _pair_set()
    other = interest("tax law")
    paper_other = item("Decisions Before Awareness", "paper", topic="tax law", abstract=paper.abstract, status="preprint")
    assert pairing.build_pairs([film, paper_other], [it, other], QUERY) == []          # different interests
    unrelated = item("Bond pricing", "paper", abstract="Interest rate term structure models only.", tags=[])
    assert pairing.build_pairs([film, unrelated], [it], QUERY) == []                   # not the same idea
    bare_paper = item("Decisions Before Awareness", "paper")                             # no abstract: unsourced
    assert pairing.build_pairs([film, bare_paper], [it], QUERY) == []


def test_one_pair_per_interest_and_each_item_used_once():
    it, film, paper = _pair_set()
    film2 = item("Another Free Will Film", "series", abstract=film.abstract, tags=["free will"], rating=7.5)
    paper2 = item("Awareness of Decisions Revisited", "paper", abstract=paper.abstract, status="preprint")
    pairs = pairing.build_pairs([film, film2, paper, paper2], [it], QUERY)
    assert len(pairs) == 1                                   # one pair per interest


def test_plan_exposes_pairs_and_speaks_the_double_feature():
    it, film, paper = _pair_set()
    plan = build_session_plan([it], [film, paper], extractor="bedrock", query=QUERY)
    assert plan["pairs"] and "double feature" in plan["spoken"]


# ---------------------------------------------------------------- challenge slot
def test_challenge_search_tags_items_with_the_counterpoint(monkeypatch):
    seen = {}

    async def fake(interest_, client):
        seen["kw"] = interest_.keywords
        return [item("Counter Film", "movie", tags=["x"], rating=8)]

    monkeypatch.setattr(C, "_search_tmdb", fake)
    it = interest(counterpoint="Some argue choice is compatible with determinism.", counter_keywords=["compatibilism"])
    out = asyncio.run(C._challenge_search(it, None))
    assert seen["kw"] == ["compatibilism"] and out[0].challenge.startswith("Some argue")


def test_challenge_slot_is_reserved_and_explained():
    it = interest()
    normal = [item(f"Film {n}", "movie", abstract="free will film", tags=["free will"], rating=7.5, score=0.9 - n / 10) for n in range(4)]
    chal = item("Compatibilism Doc", "doc", abstract="A documentary on free will and determinism", tags=["compatibilism"],
                rating=8.0, score=0.1, challenge="Some argue choice is compatible with determinism.")
    cards = rank_and_render(normal + [chal], [it], limit=6, query=QUERY)
    c = next(c for c in cards if c["title"] == "Compatibilism Doc")
    assert c["challenge"] and "pushes back on your view" in c["reasoning"]


def test_challenge_off_by_default(monkeypatch):
    monkeypatch.delenv("KNOTS_CHALLENGE", raising=False)
    assert C.challenge_enabled(None) is False and C.challenge_enabled(True) is True
    monkeypatch.setenv("KNOTS_CHALLENGE", "1")
    assert C.challenge_enabled(None) is True and C.challenge_enabled(False) is False


def test_extraction_parses_counterpoint_and_cleans_keywords():
    out = I._parse_interests({"interests": [{"topic": "free will", "concepts": ["free will"], "keywords": ["x"],
                                             "counterpoint": "Compatibilists disagree.", "counter_keywords": ["Compatibilism", 7, "compatibilism"]}]})
    assert out[0].counterpoint == "Compatibilists disagree." and out[0].counter_keywords == ["compatibilism"]


# ---------------------------------------------------------------- engine query
def test_query_text_keeps_the_newest_words():
    old = "old " * 3000
    q = engine.query_text([old, "the newest thing I asked"], None)
    assert q.endswith("the newest thing I asked") and len(q) <= engine.QUERY_CHARS
    from knots.context import Conversation
    assert "hello" in engine.query_text(None, [Conversation(turns=["hello"])])


# ---------------------------------------------------------------- judge
def test_neighbors_vote_and_leave_one_group_out():
    n = Neighbors(["whether choice is meaningful", "gap between knowledge and will", "fear of losing selfhood",
                   "anxiety about personal agency"], [1, 1, 0, 0], ["a", "b", "c", "d"])
    assert n.score("whether a choice is meaningful") > 0 > n.score("fear of losing personhood")
    assert n.score("whether choice is meaningful", exclude_group="a") != n.score("whether choice is meaningful")
    assert Neighbors([], [], []).score("x") is None


def test_judge_ships_data_and_drops_unstated_drama(monkeypatch):
    monkeypatch.delenv("KNOTS_BRAIN", raising=False)
    judge.reload()
    text = "Is free will an illusion? I keep reading that decisions happen before we are aware of them."
    kept, dropped, _ = judge.assess(text, ["whether choice is meaningful if determined", "fear of losing selfhood and the collapse of personhood"])
    assert "whether choice is meaningful if determined" in kept and "fear of losing selfhood and the collapse of personhood" in dropped


def test_judge_confidence_only_when_calibrated(monkeypatch):
    judge.reload()
    text = "whether choice is meaningful"
    monkeypatch.setattr(judge, "_meta", lambda: {"threshold": 0.0})            # no calibration block
    _, _, conf = judge.assess(text, ["whether choice is meaningful if determined"])
    assert conf == {} and judge.probability(text, "whether choice is meaningful") is None
    monkeypatch.setattr(judge, "_meta", lambda: {"threshold": 0.0, "calibration": {"a": 1.0, "b": 0.0}})
    _, _, conf = judge.assess(text, ["whether choice is meaningful if determined"])
    assert list(conf.values())[0] > 0


def test_judge_off_switch_never_drops(monkeypatch):
    monkeypatch.setenv("KNOTS_BRAIN", "off")
    t = ["fear of losing selfhood and the collapse of personhood"]
    assert judge.assess("x", t) == (t, [], {})


def test_plan_reports_withheld_themes_and_confidence_fields():
    it = interest(implicit_themes=["whether choice is meaningful"])
    it.dropped_themes = ["fear of losing selfhood"]
    it.theme_confidence = {"whether choice is meaningful": 0.8}
    th = build_session_plan([it], [], "bedrock", "")["threads"][0]
    assert th["withheld_themes"] == 1 and th["underneath_confidence"] == {"whether choice is meaningful": 0.8}


def test_new_interest_fields_survive_a_memory_round_trip():
    from knots import memory as M
    it = interest(counterpoint="c", counter_keywords=["k"], implicit_themes=["t"])
    back = M._interest_from_dict(it.to_dict())
    assert back.counterpoint == "c" and back.counter_keywords == ["k"]
    assert M._interest_from_dict({k: v for k, v in it.to_dict().items() if k != "counterpoint"}).counterpoint == ""


# ---------------------------------------------------------------- repo hygiene (regression for the .gitignore bug)
def test_trained_brain_data_ships_and_is_not_gitignored():
    for name in ("theme_critic.json", "theme_examples.jsonl", "theme_judge.json"):
        path = ROOT / "knots" / "brain" / "data" / name
        assert path.exists(), name
        ignored = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=ROOT).returncode == 0
        assert not ignored, f"{name} is gitignored, so a fresh clone would not have it"
    assert (ROOT / "knots/data/knots_profile.db").parent.name == "data"
    assert subprocess.run(["git", "check-ignore", "-q", "knots/data/knots_profile.db"], cwd=ROOT).returncode == 0


# ---------------------------------------------------------------- eval tooling
def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "eval" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_check_labels_named_errors_skips_negated_mentions():
    cl = _load_script("check_labels")
    assert cl.named_errors("This is terminological overreach and narrative projection.")[0] == "terminological_overreach"
    assert cl.named_errors("Supported, without any narrative overreach or normative expansion.") == []
    assert cl.named_errors("a causal projection error") == ["causal_projection"]


def test_build_label_sheets_carries_split_error_type_and_pairs(tmp_path, monkeypatch):
    R = _load_script("run_eval")
    for n in ("LABELS", "THEMES", "PAIRS"):
        monkeypatch.setattr(R, n, tmp_path / f"{n.lower()}.csv")
    sessions = tmp_path / "sessions.json"
    sessions.write_text(json.dumps([{"id": "s1", "texts": ["x"], "split": "holdout"}]))
    monkeypatch.setattr(R, "SESSIONS", sessions)
    card = {"url": "u", "type": "movie", "title": "T", "year": "", "evidence": [], "reasoning": "r"}
    pair = {"concept": "c", "score": 0.3, "bridge": "b", "watch": {"url": "w", "title": "W"}, "learn": {"url": "l", "title": "L"}}
    res = [{"ok": True, "session_id": "s1", "extractor": "bedrock", "texts": ["x"],
            "interests": [{"topic": "t", "implicit_themes": ["th"], "rationale": ""}],
            "plan": {"threads": [{"topic": "t"}], "picks": {"watch": [card], "read": [], "explore": []}, "pairs": [pair]}}]
    R.build_label_sheets(res)
    assert R.read_csv(R.LABELS)[0]["split"] == "holdout" and R.read_csv(R.THEMES)[0]["split"] == "holdout"
    p = R.read_csv(R.PAIRS)[0]
    assert p["watch_title"] == "W" and p["split"] == "holdout" and p["good"] == ""
    # relabel, rebuild: the label and error_type survive
    rows = R.read_csv(R.THEMES); rows[0]["supported"], rows[0]["error_type"] = "n", "narrative_projection"
    R.write_csv(R.THEMES, R.THEME_COLS, rows)
    R.build_label_sheets(res)
    t = R.read_csv(R.THEMES)[0]
    assert t["supported"] == "n" and t["error_type"] == "narrative_projection"


def test_train_rank_evaluates_on_held_out_sessions_and_pair_summary():
    import numpy as np
    TR = _load_script("train_rank")
    rng = np.random.default_rng(0)
    rows = []
    for s in range(6):
        for _ in range(8):
            x = rng.normal(size=len(fitmodel.FEATURES))
            rows.append({"session": f"s{s}", "split": "train", "type": "movie", "x": x.tolist(),
                         "y": int(x[0] + 0.3 * rng.normal() > 0), "old": float(rng.random())})
    X, y, g, loso, res, (a, b), rel, ece = TR.evaluate(rows)
    assert res["fitted (leave-session-out)"]["auc"] > 0.7           # the signal is learnable...
    assert res["fitted (leave-session-out)"]["auc"] > res["old overlap score"]["auc"]   # ...and beats noise
    EP = _load_script("eval_pairs")
    out = EP.summarize([{"good": "y", "session_id": "a", "pair_score": "0.4"}, {"good": "n", "session_id": "b", "pair_score": "0.1"},
                        {"good": "", "session_id": "c", "pair_score": "0.2"}])
    assert out["pairs"] == 2 and out["success_rate"] == 0.5 and EP.summarize([]) is None


def test_model_card_numbers_come_from_metrics_files(tmp_path, monkeypatch):
    MC = _load_script("make_model_card")
    monkeypatch.setattr(MC, "ROOT", tmp_path); monkeypatch.setattr(MC, "RES", tmp_path)
    (tmp_path / "metrics_brain.json").write_text(json.dumps({
        "train_labels": 50, "train_sessions": 5,
        "loso": {k: {"auc": 0.77, "auc_ci90": [0.6, 0.9], "within_session_auc": 0.8} for k in ("critic", "neighbors", "ensemble")}
                | {"at_threshold": {"threshold": 0.5, "kept": 10, "of": 50, "precision": 0.7, "base_rate": 0.4}},
        "calibration": {"ece": 0.05, "reliability": [{"bin": "0.0-0.2", "n": 5, "predicted": 0.1, "observed": 0.2}]}, "holdout": None}))
    MC.main()
    card = (tmp_path / "MODEL_CARD.md").read_text()
    assert "0.77 [0.60, 0.90]" in card and "Holdout: not yet measured" in card
    assert "Not yet measured" in card.split("## Fit ranker")[1]            # ranker had no metrics file
    assert "TODO (author)" in card                                          # provenance is the author's to state


# ---------------------------------------------------------------- embedder is recorded with every measurement
def test_embedder_info_reads_what_the_eval_recorded_and_never_guesses(tmp_path):
    import metrics as M
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"tmdb_live": True, "results": [{"ok": True}]}))        # written before tracking
    assert M.embedder_info(old) == {"recorded": False} and M.embedder_info(tmp_path / "missing.json") == {"recorded": False}

    cfg = {"requested": "auto", "provider": "cohere", "model_id": "global.cohere.embed-v4:0", "dimensions": 256}
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"embedder": cfg, "results": [
        {"ok": True, "embedders_used": {"cohere": 2}}, {"ok": True, "embedders_used": {"cohere": 1}}]}))
    info = M.embedder_info(ok)
    assert info["provider"] == "cohere" and info["model_id"] == "global.cohere.embed-v4:0"
    assert info["runs_by_embedder"] == {"cohere": 2} and info["fell_back_runs"] == 0

    mixed = tmp_path / "mixed.json"
    mixed.write_text(json.dumps({"embedder": cfg, "results": [
        {"ok": True, "embedders_used": {"cohere": 1}}, {"ok": True, "embedders_used": {"hashed": 1}}]}))
    info = M.embedder_info(mixed)
    assert info["provider"] == "mixed" and info["fell_back_runs"] == 1


def test_scoring_counts_the_embedder_that_actually_ran(monkeypatch):
    monkeypatch.setenv("EMBEDDER", "hashed")
    fitmodel.USED_EMBEDDERS.clear()
    on = item("The Illusion of Free Will", "paper", abstract="Neuroscience of decisions", tags=["free will"])
    fitmodel.score_items(QUERY, [on], [interest()], medium_weight)
    assert dict(fitmodel.USED_EMBEDDERS) == {"hashed": 1}
    fitmodel.USED_EMBEDDERS.clear()


def test_model_card_states_the_embedder_for_each_part_and_warns_on_fallback(tmp_path, monkeypatch):
    MC = _load_script("make_model_card")
    monkeypatch.setattr(MC, "ROOT", tmp_path); monkeypatch.setattr(MC, "RES", tmp_path)
    MC.main()                                                               # nothing recorded anywhere
    card = (tmp_path / "MODEL_CARD.md").read_text()
    assert "## Embedder" in card and card.count("not recorded") == 2        # ranker and pairs; the judge has none

    cohere = {"recorded": True, "provider": "cohere", "requested": "auto", "model_id": "global.cohere.embed-v4:0",
              "dimensions": 256, "runs_by_embedder": {"cohere": 3}, "fell_back_runs": 0}
    fell = {**cohere, "provider": "mixed", "runs_by_embedder": {"cohere": 2, "hashed": 1}, "fell_back_runs": 1}
    (tmp_path / "metrics_rank.json").write_text(json.dumps({"status": "not_trained", "reason": "x", "embedder": cohere}))
    (tmp_path / "metrics_pairs.json").write_text(json.dumps({"all": None, "embedder": fell}))
    (tmp_path / "metrics_brain.json").write_text(json.dumps({
        "embedder": {"recorded": True, "provider": "none", "note": "character n-gram TF-IDF; no embedding model"},
        "train_labels": 1, "train_sessions": 1,
        "loso": {k: {"auc": 0.7, "auc_ci90": [0.6, 0.8], "within_session_auc": 0.7} for k in ("critic", "neighbors", "ensemble")}
                | {"at_threshold": {"threshold": 0.5, "kept": 1, "of": 2, "precision": 0.5, "base_rate": 0.5}},
        "calibration": {"ece": 0.1, "reliability": []}, "holdout": None}))
    MC.main()
    card = (tmp_path / "MODEL_CARD.md").read_text()
    sect = card.split("## Embedder")[1].split("## How the labels")[0]
    assert "no embedding model is involved" in sect                         # theme judge: always, regardless of files
    assert "global.cohere.embed-v4:0" in sect and "**cohere**" in sect      # ranker
    assert "**mixed**" in sect and "different embedders" in sect            # pairs fell back: flagged
    assert "1 eval run(s)" in sect


def test_run_eval_resume_keeps_complete_runs_and_stops_cleanly_when_the_youtube_budget_is_spent(tmp_path, monkeypatch, capsys):
    import argparse
    from knots import catalogs
    R = _load_script("run_eval")
    sessions = [{"id": f"s{i}", "texts": [f"t{i}"]} for i in (1, 2, 3)]
    (tmp_path / "sessions.json").write_text(json.dumps(sessions))
    monkeypatch.setattr(R, "SESSIONS", tmp_path / "sessions.json")
    monkeypatch.setattr(R, "RESULTS", tmp_path / "results.json")
    monkeypatch.setattr(R, "ROOT", tmp_path)                                    # the summary prints paths relative to it
    for n in ("LABELS", "THEMES", "PAIRS"):
        monkeypatch.setattr(R, n, tmp_path / f"{n.lower()}.csv")
    monkeypatch.setattr(R, "set_extractor", lambda *a: None)
    monkeypatch.setattr(R, "build_label_sheets", lambda results: None)
    monkeypatch.setenv("YOUTUBE_API_KEY", "k"); monkeypatch.setenv("TMDB_API_KEY", "k")
    monkeypatch.setenv("KNOTS_YOUTUBE_CACHE", str(tmp_path / "yt.json"))

    state = {"left": 2, "ran": [], "fail": {"s2"}}
    monkeypatch.setattr(catalogs, "youtube_searches_left", lambda: state["left"])
    monkeypatch.setattr(catalogs, "youtube_blocked_reason", lambda: "")

    class FakeResult:
        interests, extractor = [], "heuristic"
        def plan(self): return {"threads": [], "picks": {"watch": [], "read": [], "explore": []}, "pairs": []}

    async def fake_run(session_texts, include_videos):
        sid = session_texts[0].replace("t", "s")
        state["ran"].append(sid)
        state["left"] -= 1
        if sid in state["fail"]:
            catalogs.FAILURES.append("YouTube:x:429")
        return FakeResult()

    monkeypatch.setattr(R.engine, "run", fake_run)
    args = lambda **kw: argparse.Namespace(extractors="heuristic", no_videos=False, resume=False, youtube_budget=None, **kw)

    R.cmd_run(args())
    out = capsys.readouterr().out
    assert state["ran"] == ["s1", "s2"]                                        # stopped before s3: budget is 0
    assert "STOPPED before s3" in out and "run --resume" in out
    saved = json.loads((tmp_path / "results.json").read_text())["results"]
    assert [e["session_id"] for e in saved] == ["s1", "s2"]                     # saved as it went
    assert saved[1]["catalog_failures"] and not saved[0]["catalog_failures"]    # s2 is INCOMPLETE

    state.update(left=5, ran=[], fail=set())                                    # next day: quota is back
    R.cmd_run(argparse.Namespace(extractors="heuristic", no_videos=False, resume=True, youtube_budget=None))
    out = capsys.readouterr().out
    assert state["ran"] == ["s2", "s3"]                                         # s1 was complete: not redone
    assert "kept (complete" in out and "STOPPED" not in out
    final = json.loads((tmp_path / "results.json").read_text())["results"]
    assert [e["session_id"] for e in final] == ["s1", "s2", "s3"] and not any(e["catalog_failures"] for e in final)


# ---------------------------------------------------------------- ranking metrics and the order-quality baseline
def test_ranking_metrics_match_hand_calculations():
    import math
    import metrics as M
    r = [1, 0, 1, 0]
    assert M.p_at_k(r, 1) == 1.0 and M.p_at_k(r, 3) == pytest.approx(2 / 3) and M.mrr(r) == 1.0
    assert M.average_precision(r) == pytest.approx((1 + 2 / 3) / 2)
    assert M.ndcg_at_k(r, 5) == pytest.approx((1 + 0.5) / (1 + 1 / math.log2(3)))      # dcg 1.5 over the ideal [1,1,0,0]
    late = [0, 0, 1]
    assert M.hit_at_k(late, 1) == 0 and M.hit_at_k(late, 3) == 1 and M.mrr(late) == pytest.approx(1 / 3)
    assert M.ndcg_at_k(late, 5) == pytest.approx(0.5) and M.p_at_k([1], 3) == 1.0       # a short list is judged on what it has
    none = [0, 0]
    assert math.isnan(M.average_precision(none)) and math.isnan(M.ndcg_at_k(none, 3)) and M.mrr(none) == 0.0


def test_summarize_rankings_resamples_sessions_and_reports_lists_with_nothing_good():
    import metrics as M
    groups = [{"session": "a", "rels": [1, 0]}, {"session": "a", "rels": [1, 1]},
              {"session": "b", "rels": [0, 1]}, {"session": "c", "rels": [0, 0]}]
    s = M.summarize_rankings(groups, n_boot=300)
    assert s["lists"] == 4 and s["sessions"] == 3 and s["lists_without_a_good_pick"] == 1
    assert s["p@1"]["mean"] == 0.5                                          # 1, 1, 0, 0
    assert s["map"]["mean"] == pytest.approx((1.0 + 1.0 + 0.5) / 3, abs=1e-3)   # the all-bad list is not scored
    assert s["p@1"]["ci90"][0] <= 0.5 <= s["p@1"]["ci90"][1]
    perfect = M.summarize_rankings([{"session": "a", "rels": [1, 1]}, {"session": "b", "rels": [1]}], n_boot=100)
    assert perfect["p@1"] == {"mean": 1.0, "ci90": [1.0, 1.0]}
    chance = M.summarize_rankings(M.expected_random([{"session": "a", "rels": [1, 0, 0, 0]}], draws=400))
    assert chance["p@1"]["mean"] == pytest.approx(0.25, abs=0.07)           # one good item in four: 25% by chance


def test_eval_rank_orders_as_read_aloud_and_leaves_out_unlabelled_and_holdout_lists():
    ER = _load_script("eval_rank")

    def card(url, fit, kind="watch", overlap=0.0):
        return {"url": url, "fit_score": fit, "kind": kind, "fit_features": {"overlap": overlap, "sim_session": fit}}

    def entry(sid, picks, mediums):
        return {"ok": True, "session_id": sid, "extractor": "bedrock", "interests": [{"mediums": mediums}],
                "plan": {"picks": picks}}

    e1 = entry("s1", {"watch": [card("w1", 0.1), card("w2", 0.9)], "read": [card("r1", 0.5, "read")], "explore": []},
               ["paper", "film", "video"])
    assert [c["url"] for c in ER.displayed_order(e1)] == ["r1", "w1", "w2"]          # lead medium first, then the rest
    e2 = entry("s2", {"watch": [card("x1", 0.3), card("x2", 0.2)], "read": [], "explore": []}, ["film"])
    e3 = entry("s3", {"watch": [card("y1", 0.3), card("y2", 0.2)], "read": [], "explore": []}, ["film"])
    e4 = entry("s4", {"watch": [card("z1", 0.3)], "read": [], "explore": []}, ["film"])
    labels = [{"session_id": "s1", "extractor": "bedrock", "url": u, "good": g}
              for u, g in (("r1", "n"), ("w1", "n"), ("w2", "y"))]
    labels += [{"session_id": "s2", "extractor": "bedrock", "url": "x1", "good": "y"},
               {"session_id": "s2", "extractor": "bedrock", "url": "x2", "good": "n"},
               {"session_id": "s3", "extractor": "bedrock", "url": "y1", "good": "y"},     # y2 unlabelled: left out
               {"session_id": "s4", "extractor": "bedrock", "url": "z1", "good": "y"}]     # holdout session
    splits = {"s1": "train", "s2": "train", "s3": "train", "s4": "holdout"}
    built = ER.build_lists([e1, e2, e3, e4], labels, splits)
    assert [e["session_id"] for e, _ in built["train"]] == ["s1", "s2"]
    assert built["excluded"]["partly_unlabelled"] == [["s3", "bedrock", 1, 2]]
    assert built["excluded"]["holdout_lists"] == 1 and len(built["holdout"]) == 1
    read = ER.groups_for(built["train"], built["lab"], "as read aloud")
    fit = ER.groups_for(built["train"], built["lab"], "fit score")
    assert read[0]["rels"] == [0, 0, 1] and fit[0]["rels"] == [1, 0, 0]               # the ranker's order puts w2 first
    res = ER.evaluate(built["train"], built["lab"])
    assert res["as read aloud"]["p@1"]["mean"] == 0.5 and res["fit score"]["p@1"]["mean"] == 1.0
    assert "random (chance)" in res and ER.evaluate([], {}) is None


def test_model_card_has_a_ranking_section_with_numbers_only_from_the_file(tmp_path, monkeypatch):
    MC = _load_script("make_model_card")
    monkeypatch.setattr(MC, "ROOT", tmp_path); monkeypatch.setattr(MC, "RES", tmp_path)
    MC.main()
    assert "Not yet measured: run `python eval/eval_rank.py`" in (tmp_path / "MODEL_CARD.md").read_text()
    cell = lambda m: {"mean": m, "ci90": [round(m - 0.1, 3), round(m + 0.1, 3)]}  # noqa: E731
    row = {"set_precision": cell(0.6), "lists_without_a_good_pick": 2, "lists": 9} | {h: cell(0.42) for h in ("p@1", "p@3", "map", "ndcg@5")}
    (tmp_path / "metrics_ranking.json").write_text(json.dumps({
        "embedder": {"recorded": True, "provider": "cohere", "requested": "auto", "model_id": "m", "dimensions": 256},
        "headline": ["p@1", "p@3", "map", "ndcg@5"],
        "bedrock": {"lists": 9, "sessions": 7, "excluded": {"partly_unlabelled": [["s1", "bedrock", 1, 2]]}, "holdout": None,
                    "train": {"as read aloud": row, "random (chance)": row}}}))
    MC.main()
    card = (tmp_path / "MODEL_CARD.md").read_text()
    sect = card.split("## Ranking quality")[1].split("## Watch-and-learn")[0]
    assert "9 complete lists from 7 train sessions" in sect and "0.42 [0.32, 0.52]" in sect
    assert "60% of everything shown was good" in sect and "1 list(s) were left out" in sect
    assert "not yet measured (holdout sessions are unlabelled)" in sect and "**cohere**" in sect


# ---------------------------------------------------------------- theme curation
from knots.brain import themes as TH  # noqa: E402

SAID = "I keep seeing memecoin threads on my Reddit feed, is this all just a bubble? Tell me more about the research."


def test_quote_supported_ignores_case_and_punctuation_and_needs_real_words():
    assert TH.quote_supported("is this all just a bubble", SAID)
    assert TH.quote_supported("IS this ALL just a bubble?!", SAID)
    assert TH.quote_supported("I keep seeing ... on my Reddit feed", SAID)          # ellipsis skips words, in order
    assert not TH.quote_supported("Reddit feed ... I keep seeing", SAID)             # out of order
    assert not TH.quote_supported("people feel anxious about money", SAID)           # never said
    assert not TH.quote_supported("a bubble", SAID)                                  # real but under 3 words
    assert not TH.quote_supported("", SAID) and not TH.quote_supported(None, SAID)


def test_curate_drops_unquoted_themes_ranks_and_caps_at_five(monkeypatch):
    monkeypatch.setattr(TH.judge, "assess", lambda text, themes: (list(themes), [], {}))
    monkeypatch.setattr(TH.judge, "raw_score", lambda text, theme: None)
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()
    text = " ".join(f"{w} one two three." for w in words)
    a = interest("a", implicit_themes=[f"t{i}" for i in range(4)] + ["invented", "unquoted"],
                 theme_evidence={**{f"t{i}": f"{words[i]} one two three" for i in range(4)}, "invented": "never said this at all"})
    b = interest("b", implicit_themes=["t4", "t5", "t0"],
                 theme_evidence={"t4": f"{words[4]} one two three", "t5": f"{words[5]} one two three", "t0": f"{words[0]} one two"})
    out = TH.curate([a, b], text, cap=5)
    kept = [t for it in out for t in it.implicit_themes]
    assert len(kept) == 5 and len(set(kept)) == 5                                     # cap holds, duplicate t0 counted once
    assert "invented" in a.dropped_themes and "quote not found" in a.theme_dropped_why["invented"]
    assert a.theme_dropped_why["unquoted"] == "no quote from the conversation"
    assert any("over the limit of 5" in why for it in out for why in it.theme_dropped_why.values())
    ranks = sorted(r for it in out for r in it.theme_rank.values())
    assert ranks == [1, 2, 3, 4, 5]


def test_curate_orders_by_judge_confidence_and_drops_what_the_judge_rejects(monkeypatch):
    conf = {"fear of missing out": 0.9, "algorithmic amplification of risk": 0.6}
    monkeypatch.setattr(TH.judge, "assess", lambda text, themes: (
        [t for t in themes if t in conf], [t for t in themes if t not in conf], conf))
    it = interest("memecoins", implicit_themes=["algorithmic amplification of risk", "fear of missing out", "moral panic"],
                  theme_evidence={"algorithmic amplification of risk": "is this all just a bubble",
                                  "fear of missing out": "I keep seeing memecoin threads on my Reddit feed",
                                  "moral panic": "Tell me more about the research"})
    TH.curate([it], SAID)
    assert it.implicit_themes[0] == "fear of missing out" and it.theme_rank["fear of missing out"] == 1
    assert it.theme_dropped_why["moral panic"] == "the judge expects this would be rejected"
    rows = TH.ranked([it])
    assert [r["theme"] for r in rows] == ["fear of missing out", "algorithmic amplification of risk"]
    assert rows[0]["evidence"].startswith("I keep seeing") and [r["rank"] for r in rows] == [1, 2]


def test_parse_themes_reads_quotes_and_legacy_strings_and_limits_to_five():
    themes, ev = I._parse_themes({"themes": [{"theme": " Fear  of Missing Out ", "evidence": " is this  all just a bubble "},
                                             {"theme": "fear of missing out", "evidence": "dup"}, {"theme": 7},
                                             "bare string", *({"theme": f"x{i}", "evidence": "a b c"} for i in range(9))]})
    assert themes[0] == "fear of missing out" and ev["fear of missing out"] == "is this all just a bubble"
    assert "bare string" in themes and "bare string" not in ev and len(themes) == 5
    legacy, lev = I._parse_themes({"implicit_themes": ["old style", 3]})
    assert legacy == ["old style"] and lev == {}                                      # no quote, so curation will drop it


def test_plan_exposes_ranked_themes_with_their_quotes(monkeypatch):
    monkeypatch.setattr(TH.judge, "assess", lambda text, themes: (list(themes), [], {}))
    it = interest("memecoins", implicit_themes=["fear of missing out"],
                  theme_evidence={"fear of missing out": "I keep seeing memecoin threads on my Reddit feed"})
    TH.curate([it], SAID)
    plan = build_session_plan([it], [item("Margin Call", "film", "memecoins", tags=["memecoins"], rating=7.0)],
                              extractor="heuristic", query=SAID)
    assert plan["themes"] and plan["themes"][0]["theme"] == "fear of missing out"
    assert plan["themes"][0]["evidence"].startswith("I keep seeing")
    assert len(plan["themes"]) <= 5


def test_run_eval_themes_command_writes_themes_run_and_leaves_pick_labels_alone(tmp_path, monkeypatch):
    import argparse
    R = _load_script("run_eval")
    (tmp_path / "sessions.json").write_text(json.dumps([{"id": "s1", "texts": [SAID]}]))
    for n, v in {"SESSIONS": "sessions.json", "RESULTS": "results.json", "THEMES_RUN": "themes_run.json",
                 "LABELS": "labels.csv", "THEMES": "themes.csv", "PAIRS": "pairs.csv"}.items():
        monkeypatch.setattr(R, n, tmp_path / v)
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.setattr(R, "set_extractor", lambda *a: None)
    (tmp_path / "labels.csv").write_text("sentinel")
    monkeypatch.setattr(TH.judge, "assess", lambda text, themes: (list(themes), [], {}))

    async def fake_extract(texts):
        it = interest("memecoins", implicit_themes=["fear of missing out"],
                      theme_evidence={"fear of missing out": "is this all just a bubble"})
        TH.curate([it], texts[0])
        return [it], "bedrock"
    monkeypatch.setattr(R.extraction, "extract", fake_extract)
    assert R.cmd_themes(argparse.Namespace(extractors="bedrock")) == 0
    run = json.loads((tmp_path / "themes_run.json").read_text())["results"]
    assert run[0]["ok"] and run[0]["interests"][0]["theme_rank"] == {"fear of missing out": 1}
    rows = list(csv.DictReader(open(tmp_path / "themes.csv", encoding="utf-8-sig")))
    assert rows[0]["theme"] == "fear of missing out" and rows[0]["evidence"] == "is this all just a bubble"
    assert rows[0]["rank"] == "1" and rows[0]["supported"] == ""
    assert (tmp_path / "labels.csv").read_text() == "sentinel"                         # pick sheet untouched
    assert not (tmp_path / "results.json").exists()


# ---------------------------------------------------------------- notes, contract, alexa
from knots import alexa, contract, notes  # noqa: E402


def film(**kw):
    base = dict(title="Margin Call", content_type="movie", source_url="https://x/Margin-Call", topic="memecoins",
                topic_overlap_score=0.8, matched_keywords=["memecoins"], rating=7.1, popularity=4200, year="2011",
                abstract="A banker watches a crash unfold overnight. It is slow and talky.")
    base.update(kw)
    return ContentItem(**base)


def test_note_uses_only_catalog_facts_and_the_persons_words():
    it = film()
    theme = {"theme": "fear of missing out", "evidence": "I keep seeing memecoin threads"}
    n = notes.build_note(it, theme)
    assert n.startswith("A banker watches a crash unfold overnight.")
    assert "7.1/10 by 4,200 people" in n and "Well known" in n
    assert "“I keep seeing memecoin threads”" in n
    assert notes.check_numbers(n, notes.allowed_text(it, theme))
    assert [f["source"] for f in notes.facts(it)] == ["catalog", "TMDB"]


def test_note_never_invents_numbers_or_guesses_unknown_citations():
    paper = item("A survey", "paper", abstract="We review the field.", is_survey=True, citations=None, status="preprint")
    n = notes.build_note(paper)
    assert "citation count not available" in n and not any(ch.isdigit() for ch in n)
    assert all(f["label"] != "citations" for f in notes.facts(paper))
    cited = item("Old paper", "paper", abstract="Classic.", citations=1520)
    assert "Highly cited: 1,520 citations" in notes.build_note(cited)


def test_rephrase_hook_is_rejected_when_it_adds_a_number():
    it = film()
    base = notes.build_note(it)
    assert notes.build_note(it, rephrase=lambda t: t + " Won 12 Oscars.") == base
    assert notes.build_note(it, rephrase=lambda t: "Reworded: " + t) == "Reworded: " + base
    assert notes.build_note(it, rephrase=lambda t: 1 / 0) == base


def test_contract_has_notes_themes_and_a_spoken_line_without_urls(monkeypatch):
    monkeypatch.setattr(TH.judge, "assess", lambda text, themes: (list(themes), [], {}))
    it = interest("memecoins", implicit_themes=["fear of missing out"],
                  theme_evidence={"fear of missing out": "I keep seeing memecoin threads on my Reddit feed"})
    TH.curate([it], SAID)
    items = [film()]
    plan = build_session_plan([it], items, extractor="heuristic", query=SAID)
    r = plan["response"]
    assert r["version"] == 1 and r["status"] in ("ok", "relaxed") and r["picks"]
    p = r["picks"][0]
    assert p["title"] == "Margin Call" and p["note"] and p["confidence"] in ("strong match", "possible match", "stretch")
    assert r["themes"][0]["evidence"].startswith("I keep seeing") and "http" not in r["say"]
    assert "Margin Call" in r["say"] and r["ask"]


def test_contract_always_offers_a_pick_when_the_catalog_has_anything():
    it = interest("memecoins")
    items = [item("Barely related", "movie", "memecoins", score=0.05, abstract="Something else entirely.")]
    r = contract.build_response({"picks": {"watch": [], "read": [], "explore": []}, "themes": [], "ask": "More?"}, items, [it])
    assert r["status"] == "relaxed" and len(r["picks"]) == 1 and r["picks"][0]["confidence"] == "stretch"
    assert "looser matches" in r["say"]


def test_contract_is_honestly_empty_when_there_is_nothing_to_recommend():
    r = contract.build_response({"picks": {"watch": [], "read": [], "explore": []}, "themes": [], "ask": "What next?"},
                                [], [interest("memecoins")])
    assert r["status"] == "empty" and r["picks"] == [] and "won't guess" in r["say"] and "What next?" in r["say"]


def test_alexa_adapter_escapes_ssml_limits_length_and_keeps_the_session_open():
    resp = {"say": "Try Tom & Jerry <3. " * 800, "ask": "Which one?", "picks": [{"title": "A&B", "year": "", "note": "n"}]}
    out = alexa.to_alexa(resp)
    ssml = out["response"]["outputSpeech"]["ssml"]
    assert ssml.startswith("<speak>") and ssml.endswith("</speak>") and len(ssml) <= 8000
    assert "&amp;" in ssml and "<3" not in ssml and "&lt;3" in ssml
    assert out["response"]["shouldEndSession"] is False and out["response"]["reprompt"]
    assert out["response"]["card"]["content"] == "A&B - n"
    assert alexa.to_alexa(resp, end_session=True)["response"].get("reprompt") is None


# ---------------------------------------------------------------- profile + retrain
from knots import profile as PF  # noqa: E402
from knots import retrain as RT  # noqa: E402


def unit(i, dim=8, j=None, mix=0.0):
    v = [0.0] * dim
    v[i] = 1.0
    if j is not None:
        v[j] = mix
    return v


def test_profile_stays_bounded_merges_similar_and_decays(monkeypatch):
    p = PF.Profile()
    day = 86400.0
    PF.update(p, [("memecoins", unit(0), 1.0)], "titan", now=0)
    PF.update(p, [("crypto speculation", unit(0, j=1, mix=0.1), 0.6)], "titan", now=day)
    assert len(p.slots) == 1 and p.slots[0].n == 2                                  # merged, not duplicated
    for k in range(60):                                                              # far more topics than slots
        PF.update(p, [(f"t{k}", [1.0 if i == k else 0.0 for i in range(64)], 1.0)], "titan", now=day * (2 + k))
    before = p.slots[0].weight
    PF.update(p, [], "titan", now=day * 400)                                         # a long silence
    assert not p.slots or p.slots[0].weight < before


def test_profile_never_mixes_embedders_and_serialises():
    p = PF.update(PF.Profile(), [("a", unit(0), 1.0)], "titan", now=1)
    assert PF.prior(p, unit(0), "hashed") == 0.0                                    # other space: no influence
    PF.update(p, [("b", unit(1), 1.0)], "hashed", now=2)
    assert p.embedder == "hashed" and [s.topic for s in p.slots] == ["b"]            # reset, not blended
    q = PF.Profile.from_dict(json.loads(json.dumps(p.to_dict())))
    assert q.slots[0].topic == "b" and q.embedder == "hashed"


def test_profile_prior_is_capped_so_the_session_always_wins():
    p = PF.update(PF.Profile(), [("a", unit(0), 5.0)], "titan", now=1)
    assert 0 < PF.prior(p, unit(0), "titan") <= PF.PRIOR_CAP
    assert PF.prior(p, unit(3), "titan") == 0.0


def test_session_mode_reads_wording_and_blends_with_a_slow_average():
    assert PF.session_mode(["Alexa, can you explain how does tokenomics work? I want to understand the research"]) > 0.7
    assert PF.session_mode(["something fun, I'm bored, movie night, just want to chill"]) < 0.3
    assert PF.session_mode(["hello there"]) is None
    long_run = PF.Profile(learn_play=0.9)
    m = PF.blend_mode(0.1, long_run)                                                 # a playful night for a serious learner
    assert m["label"] in ("play", "mixed") and m["learn_vs_play"] < 0.5 and m["from_session"]
    assert PF.blend_mode(None, long_run)["label"] == "learn" and PF.blend_mode(None, None)["label"] == "mixed"
    p = PF.update(PF.Profile(learn_play=0.9), [], "x", now=1, session_mode=0.1)
    assert 0.8 < p.learn_play < 0.9                                                  # moves slowly


def lab(user, url, good="y", source="user", ts=0, sid="s1"):
    return {"user": user, "url": url, "good": good, "source": source, "ts": ts, "session_id": sid}


def test_retrain_uses_only_human_labels_dedupes_and_caps_each_person():
    pol = RT.Policy(min_labels=5, min_users=2, per_user_cap=3, min_new=1)
    rows = [lab("a", f"u{i}", ts=i) for i in range(10)] + [lab("b", "x"), lab("b", "y", source="model"),
                                                          lab("c", "z", good="")]
    rows.append(lab("b", "x", good="n", ts=5))                                       # later label replaces the earlier one
    el = RT.eligible(rows, pol)
    assert sum(r["user"] == "a" for r in el) == 3 and not any(r["source"] == "model" for r in el)
    assert [r["good"] for r in el if r["user"] == "b"] == ["n"] and not any(r["user"] == "c" for r in el)


def test_retrain_waits_for_enough_labels_from_enough_people():
    pol = RT.Policy(min_labels=6, min_users=3, per_user_cap=10, min_new=2)
    few = [lab("a", f"u{i}") for i in range(8)]
    ok, why = RT.should_retrain(few, {}, pol)
    assert not ok and "1 people, need 3" in why
    many = few + [lab("b", "p"), lab("c", "q")]
    assert RT.should_retrain(many, {}, pol)[0]
    assert not RT.should_retrain(many, {"trained_on": 9}, pol)[0]                    # not enough NEW labels


def test_gate_demands_a_real_gain_without_regression_and_registry_can_roll_back(tmp_path):
    champ = {"p@3": 0.70, "ndcg@5": 0.80}
    assert RT.gate(champ, {"p@3": 0.73, "ndcg@5": 0.80})["promote"]
    assert not RT.gate(champ, {"p@3": 0.71, "ndcg@5": 0.85})["promote"]              # too small a gain
    bad = RT.gate(champ, {"p@3": 0.80, "ndcg@5": 0.70})
    assert not bad["promote"] and "ndcg@5" in bad["reason"]                          # regressed elsewhere
    reg = RT.Registry(tmp_path)
    v1 = reg.register({"w": 1}, champ, 200); reg.promote(v1, 200)
    v2 = reg.register({"w": 2}, {"p@3": 0.74, "ndcg@5": 0.8}, 400); reg.promote(v2, 400)
    assert reg.champion()["artifact"] == {"w": 2}
    assert reg.rollback() == v1 and reg.champion()["artifact"] == {"w": 1}
    assert reg.rollback() is None


def test_golden_contract_and_alexa_responses_have_not_drifted():
    sys.path.insert(0, str(ROOT / "tests"))
    import make_golden
    r, a = make_golden.build()
    assert r == json.loads((ROOT / "tests" / "golden" / "response_ok.json").read_text())
    assert a == json.loads((ROOT / "tests" / "golden" / "alexa_ok.json").read_text())
    assert "“" not in r["say"] and len(r["say"]) < 400                           # short and quote-free for the ear
