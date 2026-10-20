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
