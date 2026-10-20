"""Regenerates tests/golden/*.json from the real code: python tests/make_golden.py (review the diff!)."""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from knots import alexa, extraction as I  # noqa: E402
from knots.brain import themes as TH  # noqa: E402
from knots.catalogs import ContentItem  # noqa: E402
from knots.plan import build_session_plan  # noqa: E402

SAID = "I keep seeing memecoin threads on my Reddit feed, is this all just a bubble? Tell me more about the research."


def build():
    saved = (TH.judge.assess, {k: os.environ.get(k) for k in ("EMBEDDER", "KNOTS_OFFLINE")})
    os.environ["EMBEDDER"], os.environ["KNOTS_OFFLINE"] = "hashed", "1"
    TH.judge.assess = lambda t, ths: (list(ths), [], {})
    try:
        return _build()
    finally:                                   # leave no trace for other tests
        TH.judge.assess = saved[0]
        for k, v in saved[1].items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def _build():
    it = I.Interest(topic="memecoins", depth="deep", sentiment="curious", mentions=1, concepts=["memecoins"],
                    keywords=["memecoins"], implicit_themes=["fear of missing out"],
                    theme_evidence={"fear of missing out": "I keep seeing memecoin threads on my Reddit feed"})
    TH.curate([it], SAID)
    items = [ContentItem(title="Margin Call", content_type="movie", source_url="https://www.themoviedb.org/movie/59436",
                         topic="memecoins", topic_overlap_score=0.8, matched_keywords=["memecoins"], rating=7.1,
                         popularity=4200, year="2011",
                         abstract="A banker watches a crash unfold overnight. It is slow and talky.")]
    r = build_session_plan([it], items, extractor="bedrock", query=SAID)["response"]
    return r, alexa.to_alexa(r)


if __name__ == "__main__":
    r, a = build()
    (ROOT / "tests" / "golden").mkdir(exist_ok=True)
    (ROOT / "tests" / "golden" / "response_ok.json").write_text(json.dumps(r, indent=2) + "\n")
    (ROOT / "tests" / "golden" / "alexa_ok.json").write_text(json.dumps(a, indent=2) + "\n")
    print(r["say"])
