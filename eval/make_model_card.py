"""Render MODEL_CARD.md from the metrics files in eval/results/. Every number comes from a file,
never from this script; anything not yet measured says so and why.

    python eval/train_brain.py && python eval/train_rank.py && python eval/eval_pairs.py
    python eval/make_model_card.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "eval" / "results"


def load(name: str) -> dict | None:
    try:
        return json.loads((RES / name).read_text())
    except (OSError, ValueError):
        return None


def ci(v: dict) -> str:
    return f"{v['auc']:.2f} [{v['auc_ci90'][0]:.2f}, {v['auc_ci90'][1]:.2f}]"


def rel_table(rel: list[dict]) -> list[str]:
    return ["| Said | n | Predicted | Observed |", "| --- | --- | --- | --- |"] + [
        f"| {r['bin']} | {r['n']} | {r['predicted']:.2f} | {r['observed']:.2f} |" for r in rel]


def provenance() -> str:
    """Written by the author in eval/LABEL_PROVENANCE.txt (so regenerating never overwrites it)."""
    try:
        text = (ROOT / "eval" / "LABEL_PROVENANCE.txt").read_text().strip()
        if text:
            return text
    except OSError:
        pass
    return ("**TODO (author):** state truthfully how the labels and notes were made, including whether an LLM "
            "drafted any of the notes, in `eval/LABEL_PROVENANCE.txt`.")


def main() -> int:
    brain, rank, pairs = load("metrics_brain.json"), load("metrics_rank.json"), load("metrics_pairs.json")
    L = ["# Knots model card", "",
         f"Generated {date.today().isoformat()} from `eval/results/metrics_*.json` by `eval/make_model_card.py`. "
         "Numbers below are copied from those files, not written by hand.", "",
         "## What it is", "",
         "Knots reads what someone has been researching and recommends things to watch (films, series, "
         "documentaries), explainers (YouTube) and research (arXiv), including watch-and-learn pairs. "
         "Three small learned parts sit on top of retrieval from TMDB, YouTube and arXiv: a **theme judge** "
         "(which unstated themes to trust), a **fit ranker** (what fits this session), and a **pair scorer**.", "",
         "## How the labels were made", "",
         "One person labelled every row (see `eval/`). Sessions marked `holdout` in `eval/sessions.json` are "
         "never trained or tuned on; they exist to test. Labels reflect one person's taste. "
         + provenance(), "",
         "## Theme judge", ""]
    if brain:
        e = brain["loso"]["ensemble"]
        L += [f"Trained on {brain['train_labels']} labelled themes from {brain['train_sessions']} sessions. "
              "Leave-one-session-out (each session scored by a model that never saw it):", "",
              "| Model | AUC [90% interval] | Within-session AUC |", "| --- | --- | --- |"]
        for k, name in (("critic", "Feature critic (feature lists were written after seeing the labels: optimistic)"),
                        ("neighbors", "Nearest labelled themes"), ("ensemble", "Average (shipped)")):
            v = brain["loso"][k]
            L.append(f"| {name} | {ci(v)} | {v['within_session_auc']} |")
        t = brain["loso"]["at_threshold"]
        L += ["", f"Keeping themes at calibrated p >= {t['threshold']}: keeps {t['kept']} of {t['of']}, "
              f"{t['precision']:.0%} supported (base rate {t['base_rate']:.0%}).", "",
              f"Calibration (expected error {brain['calibration']['ece']:.3f}; fitted on the same held-out scores, "
              "so slightly optimistic):", ""] + rel_table(brain["calibration"]["reliability"]) + [""]
        h = brain.get("holdout")
        L.append(("**Holdout** (the clean test): " + f"{h['n']} themes, AUC {ci(h)}, "
                  f"precision at threshold {h['precision_at_threshold']} (base rate {h['base_rate']}).") if h else
                 "**Holdout: not yet measured.** Too few labelled themes in holdout sessions. Until then, "
                 "treat the numbers above as optimistic.")
    else:
        L.append("Not yet measured: run `python eval/train_brain.py`.")
    L += ["", "## Fit ranker", ""]
    if rank and rank.get("status") == "trained":
        L += [f"{rank['train_picks']} labelled picks, {rank['train_sessions']} sessions. Good rate by type: "
              f"{rank['good_rate_by_type']}.", "", "| Ranker | AUC [90% interval] | Precision@3 |", "| --- | --- | --- |"]
        for k, v in rank["loso"].items():
            L.append(f"| {k} | {ci(v)} | {v['precision_at_3']:.2f} |")
        L += ["", f"Weights: `{rank['weights']}`", ""] + rel_table(rank["calibration"]["reliability"])
        L.append(f"\nHoldout: {rank['holdout']}" if rank.get("holdout") else "\nHoldout: not yet measured.")
    else:
        L.append("Not yet measured: " + (rank or {}).get("reason", "run `python eval/train_rank.py`.") +
                 " Until trained, ranking uses documented hand-set prior weights and cards show **no** confidence number.")
    L += ["", "## Watch-and-learn pairs", ""]
    if pairs and pairs.get("all"):
        for k in ("train", "holdout"):
            v = pairs.get(k)
            L.append(f"- {k}: " + (f"{v['pairs']} labelled pairs, success {v['success_rate']:.0%} {v['success_ci90']}, "
                                   f"pair_score AUC {v['score_auc']}" if v else "none labelled"))
    else:
        L.append("Not yet measured: no pair has been labelled. The pair score and its thresholds are untuned design choices.")
    L += ["", "## Limits", "",
          "- Few sessions, one labeller, one extraction model: results will not transfer unchanged to other people.",
          "- Confidence numbers appear only where a calibration was fitted on held-out data.",
          "- Citation counts come from OpenAlex and are omitted when a title cannot be matched; an arXiv paper is "
          "labelled a preprint unless arXiv lists a journal reference.",
          "- Popularity is not a ranking signal; a quality floor and a reserved blockbuster slot are the only uses of ratings and votes.",
          "- Offline mock catalogs are never shown as real results.", ""]
    (ROOT / "MODEL_CARD.md").write_text("\n".join(L))
    print("wrote MODEL_CARD.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
