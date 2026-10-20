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


def embedder_line(info: dict | None) -> str:
    if not info or not info.get("recorded"):
        return ("not recorded (measured before the embedder was tracked): do not assume it was the semantic "
                "embedder; re-run `python eval/run_eval.py run` to record it")
    if info.get("provider") == "none":
        return info.get("note", "no embedder")
    p = info.get("provider")
    model = f" (`{info['model_id']}`, {info.get('dimensions')} dimensions)" if info.get("model_id") else ""
    if p == "hashed":
        return ("**hashed** (offline, lexical only). These numbers do not describe a semantic embedder such as "
                "Cohere or Titan; re-measure with one")
    if p == "mixed":
        return "**mixed**: " + ", ".join(f"{k} in {v} run(s)" for k, v in info["runs_by_embedder"].items()) + ", so not comparable"
    return f"**{p}**{model}, requested `{info.get('requested')}`"


def embedder_section(brain, rank, pairs) -> list[str]:
    L = ["## Embedder", "",
         "The fit ranker and the pair scorer compare texts as vectors, so their numbers depend on which embedding model "
         "produced them, and so do the pair floors and similarity thresholds (set per embedder, still untuned). "
         "Each part below says what it was measured with.", "",
         "- Theme judge: none. It scores theme phrases with character n-gram TF-IDF and a feature critic; "
         "no embedding model is involved, so its numbers do not depend on the embedder.",
         f"- Fit ranker: {embedder_line((rank or {}).get('embedder'))}",
         f"- Pair scorer: {embedder_line((pairs or {}).get('embedder'))}", ""]
    infos = [(n, (m or {}).get("embedder")) for n, m in (("fit ranker", rank), ("pair scorer", pairs))]
    infos = [(n, i) for n, i in infos if i and i.get("recorded")]
    if len({i["provider"] for _, i in infos}) > 1:
        L += ["**Warning:** the ranker and the pair scorer were measured with different embedders.", ""]
    for n, i in infos:
        if i.get("fell_back_runs"):
            L += [f"**Warning:** {i['fell_back_runs']} eval run(s) behind the {n} fell back to the hashed embedder "
                  "(throttling or an error), so those results mix embedders. Re-run before trusting them.", ""]
    return L


def ranking_section(ranking: dict | None) -> list[str]:
    L = ["## Ranking quality (what is read out first)", ""]
    if not ranking:
        return L + ["Not yet measured: run `python eval/eval_rank.py`.", ""]
    head = ranking.get("headline") or ["p@1", "p@3", "map", "ndcg@5"]
    names = {"p@1": "P@1", "p@3": "P@3", "map": "MAP", "ndcg@5": "NDCG@5"}
    L += ["Each list is the picks shown for one session. Only shown picks are labelled, so this measures the order "
          "and the precision of the shown set, not whether a better item was never retrieved. P@k is precision in the "
          "first k picks; MAP and NDCG ignore lists with nothing good in them (the count is given). Intervals "
          "resample sessions. Embedder: " + embedder_line(ranking.get("embedder")) + ".", ""]
    for scope in [k for k in ranking if isinstance(ranking[k], dict) and "train" in ranking[k]]:
        sc = ranking[scope]
        res = sc.get("train")
        if not res:
            continue
        base = res["as read aloud"]
        L += [f"**{scope}**: {sc['lists']} complete lists from {sc['sessions']} train sessions; "
              f"{base['set_precision']['mean']:.0%} of everything shown was good; "
              f"{base['lists_without_a_good_pick']} list(s) had no good pick.", "",
              "| Order | " + " | ".join(names.get(h, h) for h in head) + " |", "| --- " * (len(head) + 1) + "|"]
        for order, v in res.items():
            L.append(f"| {order} | " + " | ".join(
                f"{v[h]['mean']:.2f} [{v[h]['ci90'][0]:.2f}, {v[h]['ci90'][1]:.2f}]" for h in head) + " |")
        ex = sc.get("excluded", {})
        if ex.get("partly_unlabelled"):
            L += ["", f"{len(ex['partly_unlabelled'])} list(s) were left out because some of their picks are still unlabelled."]
        hold = sc.get("holdout")
        L += ["", "Holdout: " + ("see `metrics_ranking.json`." if hold else "not yet measured (holdout sessions are unlabelled)."), ""]
    return L + ["Rows other than \"as read aloud\" re-order the same picks, so they show what a different ordering rule "
                "would have scored, not what the live system returned.", ""]


def main() -> int:
    brain, rank, pairs = load("metrics_brain.json"), load("metrics_rank.json"), load("metrics_pairs.json")
    ranking = load("metrics_ranking.json")
    L = ["# Knots model card", "",
         f"Generated {date.today().isoformat()} from `eval/results/metrics_*.json` by `eval/make_model_card.py`. "
         "Numbers below are copied from those files, not written by hand.", "",
         "## What it is", "",
         "Knots reads what someone has been researching and recommends things to watch (films, series, "
         "documentaries), explainers (YouTube) and research (arXiv), including watch-and-learn pairs. "
         "Three small learned parts sit on top of retrieval from TMDB, YouTube and arXiv: a **theme judge** "
         "(which unstated themes to trust), a **fit ranker** (what fits this session), and a **pair scorer**.", "",
         ] + embedder_section(brain, rank, pairs) + [
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
    L += [""] + ranking_section(ranking)
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
