# Knots model card

Generated 2026-10-16 from `eval/results/metrics_*.json` by `eval/make_model_card.py`. Numbers below are copied from those files, not written by hand.

## What it is

Knots reads what someone has been researching and recommends things to watch (films, series, documentaries), explainers (YouTube) and research (arXiv), including watch-and-learn pairs. Three small learned parts sit on top of retrieval from TMDB, YouTube and arXiv: a **theme judge** (which unstated themes to trust), a **fit ranker** (what fits this session), and a **pair scorer**.

## How the labels were made

One person labelled every row (see `eval/`). Sessions marked `holdout` in `eval/sessions.json` are never trained or tuned on; they exist to test. Labels reflect one person's taste. **TODO (author):** state truthfully how the labels and notes were made, including whether an LLM drafted any of the notes, in `eval/LABEL_PROVENANCE.txt`.

## Theme judge

Trained on 101 labelled themes from 10 sessions. Leave-one-session-out (each session scored by a model that never saw it):

| Model | AUC [90% interval] | Within-session AUC |
| --- | --- | --- |
| Feature critic (feature lists were written after seeing the labels: optimistic) | 0.81 [0.72, 0.89] | 0.818 |
| Nearest labelled themes | 0.78 [0.70, 0.87] | 0.863 |
| Average (shipped) | 0.84 [0.77, 0.91] | 0.893 |

Keeping themes at calibrated p >= 0.5: keeps 36 of 101, 75% supported (base rate 44%).

Calibration (expected error 0.052; fitted on the same held-out scores, so slightly optimistic):

| Said | n | Predicted | Observed |
| --- | --- | --- | --- |
| 0.0-0.2 | 26 | 0.10 | 0.12 |
| 0.2-0.4 | 26 | 0.31 | 0.31 |
| 0.4-0.6 | 20 | 0.49 | 0.40 |
| 0.6-0.8 | 13 | 0.69 | 0.85 |
| 0.8-1.0 | 16 | 0.93 | 0.88 |

**Holdout: not yet measured.** Too few labelled themes in holdout sessions. Until then, treat the numbers above as optimistic.

## Fit ranker

Not yet measured: 107 labelled live picks, but only 0 carry stored fit features (107 do not). Re-run `python eval/run_eval.py run`, then label the new picks. Until trained, ranking uses documented hand-set prior weights and cards show **no** confidence number.

## Watch-and-learn pairs

Not yet measured: no pair has been labelled. The pair score and its thresholds are untuned design choices.

## Limits

- Few sessions, one labeller, one extraction model: results will not transfer unchanged to other people.
- Confidence numbers appear only where a calibration was fitted on held-out data.
- Citation counts come from OpenAlex and are omitted when a title cannot be matched; an arXiv paper is labelled a preprint unless arXiv lists a journal reference.
- Popularity is not a ranking signal; a quality floor and a reserved blockbuster slot are the only uses of ratings and votes.
- Offline mock catalogs are never shown as real results.
