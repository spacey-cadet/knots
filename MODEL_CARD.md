# Knots model card

Generated 2026-10-20 from `eval/results/metrics_*.json` by `eval/make_model_card.py`. Numbers below are copied from those files, not written by hand.

## What it is

Knots reads what someone has been researching and recommends things to watch (films, series, documentaries), explainers (YouTube) and research (arXiv), including watch-and-learn pairs. Three small learned parts sit on top of retrieval from TMDB, YouTube and arXiv: a **theme judge** (which unstated themes to trust), a **fit ranker** (what fits this session), and a **pair scorer**.

## Embedder

The fit ranker and the pair scorer compare texts as vectors, so their numbers depend on which embedding model produced them, and so do the pair floors and similarity thresholds (set per embedder, still untuned). Each part below says what it was measured with.

- Theme judge: none. It scores theme phrases with character n-gram TF-IDF and a feature critic; no embedding model is involved, so its numbers do not depend on the embedder.
- Fit ranker: not recorded (measured before the embedder was tracked): do not assume it was the semantic embedder; re-run `python eval/run_eval.py run` to record it
- Pair scorer: not recorded (measured before the embedder was tracked): do not assume it was the semantic embedder; re-run `python eval/run_eval.py run` to record it

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

## Ranking quality (what is read out first)

Each list is the picks shown for one session. Only shown picks are labelled, so this measures the order and the precision of the shown set, not whether a better item was never retrieved. P@k is precision in the first k picks; MAP and NDCG ignore lists with nothing good in them (the count is given). Intervals resample sessions. Embedder: **cohere** (`cohere.embed-v4:0`, 256 dimensions), requested `auto`.

**all extractors**: 40 complete lists from 22 train sessions; 54% of everything shown was good; 6 list(s) had no good pick.

| Order | P@1 | P@3 | MAP | NDCG@5 |
| --- | --- | --- | --- | --- |
| as read aloud | 0.60 [0.46, 0.73] | 0.58 [0.49, 0.68] | 0.79 [0.71, 0.86] | 0.84 [0.77, 0.90] |
| fit score | 0.70 [0.57, 0.83] | 0.59 [0.51, 0.68] | 0.85 [0.80, 0.91] | 0.89 [0.84, 0.93] |
| tag overlap | 0.47 [0.33, 0.64] | 0.53 [0.46, 0.62] | 0.72 [0.65, 0.81] | 0.78 [0.72, 0.85] |
| session fit | 0.78 [0.67, 0.88] | 0.62 [0.53, 0.70] | 0.91 [0.86, 0.94] | 0.92 [0.89, 0.95] |
| rating | 0.57 [0.45, 0.71] | 0.57 [0.47, 0.67] | 0.78 [0.69, 0.87] | 0.83 [0.75, 0.90] |
| popularity | 0.72 [0.61, 0.85] | 0.60 [0.52, 0.68] | 0.88 [0.84, 0.92] | 0.91 [0.87, 0.95] |
| random (chance) | 0.54 [0.46, 0.62] | 0.55 [0.47, 0.63] | 0.76 [0.71, 0.81] | 0.81 [0.76, 0.86] |

8 list(s) were left out because some of their picks are still unlabelled.

Holdout: not yet measured (holdout sessions are unlabelled).

**bedrock**: 20 complete lists from 20 train sessions; 71% of everything shown was good; 0 list(s) had no good pick.

| Order | P@1 | P@3 | MAP | NDCG@5 |
| --- | --- | --- | --- | --- |
| as read aloud | 0.75 [0.60, 0.90] | 0.74 [0.66, 0.82] | 0.83 [0.76, 0.90] | 0.86 [0.80, 0.92] |
| fit score | 0.80 [0.65, 0.95] | 0.72 [0.63, 0.82] | 0.83 [0.77, 0.90] | 0.87 [0.81, 0.92] |
| tag overlap | 0.60 [0.40, 0.80] | 0.72 [0.66, 0.79] | 0.79 [0.71, 0.86] | 0.83 [0.77, 0.89] |
| session fit | 0.90 [0.80, 1.00] | 0.78 [0.68, 0.87] | 0.90 [0.83, 0.95] | 0.92 [0.87, 0.96] |
| rating | 0.75 [0.60, 0.90] | 0.69 [0.60, 0.78] | 0.82 [0.75, 0.90] | 0.85 [0.78, 0.92] |
| popularity | 0.80 [0.65, 0.95] | 0.76 [0.68, 0.83] | 0.88 [0.80, 0.94] | 0.89 [0.83, 0.95] |
| random (chance) | 0.70 [0.62, 0.78] | 0.71 [0.63, 0.78] | 0.81 [0.75, 0.86] | 0.85 [0.81, 0.89] |

4 list(s) were left out because some of their picks are still unlabelled.

Holdout: not yet measured (holdout sessions are unlabelled).

**heuristic**: 20 complete lists from 20 train sessions; 38% of everything shown was good; 6 list(s) had no good pick.

| Order | P@1 | P@3 | MAP | NDCG@5 |
| --- | --- | --- | --- | --- |
| as read aloud | 0.45 [0.25, 0.65] | 0.42 [0.28, 0.57] | 0.73 [0.59, 0.86] | 0.81 [0.70, 0.91] |
| fit score | 0.60 [0.40, 0.80] | 0.46 [0.33, 0.59] | 0.89 [0.81, 0.96] | 0.91 [0.85, 0.97] |
| tag overlap | 0.35 [0.20, 0.55] | 0.34 [0.22, 0.48] | 0.63 [0.50, 0.77] | 0.71 [0.59, 0.82] |
| session fit | 0.65 [0.45, 0.80] | 0.46 [0.33, 0.59] | 0.92 [0.85, 0.98] | 0.92 [0.86, 0.98] |
| rating | 0.40 [0.25, 0.60] | 0.44 [0.28, 0.59] | 0.71 [0.58, 0.86] | 0.79 [0.69, 0.90] |
| popularity | 0.65 [0.45, 0.80] | 0.44 [0.32, 0.57] | 0.89 [0.82, 0.94] | 0.94 [0.89, 0.97] |
| random (chance) | 0.37 [0.25, 0.50] | 0.38 [0.26, 0.51] | 0.70 [0.61, 0.78] | 0.75 [0.68, 0.83] |

4 list(s) were left out because some of their picks are still unlabelled.

Holdout: not yet measured (holdout sessions are unlabelled).

Rows other than "as read aloud" re-order the same picks, so they show what a different ordering rule would have scored, not what the live system returned.


## Watch-and-learn pairs

Not yet measured: no pair has been labelled. The pair score and its thresholds are untuned design choices.

## Limits

- Few sessions, one labeller, one extraction model: results will not transfer unchanged to other people.
- Confidence numbers appear only where a calibration was fitted on held-out data.
- Citation counts come from OpenAlex and are omitted when a title cannot be matched; an arXiv paper is labelled a preprint unless arXiv lists a journal reference.
- Popularity is not a ranking signal; a quality floor and a reserved blockbuster slot are the only uses of ratings and votes.
- Offline mock catalogs are never shown as real results.
