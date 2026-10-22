# Metrics beyond accuracy, and the online metrics behind retraining

Code: `knots/beyond.py` (the definitions), `eval/eval_beyond.py` (offline report), `knots/online.py` (online
metrics and retrain triggers), `knots/retrain.py` (the decision and the gate). The server is not wired to the
online part yet: see "Not wired".

## 1. Offline: beyond accuracy

Run `python eval/eval_beyond.py` after `eval_rank.py`. It uses the same lists and orderings and the same
session-resampled 90% intervals, at the headline cut (top 3), so every row can be read next to P@3. It also writes
`eval/results/metrics_beyond.json`, which `make_model_card.py` turns into a section of the model card.

Higher means "more of the named thing", not "better". They are there to show a trade-off with accuracy, and a
ranker that wins on P@3 by showing only famous, obvious picks should be visible as exactly that.

| Metric | Definition | Needs labels |
| --- | --- | --- |
| Diversity (`ild`) | mean pairwise cosine distance between the picks' vectors | no |
| `format_spread` | entropy of the picks' formats over log(list length); 1 = every pick a different format | no |
| `topic_spread` | distinct matched topics / picks | no |
| Novelty | 1 - popularity percentile within the pick's source (TMDB votes, video views, paper citations) | no |
| Unexpectedness | 1 - mean percentile of keyword overlap and session similarity: how far the pick sits from what the person literally said | no |
| `interest_coverage` | share of the session's own interests that at least one shown pick answers | no |
| Serendipity | mean over picks of `good * unexpectedness`; also `serendipitous_share`, the share of good picks with unexpectedness >= 0.5 | yes |
| **Grounding@k** (headline) | mean over the top-k picks of `good * supported * min(E, 3) / 3`, E = independent evidence types: `keyword` (a TMDB keyword matched; films, series, docs), `theme_quote` (the interest has a judge-kept theme resting on a verbatim quote), `overview_sim` (overview similarity to the session >= 0.30, untuned; `KNOTS_GROUNDING_SIM_MIN`), `co_mention` (reserved, never fires). Distance earns nothing | yes |
| **Creative & grounded** (headline) | STRICT: share of picks that are good, supported, backed by >= 2 evidence types AND far from the person's words (unexpectedness >= 0.5); the eval prints the count (k of n picks). SOFT: mean of `good * supported * min(E,3)/3 * unexpectedness`. A bad, overreaching or unevidenced pick scores 0 however far it is, so reaching further cannot raise either form | yes |
| **Overreach rate** (gate) | share of shown picks labelled not supported. Proxy until picks carry a `supported` column: labelled bad with reason `relevance`. Lower is better; `retrain.gate` refuses a challenger whose overreach rose by more than 0.02 | yes |
| **Theme coverage** (headline) | share of a session's linkable supported themes served by at least one good, supported pick | yes |
| Coverage | across all lists: distinct items, `unique_share` (distinct / slots), `gini` of how often items repeat, `top_item_share` | no |

### How to read them honestly

- **Percentiles, not absolutes.** Novelty and unexpectedness are percentiles against every pick shown in the run
  (the pool), so 0.5 means "typical shown pick" and the numbers move if the pool does. Popularity is only compared
  inside one source, because votes, views and citations are not on one scale.
- **Unknown stays unknown.** A paper with no citation count has no novelty; a list of one pick has no diversity; a
  source with fewer than 5 reference picks has no percentile. These are left out of the mean, never counted as 0.
- **Distance is never a goal.** An earlier "creativity" metric (good * sqrt(novelty * unexpectedness)) paid for
  being far from the person's words, which rewards far-fetched picks, so it was removed. Novelty, unexpectedness and
  serendipity stay as diagnostics that describe what was shown. Overreach is the gate that stops reaching further
  being mistaken for improvement.
- **Grounding is limited by what is stored.** Under the proxy every good pick counts as supported, so only the
  overreach rate uses the proxy. Theme quotes link to picks for only some of them in older results (73 of 270 picks);
  the report says how many it could link, and re-running `run_eval.py run` stores the evidence types and quotes.
  The 0.30 similarity threshold is untuned; the report shows Grounding at 0.25, 0.30 and 0.35.
- **Diversity needs a real embedder.** With the offline hashed embedder it is a word-overlap measure and sits near
  0.9 whatever is shown. The report and model card say which embedder was used and warn when it was hashed.
- **Coverage has no catalog denominator.** TMDB, YouTube and arXiv cannot be enumerated here, so coverage is how
  repeated the shown items are, not what fraction of a catalog was reached. It is a point value with no interval:
  resampling sessions with replacement repeats sessions, which manufactures the repetition coverage measures.
  Repetition also grows with the number of slots, so compare rows from the same lists, not across scopes.
- **Same limits as the accuracy numbers.** Only shown picks are labelled, Grounding, overreach and serendipity use only
  fully labelled lists, and there are few sessions and one labeller.

## 2. Online metrics

Computed from an append-only event log (`EventLog`, JSON lines; no conversation text is logged). Events:
`impression` (a list was shown, with each pick's position and an `exploration` flag), `feedback` (explicit good /
not good), `engage` (implicit positive) and `reformulate` (implicit negative).

`window_metrics` reports, for a window of events:

- **Positive rate** of explicit answers with a 90% Wilson interval, the same among positions 1-3, and the label
  yield (answers per pick shown).
- **Exploration gap**: positive rate of ranked picks against exploration picks. If a pick shown regardless of rank
  is liked as often as a top-ranked one, the ranking is adding little. It reads `None` until an exploration slot
  exists (designed, not built).
- **Engagement rate** and **reformulation rate** (share of sessions with each), **repeat rate** (share of picks a
  person had already been shown), **empty** and **relaxed** rates from the response contract's `status`.
- **Beyond-accuracy on what was actually shown**: format and topic spread, coverage, and novelty and
  unexpectedness against a frozen reference pool.

Feedback exists only for picks that were shown, so it is biased toward what the model already favours; the position
and exploration flags are recorded so that can be measured and corrected later.

**Percentile metrics online need a frozen pool.** Against the window's own picks, novelty averages 0.5 by
construction whatever is shown. `make_baseline` freezes the reference pool when a model is promoted, and
without one novelty and unexpectedness are reported as `None`.

## 3. What triggers a retrain

Before this, the only trigger was label volume (`retrain.should_retrain`). Now:

1. **Volume** (unchanged): at least `min_labels` usable labels from at least `min_users` people, and at least
   `min_new` new ones since the last training. Defaults 200 / 20 / 100.
2. **Online alerts** compare the latest window (default 14 days) with the baseline frozen at promotion
   (`Registry.promote(..., baseline=make_baseline(events))`; rollback restores the previous champion's baseline).

| Alert | Fires when | Can a retrain fix it? |
| --- | --- | --- |
| `quality_drop` | at least 40 answers, and even the optimistic end of the current positive rate is 0.10 below the baseline | yes |
| `score_drift` | population stability index of the served fit scores against the baseline is at least 0.25 (100+ picks each side) | yes |
| `retrieval` | empty + relaxed lists rose by 0.15 or more (30+ lists) | no: check the catalogs |
| `filter_bubble` | novelty or unexpectedness fell by 0.15 or more, or repeat rate rose by 0.15 or more | no: consider the exploration slot |

A retrain-relevant alert lowers **only** the new-label requirement (`min_new` 100 to `alert_min_new` 50).
It never lowers the total labels or the number of distinct people, so one enthusiast cannot start a retrain.
`retrain_decision(labels, events, state)` returns the decision, the reason, the alerts and the window metrics.

Two rules do not change:

- **Online signals never train.** Only explicit listener feedback becomes a label (`labels_from_events`);
  engagement and reformulation can raise an alert but are never labels.
- **Alerts decide when to try, never whether it ships.** The challenger still has to beat the champion on the
  frozen expert holdout by at least +0.02 P@3 with NDCG@5 no worse than -0.01.

All thresholds are untuned defaults and placeholders until there is real listener traffic to look at. The window,
the minimum number of answers and the eased new-label requirement are env-tunable
(`KNOTS_MONITOR_WINDOW_DAYS`, `KNOTS_MONITOR_MIN_FEEDBACK`, `KNOTS_RETRAIN_ALERT_MIN_NEW`); the rest are fields of
`MonitorPolicy` in `knots/online.py`.

## Not wired

`impression` and `feedback` events are now written by the server (`knots/feedback.py`: `knots_partner`, the
`rate_pick` tool and the Alexa skill), under a salted hash of a verified identity, and `forget_me` removes them.
`python eval/monitor.py` reads the log. Not written yet: `engage` (asked for more, opened, saved) and
`reformulate` events, so engagement and reformulation rates read 0 until something records them. Nothing starts a
retrain automatically; the monitor reports whether one is due and a person runs it.
