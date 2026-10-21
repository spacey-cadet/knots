# Profile and retraining

Code: `knots/profile.py`, `knots/retrain.py` (both tested, neither wired into the server yet: see "Not wired").

## Profile (history without bloat)

- At most 20 slots (`KNOTS_PROFILE_SLOTS`): topic, ONE centroid (moving average), decayed weight, last seen.
  New evidence merges into the nearest slot (cosine >= 0.80), else opens a slot; weights halve every 45 days;
  slots under 0.05 are pruned; the lightest is evicted when full. Raw text is never stored.
- It is a prior, capped at +0.15 on a 0..1 score, so the session's own words always outweigh history.
- Vectors from different embedders are never mixed: a changed embedder starts the profile clean.

## Learn vs play (the "temp")

`learn_vs_play` in 0..1 comes from the session's wording (cue lists in `profile.py`), blended 75/25 with a slow
average of past sessions (which moves 10% per session). No cues reads as neutral, flagged `from_session: false`.
It is a keyword read, crude on purpose: it is a dial for ordering watch / read / explore, not a classifier.

## Retraining

Trains only on human labels (`source` = `user` or `expert`). Model output, notes and `error_type` never train.
Trigger (defaults, all env-tunable): >= 200 usable labels from >= 20 people, >= 100 new since the last training,
each person capped at 25 (newest kept), latest label per (person, session, url) wins.
An online alert (a quality drop or a score-distribution shift against the baseline frozen at promotion) lowers only
the new-label requirement to 50; it never lowers the totals or the number of people, and implicit signals are never
labels. See `docs/METRICS.md`.
A challenger must beat the champion on a FROZEN expert holdout by >= +0.02 P@3 with NDCG@5 no worse than -0.01.
Versions are kept in a `Registry`, each with the online baseline it was promoted with; `rollback()` restores the
previous champion and its baseline.

## Decided
Human-only labels; per-person cap; frozen expert holdout; margin gate; versioned rollback; profile is a capped prior.

## Open (needs you)
- N and M above are placeholders until you know how many listeners you will have.
- Where labels from listeners are collected (Alexa follow-up "was that a good pick?") and stored.
- An exploration slot (one pick shown regardless of rank) so listener feedback is not only on what the model
  already favours; designed, not built.
- Whether s24-s30 stay the only holdout once listener labels exist.

## Not wired
No server code calls `profile.update`, `prior`, `blend_mode`, `retrain` or the `online` event log yet: they need a verified identity
(derive it from the OAuth token on account linking; never trust a caller-supplied `user_id` on a public endpoint)
and a store. The contract already accepts `mode`.
