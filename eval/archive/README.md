# Archive

`themes_v1_labelled.csv` - 101 theme labels (supported = y/n) written against the FIRST theme extractor
(no quotes, up to 3 themes per interest). Frozen: nothing reads or rewrites it.

Why it is kept apart: labels are keyed by (session, extractor, topic, theme text). The redesigned extractor
(<= 5 themes per conversation, each resting on a quote from the person) words its themes differently, so these
rows would never match a new run and would silently orphan. They remain useful as evidence of which kinds of theme
this author accepts or rejects; `eval/train_brain.py` learns from `knots/brain/data/theme_examples.jsonl`.

The `error_type` and `note` columns are diagnostics, not ground truth. Only `supported` is.
