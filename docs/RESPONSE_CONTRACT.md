# Response contract

One shape, built by `knots/contract.py` and attached to every session plan as `plan["response"]`. Alexa, the MCP
tool and any screen render from it; nothing else is promised to them. Golden examples (regenerated from the real
code, and tested for drift): `tests/golden/response_ok.json`, `tests/golden/alexa_ok.json`
(`python tests/make_golden.py` rewrites them; review the diff).

| Field | Meaning |
| --- | --- |
| `status` | `ok`; `relaxed` (nothing passed the normal checks, so the closest items are offered and each says `stretch`/`possible match`); `empty` (the catalogs returned nothing, so there is nothing true to recommend) |
| `say` | What a voice reads. At most 2 picks, no URLs, no quotes, under ~400 characters in the golden case |
| `picks[]` | `title, type, year, url, matched_topic, note, spoken_note, why, facts[], confidence` |
| `picks[].note` | Description + popularity + the link to the person's own words. Built from catalog fields only (`knots/notes.py`) |
| `picks[].facts[]` | `{label, value, source}` e.g. `7.1/10 from 4,200 ratings (TMDB)`; unknown is omitted, never guessed |
| `themes[]` | At most 5, ranked; each carries the exact words it rests on |
| `ask` | One follow-up question |
| `mode` | Optional `{learn_vs_play, label, from_session}` once a profile supplies it |

## Rules

- **At least one pick whenever the catalogs return anything.** Ladder: normal ranking (needs receipts) -> closest
  by relevance (`relaxed`, `possible match`) -> very weak match (`stretch`). With no catalog items, `status` is
  `empty`, `picks` is `[]`, and `say` says it will not guess. This is the honest limit of "always one".
- **Notes are faithful.** Every number in a note must appear in the item's data (`check_numbers`). An optional
  `rephrase` hook may reword a note; a rewrite that adds or changes a number is discarded.
- **Voice**: Alexa's `outputSpeech` (8000 chars) is checked AFTER XML escaping; the card (8000) lists title,
  year and note; `shouldEndSession` is false with a reprompt of `ask`.

## Still open

- Alexa+ does not document how identity or context reach MCP tools. The MCP tools identify a person only from a
  verified bearer token (`knots/identity.py`) and are stateless without one. The separate Alexa custom skill
  (`knots/alexa_skill.py`) identifies people from Amazon's signed request.
- The popularity wording thresholds (`widely seen` >= 10,000 ratings etc.) are my defaults, not tuned.
- No model-written phrasing is wired in; `rephrase` is a hook only.
