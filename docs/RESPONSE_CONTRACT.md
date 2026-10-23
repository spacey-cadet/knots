# Response contract

One shape, built by `knots/contract.py` and attached to every session plan as `plan["response"]`. Alexa, the MCP
tool and any screen render from it; nothing else is promised to them. Golden examples (regenerated from the real
code, and tested for drift): `tests/golden/response_ok.json`, `tests/golden/alexa_ok.json`
(`python tests/make_golden.py` rewrites them; review the diff).

| Field | Meaning |
| --- | --- |
| `status` | `ok`; `relaxed` (the closest items are offered, each saying `stretch`/`possible match`; only when no quality floor is fitted, see Rules); `empty` (the catalogs returned nothing, or nothing cleared the fitted quality floor, so there is nothing Knots will stand behind) |
| `say` | What a voice reads. At most 2 picks, no URLs, no quotes, under ~400 characters in the golden case |
| `picks[]` | `title, type, year, url, matched_topic, note, spoken_note, facts[], confidence` (plus `why`, the templated reasoning, only when `explain` is set) |
| `picks[].note` | The persuasion: why this person would love it, tied to their own words (`knots/notewriter.py`). If no written note passes the faithfulness checks, a templated one is used: the link to the person first, then what it is, then how well regarded it is (`knots/notes.py`). Built from catalog fields and the person's own words only |
| `picks[].facts[]` | `{label, value, source}` e.g. `7.1/10 from 4,200 ratings (TMDB)`; unknown is omitted, never guessed |
| `themes[]` | At most 5, ranked; each carries the exact words it rests on |
| `ask` | One follow-up question |
| `mode` | Optional `{learn_vs_play, label, from_session}` once a profile supplies it |

## Rules

- **Fewer, better picks, or an honest empty answer; never a padded list.** With a quality floor fitted (the shipped
  default, see the README), nothing below it is shown, even to fill a reserved slot, and the `relaxed` ladder is
  not used to refill the list. If nothing clears the floor, `status` is `empty`, `picks` is `[]`, and `say` says
  Knots won't guess and `ask` asks what the person wants to dig into. Without a fitted floor (or with
  `KNOTS_FLOOR=off`) the older ladder applies: normal ranking (needs receipts) -> closest by relevance (`relaxed`,
  `possible match`) -> very weak match (`stretch`), so there is at least one pick whenever the catalogs return
  anything. The empty rate under the floor is not yet measured (the eval runs with the floor off so every candidate
  can be labelled).
- **Evidence stays internal.** Behind every card is a ledger (`receipts`: the sentence the person said, matched
  catalog tags, the item's own summary, a paper's preprint status, fit features). It decides whether a card exists,
  feeds the grounding metric and is what a note is checked against, but the server does not return it. `knots/presenter.py`
  is the one place that decides what leaves: an allowlist of card fields, with the note filling in the persuading.
  `explain=true` on `knots_partner` / `get_recommendations` returns the full cards for "why did you pick that?".
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
