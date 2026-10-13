# Evaluation

Measures whether Knots' picks and "unstated themes" are actually good, by
running 10 fixed sessions through each extractor and scoring your hand labels.

## Run it

Prerequisites: `TMDB_API_KEY` in `.env` (otherwise films are the fixed offline
mock set and the run is meaningless), plus Bedrock settings if you include it.

```bash
python eval/run_eval.py run --extractors "heuristic,bedrock=us.anthropic.claude-sonnet-4-6"
```

Optional: add more models to compare, e.g.
`"heuristic,bedrock=us.anthropic.claude-sonnet-4-6,bedrock=us.meta.llama3-3-70b-instruct-v1:0"`.
Bedrock runs in strict mode here, so a failure shows as `FAIL` and is never
quietly replaced by the heuristic.

## Label (about 1 hour)

Open both CSVs in a spreadsheet. **Hide the last column (`extractor`) while you
label**, so you judge each row without knowing which extractor produced it.
Rows are shuffled within each session for the same reason.

`eval/labels.csv`, one row per pick. Fill `good` with `y` or `n`:
- `y` = I would genuinely watch or read this for that session, and the stated
  reason holds up.
- `n` = wrong topic, a stretch, a generic title, or a reason that overclaims.
- Judge against `session_text`, not against what the film is famous for.
- Session `s10` is deliberately vague. If the text gave no real signal, picks
  are not good.
- Rows with `live = no` are offline mock items. They are skipped by `score`.

`eval/themes.csv`, one row per "unstated theme". Fill `supported` with `y` or `n`:
- `y` = the session text genuinely implies this, even if never said.
- `n` = invented or a stretch (for example "distrust of traditional finance"
  from two questions about memecoins).

Use the `note` column for anything worth quoting later. Failures and
surprises go straight into the write-up's "what didn't work" section.

## Score

```bash
python eval/run_eval.py score
```

Writes `eval/RESULTS.md`. Re-running `run` keeps the labels you already entered
(rows are matched by session, extractor and URL/theme).

## Commit these

`sessions.json`, `labels.csv`, `themes.csv`, `results/results.json` and
`RESULTS.md` are the evidence. Commit them so reviewers can check the numbers.