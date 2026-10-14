# Evaluation

Measures whether Knots' picks and "unstated themes" are actually good: 10 fixed
sessions go through each extractor, you label the results blind, and `score`
turns the labels into a table.

Needs `TMDB_API_KEY` in `.env` (without it films are a fixed offline mock set
and the run is meaningless), plus Bedrock settings if you include Bedrock.

## 1. Run

```bash
python eval/run_eval.py run --extractors "heuristic,bedrock=us.anthropic.claude-sonnet-4-6"
```

Add more models to compare, e.g. `...,bedrock=us.meta.llama3-3-70b-instruct-v1:0`.
Bedrock runs in strict mode, so a failure shows as `FAIL` and is never quietly
replaced by the heuristic. Re-running keeps labels you already entered.

## 2. Label (about 1 hour)

```bash
python eval/run_eval.py label     # http://127.0.0.1:8765/  (Ctrl+C when done)
```

Keys: `Y` yes, `N` no, `S` skip, `B` back, `C` clear, `/` note. Labels save into
the CSVs instantly, and the page never shows which extractor made a row. The
rubric is on each card. Use the **Picks** and **Themes** tabs. Rows marked
"offline mock" are hidden by default and ignored by `score`.

Prefer a spreadsheet? Edit `labels.csv` (`good`) and `themes.csv` (`supported`)
with `y`/`n`, and hide the last column (`extractor`) while you do.

## 3. Score

```bash
python eval/run_eval.py score     # writes eval/RESULTS.md
```

## Commit as evidence

`sessions.json`, `labels.csv`, `themes.csv`, `results/results.json`, `RESULTS.md`.