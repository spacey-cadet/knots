# Knots (Alexa+ MCP server)

Knots is a self-hosted MCP server for the Alexa+ track of the Amazon Developer Hackathon.
It acts as an **intellectual partner**: ask Alexa+ "what are we doing today?" and
it says what you seem to be circling, including the ideas underneath that you
never named, then suggests films and documentaries to watch, papers to read and
videos to explore, each with the reason it was picked.

The goal is to help people who think hard about abstract ideas find the films
that make those ideas concrete: entertainment that sits close to the research,
not a random feed.

The extractor runs with zero keys (heuristic fallback). Catalog results are always
real: with no TMDB key, or when a live call fails, the catalogs return nothing
rather than made-up titles. Set `KNOTS_OFFLINE=1` to use small fixed mock
catalogs for a no-network demo or tests.

Catalogs: films and documentaries and scripted TV series (TMDB, by film-tag
keyword), papers (arXiv) and YouTube explainers (3-60 minutes, title must match the
concept, 5,000+ views). Series have their own floor (rating >= 7.0, >= 150
votes).

Ranking: relevance (shared film-tag keywords) decides. Quality is a floor
(TMDB rating >= 6.5 with >= 100 votes; films before 1990 need 7.5+ and 1,000+
votes), not a score, and popularity is not a signal. One slot per session is
reserved for a lesser-known (<= 3,000 votes), highly rated (>= 7.0) pick.

## Repo layout

The whole recommendation algorithm is one package, `knots/`, with a single entry
point, `knots.engine`. `server.py` (the MCP transport) and `eval/run_eval.py`
(the evaluation harness) are thin callers of it and hold no recommendation logic.

```
knots/                  the algorithm
  engine.py             the one door: run / plan / recommend / extract / search / forget
  context.py            input scope: one or many conversations, merging, memory recall
  memory.py             SQLite ProfileStore (interests + embeddings, never words)
  embeddings.py         Titan v2 with an offline hashed fallback
  extraction.py         Bedrock extractor + heuristic fallback; best medium per interest
  catalogs.py           retrieval: TMDB films, series, docs; YouTube explainers; arXiv papers
  quality.py            the quality gate (rating, votes, age) and the reserved-slot rules
  ranking.py            relevance-first ranking, reserved slots, grounded reasons
  plan.py               watch / read / explore, spoken line, follow-up question
server.py               MCP tools, transport only
eval/                   run_eval.py, the labelling tool, your labels and results
tests/test_local.py     offline tests (no network, no keys)
```

```python
from knots import engine
plan = await engine.plan(session_texts=["..."], user_id="alex")   # same as the thinking_partner tool
```

## How it works

```
session text ──► understanding ──► retrieval ──► quality gate ──► ranking ──► plan
(Alexa+/Reddit)   Bedrock: topic,   TMDB films,   rating, votes,   relevance,   watch / read /
 + memory          concepts,        series, docs  age floor        reserved     explore + spoken
                   themes, mediums  YouTube, arXiv                  slots        line + question
```

- **Understanding (`extraction.py`).** Amazon Bedrock's Converse API with a forced
  tool call returns structured JSON: the surface topic, the abstract concepts at
  stake, unstated themes the text implies, short film-tag-style keywords and the
  best formats. Model output is treated as untrusted (validated, clamped, session
  text fenced in tags). On any failure (no credentials, no model access,
  throttling, bad JSON) it falls back to the heuristic, and the response reports
  which extractor ran.
- **Retrieval (`catalogs.py`).** TMDB is queried by *keyword*, not title, so "free
  will" finds films tagged with it rather than films named "Free Will". Papers
  from arXiv are attached as concept anchors.
- **Quality and ranking (`quality.py`, `ranking.py`).** See "Ranking" below.
  Reasons are built only from facts held: the concept and theme the extractor
  found, plus the tags that actually matched. No claims about a title beyond its
  tags.
- **Plan (`plan.py`).** Groups picks into watch / read / explore, adds a short
  `spoken` line for voice and a follow-up `ask` question.

## Tools exposed

| Tool | Use |
| --- | --- |
| `thinking_partner(session_texts, include_videos=True)` | "What are we doing today?" Threads, what's underneath, grouped picks with reasons, spoken line, follow-up question |
| `get_recommendations(session_texts, limit=6)` | Full pipeline as ranked cards |
| `extract_interests(session_texts)` | Topics, concepts, unstated themes, keywords, and which extractor ran |
| `search_content(topics)` | Catalog lookup for plain topic strings |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

To enable Bedrock: in the AWS console request access to a model in your region,
set `BEDROCK_MODEL_ID` (a model or inference-profile ID that supports Converse
with tool use) and `AWS_REGION`, and provide credentials through the standard
AWS chain. The role needs `bedrock:InvokeModel`. Never commit keys.

## Run

```bash
python -m pytest -q            # offline tests
python tests/test_local.py     # the tests, then a sample thinking_partner response
python server.py       # MCP server at http://localhost:8000/mcp
```

Verify over the real protocol:

```python
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    async with streamablehttp_client("http://localhost:8000/mcp") as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            res = await s.call_tool("thinking_partner", {
                "session_texts": ["I keep reading about memecoin tokenomics. Is it a bubble?"]})
            print(res.content[0].text)

asyncio.run(main())
```

## Input: one conversation or many

The tools take either the current conversation as `session_texts` (a list of
strings, as before) or several as `conversations`:

```json
{"conversations": [
  {"turns": ["what is a memecoin?"],     "timestamp": "2026-10-10T08:00:00Z"},
  {"turns": ["is crypto just a bubble?"], "timestamp": "2026-10-14T19:30:00Z"},
  {"turns": ["explain tokenomics"],       "timestamp": "2026-10-15T08:00:00Z"}]}
```

- **Newest conversation = what they want now.** It gets a weight boost and its
  strongest thread always stays in the top two.
- **Recurring topics = what they keep coming back to.** Each conversation is
  extracted separately (parallel, cached, so only new ones cost a call), then
  merged by shared topic or concept. Weight halves every 14 days and rises with
  the number of distinct conversations. Cards say "you've come back to X across
  N conversations".
- **Limits:** the last 8 conversations within 60 days, each trimmed to its latest
  3,000 characters. Without timestamps, order is used (the last one is "now").
- **Why not concatenate everything into one prompt:** unrelated topics blur into
  one vague profile, cost and latency grow with history, and old text crowds out
  new. (A bug of that kind was fixed: over-long input used to drop the NEWEST
  text; it now keeps the latest.)

## Memory

Pass a `user_id` and Knots remembers the person between conversations. Without
one it stays stateless: nothing is read or stored.

**What is kept, per conversation:** the extracted interests (topic, concepts,
unstated themes, keywords, best mediums), an embedding of the conversation, a
timestamp and a content key. **The words themselves are discarded** as soon as
they have been extracted and embedded. A test checks the database file for any
trace of the conversation text.

**How it is used**
- A new conversation is embedded and compared with the stored ones. The most
  related (similarity discounted by age, at most 6) are brought back, and their
  stored interests are merged with the new ones exactly like the multi-conversation
  input above: "you've come back to X across N conversations".
- With nothing said at all ("what should I explore?"), it answers from the most
  recent history alone (`extractor: "profile"`).
- Re-sending the same conversation is not double counted.
- A memory failure never breaks an answer; it is logged and the request goes on.

**Embeddings:** Amazon Titan Text Embeddings v2 (256 dimensions) when Bedrock is
configured, otherwise an offline hashed bag-of-words embedding (lexical only,
no keys). The embedder's name is stored with every vector, and vectors from
different embedders are never compared. `EMBEDDER=auto|titan|hashed`.

**Control:** `what_do_you_remember(user_id)` lists the threads held and how many
conversations they come from; `forget_me(user_id)` deletes everything stored for
that id. Storage is SQLite at `KNOTS_PROFILE_DB` (default `data/knots_profile.db`,
gitignored), capped at 500 conversations per person, behind a small `ProfileStore`
interface so DynamoDB can replace it.

**Privacy, plainly**
- Embeddings are not raw text, but they are not anonymous: they keep much of the
  meaning of what was said, and the stored "unstated themes" can be sensitive.
  Treat the store as personal data.
- Do not trust `user_id` as an argument in production. Derive it from the
  authenticated caller, or one person could read another's profile.
- A container's local disk does not survive restarts, so a deployed server needs
  a persistent store (DynamoDB) behind `ProfileStore`.
- Encrypt at rest and say what is kept in the app's privacy notice.

## Best medium per interest

Each interest carries a ranked list of `film`, `video` and `paper`. Film means
films, series and documentaries; video means an explainer; paper means research.
Bedrock ranks them for THAT interest (human and narrative themes lean to film,
abstract mechanisms to video, "what do the studies say" to paper). All three stay
possible: the best two are searched, and the third too when the person is going
deep. Cards are ordered by relevance weighted by medium rank, and the spoken line
leads with the best medium. Skipping a poor-fit medium also saves a catalog call.

## Known limitations

- **Live paths not yet tested against the real services.** Bedrock, TMDB, YouTube
  and arXiv are covered offline (fake Bedrock client, mock transports) but have
  not been called live from the build environment. Smoke-test with real keys
  before recording the demo.
- **Offline mock data is fixed** (a memecoin-themed set) and only used with
  `KNOTS_OFFLINE=1`. Without a TMDB key and without that flag, no films are returned.
- **The heuristic fallback is weak** (frequency-based; it can pick words like
  "buy" as topics). It exists to keep the server answering, not to demo.
- **No Reddit source yet.** `session_texts` expects Alexa+ conversation strings.
- **No deployment yet.** Alexa+ needs a public HTTPS endpoint with auth.
- **Ranking is simple** (keyword overlap, one item per topic and type). Next:
  embeddings + MMR diversity re-ranking and a hand-labelled precision@5 check.
