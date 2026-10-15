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

Ranking: relevance (shared film-tag keywords) decides. Quality is a floor
(TMDB rating >= 6.5 with >= 100 votes; films before 1990 need 7.5+ and 1,000+
votes), not a score, and popularity is not a signal. One slot per session is
reserved for a lesser-known (<= 3,000 votes), highly rated (>= 7.0) pick.

## How it works

```
session text ──► Bedrock extractor ──► interests ──► catalogs ──► reasoned cards
(Alexa+/Reddit)   topic, concepts,     (fallback:     TMDB: films + docs (by keyword)
                  unstated themes,      heuristic)    arXiv: papers
                  film-tag keywords                   YouTube: explainers
```

- **Extraction (`interests.py`).** Amazon Bedrock's Converse API with a forced
  tool call returns structured JSON: the surface topic, the abstract concepts at
  stake, unstated themes the text implies, and short film-tag-style keywords.
  Model output is treated as untrusted (validated, clamped, session text fenced in
  tags). On any failure (no credentials, no model access, throttling, bad JSON)
  it falls back to the heuristic, and the response reports which extractor ran.
- **Search (`catalogs.py`).** TMDB is queried by *keyword*, not title, so "free
  will" finds films tagged with it rather than films named "Free Will". For each
  keyword it runs discover twice: documentaries (the literal pick) and fiction
  (the lateral pick). Papers from arXiv are attached as concept anchors.
- **Reasons (`cards.py`).** Built only from facts held: the concept and theme the
  extractor found, plus the TMDB tags that actually matched. No claims about a
  film beyond its tags.
- **Partner mode (`partner.py`).** Groups picks into watch / read / explore, adds
  a short `spoken` line for voice and a follow-up `ask` question.

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
python test_local.py   # offline tests + a sample thinking_partner response
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
