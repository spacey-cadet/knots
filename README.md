# Research-to-Media Recommendation Engine — Alexa+ MCP server (v1)

A self-hosted MCP server, built for the Alexa+ track of the Amazon Developer
Hackathon (Build, Ship, Shape). It turns what a person has been researching
— through Alexa+ conversations, and eventually Reddit/X activity — into a
curated, cross-format recommendation feed: real movies, documentaries,
videos, and papers, each with a stated reason it was picked.

This is the **runnable v1**: it works fully offline with zero API keys
(mock data fallback per catalog), and upgrades to live results the moment
you drop in a TMDB and/or YouTube API key.

## What's in here

| File | Purpose |
| --- | --- |
| `server.py` | The MCP server itself — Streamable HTTP, spec 2025-11-25+ |
| `interests.py` | `extract_interests` — heuristic topic extraction from raw text |
| `catalogs.py` | `search_content` — TMDB / YouTube / arXiv clients + offline mocks |
| `cards.py` | `explain_recommendation` + `render_card` — the "why" + visual card payload |
| `test_local.py` | Smoke test that runs the pipeline directly, no MCP client needed |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # optional — fill in TMDB_API_KEY / YOUTUBE_API_KEY for live data
```

## Run it

**Quick check — pipeline logic only, no server, no network needed:**

```bash
python test_local.py
```

You should see extracted topics, matched candidates, and rendered cards
printed to the console in a few seconds.

**Full MCP server:**

```bash
python server.py
```

Starts a Streamable HTTP MCP server at `http://localhost:8000/mcp`. Point
an MCP client — or your Alexa+ MCP configuration — at that URL.

**Verify it over the real protocol** (this is what a genuine MCP client
does, not just a function call):

```python
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    async with streamablehttp_client("http://localhost:8000/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print(await session.list_tools())
            result = await session.call_tool("get_recommendations", {
                "session_texts": ["I've been reading about memecoin tokenomics and liquidity pools."]
            })
            print(result.content[0].text)

asyncio.run(main())
```

## Tools exposed

- `extract_interests(session_texts: list[str]) -> list[dict]`
- `search_content(topics: list[str]) -> list[dict]`
- `get_recommendations(session_texts: list[str], limit: int = 6) -> dict` — the full pipeline, what Alexa+ actually calls

## Known limitations in v1 (honest, for the submission's product feedback)

- **Topic extraction is heuristic** (frequency-based, no LLM call yet) —
  good enough to demo, but will misfire on subtler phrasing. `interests.py`
  has an `llm_extract` stub ready for a Bedrock/Claude swap — this is the
  natural AWS Builder mini-challenge hook.
- **No live Reddit, X, or Google Takeout integration yet** — `session_texts`
  currently expects Alexa+ conversation strings; wiring in Reddit's OAuth
  API is the next real step (see the spec doc's data-sources section).
  X/Twitter needs a paid API tier and isn't attempted here.
- **Card variety logic is simple** (one item per topic+type) — works for a
  demo, would want real diversity/ranking logic for a production version.
- Network access to TMDB/YouTube/arXiv wasn't available in the environment
  this was built in, so the live-API code paths are written and reviewed
  but not live-tested — the mock fallback paths are what's been verified
  end-to-end. Test with a real `TMDB_API_KEY` before the demo recording.

## Next steps toward the full spec

See the companion spec doc for the data-sources roadmap, demo script, and
track/mini-challenge fit.
