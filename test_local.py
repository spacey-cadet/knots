"""
Quick local smoke test — exercises the pipeline logic directly (no MCP
transport involved) so you can sanity-check extraction/matching/rendering
in seconds. Run: python test_local.py
"""

import asyncio
import json

from cards import rank_and_render
from catalogs import search_content
from interests import heuristic_extract

SAMPLE_SESSION = [
    "Alexa, what's a memecoin and why do people buy them?",
    "Can you explain memecoin tokenomics to me, like liquidity pools and market cap?",
    "I keep seeing memecoin threads on my Reddit feed, is this all just a bubble?",
    "Tell me more about the research behind speculative crypto assets.",
]


async def main():
    interests = heuristic_extract(SAMPLE_SESSION)
    print("=== Extracted interests ===")
    for i in interests:
        print(f"  {i.topic!r:25} depth={i.depth:8} sentiment={i.sentiment:9} mentions={i.mentions}")

    topics = [i.topic for i in interests]
    items = await search_content(topics)
    print(f"\n=== Matched {len(items)} candidate items across catalogs ===")

    cards = rank_and_render(items, interests, limit=6)
    print(f"\n=== Rendered {len(cards)} cards ===")
    print(json.dumps(cards, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
