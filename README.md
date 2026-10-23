# Knots (Alexa+ MCP server)

Knots is a grounded recommender for Alexa+: it connects what a person has been
curious about with films, documentaries, explainers and research that help make
those ideas vivid. Ask "what are we doing today?" and Knots identifies the
threads and concepts underneath recent conversation, then returns picks with a
reason tied to the person and the source. The root page (`/`) is a swipeable
project pitch; `/explore` is the live tool console.

Knots is for people who think hard about abstract ideas and love film. It helps
them find entertainment that sits close to research, rather than a random feed.
Entertainment and learning belong together, so Knots also builds
**watch-and-learn pairs** (a drama plus the explainer or paper on the same idea) and
shows, for every pick, the evidence behind it.

The extractor runs with zero keys (heuristic fallback). Catalog results are always
real: with no TMDB key, or when a live call fails, the catalogs return nothing
rather than made-up titles. Set `KNOTS_OFFLINE=1` to use small fixed mock
catalogs for a no-network demo or tests.

Catalogs: films and documentaries and scripted TV series (TMDB, by film-tag
keyword), papers (arXiv) and YouTube explainers (3-60 minutes, title must match the
concept, 5,000+ views). Series have their own floor (rating >= 7.0, >= 150
votes).

## Ranking

- **Fit decides the order.** With the person's words available (the normal path),
  `fitmodel.py` scores each candidate by similarity to what was said and to the
  interest it was found for, tag overlap, quality, how well the format suits the
  interest, and, for papers, authority (log citations, survey status). Similarities
  are z-scored within the candidate set, so weights do not depend on which embedder
  ran. Without the person's words the original rule applies unchanged: shared
  film-tag keywords, weighted by medium, rating as tie-break. Popularity is not a
  signal, so the same famous films do not win every session.
- **Weights are stated, not hidden.** The current ranker uses hand-set priors in
  `fitmodel.DEFAULT_WEIGHTS`; it is not yet trained on a sufficient labelled set.
  Cards show **no calibrated confidence** without held-out calibration data; the
  `stretch` label only identifies picks admitted by the floor allowance. Once
  fitted, weights ship with a calibration measured on held-out sessions.
- **Quality is a floor, not a score** (`quality.py`): TMDB rating >= 6.5 with >= 100
  votes; series >= 7.0 with >= 150; anything before 1990 needs 7.5+ and 1,000+
  votes.
- **Both the best-rated and the most-voted titles are retrieved** for every film
  and series keyword, so widely seen titles are candidates, not only the niche
  best-rated ones.
- **Two reserved slots, each only if a candidate exists:**
  - a **blockbuster**: a widely seen film or series (>= 8,000 votes, series >= 2,000)
    that passes the quality floor AND is nearly as relevant as the best match for
    its topic (>= 60% of it). A famous title that only loosely fits is not suggested;
  - a **gem**: a lesser-known (<= 3,000 votes; series <= 1,500) but highly rated
    (>= 7.0) pick.

  Both sit in the first three cards, so a cap on the list cannot drop them. Their
  reasons say so ("Widely watched and well rated ..." / "Lesser-known, but well
  rated ...").
- Every number above is a constant in `quality.py`.

## Trust: an internal ledger, a persuasive note, confidence, and saying nothing

- **The note is what the person reads.** Each pick carries a short note on why *they* would love it, tied to their
  own words (`notewriter.py`, with a faithful templated fallback in `notes.py`). Every number, name and quotation in
  it must trace to catalog data or the person's own words, or it is discarded.
- **Receipts are an internal ledger, not something shown by default** (`receipts.py`): behind every card sits its evidence: a sentence you said
  that shares terms with the interest (quoted, never paraphrased), the catalog tags
  that matched, the first sentences of the item's own overview or abstract, and for
  papers whether it is a **preprint, not peer reviewed** or has a journal reference
  (arXiv lists one only if the authors added it), plus an OpenAlex citation count when
  the title matches exactly. A card with no matched tags and no summary is not shown. The ledger gates cards,
  feeds the grounding metric and checks the notes, but `knots/presenter.py` keeps it out of the response; pass
  `explain=true` (for "why did you pick that?") to get it.
- **Papers are ranked on more than a keyword**: citations (omitted, never guessed,
  when unknown) and survey/review status lift accessible, well-cited overviews.
- **Confidence only when measured**: themes carry a calibrated probability
  (`underneath_confidence`) and picks carry `confidence` only where a calibration was
  fitted on held-out data. Otherwise the field is empty.
- **Saying nothing**: themes below the judge's threshold are withheld
  (`withheld_themes`), and a pair is dropped if either half is unsourced.

## Watch-and-learn pairs

`plan()` returns `pairs`: one thing to watch (film, series, documentary) and one to
learn from (paper or explainer) about the SAME interest, as a double feature.

```
pair_score = min(fit(W), fit(L)) * cos(W, L) * level_match(W, L)
```

`min` stops a strong half rescuing a weak one; `cos` requires the two to cover the
same idea; `level_match` is a ladder (drama, then explainer, then survey paper). One
pair per interest, each item used once, and the bridge sentence is built only from
facts held (shared tags that literally appear in both). The floors and the ladder are
**untuned design choices**: label pairs in the `Pairs` tab and run
`python eval/eval_pairs.py` to find out whether they are right.

## Challenge me (opt-in)

`knots_partner(..., challenge_me=True)` (or `KNOTS_CHALLENGE=1`) reserves one slot
for a well-rated title found with the extractor's `counter_keywords`: a serious view that
complicates yours. The card says what it pushes back on. Off by default.

## Repo layout

The whole recommendation algorithm is one package, `knots/`, with a single entry
point, `knots.engine`. `server.py` (the MCP transport) and `eval/run_eval.py`
(the evaluation harness) are thin callers of it and hold no recommendation logic.

```
knots/                  the algorithm
  engine.py             the one door: run / plan / recommend / extract / search / forget
  context.py            input scope: one or many conversations, merging, memory recall
  memory.py             SQLite ProfileStore (interests + embeddings, never words)
  embeddings.py         Cohere Embed v4 / Titan v2 on Bedrock, with an offline hashed fallback
  extraction.py         Bedrock extractor + heuristic fallback; best medium per interest
  catalogs.py           retrieval: TMDB films, series, docs; YouTube explainers; arXiv papers
  quality.py            the quality gate (rating, votes, age) and the reserved-slot rules
  ranking.py            ranking, reserved slots (blockbuster, gem, challenge), grounded reasons
  fitmodel.py           the fit model: similarity + quality + format + authority, calibrated if fitted
  signals.py            survey flag, reading level, level ladder, paper authority
  receipts.py           the internal evidence ledger behind every card (not shown unless explain=true)
  presenter.py          what leaves the server: an allowlist of card fields, notes filled in
  pairing.py            watch-and-learn pairs
  calibration.py        "when it says 80%, is it right 80% of the time?"
  plan.py               watch / read / explore, pairs, spoken line, follow-up question
  brain/                the learned theme judge: critic, nearest labelled themes, few-shot guidance
    data/               shipped artifacts (critic, examples, calibration); retrain with eval/train_brain.py
server.py               MCP tools, transport only
eval/                   run_eval.py, the labelling tool, labels, results, training and evaluation scripts
MODEL_CARD.md           generated from eval/results/metrics_*.json (eval/make_model_card.py)
tests/                  offline tests (no network, no keys)
```

```python
from knots import engine
plan = await engine.plan(session_texts=["..."], user_id="alex")   # same as the knots_partner tool
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
| `knots_partner(session_texts, request, include_videos=True, challenge_me=False, explain=False)` | "What are we doing today?" Threads, what's underneath, grouped picks each with a note on why you'd love it (the evidence ledger only with `explain=true`), watch-and-learn pairs, spoken line, follow-up question, and a `session_id`. `session_texts` is conversation context; `request` is the separate immediate ask, such as "what should I watch?" |
| `rate_pick(session_id, url, good)` | The person said a pick was good or not. Only for a signed-in person and only for a pick shown to them in that session; this is the feedback retraining learns from |
| `get_recommendations(session_texts, request, limit=6, challenge_me=False, explain=False)` | Full pipeline as ranked cards. Context and immediate request are separate; each card has a note on why you'd love it (the evidence ledger and templated reasoning only with `explain=true`) |
| `extract_interests(session_texts)` | Topics, concepts, unstated themes, keywords, and which extractor ran |
| `search_content(topics)` | Catalog lookup for plain topic strings |
| `what_do_you_remember()` / `forget_me()` | What is held about the signed-in person, and deleting it (memory and logged feedback) |

The server also exposes `GET /health` and `POST /alexa` (an Alexa custom-skill endpoint). See "Identity, feedback
and deployment" below.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

To enable Bedrock: in the AWS console request access to a model in your region,
set `BEDROCK_MODEL_ID` (a model or inference-profile ID that supports Converse
with tool use) and `AWS_REGION`, and provide credentials through the standard
AWS chain. The role needs `bedrock:InvokeModel`. Never commit keys.

## Knots Explorer

The pitch is served at `https://<your-host>/`; its **Try it live** button opens `/explore`. Keep any passcode in
private submission notes, not on the pitch page.

The Swagger-style tool console is at `https://<your-host>/explore`. Pick a tool, fill in the input (or start
from a sample conversation), press **Run tool**, and see the answer as cards, raw JSON, and the exact MCP request
(with a `curl` you can paste). The Good pick / Missed buttons call `rate_pick`, so the learning loop is visible too.

- **Off by default.** Set `KNOTS_EXPLORER=on`. Optionally `KNOTS_EXPLORER_PASSCODE`; the page asks for it.
- **Same server, same tools.** Calls go through the MCP server's own tool registry, so what you see is what Alexa+ gets.
- **No MCP token in the browser.** Each visitor is a random id in a signed, HttpOnly, SameSite=Strict cookie, turned
  into an opaque per-visitor user: memory and ratings work and no visitor can read another's. A `user_id` argument is
  always dropped. Without `KNOTS_USER_SALT` the page works but remembers nobody.
- **Quota guards.** Each visitor is limited to `KNOTS_EXPLORER_RATE` runs a minute on top of the global limit, and the
  YouTube toggle starts off (YouTube allows about 100 searches a day for the whole server).
- The explorer is `knots/explorer_ui/index.html`; the pitch is `knots/pitch.html`. Both use the same colors and embedded fonts.
- For recommendation tools, enter **context** (the conversation that reveals interests) separately from the
  **request** (what the person wants now, such as "what should I watch?"). With no context, a topical request can
  supply the subject by itself.

## Quality floor and written notes

- **Floor.** `eval/train_rank.py --write` can fit a similarity floor from held-out labels; it is stored in
  `knots/brain/data/rank_weights.json`. Picks below it do not count as having cleared the floor. By default, if
  fewer than three picks clear it, Knots may fill the list with nearby candidates within `KNOTS_FLOOR_SLACK`
  (default `0.10` similarity), mark them `relaxed`, and label their confidence `stretch`. Set
  `KNOTS_FLOOR_MIN_PICKS=0` for strict-floor behavior. Raw similarity only means the same thing on the embedder
  it was measured on, so a mismatched embedder cannot use the allowance.
- **Notes.** `knots/notewriter.py` asks the Bedrock model (`NOTE_MODEL_ID`, else `BEDROCK_MODEL_ID`) for a note
  and a spoken line per pick in one call. It sees only the catalog's facts and the person's quote, and a note is
  discarded unless every number, capitalised name and quotation in it traces to that data. A failure or a
  timeout keeps the templated note. `NOTES_LLM=off` disables it. The eval always runs with both off.

## Run

```bash
python -m pytest -q            # offline tests
python tests/test_local.py     # the tests, then a sample knots_partner response
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
            res = await s.call_tool("knots_partner", {
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

Give the engine a `user_id` and Knots remembers the person between conversations. Without
one it stays stateless: nothing is read or stored. The server does not take that id from the caller: it comes from
a verified sign-in (see "Identity, feedback and deployment").

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

**Embeddings:** Cohere Embed v4 (256 dimensions, up to 96 texts per request) or
Amazon Titan Text Embeddings v2 (256 dimensions, one text per request, capped at
60 requests/minute with no quota increase possible) when Bedrock is configured,
otherwise an offline hashed bag-of-words embedding (lexical only, no keys). Set
`EMBED_MODEL_ID` to a Cohere Embed v4 cross-region inference profile to use
Cohere; calls are paced (`COHERE_RPM`, `TITAN_RPM`), throttling is retried with
backoff, and a circuit breaker drops a whole run to hashed rather than mixing
embedders. The embedder's name is stored with every vector, and vectors from
different embedders are never compared. `EMBEDDER=auto|cohere|titan|hashed`.

**Control:** `what_do_you_remember` lists the threads held and how many
conversations they come from; `forget_me` deletes everything stored for the
signed-in person, including their logged feedback. Storage is SQLite at `KNOTS_PROFILE_DB` (default `data/knots_profile.db`,
gitignored), capped at 500 conversations per person, behind a small `ProfileStore`
interface so DynamoDB can replace it.

**Privacy, plainly**
- Embeddings are not raw text, but they are not anonymous: they keep much of the
  meaning of what was said, and the stored "unstated themes" can be sensitive.
  Treat the store as personal data.
- The server never trusts a `user_id` argument: the id comes from a verified bearer
  token or a signature-checked Alexa request and is stored only as a salted hash, so
  the database and event log hold no Amazon id or token subject. A caller-supplied id
  counts only when `KNOTS_ALLOW_CLIENT_USER_ID=1` (development).
- A container's local disk does not survive restarts. The EC2 setup keeps the database
  and event log on a Docker volume on the instance's disk; a platform with an ephemeral
  disk (App Runner) would need a DynamoDB store behind `ProfileStore`, which does not exist yet.
- Encrypt at rest and say what is kept in the app's privacy notice.

## Identity, feedback and deployment

- **Identity.** MCP calls carry a bearer token (a JWT) verified against your identity provider's keys
  (`KNOTS_JWKS_URL`) or a shared secret (`KNOTS_JWT_SECRET`), with a required audience. Alexa skill requests are
  checked against Amazon's signature rules, and `session.user.userId` is the person. Both become an opaque salted id
  (`KNOTS_USER_SALT`). Code: `knots/identity.py`, `knots/alexa_skill.py`.
- **Feedback.** Every response logs which picks were shown (`impression`); `rate_pick` and the skill's "good pick" /
  "not for me" log a verdict. `python eval/monitor.py` reads the log, reports the online metrics and says whether a
  retrain is due (`docs/METRICS.md`). Only explicit verdicts become training labels.
- **Alexa skill.** `alexa/` holds the skill manifest and interaction model; `alexa/README.md` has the steps.
- **Deploying.** `Dockerfile`, `docker-compose.yml` (Caddy gets the HTTPS certificate) and `docs/DEPLOY_EC2.md`.

### Planned Alexa+ integration: MCP add-on

The Alexa+ add-on has **not been created or connected yet**. Knots' MCP server is implemented at `/mcp`, and the
Explorer calls those tools as a browser MCP client. This is separate from the legacy custom skill in `alexa/`,
which receives signed requests at `/alexa`.

When connected, an Alexa+ MCP add-on will register the remote server. Alexa+ discovers its tools and their names,
descriptions and input schemas through MCP, then its own reasoning chooses which tool fits the person's request.
MCP supplies the tools and their contracts; it does not specify Alexa+'s tool-selection algorithm. The Explorer
does **not** emulate Alexa+'s natural-language understanding or automatic tool selection: a person chooses a tool
from its list, enters its arguments and sees the server response. It demonstrates the MCP request/response path,
not the Alexa+ experience or device presentation.

`knots_partner` is intended as the main user-facing recommendation tool. Its MCP description already says to use
it for requests such as these:

- “What are we doing today?”
- “What should I watch or read about free will?”
- “Connect what I’ve been curious about to a film and something to read.”

These are example phrases to include in the future add-on's `examplePhrases`; they are guidance for Alexa+'s
discovery and reasoning, not exact-match triggers. For a call, Alexa+ should pass recent discussion as
`session_texts` (context) and the immediate ask separately as `request`. `get_recommendations` is for a plain
ranked card list; `extract_interests` inspects interests without searching; `search_content` looks up topics
already supplied. The other tools handle feedback and memory actions.

The add-on package is the Alexa-side registration and description of this service; recommendation logic stays in
this server. Knots fits the general Alexa+ **MCP Toolkit** path rather than a category-specific add-on. The
onboarding flow uses the Alexa AI CLI to scaffold and deploy the package, then test it in the web simulator and on
a device before certification. Redeploy after changing tools or their descriptions. See Amazon's
[MCP Toolkit overview](https://developer.amazon.com/docs/alexaplus/add-ons/mcp-toolkit-overview.html),
[MCP onboarding guide](https://developer.amazon.com/docs/alexaplus/add-ons/mcp-toolkit-quickstart.html), and
[certification guide](https://developer.amazon.com/docs/alexaplus/add-ons/mcp-toolkit-certify.html). The toolkit
is currently limited to selected partners and the US; check the current [developer docs](https://developer.amazon.com/docs/alexaplus/add-ons/home.html)
for access and availability.

```bash
alexa-ai configure
alexa-ai new mcp --name "Knots" --locale en-US --mcp-server-url "https://<DOMAIN>/mcp"
# Review addon-package/addon.json; add the required description, example phrases, policy URLs and media.
alexa-ai deploy
# Test in the Alexa+ web simulator/device; submit for certification when ready.
```

**Authentication to finish before a connected add-on can use identity-linked features:** the supplied Inspector
capture shows the unauthenticated `initialize` got `401`; the
protected-resource metadata pointed to `https://knots-tool.duckdns.org/` as the authorization server, and the
Inspector then received `404` from `/.well-known/oauth-authorization-server`. The server accepts bearer JWTs,
but does not host the OAuth authorization/token service required for Alexa-managed account linking. Before
enabling linked profiles and feedback, deploy a compatible OAuth 2.1 authorization service and expose discovery
metadata and PKCE S256 support as required by Amazon's [authentication guide](https://developer.amazon.com/docs/alexaplus/add-ons/mcp-toolkit-authentication.html).
The capture also shows a `WWW-Authenticate` header on the `401`; Amazon's current checklist says not to send that
header. This concerns the captured deployment's auth discovery, not whether the Explorer can manually call the
MCP tools. Recheck the live endpoints during add-on onboarding.

The auth work matters for Knots' per-person memory and pick feedback. Without a verified user identity, Knots can
still make stateless recommendations, but it cannot retain a profile or accept persistent feedback for that person.
The separate Alexa custom skill already verifies Amazon-signed requests; that mechanism does not authenticate the
MCP add-on.

## Best medium per interest

Each interest carries a ranked list of `film`, `video` and `paper`. Film means
films, series and documentaries; video means an explainer; paper means research.
Bedrock ranks them for THAT interest (human and narrative themes lean to film,
abstract mechanisms to video, "what do the studies say" to paper). All three stay
possible: the best two are searched, and the third too when the person is going
deep. Cards are ordered by relevance weighted by medium rank, and the spoken line
leads with the best medium. Skipping a poor-fit medium also saves a catalog call.

## Known limitations

- **The heuristic fallback is weak** (frequency-based; it can pick words like
  "buy" as topics). It exists to keep the server answering, not to demo.
- **No Reddit source yet.** `session_texts` expects Alexa+ conversation strings.
- **Alexa+ add-on is not connected yet.** The Explorer is a manual MCP client, not an Alexa+ simulator; it cannot
  show Alexa+'s automatic tool choice, voice response or device UI. The tool descriptions name the intended
  recommendation trigger requests, but actual selection needs to be confirmed in an Alexa+ add-on test.
- **Alexa+ account-linking OAuth is not configured.** The supplied Inspector capture reaches resource metadata but
  gets 404 from OAuth authorization-server discovery. The app accepts bearer JWTs, but a compatible OAuth flow
  and per-person identity mapping still need deployment work before linked profiles and feedback can be used.
- **Pair selection is a fixed formula.** It combines the
  component picks' fit percentiles, their text similarity and a medium-level rule.
  No pairs have been labelled yet, so its thresholds and level ladder remain
  untuned. Similarity is a hashed lexical proxy unless Bedrock embeddings (Cohere
  Embed v4 or Titan) are configured; ranking and pair quality should be measured
  again with the production embedder. `eval/train_brain.py --save` trains a
  different component—the theme judge—and does not train ranking or pairs.
  `MODEL_CARD.md` reports the current state and measurement embedder.
- **One labeller, few sessions.** See `MODEL_CARD.md` for what is and is not measured.


## The brain: themes judged the way the author judges them

The "unstated themes" Bedrock produces were only 44% supported when hand-labelled (101 themes, 10 sessions).
The labels and notes in `eval/themes.csv` fall into recurring errors (over-psychologizing, terminological
overreach, narrative projection, invented "X vs Y"), recorded in the `error_type` column. Knots learns from them
in three ways (`knots/brain/`):

1. **Few-shot guidance** (`guidance.py`): distilled rules plus the accepted and rejected labelled themes nearest
   to the current text (with your reason) are added to the extraction prompt.
2. **Theme critic** (`critic.py`): a logistic model over interpretable features (stance/mechanism wording,
   unstated feelings, drama words, "vs" patterns).
3. **Nearest labelled themes** (`neighbors.py`): a theme is scored by the labelled themes it most resembles.
   No hand-written word lists, and every new label changes it immediately.

`judge.py` averages 2 and 3, calibrates the result (Platt scaling fitted on held-out scores) and withholds themes
below the threshold. Honest numbers at the time of writing, leave-one-session-out on the 10 train sessions
(`python eval/train_brain.py`; always see `MODEL_CARD.md` for the current ones): critic AUC 0.81, nearest-theme
AUC 0.78, **average 0.84 [0.77, 0.91]**; keeping calibrated p >= 0.5 keeps 36 of 101 themes at 75% supported
(base rate 44%).

**Caveats you should keep.** The critic's word lists were written after reading these same labels, so its
cross-validated number is optimistic. The only clean test is the **holdout**: sessions marked `"split": "holdout"`
in `eval/sessions.json` (s25-s30) are never trained or tuned on. They are unlabelled; label them in your own
words (no LLM drafting), then re-run `train_brain.py` to get the number to trust. Limits: one labeller, one
generator, 10 labelled sessions.

## Learning from your labels (the workflow)

```bash
python eval/run_eval.py run                 # 30 sessions through the extractors; builds the label sheets
python eval/run_eval.py run --resume        # YouTube allows ~100 searches/day: finish over several days (cached)
python eval/run_eval.py label               # tabs: Picks, Themes (+ error type), Pairs
python eval/check_labels.py                 # contradictions, missing reasons, holdout hygiene
python eval/train_brain.py --save           # theme judge: critic + neighbours + calibration (never uses holdout)
python eval/train_rank.py --write           # ranker weights + calibration (needs 20+ train sessions)
python eval/eval_pairs.py                   # how often a pair is one you would watch and read together
python eval/make_model_card.py              # MODEL_CARD.md from the metrics files
```

Rules the scripts enforce: models are scored on sessions they never saw; holdout rows are never trained on;
weights are written only with enough sessions; intervals come from resampling whole sessions. Retraining after
relabelling is one command. Put an honest account of how labels and notes were made (including any LLM help) in
`eval/LABEL_PROVENANCE.txt`; the model card quotes it. Set `KNOTS_BRAIN=off` to disable the judge and guidance.

Environment: `KNOTS_THEME_THRESHOLD` (override the shipped threshold), `KNOTS_CHALLENGE=1`, `KNOTS_OPENALEX=0`
(skip citation lookups), `OPENALEX_MAILTO` (identify yourself to OpenAlex's polite pool).
