"""The Knots Explorer (knots/explorer.py): off by default, passcode, signed visitor cookie, per-visitor memory and
rate limit, and that a client can never choose who it is."""
import pytest
from starlette.testclient import TestClient

SAID = "Is free will an illusion? Neuroscience says decisions are made before we are aware of them."
HDR = {"X-Knots-Explorer": "1"}


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    for k in ("KNOTS_JWKS_URL", "KNOTS_JWT_SECRET", "KNOTS_ALLOWED_HOSTS", "KNOTS_ALLOW_CLIENT_USER_ID", "KNOTS_EXPLORER_PASSCODE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("KNOTS_OFFLINE", "1")
    monkeypatch.setenv("EXTRACTOR", "heuristic")
    monkeypatch.setenv("KNOTS_USER_SALT", "s" * 40)
    monkeypatch.setenv("KNOTS_PROFILE_DB", str(tmp_path / "p.db"))
    monkeypatch.setenv("KNOTS_EVENT_LOG", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("KNOTS_RATE_LIMIT", "0")
    monkeypatch.setenv("KNOTS_RATE_GLOBAL", "0")
    monkeypatch.setenv("KNOTS_EXPLORER", "on")
    from knots import feedback, memory
    feedback._log = None
    if hasattr(memory, "_store"):
        memory._store = None


def client():
    import server
    return TestClient(server.create_mcp().streamable_http_app(), base_url="https://knots.example.com")


def call(c, tool, args=None, headers=HDR):
    return c.post("/explore/api/call", json={"tool": tool, "arguments": args or {}}, headers=headers)


def test_everything_is_a_404_unless_the_explorer_is_switched_on(monkeypatch):
    monkeypatch.delenv("KNOTS_EXPLORER")
    with client() as c:
        assert c.get("/explore").status_code == 404
        assert c.get("/explore/api/tools").status_code == 404
        assert call(c, "knots_partner").status_code == 404


def test_the_page_sets_a_signed_httponly_cookie_and_lists_every_tool():
    with client() as c:
        r = c.get("/explore")
        assert r.status_code == 200 and "Knots" in r.text
        cookie = r.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=strict" in cookie and "secure" in cookie
        tools = c.get("/explore/api/tools").json()["tools"]
        assert {t["name"] for t in tools} == {"knots_partner", "rate_pick", "get_recommendations", "extract_interests",
                                              "search_content", "what_do_you_remember", "forget_me"}
        kinds = {t["name"]: t["kind"] for t in tools}
        assert kinds["rate_pick"] == "write" and kinds["forget_me"] == "delete" and kinds["knots_partner"] == "read"
        st = c.get("/explore/api/status").json()
        assert st["remembers_visitors"] is True and "ranker" in st


def test_a_post_needs_json_and_the_explorer_header_and_a_known_tool():
    with client() as c:
        assert call(c, "knots_partner", headers={}).status_code == 400                      # no header: a cross-site form
        assert c.post("/explore/api/call", content="tool=x", headers=HDR).status_code == 400  # not JSON
        assert call(c, "no_such_tool").status_code == 404
        assert c.post("/explore/api/call", content=b"x" * 70_000, headers={**HDR, "content-type": "application/json"}).status_code == 413


def test_the_passcode_gates_the_api_when_set(monkeypatch):
    monkeypatch.setenv("KNOTS_EXPLORER_PASSCODE", "open sesame")
    with client() as c:
        assert c.get("/explore/api/tools").status_code == 401
        assert call(c, "knots_partner", {"session_texts": [SAID]}).status_code == 401
        ok = {**HDR, "X-Passcode": "open sesame"}
        assert c.get("/explore/api/tools", headers=ok).status_code == 200
        assert call(c, "knots_partner", {"session_texts": [SAID], "include_videos": False}, headers=ok).json()["ok"] is True


def test_a_run_returns_the_result_the_request_and_who_it_acted_as():
    with client() as c:
        c.get("/explore")
        body = call(c, "knots_partner", {"session_texts": [SAID], "include_videos": False}).json()
        assert body["ok"] and body["remembered_as"] == "visitor" and body["ms"] >= 0
        assert body["request"]["method"] == "tools/call" and body["request"]["params"]["name"] == "knots_partner"
        assert body["result"]["session_id"] and body["result"]["response"]["picks"]


def test_visitors_have_separate_memory_and_a_forged_cookie_gets_a_fresh_identity():
    with client() as a, client() as b:
        a.get("/explore"); b.get("/explore")
        call(a, "knots_partner", {"session_texts": [SAID], "include_videos": False})
        mine = call(a, "what_do_you_remember").json()["result"]
        theirs = call(b, "what_do_you_remember").json()["result"]
        assert mine["remembered_conversations"] >= 1 and theirs["remembered_conversations"] == 0
        vid = a.cookies.get("knots_visitor").split(".")[0]
        with client() as evil:
            forged = {**HDR, "cookie": f"knots_visitor={vid}.0000000000000000000a"}      # right id, wrong signature
            r = call(evil, "what_do_you_remember", headers=forged)
            assert r.json()["result"]["remembered_conversations"] == 0
            assert "knots_visitor=" in r.headers["set-cookie"] and vid not in r.headers["set-cookie"]   # a fresh identity


def test_a_user_id_argument_is_ignored_and_a_rating_closes_the_loop(monkeypatch):
    monkeypatch.setenv("KNOTS_ALLOW_CLIENT_USER_ID", "1")                                 # even in dev mode the explorer drops it
    with client() as a, client() as b:
        a.get("/explore"); b.get("/explore")
        out = call(a, "knots_partner", {"session_texts": [SAID], "include_videos": False}).json()["result"]
        pick = out["response"]["picks"][0]
        assert call(b, "what_do_you_remember", {"user_id": "anything"}).json()["result"]["remembered_conversations"] == 0
        rated = call(a, "rate_pick", {"session_id": out["session_id"], "url": pick["url"], "good": True}).json()
        assert rated["ok"] and rated["result"].get("recorded") is True
        bad = call(b, "rate_pick", {"session_id": out["session_id"], "url": pick["url"], "good": True}).json()
        assert bad["result"].get("recorded") is False                                       # someone else's session


def test_each_visitor_is_rate_limited(monkeypatch):
    monkeypatch.setenv("KNOTS_EXPLORER_RATE", "2")
    with client() as c:
        c.get("/explore")
        codes = [call(c, "what_do_you_remember").status_code for _ in range(3)]
        assert codes == [200, 200, 429]


def test_without_a_salt_the_explorer_works_but_remembers_nobody(monkeypatch):
    monkeypatch.delenv("KNOTS_USER_SALT")
    with client() as c:
        c.get("/explore")
        body = call(c, "knots_partner", {"session_texts": [SAID], "include_videos": False}).json()
        assert body["ok"] and body["remembered_as"] == "nobody"
        assert c.get("/explore/api/status").json()["remembers_visitors"] is False


def test_explain_is_a_tool_argument_and_the_ledger_only_comes_back_when_asked():
    with client() as c:
        c.get("/explore")
        tools = {t["name"]: t for t in c.get("/explore/api/tools").json()["tools"]}
        assert "explain" in tools["knots_partner"]["inputSchema"]["properties"]
        plain = call(c, "knots_partner", {"session_texts": [SAID], "include_videos": False}).json()["result"]
        card = next(x for b in plain["picks"].values() for x in b)
        assert "receipts" not in card and "fit_score" not in card and "why" not in plain["response"]["picks"][0]
        full = call(c, "knots_partner", {"session_texts": [SAID], "include_videos": False, "explain": True}).json()["result"]
        assert any("receipts" in x for b in full["picks"].values() for x in b)


def test_the_page_reads_a_pairs_bridge_and_only_shows_how_it_chose_when_explaining():
    with client() as c:
        page = c.get("/explore").text
        assert "p.bridge" in page and "p.why" not in page and "Matched on" not in page
        assert "Evidence behind this pick" in page
