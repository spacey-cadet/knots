"""Identity, rate limiting, feedback, the MCP server (auth, hosts, tools) and the Alexa skill endpoint with real
signature verification against a throwaway certificate authority."""
import asyncio
import base64
import datetime as dt
import json
import time

import pytest

from knots import alexa_skill as AS
from knots import feedback as FB
from knots import identity as ID
from knots import online as O
from knots.ratelimit import RateLimiter

SKILL = "amzn1.ask.skill.test-skill"


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    for k in ("KNOTS_JWKS_URL", "KNOTS_JWT_SECRET", "KNOTS_JWT_ISSUER", "KNOTS_JWT_AUDIENCE", "KNOTS_ALLOWED_HOSTS",
              "KNOTS_ALLOW_CLIENT_USER_ID", "KNOTS_PUBLIC_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("KNOTS_USER_SALT", "s" * 40)
    monkeypatch.setenv("KNOTS_EVENT_LOG", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("KNOTS_ALEXA_SKILL_ID", SKILL)
    monkeypatch.setenv("KNOTS_RATE_LIMIT", "0")
    monkeypatch.setenv("KNOTS_RATE_GLOBAL", "0")
    AS._pending.clear()
    AS._chain_cache.clear()
    FB._log = None


# ------------------------------------------------------------------------------------------ a throwaway PKI
def _pki(san="echo-api.amazon.com", days_valid=30, start_offset_days=-1, eku=True):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    now = dt.datetime.now(dt.timezone.utc)

    def name(cn):
        return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    ca_key = rsa.generate_private_key(65537, 2048)
    ca = (x509.CertificateBuilder().subject_name(name("Test Root")).issuer_name(name("Test Root"))
          .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
          .not_valid_before(now - dt.timedelta(days=5)).not_valid_after(now + dt.timedelta(days=365))
          .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
          .add_extension(x509.KeyUsage(True, False, False, False, False, True, False, False, False), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
          .sign(ca_key, hashes.SHA256()))
    leaf_key = rsa.generate_private_key(65537, 2048)
    b = (x509.CertificateBuilder().subject_name(name(san)).issuer_name(ca.subject).public_key(leaf_key.public_key())
         .serial_number(x509.random_serial_number())
         .not_valid_before(now + dt.timedelta(days=start_offset_days))
         .not_valid_after(now + dt.timedelta(days=start_offset_days + days_valid))
         .add_extension(x509.SubjectAlternativeName([x509.DNSName(san)]), critical=False)
         .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
         .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
         .add_extension(x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()), critical=False))
    if eku:
        b = b.add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
    leaf = b.sign(ca_key, hashes.SHA256())
    return {"roots": [ca], "chain": leaf.public_bytes(serialization.Encoding.PEM), "key": leaf_key}


@pytest.fixture(scope="module")
def pki():
    return _pki()


def sign(key, body: bytes, sha256=True) -> str:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    return base64.b64encode(key.sign(body, padding.PKCS1v15(), hashes.SHA256() if sha256 else hashes.SHA1())).decode()


def alexa_body(intent=None, kind="IntentRequest", user="amzn1.ask.account.A", attrs=None, app=SKILL, ts=None):
    req = {"type": kind, "timestamp": ts or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if intent:
        req["intent"] = intent
    return {"version": "1.0", "session": {"application": {"applicationId": app}, "user": {"userId": user},
                                          "attributes": attrs or {}}, "request": req}


URL = "https://s3.amazonaws.com/echo.api/echo-api-cert.pem"


def signed(pki, body: dict, sha256=True):
    raw = json.dumps(body).encode()
    h = {"signaturecertchainurl": URL}
    h["signature-256" if sha256 else "signature"] = sign(pki["key"], raw, sha256)
    return h, raw


async def run(pki, body, tamper=False, headers=None, **kw):
    h, raw = signed(pki, body)
    if tamper:
        raw = raw.replace(b"IntentRequest", b"IntentRequezt")
    async def fetch(_):
        return pki["chain"]
    await AS.verify_request(headers or h, raw, json.loads(raw), fetch=fetch, roots=pki["roots"], **kw)


# ------------------------------------------------------------------------------------------ alexa verification
@pytest.mark.parametrize("url,ok", [
    ("https://s3.amazonaws.com/echo.api/echo-api-cert.pem", True),
    ("https://s3.amazonaws.com:443/echo.api/echo-api-cert.pem", True),
    ("https://s3.amazonaws.com/echo.api/../echo.api/echo-api-cert.pem", True),
    ("HTTPS://S3.AMAZONAWS.COM/echo.api/echo-api-cert.pem", True),
    ("http://s3.amazonaws.com/echo.api/echo-api-cert.pem", False),
    ("https://notamazon.com/echo.api/echo-api-cert.pem", False),
    ("https://s3.amazonaws.com.evil.com/echo.api/echo-api-cert.pem", False),
    ("https://s3.amazonaws.com/EcHo.aPi/echo-api-cert.pem", False),
    ("https://s3.amazonaws.com/invalid.path/echo-api-cert.pem", False),
    ("https://s3.amazonaws.com:563/echo.api/echo-api-cert.pem", False),
    ("https://s3.amazonaws.com/echo.api/../evil/cert.pem", False),
    ("https://user@s3.amazonaws.com/echo.api/echo-api-cert.pem", False),
    ("", False), ("not a url", False)])
def test_certificate_url_rules(url, ok):
    assert AS.cert_url_ok(url) is ok


def test_a_correctly_signed_request_from_our_skill_passes_and_sha1_is_the_fallback(pki):
    body = alexa_body()
    asyncio.run(run(pki, body))
    h, raw = signed(pki, body, sha256=False)
    async def fetch(_):
        return pki["chain"]
    asyncio.run(AS.verify_request(h, raw, json.loads(raw), fetch=fetch, roots=pki["roots"]))


@pytest.mark.parametrize("name,kw", [
    ("tampered body", {"tamper": True}),
    ("another skill", {"body": alexa_body(app="amzn1.ask.skill.someone-else")}),
    ("old timestamp", {"body": alexa_body(ts="2020-01-01T00:00:00Z")}),
    ("no timestamp", {"body": {**alexa_body(), "request": {"type": "IntentRequest"}}}),
    ("bad cert url", {"headers": {"signaturecertchainurl": "https://evil.com/echo.api/x.pem", "signature-256": "AAAA"}}),
    ("no signature headers", {"headers": {"signaturecertchainurl": URL}}),
])
def test_requests_that_fail_any_rule_are_rejected(pki, name, kw):
    kw = dict(kw)
    body = kw.pop("body", alexa_body())
    with pytest.raises(AS.AlexaVerificationError):
        asyncio.run(run(pki, body, **kw))


@pytest.mark.parametrize("name,make", [
    ("wrong certificate name", lambda: _pki(san="evil.example.com")),
    ("expired certificate", lambda: _pki(days_valid=2, start_offset_days=-10)),
    ("not yet valid", lambda: _pki(start_offset_days=10)),
])
def test_bad_signing_certificates_are_rejected(name, make):
    bad = make()
    with pytest.raises(AS.AlexaVerificationError):
        asyncio.run(run(bad, alexa_body()))


def test_a_certificate_from_an_untrusted_authority_and_a_foreign_signature_are_rejected(pki):
    other = _pki()
    with pytest.raises(AS.AlexaVerificationError, match="not trusted"):
        asyncio.run(run({**other, "roots": pki["roots"]}, alexa_body()))          # chain not under the trusted root
    h, raw = signed(pki, alexa_body())
    async def fetch(_):
        return other["chain"]                                                      # a different key than the signer's
    with pytest.raises(AS.AlexaVerificationError):
        asyncio.run(AS.verify_request(h, raw, json.loads(raw), fetch=fetch, roots=other["roots"]))


def test_with_no_skill_id_configured_nothing_verifies(pki, monkeypatch):
    monkeypatch.delenv("KNOTS_ALEXA_SKILL_ID")
    with pytest.raises(AS.AlexaVerificationError):
        asyncio.run(run(pki, alexa_body()))


# ------------------------------------------------------------------------------------------ the skill itself
def fake_plan(urls=("http://a", "http://b")):
    picks = [{"title": f"T{i}", "type": "movie", "year": "2020", "url": u, "matched_topic": "t", "note": "n",
              "spoken_note": "n", "why": "w", "facts": [], "confidence": "strong match"} for i, u in enumerate(urls)]
    cards = {"watch": [{"url": u, "type": "movie", "title": f"T{i}", "fit_score": 1.0, "votes": 1000, "matched_topic": "t",
                        "fit_features": {"overlap": 1.0, "sim_session": 1.0}} for i, u in enumerate(urls)],
             "read": [], "explore": []}
    return {"extractor": "heuristic", "threads": [], "picks": cards, "spoken": "s", "ask": "Anything else?",
            "response": {"status": "ok", "say": "Try T0, a film.", "picks": picks, "themes": [], "ask": "Anything else?"}}


@pytest.fixture
def engine_calls(monkeypatch):
    calls = []

    async def plan(texts, convs, user, *a, **k):
        calls.append((list(texts), user))
        return fake_plan()
    monkeypatch.setattr(AS.engine, "plan", plan)
    return calls


def intent(name, **slots):
    return {"name": name, "slots": {k: {"name": k, "value": v} for k, v in slots.items()}}


def test_launch_help_stop_and_session_end_need_no_pipeline(engine_calls):
    assert "Knots" in asyncio.run(AS.handle_alexa(alexa_body(kind="LaunchRequest")))["response"]["outputSpeech"]["text"]
    stop = asyncio.run(AS.handle_alexa(alexa_body(intent("AMAZON.StopIntent"))))
    assert stop["response"]["shouldEndSession"] is True
    assert "good pick" in asyncio.run(AS.handle_alexa(alexa_body(intent("AMAZON.HelpIntent"))))["response"]["outputSpeech"]["text"]
    assert asyncio.run(AS.handle_alexa(alexa_body(kind="SessionEndedRequest")))["response"] == {}
    assert engine_calls == []


def test_a_topic_request_runs_the_pipeline_speaks_ssml_and_sets_up_feedback(engine_calls):
    out = asyncio.run(AS.handle_alexa(alexa_body(intent("KnotsPartnerIntent", topic="memecoins"))))
    assert engine_calls[0][0] == ["memecoins"] and engine_calls[0][1].startswith("u_")     # opaque id, not the Alexa id
    assert "amzn1" not in str(engine_calls)
    r = out["response"]
    assert r["outputSpeech"]["type"] == "SSML" and "Try T0" in r["outputSpeech"]["ssml"] and r["shouldEndSession"] is False
    assert "Was T0 a good pick" in r["reprompt"]["outputSpeech"]["text"]
    assert out["sessionAttributes"]["last_url"] == "http://a" and out["sessionAttributes"]["session_id"]
    ev = [e for e in O.EventLog(FB.log_path()).read() if e["type"] == "impression"]
    assert len(ev) == 1 and [p["url"] for p in ev[0]["picks"]] == ["http://a", "http://b"] and ev[0]["user"].startswith("u_")


def test_a_plan_with_no_interests_still_gets_an_honest_spoken_answer(monkeypatch):
    async def plan(*a, **k):
        return {"extractor": "heuristic", "threads": [], "picks": {"watch": [], "read": [], "explore": []},
                "spoken": "I do not have anything to go on yet.", "ask": "What have you been curious about lately?"}
    monkeypatch.setattr(AS.engine, "plan", plan)
    out = asyncio.run(AS.handle_alexa(alexa_body(intent("KnotsPartnerIntent"))))
    assert "anything to go on" in out["response"]["outputSpeech"]["ssml"] and out["sessionAttributes"]["last_url"] == ""
    assert [e["status"] for e in O.EventLog(FB.log_path()).read()] == ["empty"]


def test_good_and_bad_pick_become_verified_feedback_that_retraining_can_use(engine_calls):
    shown = asyncio.run(AS.handle_alexa(alexa_body(intent("KnotsPartnerIntent", topic="x"))))
    attrs = shown["sessionAttributes"]
    done = asyncio.run(AS.handle_alexa(alexa_body(intent("GoodPickIntent"), attrs=attrs)))
    assert "remember that" in done["response"]["outputSpeech"]["text"] and done["sessionAttributes"]["last_url"] == ""
    events = O.EventLog(FB.log_path()).read()
    labels = O.labels_from_events(events)
    assert [(x["good"], x["source"], x["url"]) for x in labels] == [("y", "user", "http://a")]
    assert labels[0]["user"].startswith("u_")
    # someone else cannot rate this person's session, and a made-up session cannot be rated at all
    other = alexa_body(intent("BadPickIntent"), user="amzn1.ask.account.OTHER", attrs=attrs)
    asyncio.run(AS.handle_alexa(other))
    forged = alexa_body(intent("BadPickIntent"), attrs={"session_id": "nope", "last_url": "http://a"})
    asyncio.run(AS.handle_alexa(forged))
    assert len(O.labels_from_events(O.EventLog(FB.log_path()).read())) == 1
    nothing = asyncio.run(AS.handle_alexa(alexa_body(intent("BadPickIntent"))))
    assert "do not have a pick" in nothing["response"]["outputSpeech"]["text"]


def test_a_slow_pipeline_asks_the_person_to_retry_then_serves_the_finished_answer_without_redoing_it(monkeypatch):
    calls = []

    async def slow(texts, convs, user, *a, **k):
        calls.append(1)
        await asyncio.sleep(0.25)
        return fake_plan()
    monkeypatch.setattr(AS.engine, "plan", slow)
    monkeypatch.setenv("KNOTS_ALEXA_TIMEOUT_S", "0.05")

    async def scenario():
        first = await AS.handle_alexa(alexa_body(intent("KnotsPartnerIntent", topic="slow topic")))
        await asyncio.sleep(0.4)
        second = await AS.handle_alexa(alexa_body(intent("KnotsPartnerIntent", topic="Slow Topic")))
        return first, second
    first, second = asyncio.run(scenario())
    assert "longer than usual" in first["response"]["outputSpeech"]["text"]
    assert "Try T0" in second["response"]["outputSpeech"]["ssml"] and len(calls) == 1


def test_a_pipeline_failure_is_a_polite_answer_not_a_crash(monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("bedrock exploded")
    monkeypatch.setattr(AS.engine, "plan", boom)
    out = asyncio.run(AS.handle_alexa(alexa_body(intent("KnotsPartnerIntent", topic="x"))))
    assert "something went wrong" in out["response"]["outputSpeech"]["text"] and "exploded" not in json.dumps(out)


class FakeRequest:
    def __init__(self, headers, raw):
        self.headers, self._raw = headers, raw

    async def body(self):
        return self._raw


def test_the_http_route_verifies_before_it_does_any_work(pki, engine_calls):
    limiter = RateLimiter(per_key=0, overall=0)
    body = alexa_body(intent("KnotsPartnerIntent", topic="memecoins"))
    h, raw = signed(pki, body)

    async def fetch(_):
        return pki["chain"]
    ok = asyncio.run(AS.handle_http(FakeRequest({**h, "content-length": str(len(raw))}, raw), limiter, fetch, pki["roots"]))
    assert ok.status_code == 200 and b"Try T0" in ok.body and len(engine_calls) == 1
    bad = asyncio.run(AS.handle_http(FakeRequest(h, raw.replace(b"memecoins", b"anything")), limiter, fetch, pki["roots"]))
    assert bad.status_code == 400 and len(engine_calls) == 1                       # rejected: the pipeline never ran
    junk = asyncio.run(AS.handle_http(FakeRequest(h, b"not json"), limiter, fetch, pki["roots"]))
    assert junk.status_code == 400
    big = asyncio.run(AS.handle_http(FakeRequest({"content-length": str(10**7)}, b""), limiter, fetch, pki["roots"]))
    assert big.status_code == 413


def test_the_http_route_is_off_without_a_skill_id_and_rate_limits_after_verifying(pki, engine_calls, monkeypatch):
    body = alexa_body(intent("KnotsPartnerIntent", topic="x"))
    h, raw = signed(pki, body)

    async def fetch(_):
        return pki["chain"]
    tight = RateLimiter(per_key=1, window_s=60, overall=0)
    assert asyncio.run(AS.handle_http(FakeRequest(h, raw), tight, fetch, pki["roots"])).status_code == 200
    limited = asyncio.run(AS.handle_http(FakeRequest(h, raw), tight, fetch, pki["roots"]))
    assert limited.status_code == 200 and b"lot of requests" in limited.body and len(engine_calls) == 1
    monkeypatch.delenv("KNOTS_ALEXA_SKILL_ID")
    assert asyncio.run(AS.handle_http(FakeRequest(h, raw), tight, fetch, pki["roots"])).status_code == 503


# ------------------------------------------------------------------------------------------ identity
def test_hashed_ids_are_stable_namespaced_opaque_and_need_a_salt(monkeypatch):
    a = ID.hash_user("amzn1.ask.account.A", "alexa")
    assert a == ID.hash_user("amzn1.ask.account.A", "alexa") and a.startswith("u_") and "amzn1" not in a
    assert a != ID.hash_user("amzn1.ask.account.A", "mcp") and a != ID.hash_user("amzn1.ask.account.B", "alexa")
    monkeypatch.setenv("KNOTS_USER_SALT", "t" * 40)
    assert ID.hash_user("amzn1.ask.account.A", "alexa") != a                       # a different salt is a different pseudonym
    assert ID.hash_user("", "alexa") is None
    monkeypatch.delenv("KNOTS_USER_SALT")
    assert ID.hash_user("amzn1.ask.account.A", "alexa") is None


def test_a_caller_supplied_user_id_is_not_an_identity_unless_development_mode_is_on(monkeypatch):
    assert ID.resolve_user("someone-elses-id") is None
    monkeypatch.setenv("KNOTS_ALLOW_CLIENT_USER_ID", "1")
    assert ID.resolve_user("dev-user").startswith("u_") and ID.resolve_user(None) is None
    monkeypatch.setattr(ID, "current_user", lambda: "u_verified")
    assert ID.resolve_user("dev-user") == "u_verified"                              # the verified caller always wins


def _token(secret="k" * 40, **over):
    import jwt
    claims = {"sub": "alex", "aud": "https://knots.example.com/mcp", "iss": "https://issuer.example.com",
              "exp": int(time.time()) + 600, **over}
    return jwt.encode({k: v for k, v in claims.items() if v is not None}, secret, algorithm="HS256")


def test_jwt_verifier_accepts_only_valid_tokens_for_this_server(monkeypatch):
    v = ID.JwtVerifier(issuer="https://issuer.example.com", audience="https://knots.example.com/mcp", secret="k" * 40)
    ok = asyncio.run(v.verify_token(_token(scope="read write")))
    assert ok.subject == "alex" and ok.scopes == ["read", "write"] and ok.resource == "https://knots.example.com/mcp"
    for bad in (_token(secret="x" * 40), _token(aud="https://other.example.com"), _token(iss="https://evil.example.com"),
                _token(exp=int(time.time()) - 600), _token(sub=None), _token(exp=None), "garbage"):
        assert asyncio.run(v.verify_token(bad)) is None
    for kw in ({"secret": "short"}, {"audience": None}):
        with pytest.raises(ValueError):
            ID.JwtVerifier(**{"audience": "https://x", "secret": "k" * 40, **kw})
    with pytest.raises(ValueError):
        ID.JwtVerifier(audience="https://x")                                        # neither keys nor a secret


def test_an_unsigned_none_algorithm_token_is_rejected():
    import jwt
    v = ID.JwtVerifier(audience="https://knots.example.com/mcp", secret="k" * 40)
    forged = jwt.encode({"sub": "alex", "aud": "https://knots.example.com/mcp", "exp": int(time.time()) + 600}, None, algorithm="none")
    assert asyncio.run(v.verify_token(forged)) is None


# ------------------------------------------------------------------------------------------ rate limit and feedback
def test_rate_limit_is_per_key_per_window_and_shared_budget_is_not_eaten_by_refusals():
    r = RateLimiter(per_key=2, window_s=10, overall=3)
    assert r.allow("a", 0) and r.allow("a", 1) and not r.allow("a", 2)
    assert r.allow("b", 3) and not r.allow("c", 4)                                   # 3 allowed overall in this window
    assert r.allow("a", 11) and r.allow("a", 12)                                      # a new window
    assert RateLimiter(per_key=0, overall=0).allow("x") is True
    many = RateLimiter(per_key=1, window_s=10, overall=0)
    for i in range(20000):
        many.allow(f"k{i}", 0)
    assert len(many._counts) <= 10000


def test_feedback_needs_a_person_and_a_pick_that_was_shown_to_them():
    plan = fake_plan()
    sid = FB.new_session_id()
    assert FB.record_impression(plan, "u_a", sid) and FB.record_impression(plan, None, "anon-session")
    assert FB.record_feedback("u_a", sid, "http://b", False)["recorded"]
    assert not FB.record_feedback(None, sid, "http://b", True)["recorded"]            # anonymous: nobody to cap
    assert not FB.record_feedback("u_other", sid, "http://b", True)["recorded"]       # not their session
    assert not FB.record_feedback("u_a", sid, "http://not-shown", True)["recorded"]
    assert not FB.record_feedback("u_a", "anon-session", "http://a", True)["recorded"]  # that impression had no person
    assert not FB.record_feedback("u_a", "", "http://a", True)["recorded"]
    labels = O.labels_from_events(O.EventLog(FB.log_path()).read())
    assert [(x["user"], x["good"], x["url"]) for x in labels] == [("u_a", "n", "http://b")]
    assert FB.forget_events("u_a") == 2 and FB.forget_events("u_a") == 0 and FB.forget_events(None) == 0


# ------------------------------------------------------------------------------------------ the MCP server
@pytest.fixture
def srv(monkeypatch):
    import server
    return server


def test_the_server_registers_every_tool_a_health_route_and_the_alexa_route(srv):
    app = srv.create_mcp()
    names = {t.name for t in asyncio.run(app.list_tools())}
    assert names == {"knots_partner", "rate_pick", "get_recommendations", "extract_interests", "search_content",
                     "what_do_you_remember", "forget_me"}
    paths = {r.path for r in app.streamable_http_app().routes}
    assert {"/health", "/alexa", "/mcp"} <= paths


def test_binding_host_comes_from_the_environment_so_a_container_is_reachable(srv, monkeypatch):
    assert srv.create_mcp().settings.host == "127.0.0.1"
    monkeypatch.setenv("KNOTS_HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "9100")
    s = srv.create_mcp().settings
    assert (s.host, s.port) == ("0.0.0.0", 9100)


def test_knots_partner_logs_an_impression_returns_a_session_id_and_ignores_a_claimed_user(srv, engine_calls):
    out = asyncio.run(srv.knots_partner(["memes"], user_id="victim"))
    assert out["session_id"] and engine_calls[0][1] is None                           # the claimed id was not used
    ev = O.EventLog(FB.log_path()).read()
    assert ev[0]["type"] == "impression" and ev[0]["session_id"] == out["session_id"] and ev[0]["user"] == ""


def test_rate_pick_works_for_the_verified_caller_only(srv, engine_calls, monkeypatch):
    monkeypatch.setattr(ID, "current_user", lambda: "u_me")
    out = asyncio.run(srv.knots_partner(["memes"]))
    sid, url = out["session_id"], out["response"]["picks"][0]["url"]
    assert engine_calls[-1][1] == "u_me"
    assert asyncio.run(srv.rate_pick(sid, url, True))["recorded"]
    monkeypatch.setattr(ID, "current_user", lambda: "u_someone_else")
    assert not asyncio.run(srv.rate_pick(sid, url, False))["recorded"]
    monkeypatch.setattr(ID, "current_user", lambda: None)
    assert "verified identity" in asyncio.run(srv.rate_pick(sid, url, True))["reason"]


def test_memory_tools_refuse_unsigned_callers_and_forget_also_clears_the_event_log(srv, engine_calls, monkeypatch):
    assert asyncio.run(srv.what_do_you_remember("anyone"))["remembered_conversations"] == 0
    assert asyncio.run(srv.forget_me("anyone"))["deleted"] == 0
    monkeypatch.setattr(ID, "current_user", lambda: "u_me")
    asyncio.run(srv.knots_partner(["memes"]))
    forgotten = asyncio.run(srv.forget_me())
    assert forgotten["events_deleted"] == 1 and O.EventLog(FB.log_path()).read() == []


def test_over_the_rate_limit_a_tool_call_fails_politely(srv, engine_calls, monkeypatch):
    monkeypatch.setattr(srv, "limiter", RateLimiter(per_key=1, window_s=60, overall=0))
    asyncio.run(srv.knots_partner(["a"]))
    with pytest.raises(ValueError, match="Too many requests"):
        asyncio.run(srv.knots_partner(["b"]))


def test_over_http_health_is_open_the_mcp_endpoint_needs_a_token_and_hosts_are_checked(srv, monkeypatch):
    from starlette.testclient import TestClient
    monkeypatch.setenv("KNOTS_JWT_SECRET", "k" * 40)
    monkeypatch.setenv("KNOTS_JWT_AUDIENCE", "https://knots.example.com/mcp")
    monkeypatch.setenv("KNOTS_JWT_ISSUER", "https://issuer.example.com")
    monkeypatch.setenv("KNOTS_ALLOWED_HOSTS", "knots.example.com")
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}
    mcp_headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    with TestClient(srv.create_mcp().streamable_http_app(), base_url="https://knots.example.com") as c:
        assert c.get("/health").json() == {"ok": True}
        assert c.post("/mcp", json=init, headers=mcp_headers).status_code == 401
        assert c.post("/mcp", json=init, headers={**mcp_headers, "Authorization": "Bearer " + _token()}).status_code == 200
        assert c.post("/mcp", json=init, headers={**mcp_headers, "Authorization": "Bearer " + _token(aud="https://x")}).status_code == 401
        assert c.post("/alexa", content=b"{}").status_code == 400                      # reachable, but unsigned: refused
    with TestClient(srv.create_mcp().streamable_http_app(), base_url="https://evil.example.org") as c:
        assert c.post("/mcp", json=init, headers={**mcp_headers, "Authorization": "Bearer " + _token()}).status_code in (400, 403, 421)


# ------------------------------------------------------------------------------------------ skill package
def _load_script(name):
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    import sys
    sys.path.insert(0, str(root / "eval"))
    spec = importlib.util.spec_from_file_location(name, root / "eval" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_skill_package_matches_what_the_handler_understands():
    from pathlib import Path
    pkg = Path(__file__).resolve().parent.parent / "alexa" / "skill-package"
    model = json.loads((pkg / "interactionModels" / "custom" / "en-US.json").read_text())["interactionModel"]["languageModel"]
    manifest = json.loads((pkg / "skill.json").read_text())["manifest"]
    assert len(model["invocationName"].split()) >= 2 and model["invocationName"] == model["invocationName"].lower()
    intents = {i["name"]: i for i in model["intents"]}
    assert {"KnotsPartnerIntent", "GoodPickIntent", "BadPickIntent", "AMAZON.HelpIntent", "AMAZON.StopIntent",
            "AMAZON.CancelIntent", "AMAZON.FallbackIntent"} <= set(intents)
    seen = []
    for i in intents.values():
        declared = {s["name"] for s in i.get("slots", [])}
        for sample in i["samples"]:
            seen.append(sample.lower())
            slots = {w.strip("{}") for w in sample.split() if w.startswith("{")}
            assert slots <= declared, sample
            if slots:                                                  # a free-text slot needs words around it
                assert len(sample.split()) >= 2, sample
    assert len(seen) == len(set(seen)), "the same utterance appears under two intents"
    topic = intents["KnotsPartnerIntent"]["slots"][0]
    assert topic["name"] == "topic" and topic["type"] == "AMAZON.SearchQuery"       # what handle_alexa reads
    ep = manifest["apis"]["custom"]["endpoint"]
    assert ep["uri"].endswith("/alexa") and ep["sslCertificateType"] == "Trusted"
    for phrase in manifest["publishingInformation"]["locales"]["en-US"]["examplePhrases"]:
        assert phrase.startswith("Alexa, ") and model["invocationName"] in phrase


# ------------------------------------------------------------------------------------------ monitor
def _shown_and_rated(n_users, good_every):
    for i in range(n_users):
        sid = FB.new_session_id()
        FB.record_impression(fake_plan((f"http://p{i}a", f"http://p{i}b", f"http://p{i}c")), f"u_{i}", sid)
        for j, u in enumerate(("a", "b", "c")):
            FB.record_feedback(f"u_{i}", sid, f"http://p{i}{u}", (i + j) % good_every == 0)


def test_monitor_reports_a_decision_and_can_freeze_a_baseline(tmp_path, capsys):
    from knots import retrain
    MON = _load_script("monitor")
    assert MON.main([]) == 0 and "No events" in capsys.readouterr().out
    _shown_and_rated(30, 2)
    events = FB.event_log().read()
    state = {"trained_on": 0, "baseline": None}
    out = MON.report(events, state, 14)
    assert out["labels_total"] == 90 and out["usable_labels"] == 90 and out["metrics"]["lists"] == 30
    reg = retrain.Registry(tmp_path / "models")
    base = MON.freeze_baseline(reg, events)
    assert reg.state()["baseline"]["metrics"]["lists"] == 30 and reg.state()["champion"] is None and base["pool"]
    assert MON.main(["--registry", str(tmp_path / "models")]) == 0
    text = capsys.readouterr().out
    assert "30 lists" in text and "baseline: frozen" in text and "retrain:" in text
    assert MON.main(["--freeze-baseline"]) == 2                                        # needs somewhere to write it
    capsys.readouterr()
    assert MON.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out)["labels_total"] == 90
