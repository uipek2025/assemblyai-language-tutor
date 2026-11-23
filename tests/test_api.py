"""Backend routes with AssemblyAI's HTTP APIs mocked at the transport layer."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


NO_ACCESS = (400, '{"metadata":{"errors":["Your account does not have access to this LLM Gateway model"]}}')


class FakeUpstream:
    def __init__(self, llm_replies=None, catalog=None):
        self.calls: list[httpx.Request] = []
        self.llm_replies = list(llm_replies or [])
        self.catalog = catalog or [{"id": "gemini-2.5-flash-lite"}, {"id": "gpt-4.1"}]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        url = str(request.url)
        if url.endswith("/v1/models"):
            return httpx.Response(200, json={"data": self.catalog})
        if "/v3/token" in url:
            assert request.headers["Authorization"] == "test-key"
            assert request.url.params["expires_in_seconds"] == "60"
            return httpx.Response(200, json={"token": "tmp-123", "expires_in_seconds": 60})
        if url.endswith("/chat/completions"):
            assert request.headers["authorization"] == "test-key"
            status, payload = self.llm_replies.pop(0)
            if status != 200:
                return httpx.Response(status, text=payload)
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})
        if url.endswith("/v1/speech"):
            body = json.loads(request.content)
            return httpx.Response(200, content=b"RIFF....WAVEfmt " + body["language"].encode(),
                                  headers={"content-type": "audio/wav", "server-timing": "ttfa;dur=88.5"})
        return httpx.Response(404, text=f"unexpected {url}")

    def bodies(self, suffix):
        return [json.loads(c.content) for c in self.calls if str(c.url).endswith(suffix)]


TEACH = {
    "segments": [
        {"lang": "en", "text": "In Spanish you'd say"},
        {"lang": "es", "text": "¿Dónde está la biblioteca?"},
        {"lang": "en", "text": "Now you try."},
    ],
    "practice_phrase": {"text": "¿Dónde está la biblioteca?", "lang": "es",
                        "pronunciation": "DOHN-deh eh-STAH lah bee-blee-oh-TEH-kah", "meaning": "Where is the library?"},
}


@pytest.fixture
def make_client():
    def _make(upstream, **settings):
        settings.setdefault("llm_startup_probe", False)
        app = create_app(Settings(api_key="test-key", **settings), transport=httpx.MockTransport(upstream.handler))
        return TestClient(app)
    return _make


def test_session_mints_token_and_returns_recipe(make_client):
    up = FakeUpstream()
    with make_client(up) as client:
        data = client.post("/api/session", json={"purpose": "tutor"}).json()
        health = client.get("/api/health").json()
    assert data["token"] == "tmp-123"
    assert data["ws_url"] == "wss://streaming.assemblyai.com/v3/ws"
    assert data["connect_params"]["speech_model"] == "assemblyai-stt-3-7-realtime-agent"
    assert data["turn_presets"]["practice"]["max_turn_silence"] > data["turn_presets"]["conversation"]["max_turn_silence"]
    assert "test-key" not in json.dumps(data)  # the API key never reaches the browser
    assert health["llm_model_listed"] is True


def test_tutor_teaches_with_structured_output(make_client):
    up = FakeUpstream([(200, TEACH)])
    with make_client(up) as client:
        r = client.post("/api/tutor", json={"transcript": "how do you say where is the library in spanish"})
    data = r.json()
    assert r.status_code == 200
    assert [s["lang"] for s in data["segments"]] == ["en", "es", "en"]
    assert data["practice"]["text"] == "¿Dónde está la biblioteca?"
    assert data["grade"] is None
    body = up.bodies("/chat/completions")[0]
    assert body["model"] == "gemini-2.5-flash-lite"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True


def test_practice_attempt_is_graded_in_code_and_grounds_the_llm(make_client):
    feedback = {"segments": [{"lang": "en", "text": "Close! Biblioteca, not librería."}], "practice_phrase": TEACH["practice_phrase"]}
    up = FakeUpstream([(200, feedback)])
    with make_client(up) as client:
        data = client.post("/api/tutor", json={
            "transcript": "dónde está la librería",
            "words": [{"text": w, "confidence": 0.9} for w in "dónde está la librería".split()],
            "practice": {"text": "¿Dónde está la biblioteca?", "lang": "es"},
            "stt_language": "es", "stt_language_confidence": 0.93,
        }).json()
    assert data["grade"]["verdict"] in ("close", "retry")
    prompt = up.bodies("/chat/completions")[0]["messages"][-1]["content"]
    assert "PRACTICE RESULT" in prompt and "biblioteca=MISSED" in prompt


def test_falls_back_when_structured_outputs_rejected(make_client):
    up = FakeUpstream([(400, '{"error": "response_format is invalid: unsupported"}'), (200, TEACH)])
    with make_client(up) as client:
        data = client.post("/api/tutor", json={"transcript": "hola"}).json()
    bodies = up.bodies("/chat/completions")
    assert "response_format" in bodies[0] and "response_format" not in bodies[1]
    assert data["meta"]["structured"] is False


def test_falls_back_to_next_model_when_one_is_unavailable(make_client):
    up = FakeUpstream([(404, "model not found"), (200, TEACH)])
    with make_client(up, llm_fallback_models=("gpt-4.1",)) as client:
        data = client.post("/api/tutor", json={"transcript": "hola"}).json()
    assert data["meta"]["model"] == "gpt-4.1"


def test_model_without_access_is_skipped_at_once_and_remembered(make_client):
    """Free AssemblyAI keys can only call a couple of Gateway models: don't burn a retry per turn."""
    up = FakeUpstream([NO_ACCESS, (200, TEACH), (200, TEACH)])
    with make_client(up, llm_fallback_models=("gemma-4-31b",)) as client:
        first = client.post("/api/tutor", json={"transcript": "hola"}).json()
        second = client.post("/api/tutor", json={"transcript": "hola"}).json()
        health = client.get("/api/health").json()
    models = [b["model"] for b in up.bodies("/chat/completions")]
    assert models == ["gemini-2.5-flash-lite", "gemma-4-31b", "gemma-4-31b"]  # no schema-less retry, no re-try next turn
    assert first["meta"]["model"] == second["meta"]["model"] == "gemma-4-31b"
    assert "gemini-2.5-flash-lite" in health["llm"]["unavailable"]
    assert health["llm"]["active"] == "gemma-4-31b"


def test_startup_probe_finds_a_usable_model_before_the_first_turn(make_client):
    up = FakeUpstream([NO_ACCESS, (200, {"ok": True}), (200, TEACH)])
    with make_client(up, llm_startup_probe=True, llm_fallback_models=("gemma-4-31b",)) as client:
        data = client.post("/api/tutor", json={"transcript": "hola"}).json()
    bodies = up.bodies("/chat/completions")
    assert [b["max_tokens"] for b in bodies[:2]] == [8, 8]  # the probe costs a few tokens
    assert bodies[2]["model"] == "gemma-4-31b"  # the learner's turn goes straight to a working model
    assert data["meta"]["model"] == "gemma-4-31b"


def test_no_usable_model_gives_an_actionable_error(make_client):
    up = FakeUpstream([NO_ACCESS])
    with make_client(up, llm_fallback_models=()) as client:
        r = client.post("/api/tutor", json={"transcript": "hola"})
        again = client.post("/api/tutor", json={"transcript": "hola"})
    assert r.status_code == 502 and "paid AssemblyAI account" in r.json()["error"]
    assert again.status_code == 502 and len(up.bodies("/chat/completions")) == 1  # fails fast, no new request


def test_catalog_routes_models_without_schema_support_to_prompt_only_json(make_client):
    catalog = [{"id": "qwen3.5-4b-32k-fast", "supported_parameters": ["max_tokens", "temperature"]}]
    up = FakeUpstream([(200, TEACH)], catalog=catalog)
    with make_client(up, llm_model="qwen3.5-4b-32k-fast", llm_fallback_models=()) as client:
        data = client.post("/api/tutor", json={"transcript": "hola"}).json()
    body = up.bodies("/chat/completions")[0]
    assert "response_format" not in body  # no wasted 400 first
    assert "JSON Schema" in body["messages"][0]["content"] and '"practice_phrase"' in body["messages"][0]["content"]
    assert data["meta"]["structured"] is False


def test_eu_region_skips_models_not_served_from_the_eu(make_client):
    catalog = [{"id": "gemini-2.5-flash-lite", "available_regions": ["us", "eu"]},
               {"id": "gemma-4-31b", "available_regions": ["us"]}]
    up = FakeUpstream([(500, "boom")], catalog=catalog)
    with make_client(up, region="eu", llm_fallback_models=("gemma-4-31b",)) as client:
        client.post("/api/tutor", json={"transcript": "hola"})
    calls = [c for c in up.calls if str(c.url).endswith("/chat/completions")]
    assert [json.loads(c.content)["model"] for c in calls] == ["gemini-2.5-flash-lite"]
    assert all(c.url.host == "llm-gateway.eu.assemblyai.com" for c in calls)


def test_reply_is_sanitised(make_client):
    messy = {
        "segments": [{"lang": "EN", "text": " Hi "}, {"lang": "en", "text": "there."}, {"lang": "xx", "text": "?"}, {"lang": "es", "text": ""}],
        "practice_phrase": {"text": "Bonjour", "lang": "fr", "pronunciation": "", "meaning": ""},
    }
    up = FakeUpstream([(200, messy)])
    with make_client(up) as client:
        data = client.post("/api/tutor", json={"transcript": "hi"}).json()
    assert data["segments"] == [{"lang": "en", "text": "Hi there. ?"}]  # merged, unknown lang -> native
    assert data["practice"] is None  # only the target language is practised


def test_tts_maps_language_caches_and_refuses_unsupported(make_client):
    up = FakeUpstream()
    with make_client(up) as client:
        first = client.post("/api/tts", json={"text": "Hola", "lang": "es"})
        second = client.post("/api/tts", json={"text": "Hola", "lang": "es"})
        japanese = client.post("/api/tts", json={"text": "こんにちは", "lang": "ja"})
    assert first.status_code == 200 and first.headers["content-type"] == "audio/wav"
    assert first.content.endswith(b"spanish")
    assert first.headers["x-tts-ttfa-ms"] == "88.5"
    assert second.headers["x-tts-cached"] == "true"
    assert len(up.bodies("/v1/speech")) == 1
    assert up.bodies("/v1/speech")[0]["voice_id"] == "aai-kyla"
    assert japanese.status_code == 422 and japanese.json()["fallback"] == "browser"


def test_greeting(make_client):
    hello = {"segments": [{"lang": "en", "text": "Hi, I'm Lumi."}, {"lang": "es", "text": "¡Hola!"}],
             "practice_phrase": {"text": "", "lang": "", "pronunciation": "", "meaning": ""}}
    up = FakeUpstream([(200, hello)])
    with make_client(up) as client:
        data = client.post("/api/tutor", json={"greet": True}).json()
        empty = client.post("/api/tutor", json={"transcript": "  "})
    assert data["segments"][1]["lang"] == "es" and data["practice"] is None
    assert empty.status_code == 400


ATTEMPT = {
    "transcript": "dónde está la librería",
    "practice": {"text": "¿Dónde está la biblioteca?", "lang": "es"},
    "stt_language": "es", "stt_language_confidence": 0.93,
}


def test_failed_attempt_keeps_practising_even_if_the_model_drops_the_phrase(make_client):
    reply = {"segments": [{"lang": "en", "text": "Almost! Library is"}, {"lang": "es", "text": "biblioteca"}],
             "practice_phrase": {"text": "", "lang": "", "pronunciation": "", "meaning": ""}}
    up = FakeUpstream([(200, reply)])
    with make_client(up) as client:
        data = client.post("/api/tutor", json=ATTEMPT).json()
    assert data["grade"]["verdict"] in ("close", "retry")
    assert data["practice"]["text"] == "¿Dónde está la biblioteca?"
    assert data["meta"]["practice_source"] == "carried"


def test_correct_attempt_ends_practice_even_if_the_model_repeats_the_phrase(make_client):
    up = FakeUpstream([(200, {**TEACH, "segments": [{"lang": "es", "text": "¡Perfecto!"}]})])
    with make_client(up) as client:
        data = client.post("/api/tutor", json={**ATTEMPT, "transcript": "dónde está la biblioteca"}).json()
    assert data["grade"]["verdict"] == "correct"
    assert data["practice"] is None


def test_phrase_is_inferred_when_a_prompt_only_model_ignores_the_contract(make_client):
    sloppy = {"segments": [{"lang": "en", "text": "You can say"}, {"lang": "es", "text": "Me gustaría un café, por favor."}]}
    up = FakeUpstream([(200, sloppy)])
    with make_client(up) as client:
        data = client.post("/api/tutor", json={"transcript": "how do I order a coffee"}).json()
    assert data["meta"]["missing"] == ["practice_phrase"]
    assert data["practice"]["text"] == "Me gustaría un café, por favor."
    assert data["meta"]["practice_source"] == "inferred"


def test_greeting_never_starts_practice_by_inference(make_client):
    up = FakeUpstream([(200, {"segments": [{"lang": "es", "text": "¡Hola!"}, {"lang": "en", "text": "I'm Lumi."}]})])
    with make_client(up) as client:
        data = client.post("/api/tutor", json={"greet": True}).json()
    assert data["practice"] is None


def test_roast_fills_keys_a_fallback_model_dropped(make_client):
    up = FakeUpstream([(200, {"score": 6, "spoken_verdict": "Lead with the customer."})])
    with make_client(up) as client:
        data = client.post("/api/roast", json={"transcript": "We build voice agents", "stats": {"wpm": 150}}).json()
    assert data["score"] == 6 and data["missing"] == [] and data["cut_this"] == ""
