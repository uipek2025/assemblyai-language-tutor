"""LLM Gateway client helpers: schema checks, defaults, prompt-only JSON, model memory."""
import httpx
import pytest

from app.config import Settings
from app.llm import LLMGateway, fill_defaults, missing_required, with_json_instruction
from app.pitch import ROAST_SCHEMA
from app.tutor import REPLY_SCHEMA


def test_missing_required_walks_nested_objects_and_arrays():
    assert missing_required({"segments": [], "practice_phrase": {"text": "", "lang": "", "pronunciation": "", "meaning": ""}}, REPLY_SCHEMA) == []
    assert missing_required({"segments": [{"text": "hola"}]}, REPLY_SCHEMA) == ["segments[0].lang", "practice_phrase"]
    assert missing_required({"segments": "hola", "practice_phrase": None}, REPLY_SCHEMA) == ["segments", "practice_phrase"]


def test_fill_defaults_gives_every_required_key_a_typed_empty_value():
    filled = fill_defaults({"score": 7}, ROAST_SCHEMA)
    assert filled["score"] == 7 and filled["missing"] == [] and filled["one_liner"] == ""
    assert missing_required(filled, ROAST_SCHEMA) == []
    assert fill_defaults(None, REPLY_SCHEMA)["practice_phrase"] == {"text": "", "lang": "", "pronunciation": "", "meaning": ""}


def test_json_instruction_goes_into_the_existing_system_prompt():
    msgs = [{"role": "system", "content": "You are Lumi."}, {"role": "user", "content": "hola"}]
    out = with_json_instruction(msgs, REPLY_SCHEMA)
    assert out[0]["content"].startswith("You are Lumi.") and "JSON Schema" in out[0]["content"]
    assert msgs[0]["content"] == "You are Lumi."  # the caller's list is not mutated
    assert with_json_instruction([{"role": "user", "content": "hi"}], REPLY_SCHEMA)[0]["role"] == "system"


@pytest.mark.asyncio
async def test_unavailable_models_are_retried_after_a_while(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("app.llm.time.monotonic", lambda: clock[0])
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
        llm = LLMGateway(Settings(api_key="k", llm_model="a", llm_fallback_models=("b",)), client, retry_unavailable_after_s=60)
        llm.mark_unavailable("a", "no access")
        assert llm.usable_models == ["b"]
        clock[0] += 61  # e.g. the customer added a card in the meantime
        assert llm.usable_models == ["a", "b"]


@pytest.mark.asyncio
async def test_bad_api_key_fails_fast_without_walking_the_chain():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, text="Invalid API key")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        llm = LLMGateway(Settings(api_key="bad", llm_model="a", llm_fallback_models=("b", "c")), client)
        with pytest.raises(Exception, match="rejected the API key"):
            await llm.chat_json([{"role": "user", "content": "hi"}], REPLY_SCHEMA, "tutor_reply")
    assert len(calls) == 1
