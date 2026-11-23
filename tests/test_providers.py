"""The harness's streaming client, end to end against a local protocol fake."""
import numpy as np
import pytest

from evals.metrics import speech_end_ms, turn_outcome
from evals.providers import AssemblyAIProvider
from tests.fake_assemblyai import API_KEY, FakeAssemblyAI

SR = 16000


def tone(seconds):
    t = np.arange(int(SR * seconds)) / SR
    return 0.5 * np.sin(2 * np.pi * 220 * t)


def pcm(*parts):
    audio = np.concatenate(parts)
    return audio, (audio * 32767).astype("<i2").tobytes()


def provider(url, max_silence, **kw):
    return AssemblyAIProvider(
        api_key=kw.pop("api_key", API_KEY),
        connect_params={"speech_model": "assemblyai-stt-3-7-realtime-agent", "max_turn_silence": max_silence,
                        "language_detection": True},
        url=url,
        trailing_silence_ms=kw.pop("trailing", 2000),
        realtime_factor=kw.pop("speed", 5.0),
    )


async def test_single_turn_latency_and_update_configuration():
    audio, data = pcm(np.zeros(SR // 4), tone(1.0))
    async with FakeAssemblyAI(["dónde está la biblioteca"]) as fake:
        result = await provider(fake.url, 600).transcribe(data, SR, {"language_codes": ["en", "es"]})

    assert result.error is None
    assert result.model == "assemblyai-stt-3-7-realtime-agent"
    assert fake.updates == [{"type": "UpdateConfiguration", "language_codes": ["en", "es"]}]
    assert fake.connect_params[0]["language_detection"] == "true"  # bools sent the way the API expects
    assert result.transcript == "dónde está la biblioteca"
    assert len(result.turns) == 1 and not result.turns[0].forced
    assert result.partial_count > 0

    outcome = turn_outcome([t.as_dict() for t in result.turns], speech_end_ms(audio, SR))
    assert outcome["premature"] == 0 and not outcome["missed"]
    assert outcome["eot_latency_ms"] == pytest.approx(600, abs=150)


async def test_hesitation_gets_cut_off_with_impatient_turn_detection():
    # "Dónde ... (0.8 s pause) ... está la biblioteca" - the learner isn't done.
    audio, data = pcm(np.zeros(SR // 4), tone(0.5), np.zeros(int(SR * 0.8)), tone(0.6))
    end = speech_end_ms(audio, SR)

    async with FakeAssemblyAI(["dónde", "está la biblioteca"]) as fake:
        impatient = await provider(fake.url, 500).transcribe(data, SR)
    async with FakeAssemblyAI(["dónde está la biblioteca"]) as fake:
        patient = await provider(fake.url, 1200).transcribe(data, SR)

    cut = turn_outcome([t.as_dict() for t in impatient.turns], end)
    ok = turn_outcome([t.as_dict() for t in patient.turns], end)
    assert cut["premature"] == 1  # what a sweep of max_turn_silence makes visible
    assert ok["premature"] == 0
    assert ok["eot_latency_ms"] > cut["eot_latency_ms"]  # ...and the price of patience


async def test_turn_only_closed_by_terminate_is_flagged_as_forced():
    audio, data = pcm(np.zeros(SR // 4), tone(0.6))
    async with FakeAssemblyAI(["hola"]) as fake:
        result = await provider(fake.url, 5000, trailing=500).transcribe(data, SR)
    assert result.turns and result.turns[0].forced
    assert turn_outcome([t.as_dict() for t in result.turns], speech_end_ms(audio, SR))["missed"]


async def test_bad_key_is_reported_not_raised():
    _, data = pcm(tone(0.2))
    async with FakeAssemblyAI(["x"]) as fake:
        result = await provider(fake.url, 600, api_key="wrong").transcribe(data, SR)
    assert result.error
