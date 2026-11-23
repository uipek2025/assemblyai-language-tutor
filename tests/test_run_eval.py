"""The eval CLI end to end: WAV files + manifest -> streaming -> metrics -> report."""
import json

import numpy as np
import soundfile as sf

from evals.run_eval import evaluate, expand_sweep, item_settings, write_report
from tests.fake_assemblyai import API_KEY, FakeAssemblyAI

SR = 16000


def tone(seconds):
    t = np.arange(int(SR * seconds)) / SR
    return 0.5 * np.sin(2 * np.pi * 220 * t)


def silence(seconds):
    return np.zeros(int(SR * seconds))


def write_set(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    clips = {
        "q01": np.concatenate([silence(0.2), tone(1.2)]),
        "p04": np.concatenate([silence(0.2), tone(0.4), silence(0.8), tone(0.6)]),  # hesitation
        "p05": np.concatenate([silence(0.2), tone(1.0)]),
    }
    for cid, audio in clips.items():
        sf.write(audio_dir / f"{cid}.wav", audio, SR, subtype="PCM_16")
    items = [
        {"id": "q01", "audio": "audio/q01.wav", "reference": "what does me gustaría mean",
         "target_spans": ["me gustaría"], "expected_language": "en", "tags": ["question", "code_switch"]},
        {"id": "p04", "audio": "audio/p04.wav", "reference": "dónde está la biblioteca",
         "target": "¿Dónde está la biblioteca?", "label": "correct", "agent_context": "Now you try: ¿Dónde está la biblioteca?",
         "expected_language": "es", "tags": ["practice", "hesitation"]},
        {"id": "p05", "audio": "audio/p05.wav", "reference": "dónde está la librería",
         "target": "¿Dónde está la biblioteca?", "label": "incorrect", "expected_language": "es", "tags": ["practice", "wrong_word"]},
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", encoding="utf-8")
    return manifest


def config(url):
    return {
        "name": "learner",
        "connect_params": {"speech_model": "assemblyai-stt-3-7-realtime-agent", "max_turn_silence": 600},
        "practice_overrides": {"max_turn_silence": 1500},
        "language_codes": ["en", "es"],
        "use_agent_context": True,
        "url": url,
        "realtime_factor": 5,
    }


def test_item_settings_mirror_the_live_app():
    cfg = {**config("ws://x"), "boost_target_keyterms": True}
    params, update = item_settings({"tags": ["practice"], "target": "La cuenta, por favor.", "agent_context": "Try it"}, cfg)
    assert params["max_turn_silence"] == 1500  # practice preset
    assert update["language_codes"] == ["en", "es"] and update["agent_context"] == "Try it"
    assert update["keyterms_prompt"][0] == "La cuenta por favor"


def test_sweep_expansion():
    variants = expand_sweep("learner", config("ws://x"), "practice.max_turn_silence=500,2400")
    assert [v[1]["practice_overrides"]["max_turn_silence"] for v in variants] == [500, 2400]
    assert expand_sweep("learner", config("ws://x"), "boost_target_keyterms=true")[0][1]["boost_target_keyterms"] is True


async def test_end_to_end_with_sweep(tmp_path):
    manifest = write_set(tmp_path)
    scripts = [
        # sweep value 500 ms: the hesitating practice attempt gets cut in two
        ["what does me gustaría mean"], ["dónde", "está la biblioteca"], ["dónde está la librería"],
        # sweep value 1500 ms: patient enough
        ["what does me gustaría mean"], ["dónde está la biblioteca"], ["dónde está la librería"],
    ]
    async with FakeAssemblyAI([], per_connection=scripts) as fake:
        results = await evaluate(
            manifest, [config(fake.url)], "practice.max_turn_silence=500,1500", API_KEY, concurrency=1
        )

    (impatient, rows_a, _), (patient, rows_b, _) = results.values()
    assert impatient["errors"] == patient["errors"] == 0
    assert impatient["cutoff_rate_hesitation"] == 1.0 and patient["cutoff_rate_hesitation"] == 0.0
    assert patient["eot_p50_ms"] is not None
    assert patient["span_accuracy"] == 1.0
    assert patient["false_accept_rate"] == 0.0  # "librería" was not praised
    assert patient["false_reject_rate"] == 0.0
    assert patient["wer"] == 0.0
    # the agent's last line was passed as context only where the manifest had one
    assert sum("agent_context" in u for u in fake.updates) == 2

    report = write_report(tmp_path / "out", results)
    text = report.read_text(encoding="utf-8")
    assert "Premature cut-off rate" in text and "Smoke test only" in text
    assert (tmp_path / "out" / "items.csv").exists()
