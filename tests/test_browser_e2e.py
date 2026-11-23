"""Full browser test: real Chrome, fake microphone, fake AssemblyAI.

Chrome plays a speech-like WAV as the microphone. The real front end captures
it with the AudioWorklet, streams it to a local fake of the realtime
WebSocket (authenticated with the temp token), receives turns, calls the real
FastAPI backend (whose AssemblyAI HTTP calls are mocked) and plays TTS audio.
Skipped automatically when Chrome or Playwright isn't installed.
"""
from __future__ import annotations

import asyncio
import io
import json
import shutil
import socket
import threading
import time
import wave

import httpx
import numpy as np
import pytest
import uvicorn

from app.config import Settings
from app.main import create_app
from tests.fake_assemblyai import FakeAssemblyAI

sync_api = pytest.importorskip("playwright.sync_api")
CHROME = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
pytestmark = pytest.mark.skipif(not CHROME, reason="needs a local Chrome/Chromium")

TEACH = {
    "segments": [
        {"lang": "en", "text": "In Spanish you'd say"},
        {"lang": "es", "text": "¿Dónde está la biblioteca?"},
        {"lang": "en", "text": "Now you try."},
    ],
    "practice_phrase": {"text": "¿Dónde está la biblioteca?", "lang": "es", "pronunciation": "DOHN-deh eh-STAH",
                        "meaning": "Where is the library?"},
}
HELLO = {"segments": [{"lang": "en", "text": "Hi, I'm Lumi."}, {"lang": "es", "text": "¡Hola!"}],
         "practice_phrase": {"text": "", "lang": "", "pronunciation": "", "meaning": ""}}
FEEDBACK = {"segments": [{"lang": "en", "text": "Close. Say it again:"}, {"lang": "es", "text": "¿Dónde está la biblioteca?"}],
            "practice_phrase": TEACH["practice_phrase"]}


def wav_bytes(signal: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def fake_mic_wav(path, sr=48000):
    """1.4 s of vowel-like, syllable-modulated sound, then 3.2 s of silence (Chrome loops it)."""
    t = np.arange(int(sr * 1.4)) / sr
    voice = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate([180, 360, 720, 1080]))
    voice *= 0.5 * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)) / 2
    path.write_bytes(wav_bytes(np.concatenate([voice, np.zeros(int(sr * 3.2))]), sr))


def upstream_handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if url.endswith("/v1/models"):
        return httpx.Response(200, json={"data": [{"id": "gemini-2.5-flash-lite"}]})
    if "/v3/token" in url:
        return httpx.Response(200, json={"token": "tmp-123", "expires_in_seconds": 60})
    if url.endswith("/chat/completions"):
        last = json.loads(request.content)["messages"][-1]["content"]
        reply = HELLO if "Session just started" in last else FEEDBACK if "PRACTICE RESULT" in last else TEACH
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(reply)}}]})
    if url.endswith("/v1/speech"):
        sr = 24000
        t = np.arange(int(sr * 0.25)) / sr
        return httpx.Response(200, content=wav_bytes(0.2 * np.sin(2 * np.pi * 440 * t), sr),
                              headers={"content-type": "audio/wav"})
    return httpx.Response(404)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def stack(tmp_path):
    script = ["how do you say where is the library in spanish", "dónde está la librería", "dónde está la biblioteca"] * 4
    fake = FakeAssemblyAI(script, speech_threshold=300)
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run_fake():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(fake.__aenter__())
        ready.set()
        loop.run_forever()

    threading.Thread(target=run_fake, daemon=True).start()
    ready.wait(5)

    port = free_port()
    app = create_app(Settings(api_key="test-key", streaming_url_override=fake.url),
                     transport=httpx.MockTransport(upstream_handler))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)

    mic = tmp_path / "mic.wav"
    fake_mic_wav(mic)
    yield f"http://127.0.0.1:{port}", fake, mic
    server.should_exit = True
    loop.call_soon_threadsafe(loop.stop)


def test_tutor_loop_in_real_chrome(stack):
    base, fake, mic = stack
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME,
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={mic}",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(base)
        page.click("#start")

        page.wait_for_selector(".bubble.tutor mark", timeout=15000)  # greeting, Spanish segment highlighted
        page.wait_for_selector(".bubble.user", timeout=20000)  # first learner turn from the fake mic
        page.wait_for_function(
            "document.getElementById('phraseText').textContent.includes('biblioteca')", timeout=20000
        )
        page.wait_for_selector("#chips .chip", timeout=30000)  # a practice attempt was graded

        model = page.text_content("#hud-model")
        eot = page.text_content("#hud-eot")
        verdict = page.text_content("#verdict")
        statuses = page.eval_on_selector_all("#chips .chip", "els => els.map(e => e.className)")
        page.click("#start")  # stop
        browser.close()

    assert not errors, errors
    assert model == "assemblyai-stt-3-7-realtime-agent"
    assert eot.endswith("ms")
    assert any(s in verdict for s in ("Nailed it", "Almost", "try again"))
    assert len(statuses) == 4
    params = fake.connect_params[0]
    assert params["token"] == "tmp-123" and params["encoding"] == "pcm_s16le" and params["language_detection"] == "true"
    assert int(params["sample_rate"]) in (16000, 44100, 48000)
    assert {"type": "UpdateConfiguration", "language_codes": ["en", "es"]} in fake.updates
    assert any(u.get("max_turn_silence") == 2400 for u in fake.updates)  # practice preset applied
    assert any("agent_context" in u for u in fake.updates)  # Lumi's words given to the STT as context
    assert fake.audio_chunks > 20
