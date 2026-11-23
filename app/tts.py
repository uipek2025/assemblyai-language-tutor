"""AssemblyAI-TTS-1 client.

TTS-1 does not detect language: every request must say which language the text
is in, and any voice can speak all supported languages. That pairs naturally
with the tutor's per-language reply segments: one consistent tutor voice that
pronounces "¿Dónde está la biblioteca?" as Spanish and the explanation around
it as English.
"""
from __future__ import annotations

import logging
import re
import time
from collections import OrderedDict

import httpx

from .config import TTS_LANGUAGES, Settings

log = logging.getLogger("lumi.tts")


class TTSUnsupportedLanguage(ValueError):
    pass


class TTSError(RuntimeError):
    pass


class TextToSpeech:
    def __init__(self, settings: Settings, client: httpx.AsyncClient, cache_size: int = 256):
        self.settings = settings
        self.client = client
        self.voice_id = settings.tts_voice_id
        self._cache: OrderedDict[tuple[str, str, str], tuple[bytes, dict]] = OrderedDict()
        self._cache_size = cache_size

    @staticmethod
    def supports(lang: str) -> bool:
        return (lang or "")[:2].lower() in TTS_LANGUAGES

    async def synthesize(self, text: str, lang: str) -> tuple[bytes, dict]:
        lang = (lang or "")[:2].lower()
        if lang not in TTS_LANGUAGES:
            raise TTSUnsupportedLanguage(lang)
        text = text.strip()[:5000]
        if not text:
            raise TTSError("empty text")

        # Learners replay phrases constantly ("say it again", "slower"). Caching
        # identical requests saves latency and money (TTS bills per character).
        key = (text, lang, self.voice_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            audio, meta = self._cache[key]
            return audio, {**meta, "cached": True}

        audio, meta = await self._request(text, lang, self.voice_id)
        self._cache[key] = (audio, meta)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return audio, meta

    async def _request(self, text: str, lang: str, voice_id: str) -> tuple[bytes, dict]:
        if not self.settings.api_key:
            raise TTSError("ASSEMBLYAI_API_KEY is not set")
        body = {"text": text, "language": TTS_LANGUAGES[lang], "container": "wav", "stream": False}
        if voice_id:
            body["voice_id"] = voice_id
        headers = {"Authorization": self.settings.api_key, "Content-Type": "application/json"}

        started = time.perf_counter()
        try:
            resp = await self.client.post(f"{self.settings.tts_base}/v1/speech", headers=headers, json=body, timeout=20)
        except httpx.HTTPError as exc:
            raise TTSError(f"network error: {exc!r}") from exc

        if resp.status_code == 422 and voice_id and "voice" in resp.text.lower():
            # Configured voice not available on this account/region: fall back
            # to the language's default voice rather than going silent.
            log.warning("TTS voice %r refused (%s); falling back to default voice", voice_id, resp.text[:200])
            self.voice_id = ""
            return await self._request(text, lang, "")

        if resp.status_code != 200:
            raise TTSError(f"HTTP {resp.status_code}: {resp.text[:300]}")

        ttfa = None
        match = re.search(r"ttfa;dur=([\d.]+)", resp.headers.get("server-timing", ""))
        if match:
            ttfa = float(match.group(1))
        meta = {
            "tts_ms": round((time.perf_counter() - started) * 1000),
            "server_ttfa_ms": ttfa,
            "voice": resp.headers.get("x-tts-voice-id", voice_id or "default"),
            "cached": False,
        }
        return resp.content, meta
