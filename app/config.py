"""Runtime settings.

Everything that a customer would tune during an evaluation lives here, so the
same values are used by the live app *and* by the eval harness.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


# Languages AssemblyAI-TTS-1 can speak. The API wants the full English name and
# does not auto-detect, which is exactly why the tutor splits every reply into
# per-language segments (see app/tutor.py).
TTS_LANGUAGES = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
}

# A friendly subset of the 32 languages the realtime STT models code-switch
# across. Languages outside TTS_LANGUAGES still work for listening; the browser
# falls back to its own speech synthesis for the tutor's voice.
UI_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "tr": "Turkish",
    "ja": "Japanese",
    "zh": "Mandarin",
    "ko": "Korean",
    "hi": "Hindi",
    "ar": "Arabic",
}

# Turn detection is punctuation-based on the current realtime models:
#   min_turn_silence  -> silence before a speculative "is this turn finished?" check
#   max_turn_silence  -> hard cap; the turn ends regardless of punctuation
# Learners hesitate far more than typical callers, and they hesitate *most*
# while attempting a foreign phrase. So the agent switches presets with the
# conversation state instead of using one global setting. These numbers are a
# starting point - the eval harness sweeps them on the customer's own audio.
TURN_PRESETS = {
    "conversation": {"min_turn_silence": 160, "max_turn_silence": 1600},
    "practice": {"min_turn_silence": 560, "max_turn_silence": 2400},
}

# Below this STT word confidence a matched word is flagged "unclear" in practice
# feedback. Confidence is only a proxy for intelligibility; calibrate it on
# labelled attempts (evals/) before trusting it.
LOW_CONFIDENCE = float(_env("GRADING_LOW_CONFIDENCE", "0.5"))


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default_factory=lambda: _env("ASSEMBLYAI_API_KEY"))
    # "global" routes to the nearest region; "eu" keeps audio, text and LLM
    # traffic inside the EU (a common requirement for EU learners).
    region: str = field(default_factory=lambda: _env("ASSEMBLYAI_REGION", "global").lower())
    speech_model: str = field(
        default_factory=lambda: _env("ASSEMBLYAI_SPEECH_MODEL", "assemblyai-stt-3-7-realtime-agent")
    )
    # Monologue model for the extra-credit pitch page (captions, not turn-taking).
    caption_model: str = field(
        default_factory=lambda: _env("ASSEMBLYAI_CAPTION_MODEL", "assemblyai-stt-3-7-realtime")
    )
    # The LLM chain, tried in order (see app/llm.py). Picked from the Gateway's own
    # published benchmarks for short voice turns:
    #   gemini-2.5-flash-lite  ~0.7 s p50, strict JSON schema, US+EU, cheapest  (paid accounts)
    #   qwen3-32B              ~0.5 s p50, strict JSON schema, US+EU            (paid accounts)
    #   gemma-4-31b            strict JSON schema, US only, slower tail         (also works on free keys)
    #   qwen3.5-4b-32k-fast    fastest, prompt-only JSON, weakest               (also works on free keys)
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gemini-2.5-flash-lite"))
    llm_fallback_models: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            m.strip()
            for m in _env("LLM_FALLBACK_MODELS", "qwen3-32B,gemma-4-31b,qwen3.5-4b-32k-fast").split(",")
            if m.strip()
        )
    )
    # At startup, spend a few tokens finding the first model in the chain this key
    # can call, so the learner's first turn doesn't pay for that discovery.
    llm_startup_probe: bool = field(default_factory=lambda: _env("LLM_STARTUP_PROBE", "1") not in ("0", "false", "no"))
    # One voice for every language keeps the tutor sounding like one person.
    # Leave empty to use each language's default voice.
    tts_voice_id: str = field(default_factory=lambda: _env("TTS_VOICE_ID", "aai-kyla"))
    # Only for local testing against a fake streaming server (tests/test_browser_e2e.py).
    streaming_url_override: str = field(default_factory=lambda: _env("ASSEMBLYAI_STREAMING_URL"))

    @property
    def is_eu(self) -> bool:
        return self.region == "eu"

    @property
    def streaming_ws_url(self) -> str:
        return self.streaming_url_override or f"wss://{self.streaming_host}/v3/ws"

    @property
    def streaming_host(self) -> str:
        return {
            "eu": "streaming.eu.assemblyai.com",
            "us": "streaming.us.assemblyai.com",
        }.get(self.region, "streaming.assemblyai.com")

    @property
    def llm_base(self) -> str:
        host = "llm-gateway.eu.assemblyai.com" if self.is_eu else "llm-gateway.assemblyai.com"
        return f"https://{host}/v1"

    @property
    def tts_base(self) -> str:
        return "https://tts.eu.assemblyai.com" if self.is_eu else "https://tts.us.assemblyai.com"
