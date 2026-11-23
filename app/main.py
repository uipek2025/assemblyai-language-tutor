"""FastAPI server.

The server is deliberately thin and stateless:
- It mints short-lived streaming tokens so the API key never reaches the
  browser. Audio then goes browser -> AssemblyAI directly (no extra hop).
- It grades practice attempts deterministically and calls LLM Gateway.
- It proxies TTS-1 (keeping the key server-side) with a small cache.
Conversation history lives in the browser and is sent with each turn, so any
replica can serve any request.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import TTS_LANGUAGES, TURN_PRESETS, UI_LANGUAGES, Settings
from .grading import grade_attempt, looks_like_question
from .llm import LLMGateway, LLMGatewayError, fill_defaults
from .pitch import ROAST_SCHEMA, roast_messages
from .tts import TextToSpeech, TTSError, TTSUnsupportedLanguage
from .tutor import REPLY_SCHEMA, build_messages, greeting_messages, sanitize_reply

log = logging.getLogger("lumi")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


# ---------- request models ----------
class Word(BaseModel):
    text: str = ""
    start: float | None = None
    end: float | None = None
    confidence: float | None = None


class PracticePhrase(BaseModel):
    text: str
    lang: str = ""


class TutorRequest(BaseModel):
    native_language: str = "en"
    target_language: str = "es"
    transcript: str = ""
    greet: bool = False  # first turn of a session: Lumi says hello
    words: list[Word] = Field(default_factory=list)
    stt_language: str | None = None
    stt_language_confidence: float | None = None
    practice: PracticePhrase | None = None
    history: list[dict[str, str]] = Field(default_factory=list)


class TTSRequest(BaseModel):
    text: str
    lang: str


class RoastRequest(BaseModel):
    transcript: str
    stats: dict[str, Any] = Field(default_factory=dict)


class SessionRequest(BaseModel):
    purpose: str = "tutor"  # tutor | pitch | record


def create_app(settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http = httpx.AsyncClient(transport=transport, timeout=30)
        app.state.llm = LLMGateway(settings, app.state.http)
        app.state.tts = TextToSpeech(settings, app.state.http)
        app.state.llm_model_listed = None
        try:  # non-fatal sanity check: is the configured LLM offered, and how should we call it?
            catalog = await app.state.llm.catalog()
            available = [m["id"] for m in catalog]
            app.state.llm_model_listed = settings.llm_model in available
            app.state.llm.apply_catalog(catalog)
            if available and not app.state.llm_model_listed:
                log.warning("LLM_MODEL %r not in LLM Gateway model list. Some available: %s",
                            settings.llm_model, ", ".join(available[:15]))
        except Exception as exc:  # noqa: BLE001 - startup must not fail on this
            log.info("Could not list LLM Gateway models (%s); continuing", exc)
        if settings.llm_startup_probe and settings.api_key:
            try:
                active = await app.state.llm.probe()
            except Exception as exc:  # noqa: BLE001
                active = None
                log.info("LLM probe failed (%s); continuing", exc)
            if active:
                mode = "prompt-only JSON" if active in app.state.llm.no_schema else "strict JSON schema"
                log.info("LLM Gateway: answering with %s (%s)", active, mode)
            elif app.state.llm.unavailable and not app.state.llm.usable_models:
                log.error("LLM Gateway: no model in the chain is usable with this key. %s",
                          "Most Gateway models need a paid AssemblyAI account.")
        yield
        await app.state.http.aclose()

    app = FastAPI(title="Lumi — AssemblyAI language tutor", lifespan=lifespan)

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "api_key_configured": bool(settings.api_key),
            "region": settings.region,
            "speech_model": settings.speech_model,
            "llm_model": settings.llm_model,
            "llm_model_listed": app.state.llm_model_listed,
            "llm": app.state.llm.status(),
            "tts_voice_id": app.state.tts.voice_id or "default",
        }

    @app.post("/api/session")
    async def session(req: SessionRequest) -> dict[str, Any]:
        """Mint a single-use streaming token and return the connection recipe."""
        if not settings.api_key:
            raise HTTPException(500, "ASSEMBLYAI_API_KEY is not set on the server")
        try:
            resp = await app.state.http.get(
                f"https://{settings.streaming_host}/v3/token",
                params={"expires_in_seconds": 60, "max_session_duration_seconds": 1800},
                headers={"Authorization": settings.api_key},
                timeout=10,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Could not reach AssemblyAI token endpoint: {exc!r}") from exc
        if resp.status_code != 200:
            raise HTTPException(502, f"Token request failed ({resp.status_code}): {resp.text[:300]}")

        if req.purpose == "pitch":
            model, params = settings.caption_model, {"min_turn_silence": 560, "max_turn_silence": 2560}
        else:
            model, params = settings.speech_model, dict(TURN_PRESETS["conversation"])
        return {
            "token": resp.json()["token"],
            "ws_url": settings.streaming_ws_url,
            "connect_params": {
                "speech_model": model,
                "encoding": "pcm_s16le",
                "language_detection": "true",
                **params,
            },
            "turn_presets": TURN_PRESETS,
            "tts_languages": sorted(TTS_LANGUAGES),
            "ui_languages": UI_LANGUAGES,
        }

    @app.post("/api/tutor")
    async def tutor(req: TutorRequest) -> JSONResponse:
        native = req.native_language[:2].lower()
        target = req.target_language[:2].lower()
        transcript = req.transcript.strip()
        words = [w.model_dump() for w in req.words]
        if not transcript and not req.greet:
            raise HTTPException(400, "Empty transcript")

        grade = None
        practice_text = req.practice.text.strip() if req.practice and req.practice.text.strip() else None
        if req.greet:
            messages = greeting_messages(native, target)
        else:
            if practice_text:
                grade = grade_attempt(transcript, practice_text, words)
                if grade.verdict != "correct" and looks_like_question(
                    req.stt_language, req.stt_language_confidence, native, grade
                ):
                    grade.verdict = "not_an_attempt"
            messages = build_messages(native, target, req.history, transcript, practice_text, grade, req.stt_language)
        try:
            raw, meta = await app.state.llm.chat_json(messages, REPLY_SCHEMA, "tutor_reply")
        except LLMGatewayError as exc:
            log.error("LLM Gateway failed: %s", exc)
            return JSONResponse(
                status_code=502,
                content={"error": str(exc), "grade": grade.to_dict() if grade else None},
            )
        reply = sanitize_reply(
            raw, native, target,
            current_practice={"text": practice_text, "lang": target} if practice_text else None,
            verdict=grade.verdict if grade else None,
            greet=req.greet,
        )
        meta["practice_source"] = reply.pop("practice_source")
        return JSONResponse({**reply, "grade": grade.to_dict() if grade else None, "meta": meta})

    @app.post("/api/tts")
    async def tts(req: TTSRequest) -> Response:
        try:
            audio, meta = await app.state.tts.synthesize(req.text, req.lang)
        except TTSUnsupportedLanguage:
            return JSONResponse(status_code=422, content={"fallback": "browser", "reason": "language not in TTS-1"})
        except TTSError as exc:
            log.warning("TTS failed: %s", exc)
            return JSONResponse(status_code=502, content={"fallback": "browser", "reason": str(exc)})
        headers = {"X-TTS-Cached": str(meta["cached"]).lower(), "Cache-Control": "no-store"}
        if meta.get("server_ttfa_ms") is not None:
            headers["X-TTS-TTFA-Ms"] = str(meta["server_ttfa_ms"])
        return Response(content=audio, media_type="audio/wav", headers=headers)

    @app.post("/api/roast")
    async def roast(req: RoastRequest) -> JSONResponse:
        if not req.transcript.strip():
            raise HTTPException(400, "Empty pitch")
        try:
            raw, meta = await app.state.llm.chat_json(
                roast_messages(req.transcript.strip(), req.stats), ROAST_SCHEMA, "pitch_feedback", max_tokens=600
            )
        except LLMGatewayError as exc:
            return JSONResponse(status_code=502, content={"error": str(exc)})
        # Prompt-only fallback models sometimes drop keys; the page should never crash on that.
        return JSONResponse({**fill_defaults(raw, ROAST_SCHEMA), "meta": meta})

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
