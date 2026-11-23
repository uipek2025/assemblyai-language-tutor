"""Streaming providers behind one interface, so every vendor gets identical treatment.

Fairness rules baked in here (the usual ways evaluations get skewed):
- Same audio bytes, same 50 ms chunks, paced in real time. Uploading a file
  faster than real time changes how streaming models behave and hides latency.
- Trailing silence after speech, as a live mic would send, so endpointing
  happens naturally instead of being forced by closing the stream.
- Latency measured on the audio timeline (audio sent when the final arrived),
  identically for every provider.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlencode

import websockets


@dataclass
class Turn:
    order: int
    transcript: str
    words: list[dict]
    receipt_audio_ms: float
    language_code: str | None = None
    language_confidence: float | None = None
    forced: bool = False  # arrived only after we sent Terminate (a flush, not a real endpoint)

    def as_dict(self) -> dict:
        return {
            "order": self.order,
            "transcript": self.transcript,
            "receipt_audio_ms": self.receipt_audio_ms,
            "forced": self.forced,
            "language_code": self.language_code,
            "language_confidence": self.language_confidence,
        }


@dataclass
class StreamResult:
    provider: str
    model: str | None = None
    turns: list[Turn] = field(default_factory=list)
    partial_count: int = 0
    first_partial_audio_ms: float | None = None
    error: str | None = None

    @property
    def transcript(self) -> str:
        return " ".join(t.transcript for t in self.turns if t.transcript).strip()

    @property
    def words(self) -> list[dict]:
        return [w for t in self.turns for w in (t.words or [])]


class StreamingProvider(Protocol):
    name: str

    async def transcribe(self, pcm16: bytes, sample_rate: int, update: dict | None = None) -> StreamResult: ...


class AssemblyAIProvider:
    """AssemblyAI realtime STT over the v3 WebSocket API."""

    name = "assemblyai"

    def __init__(
        self,
        api_key: str,
        connect_params: dict | None = None,
        host: str = "streaming.assemblyai.com",
        chunk_ms: int = 50,
        trailing_silence_ms: int = 3000,
        realtime_factor: float = 1.0,
        url: str | None = None,  # override (tests use a local fake server)
        timeout_s: float = 20,
    ):
        self.api_key = api_key
        self.connect_params = dict(connect_params or {})
        self.host = host
        self.chunk_ms = chunk_ms
        self.trailing_silence_ms = trailing_silence_ms
        self.realtime_factor = realtime_factor
        self.url = url
        self.timeout_s = timeout_s

    def _url(self, sample_rate: int) -> str:
        params = {}
        for key, value in self.connect_params.items():
            if isinstance(value, (list, dict)):
                continue  # arrays go in UpdateConfiguration after Begin
            params[key] = str(value).lower() if isinstance(value, bool) else value
        params.update(sample_rate=sample_rate, encoding="pcm_s16le")
        base = self.url or f"wss://{self.host}/v3/ws"
        return f"{base}?{urlencode(params)}"

    async def transcribe(self, pcm16: bytes, sample_rate: int, update: dict | None = None) -> StreamResult:
        result = StreamResult(provider=self.name)
        sent_samples = 0
        terminate_sent = False

        try:
            async with websockets.connect(
                self._url(sample_rate),
                additional_headers={"Authorization": self.api_key},
                max_size=None,
                open_timeout=self.timeout_s,
            ) as ws:
                begin = json.loads(await asyncio.wait_for(ws.recv(), timeout=self.timeout_s))
                if begin.get("type") != "Begin":
                    raise RuntimeError(f"expected Begin, got {begin}")
                result.model = (begin.get("configuration") or {}).get("model") or self.connect_params.get("speech_model")
                if update:
                    await ws.send(json.dumps({"type": "UpdateConfiguration", **update}))

                async def send_audio() -> None:
                    nonlocal sent_samples, terminate_sent
                    chunk_bytes = int(sample_rate * self.chunk_ms / 1000) * 2
                    audio = pcm16 + bytes(int(sample_rate * self.trailing_silence_ms / 1000) * 2)
                    loop = asyncio.get_running_loop()
                    t0 = loop.time()
                    for i, offset in enumerate(range(0, len(audio), chunk_bytes)):
                        due = t0 + (i * self.chunk_ms / 1000) / self.realtime_factor
                        delay = due - loop.time()
                        if delay > 0:
                            await asyncio.sleep(delay)
                        chunk = audio[offset : offset + chunk_bytes]
                        await ws.send(chunk)
                        sent_samples += len(chunk) // 2
                    terminate_sent = True
                    await ws.send(json.dumps({"type": "Terminate"}))

                async def receive() -> None:
                    async for raw in ws:
                        if isinstance(raw, bytes):
                            continue
                        msg = json.loads(raw)
                        kind = msg.get("type")
                        audio_ms = sent_samples / sample_rate * 1000
                        if kind == "Turn":
                            if msg.get("end_of_turn"):
                                if any(t.order == msg.get("turn_order") for t in result.turns):
                                    continue  # one final per turn, whatever the formatting mode
                                result.turns.append(
                                    Turn(
                                        order=msg.get("turn_order", len(result.turns)),
                                        transcript=msg.get("transcript", ""),
                                        words=msg.get("words") or [],
                                        receipt_audio_ms=audio_ms,
                                        language_code=msg.get("language_code"),
                                        language_confidence=msg.get("language_confidence"),
                                        forced=terminate_sent,
                                    )
                                )
                            else:
                                result.partial_count += 1
                                if result.first_partial_audio_ms is None and msg.get("transcript"):
                                    result.first_partial_audio_ms = audio_ms
                        elif kind == "Termination":
                            return
                        elif msg.get("error"):
                            result.error = str(msg["error"])

                receiver = asyncio.create_task(receive())
                await send_audio()
                await asyncio.wait_for(receiver, timeout=self.timeout_s)
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
            result.error = result.error or f"{type(exc).__name__}: {exc}"
        return result


class IncumbentProvider:
    """Plug the customer's current provider in here, behind the same interface.

    Implement ``transcribe`` with the incumbent's streaming API. Keep the rules
    above: identical audio and pacing, the incumbent's *own recommended*
    settings (a fair fight is the only result their data team will trust),
    and the same region where possible.
    """

    name = "incumbent"

    def __init__(self, **_: object):
        pass

    async def transcribe(self, pcm16: bytes, sample_rate: int, update: dict | None = None) -> StreamResult:
        raise NotImplementedError("Add the incumbent provider's streaming client here.")


PROVIDERS = {"assemblyai": AssemblyAIProvider, "incumbent": IncumbentProvider}
