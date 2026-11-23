"""A tiny stand-in for AssemblyAI's v3 realtime WebSocket, for offline tests.

It follows the documented protocol shape (Begin -> partial/final Turn ->
Termination, UpdateConfiguration, Terminate) and a crude punctuation-free turn
detector: a turn ends after `max_turn_silence` ms of silence. Good enough to
test the harness's pacing, timing maths and turn bookkeeping without a key.
"""
from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import numpy as np
import websockets

API_KEY = "test-key"
TEMP_TOKEN = "tmp-123"


class FakeAssemblyAI:
    def __init__(self, script: list[str], per_connection: list[list[str]] | None = None, speech_threshold: int = 1000):
        self.script = list(script)  # transcript text for each successive turn
        self.per_connection = per_connection  # optional: a different script per connection
        self.speech_threshold = speech_threshold
        self.updates: list[dict] = []
        self.connect_params: list[dict] = []
        self.audio_chunks = 0
        self.server = None

    async def __aenter__(self):
        self.server = await websockets.serve(self.handler, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}/v3/ws"
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()

    async def handler(self, ws):
        query = {k: v[0] for k, v in parse_qs(urlparse(ws.request.path).query).items()}
        # Server-side clients use the API key header; browsers use a temporary token.
        if ws.request.headers.get("Authorization") != API_KEY and query.get("token") != TEMP_TOKEN:
            await ws.close(1008, "Unauthorized")
            return
        script = self.per_connection[len(self.connect_params)] if self.per_connection else self.script
        self.connect_params.append(query)
        sr = int(query["sample_rate"])
        max_silence = int(query.get("max_turn_silence", 1280))
        await ws.send(json.dumps({"type": "Begin", "id": "fake", "expires_at": 0,
                                  "configuration": {"model": query.get("speech_model")}}))

        received = 0  # samples
        speech_start = speech_end = None
        silence_ms = 0.0
        turn = 0

        async def emit(final: bool):
            nonlocal speech_start, speech_end, turn
            text = script[turn] if turn < len(script) else ""
            tokens = text.split()
            span = max(1.0, (speech_end or 0) - (speech_start or 0))
            words = [
                {
                    "text": tok,
                    "start": round(speech_start + span * i / len(tokens)),
                    "end": round(speech_start + span * (i + 1) / len(tokens)),
                    "confidence": 0.9,
                    "word_is_final": final,
                }
                for i, tok in enumerate(tokens)
            ]
            await ws.send(json.dumps({
                "type": "Turn", "turn_order": turn, "end_of_turn": final, "turn_is_formatted": final,
                "transcript": text, "utterance": text if final else "", "words": words,
                "end_of_turn_confidence": 1.0 if final else 0.0,
                "language_code": "es" if final else None, "language_confidence": 0.9 if final else None,
            }))
            if final:
                turn += 1
                speech_start = speech_end = None

        async for message in ws:
            if isinstance(message, str):
                msg = json.loads(message)
                if msg["type"] == "UpdateConfiguration":
                    self.updates.append(msg)
                    max_silence = int(msg.get("max_turn_silence", max_silence))
                elif msg["type"] == "Terminate":
                    if speech_start is not None:
                        await emit(final=True)
                    await ws.send(json.dumps({"type": "Termination", "audio_duration_seconds": received / sr}))
                    await ws.close()
                    return
                continue

            samples = np.frombuffer(message, dtype="<i2")
            self.audio_chunks += 1
            chunk_ms = len(samples) / sr * 1000
            t_end = (received + len(samples)) / sr * 1000
            received += len(samples)
            if np.abs(samples).max(initial=0) > self.speech_threshold:  # "speech"
                if speech_start is None:
                    speech_start = t_end - chunk_ms
                speech_end = t_end
                silence_ms = 0.0
                await emit(final=False)
            elif speech_start is not None:
                silence_ms += chunk_ms
                if silence_ms >= max_silence:
                    await emit(final=True)
