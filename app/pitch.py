"""Extra credit: a 60-second YC-style pitch coach.

Same building blocks as the tutor (streaming STT, LLM Gateway, TTS-1), pointed
at something every founder in a YC batch is doing this week: rehearsing a pitch.
The browser computes delivery stats from word timestamps; the LLM gives the
content feedback.
"""
from __future__ import annotations

from typing import Any

ROAST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "description": "1-10, how fundable this pitch sounded"},
        "one_liner": {"type": "string", "description": "What the company does, as the listener understood it"},
        "strongest_moment": {"type": "string"},
        "missing": {"type": "array", "items": {"type": "string"}, "description": "Key pitch elements not covered"},
        "cut_this": {"type": "string", "description": "The sentence to delete, quoted"},
        "better_opening": {"type": "string", "description": "A sharper first sentence they could use"},
        "spoken_verdict": {"type": "string", "description": "Max 40 words, read aloud, direct but kind"},
    },
    "required": ["score", "one_liner", "strongest_moment", "missing", "cut_this", "better_opening", "spoken_verdict"],
    "additionalProperties": False,
}

ROAST_SYSTEM = """You are a seasoned YC partner giving a founder fast, direct, kind feedback on a ~60 second spoken pitch.
Judge clarity first: could a smart stranger repeat what the company does? Then check for: problem, who has it, solution, traction or evidence, why now, and the ask.
Use the delivery stats provided (pace, hedge words, long pauses) only if they matter.
Be specific and quote the founder. No generic advice. Return JSON matching the schema."""


def roast_messages(transcript: str, stats: dict[str, Any]) -> list[dict[str, str]]:
    stat_line = ", ".join(f"{k}={v}" for k, v in stats.items())
    return [
        {"role": "system", "content": ROAST_SYSTEM},
        {"role": "user", "content": f'Pitch transcript: "{transcript}"\nDelivery stats: {stat_line}'},
    ]
