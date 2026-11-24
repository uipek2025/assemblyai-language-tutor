"""The tutor's brain: prompt, reply schema, and reply sanitising.

The LLM never decides whether an attempt was right (see grading.py). It gets
the grade as a fact and turns it into encouraging, specific, *spoken* feedback.
"""
from __future__ import annotations

from typing import Any

from .config import UI_LANGUAGES
from .grading import Grade, normalize_tokens

REPLY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "description": "The spoken reply, split into consecutive pieces by language.",
            "items": {
                "type": "object",
                "properties": {
                    "lang": {"type": "string", "description": "ISO 639-1 code of this piece"},
                    "text": {"type": "string"},
                },
                "required": ["lang", "text"],
                "additionalProperties": False,
            },
        },
        "practice_phrase": {
            "type": "object",
            "description": "The phrase the learner should say next. All fields empty if none.",
            "properties": {
                "text": {"type": "string"},
                "lang": {"type": "string"},
                "pronunciation": {"type": "string"},
                "meaning": {"type": "string"},
            },
            "required": ["text", "lang", "pronunciation", "meaning"],
            "additionalProperties": False,
        },
    },
    "required": ["segments", "practice_phrase"],
    "additionalProperties": False,
}


def language_name(code: str) -> str:
    return UI_LANGUAGES.get((code or "")[:2], code)


def system_prompt(native: str, target: str) -> str:
    n, t = language_name(native), language_name(target)
    return f"""You are Lumi, a warm, concise voice tutor. The learner's native language is {n} ("{native}"). They are learning {t} ("{target}").

This is a spoken conversation. Your reply is read aloud by text-to-speech, so:
- Keep replies short: at most about 35 words. No markdown, lists or emojis.
- Return JSON matching the schema. Split the reply into "segments" by language. Explanations go in {n} segments (lang "{native}"). Every {t} word or phrase goes in its own {t} segment (lang "{target}") so it is pronounced correctly. Never embed {t} words inside a {n} segment.

Teaching flow:
1. When asked how to say something, give the most common natural {t} phrasing, then invite the learner to say it back. The {t} phrase itself must be its own segment. Never end on "we say", "you'd say" or "you say" without that segment.
2. Whenever the learner should say a {t} phrase next (new phrase or retry), fill "practice_phrase": exact {t} text with correct accents and punctuation, lang "{target}", a simple respelling for a {n} speaker in "pronunciation", and a short {n} "meaning". Otherwise set all four fields to "".
3. If a PRACTICE RESULT is given, base feedback only on it. Never claim you heard stress, intonation or accent: speech recognition can only tell what was understood.
   - correct: praise briefly, then offer a next step (leave practice_phrase empty unless you teach something new).
   - close or retry: name the specific missed or unclear words, model the phrase once more, and keep the same practice_phrase.
   - not_an_attempt: the learner asked something instead. Answer it, then invite them to try the phrase again (keep practice_phrase).
4. If the learner speaks {t}, reply partly in {t} at their level.
5. Accuracy matters more than anything: never teach a phrase you are unsure of."""


def greeting_messages(native: str, target: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt(native, target)},
        {
            "role": "user",
            "content": f"(Session just started. Greet the learner in one short sentence that includes a "
            f"{language_name(target)} hello, and invite them to ask how to say something. No practice phrase yet.)",
        },
    ]


def build_messages(
    native: str,
    target: str,
    history: list[dict[str, str]],
    transcript: str,
    practice_text: str | None,
    grade: Grade | None,
    stt_language: str | None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt(native, target)}]
    for turn in history[-10:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content[:600]})

    lines = [f'Learner said (speech-to-text): "{transcript}"']
    if stt_language:
        lines.append(f"Detected language: {stt_language}")
    if practice_text and grade is not None:
        lines.append(f'PRACTICE RESULT for target "{practice_text}": {grade.summary()}')
    messages.append({"role": "user", "content": "\n".join(lines)})
    return messages


def sanitize_reply(
    data: dict[str, Any],
    native: str,
    target: str,
    *,
    current_practice: dict[str, str] | None = None,
    verdict: str | None = None,
    greet: bool = False,
) -> dict[str, Any]:
    """Make the LLM's reply safe for the voice loop, whatever it sent back.

    The practice loop is state, so code owns it, not the model:
    - a failed attempt (close / retry / not_an_attempt) never ends practice;
    - a correct attempt ends practice of that phrase, even if the model repeats it;
    - if a prompt-only fallback model drops ``practice_phrase`` altogether while
      teaching, the target-language phrase it just modelled becomes the practice.
    ``practice_source`` says which rule produced the phrase: model | carried | inferred.
    """
    segments: list[dict[str, str]] = []
    for seg in data.get("segments") or []:
        if not isinstance(seg, dict):
            continue
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        lang = str(seg.get("lang", "") or native).strip().lower()[:2] or native
        if lang not in (native, target):
            lang = native
        if segments and segments[-1]["lang"] == lang:  # merge adjacent same-language pieces
            segments[-1]["text"] = f'{segments[-1]["text"]} {text}'
        else:
            segments.append({"lang": lang, "text": text})
    if not segments:
        segments = [{"lang": native, "text": "Sorry, could you say that again?"}]

    raw = data.get("practice_phrase")
    practice, source = None, None
    if isinstance(raw, dict) and str(raw.get("text", "")).strip():
        candidate = {
            "text": str(raw.get("text", "")).strip(),
            "lang": (str(raw.get("lang", "")).strip().lower()[:2] or target),
            "pronunciation": str(raw.get("pronunciation", "")).strip(),
            "meaning": str(raw.get("meaning", "")).strip(),
        }
        if candidate["lang"] == target:  # only practise the language being learned
            practice, source = candidate, "model"

    if current_practice and verdict == "correct" and practice and _same_phrase(practice["text"], current_practice["text"]):
        practice, source = None, None
    elif current_practice and verdict in ("close", "retry", "not_an_attempt") and practice is None:
        practice = {"text": current_practice["text"], "lang": target, "pronunciation": "", "meaning": ""}
        source = "carried"
    elif practice is None and verdict is None and not greet and not isinstance(raw, dict):
        modelled = [s["text"] for s in segments if s["lang"] == target and s["lang"] != native]
        if modelled:
            practice = {"text": max(modelled, key=len), "lang": target, "pronunciation": "", "meaning": ""}
            source = "inferred"

    # The model sometimes announces a phrase ("we say") and puts it only on the
    # practice card, or nowhere in the audio. Say it if it was not already spoken.
    if practice and not _phrase_spoken(segments, target, practice["text"]):
        segments.append({"lang": target, "text": practice["text"]})

    return {"segments": segments, "practice": practice, "practice_source": source}


def _same_phrase(a: str, b: str) -> bool:
    return normalize_tokens(a) == normalize_tokens(b)


def _phrase_spoken(segments: list[dict[str, str]], target: str, phrase: str) -> bool:
    """True when the practice phrase's words already occur, in order, in a target segment."""
    want = normalize_tokens(phrase)
    if not want:
        return True
    spoken = normalize_tokens(" ".join(s["text"] for s in segments if s["lang"] == target))
    start = 0
    for token in want:
        try:
            start = spoken.index(token, start) + 1
        except ValueError:
            return False
    return True


def reply_text(segments: list[dict[str, str]]) -> str:
    return " ".join(s["text"] for s in segments)
