"""Deterministic grading of a learner's attempt at a target phrase.

Why grade in code instead of asking the LLM "did they say it right?":
- LLMs are unreliable judges of string similarity and will happily praise a
  wrong attempt. A wrong "well done" is the worst failure a tutor can have.
- Deterministic scores are explainable, testable, and the *same* function is
  used by the eval harness to measure false-accept / false-reject rates.

What this measures honestly: intelligibility. Did a strong speech model
understand the attempt as the target words? It cannot hear stress or
intonation. Phoneme-level scoring would need a dedicated pronunciation
assessment model on top; that is called out as a next step, not faked.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any

from .config import LOW_CONFIDENCE

_PUNCT = re.compile(r"[^\w\s'’-]", re.UNICODE)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if unicodedata.category(c) != "Mn"))


def normalize_tokens(text: str, keep_accents: bool = False) -> list[str]:
    """Lower-case, drop punctuation (keeping in-word apostrophes/hyphens), split."""
    text = unicodedata.normalize("NFKC", text or "").lower().replace("’", "'")
    text = _PUNCT.sub(" ", text).replace("_", " ")
    if not keep_accents:
        text = strip_accents(text)
    tokens = [t.strip("'-") for t in text.split()]
    return [t for t in tokens if t]


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() if (a or b) else 1.0


@dataclass
class WordResult:
    target: str
    status: str  # ok | unclear | close | missed
    heard: str = ""
    confidence: float | None = None


@dataclass
class Grade:
    verdict: str  # correct | close | retry | no_speech | not_an_attempt
    score: float
    heard: str
    words: list[WordResult] = field(default_factory=list)
    extra_words: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        """Compact line handed to the LLM so feedback is grounded in facts."""
        parts = []
        for w in self.words:
            if w.status == "ok":
                parts.append(f"{w.target}=ok")
            elif w.status == "unclear":
                parts.append(f"{w.target}=UNCLEAR(low confidence)")
            elif w.status == "close":
                parts.append(f'{w.target}=CLOSE(heard "{w.heard}")')
            else:
                parts.append(f"{w.target}=MISSED" + (f' (heard "{w.heard}")' if w.heard else ""))
        return f"verdict={self.verdict}, score={self.score:.2f}; " + ", ".join(parts)


_WEIGHTS = {"ok": 1.0, "unclear": 0.85, "close": 0.6, "missed": 0.0}


def _heard_tokens(transcript: str, words: list[dict] | None) -> list[tuple[str, float | None]]:
    """Tokens of what the STT heard, each carrying its word-level confidence."""
    if words:
        out: list[tuple[str, float | None]] = []
        for w in words:
            conf = w.get("confidence")
            for tok in normalize_tokens(str(w.get("text", ""))):
                out.append((tok, float(conf) if isinstance(conf, (int, float)) else None))
        if out:
            return out
    return [(tok, None) for tok in normalize_tokens(transcript)]


def grade_attempt(
    transcript: str,
    target: str,
    words: list[dict] | None = None,
    low_confidence: float = LOW_CONFIDENCE,
) -> Grade:
    # Compare without accents (STT orthography for "dónde" vs "donde" isn't
    # something the learner controls), but show the target as it's written.
    display = normalize_tokens(target, keep_accents=True)
    target_tokens = [strip_accents(t) for t in display]
    heard = _heard_tokens(transcript, words)
    heard_tokens = [t for t, _ in heard]
    heard_text = (transcript or "").strip() or " ".join(heard_tokens)

    if not target_tokens:
        return Grade("retry", 0.0, heard_text)
    if not heard_tokens:
        return Grade("no_speech", 0.0, "", [WordResult(t, "missed") for t in display])

    results: list[WordResult | None] = [None] * len(target_tokens)
    extra: list[str] = []
    matcher = SequenceMatcher(None, target_tokens, heard_tokens, autojunk=False)

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                tok, conf = heard[j1 + k]
                low = conf is not None and conf < low_confidence
                results[i1 + k] = WordResult(target_tokens[i1 + k], "unclear" if low else "ok", tok, conf)
        elif op == "delete":
            for k in range(i1, i2):
                results[k] = WordResult(target_tokens[k], "missed")
        elif op == "insert":
            extra.extend(heard_tokens[j1:j2])
        else:  # replace
            tgt, hrd = target_tokens[i1:i2], heard_tokens[j1:j2]
            confs = [c for _, c in heard[j1:j2] if c is not None]
            block_conf = min(confs) if confs else None
            if "".join(tgt) == "".join(hrd):
                # Same letters, different word boundaries ("biblio teca") - fine.
                for k, t in enumerate(tgt):
                    low = block_conf is not None and block_conf < low_confidence
                    results[i1 + k] = WordResult(t, "unclear" if low else "ok", " ".join(hrd), block_conf)
            elif len(tgt) == len(hrd):
                for k, t in enumerate(tgt):
                    h, c = heard[j1 + k]
                    status = "close" if _similarity(t, h) >= 0.75 else "missed"
                    results[i1 + k] = WordResult(t, status, h, c)
            else:
                joined = " ".join(hrd)
                for k, t in enumerate(tgt):
                    best = max(hrd, key=lambda h: _similarity(t, h))
                    status = "close" if _similarity(t, best) >= 0.75 else "missed"
                    results[i1 + k] = WordResult(t, status, best if status == "close" else joined, block_conf)

    word_results = [r for r in results if r is not None]
    for r, shown in zip(word_results, display):
        r.target = shown
    score = sum(_WEIGHTS[r.status] for r in word_results) / len(word_results)
    bad = [r for r in word_results if r.status in ("missed", "close")]
    unclear = [r for r in word_results if r.status == "unclear"]

    if not bad and len(unclear) <= max(1, len(word_results) // 4):
        verdict = "correct"
    elif score >= 0.6:
        verdict = "close"
    else:
        verdict = "retry"
    return Grade(verdict, round(score, 3), heard_text, word_results, extra)


def looks_like_question(stt_language: str | None, stt_confidence: float | None, native: str, grade: Grade) -> bool:
    """Use language detection to tell an attempt from a question.

    If a learner who was asked to repeat a Spanish phrase replies confidently in
    their native language and barely matches the target, they are asking
    something ("wait, what does biblioteca mean?"), not attempting it.
    """
    return bool(
        stt_language
        and stt_language[:2] == native[:2]
        and (stt_confidence or 0) >= 0.7
        and grade.score < 0.5
    )
