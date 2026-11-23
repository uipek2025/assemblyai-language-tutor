"""Scoring functions for the evaluation harness (pure, unit-tested).

Why these metrics, for a language tutor specifically:
- Plain WER is dominated by easy carrier words ("how do you say ... in
  Spanish") and hides errors on the one foreign word that matters. So we
  report target-span accuracy separately: did "me gustaría" come through?
- End-of-turn behaviour gets its own metrics: cutting a hesitating learner off
  is the most visible failure in this product, and latency is the cost of
  being patient. The two are a trade-off, so we sweep it instead of arguing.
- The grader's false-accept / false-reject rates are measured on labelled
  attempts, because a speech model that "auto-corrects" a wrong attempt into
  the right phrase makes the tutor praise a mistake.
"""
from __future__ import annotations

import math
import random
from statistics import mean
from typing import Callable, Iterable, Sequence

import numpy as np

from app.grading import normalize_tokens


# ---------------------------------------------------------------- accuracy
def edit_ops(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) of a minimum edit alignment."""
    n, m = len(ref), len(hyp)
    # dp[i][j] = (cost, subs, dels, ins)
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            c_sub, s, d, ins = dp[i - 1][j - 1]
            sub = (c_sub + 1, s + 1, d, ins)
            c_del, s, d, ins = dp[i - 1][j]
            dele = (c_del + 1, s, d + 1, ins)
            c_ins, s, d, ins = dp[i][j - 1]
            insert = (c_ins + 1, s, d, ins + 1)
            dp[i][j] = min(sub, dele, insert, key=lambda t: t[0])
    _, s, d, ins = dp[n][m]
    return s, d, ins


def wer_counts(reference: str, hypothesis: str) -> dict[str, int]:
    """Counts for corpus-level WER. Same normaliser on both sides (always)."""
    ref = normalize_tokens(reference)
    hyp = normalize_tokens(hypothesis)
    s, d, i = edit_ops(ref, hyp)
    return {"sub": s, "del": d, "ins": i, "n": len(ref)}


def corpus_wer(counts: Iterable[dict[str, int]]) -> float | None:
    counts = list(counts)
    n = sum(c["n"] for c in counts)
    if n == 0:
        return None
    return sum(c["sub"] + c["del"] + c["ins"] for c in counts) / n


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = list(reference), list(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    s, d, i = edit_ops(ref, hyp)
    return (s + d + i) / len(ref)


def span_hit(span: str, hypothesis: str, keep_accents: bool = False) -> bool:
    """Did the target span appear verbatim (as a token sequence) in the hypothesis?"""
    s = normalize_tokens(span, keep_accents=keep_accents)
    h = normalize_tokens(hypothesis, keep_accents=keep_accents)
    if not s:
        return True
    return any(h[k : k + len(s)] == s for k in range(len(h) - len(s) + 1))


def span_cer(span: str, hypothesis: str) -> float:
    """Best character error rate of the span against any similar-length window."""
    s_tokens = normalize_tokens(span)
    h_tokens = normalize_tokens(hypothesis)
    target = " ".join(s_tokens)
    if not target:
        return 0.0
    if not h_tokens:
        return 1.0
    best = 1.0
    n = len(s_tokens)
    for width in range(max(1, n - 1), n + 2):
        for k in range(0, max(1, len(h_tokens) - width + 1)):
            best = min(best, cer(target, " ".join(h_tokens[k : k + width])))
    return min(best, 1.0)


# ----------------------------------------------------------- end of turn
def speech_end_ms(samples: np.ndarray, sample_rate: int, frame_ms: int = 30, rel_db: float = -30.0) -> float | None:
    """Vendor-independent speech end: last frame within ``rel_db`` of the loudest.

    Using the providers' own word timestamps to define "speech end" would let
    each vendor grade its own homework, so the harness uses this (or a
    manually annotated ``speech_end_s`` in the manifest) for both.
    """
    if samples.size == 0:
        return None
    frame = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = samples.size // frame
    if n_frames == 0:
        return None
    frames = samples[: n_frames * frame].reshape(n_frames, frame).astype(np.float64)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    peak = float(rms.max())
    if peak < 1e-4:
        return None
    threshold = max(peak * 10 ** (rel_db / 20), 1e-4)
    active = np.nonzero(rms >= threshold)[0]
    if active.size == 0:
        return None
    return (int(active[-1]) + 1) * frame_ms


def turn_outcome(turns: list[dict], speech_end: float | None, grace_ms: float = 250) -> dict:
    """Classify end-of-turn behaviour for a single-turn utterance.

    ``turns`` are final turns with ``receipt_audio_ms`` (how much audio had been
    sent when the final arrived) and ``forced`` (arrived only after Terminate).
    - premature cutoff (FP): a final arrived before the learner finished.
    - missed endpoint (FN): no natural final after speech ended.
    - latency: speech end -> the final that closed the turn.
    """
    if speech_end is None:
        return {"premature": 0, "missed": not turns, "eot_latency_ms": None, "tp": 0}
    premature = sum(1 for t in turns if not t["forced"] and t["receipt_audio_ms"] < speech_end - grace_ms)
    closing = next((t for t in turns if not t["forced"] and t["receipt_audio_ms"] >= speech_end - grace_ms), None)
    latency = round(closing["receipt_audio_ms"] - speech_end) if closing else None
    return {
        "premature": premature,
        "missed": closing is None,
        "eot_latency_ms": latency,
        "tp": int(closing is not None),
    }


# ------------------------------------------------------------- statistics
def percentile(values: Sequence[float], p: float) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    k = (len(values) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def bootstrap_ci(
    items: Sequence,
    stat: Callable[[Sequence], float | None],
    n_resamples: int = 1000,
    seed: int = 7,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI over *items* (utterances), not over words."""
    items = list(items)
    if len(items) < 2:
        return None, None
    rng = random.Random(seed)
    stats = []
    for _ in range(n_resamples):
        sample = [items[rng.randrange(len(items))] for _ in items]
        value = stat(sample)
        if value is not None:
            stats.append(value)
    if not stats:
        return None, None
    return percentile(stats, 100 * alpha / 2), percentile(stats, 100 * (1 - alpha / 2))


def rate(flags: Sequence[bool]) -> float | None:
    flags = list(flags)
    return mean(1.0 if f else 0.0 for f in flags) if flags else None
