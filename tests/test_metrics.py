import jiwer
import numpy as np
import pytest

from evals.metrics import (
    bootstrap_ci,
    corpus_wer,
    percentile,
    span_cer,
    span_hit,
    speech_end_ms,
    turn_outcome,
    wer_counts,
)


@pytest.mark.parametrize(
    "ref,hyp",
    [
        ("how do you say where is the library", "how do you say where is the library"),
        ("what does me gustaria mean", "what does me gusta ria mean"),
        ("is it el agua or la agua", "is it el agua"),
        ("teach me to ask for the bill", "please teach me how to ask for the bill now"),
    ],
)
def test_wer_matches_jiwer(ref, hyp):
    counts = wer_counts(ref, hyp)
    assert corpus_wer([counts]) == pytest.approx(jiwer.wer(ref, hyp))


def test_wer_uses_same_normaliser_on_both_sides():
    # Formatting differences (case, punctuation, accents) must not count as errors.
    assert corpus_wer([wer_counts("¿Qué tal?", "que tal")]) == 0.0


def test_span_hit_is_what_matters_for_a_tutor():
    hyp = "what's the difference between sir and estar"
    assert span_hit("estar", hyp)
    assert not span_hit("ser", hyp)  # overall WER is low, but the key word is wrong
    assert corpus_wer([wer_counts("what's the difference between ser and estar", hyp)]) < 0.2


def test_accent_sensitive_span():
    assert span_hit("me gustaría", "me gustaria")
    assert not span_hit("me gustaría", "me gustaria", keep_accents=True)


def test_span_cer_finds_best_window():
    assert span_cer("ferrocarril", "how do you pronounce ferro carril") < 0.15
    assert span_cer("ferrocarril", "how do you pronounce it") > 0.5


def test_speech_end_from_audio():
    sr = 16000
    t = np.arange(int(sr * 1.0)) / sr
    tone = 0.5 * np.sin(2 * np.pi * 220 * t)
    audio = np.concatenate([np.zeros(sr // 2), tone, np.zeros(sr)])
    assert speech_end_ms(audio, sr) == pytest.approx(1500, abs=40)
    assert speech_end_ms(np.zeros(sr), sr) is None


def test_turn_outcome_classifies_cutoffs_and_latency():
    clean = turn_outcome([{"receipt_audio_ms": 2600, "forced": False}], speech_end=2000)
    assert clean == {"premature": 0, "missed": False, "eot_latency_ms": 600, "tp": 1}

    cut = turn_outcome(
        [{"receipt_audio_ms": 1200, "forced": False}, {"receipt_audio_ms": 2700, "forced": False}], speech_end=2000
    )
    assert cut["premature"] == 1 and cut["eot_latency_ms"] == 700

    missed = turn_outcome([{"receipt_audio_ms": 6000, "forced": True}], speech_end=2000)
    assert missed["missed"] and missed["eot_latency_ms"] is None


def test_percentile_and_bootstrap():
    values = list(range(1, 101))
    assert percentile(values, 50) == pytest.approx(50.5)
    lo, hi = bootstrap_ci(values, lambda s: sum(s) / len(s))
    assert lo < 50.5 < hi
    assert bootstrap_ci([1], lambda s: s[0]) == (None, None)
