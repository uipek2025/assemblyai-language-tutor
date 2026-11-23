from app.grading import grade_attempt, looks_like_question, normalize_tokens


def words(text, conf=0.95):
    return [{"text": w, "confidence": conf} for w in text.split()]


def test_normalize_handles_spanish_punctuation_and_accents():
    assert normalize_tokens("¿Dónde está la biblioteca?") == ["donde", "esta", "la", "biblioteca"]
    assert normalize_tokens("¿Dónde está?", keep_accents=True) == ["dónde", "está"]
    assert normalize_tokens("J’ai faim!") == ["j'ai", "faim"]


def test_exact_attempt_is_correct_even_without_accents_from_stt():
    g = grade_attempt("donde esta la biblioteca", "¿Dónde está la biblioteca?", words("donde esta la biblioteca"))
    assert g.verdict == "correct"
    assert g.score == 1.0
    assert [w.target for w in g.words] == ["dónde", "está", "la", "biblioteca"]  # shown as written
    assert g.heard == "donde esta la biblioteca"


def test_word_split_differently_still_counts():
    g = grade_attempt("donde esta la biblio teca", "¿Dónde está la biblioteca?")
    assert g.verdict == "correct"
    assert [w.status for w in g.words] == ["ok", "ok", "ok", "ok"]


def test_wrong_word_is_not_praised():
    # librería = bookshop: a classic learner mistake that must not get a "well done".
    g = grade_attempt("donde esta la libreria", "¿Dónde está la biblioteca?", words("donde esta la libreria"))
    assert g.verdict != "correct"
    assert g.words[-1].status == "missed"
    assert "biblioteca=MISSED" in g.summary()


def test_near_miss_is_close():
    g = grade_attempt("me gustaria un cafe por favo", "Me gustaría un café, por favor.")
    assert g.verdict in ("correct", "close")
    assert g.words[-1].status == "close"


def test_missing_words_are_flagged():
    g = grade_attempt("la cuenta por", "La cuenta, por favor.")
    assert g.verdict == "close"
    assert g.words[-1].status == "missed"


def test_low_confidence_word_is_unclear():
    w = words("donde esta la biblioteca")
    w[3]["confidence"] = 0.2
    g = grade_attempt("donde esta la biblioteca", "¿Dónde está la biblioteca?", w)
    assert g.words[3].status == "unclear"
    assert g.verdict == "correct"  # one unclear word is coaching, not failure


def test_silence_is_no_speech():
    assert grade_attempt("", "Hola").verdict == "no_speech"


def test_fillers_around_attempt_are_ignored():
    g = grade_attempt("um okay donde esta la biblioteca", "¿Dónde está la biblioteca?")
    assert g.verdict == "correct"
    assert g.extra_words == ["um", "okay"]


def test_language_detection_separates_questions_from_attempts():
    g = grade_attempt("wait what does biblioteca mean", "¿Dónde está la biblioteca?")
    assert looks_like_question("en", 0.95, "en", g)
    assert not looks_like_question("es", 0.95, "en", g)
