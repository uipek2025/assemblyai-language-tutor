// Lumi tutor orchestration: mic -> AssemblyAI realtime STT -> /api/tutor
// (grading + LLM Gateway) -> TTS-1 per language segment -> back to listening.
import { startMic } from './mic.js';
import { SttSession } from './stt.js';
import { speakSegments, stopSpeaking } from './voice.js';
import { LANGUAGES, TTS_LANGS, $, postJSON, sleep, toast, keyterms, textOf } from './common.js';

const state = {
  phase: 'idle',
  session: null,
  mic: null,
  stt: null,
  native: 'en',
  target: 'es',
  practice: null, // phrase the learner should say next
  cardPhrase: null, // phrase shown on the card (stays after practice ends)
  keytermsActive: false,
  history: [],
  muted: false,
  interrupted: false,
  stopping: false,
};

const PHASE_LABEL = { idle: 'Press Start', listening: 'Listening…', thinking: 'Thinking…', speaking: 'Speaking…' };
const VERDICT = {
  correct: ['✓ Nailed it', 'good'],
  close: ['Almost there', 'warn'],
  retry: ["Let's try again", 'bad'],
  no_speech: ["Didn't catch that", 'bad'],
  not_an_attempt: ['Sounded like a question — answered it', 'info'],
};
const zeroBuffers = new Map();
const silence = (n) => zeroBuffers.get(n) || zeroBuffers.set(n, new ArrayBuffer(n)).get(n);
let clipStartedAt = 0;
let clipGuardMs = 0;

// ---------------------------------------------------------------- UI helpers
function setPhase(phase) {
  state.phase = phase;
  document.body.dataset.phase = phase;
  $('phase').textContent = PHASE_LABEL[phase];
}

function hud(key, value) {
  $(`hud-${key}`).textContent = value;
}

function setRunning(running) {
  $('start').textContent = running ? 'Stop' : 'Start';
  $('start').classList.toggle('danger', running);
}

function addUserBubble(text, lang) {
  const el = document.createElement('div');
  el.className = 'bubble user';
  el.textContent = text;
  if (lang) {
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.textContent = lang;
    el.appendChild(tag);
  }
  appendToLog(el);
}

function addTutorBubble(segments) {
  const el = document.createElement('div');
  el.className = 'bubble tutor';
  for (const seg of segments) {
    const span = document.createElement(seg.lang === state.target ? 'mark' : 'span');
    span.textContent = `${seg.text} `;
    if (seg.lang === state.target) {
      span.title = 'Click to hear again';
      span.onclick = () => replay([{ lang: seg.lang, text: seg.text }], 1);
    }
    el.appendChild(span);
  }
  appendToLog(el);
}

/** Colour the learner's words under their attempt so the log keeps a record. */
function annotateLastAttempt(grade) {
  const bubbles = $('log').querySelectorAll('.bubble.user');
  const last = bubbles[bubbles.length - 1];
  if (!last || !grade.words?.length) return;
  const row = document.createElement('div');
  row.className = 'chips small';
  for (const w of grade.words) {
    const chip = document.createElement('span');
    chip.className = `chip ${w.status}`;
    chip.textContent = w.target;
    row.appendChild(chip);
  }
  last.appendChild(row);
}

function appendToLog(el) {
  $('log').appendChild(el);
  $('log').scrollTop = $('log').scrollHeight;
}

function showPartial(text) {
  $('partial').textContent = text;
}

function renderCard(phrase) {
  state.cardPhrase = phrase;
  $('phraseCard').hidden = !phrase;
  if (!phrase) return;
  $('phraseLabel').textContent = 'Say it back';
  $('phraseText').textContent = phrase.text;
  $('phrasePron').textContent = phrase.pronunciation ? `🗣 ${phrase.pronunciation}` : '';
  $('phraseMeaning').textContent = phrase.meaning || '';
  $('result').hidden = true;
}

function showGrade(grade) {
  const [label, tone] = VERDICT[grade.verdict] || [grade.verdict, 'info'];
  $('result').hidden = false;
  $('verdict').textContent = `${label} · ${Math.round(grade.score * 100)}%`;
  $('verdict').className = `verdict ${tone}`;
  const chips = $('chips');
  chips.replaceChildren();
  for (const w of grade.words) {
    const chip = document.createElement('span');
    chip.className = `chip ${w.status}`;
    chip.textContent = w.target;
    const conf = w.confidence != null ? `, confidence ${w.confidence.toFixed(2)}` : '';
    chip.title = `${w.status}${w.heard ? ` — heard "${w.heard}"` : ''}${conf}`;
    chips.appendChild(chip);
  }
  $('heard').textContent = grade.heard ? `Heard: “${grade.heard}”` : '';
}

function updateTtsNote() {
  const t = state.target;
  $('ttsNote').textContent = TTS_LANGS.includes(t)
    ? ''
    : `${LANGUAGES[t]} is understood by the STT model, but AssemblyAI-TTS-1 doesn't speak it yet — Lumi will use your browser's voice for ${LANGUAGES[t]} phrases.`;
  $('slowLabel').textContent = LANGUAGES[t];
}

// ------------------------------------------------------ STT configuration
const languageCodes = () => [...new Set([state.native, state.target])];
const allowInterrupt = () => $('interrupt').checked;
// A one-word French clip is easy for the speakers to trip. Ignore barge-in until
// that clip has had time to finish, and at most 1.5 s into a longer one.
const canBargeIn = () => allowInterrupt() && state.phase === 'speaking'
  && performance.now() - clipStartedAt > clipGuardMs;

/** Turn detection follows the conversation state: be patient while practising. */
function applyTurnPreset() {
  if (!state.stt?.isOpen) return;
  const mode = state.practice ? 'practice' : 'conversation';
  const preset = state.session.turn_presets[mode];
  const config = { ...preset };
  if (state.practice && $('boost').checked) {
    config.keyterms_prompt = keyterms(state.practice.text);
    state.keytermsActive = true;
  } else if (state.keytermsActive) {
    config.keyterms_prompt = [];
    state.keytermsActive = false;
  }
  state.stt.update(config);
  hud('turn', `${mode} · up to ${preset.max_turn_silence} ms silence`);
}

function setPractice(phrase) {
  state.practice = phrase;
  if (phrase && phrase.text !== state.cardPhrase?.text) renderCard(phrase);
  if (!phrase && state.cardPhrase) $('phraseLabel').textContent = 'Practised';
  applyTurnPreset();
}

// --------------------------------------------------------------- speaking
function interrupt() {
  state.interrupted = true;
  stopSpeaking();
}

async function speak(segments, rate) {
  state.interrupted = false;
  setPhase('speaking');
  state.muted = !allowInterrupt(); // half-duplex unless the user opted in
  try {
    const { firstAudioMs, usedFallback } = await speakSegments(segments, {
      rateFor: (seg) => rate ?? (seg.lang === state.target && $('slow').checked ? 0.85 : 1),
      isInterrupted: () => state.interrupted,
      onClipStart: (durationMs) => {
        clipStartedAt = performance.now();
        clipGuardMs = Math.min(Math.max(durationMs || 800, 400), 1500);
      },
    });
    if (firstAudioMs != null) hud('tts', `${firstAudioMs} ms${usedFallback ? ' (browser voice)' : ''}`);
  } finally {
    await sleep(150); // let the speaker tail die out before the mic reopens
    state.muted = false;
    setPhase(state.stt?.isOpen ? 'listening' : 'idle');
  }
}

async function replay(segments, rate) {
  if (state.phase === 'thinking' || state.phase === 'speaking') return;
  await speak(segments, rate);
}

// ----------------------------------------------------------- tutor turns
async function callTutor(payload) {
  try {
    const data = await postJSON('/api/tutor', payload);
    if (data.meta) hud('llm', `${data.meta.llm_ms} ms · ${data.meta.model}${data.meta.structured ? '' : ' (prompt-only JSON)'}`);
    return data;
  } catch (err) {
    toast(`Tutor error: ${err.message}`);
    return null;
  }
}

async function deliver(data) {
  if (data.grade) {
    showGrade(data.grade);
    annotateLastAttempt(data.grade);
  }
  setPractice(data.practice);
  addTutorBubble(data.segments);
  // Tell the STT what Lumi just said: context makes the learner's reply easier to transcribe.
  state.stt?.update({ agent_context: textOf(data.segments) });
  await speak(data.segments);
}

async function greet() {
  setPhase('thinking');
  const data = await callTutor({
    native_language: state.native,
    target_language: state.target,
    transcript: '',
    greet: true,
  });
  const segments = data?.segments || [
    { lang: 'en', text: `Hi, I'm Lumi. Ask me how to say anything in ${LANGUAGES[state.target]}.` },
  ];
  state.history.push({ role: 'assistant', content: textOf(segments) });
  await deliver({ segments, practice: null, grade: null });
}

async function respond(turn) {
  setPhase('thinking');
  const data = await callTutor({
    native_language: state.native,
    target_language: state.target,
    transcript: turn.transcript,
    words: (turn.words || []).map(({ text, start, end, confidence }) => ({ text, start, end, confidence })),
    stt_language: turn.language_code || null,
    stt_language_confidence: turn.language_confidence ?? null,
    practice: state.practice ? { text: state.practice.text, lang: state.practice.lang } : null,
    history: state.history.slice(-10),
  });
  if (!data) {
    setPhase(state.stt?.isOpen ? 'listening' : 'idle');
    return;
  }
  state.history.push({ role: 'user', content: turn.transcript }, { role: 'assistant', content: textOf(data.segments) });
  await deliver(data);
}

function onTurn(turn, final) {
  if (!final) {
    if (turn.transcript) {
      if (canBargeIn()) interrupt();
      showPartial(turn.transcript);
    }
    return;
  }
  showPartial('');
  const text = (turn.transcript || '').trim();
  if (!text || state.phase === 'thinking') return; // ignore overlap while a reply is being prepared
  if (state.phase === 'speaking') {
    if (!canBargeIn()) return;
    interrupt();
  }
  hud('eot', turn.eotLatencyMs != null ? `${turn.eotLatencyMs} ms` : '–');
  hud('lang', turn.language_code ? `${turn.language_code} (${(turn.language_confidence ?? 0).toFixed(2)})` : '–');
  addUserBubble(text, turn.language_code);
  respond(turn);
}

function onSpeechStarted() {
  if (canBargeIn()) interrupt(); // barge-in
}

function onSttClose(ev) {
  if (!state.stopping && state.mic) {
    toast(`Speech connection closed (${ev.code}${ev.reason ? `: ${ev.reason}` : ''}). Press Start to reconnect.`);
    stop();
  }
}

function onMicChunk(pcm, rms) {
  $('level').style.width = `${Math.min(100, Math.round(rms * 400))}%`;
  // While Lumi talks (half-duplex) we keep the stream alive with silence so the
  // server's audio timeline - and our latency maths - stays continuous.
  state.stt?.sendAudio(state.muted ? silence(pcm.byteLength) : pcm);
}

// ------------------------------------------------------------ start/stop
async function start() {
  $('start').disabled = true;
  try {
    state.session = await postJSON('/api/session', { purpose: 'tutor' });
    state.mic = await startMic({ onChunk: onMicChunk });
    state.stt = new SttSession({
      wsUrl: state.session.ws_url,
      token: state.session.token,
      params: state.session.connect_params,
      sampleRate: state.mic.sampleRate,
      onTurn,
      onSpeechStarted,
      onClose: onSttClose,
      onError: (err) => toast(err.message),
    });
    const begin = await state.stt.connect();
    // Unknown connection params are ignored rather than rejected, so confirm
    // the session is really running the model we asked for.
    const requested = state.session.connect_params.speech_model;
    const actual = begin.configuration?.model || requested;
    hud('model', actual === requested ? actual : `${actual} (asked for ${requested})`);
    state.stt.update({ language_codes: languageCodes() }); // steer to the two languages in play
    applyTurnPreset();
    setRunning(true);
    state.history = [];
    await greet();
  } catch (err) {
    toast(err.message);
    await stop();
  } finally {
    $('start').disabled = false;
  }
}

async function stop() {
  state.stopping = true;
  interrupt();
  state.stt?.terminate();
  state.stt = null;
  await state.mic?.stop();
  state.mic = null;
  state.practice = null;
  setRunning(false);
  setPhase('idle');
  showPartial('');
  $('level').style.width = '0%';
  state.stopping = false;
}

// ------------------------------------------------------------- wiring
function fillSelect(id, value) {
  const select = $(id);
  for (const [code, name] of Object.entries(LANGUAGES)) {
    const opt = new Option(TTS_LANGS.includes(code) ? name : `${name} (listen only)`, code);
    select.add(opt);
  }
  select.value = value;
}

fillSelect('native', state.native);
fillSelect('target', state.target);
updateTtsNote();

$('start').onclick = (e) => {
  e.currentTarget.blur(); // so Space doesn't press the button
  return state.mic ? stop() : start();
};

for (const id of ['native', 'target']) {
  $(id).onchange = () => {
    state.native = $('native').value;
    state.target = $('target').value;
    if (state.native === state.target) toast('Pick two different languages.');
    state.practice = null;
    renderCard(null);
    updateTtsNote();
    state.stt?.update({ language_codes: languageCodes() });
    applyTurnPreset();
  };
}

$('boost').onchange = applyTurnPreset;
$('hear').onclick = () => state.cardPhrase && replay([{ lang: state.cardPhrase.lang, text: state.cardPhrase.text }], 1);
$('hearSlow').onclick = () => state.cardPhrase && replay([{ lang: state.cardPhrase.lang, text: state.cardPhrase.text }], 0.7);

const isFormField = (el) => ['INPUT', 'SELECT', 'TEXTAREA'].includes(el?.tagName);
document.addEventListener('keydown', (e) => {
  if (e.code !== 'Space' || !state.stt?.isOpen || isFormField(document.activeElement)) return;
  e.preventDefault();
  if (state.phase === 'listening' && !e.repeat) {
    state.stt.forceEndpoint(); // learner says "I'm done" instead of waiting out the silence
    toast('Turn ended', 900);
  }
});
window.addEventListener('beforeunload', () => state.stt?.terminate());
