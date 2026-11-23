// Extra credit: a 60-second pitch coach for a YC batch demo.
// Realtime STT (captions model) -> live delivery stats from word timestamps ->
// LLM Gateway feedback -> verdict spoken by TTS-1.
import { startMic } from './mic.js';
import { SttSession } from './stt.js';
import { speakSegments } from './voice.js';
import { $, postJSON, sleep, toast } from './common.js';

const PITCH_SECONDS = 60;
const HEDGES = ['basically', 'actually', 'literally', 'like', 'just', 'kind of', 'sort of', 'i think', 'you know', 'i guess', 'maybe', 'um', 'uh'];

let mic = null;
let stt = null;
let finals = [];
let partial = '';
let timer = null;
let running = false;

const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function stats() {
  const words = finals.flatMap((t) => t.words || []).filter((w) => w.text);
  const spanMs = words.length > 1 ? words[words.length - 1].end - words[0].start : 0;
  const text = ` ${finals.map((t) => t.transcript).join(' ').toLowerCase().replace(/[^\p{L}\p{N}'\s]/gu, ' ')} `;
  const hedgeHits = {};
  for (const h of HEDGES) {
    const count = text.split(` ${h} `).length - 1;
    if (count) hedgeHits[h] = count;
  }
  let pauses = 0;
  for (let i = 1; i < words.length; i++) if (words[i].start - words[i - 1].end > 1200) pauses++;
  return {
    words: words.length,
    wpm: spanMs > 5000 ? Math.round(words.length / (spanMs / 60000)) : null,
    hedge_words: Object.values(hedgeHits).reduce((a, b) => a + b, 0),
    hedges: Object.entries(hedgeHits).map(([h, n]) => `${h}×${n}`).join(' ') || 'none',
    long_pauses: pauses,
    seconds: Math.round(spanMs / 1000),
  };
}

function render() {
  const s = stats();
  $('s-wpm').textContent = s.wpm ?? '–';
  $('s-hedges').textContent = s.hedge_words;
  $('s-pauses').textContent = s.long_pauses;
  $('s-words').textContent = s.words;
  const pattern = new RegExp(`\\b(${HEDGES.join('|')})\\b`, 'gi');
  const said = escapeHtml(finals.map((t) => t.transcript).join(' ')).replace(pattern, '<span class="hedge">$1</span>');
  $('live').innerHTML = `${said} <span class="partial-inline">${escapeHtml(partial)}</span>`;
}

function onTurn(turn, final) {
  if (final) {
    if (turn.transcript) finals.push(turn);
    partial = '';
  } else {
    partial = turn.transcript || '';
  }
  render();
}

async function start() {
  finals = [];
  partial = '';
  $('feedback').hidden = true;
  render();
  const session = await postJSON('/api/session', { purpose: 'pitch' });
  mic = await startMic({ onChunk: (pcm) => stt?.sendAudio(pcm) });
  stt = new SttSession({
    wsUrl: session.ws_url,
    token: session.token,
    params: session.connect_params,
    sampleRate: mic.sampleRate,
    onTurn,
    onError: (e) => toast(e.message),
  });
  await stt.connect();
  stt.update({ language_codes: ['en'] });
  running = true;
  $('go').textContent = 'Finish';
  const endsAt = Date.now() + PITCH_SECONDS * 1000;
  timer = setInterval(() => {
    const left = Math.max(0, Math.round((endsAt - Date.now()) / 1000));
    $('timer').textContent = `${Math.floor(left / 60)}:${String(left % 60).padStart(2, '0')}`;
    if (left === 0) finish();
  }, 250);
}

async function finish() {
  if (!running) return;
  running = false;
  clearInterval(timer);
  $('go').disabled = true;
  $('go').textContent = 'Thinking…';
  stt.terminate(); // flushes the in-flight turn
  await sleep(1600);
  await mic.stop();
  mic = null;
  stt = null;
  const transcript = finals.map((t) => t.transcript).join(' ').trim();
  try {
    if (!transcript) throw new Error("I didn't hear a pitch — try again.");
    const fb = await postJSON('/api/roast', { transcript, stats: stats() });
    $('f-score').textContent = `${fb.score}/10`;
    $('f-oneliner').textContent = fb.one_liner;
    $('f-strongest').textContent = fb.strongest_moment;
    $('f-missing').textContent = (fb.missing || []).join(' · ') || 'Nothing major';
    $('f-cut').textContent = fb.cut_this;
    $('f-opening').textContent = fb.better_opening;
    $('feedback').hidden = false;
    await speakSegments([{ lang: 'en', text: fb.spoken_verdict }]);
  } catch (err) {
    toast(err.message);
  } finally {
    $('go').disabled = false;
    $('go').textContent = 'Pitch again';
  }
}

$('go').onclick = (e) => {
  e.currentTarget.blur();
  (running ? finish() : start()).catch((err) => {
    toast(err.message);
    running = false;
    clearInterval(timer);
    $('go').disabled = false;
    $('go').textContent = 'Start pitch';
  });
};
