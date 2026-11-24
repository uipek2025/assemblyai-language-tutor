// Speaking: AssemblyAI-TTS-1 through our server, one request per language
// segment, all fetched in parallel and played in order. If a language is not
// covered by TTS-1 (or TTS fails), that segment falls back to the browser's
// own speech synthesis so the lesson never goes silent.

let current = null; // { stop() } for whatever is playing now

export function stopSpeaking() {
  if (current) current.stop();
  current = null;
  window.speechSynthesis?.cancel();
}

async function fetchClip(seg) {
  try {
    const res = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: seg.text, lang: seg.lang }),
    });
    if (!res.ok) return { blob: null };
    return { blob: await res.blob(), cached: res.headers.get('X-TTS-Cached') === 'true' };
  } catch {
    return { blob: null };
  }
}

function playBlob(blob, rate, onStart) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(blob);
    const audio = new Audio();
    audio.preload = 'auto';
    audio.preservesPitch = true;
    let settled = false;
    const done = () => {
      if (settled) return;
      settled = true;
      URL.revokeObjectURL(url);
      resolve();
    };
    // Setting playbackRate before the duration is known makes Chrome skip the
    // start of a short clip. French words ("Parfait", "merci") are one clip
    // played slightly slower, so the beginning of the word disappeared.
    let started = false;
    const start = () => {
      if (started || settled) return;
      started = true;
      audio.playbackRate = rate || 1;
      const durationMs = Number.isFinite(audio.duration) ? audio.duration * 1000 : 0;
      onStart?.(durationMs);
      audio.play().catch(done);
    };
    current = { stop: () => { audio.pause(); done(); } };
    audio.onended = done;
    audio.onerror = done;
    audio.onloadedmetadata = start;
    audio.src = url;
    if (audio.readyState >= 1) start();
  });
}

function speakWithBrowser(text, lang, rate) {
  return new Promise((resolve) => {
    if (!window.speechSynthesis) return resolve();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang;
    u.rate = rate;
    const voice = speechSynthesis.getVoices().find((v) => v.lang?.toLowerCase().startsWith(lang));
    if (voice) u.voice = voice;
    u.onend = resolve;
    u.onerror = resolve;
    current = { stop: () => { speechSynthesis.cancel(); resolve(); } };
    speechSynthesis.speak(u);
  });
}

/**
 * Speak segments [{lang, text}] in order.
 * @param {(seg) => number} rateFor  playback rate per segment (e.g. slower for the target language)
 * @param {() => boolean} isInterrupted checked between segments
 * @returns {Promise<{firstAudioMs: number|null, usedFallback: boolean}>}
 */
export async function speakSegments(segments, { rateFor = () => 1, isInterrupted = () => false, onClipStart = () => {} } = {}) {
  const started = performance.now();
  const clips = segments.map(fetchClip); // parallel prefetch
  let firstAudioMs = null;
  let usedFallback = false;
  for (let i = 0; i < segments.length; i++) {
    if (isInterrupted()) break;
    const clip = await clips[i];
    if (isInterrupted()) break;
    if (firstAudioMs === null) firstAudioMs = Math.round(performance.now() - started);
    const seg = segments[i];
    if (clip.blob) {
      await playBlob(clip.blob, rateFor(seg), onClipStart);
    } else {
      usedFallback = true;
      onClipStart(Math.max(400, seg.text.length * 70));
      await speakWithBrowser(seg.text, seg.lang, rateFor(seg));
    }
  }
  current = null;
  return { firstAudioMs, usedFallback };
}
