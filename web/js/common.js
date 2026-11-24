// Small shared helpers for the three pages.

// Languages the tutor offers. AssemblyAI-TTS-1 speaks each of these with a voice recorded in that language.
export const LANGUAGES = {
  en: 'English',
  es: 'Spanish',
  fr: 'French',
  de: 'German',
  it: 'Italian',
  pt: 'Portuguese',
};

export const TTS_LANGS = ['en', 'es', 'fr', 'de', 'it', 'pt'];

export const $ = (id) => document.getElementById(id);
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function postJSON(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON error body */ }
  if (!res.ok) throw new Error(data?.detail || data?.error || `HTTP ${res.status}`);
  return data;
}

let toastTimer;
export function toast(message, ms = 5000) {
  const el = $('toast');
  if (!el) return console.warn(message);
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, ms);
}

export const textOf = (segments) => segments.map((s) => s.text).join(' ');

/** Keyterms for a practice phrase: the whole phrase plus its longer words. */
export function keyterms(phrase) {
  const clean = phrase.replace(/[¿?¡!.,;:"“”«»]/g, '').trim();
  const words = clean.split(/\s+/).filter((w) => w.length >= 4);
  return [...new Set([clean, ...words])].slice(0, 10);
}
