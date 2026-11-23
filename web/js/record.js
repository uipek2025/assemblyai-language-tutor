// Records a small labelled eval set from the microphone: 16 kHz mono WAVs plus
// a manifest.jsonl in the format evals/run_eval.py expects.
import { startMic } from './mic.js';
import { $, toast } from './common.js';

let prompts = [];
let mic = null;
let recording = null; // { id, chunks, button }
const clips = new Map(); // id -> true once saved

function wavBlob(chunks, sampleRate) {
  const dataLen = chunks.reduce((n, c) => n + c.byteLength, 0);
  const buf = new ArrayBuffer(44 + dataLen);
  const v = new DataView(buf);
  const ascii = (offset, s) => [...s].forEach((ch, i) => v.setUint8(offset + i, ch.charCodeAt(0)));
  ascii(0, 'RIFF'); v.setUint32(4, 36 + dataLen, true); ascii(8, 'WAVE');
  ascii(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  ascii(36, 'data'); v.setUint32(40, dataLen, true);
  let offset = 44;
  for (const c of chunks) {
    new Uint8Array(buf, offset, c.byteLength).set(new Uint8Array(c));
    offset += c.byteLength;
  }
  return new Blob([buf], { type: 'audio/wav' });
}

function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

async function toggle(prompt, button, row) {
  if (recording && recording.id !== prompt.id) return toast('Stop the current recording first.');
  if (!mic) {
    mic = await startMic({
      onChunk: (pcm) => { if (recording) recording.chunks.push(pcm); },
    });
  }
  if (!recording) {
    recording = { id: prompt.id, chunks: [] };
    button.textContent = '■ Stop';
    button.classList.add('danger');
    return;
  }
  // Drop the last 150 ms so the mouse click on "Stop" isn't mistaken for speech
  // by the harness's vendor-independent speech-end detector.
  const chunks = recording.chunks.length > 6 ? recording.chunks.slice(0, -3) : recording.chunks;
  recording = null;
  button.textContent = '● Re-record';
  button.classList.remove('danger');
  download(wavBlob(chunks, mic.sampleRate), `${prompt.id}.wav`);
  clips.set(prompt.id, true);
  row.classList.add('done');
  $('count').textContent = `${clips.size}/${prompts.length} recorded`;
}

function render() {
  const container = $('prompts');
  for (const p of prompts) {
    const row = document.createElement('div');
    row.className = 'prompt-row';
    row.innerHTML = `<div class="id"></div><div><div class="say"></div><div class="meta"></div></div>`;
    row.querySelector('.id').textContent = p.id;
    row.querySelector('.say').textContent = p.say;
    row.querySelector('.meta').textContent = p.tags.join(' · ') + (p.label ? ` · label: ${p.label}` : '');
    const button = document.createElement('button');
    button.textContent = '● Record';
    button.onclick = () => toggle(p, button, row).catch((e) => toast(e.message));
    row.appendChild(button);
    container.appendChild(row);
  }
}

$('manifest').onclick = () => {
  const lines = prompts
    .filter((p) => clips.has(p.id))
    .map(({ say, ...rest }) => JSON.stringify({ ...rest, audio: `audio/${rest.id}.wav` }));
  if (!lines.length) return toast('Record at least one prompt first.');
  download(new Blob([lines.join('\n') + '\n'], { type: 'application/jsonl' }), 'manifest.jsonl');
};

prompts = await (await fetch('/eval-prompts.json')).json();
render();
