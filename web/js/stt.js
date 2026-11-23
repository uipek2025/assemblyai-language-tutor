// Minimal browser client for AssemblyAI realtime STT (v3 WebSocket protocol).
//
// - Authenticates with a single-use temporary token minted by our server, so
//   the API key never reaches the browser.
// - Audio goes browser -> AssemblyAI directly: one network hop, lowest latency.
// - A turn is processed exactly once (dedupe on turn_order), whichever model
//   or formatting mode is in use.
// - Measures end-of-turn latency the way that matters for a voice agent:
//   how much audio had been sent past the last spoken word when the final
//   transcript arrived (silence wait + processing + network).

export class SttSession {
  constructor({ wsUrl, token, params, sampleRate, onTurn, onSpeechStarted, onClose, onError }) {
    Object.assign(this, { wsUrl, token, params, sampleRate, onTurn, onSpeechStarted, onClose, onError });
    this.samplesSent = 0;
    this.finalized = new Set();
    this.began = false;
    this.beginInfo = null;
  }

  connect() {
    const qs = new URLSearchParams({ ...this.params, sample_rate: String(this.sampleRate), token: this.token });
    this.ws = new WebSocket(`${this.wsUrl}?${qs}`);
    this.ws.binaryType = 'arraybuffer';

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Timed out waiting for the STT session to begin')), 10000);
      this.ws.onmessage = (ev) => {
        if (typeof ev.data !== 'string') return;
        let msg;
        try { msg = JSON.parse(ev.data); } catch { return; }
        if (msg.type === 'Begin') {
          clearTimeout(timer);
          this.began = true;
          this.beginInfo = msg;
          resolve(msg);
          return;
        }
        this.#handle(msg);
      };
      this.ws.onclose = (ev) => {
        clearTimeout(timer);
        if (!this.began) reject(new Error(`STT connection closed before start (${ev.code} ${ev.reason || ''})`));
        this.onClose?.(ev);
      };
      this.ws.onerror = () => this.onError?.(new Error('STT WebSocket error'));
    });
  }

  get isOpen() {
    return this.began && this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  /** Milliseconds of audio sent so far - the server's timeline for word timestamps. */
  get audioMs() {
    return (this.samplesSent / this.sampleRate) * 1000;
  }

  sendAudio(pcmBuffer) {
    if (!this.isOpen) return;
    this.ws.send(pcmBuffer);
    this.samplesSent += pcmBuffer.byteLength / 2;
  }

  /** Change turn detection, context, keyterms or language steering mid-session. */
  update(config) {
    if (this.isOpen) this.ws.send(JSON.stringify({ type: 'UpdateConfiguration', ...config }));
  }

  /** "I'm done speaking" - end the current turn immediately. */
  forceEndpoint() {
    if (this.isOpen) this.ws.send(JSON.stringify({ type: 'ForceEndpoint' }));
  }

  terminate() {
    if (this.isOpen) this.ws.send(JSON.stringify({ type: 'Terminate' }));
    setTimeout(() => { try { this.ws.close(); } catch { /* noop */ } }, 1500);
  }

  #handle(msg) {
    if (msg.type === 'Turn') {
      const final = !!msg.end_of_turn;
      if (final) {
        if (this.finalized.has(msg.turn_order)) return; // never process a turn twice
        this.finalized.add(msg.turn_order);
        const words = msg.words || [];
        const lastEnd = words.length ? words[words.length - 1].end : null;
        msg.eotLatencyMs = lastEnd != null ? Math.max(0, Math.round(this.audioMs - lastEnd)) : null;
      }
      this.onTurn?.(msg, final);
    } else if (msg.type === 'SpeechStarted') {
      this.onSpeechStarted?.(msg);
    } else if (msg.type === 'Termination') {
      /* session closed cleanly */
    } else if (msg.error || msg.type === 'Error') {
      this.onError?.(new Error(msg.error || JSON.stringify(msg)));
    }
  }
}
