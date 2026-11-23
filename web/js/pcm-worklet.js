// Runs on the audio rendering thread.
// Converts Float32 microphone samples to 16-bit PCM and posts fixed 50 ms
// chunks to the main thread (AssemblyAI recommends 50-100 ms chunks), plus an
// RMS level for the mic meter. The node writes no output, so connecting it to
// the destination (to keep it running) plays silence.
class PcmChunker extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunkSamples = Math.round(sampleRate * 0.05);
    this.buf = new Int16Array(this.chunkSamples);
    this.n = 0;
    this.sumSq = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) {
      for (let i = 0; i < channel.length; i++) {
        const s = Math.max(-1, Math.min(1, channel[i]));
        this.sumSq += s * s;
        this.buf[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
        if (this.n === this.chunkSamples) {
          const rms = Math.sqrt(this.sumSq / this.chunkSamples);
          this.port.postMessage({ pcm: this.buf.buffer, rms }, [this.buf.buffer]);
          this.buf = new Int16Array(this.chunkSamples);
          this.n = 0;
          this.sumSq = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor('pcm-chunker', PcmChunker);
