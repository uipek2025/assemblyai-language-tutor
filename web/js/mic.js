// Microphone capture -> 16-bit PCM chunks via an AudioWorklet.
// Echo cancellation is on so the tutor's own voice is less likely to be
// transcribed when interruptions are enabled.

const CONSTRAINTS = {
  audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
};

async function buildGraph(stream, sampleRate) {
  const ctx = sampleRate ? new AudioContext({ sampleRate }) : new AudioContext();
  try {
    // Some browsers (e.g. Firefox) refuse to connect a 48 kHz mic to a 16 kHz
    // context; that throws here and we retry at the device's native rate.
    const source = ctx.createMediaStreamSource(stream);
    await ctx.audioWorklet.addModule(new URL('./pcm-worklet.js', import.meta.url));
    const node = new AudioWorkletNode(ctx, 'pcm-chunker');
    source.connect(node);
    node.connect(ctx.destination);
    if (ctx.state === 'suspended') await ctx.resume();
    return { ctx, source, node };
  } catch (err) {
    await ctx.close();
    throw err;
  }
}

export async function startMic({ onChunk, preferredSampleRate = 16000 }) {
  const stream = await navigator.mediaDevices.getUserMedia(CONSTRAINTS);
  let graph;
  try {
    graph = await buildGraph(stream, preferredSampleRate);
  } catch {
    // AssemblyAI accepts 8-96 kHz, so the native rate is fine - we just tell
    // the server which rate we are sending.
    graph = await buildGraph(stream, null);
  }
  graph.node.port.onmessage = (e) => onChunk(e.data.pcm, e.data.rms);

  return {
    sampleRate: graph.ctx.sampleRate,
    async stop() {
      try {
        graph.source.disconnect();
        graph.node.disconnect();
      } catch { /* already disconnected */ }
      stream.getTracks().forEach((t) => t.stop());
      await graph.ctx.close();
    },
  };
}
