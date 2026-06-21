const recordButton = document.querySelector("#recordButton");
const stopButton = document.querySelector("#stopButton");
const predictButton = document.querySelector("#predictButton");
const playback = document.querySelector("#playback");
const statusEl = document.querySelector("#status");
const timerEl = document.querySelector("#timer");
const resultLabel = document.querySelector("#resultLabel");
const positiveBar = document.querySelector("#positiveBar");
const negativeBar = document.querySelector("#negativeBar");
const positiveValue = document.querySelector("#positiveValue");
const negativeValue = document.querySelector("#negativeValue");
const canvas = document.querySelector("#meter");
const ctx = canvas.getContext("2d");

let audioContext;
let stream;
let source;
let processor;
let chunks = [];
let recordingStartedAt = 0;
let timerId;
let lastBlob;
let lastSamples = new Float32Array();
let lastSampleRate = 0;

const maxRecordingSeconds = 6;

function setStatus(value) {
  statusEl.textContent = value;
}

function setResult(result) {
  const positive = Math.max(0, Math.min(1, result.positive_probability || 0));
  const negative = Math.max(0, Math.min(1, result.negative_probability || 0));
  resultLabel.textContent = result.label ? result.label.toUpperCase() : "No result";
  positiveBar.style.width = `${Math.round(positive * 100)}%`;
  negativeBar.style.width = `${Math.round(negative * 100)}%`;
  positiveValue.textContent = `${Math.round(positive * 100)}%`;
  negativeValue.textContent = `${Math.round(negative * 100)}%`;
}

function drawMeter(samples = lastSamples) {
  const width = canvas.width;
  const height = canvas.height;
  ctx.fillStyle = "#101828";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#5eead4";
  ctx.lineWidth = 2;
  ctx.beginPath();

  const step = Math.max(1, Math.floor(samples.length / width));
  const mid = height / 2;
  for (let x = 0; x < width; x += 1) {
    const index = x * step;
    const value = samples[index] || 0;
    const y = mid + value * mid * 0.86;
    if (x === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
}

function updateTimer() {
  const elapsed = (performance.now() - recordingStartedAt) / 1000;
  timerEl.textContent = elapsed.toFixed(1).padStart(4, "0");
  if (elapsed >= maxRecordingSeconds) {
    stopRecording();
  }
}

function mergeChunks(buffers) {
  const totalLength = buffers.reduce((sum, buffer) => sum + buffer.length, 0);
  const result = new Float32Array(totalLength);
  let offset = 0;
  for (const buffer of buffers) {
    result.set(buffer, offset);
    offset += buffer.length;
  }
  return result;
}

function floatTo16BitPcm(view, offset, input) {
  for (let i = 0; i < input.length; i += 1, offset += 2) {
    const sample = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i += 1) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);
  floatTo16BitPcm(view, 44, samples);
  return new Blob([view], { type: "audio/wav" });
}

async function startRecording() {
  stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  audioContext = new AudioContext();
  chunks = [];
  lastBlob = null;
  lastSampleRate = audioContext.sampleRate;
  source = audioContext.createMediaStreamSource(stream);
  processor = audioContext.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    const copy = new Float32Array(input);
    chunks.push(copy);
    lastSamples = copy;
    drawMeter(copy);
  };
  source.connect(processor);
  processor.connect(audioContext.destination);

  recordButton.disabled = true;
  stopButton.disabled = false;
  predictButton.disabled = true;
  playback.removeAttribute("src");
  setResult({});
  setStatus("Recording");
  recordingStartedAt = performance.now();
  timerId = window.setInterval(updateTimer, 100);
  updateTimer();
}

function stopRecording() {
  if (timerId) {
    window.clearInterval(timerId);
    timerId = null;
  }
  if (processor) {
    processor.disconnect();
    processor.onaudioprocess = null;
    processor = null;
  }
  if (source) {
    source.disconnect();
    source = null;
  }
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }

  const samples = mergeChunks(chunks);
  lastSamples = samples;
  lastBlob = encodeWav(samples, lastSampleRate);
  playback.src = URL.createObjectURL(lastBlob);
  drawMeter(samples);

  recordButton.disabled = false;
  stopButton.disabled = true;
  predictButton.disabled = samples.length === 0;
  setStatus("Recorded");
}

async function predictRecording() {
  if (!lastBlob) {
    return;
  }
  setStatus("Analyzing");
  predictButton.disabled = true;
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "audio/wav" },
      body: lastBlob,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Prediction failed");
    }
    setResult(payload);
    setStatus("Done");
  } catch (error) {
    resultLabel.textContent = error.message;
    setStatus("Error");
  } finally {
    predictButton.disabled = false;
  }
}

recordButton.addEventListener("click", () => {
  startRecording().catch((error) => {
    setStatus("Error");
    resultLabel.textContent = error.message;
    recordButton.disabled = false;
    stopButton.disabled = true;
  });
});
stopButton.addEventListener("click", stopRecording);
predictButton.addEventListener("click", predictRecording);

drawMeter();
