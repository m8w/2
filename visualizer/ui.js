// Wires the DOM controls (Sources panel + Audio Mixer panel) to the
// AudioMixer (audio.js) and the layer state consumed by the compositor (app.js).

document.querySelectorAll('.source-row').forEach((row) => {
  const layerName = row.dataset.layer;
  const layer = window.vizState.layers[layerName];

  const enabledEl = row.querySelector('.src-enabled');
  enabledEl.checked = layer.enabled;
  enabledEl.addEventListener('change', () => { layer.enabled = enabledEl.checked; });

  const opacityEl = row.querySelector('.src-opacity');
  opacityEl.value = layer.opacity;
  opacityEl.addEventListener('input', () => { layer.opacity = parseFloat(opacityEl.value); });

  const reactivityEl = row.querySelector('.src-reactivity');
  reactivityEl.value = layer.reactivity;
  reactivityEl.addEventListener('input', () => { layer.reactivity = parseFloat(reactivityEl.value); });

  const blendEl = row.querySelector('.src-blend');
  if (blendEl) {
    blendEl.value = String(layer.blend);
    blendEl.addEventListener('change', () => { layer.blend = parseInt(blendEl.value, 10); });
  }
});

// ---- File channel ----
const fileInput = document.getElementById('file-input');
const fileLoadBtn = document.getElementById('file-load-btn');
const filePlayBtn = document.getElementById('file-play-btn');
const fileMuteBtn = document.getElementById('file-mute-btn');
const fileGain = document.getElementById('file-gain');
const fileName = document.getElementById('file-name');

fileLoadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', async () => {
  const f = fileInput.files[0];
  if (!f) return;
  fileName.textContent = f.name;
  await window.mixer.loadFile(f);
  filePlayBtn.disabled = false;
  filePlayBtn.textContent = 'Pause';
});
filePlayBtn.addEventListener('click', () => {
  window.mixer.togglePlay();
  setTimeout(() => {
    filePlayBtn.textContent = window.mixer.audioEl && window.mixer.audioEl.paused ? 'Play' : 'Pause';
  }, 50);
});
fileMuteBtn.addEventListener('click', () => {
  const m = !window.mixer.file.muted;
  window.mixer.file.setMuted(m);
  fileMuteBtn.classList.toggle('muted', m);
});
fileGain.addEventListener('input', () => window.mixer.file.setGain(parseFloat(fileGain.value)));
window.mixer.file.setGain(parseFloat(fileGain.value));

// ---- Mic channel ----
const micEnableBtn = document.getElementById('mic-enable-btn');
const micMuteBtn = document.getElementById('mic-mute-btn');
const micGain = document.getElementById('mic-gain');

micEnableBtn.addEventListener('click', async () => {
  micEnableBtn.disabled = true;
  micEnableBtn.textContent = '...';
  try {
    await window.mixer.enableMic();
    micEnableBtn.textContent = 'Enabled';
    micEnableBtn.classList.add('active');
  } catch (e) {
    micEnableBtn.disabled = false;
    micEnableBtn.textContent = 'Enable';
    alert('Microphone access failed: ' + e.message);
  }
});
micMuteBtn.addEventListener('click', () => {
  const m = !window.mixer.mic.muted;
  window.mixer.mic.setMuted(m);
  micMuteBtn.classList.toggle('muted', m);
});
micGain.addEventListener('input', () => window.mixer.mic.setGain(parseFloat(micGain.value)));
window.mixer.mic.setGain(parseFloat(micGain.value));

// ---- Meters, updated every rendered frame ----
const fileMeter = document.getElementById('file-meter');
const micMeter = document.getElementById('mic-meter');
const bassMeter = document.getElementById('bass-meter');
const midMeter = document.getElementById('mid-meter');
const trebleMeter = document.getElementById('treble-meter');

window.addEventListener('vizframe', (e) => {
  const mixer = e.detail.mixer;
  fileMeter.style.width = Math.min(100, mixer.file.level() * 220) + '%';
  micMeter.style.width = Math.min(100, mixer.mic.level() * 220) + '%';
  bassMeter.style.width = Math.min(100, mixer.bands.bass * 100) + '%';
  midMeter.style.width = Math.min(100, mixer.bands.mid * 100) + '%';
  trebleMeter.style.width = Math.min(100, mixer.bands.treble * 100) + '%';
});
