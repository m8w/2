// Audio mixer engine: two input channels (file playback + live mic), each with
// its own gain (fader) and analyser (for metering), summed into a mix bus that
// feeds a master analyser. Frequency-band energies and a beat/transient pulse
// are extracted every frame and exposed for the visual layers to consume.

class Channel {
  constructor(ctx, mixBus, name) {
    this.ctx = ctx;
    this.name = name;
    this.sourceNode = null;
    this.gainNode = ctx.createGain();
    this.gainNode.gain.value = 0.8;
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.analyser.smoothingTimeConstant = 0.0; // we do our own smoothing
    this.muted = false;
    this._preMuteGain = 0.8;
    this.gainNode.connect(this.analyser);
    this.gainNode.connect(mixBus);
    this._freqData = new Uint8Array(this.analyser.frequencyBinCount);
  }

  connectSource(node) {
    if (this.sourceNode) {
      try { this.sourceNode.disconnect(); } catch (e) {}
    }
    this.sourceNode = node;
    node.connect(this.gainNode);
  }

  setGain(v) {
    this._preMuteGain = v;
    if (!this.muted) this.gainNode.gain.setTargetAtTime(v, this.ctx.currentTime, 0.01);
  }

  setMuted(m) {
    this.muted = m;
    this.gainNode.gain.setTargetAtTime(m ? 0.0 : this._preMuteGain, this.ctx.currentTime, 0.01);
  }

  // RMS level 0..1 for meters
  level() {
    this.analyser.getByteTimeDomainData(this._freqData);
    let sum = 0;
    for (let i = 0; i < this._freqData.length; i++) {
      const v = (this._freqData[i] - 128) / 128;
      sum += v * v;
    }
    return Math.sqrt(sum / this._freqData.length);
  }
}

class AudioMixer {
  constructor() {
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.mixBus = this.ctx.createGain();
    this.mixBus.gain.value = 1.0;

    this.masterAnalyser = this.ctx.createAnalyser();
    this.masterAnalyser.fftSize = 2048;
    this.masterAnalyser.smoothingTimeConstant = 0.0;
    this.mixBus.connect(this.masterAnalyser);
    this.masterAnalyser.connect(this.ctx.destination);

    this.file = new Channel(this.ctx, this.mixBus, 'file');
    this.mic = new Channel(this.ctx, this.mixBus, 'mic');

    this._freq = new Uint8Array(this.masterAnalyser.frequencyBinCount);
    this._time = new Uint8Array(this.masterAnalyser.fftSize);

    // smoothed band energies
    this.bands = { bass: 0, mid: 0, treble: 0, amp: 0 };
    this._bandAttack = 0.55;
    this._bandRelease = 0.08;

    // beat detection: short-term bass energy vs a slower running average
    this.beatPulse = 0;
    this._bassAvg = 0;
    this._beatCooldown = 0;
  }

  async loadFile(file) {
    const url = URL.createObjectURL(file);
    if (!this.audioEl) {
      this.audioEl = new Audio();
      this.audioEl.crossOrigin = 'anonymous';
    }
    this.audioEl.src = url;
    this.audioEl.loop = true;
    if (!this._fileNode) {
      this._fileNode = this.ctx.createMediaElementSource(this.audioEl);
      this.file.connectSource(this._fileNode);
    }
    await this.ctx.resume();
    await this.audioEl.play();
  }

  togglePlay() {
    if (!this.audioEl) return;
    if (this.audioEl.paused) this.audioEl.play(); else this.audioEl.pause();
  }

  async enableMic() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    const node = this.ctx.createMediaStreamSource(stream);
    this.mic.connectSource(node);
    await this.ctx.resume();
  }

  // call once per animation frame
  update(dt) {
    const analyser = this.masterAnalyser;
    analyser.getByteFrequencyData(this._freq);
    analyser.getByteTimeDomainData(this._time);

    const sampleRate = this.ctx.sampleRate;
    const binHz = sampleRate / analyser.fftSize;
    const bandAvg = (fLo, fHi) => {
      const iLo = Math.max(1, Math.floor(fLo / binHz));
      const iHi = Math.min(this._freq.length - 1, Math.ceil(fHi / binHz));
      if (iHi <= iLo) return 0;
      let sum = 0;
      for (let i = iLo; i < iHi; i++) sum += this._freq[i];
      return (sum / (iHi - iLo)) / 255;
    };

    const rawBass = bandAvg(20, 160);
    const rawMid = bandAvg(160, 2000);
    const rawTreble = bandAvg(2000, 9000);

    let rmsSum = 0;
    for (let i = 0; i < this._time.length; i++) {
      const v = (this._time[i] - 128) / 128;
      rmsSum += v * v;
    }
    const rawAmp = Math.sqrt(rmsSum / this._time.length);

    const smooth = (prev, raw) => {
      const k = raw > prev ? this._bandAttack : this._bandRelease;
      return prev + (raw - prev) * k;
    };
    this.bands.bass = smooth(this.bands.bass, rawBass);
    this.bands.mid = smooth(this.bands.mid, rawMid);
    this.bands.treble = smooth(this.bands.treble, rawTreble);
    this.bands.amp = smooth(this.bands.amp, rawAmp);

    // beat: instantaneous bass spikes above a slow running average
    this._bassAvg += (rawBass - this._bassAvg) * 0.02;
    this._beatCooldown = Math.max(0, this._beatCooldown - dt);
    if (rawBass > this._bassAvg * 1.35 + 0.05 && this._beatCooldown <= 0) {
      this.beatPulse = 1.0;
      this._beatCooldown = 0.12;
    }
    this.beatPulse *= Math.pow(0.02, dt); // fast exponential decay
  }
}

window.AudioMixer = AudioMixer;
