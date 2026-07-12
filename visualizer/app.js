// Compositor: renders each visual "source" (layer) to its own offscreen
// framebuffer, then composites them onto the visible canvas. Layer params
// (enabled, opacity, blend mode, reactivity) come from the Sources panel;
// audio uniforms come from AudioMixer.

const canvas = document.getElementById('gl');
const gl = canvas.getContext('webgl2', { antialias: false, powerPreference: 'high-performance' });
if (!gl) {
  document.body.innerHTML = '<p style="color:#eee;font-family:sans-serif;padding:2rem">WebGL2 is required and is not available in this browser.</p>';
  throw new Error('no webgl2');
}

function compile(type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    console.error(log, src);
    throw new Error(log);
  }
  return sh;
}

function makeProgram(vsSrc, fsSrc) {
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    console.error(log);
    throw new Error(log);
  }
  return prog;
}

// fullscreen triangle, shared by every pass
const quadBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
function bindQuad(prog) {
  const loc = gl.getAttribLocation(prog, 'aPos');
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
}

const shaderSrc = window.Shaders;
const progGold = makeProgram(shaderSrc.VS_FULLSCREEN, shaderSrc.FS_GOLD);
const progRings = makeProgram(shaderSrc.VS_FULLSCREEN, shaderSrc.FS_RINGS);
const progComposite = makeProgram(shaderSrc.VS_FULLSCREEN, shaderSrc.FS_COMPOSITE);

function makeLayerTarget() {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { tex, fbo, w: 0, h: 0 };
}

function resizeLayerTarget(target, w, h) {
  if (target.w === w && target.h === h) return;
  target.w = w; target.h = h;
  gl.bindTexture(gl.TEXTURE_2D, target.tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
}

const layerA = makeLayerTarget(); // molten gold
const layerB = makeLayerTarget(); // beat rings

// ---- state driven by the Sources panel ----
const state = {
  layers: {
    gold: { enabled: true, opacity: 1.0, reactivity: 1.0 },
    rings: { enabled: true, opacity: 0.8, reactivity: 1.0, blend: 1 }, // 0 normal,1 add,2 screen
  },
};

let seed = Math.random() * 100.0;
let mouseTarget = [0, 0];
let mouseSmooth = [0, 0];

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 1.75);
  const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  resizeLayerTarget(layerA, w, h);
  resizeLayerTarget(layerB, w, h);
}
window.addEventListener('resize', resize);

canvas.addEventListener('pointermove', (e) => {
  const rect = canvas.getBoundingClientRect();
  mouseTarget = [
    ((e.clientX - rect.left) / rect.width) * 2 - 1,
    -(((e.clientY - rect.top) / rect.height) * 2 - 1),
  ];
});
canvas.addEventListener('pointerdown', () => { seed = Math.random() * 100.0; });

function setAudioUniforms(prog, mixer, reactivity) {
  gl.uniform2f(gl.getUniformLocation(prog, 'uResolution'), canvas.width, canvas.height);
  gl.uniform1f(gl.getUniformLocation(prog, 'uTime'), performance.now() / 1000);
  gl.uniform2f(gl.getUniformLocation(prog, 'uMouse'), mouseSmooth[0], mouseSmooth[1]);
  gl.uniform1f(gl.getUniformLocation(prog, 'uSeed'), seed);
  gl.uniform1f(gl.getUniformLocation(prog, 'uBass'), mixer.bands.bass);
  gl.uniform1f(gl.getUniformLocation(prog, 'uMid'), mixer.bands.mid);
  gl.uniform1f(gl.getUniformLocation(prog, 'uTreble'), mixer.bands.treble);
  gl.uniform1f(gl.getUniformLocation(prog, 'uAmp'), mixer.bands.amp);
  gl.uniform1f(gl.getUniformLocation(prog, 'uBeat'), mixer.beatPulse);
  gl.uniform1f(gl.getUniformLocation(prog, 'uReactivity'), reactivity);
}

function renderLayer(prog, target, mixer, reactivity) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
  gl.viewport(0, 0, target.w, target.h);
  gl.useProgram(prog);
  bindQuad(prog);
  setAudioUniforms(prog, mixer, reactivity);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

function composite() {
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.useProgram(progComposite);
  bindQuad(progComposite);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, layerA.tex);
  gl.uniform1i(gl.getUniformLocation(progComposite, 'uLayerA'), 0);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, layerB.tex);
  gl.uniform1i(gl.getUniformLocation(progComposite, 'uLayerB'), 1);

  gl.uniform1f(gl.getUniformLocation(progComposite, 'uOpacityA'), state.layers.gold.opacity);
  gl.uniform1f(gl.getUniformLocation(progComposite, 'uOpacityB'), state.layers.rings.opacity);
  gl.uniform1i(gl.getUniformLocation(progComposite, 'uBlendB'), state.layers.rings.blend);
  gl.uniform1i(gl.getUniformLocation(progComposite, 'uEnabledA'), state.layers.gold.enabled ? 1 : 0);
  gl.uniform1i(gl.getUniformLocation(progComposite, 'uEnabledB'), state.layers.rings.enabled ? 1 : 0);

  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

// ---- main loop ----
window.mixer = new AudioMixer();
let lastT = performance.now();

function frame(now) {
  const dt = Math.min(0.1, (now - lastT) / 1000);
  lastT = now;
  resize();

  mouseSmooth[0] += (mouseTarget[0] - mouseSmooth[0]) * 0.04;
  mouseSmooth[1] += (mouseTarget[1] - mouseSmooth[1]) * 0.04;

  window.mixer.update(dt);

  if (state.layers.gold.enabled) renderLayer(progGold, layerA, window.mixer, state.layers.gold.reactivity);
  if (state.layers.rings.enabled) renderLayer(progRings, layerB, window.mixer, state.layers.rings.reactivity);
  composite();

  window.dispatchEvent(new CustomEvent('vizframe', { detail: { mixer: window.mixer } }));
  requestAnimationFrame(frame);
}
resize();
requestAnimationFrame(frame);

window.vizState = state;
