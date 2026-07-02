// GLSL sources for the compositor. Each visual "source" is a full-screen
// fragment shader rendered to its own offscreen texture; the compositor then
// layers those textures together (see app.js). All layer shaders share the
// same audio uniforms so they react to the live mix.

const VS_FULLSCREEN = `#version 300 es
in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
`;

const NOISE_LIB = `
vec2 hash2(vec2 p, float seed) {
  p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
  return -1.0 + 2.0 * fract(sin(p + seed * 17.0) * 43758.5453123);
}
float vnoise(vec2 p, float seed) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(dot(hash2(i + vec2(0.0,0.0), seed), f - vec2(0.0,0.0)),
        dot(hash2(i + vec2(1.0,0.0), seed), f - vec2(1.0,0.0)), u.x),
    mix(dot(hash2(i + vec2(0.0,1.0), seed), f - vec2(0.0,1.0)),
        dot(hash2(i + vec2(1.0,1.0), seed), f - vec2(1.0,1.0)), u.x),
    u.y
  );
}
float fbm(vec2 p, float seed) {
  float sum = 0.0, amp = 0.5;
  mat2 rot = mat2(0.8, -0.6, 0.6, 0.8);
  for (int i = 0; i < 6; i++) {
    sum += amp * vnoise(p, seed);
    p = rot * p * 2.02;
    amp *= 0.52;
  }
  return sum;
}
`;

// Shared uniform block declared by every layer shader.
const AUDIO_UNIFORMS = `
uniform vec2  uResolution;
uniform float uTime;
uniform vec2  uMouse;
uniform float uSeed;
uniform float uBass;
uniform float uMid;
uniform float uTreble;
uniform float uAmp;
uniform float uBeat;
uniform float uReactivity;
`;

// ---- Layer A: molten gold domain-warped fractal flow ----
const FS_GOLD = `#version 300 es
precision highp float;
${AUDIO_UNIFORMS}
out vec4 fragColor;
${NOISE_LIB}

float warpedField(vec2 p, float t, float seed, out vec2 gradHint) {
  vec2 q = vec2(fbm(p + vec2(0.0, 0.0) + 0.05 * t, seed), fbm(p + vec2(5.2, 1.3) - 0.04 * t, seed));
  vec2 r = vec2(fbm(p + 4.0 * q + vec2(1.7, 9.2) + 0.09 * t, seed), fbm(p + 4.0 * q + vec2(8.3, 2.8) + 0.07 * t, seed));
  gradHint = r;
  return fbm(p + 4.0 * r, seed);
}

vec3 goldPalette(float t, float hueShift) {
  t = clamp((t - 0.58) * 1.9 + 0.42, 0.0, 1.0);
  vec3 c0 = vec3(0.05, 0.025, 0.01);
  vec3 c1 = vec3(0.42, 0.17, 0.02);
  vec3 c2 = vec3(0.92, 0.50, 0.06);
  vec3 c3 = vec3(1.00, 0.84, 0.28);
  vec3 c4 = vec3(1.00, 0.98, 0.86);
  vec3 col = mix(c0, c1, smoothstep(0.0, 0.28, t));
  col = mix(col, c2, smoothstep(0.24, 0.50, t));
  col = mix(col, c3, smoothstep(0.46, 0.72, t));
  col = mix(col, c4, smoothstep(0.70, 0.96, t));
  // subtle hue rotation driven by mids, keeps it in the warm range but adds motion
  float s = sin(hueShift), c = cos(hueShift);
  mat3 rot = mat3(1.0,0.0,0.0, 0.0,c,-s, 0.0,s,c);
  return clamp(col + (rot * col - col) * 0.35, 0.0, 1.5);
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution) / uResolution.y;
  vec2 drift = vec2(uTime * 0.015, -uTime * 0.011);
  vec2 p = uv * 2.6 + drift + uMouse * 0.35;

  float warpSpeed = (0.4 + uBass * 1.8 * uReactivity);
  float t = uTime * warpSpeed;
  vec2 gradHint;
  float f = warpedField(p, t, uSeed, gradHint);
  float field = f * 0.5 + 0.5;

  vec3 col = goldPalette(field, uMid * 2.2 * uReactivity);

  float e = 0.0025;
  vec2 gh;
  float fx = warpedField(p + vec2(e, 0.0), t, uSeed, gh);
  float fy = warpedField(p + vec2(0.0, e), t, uSeed, gh);
  vec3 normal = normalize(vec3(-(fx - f), -(fy - f), e * 6.0));
  vec3 lightDir = normalize(vec3(0.5, 0.65, 0.7));
  float diff = clamp(dot(normal, lightDir), 0.0, 1.0);
  col *= 0.55 + 0.65 * diff;

  vec3 viewDir = vec3(0.0, 0.0, 1.0);
  vec3 halfV = normalize(lightDir + viewDir);
  float spec = pow(clamp(dot(normal, halfV), 0.0, 1.0), 24.0);
  float sparkleField = fbm(p * 14.0 + 13.7 - t * 0.6, uSeed);
  float sparkleThresh = mix(0.55, 0.35, clamp(uTreble * uReactivity, 0.0, 1.0));
  float sparkleMask = smoothstep(sparkleThresh, 0.88, sparkleField);
  float twinkle = 0.5 + 0.5 * sin(uTime * 5.0 + sparkleField * 60.0);
  vec3 specCol = vec3(1.0, 0.97, 0.85) * spec * (5.5 + uTreble * 6.0 * uReactivity);
  specCol += vec3(1.0, 0.95, 0.72) * sparkleMask * twinkle * (2.6 + uTreble * 3.0 * uReactivity);
  col += specCol;

  float glow = smoothstep(0.6, 1.0, field);
  col += vec3(1.0, 0.6, 0.12) * glow * (0.4 + uAmp * 0.6 * uReactivity);

  // beat kick: brief warm flash across the whole field
  col += vec3(1.0, 0.7, 0.3) * uBeat * 0.18 * uReactivity;

  float streak = fbm(vec2(p.y * 3.0, t * 0.3), uSeed);
  float glitchLine = step(0.965, fract(streak * 4.0 + t * 0.05));
  vec2 shift = vec2(glitchLine * 0.006 * sin(t * 9.0 + p.y * 40.0), 0.0);
  vec3 chroma;
  chroma.r = goldPalette(field + 0.02 + shift.x, 0.0).r;
  chroma.g = col.g;
  chroma.b = goldPalette(field - 0.02 - shift.x, 0.0).b * 0.9;
  col = mix(col, chroma, glitchLine * 0.6);

  float vig = smoothstep(1.15, 0.2, length(uv));
  col *= mix(0.65, 1.0, vig);

  float grain = (fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233)) + uTime * 60.0) * 43758.5453) - 0.5) * 0.03;
  col += grain;

  col *= 1.15;
  col = col / (1.0 + col);
  col = pow(col, vec3(0.78));

  fragColor = vec4(col, 1.0);
}
`;

// ---- Layer B: beat-reactive expanding rings, layered additively over the gold ----
const FS_RINGS = `#version 300 es
precision highp float;
${AUDIO_UNIFORMS}
out vec4 fragColor;
${NOISE_LIB}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution) / uResolution.y;
  uv -= uMouse * 0.2;
  float r = length(uv);
  float ang = atan(uv.y, uv.x);

  float wobble = fbm(vec2(ang * 2.0, uTime * 0.2), uSeed) * 0.06 * (0.5 + uMid);
  float rings = sin((r + wobble) * 32.0 - uTime * (2.0 + uTreble * 6.0 * uReactivity));
  float ringMask = smoothstep(0.94, 1.0, rings);

  float pulse = uBeat * uReactivity;
  float pulseRing = smoothstep(0.02, 0.0, abs(r - pulse * 0.9 - 0.05));

  vec3 iceCol = mix(vec3(0.55, 0.85, 1.0), vec3(1.0, 1.0, 0.95), uTreble);
  vec3 col = iceCol * ringMask * (0.35 + 0.5 * uAmp * uReactivity);
  col += vec3(0.8, 0.95, 1.0) * pulseRing * 1.4;

  float vig = smoothstep(1.1, 0.1, r);
  col *= vig;

  fragColor = vec4(col, ringMask * 0.8 + pulseRing);
}
`;

// ---- Compositor: blends two layer textures onto the canvas ----
const FS_COMPOSITE = `#version 300 es
precision highp float;
uniform sampler2D uLayerA;
uniform sampler2D uLayerB;
uniform float uOpacityA;
uniform float uOpacityB;
uniform int   uBlendB; // 0 = normal(alpha), 1 = additive, 2 = screen
uniform bool  uEnabledA;
uniform bool  uEnabledB;
out vec4 fragColor;

void main() {
  vec2 uv = gl_FragCoord.xy / vec2(textureSize(uLayerA, 0));
  vec3 base = uEnabledA ? texture(uLayerA, uv).rgb * uOpacityA : vec3(0.0);

  if (uEnabledB) {
    vec4 b = texture(uLayerB, uv);
    vec3 bc = b.rgb * uOpacityB;
    if (uBlendB == 1) {
      base += bc;
    } else if (uBlendB == 2) {
      base = 1.0 - (1.0 - base) * (1.0 - bc);
    } else {
      base = mix(base, bc, clamp(b.a * uOpacityB, 0.0, 1.0));
    }
  }
  fragColor = vec4(base, 1.0);
}
`;

window.Shaders = { VS_FULLSCREEN, FS_GOLD, FS_RINGS, FS_COMPOSITE };
