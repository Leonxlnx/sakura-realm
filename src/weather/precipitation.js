/**
 * precipitation.js - rain, rain impact, snow and lightning.
 *
 * Design notes worth reading before editing:
 *
 * 1. RAIN and SNOW share one idea: a box-shaped volume *centred on the camera*
 *    that wraps particles modulo the box instead of respawning them. Wrapping
 *    keeps density exactly constant while the player walks or flies and costs one
 *    `mod()`; respawning always produces visible density waves behind a moving
 *    camera. Everything in the camera-following group stores camera-relative
 *    positions, and the group itself is translated to the camera each frame - 
 *    that keeps float precision high and makes the wrap a single subtraction.
 *
 * 2. Rain is fully analytic on the GPU (position = f(seed, time)); it is by far
 *    the largest particle count and must never touch the CPU. Snow is
 *    CPU-integrated because it has to be advected by `state.wind.sample`, which
 *    is a CPU function; the wind fetch is amortised round-robin over six frames.
 *
 * 3. Every primitive here goes sub-pixel at distance, and thin sub-pixel quads
 *    are the worst source of shimmer in any weather effect. All four materials
 *    widen the primitive to a minimum pixel footprint and scale alpha by the
 *    inverse - energy-conserving anti-aliasing, effectively free.
 *
 * 4. Splash and droplet work is strictly camera-local. Nobody sees a splash at
 *    80 m, so nothing is spent there.
 */

import * as THREE from 'three';
import { EVENTS } from '../core/state.js';
import { clamp01, lerp, smoothstep, damp, makeRNG, TAU } from '../core/math.js';

// ---------------------------------------------------------------------------
// Quality tiers. Typed arrays are always allocated at the ULTRA size (~1.2 MB
// total) so a tier change never reallocates a GPU buffer mid-flight; only the
// active counts and the volume extents move. Particle density per cubic metre is
// held roughly constant across tiers - LOW buys its speed with a shorter view
// distance into the rain, which reads as heavier haze rather than thinner rain.
// ---------------------------------------------------------------------------

const TIERS = {
  low:    { rain: 4200,  snow: 1500, splash: 64,  drops: 20, rainBox: [46, 30, 46], snowBox: [30, 22, 30], rainFade: 21, snowFade: 15, splashR: 6.5,  minPx: 1.50, crown: false, boltSeg: 160 },
  medium: { rain: 8200,  snow: 2600, splash: 128, drops: 36, rainBox: [58, 36, 58], snowBox: [38, 26, 38], rainFade: 26, snowFade: 19, splashR: 8.0,  minPx: 1.40, crown: true,  boltSeg: 256 },
  high:   { rain: 14000, snow: 4000, splash: 224, drops: 56, rainBox: [72, 42, 72], snowBox: [46, 30, 46], rainFade: 33, snowFade: 23, splashR: 10.0, minPx: 1.35, crown: true,  boltSeg: 384 },
  ultra:  { rain: 21000, snow: 6000, splash: 320, drops: 80, rainBox: [84, 48, 84], snowBox: [54, 34, 54], rainFade: 39, snowFade: 27, splashR: 12.0, minPx: 1.30, crown: true,  boltSeg: 512 },
};
const MAX = TIERS.ultra;

const MAX_BOLT_POINTS = 1024;
const MAX_BOLT_SEGS = TIERS.ultra.boltSeg;
const MAX_CHANNELS = 20;
const THUNDER_EVENT = 'weather:thunder';
const SPEED_OF_SOUND = 343;

// ---------------------------------------------------------------------------
// Module-scope scratch. Nothing in update() may allocate.
// ---------------------------------------------------------------------------

const _camPos = new THREE.Vector3();
const _prevCam = new THREE.Vector3();
const _camDelta = new THREE.Vector3();
const _fwd = new THREE.Vector3();
const _v3 = new THREE.Vector3();
const _n3 = new THREE.Vector3();
const _col = new THREE.Color();
const _col2 = new THREE.Color();
const _size = new THREE.Vector2();
const _windOut = { x: 0, y: 0, z: 0 };
const _strikeEvent = { distance: 0, intensity: 0, visible: false, thunderDelay: 0, position: new THREE.Vector3() };
const _thunderEvent = { distance: 0, intensity: 0 };
const _quadSide = [-1, 1, 1, -1];
const _quadEnd = [0, 0, 1, 1];

// Snow flutter frequencies, quantised into buckets. Eight distinct spiral rates
// is far more variety than the eye can separate, and it turns three transcendental
// calls per flake per frame into eight for the whole field.
const SNOW_BUCKETS = 8;
const SNOW_FREQ = new Float32Array(SNOW_BUCKETS);
for (let k = 0; k < SNOW_BUCKETS; k++) SNOW_FREQ[k] = 0.5 + (k / (SNOW_BUCKETS - 1)) * 1.5;
const _rotC = new Float32Array(SNOW_BUCKETS);
const _rotS = new Float32Array(SNOW_BUCKETS);

// ---------------------------------------------------------------------------
// Shared GLSL
// ---------------------------------------------------------------------------

const GLSL_COMMON = /* glsl */ `
  // Fold a world point into the camera-centred volume. Because the fold is
  // relative to the camera, moving never changes local density.
  vec3 wrapVolume(vec3 world, vec3 cam, vec3 box) {
    return mod(world - cam + box * 0.5, box) - box * 0.5;
  }

  // Fade before the wrap seam (Chebyshev, so the box corners still get used) and
  // again with range, so nothing ever pops into existence at the boundary.
  float volumeFade(vec3 rel, vec3 box, float fadeEnd, float dist) {
    vec3 n = abs(rel) / (box * 0.5);
    float edge = max(max(n.x, n.y), n.z);
    return (1.0 - smoothstep(0.76, 0.99, edge)) * (1.0 - smoothstep(fadeEnd * 0.62, fadeEnd, dist));
  }

  // Grow a primitive to a minimum pixel footprint and return the brightness scale
  // that keeps its integrated energy unchanged. Without this, distant rain and
  // snow scintillate: the classic "that's a particle system" tell.
  float pixelWiden(inout float w, float dist, float pixelScale, float minPixels) {
    float widened = max(w, dist * pixelScale * minPixels);
    float e = w / widened;
    w = widened;
    return e;
  }

  // The tree keeps the rain off you. uCanopy = (centre.xyz, radius); w <= 0 disables.
  float canopyShelter(vec3 world, vec4 canopy, float strength) {
    if (canopy.w <= 0.0) return 0.0;
    float r = length(world.xz - canopy.xz);
    float radial = 1.0 - smoothstep(canopy.w * 0.5, canopy.w * 1.06, r);
    float below = smoothstep(canopy.y + 1.5, canopy.y - 2.0, world.y);
    return radial * below * strength;
  }

  // Rain and snow arrive in curtains, not as a uniform field. Two incommensurable
  // travelling waves give a slow non-repeating swell in density.
  float precipCurtain(vec2 xz, float t, float amount) {
    float a = sin(dot(xz, vec2(0.061, 0.043)) - t * 0.55);
    float b = sin(dot(xz, vec2(-0.027, 0.035)) - t * 0.31 + 1.7);
    return 1.0 - amount * (0.5 - 0.5 * (a * 0.6 + b * 0.4));
  }
`;

const GLSL_LIGHT_UNIFORMS = /* glsl */ `
  uniform vec3 uCamPos, uKeyDir, uKeyLight, uSkyLight, uFogColor, uFlashColor;
  uniform vec4 uCanopy;
  uniform float uTime, uPixelScale, uMinPixels, uFlash, uSunBreak;
`;

// ---------------------------------------------------------------------------
// Rain
// ---------------------------------------------------------------------------

const RAIN_VERT = /* glsl */ `
  ${GLSL_LIGHT_UNIFORMS}
  uniform vec3 uBox, uCamVel;
  uniform vec2 uWind;
  uniform float uFall, uStretch, uWidth, uAlpha, uFadeEnd, uCount;

  attribute vec3 aSeed;
  attribute vec4 aRand;
  attribute float aIndex;

  varying vec3 vColor;
  varying float vAlpha;
  varying vec2 vQ;

  ${GLSL_COMMON}

  void main() {
    float speed = uFall * aRand.x;
    // Small drops are dragged closer to true wind speed than large ones. That
    // spread is what makes wind-driven rain read as depth instead of a wall.
    float lag = 0.66 + 0.46 * aRand.w;

    vec3 world = aSeed * uBox;
    world.y -= speed * uTime;
    world.xz += uWind * (uTime * lag);
    float wob = aRand.w * 43.7 + uTime * 1.7;
    world.x += 0.05 * sin(wob);
    world.z += 0.05 * cos(wob * 0.83);

    vec3 rel = wrapVolume(world, uCamPos, uBox);
    float dist = length(rel);
    vec3 Vp = -rel / max(dist, 1e-4);          // fragment -> camera

    // Motion blur is about velocity *relative to the observer*: fly forward and
    // the rain rotates to streak at you, exactly like a windscreen at speed.
    // Only the smear is affected - the drops themselves stay world-anchored.
    vec3 vel = vec3(uWind.x * lag, -speed, uWind.y * lag) - uCamVel;
    float vlen = length(vel);
    vec3 axis = vel / max(vlen, 1e-4);

    // The streak is a motion-blur smear: oriented along true velocity, length is
    // speed times shutter time, so wind shear tilts the entire field correctly.
    vec3 sideRaw = cross(axis, Vp);
    float sinA = length(sideRaw);
    vec3 side = sinA > 1e-3 ? sideRaw / sinA : vec3(1.0, 0.0, 0.0);

    float len = vlen * uStretch * (0.68 + 0.7 * aRand.y);
    float w = uWidth * (0.75 + 0.6 * aRand.y);
    float energy = pixelWiden(w, dist, uPixelScale, uMinPixels);

    // Drops inside the near focal plane are out of focus, not razor sharp.
    float coc = smoothstep(2.4, 0.35, dist);
    w += coc * 0.055;
    len = mix(len, len * 0.75, coc);
    energy *= 1.0 - 0.62 * coc;

    vec3 worldPos = uCamPos + rel;
    float fade = volumeFade(rel, uBox, uFadeEnd, dist);
    fade *= smoothstep(0.16, 0.85, dist);
    fade *= 1.0 - canopyShelter(worldPos, uCanopy, 0.92);
    // Instances past the active count fade out over a narrow band rather than
    // being cut, so changing intensity never pops a block of drops in or out.
    fade *= 1.0 - smoothstep(uCount - 0.06, uCount, aIndex);
    fade *= precipCurtain(worldPos.xz, uTime, 0.45);
    // Seen end-on a streak is a dot, not a bar - dim it rather than show a slab.
    float a = uAlpha * aRand.z * energy * fade * mix(0.16, 1.0, sinA);

    if (a < 0.0035) {
      vColor = vec3(0.0); vAlpha = 0.0; vQ = vec2(0.0);
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);   // outside clip space: the quad dies
      return;
    }

    vec3 p = rel + axis * (position.y * len) + side * (position.x * w);

    // A rain streak is a dielectric cylinder - lit exactly like a hair strand,
    // so Kajiya-Kay is both the physically right and the cheapest model here.
    float TdotL = dot(axis, uKeyDir);
    float TdotV = dot(axis, Vp);
    float sinTL = sqrt(max(0.0, 1.0 - TdotL * TdotL));
    float sinTV = sqrt(max(0.0, 1.0 - TdotV * TdotV));
    float spec = pow(max(0.0, sinTL * sinTV - TdotL * TdotV), 22.0);
    // Drops refract strongly forward: rain glows when you look toward the sun.
    float fwd = pow(max(0.0, dot(-Vp, uKeyDir)), 5.0);

    vec3 col = uSkyLight * (0.72 + 0.4 * aRand.y)
             + uKeyLight * uSunBreak * (0.10 + spec * 3.2 + fwd * 1.1)
             + uFlashColor * (uFlash * 1.7);
    col = mix(col, uFogColor, smoothstep(0.12, 1.0, dist / uFadeEnd) * 0.5);

    vColor = col;
    vAlpha = a;
    vQ = position.xy * 2.0;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }
`;

const RAIN_FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  varying vec2 vQ;

  void main() {
    float across = 1.0 - vQ.x * vQ.x;
    float along = 1.0 - smoothstep(0.5, 1.0, abs(vQ.y));
    float a = vAlpha * max(across, 0.0) * sqrt(max(across, 0.0)) * along;
    if (a < 0.002) discard;
    gl_FragColor = vec4(vColor, a);
  }
`;

// ---------------------------------------------------------------------------
// Snow
// ---------------------------------------------------------------------------

const SNOW_VERT = /* glsl */ `
  ${GLSL_LIGHT_UNIFORMS}
  uniform vec3 uBox;
  uniform float uSize, uAlpha, uFadeEnd, uCount;

  attribute vec3 aOffset;   // camera-relative, CPU-integrated
  attribute vec4 aRand;     // x size, y tumble rate, z opacity, w phase
  attribute float aIndex;

  varying vec3 vColor;
  varying float vAlpha;
  varying vec2 vUvS;

  ${GLSL_COMMON}

  void main() {
    vec3 rel = aOffset;
    float dist = length(rel);
    vec3 Vp = -rel / max(dist, 1e-4);

    // Tumble. A flake is a flat plate: it spins and periodically turns edge-on,
    // which reads as a twinkle. Without this, snow is a field of sliding dots.
    float ph = aRand.w * 6.28318 + uTime * (0.7 + 1.9 * aRand.y);
    // (named faceOn, not flat: 'flat' is a reserved interpolation qualifier in
    // GLSL ES 3.00, which three transpiles this shader to on WebGL2.)
    float faceOn = abs(cos(ph * 0.57));
    float rot = ph * 0.5;
    float cs = cos(rot), sn = sin(rot);
    vec2 q = vec2(position.x * mix(0.14, 1.0, faceOn), position.y);
    q = vec2(q.x * cs - q.y * sn, q.x * sn + q.y * cs);

    float w = uSize * (0.5 + 1.0 * aRand.x);
    float energy = pixelWiden(w, dist, uPixelScale, uMinPixels);
    float coc = smoothstep(2.8, 0.3, dist);
    w *= 1.0 + coc * 2.4;
    energy *= 1.0 - 0.74 * coc;

    vec3 worldPos = uCamPos + rel;
    float fade = volumeFade(rel, uBox, uFadeEnd, dist);
    fade *= smoothstep(0.12, 0.7, dist);
    fade *= 1.0 - canopyShelter(worldPos, uCanopy, 0.5);
    fade *= 1.0 - smoothstep(uCount - 0.06, uCount, aIndex);
    fade *= precipCurtain(worldPos.xz, uTime * 0.6, 0.3);
    float a = uAlpha * aRand.z * energy * fade * mix(0.4, 1.0, faceOn);

    if (a < 0.003) {
      vColor = vec3(0.0); vAlpha = 0.0; vUvS = vec2(0.5);
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      return;
    }

    // Camera basis straight out of the view matrix - stable while the flake spins.
    vec3 right = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
    vec3 up    = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);
    vec3 p = rel + right * (q.x * w) + up * (q.y * w);

    float fwd = pow(max(0.0, dot(-Vp, uKeyDir)), 3.0);
    vec3 col = uSkyLight * 1.05
             + uKeyLight * uSunBreak * (0.30 + 0.75 * fwd)
             + uFlashColor * (uFlash * 1.3);
    col = mix(col, uFogColor, smoothstep(0.1, 1.0, dist / uFadeEnd) * 0.62);

    vColor = col;
    vAlpha = a;
    vUvS = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }
`;

const SNOW_FRAG = /* glsl */ `
  uniform sampler2D uSprite;
  varying vec3 vColor;
  varying float vAlpha;
  varying vec2 vUvS;

  void main() {
    float m = texture2D(uSprite, vUvS).a;
    float a = vAlpha * m;
    if (a < 0.002) discard;
    gl_FragColor = vec4(vColor, min(a, 1.0));
  }
`;

// ---------------------------------------------------------------------------
// Splash - one source, two variants (#define CROWN builds the vertical burst).
// ---------------------------------------------------------------------------

const SPLASH_VERT = /* glsl */ `
  ${GLSL_LIGHT_UNIFORMS}
  uniform float uAlpha, uFadeEndSplash;

  attribute vec3 aPos;      // camera-relative impact point
  attribute vec3 aNrm;      // ground normal
  attribute vec4 aParams;   // x life 0..1, y radius, z seed, w strength

  varying vec2 vLocal;
  varying vec3 vColor;
  varying float vAlpha;
  varying float vT;
  varying float vSeed;

  void main() {
    float R = aParams.y;
    vec3 rel = aPos;
    float dist = length(rel);
    vec3 Vp = -rel / max(dist, 1e-4);
    vec3 n = aNrm;
    vec3 p;
    float grazing;

    #ifdef CROWN
      vec3 right = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
      vec3 up    = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);
      float s = R * 2.4;
      p = rel + right * (position.x * s) + up * ((position.y + 0.36) * s);
      vLocal = vec2(position.x * 2.0, (position.y + 0.5) * 2.0);
      grazing = 1.0;
    #else
      vec3 ref = abs(n.y) > 0.99 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
      vec3 tx = normalize(cross(ref, n));
      vec3 tz = cross(n, tx);
      float s = R * 2.2;
      // Lift a hair off the surface: the ground has its own microrelief and
      // z-fighting on a one-centimetre ripple is extremely visible.
      p = rel + tx * (position.x * s) + tz * (position.y * s) + n * 0.015;
      vLocal = position.xy * 2.2;
      // Wet ground is near-mirror at grazing angles, so ripples brighten there.
      grazing = mix(1.7, 0.85, clamp(dot(n, Vp), 0.0, 1.0));
    #endif

    float distFade = 1.0 - smoothstep(uFadeEndSplash * 0.55, uFadeEndSplash, dist);
    float mirror = pow(max(0.0, dot(reflect(-uKeyDir, n), Vp)), 20.0);
    vec3 col = uSkyLight * 0.85
             + uKeyLight * uSunBreak * (0.18 + mirror * 2.2)
             + uFlashColor * (uFlash * 1.4);
    col = mix(col, uFogColor, smoothstep(0.2, 1.0, dist / uFadeEndSplash) * 0.4);

    vColor = col;
    vAlpha = uAlpha * aParams.w * distFade * grazing;
    vT = aParams.x;
    vSeed = aParams.z;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }
`;

const SPLASH_FRAG = /* glsl */ `
  varying vec2 vLocal;
  varying vec3 vColor;
  varying float vAlpha;
  varying float vT;
  varying float vSeed;

  void main() {
    float t = vT;
    float a;

    #ifdef CROWN
      // Four ballistic beads plus the thin sheet they tear off. The crown only
      // exists for the first fifth of the impact, so it has to be cheap.
      float k = clamp(t / 0.22, 0.0, 1.0);
      if (k >= 1.0) discard;
      vec2 q = vLocal;
      a = 0.0;
      for (int i = 0; i < 4; i++) {
        float fi = float(i);
        float s = fract(vSeed * (3.13 + fi * 1.77) + fi * 0.37);
        vec2 b = vec2((s - 0.5) * 1.9 * k, (1.5 + s * 0.7) * k - 1.35 * k * k);
        vec2 d = q - b;
        a += exp(-dot(d, d) * 95.0);
      }
      float sheetY = max(q.y - 0.28 * k, 0.0) * 4.0;
      float sheet = exp(-q.x * q.x * 5.0) * exp(-sheetY * sheetY);
      a = (a + sheet * 0.75) * (1.0 - k) * vAlpha;
    #else
      float r = length(vLocal);
      // Capillary ring: the crest decelerates like sqrt(t), widening and
      // softening as it loses energy.
      float rr = sqrt(max(t, 0.0)) * 0.95;
      float w = 0.10 + 0.30 * t;
      float d0 = (r - rr) / w;
      float ring = exp(-d0 * d0);
      // Second, weaker crest launched when the crown collapses back in.
      float t2 = max(t - 0.24, 0.0) / 0.76;
      float d1 = (r - sqrt(t2) * 0.62) / (w * 0.85);
      float ring2 = exp(-d1 * d1) * 0.42 * step(0.24, t);
      float core = exp(-r * r * 26.0) * (1.0 - smoothstep(0.0, 0.13, t)) * 0.9;
      float decay = (1.0 - t) * (1.0 - t);
      a = vAlpha * ((ring + ring2) * decay + core);
    #endif

    if (a < 0.002) discard;
    gl_FragColor = vec4(vColor, min(a, 1.0));
  }
`;

// ---------------------------------------------------------------------------
// Running / falling droplets
// ---------------------------------------------------------------------------

const DROP_VERT = /* glsl */ `
  ${GLSL_LIGHT_UNIFORMS}
  uniform float uAlpha, uFadeEndDrop;

  attribute vec3 aPos;
  attribute vec3 aVel;
  attribute vec4 aParams;   // x size, y fade-in, z facing, w seed

  varying vec2 vQ;
  varying vec3 vColor;
  varying float vAlpha;

  ${GLSL_COMMON}

  void main() {
    vec3 rel = aPos;
    float dist = length(rel);
    vec3 Vp = -rel / max(dist, 1e-4);

    float vlen = length(aVel);
    vec3 axis = vlen > 1e-3 ? aVel / vlen : vec3(0.0, -1.0, 0.0);
    vec3 sideRaw = cross(axis, Vp);
    float sinA = length(sideRaw);
    vec3 side = sinA > 1e-3 ? sideRaw / sinA : vec3(1.0, 0.0, 0.0);

    float w = aParams.x;
    float energy = pixelWiden(w, dist, uPixelScale, uMinPixels * 1.6);
    // A bead at rest is round; a running bead stretches into a comet.
    float len = max(w, w * 0.9 + vlen * 0.10);

    float fade = (1.0 - smoothstep(uFadeEndDrop * 0.6, uFadeEndDrop, dist)) * smoothstep(0.1, 0.5, dist);
    vAlpha = uAlpha * aParams.y * aParams.z * energy * fade * mix(0.45, 1.0, sinA);
    if (vAlpha < 0.004) {
      vQ = vec2(0.0); vColor = vec3(0.0);
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      return;
    }

    // A bead is a lens: there is always a specular point somewhere on it, so the
    // key term is broad rather than a sharp lobe.
    float spec = 0.55 + 0.45 * max(0.0, dot(uKeyDir, Vp));
    vColor = uSkyLight * 1.15 + uKeyLight * uSunBreak * spec * 0.9 + uFlashColor * uFlash * 1.5;

    vec3 p = rel + axis * (position.y * len) + side * (position.x * w);
    vQ = position.xy * 2.0;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }
`;

const DROP_FRAG = /* glsl */ `
  varying vec2 vQ;
  varying vec3 vColor;
  varying float vAlpha;

  void main() {
    float u = vQ.x, v = vQ.y;
    // Teardrop: blunt at the leading (lower) end, tapering behind.
    float taper = 1.0 + 0.55 * smoothstep(-0.2, 1.0, v);
    float body = exp(-(u * u * taper * taper * 3.2 + v * v * 1.6));
    float gx = u + 0.3, gy = v - 0.28;
    float glint = exp(-(gx * gx + gy * gy) * 26.0);
    float a = vAlpha * (body + glint * 0.9);
    if (a < 0.002) discard;
    gl_FragColor = vec4(vColor * (1.0 + glint * 1.6), min(a, 1.0));
  }
`;

// ---------------------------------------------------------------------------
// Lightning bolt ribbon
// ---------------------------------------------------------------------------

const BOLT_VERT = /* glsl */ `
  uniform float uPixelScale, uMinPixels, uFlash, uWidth;
  uniform vec3 uFogColor;

  attribute vec3 aTan;
  attribute vec3 aOfs;      // x side (-1/+1), y width scale, z brightness

  varying float vU;
  varying float vBright;
  varying vec3 vTint;

  void main() {
    vec3 world = position;
    vec3 toCam = cameraPosition - world;
    float dist = length(toCam);
    vec3 Vp = toCam / max(dist, 1e-4);
    vec3 side = cross(aTan, Vp);
    float sl = length(side);
    side = sl > 1e-4 ? side / sl : vec3(1.0, 0.0, 0.0);

    float w = uWidth * aOfs.y;
    float widened = max(w, dist * uPixelScale * uMinPixels * 2.0);
    float energy = w / widened;

    // Aerial perspective: a strike three kilometres out is a soft blue smear, not
    // a white wire. This one term carries most of the sense of strike distance.
    float ext = exp(-dist * 0.00042);
    vTint = mix(uFogColor, vec3(0.86, 0.90, 1.05), ext);
    vBright = aOfs.z * uFlash * mix(0.30, 1.0, ext) * energy;
    vU = aOfs.x;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(world + side * (aOfs.x * widened), 1.0);
  }
`;

const BOLT_FRAG = /* glsl */ `
  varying float vU;
  varying float vBright;
  varying vec3 vTint;

  void main() {
    float u2 = vU * vU;
    float core = exp(-u2 * 26.0);
    float halo = exp(-u2 * 2.6) * 0.30;
    gl_FragColor = vec4(vTint * (core * 2.4 + halo) * vBright, 1.0);
  }
`;

// ---------------------------------------------------------------------------

export class Precipitation {
  constructor(ctx) {
    this.ctx = ctx;
    this.state = ctx.state;
    this.bus = ctx.bus;
    this.camera = ctx.camera;
    this.renderer = ctx.renderer;
    this.scene = ctx.scene;
    this.textures = ctx.textures;

    this.terrain = null;
    this.tree = null;

    this.rng = makeRNG(0x5a4b12);
    this.tier = TIERS[this.state.quality?.tier] || TIERS.high;

    /** Smoothed intensities - weather transitions must never snap. */
    this.rain = 0;
    this.snow = 0;
    /** Multi-pulse lightning envelope, also published to state.weather.lightning. */
    this.flash = 0;
    this.boltFlash = 0;
    /** Thunder loudness envelope, 0..1. Readable by the HUD. */
    this.thunder = 0;
    this.audioEnabled = true;

    this.group = new THREE.Group();          // follows the camera
    this.group.name = 'Precipitation';
    this.worldGroup = new THREE.Group();     // stays at the world origin
    this.worldGroup.name = 'PrecipitationWorld';
    this.worldGroup.matrixAutoUpdate = false;

    this._pixelScale = 0.002;
    this._time = 0;
    this._splashAcc = 0;
    this._dropAcc = 0;
    this._snowCursor = 0;
    this._snowSimulated = 0;
    this._camValid = false;
    this._camVel = new THREE.Vector3();
    this._lastLightningWrite = 0;

    // Lightning scheduling
    this._nextStrike = 6 + this.rng() * 10;
    this._strikeAge = 1e3;
    this._boltAge = 1e3;
    this._boltActive = false;
    this._burstLeft = 0;
    this._burstTimer = 0;
    this._pulseT = new Float32Array(6);
    this._pulseA = new Float32Array(6);
    this._pulseCount = 0;
    this._thunderTime = new Float32Array(8);
    this._thunderDist = new Float32Array(8);
    this._thunderAmp = new Float32Array(8);
    this._thunderCount = 0;

    this._audioCtx = null;
    this._audioGain = null;
    this._noiseBuf = null;
    this._unlockAudio = () => this._initAudio();
  }

  // -------------------------------------------------------------------------
  // Build
  // -------------------------------------------------------------------------

  async init() {
    this._sharedUniforms = {
      uTime: { value: 0 },
      uCamPos: { value: new THREE.Vector3() },
      uKeyDir: { value: new THREE.Vector3(0, 1, 0) },
      uKeyLight: { value: new THREE.Color(1, 1, 1) },
      uSkyLight: { value: new THREE.Color(0.5, 0.55, 0.62) },
      uFogColor: { value: new THREE.Color(0.7, 0.75, 0.82) },
      uFlashColor: { value: new THREE.Color(0.78, 0.85, 1.05) },
      uCanopy: { value: new THREE.Vector4(0, 0, 0, 0) },
      uPixelScale: { value: 0.002 },
      uMinPixels: { value: this.tier.minPx },
      uFlash: { value: 0 },
      uSunBreak: { value: 1 },
    };

    this._buildRain();
    this._buildSnow();
    this._buildSplashes();
    this._buildDroplets();
    this._buildBoltMesh();

    this.scene.add(this.group);
    this.scene.add(this.worldGroup);

    if (typeof window !== 'undefined') {
      window.addEventListener('pointerdown', this._unlockAudio, { once: true, passive: true });
      window.addEventListener('keydown', this._unlockAudio, { once: true, passive: true });
    }
  }

  /** A unit quad carrying only `position` in [-0.5,0.5] and `uv` in [0,1]. */
  _quad() {
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      -0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0,
    ]), 3));
    g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]), 2));
    g.setIndex(new THREE.BufferAttribute(new Uint16Array([0, 1, 2, 0, 2, 3]), 1));
    g.instanceCount = 0;
    return g;
  }

  /** Common setup for a camera-local instanced particle mesh. */
  _addParticleMesh(geometry, material, renderOrder) {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;      // the volume always contains the camera
    mesh.matrixAutoUpdate = false;   // identity; the parent group carries the offset
    mesh.renderOrder = renderOrder;
    mesh.visible = false;
    this.group.add(mesh);
    return mesh;
  }

  _buildRain() {
    const n = MAX.rain;
    const seed = new Float32Array(n * 3);
    const rand = new Float32Array(n * 4);
    const index = new Float32Array(n);
    const r = makeRNG(0x1f3a77);
    for (let i = 0; i < n; i++) {
      seed[i * 3] = r();
      seed[i * 3 + 1] = r();
      seed[i * 3 + 2] = r();
      // Drop-size distribution is heavily skewed toward small drops (Marshall-Palmer
      // is exponential); cubing a uniform is a cheap, close-enough stand-in.
      const s = r();
      rand[i * 4] = 0.78 + s * 0.5;          // fall-speed multiplier
      rand[i * 4 + 1] = s * s * s;           // size / length multiplier
      rand[i * 4 + 2] = 0.55 + r() * 0.65;   // per-drop opacity
      rand[i * 4 + 3] = r();                 // phase / drag lag
      index[i] = i / n;
    }
    const g = this._quad();
    g.setAttribute('aSeed', new THREE.InstancedBufferAttribute(seed, 3));
    g.setAttribute('aRand', new THREE.InstancedBufferAttribute(rand, 4));
    g.setAttribute('aIndex', new THREE.InstancedBufferAttribute(index, 1));

    this.rainUniforms = {
      ...this._sharedUniforms,
      uBox: { value: new THREE.Vector3(...this.tier.rainBox) },
      uCamVel: { value: new THREE.Vector3() },
      uWind: { value: new THREE.Vector2(0, 0) },
      uFall: { value: 8.5 },
      uStretch: { value: 0.05 },
      uWidth: { value: 0.011 },
      uAlpha: { value: 0.3 },
      uFadeEnd: { value: this.tier.rainFade },
      uCount: { value: 0 },
    };
    this.rainMat = new THREE.ShaderMaterial({
      uniforms: this.rainUniforms,
      vertexShader: RAIN_VERT,
      fragmentShader: RAIN_FRAG,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      fog: false,
    });
    this.rainGeo = g;
    this.rainMesh = this._addParticleMesh(g, this.rainMat, 12);
  }

  _buildSnow() {
    const n = MAX.snow;
    this._snowRel = new Float32Array(n * 3);
    this._snowVel = new Float32Array(n * 3);
    this._snowAdv = new Float32Array(n * 3);
    this._snowAmp = new Float32Array(n);
    this._snowBucket = new Uint8Array(n);
    this._snowOsc = new Float32Array(n * 2);
    this._snowTerminal = new Float32Array(n);
    const rand = new Float32Array(n * 4);
    const index = new Float32Array(n);
    const r = makeRNG(0x77c311);
    this._snowRng = r;
    for (let i = 0; i < n; i++) {
      rand[i * 4] = r();                     // size
      rand[i * 4 + 1] = r();                 // tumble rate
      rand[i * 4 + 2] = 0.6 + r() * 0.5;     // opacity
      rand[i * 4 + 3] = r();                 // phase
      index[i] = i / n;
      // Flutter: the sideways spiral a plate-shaped flake carves as it stalls and
      // slips. Each flake carries a unit 2-vector that is *rotated* a fixed angle
      // per frame instead of being evaluated with sin/cos - see _updateSnow.
      // Frequencies are quantised into buckets so the per-frame rotation for a
      // whole bucket is computed once, not once per flake.
      this._snowAmp[i] = 0.18 + r() * 0.5;
      this._snowBucket[i] = Math.min(SNOW_BUCKETS - 1, Math.floor(r() * SNOW_BUCKETS));
      const ph = r() * TAU;
      this._snowOsc[i * 2] = Math.cos(ph);
      this._snowOsc[i * 2 + 1] = Math.sin(ph);
      this._snowTerminal[i] = 0.45 + r() * 0.75;
    }

    const g = this._quad();
    this._snowAttr = new THREE.InstancedBufferAttribute(this._snowRel, 3);
    this._snowAttr.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('aOffset', this._snowAttr);
    g.setAttribute('aRand', new THREE.InstancedBufferAttribute(rand, 4));
    g.setAttribute('aIndex', new THREE.InstancedBufferAttribute(index, 1));

    this.snowUniforms = {
      ...this._sharedUniforms,
      uBox: { value: new THREE.Vector3(...this.tier.snowBox) },
      uSize: { value: 0.055 },
      uAlpha: { value: 0.85 },
      uFadeEnd: { value: this.tier.snowFade },
      uCount: { value: 0 },
      uSprite: { value: this._snowSprite() },
    };
    this.snowMat = new THREE.ShaderMaterial({
      uniforms: this.snowUniforms,
      vertexShader: SNOW_VERT,
      fragmentShader: SNOW_FRAG,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      fog: false,
    });
    this.snowGeo = g;
    this.snowMesh = this._addParticleMesh(g, this.snowMat, 12);
  }

  /**
   * Soft flake sprite: a wide gaussian core with a faint six-fold structure.
   * Mipmapped, so a flake covering two pixels resolves to a smooth blob instead
   * of a flickering speck.
   */
  _snowSprite() {
    const make = () => {
      const draw = (c, size) => {
        const h = size * 0.5;
        c.clearRect(0, 0, size, size);
        const g = c.createRadialGradient(h, h, 0, h, h, h * 0.92);
        g.addColorStop(0.0, 'rgba(255,255,255,1)');
        g.addColorStop(0.35, 'rgba(255,255,255,0.72)');
        g.addColorStop(0.7, 'rgba(255,255,255,0.18)');
        g.addColorStop(1.0, 'rgba(255,255,255,0)');
        c.fillStyle = g;
        c.fillRect(0, 0, size, size);
        // Six arms, very faint: enough structure to survive the first mip and
        // give large near-camera flakes a crystalline hint, never a snowflake icon.
        c.globalCompositeOperation = 'lighter';
        c.strokeStyle = 'rgba(255,255,255,0.28)';
        c.lineWidth = Math.max(1, size * 0.035);
        c.lineCap = 'round';
        for (let i = 0; i < 6; i++) {
          const a = (i / 6) * Math.PI * 2;
          c.beginPath();
          c.moveTo(h, h);
          c.lineTo(h + Math.cos(a) * h * 0.66, h + Math.sin(a) * h * 0.66);
          c.stroke();
        }
        c.globalCompositeOperation = 'source-over';
      };
      let tex;
      if (this.textures?.canvas) {
        tex = this.textures.canvas(64, draw);
      } else {
        const cv = document.createElement('canvas');
        cv.width = cv.height = 64;
        draw(cv.getContext('2d'), 64);
        tex = new THREE.CanvasTexture(cv);
        this._ownSprite = tex;
      }
      tex.generateMipmaps = true;
      tex.minFilter = THREE.LinearMipmapLinearFilter;
      tex.magFilter = THREE.LinearFilter;
      tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
      tex.needsUpdate = true;
      return tex;
    };
    return this.textures?.get ? this.textures.get('precip/snowflake', make) : make();
  }

  _buildSplashes() {
    const n = MAX.splash;
    this._spPos = new Float32Array(n * 3);
    this._spNrm = new Float32Array(n * 3);
    this._spParams = new Float32Array(n * 4);
    this._spAge = new Float32Array(n);
    this._spLife = new Float32Array(n);
    this._spCount = 0;

    const posAttr = new THREE.InstancedBufferAttribute(this._spPos, 3);
    const nrmAttr = new THREE.InstancedBufferAttribute(this._spNrm, 3);
    const parAttr = new THREE.InstancedBufferAttribute(this._spParams, 4);
    posAttr.setUsage(THREE.DynamicDrawUsage);
    nrmAttr.setUsage(THREE.DynamicDrawUsage);
    parAttr.setUsage(THREE.DynamicDrawUsage);
    this._spAttrs = [posAttr, nrmAttr, parAttr];

    this.splashUniforms = {
      ...this._sharedUniforms,
      uAlpha: { value: 0.55 },
      uFadeEndSplash: { value: this.tier.splashR * 1.25 },
    };
    const mk = (crown) => {
      const g = this._quad();
      g.setAttribute('aPos', posAttr);
      g.setAttribute('aNrm', nrmAttr);
      g.setAttribute('aParams', parAttr);
      const m = new THREE.ShaderMaterial({
        uniforms: this.splashUniforms,
        defines: crown ? { CROWN: '1' } : {},
        vertexShader: SPLASH_VERT,
        fragmentShader: SPLASH_FRAG,
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
        fog: false,
      });
      return { g, m, mesh: this._addParticleMesh(g, m, crown ? 11 : 10) };
    };
    this._ring = mk(false);
    this._crown = mk(true);
  }

  _buildDroplets() {
    const n = MAX.drops;
    this._drPos = new Float32Array(n * 3);
    this._drVel = new Float32Array(n * 3);
    this._drParams = new Float32Array(n * 4);
    this._drKind = new Uint8Array(n);       // 0 = running on the trunk, 1 = free fall
    this._drTheta = new Float32Array(n);
    this._drStick = new Float32Array(n);
    this._drAge = new Float32Array(n);
    this._drCount = 0;

    const posAttr = new THREE.InstancedBufferAttribute(this._drPos, 3);
    const velAttr = new THREE.InstancedBufferAttribute(this._drVel, 3);
    const parAttr = new THREE.InstancedBufferAttribute(this._drParams, 4);
    posAttr.setUsage(THREE.DynamicDrawUsage);
    velAttr.setUsage(THREE.DynamicDrawUsage);
    parAttr.setUsage(THREE.DynamicDrawUsage);
    this._drAttrs = [posAttr, velAttr, parAttr];

    const g = this._quad();
    g.setAttribute('aPos', posAttr);
    g.setAttribute('aVel', velAttr);
    g.setAttribute('aParams', parAttr);
    this.dropUniforms = {
      ...this._sharedUniforms,
      uAlpha: { value: 0.9 },
      uFadeEndDrop: { value: 16 },
    };
    this.dropMat = new THREE.ShaderMaterial({
      uniforms: this.dropUniforms,
      vertexShader: DROP_VERT,
      fragmentShader: DROP_FRAG,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      fog: false,
    });
    this.dropGeo = g;
    this.dropMesh = this._addParticleMesh(g, this.dropMat, 11);
  }

  _buildBoltMesh() {
    const maxVerts = MAX_BOLT_SEGS * 4;
    this._boltPos = new Float32Array(maxVerts * 3);
    this._boltTan = new Float32Array(maxVerts * 3);
    this._boltOfs = new Float32Array(maxVerts * 3);
    const idx = new Uint16Array(MAX_BOLT_SEGS * 6);
    for (let s = 0; s < MAX_BOLT_SEGS; s++) {
      const v = s * 4, o = s * 6;
      idx[o] = v; idx[o + 1] = v + 1; idx[o + 2] = v + 2;
      idx[o + 3] = v; idx[o + 4] = v + 2; idx[o + 5] = v + 3;
    }
    const g = new THREE.BufferGeometry();
    const pa = new THREE.BufferAttribute(this._boltPos, 3);
    const ta = new THREE.BufferAttribute(this._boltTan, 3);
    const oa = new THREE.BufferAttribute(this._boltOfs, 3);
    pa.setUsage(THREE.DynamicDrawUsage);
    ta.setUsage(THREE.DynamicDrawUsage);
    oa.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('position', pa);
    g.setAttribute('aTan', ta);
    g.setAttribute('aOfs', oa);
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.setDrawRange(0, 0);
    this._boltAttrs = [pa, ta, oa];

    this.boltUniforms = {
      uPixelScale: this._sharedUniforms.uPixelScale,
      uMinPixels: this._sharedUniforms.uMinPixels,
      uFogColor: this._sharedUniforms.uFogColor,
      uFlash: { value: 0 },
      uWidth: { value: 2.2 },
    };
    this.boltMat = new THREE.ShaderMaterial({
      uniforms: this.boltUniforms,
      vertexShader: BOLT_VERT,
      fragmentShader: BOLT_FRAG,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      fog: false,
    });
    this.boltGeo = g;
    this.boltMesh = new THREE.Mesh(g, this.boltMat);
    this.boltMesh.frustumCulled = false;
    this.boltMesh.matrixAutoUpdate = false;
    this.boltMesh.renderOrder = 13;
    this.boltMesh.visible = false;
    this.worldGroup.add(this.boltMesh);

    // Scratch for midpoint displacement - reused, never reallocated.
    this._ptA = new Float32Array(MAX_BOLT_POINTS * 3);
    this._ptB = new Float32Array(MAX_BOLT_POINTS * 3);
    this._chStart = new Int32Array(MAX_CHANNELS);
    this._chCount = new Int32Array(MAX_CHANNELS);
    this._chWidth = new Float32Array(MAX_CHANNELS);
    this._chBright = new Float32Array(MAX_CHANNELS);
    this._chPts = new Float32Array(MAX_BOLT_POINTS * 3);
  }

  link(systems) {
    this.terrain = systems.terrain || null;
    this.tree = systems.tree || null;
  }

  // -------------------------------------------------------------------------
  // Frame
  // -------------------------------------------------------------------------

  update(dt, state) {
    if (!this.rainMesh) return;
    this._time = state.time.elapsed;

    this.camera.getWorldPosition(_camPos);
    if (this._camValid) _camDelta.subVectors(_prevCam, _camPos);
    else { _camDelta.set(0, 0, 0); this._camValid = true; }
    _prevCam.copy(_camPos);
    // Everything camera-local lives in a group pinned to the camera, so the
    // shaders work in small relative coordinates and precision never degrades.
    this.group.position.copy(_camPos);

    // Smoothed camera velocity, for relative-motion streaking. Clamped so a
    // teleport or a frame hitch cannot produce a hundred-metre smear.
    if (dt > 1e-5) {
      _v3.copy(_camDelta).multiplyScalar(-1 / dt);
      const speed = _v3.length();
      if (speed > 30) _v3.multiplyScalar(30 / speed);
      this._camVel.lerp(_v3, 1 - Math.exp(-9 * dt));
    }

    const targetRain = clamp01(state.weather.rainIntensity);
    const targetSnow = clamp01(state.weather.snowIntensity);
    this.rain = damp(this.rain, targetRain, 0.9, dt);
    this.snow = damp(this.snow, targetSnow, 0.7, dt);
    if (Math.abs(this.rain - targetRain) < 1e-4) this.rain = targetRain;
    if (Math.abs(this.snow - targetSnow) < 1e-4) this.snow = targetSnow;

    // Lightning first: this frame's flash feeds every material's lighting term.
    this._updateLightning(dt, state);
    this._updateEnvironment(dt, state);
    this._updateRain(dt, state);
    this._updateSnow(dt, state);
    this._updateDroplets(dt, state);
    this._updateSplashes(dt, state);
  }

  /** One pass over everything the five materials share. */
  _updateEnvironment(dt, state) {
    const u = this._sharedUniforms;
    u.uTime.value = this._time;
    u.uCamPos.value.copy(_camPos);

    // Angular size of one pixel - drives every minimum-width computation. Read
    // from the real drawing buffer so it survives adaptive resolution changes.
    this.renderer.getDrawingBufferSize(_size);
    const h = Math.max(1, _size.y);
    const fov = (this.camera.fov || 60) * Math.PI / 180;
    this._pixelScale = (2 * Math.tan(fov * 0.5)) / h;
    u.uPixelScale.value = this._pixelScale;

    // Key light: whichever of sun/moon is actually doing the lighting.
    const sun = state.sun, moon = state.moon;
    const sunW = Math.max(0, sun.intensity) * sun.visibility;
    const moonW = Math.max(0, moon.intensity) * moon.visibility;
    const total = sunW + moonW;
    if (total > 1e-5) {
      const f = sunW / total;
      _v3.copy(sun.direction).multiplyScalar(f).addScaledVector(moon.direction, 1 - f);
      if (_v3.lengthSq() < 1e-6) _v3.copy(f > 0.5 ? sun.direction : moon.direction);
      u.uKeyDir.value.copy(_v3.normalize());
      _col.copy(sun.color).multiplyScalar(Math.min(sun.intensity, 6) * sun.visibility);
      _col2.copy(moon.color).multiplyScalar(Math.min(moon.intensity, 6) * moon.visibility);
      u.uKeyLight.value.copy(_col).add(_col2).multiplyScalar(0.26);
    } else {
      u.uKeyDir.value.set(0, 1, 0);
      u.uKeyLight.value.setRGB(0, 0, 0);
    }

    // Sky term. Rain has to stay legible against a black storm cloud, so the
    // colour leans on the horizon (which stays bright under overcast) rather than
    // the zenith, and keeps a small floor scaled by ambient so it never goes flat.
    const cov = clamp01(state.clouds.coverage);
    const amb = Math.max(0.06, state.sky.ambientIntensity);
    _col.copy(state.sky.horizonColor).lerp(state.sky.zenithColor, 0.28);
    _col.multiplyScalar(amb * (0.75 + 0.25 * (1 - cov)) * 0.55);
    const floor = 0.045 * amb;
    u.uSkyLight.value.setRGB(
      Math.max(_col.r, floor),
      Math.max(_col.g, floor),
      Math.max(_col.b, floor * 1.06)   // a hair cooler in shadow, like real skylight
    );
    u.uFogColor.value.copy(state.weather.fogColor);

    // Sun through a break in the deck: this is what makes rain glitter.
    u.uSunBreak.value = clamp01(Math.max(sun.visibility, moon.visibility * 0.6)) * (1 - 0.72 * cov);
    u.uFlash.value = this.flash;
    this.boltUniforms.uFlash.value = this.boltFlash;

    // Canopy shelter volume.
    const c = this.tree?.canopyCenter;
    const r = this.tree?.canopyRadius;
    const cv = u.uCanopy.value;
    if (c && typeof r === 'number' && r > 0.5 && Number.isFinite(c.x)) cv.set(c.x, c.y, c.z, r);
    else cv.set(0, 0, 0, 0);
  }

  _updateRain(dt, state) {
    const f = this.rain;
    const vis = f > 0.004;
    this.rainMesh.visible = vis;
    if (!vis) { this.rainGeo.instanceCount = 0; return; }

    const u = this.rainUniforms;
    const w = state.wind;
    // Rain is dragged by the mean wind, not the turbulent field; drops never
    // fully reach wind speed, hence the scale-down.
    const speed = w.strength * w.gust * 0.62;
    u.uWind.value.set(w.direction.x * speed, w.direction.y * speed);
    u.uCamVel.value.copy(this._camVel);
    u.uFall.value = 7.4 + 4.2 * f;
    u.uStretch.value = 0.042 + 0.028 * f;      // effective shutter time
    u.uWidth.value = 0.0095 + 0.006 * f;
    u.uAlpha.value = 0.16 + 0.30 * f;
    u.uFadeEnd.value = this.tier.rainFade;

    // Count ramps with the square root of intensity: perceived density is far
    // from linear in drop count, and this keeps drizzle from reading as a wall.
    const frac = clamp01(Math.sqrt(f) * 1.02);
    u.uCount.value = frac;
    this.rainGeo.instanceCount = Math.max(1, Math.ceil(this.tier.rain * frac));
  }

  _updateSnow(dt, state) {
    const f = this.snow;
    const vis = f > 0.004;
    this.snowMesh.visible = vis;
    if (!vis) {
      this.snowGeo.instanceCount = 0;
      // Drop the simulated set so nothing reappears at a stale camera position.
      this._snowSimulated = 0;
      return;
    }

    const box = this.snowUniforms.uBox.value;
    const bx = box.x, by = box.y, bz = box.z;
    const hx = bx * 0.5, hy = by * 0.5, hz = bz * 0.5;

    const frac = clamp01(Math.sqrt(f) * 1.05);
    const active = Math.max(1, Math.ceil(this.tier.snow * frac));
    this.snowUniforms.uCount.value = frac;
    this.snowUniforms.uFadeEnd.value = this.tier.snowFade;
    this.snowUniforms.uSize.value = 0.045 + 0.045 * f;
    this.snowUniforms.uAlpha.value = 0.5 + 0.45 * f;

    // Flakes that were dormant (the count grew, or this is the first frame) are
    // placed fresh, high in the volume, so they fall in rather than blink in.
    if (active > this._snowSimulated) {
      for (let i = this._snowSimulated; i < active; i++) this._placeSnow(i, bx, by, bz, true);
    }
    this._snowSimulated = active;

    const pos = this._snowRel, vel = this._snowVel, adv = this._snowAdv;
    const amp = this._snowAmp, term = this._snowTerminal;
    const osc = this._snowOsc, bucket = this._snowBucket;
    const t = this._time;
    const sample = state.wind.sample;

    // Refresh the cached wind vector for a slice of flakes each frame. The field
    // that actually carries snow is the large-scale one; it varies over metres and
    // seconds, so a 6 Hz refresh is indistinguishable from per-frame sampling and
    // costs a tenth as much. Whatever wind.sample does internally, we call it
    // active/10 times a frame, not `active` times.
    const slice = Math.ceil(active / 10);
    let cur = this._snowCursor;
    for (let s = 0; s < slice; s++) {
      if (cur >= active) cur = 0;
      const i3 = cur * 3;
      sample(_camPos.x + pos[i3], _camPos.z + pos[i3 + 2], t, _windOut);
      adv[i3] = _windOut.x;
      adv[i3 + 1] = _windOut.y;
      adv[i3 + 2] = _windOut.z;
      cur++;
    }
    this._snowCursor = cur;

    const dx = _camDelta.x, dy = _camDelta.y, dz = _camDelta.z;
    // Inertia: wind is a target velocity, not an impulse. Flakes lag behind
    // gusts, which is precisely what separates snow from confetti.
    const k = 1 - Math.exp(-2.2 * dt);
    // One rotation per frequency bucket for the whole field.
    for (let b = 0; b < SNOW_BUCKETS; b++) {
      const ang = SNOW_FREQ[b] * dt;
      _rotC[b] = Math.cos(ang);
      // Alternate buckets spiral the other way. Flakes all turning the same
      // direction is a uniformity you notice without being able to name it.
      _rotS[b] = (b & 1) ? -Math.sin(ang) : Math.sin(ang);
    }

    for (let i = 0; i < active; i++) {
      const i3 = i * 3;
      const b = bucket[i];
      const cd = _rotC[b], sd = _rotS[b];
      const o2 = i * 2;
      const ox = osc[o2], oy = osc[o2 + 1];
      let cx = ox * cd - oy * sd;
      let cy = ox * sd + oy * cd;
      // First-order renormalisation: without it the repeated rotation slowly
      // drifts off the unit circle in float32 and the spiral inflates.
      const inv = 1.5 - 0.5 * (cx * cx + cy * cy);
      cx *= inv; cy *= inv;
      osc[o2] = cx; osc[o2 + 1] = cy;

      // Horizontal pair traces the helix; the vertical bob is its second
      // harmonic, which sin(2a) = 2 sin(a) cos(a) hands us for two multiplies.
      const fa = amp[i];
      const fx = cx * fa;
      const fz = cy * fa;
      const fy = 2 * cx * cy * fa * 0.35;

      vel[i3] += ((adv[i3] * 0.92 + fx) - vel[i3]) * k;
      vel[i3 + 1] += ((adv[i3 + 1] * 0.5 - term[i] + fy) - vel[i3 + 1]) * k;
      vel[i3 + 2] += ((adv[i3 + 2] * 0.92 + fz) - vel[i3 + 2]) * k;

      let px = pos[i3] + vel[i3] * dt + dx;
      let py = pos[i3 + 1] + vel[i3 + 1] * dt + dy;
      let pz = pos[i3 + 2] + vel[i3 + 2] * dt + dz;

      // Wrap in the camera-local frame: constant density forever. A flake moves
      // far less than a box per frame, so the branch is almost never taken and
      // the general modulo only runs after a teleport.
      if (px > hx || px < -hx) px -= bx * Math.floor((px + hx) / bx);
      if (py > hy || py < -hy) py -= by * Math.floor((py + hy) / by);
      if (pz > hz || pz < -hz) pz -= bz * Math.floor((pz + hz) / bz);

      pos[i3] = px;
      pos[i3 + 1] = py;
      pos[i3 + 2] = pz;
    }

    this._snowAttr.needsUpdate = true;
    this.snowGeo.instanceCount = active;
  }

  _placeSnow(i, bx, by, bz, biasTop) {
    const r = this._snowRng;
    const i3 = i * 3;
    this._snowRel[i3] = (r() - 0.5) * bx;
    this._snowRel[i3 + 1] = biasTop ? (0.18 + r() * 0.32) * by : (r() - 0.5) * by;
    this._snowRel[i3 + 2] = (r() - 0.5) * bz;
    this._snowVel[i3] = 0;
    this._snowVel[i3 + 1] = -this._snowTerminal[i];
    this._snowVel[i3 + 2] = 0;
    this._snowAdv[i3] = 0;
    this._snowAdv[i3 + 1] = 0;
    this._snowAdv[i3 + 2] = 0;
  }

  // -------------------------------------------------------------------------
  // Impacts
  // -------------------------------------------------------------------------

  _updateSplashes(dt, state) {
    const pos = this._spPos, par = this._spParams;
    const age = this._spAge, life = this._spLife;
    const dx = _camDelta.x, dy = _camDelta.y, dz = _camDelta.z;

    // Age, slide and compact. Splashes are world-anchored, so in the camera-local
    // frame they move by exactly the camera delta.
    let count = this._spCount;
    for (let i = 0; i < count; i++) {
      age[i] += dt;
      if (age[i] >= life[i]) {
        count--;
        this._swapSplash(i, count);
        i--;
        continue;
      }
      const i3 = i * 3;
      pos[i3] += dx; pos[i3 + 1] += dy; pos[i3 + 2] += dz;
      par[i * 4] = age[i] / life[i];
    }

    const f = this.rain;
    if (f > 0.02) {
      const R = this.tier.splashR;
      const maxSp = this.tier.splash;
      // Impacts per second inside the visible disc, tuned so light rain stipples
      // the ground and a downpour makes it boil. Quadratic because both drop
      // count and drop rate climb with intensity.
      this._splashAcc += (26 + 620 * f * f) * dt;
      if (this._splashAcc > maxSp) this._splashAcc = maxSp;
      // Bias the disc down the view direction: splashes behind you are wasted.
      this.camera.getWorldDirection(_fwd);
      _fwd.y = 0;
      if (_fwd.lengthSq() < 1e-6) _fwd.set(0, 0, -1);
      _fwd.normalize();
      const cx = _camPos.x + _fwd.x * R * 0.3;
      const cz = _camPos.z + _fwd.z * R * 0.3;
      const canopy = this._sharedUniforms.uCanopy.value;

      while (this._splashAcc >= 1 && count < maxSp) {
        this._splashAcc -= 1;
        const rr = R * Math.sqrt(this.rng());
        const th = this.rng() * TAU;
        const wx = cx + Math.cos(th) * rr;
        const wz = cz + Math.sin(th) * rr;
        // The canopy shelters the ground under it - no splashes there, which is
        // exactly what makes the drip line at its edge read.
        if (canopy.w > 0.5) {
          const d = Math.hypot(wx - canopy.x, wz - canopy.z);
          const shelter = 1 - smoothstep(canopy.w * 0.5, canopy.w * 1.06, d);
          if (this.rng() < shelter * 0.92) continue;
        }
        this._spawnSplash(wx, wz, 0.55 + 0.75 * f, count++);
      }
    }
    this._spCount = count;

    const vis = count > 0;
    const crown = vis && this.tier.crown;
    this._ring.mesh.visible = vis;
    this._crown.mesh.visible = crown;
    this._ring.g.instanceCount = count;
    this._crown.g.instanceCount = crown ? count : 0;
    this.splashUniforms.uFadeEndSplash.value = this.tier.splashR * 1.25;
    this.splashUniforms.uAlpha.value = 0.30 + 0.35 * clamp01(state.weather.wetness + this.rain * 0.5);
    if (vis) for (let i = 0; i < this._spAttrs.length; i++) this._spAttrs[i].needsUpdate = true;
  }

  /** Spawn one impact at world x/z, placed on the ground and oriented to it. */
  _spawnSplash(wx, wz, strength, slot) {
    const i3 = slot * 3, i4 = slot * 4;
    const gy = this.terrain?.getHeight ? this.terrain.getHeight(wx, wz) : 0;
    this._spPos[i3] = wx - _camPos.x;
    this._spPos[i3 + 1] = gy - _camPos.y;
    this._spPos[i3 + 2] = wz - _camPos.z;
    if (this.terrain?.getNormal) {
      this.terrain.getNormal(wx, wz, _n3);
      const l = _n3.length();
      if (l > 1e-4) _n3.multiplyScalar(1 / l); else _n3.set(0, 1, 0);
    } else _n3.set(0, 1, 0);
    this._spNrm[i3] = _n3.x; this._spNrm[i3 + 1] = _n3.y; this._spNrm[i3 + 2] = _n3.z;
    const r = this.rng();
    this._spParams[i4] = 0;
    this._spParams[i4 + 1] = 0.11 + r * 0.13;         // maximum ring radius, metres
    this._spParams[i4 + 2] = this.rng();              // seed for the crown beads
    this._spParams[i4 + 3] = strength * (0.6 + 0.6 * this.rng());
    this._spAge[slot] = 0;
    this._spLife[slot] = 0.42 + r * 0.34;
  }

  _swapSplash(a, b) {
    if (a === b) return;
    const a3 = a * 3, b3 = b * 3, a4 = a * 4, b4 = b * 4;
    for (let k = 0; k < 3; k++) {
      let t = this._spPos[a3 + k]; this._spPos[a3 + k] = this._spPos[b3 + k]; this._spPos[b3 + k] = t;
      t = this._spNrm[a3 + k]; this._spNrm[a3 + k] = this._spNrm[b3 + k]; this._spNrm[b3 + k] = t;
    }
    for (let k = 0; k < 4; k++) {
      const t = this._spParams[a4 + k]; this._spParams[a4 + k] = this._spParams[b4 + k]; this._spParams[b4 + k] = t;
    }
    let t = this._spAge[a]; this._spAge[a] = this._spAge[b]; this._spAge[b] = t;
    t = this._spLife[a]; this._spLife[a] = this._spLife[b]; this._spLife[b] = t;
  }

  // -------------------------------------------------------------------------
  // Droplets: beads running down the trunk, and the canopy drip line
  // -------------------------------------------------------------------------

  _updateDroplets(dt, state) {
    const wet = clamp01(Math.max(state.weather.wetness, this.rain));
    const maxDrops = this.tier.drops;
    const trunk = this.tree?.position;
    const canopy = this._sharedUniforms.uCanopy.value;
    const hasTree = !!trunk && Number.isFinite(trunk.x);
    // Only worth simulating while the player can see the surface it runs on.
    const nearTree = hasTree ? _camPos.distanceTo(trunk) < 26 : false;
    const trunkR = typeof this.tree?.trunkRadius === 'number' ? this.tree.trunkRadius : 0.42;

    const pos = this._drPos, vel = this._drVel, par = this._drParams;
    let count = this._drCount;

    for (let i = 0; i < count; i++) {
      const i3 = i * 3, i4 = i * 4;
      this._drAge[i] += dt;
      let wx = pos[i3] + _camPos.x;
      let wy = pos[i3 + 1] + _camPos.y;
      let wz = pos[i3 + 2] + _camPos.z;

      if (this._drKind[i] === 0 && hasTree) {
        // Running bead. Water on bark is stick-slip: surface tension pins it,
        // then it releases and accelerates. That stutter is the whole effect.
        this._drStick[i] -= dt;
        if (this._drStick[i] <= 0) {
          vel[i3 + 1] -= 1.15 * dt;
          if (vel[i3 + 1] < -0.85) vel[i3 + 1] = -0.85;
          if (this.rng() < dt * 0.9) {
            this._drStick[i] = 0.08 + this.rng() * 0.34;
            vel[i3 + 1] *= 0.12;
          }
        } else {
          vel[i3 + 1] *= Math.exp(-9 * dt);
        }
        // Wander around the trunk as the bead follows the bark grooves.
        this._drTheta[i] += Math.sin(this._drAge[i] * 2.3 + par[i4 + 3] * 9.1) * 0.22 * dt;
        const h = wy - trunk.y;
        const rad = trunkR * Math.max(0.35, 1 - 0.055 * Math.max(h, 0));
        wy += vel[i3 + 1] * dt;
        wx = trunk.x + Math.cos(this._drTheta[i]) * rad;
        wz = trunk.z + Math.sin(this._drTheta[i]) * rad;
        vel[i3] = 0; vel[i3 + 2] = 0;
        // Only beads on the camera side of the trunk are actually visible.
        const ox = wx - trunk.x, oz = wz - trunk.z;
        const tx = _camPos.x - trunk.x, tz = _camPos.z - trunk.z;
        const facing = (ox * tx + oz * tz) / (Math.hypot(ox, oz) * Math.hypot(tx, tz) + 1e-5);
        par[i4 + 2] = clamp01(facing * 1.4 + 0.15);
        if (wy <= trunk.y + 0.03) {
          if (this._spCount < this.tier.splash) this._spawnSplash(wx, wz, 0.7, this._spCount++);
          count--; this._swapDrop(i, count); i--; continue;
        }
      } else {
        // Free-falling drip off the canopy: wind pushes it, gravity wins.
        state.wind.sample(wx, wz, this._time, _windOut);
        const kd = 1 - Math.exp(-2.5 * dt);
        vel[i3] += (_windOut.x * 0.5 - vel[i3]) * kd;
        vel[i3 + 2] += (_windOut.z * 0.5 - vel[i3 + 2]) * kd;
        vel[i3 + 1] -= 9.0 * dt;
        if (vel[i3 + 1] < -7.5) vel[i3 + 1] = -7.5;
        wx += vel[i3] * dt; wy += vel[i3 + 1] * dt; wz += vel[i3 + 2] * dt;
        par[i4 + 2] = 1;
        const gy = this.terrain?.getHeight ? this.terrain.getHeight(wx, wz) : 0;
        if (wy <= gy + 0.02 || this._drAge[i] > 6) {
          if (this._spCount < this.tier.splash) this._spawnSplash(wx, wz, 0.9, this._spCount++);
          count--; this._swapDrop(i, count); i--; continue;
        }
      }

      pos[i3] = wx - _camPos.x;
      pos[i3 + 1] = wy - _camPos.y;
      pos[i3 + 2] = wz - _camPos.z;
      // Fade in over the first moments so a bead never blinks into existence.
      par[i4 + 1] = Math.min(1, this._drAge[i] * 6);
    }

    // Spawn. Beads only exist while the surface is genuinely wet.
    if (hasTree && nearTree && wet > 0.05 && count < maxDrops) {
      this._dropAcc += dt * (2.5 + 14 * wet);
      if (this._dropAcc > maxDrops) this._dropAcc = maxDrops;
      while (this._dropAcc >= 1 && count < maxDrops) {
        this._dropAcc -= 1;
        const runner = this.rng() < 0.55 || canopy.w <= 0.5;
        const i3 = count * 3, i4 = count * 4;
        if (runner) {
          const th = this.rng() * TAU;
          const h = 0.6 + this.rng() * 4.2;
          const rad = trunkR * Math.max(0.35, 1 - 0.055 * h);
          this._drKind[count] = 0;
          this._drTheta[count] = th;
          this._drStick[count] = this.rng() * 0.25;
          pos[i3] = trunk.x + Math.cos(th) * rad - _camPos.x;
          pos[i3 + 1] = trunk.y + h - _camPos.y;
          pos[i3 + 2] = trunk.z + Math.sin(th) * rad - _camPos.z;
          vel[i3] = 0; vel[i3 + 1] = -0.05; vel[i3 + 2] = 0;
          par[i4] = 0.016 + this.rng() * 0.014;
        } else {
          const th = this.rng() * TAU;
          const rad = canopy.w * (0.45 + 0.55 * Math.sqrt(this.rng()));
          this._drKind[count] = 1;
          pos[i3] = canopy.x + Math.cos(th) * rad - _camPos.x;
          pos[i3 + 1] = canopy.y - canopy.w * 0.32 - _camPos.y;
          pos[i3 + 2] = canopy.z + Math.sin(th) * rad - _camPos.z;
          vel[i3] = 0; vel[i3 + 1] = -0.4; vel[i3 + 2] = 0;
          par[i4] = 0.014 + this.rng() * 0.012;
        }
        par[i4 + 1] = 0;
        par[i4 + 2] = 1;
        par[i4 + 3] = this.rng();
        this._drAge[count] = 0;
        count++;
      }
    }

    this._drCount = count;
    this.dropMesh.visible = count > 0;
    this.dropGeo.instanceCount = count;
    this.dropUniforms.uAlpha.value = 0.55 + 0.45 * wet;
    if (count > 0) for (let i = 0; i < this._drAttrs.length; i++) this._drAttrs[i].needsUpdate = true;
  }

  _swapDrop(a, b) {
    if (a === b) return;
    const a3 = a * 3, b3 = b * 3, a4 = a * 4, b4 = b * 4;
    for (let k = 0; k < 3; k++) {
      let t = this._drPos[a3 + k]; this._drPos[a3 + k] = this._drPos[b3 + k]; this._drPos[b3 + k] = t;
      t = this._drVel[a3 + k]; this._drVel[a3 + k] = this._drVel[b3 + k]; this._drVel[b3 + k] = t;
    }
    for (let k = 0; k < 4; k++) {
      const t = this._drParams[a4 + k]; this._drParams[a4 + k] = this._drParams[b4 + k]; this._drParams[b4 + k] = t;
    }
    let t = this._drKind[a]; this._drKind[a] = this._drKind[b]; this._drKind[b] = t;
    t = this._drTheta[a]; this._drTheta[a] = this._drTheta[b]; this._drTheta[b] = t;
    t = this._drStick[a]; this._drStick[a] = this._drStick[b]; this._drStick[b] = t;
    t = this._drAge[a]; this._drAge[a] = this._drAge[b]; this._drAge[b] = t;
  }

  // -------------------------------------------------------------------------
  // Lightning
  // -------------------------------------------------------------------------

  _updateLightning(dt, state) {
    // Storm energy: explicit storminess, or rain heavy enough to imply a cell.
    const storm = clamp01(Math.max(state.clouds.storminess, ((this.rain - 0.62) / 0.38) * 0.75));

    if (storm > 0.02) {
      this._nextStrike -= dt * (0.25 + 1.9 * storm * storm);
      if (this._burstLeft > 0) {
        this._burstTimer -= dt;
        if (this._burstTimer <= 0) {
          this._burstLeft--;
          this._burstTimer = 0.25 + this.rng() * 1.2;
          this.strike(storm);
        }
      }
      if (this._nextStrike <= 0) {
        this._nextStrike = lerp(16, 3.2, storm) * (0.55 + this.rng() * 0.9);
        this.strike(storm);
        // Cells fire in bursts; one strike every N seconds reads as a metronome.
        if (this.rng() < 0.4) {
          this._burstLeft = 1 + Math.floor(this.rng() * 3);
          this._burstTimer = 0.3 + this.rng() * 0.9;
        }
      }
    } else {
      if (this._nextStrike < 4) this._nextStrike = 4;
      this._burstLeft = 0;
    }

    // Envelope. Attack is a couple of frames; the long tail is the cloud mass
    // still glowing after the channel itself is gone.
    this._strikeAge += dt;
    this._boltAge += dt;
    let env = 0;
    let boltEnv = 0;
    for (let i = 0; i < this._pulseCount; i++) {
      const t = this._strikeAge - this._pulseT[i];
      if (t < 0) continue;
      const rise = clamp01(t / 0.012);
      const a = this._pulseA[i] * rise;
      const e = a * Math.exp(-t * (14 + i * 5));
      if (e > env) env = e;
      const b = a * Math.exp(-t * (30 + i * 9));
      if (b > boltEnv) boltEnv = b;
    }
    if (this._pulseCount > 0) {
      const after = this._pulseA[0] * 0.22 * Math.exp(-this._strikeAge * 3.1);
      if (after > env) env = after;
    }
    this.flash = env;
    this.boltFlash = boltEnv;

    // Cooperative write: state.weather.lightning is owned by the weather system,
    // so only overwrite the value we ourselves left there last frame; otherwise
    // take the brighter of the two so neither system can erase the other's flash.
    const cur = state.weather.lightning;
    state.weather.lightning = Math.abs(cur - this._lastLightningWrite) < 1e-4 ? env : Math.max(cur, env);
    this._lastLightningWrite = state.weather.lightning;

    if (this._boltActive && (boltEnv < 0.004 || this._boltAge > 1.2)) {
      this._boltActive = false;
      this.boltMesh.visible = false;
      this.boltGeo.setDrawRange(0, 0);
    }

    // Thunder queue: sound arrives distance / 343 m/s after the flash.
    if (this.thunder > 0) this.thunder = Math.max(0, this.thunder - dt * 0.55);
    for (let i = 0; i < this._thunderCount; i++) {
      this._thunderTime[i] -= dt;
      if (this._thunderTime[i] > 0) continue;
      const d = this._thunderDist[i];
      const amp = this._thunderAmp[i];
      this.thunder = Math.min(1, this.thunder + amp);
      _thunderEvent.distance = d;
      _thunderEvent.intensity = amp;
      this.bus?.emit(THUNDER_EVENT, _thunderEvent);
      this._playThunder(d, amp);
      this._thunderCount--;
      this._thunderTime[i] = this._thunderTime[this._thunderCount];
      this._thunderDist[i] = this._thunderDist[this._thunderCount];
      this._thunderAmp[i] = this._thunderAmp[this._thunderCount];
      i--;
    }
  }

  /**
   * Fire a strike. Public so the HUD or a debug key can trigger one.
   * @param {number} storm 0..1 storm energy; drives distance and brightness.
   */
  strike(storm = 1) {
    const rng = this.rng;
    // Most flashes are distant intracloud illumination; the near cloud-to-ground
    // bolt you actually see is the exception, which is what makes it land.
    const distance = lerp(420, 4200, Math.pow(rng(), 0.55)) * lerp(1.25, 0.75, storm);
    const near = clamp01(1 - (distance - 400) / 3600);
    const amp = clamp01((0.32 + 0.85 * near * near) * (0.55 + 0.6 * storm));

    // Multi-flicker: 2-5 return strokes, each dimmer, 30-140 ms apart.
    const strokes = 2 + Math.floor(rng() * 4);
    this._pulseCount = Math.min(strokes, this._pulseT.length);
    let t = 0;
    for (let i = 0; i < this._pulseCount; i++) {
      this._pulseT[i] = t;
      this._pulseA[i] = amp * (i === 0 ? 1 : 0.35 + rng() * 0.55) * (1 - i * 0.09);
      t += 0.028 + rng() * 0.11;
    }
    this._strikeAge = 0;

    const az = rng() * TAU;
    const gx = _camPos.x + Math.cos(az) * distance;
    const gz = _camPos.z + Math.sin(az) * distance;
    const visible = distance < 3000 && rng() < 0.34;
    if (visible) {
      this._generateBolt(gx, gz, distance, amp);
    } else {
      this._boltActive = false;
      this.boltMesh.visible = false;
      this.boltGeo.setDrawRange(0, 0);
    }

    const delay = distance / SPEED_OF_SOUND;
    if (this._thunderCount < this._thunderTime.length) {
      const i = this._thunderCount++;
      this._thunderTime[i] = delay;
      this._thunderDist[i] = distance;
      this._thunderAmp[i] = clamp01(0.25 + 0.85 * near * near) * (0.5 + 0.5 * amp);
    }

    _strikeEvent.distance = distance;
    _strikeEvent.intensity = amp;
    _strikeEvent.visible = visible;
    _strikeEvent.thunderDelay = delay;
    _strikeEvent.position.set(gx, this.terrain?.getHeight ? this.terrain.getHeight(gx, gz) : 0, gz);
    this.bus?.emit(EVENTS.LIGHTNING_STRIKE, _strikeEvent);
  }

  /** Recursive midpoint displacement with forks, written into the ribbon buffer. */
  _generateBolt(gx, gz, distance, amp) {
    const rng = this.rng;
    const top = Math.max(320, this.state.clouds.altitude || 900);
    const ground = this.terrain?.getHeight ? this.terrain.getHeight(gx, gz) : 0;
    // The channel starts inside the cloud base, offset from the ground point:
    // lightning is never a plumb line.
    const spread = (top - ground) * 0.22;
    const sx = gx + (rng() - 0.5) * spread;
    const sz = gz + (rng() - 0.5) * spread;

    // Two axes perpendicular to the overall channel direction. Displacing only in
    // that plane keeps the bolt from wandering upward or doubling back on itself.
    let ax = gx - sx, ay = ground - top, az = gz - sz;
    const al = Math.hypot(ax, ay, az) || 1;
    ax /= al; ay /= al; az /= al;
    let px = -ay, py = ax, pz = 0;
    let pl = Math.hypot(px, py, pz);
    if (pl < 1e-3) { px = 1; py = 0; pz = 0; pl = 1; }
    px /= pl; py /= pl; pz /= pl;
    const qx = ay * pz - az * py, qy = az * px - ax * pz, qz = ax * py - ay * px;

    // --- main channel ---
    let src = this._ptA, dst = this._ptB;
    src[0] = sx; src[1] = top; src[2] = sz;
    src[3] = gx; src[4] = ground; src[5] = gz;
    let n = 2;
    for (let it = 0; it < 6; it++) {
      if (n * 2 - 1 > MAX_BOLT_POINTS) break;
      n = this._subdivide(src, n, dst, 0.30 * Math.pow(0.62, it), px, py, pz, qx, qy, qz, rng);
      const tmp = src; src = dst; dst = tmp;
    }

    const pts = this._chPts;
    const maxSeg = this.tier.boltSeg;
    this._ptCount = 0;
    this._chTotal = 0;
    this._segTotal = 0;
    this._pushChannel(src, n, 1, 1, maxSeg);
    const mainCount = this._chTotal > 0 ? this._chCount[0] : 0;

    // --- forks: the channel splits as it nears the ground, not at the cloud ---
    const forkTarget = 3 + Math.floor(rng() * 6);
    for (let f = 0; f < forkTarget && mainCount > 4; f++) {
      const idx = Math.floor(lerp(mainCount * 0.2, mainCount * 0.92, rng()));
      const bi = idx * 3;
      const bx = pts[bi], by = pts[bi + 1], bz = pts[bi + 2];
      const remaining = by - ground;
      if (remaining < 30) continue;
      const len = remaining * (0.22 + rng() * 0.5);
      const s1 = (rng() - 0.5) * 2;
      const s2 = (rng() - 0.5) * 2;
      const dirx = ax * 0.55 + px * s1 * 0.9 + qx * s2 * 0.9;
      const diry = ay * 0.9 + py * s1 * 0.5 + qy * s2 * 0.5;
      const dirz = az * 0.55 + pz * s1 * 0.9 + qz * s2 * 0.9;
      const dl = Math.hypot(dirx, diry, dirz) || 1;
      let fs = this._ptA, fd = this._ptB;
      fs[0] = bx; fs[1] = by; fs[2] = bz;
      fs[3] = bx + (dirx / dl) * len;
      fs[4] = Math.max(ground, by + (diry / dl) * len);
      fs[5] = bz + (dirz / dl) * len;
      let fn = 2;
      const fIters = 3 + Math.floor(rng() * 2);
      for (let it = 0; it < fIters; it++) {
        if (fn * 2 - 1 > MAX_BOLT_POINTS) break;
        fn = this._subdivide(fs, fn, fd, 0.34 * Math.pow(0.6, it), px, py, pz, qx, qy, qz, rng);
        const tmp = fs; fs = fd; fd = tmp;
      }
      this._pushChannel(fs, fn, 0.42 + rng() * 0.3, 0.32 + rng() * 0.3, maxSeg);
    }

    // --- ribbon ---
    const P = this._boltPos, T = this._boltTan, O = this._boltOfs;
    let v = 0, seg = 0;
    for (let c = 0; c < this._chTotal; c++) {
      const start = this._chStart[c], cnt = this._chCount[c];
      const cw = this._chWidth[c], cb = this._chBright[c];
      for (let i = 0; i < cnt - 1 && seg < maxSeg; i++) {
        const i0 = (start + i) * 3, i1 = (start + i + 1) * 3;
        // Smoothed tangents at the shared points miter the joints. Without this
        // the ribbon opens a gap at every sharp kink, and a bolt is all kinks.
        this._tangentAt(pts, start, cnt, i, _v3);
        const t0x = _v3.x, t0y = _v3.y, t0z = _v3.z;
        this._tangentAt(pts, start, cnt, i + 1, _v3);
        const t1x = _v3.x, t1y = _v3.y, t1z = _v3.z;
        // Taper toward the tip, and flicker segment brightness a little so the
        // channel is not a uniform tube of light.
        const f0 = 1 - (i / cnt) * 0.55, f1 = 1 - ((i + 1) / cnt) * 0.55;
        const jitter = 0.75 + rng() * 0.5;
        for (let k = 0; k < 4; k++) {
          const end = _quadEnd[k];
          const si = end ? i1 : i0;
          const wf = end ? f1 : f0;
          const v3 = v * 3;
          P[v3] = pts[si]; P[v3 + 1] = pts[si + 1]; P[v3 + 2] = pts[si + 2];
          T[v3] = end ? t1x : t0x;
          T[v3 + 1] = end ? t1y : t0y;
          T[v3 + 2] = end ? t1z : t0z;
          O[v3] = _quadSide[k];
          O[v3 + 1] = cw * wf;
          O[v3 + 2] = cb * wf * jitter;
          v++;
        }
        seg++;
      }
    }

    for (let i = 0; i < this._boltAttrs.length; i++) this._boltAttrs[i].needsUpdate = true;
    this.boltGeo.setDrawRange(0, seg * 6);
    this.boltUniforms.uWidth.value = lerp(1.4, 5.5, clamp01(distance / 3000)) * (0.7 + 0.6 * amp);
    this._boltActive = seg > 0;
    this._boltAge = 0;
    this.boltMesh.visible = this._boltActive;
  }

  /** Copy a generated polyline into the channel table. */
  _pushChannel(buf, count, width, bright, maxSeg) {
    const c = this._chTotal;
    if (c >= MAX_CHANNELS) return;
    if (this._ptCount + count > MAX_BOLT_POINTS) return;
    if (this._segTotal + count - 1 > maxSeg) return;
    this._chStart[c] = this._ptCount;
    this._chCount[c] = count;
    this._chWidth[c] = width;
    this._chBright[c] = bright;
    const dst = this._chPts;
    const base = this._ptCount * 3;
    for (let i = 0; i < count * 3; i++) dst[base + i] = buf[i];
    this._ptCount += count;
    this._segTotal += count - 1;
    this._chTotal = c + 1;
  }

  /** Central-difference tangent along a channel, normalised. */
  _tangentAt(pts, start, count, i, out) {
    const a = Math.max(0, i - 1), b = Math.min(count - 1, i + 1);
    const ai = (start + a) * 3, bi = (start + b) * 3;
    const x = pts[bi] - pts[ai], y = pts[bi + 1] - pts[ai + 1], z = pts[bi + 2] - pts[ai + 2];
    const l = Math.hypot(x, y, z);
    if (l < 1e-5) out.set(0, -1, 0);
    else out.set(x / l, y / l, z / l);
    return out;
  }

  /** One level of midpoint displacement. Returns the new point count. */
  _subdivide(src, n, dst, amp, px, py, pz, qx, qy, qz, rng) {
    let m = 0;
    for (let i = 0; i < n - 1; i++) {
      const a = i * 3, b = (i + 1) * 3;
      const ax = src[a], ay = src[a + 1], az = src[a + 2];
      const bx = src[b], by = src[b + 1], bz = src[b + 2];
      const segLen = Math.hypot(bx - ax, by - ay, bz - az);
      const d1 = (rng() * 2 - 1) * amp * segLen;
      const d2 = (rng() * 2 - 1) * amp * segLen;
      dst[m * 3] = ax; dst[m * 3 + 1] = ay; dst[m * 3 + 2] = az; m++;
      dst[m * 3] = (ax + bx) * 0.5 + px * d1 + qx * d2;
      dst[m * 3 + 1] = (ay + by) * 0.5 + py * d1 + qy * d2;
      dst[m * 3 + 2] = (az + bz) * 0.5 + pz * d1 + qz * d2;
      m++;
    }
    const last = (n - 1) * 3;
    dst[m * 3] = src[last]; dst[m * 3 + 1] = src[last + 1]; dst[m * 3 + 2] = src[last + 2];
    return m + 1;
  }

  // -------------------------------------------------------------------------
  // Thunder - synthesised, no assets, created only after a real user gesture.
  // -------------------------------------------------------------------------

  _initAudio() {
    if (this._audioCtx || !this.audioEnabled) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    try {
      const ac = new AC();
      if (ac.state === 'suspended') ac.resume().catch(() => {});
      const gain = ac.createGain();
      gain.gain.value = 0.32;
      gain.connect(ac.destination);
      // Two seconds of pink-ish noise (Voss-McCartney style filter cascade),
      // reused for every peal. Pink is the right spectrum: thunder is broadband
      // with far more energy low down than white noise would give.
      const sr = ac.sampleRate;
      const buf = ac.createBuffer(1, Math.floor(sr * 2), sr);
      const d = buf.getChannelData(0);
      let b0 = 0, b1 = 0, b2 = 0;
      const nrng = makeRNG(0x2ad13);
      for (let i = 0; i < d.length; i++) {
        const w = nrng() * 2 - 1;
        b0 = 0.99765 * b0 + w * 0.0990460;
        b1 = 0.96300 * b1 + w * 0.2965164;
        b2 = 0.57000 * b2 + w * 1.0526913;
        d[i] = (b0 + b1 + b2 + w * 0.1848) * 0.22;
      }
      this._audioCtx = ac;
      this._audioGain = gain;
      this._noiseBuf = buf;
    } catch (err) {
      this._audioCtx = null;   // no audio device / blocked: stay silent, never throw
    }
  }

  _playThunder(distance, amp) {
    const ac = this._audioCtx;
    if (!ac || !this.audioEnabled || !this._noiseBuf || ac.state !== 'running') return;
    const now = ac.currentTime;
    const near = clamp01(1 - distance / 4200);
    const src = ac.createBufferSource();
    src.buffer = this._noiseBuf;
    src.loop = true;
    src.playbackRate.value = 0.55 + near * 0.5;
    const lp = ac.createBiquadFilter();
    lp.type = 'lowpass';
    // Air absorption is strongly frequency dependent: distance eats the crack and
    // leaves only the rumble. This one filter is the whole sense of distance.
    lp.frequency.value = 160 + 2600 * Math.pow(near, 2.2);
    lp.Q.value = 0.7;
    const g = ac.createGain();
    const dur = 1.6 + (1 - near) * 3.4;
    const attack = 0.004 + (1 - near) * 0.35;
    g.gain.setValueAtTime(0.0001, now);
    g.gain.linearRampToValueAtTime(Math.max(0.001, amp * (0.35 + 0.65 * near)), now + attack);
    g.gain.exponentialRampToValueAtTime(0.0001, now + dur);
    src.connect(lp); lp.connect(g); g.connect(this._audioGain);
    src.start(now);
    src.stop(now + dur + 0.05);
    src.onended = () => {
      try { src.disconnect(); lp.disconnect(); g.disconnect(); } catch (err) { /* already torn down */ }
    };
  }

  /** Turn synthesised thunder on or off at runtime. */
  setThunderAudio(on) {
    this.audioEnabled = !!on;
    if (this._audioGain) this._audioGain.gain.value = on ? 0.32 : 0;
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  onQualityChange(quality) {
    const tier = TIERS[quality?.tier] || TIERS[this.state.quality.tier] || TIERS.high;
    this.tier = tier;
    if (!this.rainUniforms) return;
    this.rainUniforms.uBox.value.set(tier.rainBox[0], tier.rainBox[1], tier.rainBox[2]);
    this.rainUniforms.uFadeEnd.value = tier.rainFade;
    this.snowUniforms.uBox.value.set(tier.snowBox[0], tier.snowBox[1], tier.snowBox[2]);
    this.snowUniforms.uFadeEnd.value = tier.snowFade;
    this._sharedUniforms.uMinPixels.value = tier.minPx;
    // Reseat every live flake into the new volume so none sits outside it.
    for (let i = 0; i < this._snowSimulated; i++) {
      this._placeSnow(i, tier.snowBox[0], tier.snowBox[1], tier.snowBox[2], false);
    }
    this._snowAttr.needsUpdate = true;
    if (this._spCount > tier.splash) this._spCount = tier.splash;
    if (this._drCount > tier.drops) this._drCount = tier.drops;
    if (!tier.crown) this._crown.mesh.visible = false;
  }

  resize() {
    // uPixelScale is recomputed every frame from the drawing buffer, so there is
    // nothing to do here. Kept so the lifecycle contract stays explicit.
  }

  dispose() {
    if (typeof window !== 'undefined') {
      window.removeEventListener('pointerdown', this._unlockAudio);
      window.removeEventListener('keydown', this._unlockAudio);
    }
    this.scene?.remove(this.group);
    this.scene?.remove(this.worldGroup);
    const meshes = [this.rainMesh, this.snowMesh, this._ring?.mesh, this._crown?.mesh, this.dropMesh, this.boltMesh];
    for (const m of meshes) {
      if (!m) continue;
      m.geometry?.dispose();
      m.material?.dispose();
    }
    // The snow sprite belongs to the TextureFactory unless we had to make our own.
    this._ownSprite?.dispose();
    if (this._audioCtx) {
      try { this._audioCtx.close(); } catch (err) { /* already closed */ }
      this._audioCtx = null;
    }
  }
}
