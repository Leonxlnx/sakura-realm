/**
 * math.js - dependency-free noise + numeric helpers shared by every system.
 *
 * Everything here is deterministic: same seed in, same world out. Systems that
 * need matching randomness across modules (grass placement vs. scatter placement,
 * for example) must derive from the same seed rather than Math.random().
 */

// ---------------------------------------------------------------------------
// Scalars
// ---------------------------------------------------------------------------

export const PI = Math.PI;
export const TAU = Math.PI * 2;
export const DEG2RAD = Math.PI / 180;
export const RAD2DEG = 180 / Math.PI;

export const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
export const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
export const saturate = clamp01;
export const lerp = (a, b, t) => a + (b - a) * t;
export const inverseLerp = (a, b, v) => (a === b ? 0 : (v - a) / (b - a));
export const remap = (v, a, b, c, d) => c + ((v - a) / (b - a)) * (d - c);
export const remapClamped = (v, a, b, c, d) => clamp(remap(v, a, b, c, d), Math.min(c, d), Math.max(c, d));
export const step = (edge, x) => (x < edge ? 0 : 1);
export const sign = (x) => (x > 0 ? 1 : x < 0 ? -1 : 0);
export const fract = (x) => x - Math.floor(x);
export const mod = (x, m) => ((x % m) + m) % m;

export function smoothstep(edge0, edge1, x) {
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

export function smootherstep(edge0, edge1, x) {
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * t * (t * (t * 6 - 15) + 10);
}

/**
 * Framerate-independent exponential smoothing.
 * `lambda` is the decay rate: higher = snappier. Prefer this over naive
 * `lerp(a, b, 0.1)` in update loops, which changes behaviour with framerate.
 */
export const damp = (a, b, lambda, dt) => lerp(a, b, 1 - Math.exp(-lambda * dt));

/** Angular damp that takes the short way around the circle. */
export function dampAngle(a, b, lambda, dt) {
  let d = mod(b - a + PI, TAU) - PI;
  return a + d * (1 - Math.exp(-lambda * dt));
}

export const pingPong = (x, len) => len - Math.abs(mod(x, len * 2) - len);

// ---------------------------------------------------------------------------
// Easing (all map 0..1 -> 0..1)
// ---------------------------------------------------------------------------

export const Easing = {
  linear: (t) => t,
  inQuad: (t) => t * t,
  outQuad: (t) => t * (2 - t),
  inOutQuad: (t) => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t),
  inCubic: (t) => t * t * t,
  outCubic: (t) => --t * t * t + 1,
  inOutCubic: (t) => (t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1),
  inExpo: (t) => (t === 0 ? 0 : Math.pow(2, 10 * t - 10)),
  outExpo: (t) => (t === 1 ? 1 : 1 - Math.pow(2, -10 * t)),
  inOutSine: (t) => -(Math.cos(PI * t) - 1) / 2,
  outBack: (t) => 1 + 2.70158 * Math.pow(t - 1, 3) + 1.70158 * Math.pow(t - 1, 2),
  outElastic: (t) =>
    t === 0 ? 0 : t === 1 ? 1 : Math.pow(2, -10 * t) * Math.sin(((t * 10 - 0.75) * TAU) / 3) + 1,
};

// ---------------------------------------------------------------------------
// Deterministic hashing / PRNG
// ---------------------------------------------------------------------------

/** 32-bit integer hash (Wang-style). Returns 0..1. */
export function hash1(n) {
  n = (n << 13) ^ n;
  n = (n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff;
  return n / 0x7fffffff;
}

export function hash2(x, y) {
  let h = x * 374761393 + y * 668265263;
  h = (h ^ (h >> 13)) * 1274126177;
  return ((h ^ (h >> 16)) >>> 0) / 4294967295;
}

export function hash3(x, y, z) {
  let h = x * 374761393 + y * 668265263 + z * 2147483647;
  h = (h ^ (h >> 13)) * 1274126177;
  return ((h ^ (h >> 16)) >>> 0) / 4294967295;
}

/**
 * mulberry32 - small, fast, well-distributed seeded PRNG.
 * `const rand = makeRNG(1337); rand();`
 */
export function makeRNG(seed = 1) {
  let a = seed >>> 0;
  const rand = () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  rand.range = (min, max) => min + rand() * (max - min);
  rand.int = (min, max) => Math.floor(min + rand() * (max - min + 1));
  rand.sign = () => (rand() < 0.5 ? -1 : 1);
  rand.pick = (arr) => arr[Math.floor(rand() * arr.length)];
  /** Box-Muller normal distribution. */
  rand.gaussian = (mean = 0, stdev = 1) => {
    const u = 1 - rand();
    const v = rand();
    return mean + stdev * Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v);
  };
  return rand;
}

// ---------------------------------------------------------------------------
// Simplex noise (Ashima/Gustavson derived, 2D + 3D)
// ---------------------------------------------------------------------------

const GRAD3 = new Float32Array([
  1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1, 0,
  1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, -1,
  0, 1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1,
]);

const F2 = 0.5 * (Math.sqrt(3) - 1);
const G2 = (3 - Math.sqrt(3)) / 6;
const F3 = 1 / 3;
const G3 = 1 / 6;

/**
 * Creates an independent simplex noise instance with its own permutation table.
 * Two instances with different seeds produce uncorrelated fields - use separate
 * instances for terrain vs. clouds vs. wind so tweaking one never shifts another.
 */
export function createNoise(seed = 0) {
  const rand = makeRNG(seed + 1);
  const p = new Uint8Array(256);
  for (let i = 0; i < 256; i++) p[i] = i;
  // Fisher-Yates with the seeded RNG.
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const t = p[i];
    p[i] = p[j];
    p[j] = t;
  }
  const perm = new Uint8Array(512);
  const permMod12 = new Uint8Array(512);
  for (let i = 0; i < 512; i++) {
    perm[i] = p[i & 255];
    permMod12[i] = perm[i] % 12;
  }

  /** 2D simplex noise, range roughly -1..1. */
  function noise2D(xin, yin) {
    let n0 = 0, n1 = 0, n2 = 0;
    const s = (xin + yin) * F2;
    const i = Math.floor(xin + s);
    const j = Math.floor(yin + s);
    const t = (i + j) * G2;
    const x0 = xin - (i - t);
    const y0 = yin - (j - t);

    let i1, j1;
    if (x0 > y0) { i1 = 1; j1 = 0; } else { i1 = 0; j1 = 1; }

    const x1 = x0 - i1 + G2;
    const y1 = y0 - j1 + G2;
    const x2 = x0 - 1 + 2 * G2;
    const y2 = y0 - 1 + 2 * G2;

    const ii = i & 255;
    const jj = j & 255;

    let t0 = 0.5 - x0 * x0 - y0 * y0;
    if (t0 >= 0) {
      const gi0 = permMod12[ii + perm[jj]] * 3;
      t0 *= t0;
      n0 = t0 * t0 * (GRAD3[gi0] * x0 + GRAD3[gi0 + 1] * y0);
    }
    let t1 = 0.5 - x1 * x1 - y1 * y1;
    if (t1 >= 0) {
      const gi1 = permMod12[ii + i1 + perm[jj + j1]] * 3;
      t1 *= t1;
      n1 = t1 * t1 * (GRAD3[gi1] * x1 + GRAD3[gi1 + 1] * y1);
    }
    let t2 = 0.5 - x2 * x2 - y2 * y2;
    if (t2 >= 0) {
      const gi2 = permMod12[ii + 1 + perm[jj + 1]] * 3;
      t2 *= t2;
      n2 = t2 * t2 * (GRAD3[gi2] * x2 + GRAD3[gi2 + 1] * y2);
    }
    return 70 * (n0 + n1 + n2);
  }

  /** 3D simplex noise, range roughly -1..1. */
  function noise3D(xin, yin, zin) {
    let n0 = 0, n1 = 0, n2 = 0, n3 = 0;
    const s = (xin + yin + zin) * F3;
    const i = Math.floor(xin + s);
    const j = Math.floor(yin + s);
    const k = Math.floor(zin + s);
    const t = (i + j + k) * G3;
    const x0 = xin - (i - t);
    const y0 = yin - (j - t);
    const z0 = zin - (k - t);

    let i1, j1, k1, i2, j2, k2;
    if (x0 >= y0) {
      if (y0 >= z0)      { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
      else if (x0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; }
      else               { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; }
    } else {
      if (y0 < z0)       { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; }
      else if (x0 < z0)  { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; }
      else               { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
    }

    const x1 = x0 - i1 + G3,      y1 = y0 - j1 + G3,      z1 = z0 - k1 + G3;
    const x2 = x0 - i2 + 2 * G3,  y2 = y0 - j2 + 2 * G3,  z2 = z0 - k2 + 2 * G3;
    const x3 = x0 - 1 + 3 * G3,   y3 = y0 - 1 + 3 * G3,   z3 = z0 - 1 + 3 * G3;

    const ii = i & 255, jj = j & 255, kk = k & 255;

    let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
    if (t0 >= 0) {
      const gi0 = permMod12[ii + perm[jj + perm[kk]]] * 3;
      t0 *= t0;
      n0 = t0 * t0 * (GRAD3[gi0] * x0 + GRAD3[gi0 + 1] * y0 + GRAD3[gi0 + 2] * z0);
    }
    let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
    if (t1 >= 0) {
      const gi1 = permMod12[ii + i1 + perm[jj + j1 + perm[kk + k1]]] * 3;
      t1 *= t1;
      n1 = t1 * t1 * (GRAD3[gi1] * x1 + GRAD3[gi1 + 1] * y1 + GRAD3[gi1 + 2] * z1);
    }
    let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
    if (t2 >= 0) {
      const gi2 = permMod12[ii + i2 + perm[jj + j2 + perm[kk + k2]]] * 3;
      t2 *= t2;
      n2 = t2 * t2 * (GRAD3[gi2] * x2 + GRAD3[gi2 + 1] * y2 + GRAD3[gi2 + 2] * z2);
    }
    let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
    if (t3 >= 0) {
      const gi3 = permMod12[ii + 1 + perm[jj + 1 + perm[kk + 1]]] * 3;
      t3 *= t3;
      n3 = t3 * t3 * (GRAD3[gi3] * x3 + GRAD3[gi3 + 1] * y3 + GRAD3[gi3 + 2] * z3);
    }
    return 32 * (n0 + n1 + n2 + n3);
  }

  /** Fractal Brownian motion over 2D simplex. */
  function fbm2D(x, y, octaves = 4, lacunarity = 2, gain = 0.5) {
    let sum = 0, amp = 0.5, freq = 1, norm = 0;
    for (let i = 0; i < octaves; i++) {
      sum += amp * noise2D(x * freq, y * freq);
      norm += amp;
      freq *= lacunarity;
      amp *= gain;
    }
    return sum / norm;
  }

  function fbm3D(x, y, z, octaves = 4, lacunarity = 2, gain = 0.5) {
    let sum = 0, amp = 0.5, freq = 1, norm = 0;
    for (let i = 0; i < octaves; i++) {
      sum += amp * noise3D(x * freq, y * freq, z * freq);
      norm += amp;
      freq *= lacunarity;
      amp *= gain;
    }
    return sum / norm;
  }

  /** Ridged multifractal - good for mountain silhouettes and cloud wisps. */
  function ridged2D(x, y, octaves = 4, lacunarity = 2, gain = 0.5) {
    let sum = 0, amp = 0.5, freq = 1, norm = 0;
    for (let i = 0; i < octaves; i++) {
      const n = 1 - Math.abs(noise2D(x * freq, y * freq));
      sum += amp * n * n;
      norm += amp;
      freq *= lacunarity;
      amp *= gain;
    }
    return sum / norm;
  }

  /** Billowy noise - puffy, cumulus-like. */
  function billow2D(x, y, octaves = 4, lacunarity = 2, gain = 0.5) {
    let sum = 0, amp = 0.5, freq = 1, norm = 0;
    for (let i = 0; i < octaves; i++) {
      sum += amp * Math.abs(noise2D(x * freq, y * freq));
      norm += amp;
      freq *= lacunarity;
      amp *= gain;
    }
    return sum / norm;
  }

  /**
   * 2D curl of the noise field - divergence-free, so particles advected by it
   * swirl instead of clumping. This is what drives gusts and petal drift.
   */
  function curl2D(x, y, eps = 0.0001, out = { x: 0, y: 0 }) {
    const n1 = noise2D(x, y + eps);
    const n2 = noise2D(x, y - eps);
    const n3 = noise2D(x + eps, y);
    const n4 = noise2D(x - eps, y);
    out.x = (n1 - n2) / (2 * eps);
    out.y = -(n3 - n4) / (2 * eps);
    return out;
  }

  /** 3D curl noise for volumetric advection (petals tumbling in air). */
  function curl3D(x, y, z, eps = 0.0001, out = { x: 0, y: 0, z: 0 }) {
    const dydz = (noise3D(x, y + eps, z) - noise3D(x, y - eps, z)) / (2 * eps);
    const dzdy = (noise3D(x, y, z + eps) - noise3D(x, y, z - eps)) / (2 * eps);
    const dzdx = (noise3D(x + eps, y, z) - noise3D(x - eps, y, z)) / (2 * eps);
    const dxdz = (noise3D(x, y, z + eps + 19.7) - noise3D(x, y, z - eps + 19.7)) / (2 * eps);
    const dxdy = (noise3D(x, y + eps + 31.3, z) - noise3D(x, y - eps + 31.3, z)) / (2 * eps);
    const dydx = (noise3D(x + eps + 47.1, y, z) - noise3D(x - eps + 47.1, y, z)) / (2 * eps);
    out.x = dydz - dzdy;
    out.y = dzdx - dxdz;
    out.z = dxdy - dydx;
    return out;
  }

  return { noise2D, noise3D, fbm2D, fbm3D, ridged2D, billow2D, curl2D, curl3D, perm };
}

/** Shared default instance for casual use. Seed-sensitive systems make their own. */
export const noise = createNoise(1337);

// ---------------------------------------------------------------------------
// Worley / cellular noise - cloud shape detail and grass clumping
// ---------------------------------------------------------------------------

/** Returns distance to nearest feature point, roughly 0..1. */
export function worley2D(x, y, seed = 0) {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  let minDist = 8;
  for (let j = -1; j <= 1; j++) {
    for (let i = -1; i <= 1; i++) {
      const px = hash3(xi + i, yi + j, seed);
      const py = hash3(xi + i, yi + j, seed + 71);
      const dx = i + px - xf;
      const dy = j + py - yf;
      const d = dx * dx + dy * dy;
      if (d < minDist) minDist = d;
    }
  }
  return Math.sqrt(minDist);
}

export function worley3D(x, y, z, seed = 0) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const xf = x - xi, yf = y - yi, zf = z - zi;
  let minDist = 8;
  for (let k = -1; k <= 1; k++) {
    for (let j = -1; j <= 1; j++) {
      for (let i = -1; i <= 1; i++) {
        const px = hash3(xi + i, yi + j, zi + k);
        const py = hash3(xi + i + 131, yi + j, zi + k);
        const pz = hash3(xi + i, yi + j + 197, zi + k);
        const dx = i + px - xf;
        const dy = j + py - yf;
        const dz = k + pz - zf;
        const d = dx * dx + dy * dy + dz * dz;
        if (d < minDist) minDist = d;
      }
    }
  }
  return Math.sqrt(minDist);
}

// ---------------------------------------------------------------------------
// Sampling / distribution helpers
// ---------------------------------------------------------------------------

/**
 * Low-discrepancy 2D sequence. Far more even than Math.random() for scattering
 * grass and props - no visible clumps or bald patches.
 */
export function halton(index, base) {
  let result = 0;
  let f = 1 / base;
  let i = index;
  while (i > 0) {
    result += f * (i % base);
    i = Math.floor(i / base);
    f /= base;
  }
  return result;
}

/** Golden-ratio spiral point set - the most even disc distribution for the cost. */
export function fibonacciDisc(i, count, out = { x: 0, y: 0 }) {
  const golden = Math.PI * (3 - Math.sqrt(5));
  const r = Math.sqrt(i / count);
  const theta = i * golden;
  out.x = Math.cos(theta) * r;
  out.y = Math.sin(theta) * r;
  return out;
}

/** Even points on a sphere - used for star placement and probe directions. */
export function fibonacciSphere(i, count, out = { x: 0, y: 0, z: 0 }) {
  const golden = Math.PI * (3 - Math.sqrt(5));
  const y = 1 - (i / (count - 1)) * 2;
  const radius = Math.sqrt(Math.max(0, 1 - y * y));
  const theta = golden * i;
  out.x = Math.cos(theta) * radius;
  out.y = y;
  out.z = Math.sin(theta) * radius;
  return out;
}

// ---------------------------------------------------------------------------
// Colour helpers (plain numbers - no three.js dependency in this file)
// ---------------------------------------------------------------------------

/** Kelvin -> linear RGB. Physically-ish; good enough for sun/moon tinting. */
export function kelvinToRGB(kelvin, out = { r: 1, g: 1, b: 1 }) {
  const t = clamp(kelvin, 1000, 40000) / 100;
  let r, g, b;
  if (t <= 66) {
    r = 255;
    g = 99.4708025861 * Math.log(t) - 161.1195681661;
    b = t <= 19 ? 0 : 138.5177312231 * Math.log(t - 10) - 305.0447927307;
  } else {
    r = 329.698727446 * Math.pow(t - 60, -0.1332047592);
    g = 288.1221695283 * Math.pow(t - 60, -0.0755148492);
    b = 255;
  }
  out.r = clamp01(r / 255);
  out.g = clamp01(g / 255);
  out.b = clamp01(b / 255);
  return out;
}

/** Exposure value -> linear multiplier, matching photographic stops. */
export const ev100ToExposure = (ev100) => 1 / (1.2 * Math.pow(2, ev100));

// ---------------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------------

/** Rolling average, used by the perf monitor to avoid jittery readouts. */
export class RollingAverage {
  constructor(size = 60) {
    this.size = size;
    this.values = new Float32Array(size);
    this.index = 0;
    this.count = 0;
    this.sum = 0;
  }
  push(v) {
    if (this.count === this.size) this.sum -= this.values[this.index];
    else this.count++;
    this.values[this.index] = v;
    this.sum += v;
    this.index = (this.index + 1) % this.size;
    return this.sum / this.count;
  }
  get average() {
    return this.count ? this.sum / this.count : 0;
  }
  reset() {
    this.index = 0;
    this.count = 0;
    this.sum = 0;
    this.values.fill(0);
  }
}
