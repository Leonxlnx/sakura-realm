/**
 * weather/wind.js - WindField
 *
 * Owns `state.wind` and installs the real `state.wind.sample()` that grass,
 * petals, the tree, precipitation and the cloud advection all call.
 *
 * ---------------------------------------------------------------------------
 * WHY IT IS BUILT THIS WAY
 * ---------------------------------------------------------------------------
 * 1. DIVERGENCE-FREE BY CONSTRUCTION, NOT BY FINITE DIFFERENCES.
 *    The naive "curl noise" is `curl2D()` on a simplex field: four noise taps
 *    per octave, twelve for three octaves. At ~100 ns a tap that is 1.2 us per
 *    sample - 3.6 ms/frame for 3000 petals. Unshippable on a 780M.
 *
 *    Instead each octave is a *band-limited spectral field* baked once into a
 *    small periodic lookup table. Every Fourier mode is written as
 *        v(p) = a * perp(k) * cos(k.p + phi)
 *    and since v is perpendicular to its own wavevector, div v = -a (k . perp(k))
 *    sin(...) = 0 EXACTLY, for every mode, hence for the sum. The baked table is
 *    therefore an exactly incompressible 2D velocity field; the only divergence
 *    left is the tiny reconstruction error of the interpolator, and the tables
 *    are deliberately oversampled (>= 9 texels per shortest wavelength) so that
 *    error is fractions of a percent. Advected particles swirl; they never clump.
 *
 *    Runtime cost per octave is one faded-bilinear fetch (~12 flops, 4 reads).
 *
 * 2. FROZEN TURBULENCE, NOT `noise(x, t)`.
 *    Taylor's frozen-turbulence hypothesis says atmospheric eddies are advected
 *    by the mean flow far faster than they evolve. So the octaves are *scrolled*
 *    through the world at (a multiple of) the mean wind velocity rather than
 *    animated on a time axis. That is both cheaper and more correct, and it is
 *    what makes a gust actually *arrive* instead of materialising in place.
 *    The scroll offset is INTEGRATED (`off += U * dt`), never `U * t` - the
 *    latter teleports the entire field whenever wind speed changes.
 *
 * 3. GUST COHERENCE IS THE WHOLE POINT.
 *    `getGustAt(x, z, t)` is a travelling scalar wave field: two sets of banded
 *    ripples sweeping downwind with wavy (noise-bent) crests, plus a pool of
 *    discrete gust FRONTS with a sharp leading edge and a long decaying tail.
 *    Grass, tree and petals all read the same function, so a front sweeping the
 *    field bends everything it touches at the same instant and nothing before
 *    it. Every system wobbling independently in place is the single loudest
 *    "this is fake" tell in a wind implementation.
 *
 * 4. NO ALLOCATION IN THE HOT PATH. The front pool is preallocated and
 *    swap-compacted, every loop is bounded by a quality-derived constant, and
 *    all per-sample intermediates are locals. The one residue JavaScript will
 *    not let us remove is the boxing of a double return value when V8 declines
 *    to inline `getGustAt` - 16 bytes a call, measured at ~50 KB/frame with
 *    3000 petals, which is a zero-survivor scavenge every few seconds. `update()`
 *    itself measures at zero GC events over 20 000 frames.
 *
 * Extra API installed on `state.wind` (beyond the documented `sample`):
 *    getGustAt(x, z, t) -> 0..2      coherent gust multiplier
 *    shearAt(y)         -> 0..2.5    log-law boundary-layer profile
 *    sample3(x,y,z,t,out)            sample() with shear applied
 */

import {
  TAU,
  clamp,
  clamp01,
  lerp,
  damp,
  dampAngle,
  mod,
  makeRNG,
  createNoise,
} from '../core/math.js';
import { EVENTS, WEATHER } from '../core/state.js';

// ---------------------------------------------------------------------------
// Tuning constants - all in world metres / seconds. Nothing here is arbitrary;
// each value is annotated with what it buys.
// ---------------------------------------------------------------------------

/**
 * The three turbulence octaves.
 *   tile  - spatial period of the baked field. Chosen mutually non-commensurate
 *           (224 / 62 / 17) so the combined pattern never visibly repeats.
 *   n     - LUT resolution, power of two so wrapping is a bitwise AND.
 *   kMin/kMax - integer wavenumber annulus in cycles-per-tile. tile/kMax is the
 *           shortest wavelength; keep it >= 9 * (tile/n) or bilinear reconstruction
 *           starts to show the grid.
 *   gain  - RMS speed this octave contributes, as a fraction of the mean speed.
 *   turbA/turbB - gain multiplier = turbA + turbB * state.wind.turbulence. The
 *           flutter octave is almost entirely turbulence-driven, the large-scale
 *           meander is not (a calm day still has lazy directional wander).
 *   adv   - advection speed as a multiple of the mean wind. Slightly different
 *           per octave so the layers slide against each other and decorrelate.
 *   perp  - cross-wind drift, same units. Breaks the "conveyor belt" read.
 *   drift - absolute minimum drift speed (m/s) on a slowly rotating heading, so
 *           the field still breathes when the mean wind is nearly zero (fog).
 *   wScale- vertical fluctuation as a fraction of this octave's horizontal gain.
 *           Real surface-layer sigma_w is about half sigma_u.
 */
const OCTAVES = [
  { tile: 224, n: 32, kMin: 1, kMax: 3, gain: 0.30, turbA: 1.00, turbB: 0.00, adv: 1.00, perp: 0.05, drift: 0.00, wScale: 0.12 },
  { tile: 62,  n: 64, kMin: 2, kMax: 5, gain: 0.20, turbA: 0.55, turbB: 0.55, adv: 1.06, perp: 0.14, drift: 0.18, wScale: 0.35 },
  { tile: 17,  n: 64, kMin: 3, kMax: 7, gain: 0.18, turbA: 0.15, turbB: 1.25, adv: 1.13, perp: -0.22, drift: 0.55, wScale: 0.50 },
];

/**
 * Index of the highest-frequency octave. This is the one LOW drops, so it is
 * named rather than written as a literal 2 in three places.
 */
const FLUTTER_OCTAVE = OCTAVES.length - 1;

/** Periodic 1D profile tables. 256 is plenty for wavelengths >= 18 m. */
const PROFILE_N = 256;

/** Hard ceiling on the front pool. ULTRA uses all of it; LOW uses three. */
const MAX_FRONTS = 8;

/**
 * Per-weather wind character. strength is m/s-ish, gustiness is the swing of
 * the gust envelope, turbulence drives the flutter octave.
 * Fog is deliberately the *calmest* setting - radiation fog only forms in still
 * air, and a fog bank in a gale reads as broken.
 */
const PROFILES = {
  [WEATHER.CLEAR]:         { strength: 1.9,  gustiness: 0.22, turbulence: 0.26 },
  [WEATHER.PARTLY_CLOUDY]: { strength: 3.2,  gustiness: 0.35, turbulence: 0.40 },
  [WEATHER.OVERCAST]:      { strength: 4.4,  gustiness: 0.46, turbulence: 0.46 },
  [WEATHER.RAIN]:          { strength: 5.6,  gustiness: 0.56, turbulence: 0.62 },
  [WEATHER.STORM]:         { strength: 10.0, gustiness: 0.86, turbulence: 0.90 },
  [WEATHER.PETAL_STORM]:   { strength: 7.6,  gustiness: 0.70, turbulence: 0.80 },
  [WEATHER.FOG]:           { strength: 0.95, gustiness: 0.13, turbulence: 0.16 },
};
const FALLBACK_PROFILE = PROFILES[WEATHER.PARTLY_CLOUDY];

/** Grassland aerodynamic roughness length, and the reference height (2 m). */
const ROUGHNESS_Z0 = 0.06;
const INV_LOG_REF = 1 / Math.log((2.0 + ROUGHNESS_Z0) / ROUGHNESS_Z0);

/** Shared output object for callers that omit `out`. Documented, never resized. */
const SHARED_OUT = { x: 0, y: 0, z: 0 };

// ---------------------------------------------------------------------------
// Bake helpers (init-time only - allocation here is fine)
// ---------------------------------------------------------------------------

/**
 * Integer lattice points in the annulus kMin <= |k| <= kMax, restricted to a
 * half-plane so we never include a mode and its conjugate twice (which would
 * just double an amplitude and skew the spectrum).
 */
function collectModes(kMin, kMax) {
  const out = [];
  for (let kz = 0; kz <= kMax; kz++) {
    for (let kx = -kMax; kx <= kMax; kx++) {
      if (kz === 0 && kx <= 0) continue; // drop DC and the mirrored half-row
      const k = Math.sqrt(kx * kx + kz * kz);
      if (k < kMin - 1e-6 || k > kMax + 1e-6) continue;
      out.push(kx, kz, k);
    }
  }
  return out;
}

/**
 * Bakes one octave: a periodic, exactly divergence-free 2D velocity field plus
 * an uncorrelated scalar used for the vertical component.
 *
 * Amplitudes follow |k|^(-4/3), which is the per-mode velocity amplitude implied
 * by a Kolmogorov k^(-5/3) energy spectrum once you account for the number of
 * lattice modes on a shell. Rayleigh-distributed amplitudes with uniform phases
 * make the result a proper Gaussian random field rather than a stack of visible
 * sine ripples.
 */
function bakeOctave(spec, seed) {
  const { tile, n, kMin, kMax } = spec;
  const modes = collectModes(kMin, kMax);
  const m = modes.length / 3;
  const rand = makeRNG(seed);
  const cell = tile / n;
  const angFreq = TAU / tile;
  const count = n * n;

  // Accumulate in float64 so 60+ modes summing into the same texel do not
  // pick up float32 rounding on every partial sum.
  const acc = new Float64Array(count * 3);

  for (let q = 0; q < m; q++) {
    const kx = modes[q * 3];
    const kz = modes[q * 3 + 1];
    const k = modes[q * 3 + 2];
    const amp = Math.pow(k, -4 / 3);
    // Rayleigh magnitude -> Gaussian field. 1 - rand() keeps the log finite.
    const rv = Math.sqrt(-2 * Math.log(1 - rand()));
    const rw = Math.sqrt(-2 * Math.log(1 - rand()));
    // Velocity strictly perpendicular to k => this mode has zero divergence.
    const ax = (amp * rv * kz) / k;
    const az = (amp * rv * -kx) / k;
    const aw = amp * rw;

    const phV = rand() * TAU;
    const phW = rand() * TAU;
    const cV = Math.cos(phV), sV = Math.sin(phV);
    const cW = Math.cos(phW), sW = Math.sin(phW);

    // Walking a row of the grid advances theta by a constant, so the row is a
    // repeated 2D rotation of (cos, sin) - two trig calls per row instead of
    // two per texel. Cuts the whole bake from ~40 ms to ~4 ms; float64 drift
    // over 64 steps is ~1e-15 and irrelevant.
    const stepX = kx * angFreq * cell;
    const stepZ = kz * angFreq * cell;
    const cd = Math.cos(stepX), sd = Math.sin(stepX);

    for (let j = 0; j < n; j++) {
      const th0 = stepZ * j;
      let c = Math.cos(th0);
      let s = Math.sin(th0);
      let o = j * n * 3;
      for (let i = 0; i < n; i++, o += 3) {
        // cos(theta + phi) = cos(theta)cos(phi) - sin(theta)sin(phi)
        const v = c * cV - s * sV;
        const w = c * cW - s * sW;
        acc[o] += ax * v;
        acc[o + 1] += az * v;
        acc[o + 2] += aw * w;
        const nc = c * cd - s * sd;
        s = s * cd + c * sd;
        c = nc;
      }
    }
  }

  // Normalise so "gain" means "RMS speed as a fraction of the mean wind",
  // independent of how many modes happened to land in the annulus.
  let sumH = 0;
  let sumW = 0;
  for (let i = 0; i < count; i++) {
    const o = i * 3;
    sumH += acc[o] * acc[o] + acc[o + 1] * acc[o + 1];
    sumW += acc[o + 2] * acc[o + 2];
  }
  const sh = sumH > 0 ? 1 / Math.sqrt(sumH / count) : 0;
  const sw = sumW > 0 ? 1 / Math.sqrt(sumW / count) : 0;

  const data = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const o = i * 3;
    data[o] = acc[o] * sh;
    data[o + 1] = acc[o + 1] * sh;
    data[o + 2] = acc[o + 2] * sw;
  }

  return {
    data,
    n,
    tile,
    invCell: n / tile,
    gain: spec.gain,
    turbA: spec.turbA,
    turbB: spec.turbB,
    adv: spec.adv,
    perp: spec.perp,
    drift: spec.drift,
    wScale: spec.wScale,
    // Per-frame derived state (written in update, read in sample).
    ox: 0,
    oz: 0,
    avx: 0,
    avz: 0,
    gainNow: 0,
    wGainNow: 0,
  };
}

/**
 * Periodic 1D scalar profile from a harmonic series.
 * `skew > 0` sharpens crests and softens troughs, which is how real gust
 * records look: brief peaks over a lower, flatter baseline. Symmetric noise
 * reads as a sine wave and kills the "here it comes" feeling.
 * `normalize` is 'rms' (unit standard deviation) or 'peak' (range +-1).
 */
function bakeProfile(harmonics, falloff, seed, skew, normalize) {
  const rand = makeRNG(seed);
  const amps = new Float64Array(harmonics);
  const phases = new Float64Array(harmonics);
  for (let k = 1; k <= harmonics; k++) {
    amps[k - 1] = Math.pow(k, -falloff) * Math.sqrt(-2 * Math.log(1 - rand()));
    phases[k - 1] = rand() * TAU;
  }

  const d = new Float32Array(PROFILE_N);
  for (let i = 0; i < PROFILE_N; i++) {
    const u = (i / PROFILE_N) * TAU;
    let s = 0;
    for (let k = 1; k <= harmonics; k++) s += amps[k - 1] * Math.cos(u * k + phases[k - 1]);
    d[i] = s;
  }

  const rescale = (mode) => {
    if (mode === 'peak') {
      let peak = 0;
      for (let i = 0; i < PROFILE_N; i++) peak = Math.max(peak, Math.abs(d[i]));
      if (peak > 0) for (let i = 0; i < PROFILE_N; i++) d[i] /= peak;
    } else {
      let sum = 0;
      let mean = 0;
      for (let i = 0; i < PROFILE_N; i++) mean += d[i];
      mean /= PROFILE_N;
      for (let i = 0; i < PROFILE_N; i++) sum += (d[i] - mean) * (d[i] - mean);
      const rms = Math.sqrt(sum / PROFILE_N);
      const inv = rms > 0 ? 1 / rms : 0;
      for (let i = 0; i < PROFILE_N; i++) d[i] = (d[i] - mean) * inv;
    }
  };

  rescale('rms');
  if (skew !== 0) {
    // v + s(v^2 - 1): positive skew with the mean preserved for unit-variance v,
    // and monotonic for v > -1/(2s), i.e. everywhere that matters here.
    for (let i = 0; i < PROFILE_N; i++) {
      const v = d[i];
      d[i] = v + skew * (v * v - 1);
    }
  }
  rescale(normalize);
  return d;
}

// ---------------------------------------------------------------------------

export class WindField {
  constructor(ctx) {
    this.ctx = ctx;
    this._state = ctx.state;
    this._bus = ctx.bus;

    const seed = 0x5a4b17;
    this._rand = makeRNG(seed ^ 0x9e3779b1);
    this._dirNoise = createNoise(seed + 91);

    // --- layers -------------------------------------------------------------
    /** @type {Array<ReturnType<typeof bakeOctave>>} */
    this._layers = [];
    /** 0 until init() has baked; keeps sample() branch-free and safe pre-init. */
    this._octaveCount = 0;
    this._ready = false;
    /** Crossfade for the flutter octave so LOW/HIGH switching never pops. */
    this._flutterMix = 1;
    this._flutterTarget = 1;

    // --- scratch (never allocate in sample()) -------------------------------
    this._time = 0;

    // --- mean flow ----------------------------------------------------------
    const d = this._state.wind.direction;
    this._baseHeading = Math.atan2(d.y, d.x);
    this._heading = this._baseHeading;
    this._headingPhase = 0;
    this._driftAngle = this._rand() * TAU;

    // --- gust bands ---------------------------------------------------------
    // Two travelling banded wave sets at slightly different headings, speeds and
    // wavelengths. Their interference is what turns straight stripes into the
    // roaming patches you see on a real grass field.
    const profileA = bakeProfile(7, 0.9, seed + 11, 0.26, 'rms');
    const profileB = bakeProfile(6, 1.0, seed + 29, 0.26, 'rms');
    /** Peak-normalised to +-1 exactly, so `wobble` is a hard bound in metres. */
    this._lateral = bakeProfile(4, 1.1, seed + 47, 0.0, 'peak');

    this._bands = [
      {
        profile: profileA,
        angle: -0.16, speedFactor: 0.98, minSpeed: 0.30,
        invLen: 1 / 128, wobble: 26, latInv: 1 / 165, latPhase: 0.0,
        dx: 1, dz: 0, nx: 0, nz: 1, travel: this._rand(), rate: 0,
      },
      {
        profile: profileB,
        angle: 0.21, speedFactor: 1.12, minSpeed: 0.45,
        invLen: 1 / 197, wobble: 34, latInv: 1 / 240, latPhase: 0.37,
        dx: 1, dz: 0, nx: 0, nz: 1, travel: this._rand(), rate: 0,
      },
    ];
    /** Scales the banded component. Tuned so a storm peaks near the soft ceiling. */
    this._bandGain = 0.6;

    // --- discrete gust fronts ----------------------------------------------
    // Built by push, not `new Array(n)`: the latter stays HOLEY in V8 even once
    // filled, and this array is indexed once per front per sample.
    this._fronts = [];
    for (let i = 0; i < MAX_FRONTS; i++) {
      this._fronts.push({
        dx: 1, dz: 0,
        pos: 0, speed: 1, speedRatio: 1,
        rise: 6, invRise: 1 / 6, invFall: 1 / 40, cutoff: 46,
        peak: 0, amp: 0, age: 0, fadeIn: 3,
        wobble: 12, latInv: 1 / 120, latPhase: 0,
      });
    }
    this._frontCount = 0;
    /** Exponential inter-arrival counter - a proper Poisson process, not a metronome. */
    this._nextSpawn = -Math.log(1 - this._rand());
    this._maxFronts = 6;

    this._applyTier(this._state.quality?.tier);

    // A downburst outflow follows a strike: the gust arrives seconds after the
    // flash, which is exactly the beat that sells a storm.
    this._offLightning = this._bus?.on(EVENTS.LIGHTNING_STRIKE, () => this._spawnDownburst());

    // Public entry points, created once as arrow closures so they survive being
    // detached (`state.wind.sample`, or a sibling caching `wind.getGustAt`).
    //
    // Deliberately NOT `Function.prototype.bind`. Measured: a bound function
    // invoked through a property load is not reduced by TurboFan and allocates
    // an argument list on every call - 4M samples produced 280 scavenges bound
    // versus 2 unbound. At 3000 petals a frame that is a GC hitch every few
    // seconds, which is the exact failure the no-allocation rule exists to stop.
    // The real work stays in prototype methods so the internal calls
    // (`this._gustImpl` inside `_sampleImpl`) remain monomorphic and inlinable.
    this.sample = (x, z, t, out) => this._sampleImpl(x, z, t, out);
    this.sample3 = (x, y, z, t, out) => this._sample3Impl(x, y, z, t, out);
    this.getGustAt = (x, z, t) => this._gustImpl(x, z, t);
    this.shearAt = (y) => this._shearImpl(y);

    const w = this._state.wind;
    w.sample = this.sample;
    w.sample3 = this.sample3;
    w.getGustAt = this.getGustAt;
    w.shearAt = this.shearAt;
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  async init() {
    for (let i = 0; i < OCTAVES.length; i++) {
      this._layers.push(bakeOctave(OCTAVES[i], 0x5a4b17 + i * 7919));
      // Start each octave at a random phase so the three tiles are not aligned
      // at the origin, which would otherwise produce one suspiciously calm spot.
      const L = this._layers[i];
      L.ox = this._rand() * L.tile;
      L.oz = this._rand() * L.tile;
    }
    this._ready = true;
    this._octaveCount = this._flutterMix > 0 ? OCTAVES.length : FLUTTER_OCTAVE;
  }

  onQualityChange(quality) {
    this._applyTier(quality?.tier);
  }

  _applyTier(tier) {
    switch (tier) {
      case 'low':
        this._maxFronts = 3;
        this._flutterTarget = 0;
        break;
      case 'medium':
        this._maxFronts = 4;
        this._flutterTarget = 1;
        break;
      case 'ultra':
        this._maxFronts = MAX_FRONTS;
        this._flutterTarget = 1;
        break;
      default:
        this._maxFronts = 6;
        this._flutterTarget = 1;
        break;
    }
    // Before the first frame there is nothing on screen to pop, so snap.
    if (!this._ready) this._flutterMix = this._flutterTarget;
    // Existing fronts are never culled on the spot - that would make the wind
    // visibly jump. `_maxFronts` gates spawning; the pool drains naturally and
    // the cost falls within a few seconds.
  }

  dispose() {
    if (this._offLightning) this._offLightning();
    const w = this._state.wind;
    // Leave a harmless zero-wind stub rather than a dangling reference into a
    // disposed system, so anything still holding `state` cannot throw.
    if (w.sample === this.sample) {
      w.sample = (x, z, t, out) => {
        const o = out || SHARED_OUT;
        o.x = 0; o.y = 0; o.z = 0;
        return o;
      };
    }
    // Identity-checked so a hot-reloaded replacement WindField that has already
    // installed itself is not torn down by the outgoing one's dispose().
    if (w.sample3 === this.sample3) delete w.sample3;
    if (w.getGustAt === this.getGustAt) delete w.getGustAt;
    if (w.shearAt === this.shearAt) delete w.shearAt;
    this._layers.length = 0;
    this._octaveCount = 0;
    this._ready = false;
  }

  // -------------------------------------------------------------------------
  // Per-frame
  // -------------------------------------------------------------------------

  update(dt, state) {
    const w = state.wind;
    const weather = state.weather;

    // --- weather -> wind character -----------------------------------------
    const a = PROFILES[weather.type] || FALLBACK_PROFILE;
    const b = PROFILES[weather.target] || a;
    const k = clamp01(weather.transition);
    const tStrength = lerp(a.strength, b.strength, k);
    const tGust = lerp(a.gustiness, b.gustiness, k);
    const tTurb = lerp(a.turbulence, b.turbulence, k);

    // Air has inertia: the mean flow lags the weather state by several seconds.
    // Turbulence and gustiness respond faster than bulk speed, which is what
    // makes a squall feel like it "arrives" ahead of the rain.
    w.strength = damp(w.strength, tStrength, 0.28, dt);
    w.gustiness = damp(w.gustiness, tGust, 0.55, dt);
    w.turbulence = damp(w.turbulence, tTurb, 0.55, dt);

    // --- heading ------------------------------------------------------------
    // Slow synoptic veer plus a wander whose amplitude and rate rise with
    // gustiness. A dead-steady heading is an instant tell.
    this._baseHeading += dt * 0.0015;
    this._headingPhase += dt * (0.012 + 0.030 * w.gustiness);
    const wander = this._dirNoise.fbm2D(this._headingPhase, 4.7, 3, 2.0, 0.5);
    const targetHeading = this._baseHeading + wander * (0.5 + 1.1 * w.gustiness);
    this._heading = dampAngle(this._heading, targetHeading, 0.6, dt);
    w.direction.x = Math.cos(this._heading);
    w.direction.y = Math.sin(this._heading);

    const ux = w.direction.x * w.strength;
    const uz = w.direction.y * w.strength;

    // --- octave advection + gains ------------------------------------------
    this._flutterMix = damp(this._flutterMix, this._flutterTarget, 3.0, dt);
    if (this._flutterMix < 0.0015) this._flutterMix = 0;
    else if (this._flutterMix > 0.9985) this._flutterMix = 1;

    this._driftAngle = mod(this._driftAngle + dt * 0.07, TAU);
    const dcx = Math.cos(this._driftAngle);
    const dcz = Math.sin(this._driftAngle);
    const turb = w.turbulence;

    for (let i = 0; i < this._layers.length; i++) {
      const L = this._layers[i];
      L.avx = ux * L.adv - uz * L.perp + dcx * L.drift;
      L.avz = uz * L.adv + ux * L.perp + dcz * L.drift;
      // Wrapped into one tile period: the field is periodic anyway, and this
      // keeps the offset from growing until float precision degrades after an
      // hour of play.
      L.ox = mod(L.ox + L.avx * dt, L.tile);
      L.oz = mod(L.oz + L.avz * dt, L.tile);
      let g = L.gain * (L.turbA + L.turbB * turb);
      if (i === FLUTTER_OCTAVE) g *= this._flutterMix;
      L.gainNow = g;
      L.wGainNow = g * L.wScale;
    }
    if (this._ready) {
      this._octaveCount = this._flutterMix > 0 ? OCTAVES.length : FLUTTER_OCTAVE;
    }

    // --- travelling gust bands ---------------------------------------------
    for (let i = 0; i < this._bands.length; i++) {
      const B = this._bands[i];
      const h = this._heading + B.angle;
      B.dx = Math.cos(h);
      B.dz = Math.sin(h);
      B.nx = -B.dz;
      B.nz = B.dx;
      // Bands are advected features, so they travel at roughly the mean wind.
      // minSpeed keeps them crawling even in fog, where U is under 1 m/s.
      B.rate = (w.strength * B.speedFactor + B.minSpeed) * B.invLen;
      B.travel = mod(B.travel + B.rate * dt, 1);
    }

    // --- discrete fronts ----------------------------------------------------
    // Wind runs before the player in main.js, so this is last frame's position.
    // A one-frame lag on "where should the next gust be aimed" is invisible.
    const px = state.player.position.x;
    const pz = state.player.position.z;
    this._updateFronts(dt, w, px, pz);

    // Everything above is integrated to the END of this frame, which is what
    // state.time.elapsed already reflects - so a caller passing `elapsed` gets
    // an exact zero time-delta correction inside sample()/getGustAt().
    this._time = state.time.elapsed;

    // --- published scalar envelope ------------------------------------------
    // Sampled from the real field at the listener so anything that just does
    // `strength * gust` stays in sync with anything that calls the field. The
    // damp only exists to absorb teleports/fast flight, not to smooth the field.
    const local = this._gustImpl(px, pz, this._time);
    w.gust = damp(w.gust, local, 12.0, dt);
  }

  _updateFronts(dt, w, px, pz) {
    // Spawn (Poisson). Rate is in "expected fronts per second"; the pool size
    // caps the density, and because front travel time scales as 1/strength a
    // storm naturally delivers them several times more often than a calm day.
    this._nextSpawn -= dt * (0.05 + 0.55 * w.gustiness);
    // Bounded: a zero-length inter-arrival draw is vanishingly unlikely but must
    // not be able to spin the loop.
    for (let guard = 0; guard < 4 && this._nextSpawn <= 0; guard++) {
      this._nextSpawn += -Math.log(1 - this._rand()) + 1e-4;
      // One slot is held back so a lightning downburst always has somewhere to
      // go - a strike that produces no gust is a wasted dramatic beat.
      if (this._frontCount < this._maxFronts - 1) this._spawnFront(w.strength, px, pz, 1);
    }
    if (this._nextSpawn <= 0) this._nextSpawn = 0.5;

    for (let i = this._frontCount - 1; i >= 0; i--) {
      const f = this._fronts[i];
      // Fronts are carried by the flow, so they track changes in mean speed.
      f.speed = damp(f.speed, w.strength * f.speedRatio + 0.8, 0.4, dt);
      f.pos += f.speed * dt;
      f.age += dt;
      const s = clamp01(f.age * (1 / f.fadeIn));
      f.amp = f.peak * s * s * (3 - 2 * s);

      const behind = f.pos - (px * f.dx + pz * f.dz);
      // Retire once fully past the listener, or once the listener has flown far
      // upwind of it. The age cap is a last-resort leak guard and is gated on
      // the front being out of sight - culling one mid-sweep would snap the
      // gust, which is precisely the discontinuity this system exists to avoid.
      const stale = f.age > 300 && (behind > f.cutoff || behind < -260);
      if (behind > f.cutoff + 90 || behind < -900 || stale) {
        this._frontCount--;
        const last = this._fronts[this._frontCount];
        this._fronts[this._frontCount] = f;
        this._fronts[i] = last;
      }
    }
  }

  _spawnFront(strength, px, pz, boost) {
    const r = this._rand;
    const f = this._fronts[this._frontCount++];

    const ang = this._heading + (r() - 0.5) * 0.5;
    f.dx = Math.cos(ang);
    f.dz = Math.sin(ang);

    f.speedRatio = 1.05 + 0.35 * r();
    f.speed = Math.max(1.2, strength * f.speedRatio + 0.8) * boost;

    f.rise = r.range(3, 11) / boost;      // sharper leading edge on a downburst
    f.invRise = 1 / f.rise;
    const fall = r.range(22, 85) * (0.75 + 0.4 * boost);
    f.invFall = 1 / fall;
    f.cutoff = f.rise + fall;

    // A downburst is not just a scaled ordinary gust - it is reliably strong,
    // so the whole draw range shifts up with `boost` rather than only stretching.
    const shift = 0.45 * (boost - 1);
    f.peak = r.range(0.30 + shift, 0.90 + shift) * boost;
    f.amp = 0;
    f.age = 0;
    f.fadeIn = 3.0;

    f.wobble = r.range(8, 26);
    f.latInv = 1 / r.range(70, 210);
    f.latPhase = r();

    // Born upwind and out of sight: the influence band lies entirely between
    // `pos - cutoff` and `pos`, i.e. at least 110 m beyond the grass horizon.
    f.pos = px * f.dx + pz * f.dz - r.range(110, 300);
  }

  _spawnDownburst() {
    const p = this._state.player.position;
    if (this._frontCount >= MAX_FRONTS) {
      // Pool physically full (only reachable on ULTRA). Steal the slot from
      // whichever front is contributing least here - either fully passed, or
      // still so far upwind that nothing on screen is touched by it.
      let victim = -1;
      let bestScore = 0;
      for (let i = 0; i < this._frontCount; i++) {
        const f = this._fronts[i];
        const behind = f.pos - (p.x * f.dx + p.z * f.dz);
        const score = behind > f.cutoff ? behind - f.cutoff : -behind - 200;
        if (score > bestScore) { bestScore = score; victim = i; }
      }
      if (victim < 0) return; // every front is currently visible; skip the beat
      this._frontCount--;
      const last = this._fronts[this._frontCount];
      this._fronts[this._frontCount] = this._fronts[victim];
      this._fronts[victim] = last;
    }
    this._spawnFront(this._state.wind.strength, p.x, p.z, 1.7);
    // Pull it in closer than a normal front so the gust lands a few seconds
    // after the flash rather than half a minute later.
    const f = this._fronts[this._frontCount - 1];
    f.pos += 60;
    f.fadeIn = 1.4;
  }

  // -------------------------------------------------------------------------
  // Field queries - hot path, zero allocation
  // -------------------------------------------------------------------------

  /**
   * World-space wind velocity at (x, z).
   * @param {number} x @param {number} z
   * @param {number} t world time in seconds (state.time.elapsed)
   * @param {{x:number,y:number,z:number}} [out] filled and returned; omitting it
   *        returns a SHARED scratch object that the next call overwrites.
   */
  _sampleImpl(x, z, t, out) {
    const o = out || SHARED_OUT;
    const w = this._state.wind;

    // Sub-frame extrapolation. Clamped so a caller passing a private clock can
    // never fling the field across the world.
    const dtc = clamp(t - this._time, -0.5, 0.5);

    const gust = this._gustImpl(x, z, t);
    const speed = w.strength * gust;
    // Turbulence intensity is a *ratio*, so scaling the fluctuations by `speed`
    // already makes a gust chop harder. Real gust fronts are additionally more
    // turbulent than the flow they displace, hence the mild super-proportional
    // term - the moment a front lands, the motion also gets rougher, not just
    // bigger, which is most of what sells it on grass and blossom.
    const chop = 0.75 + 0.25 * gust;

    let tx = 0;
    let tz = 0;
    let ty = 0;

    const layers = this._layers;
    const n = this._octaveCount;
    for (let li = 0; li < n; li++) {
      const L = layers[li];

      // Faded-bilinear fetch from this octave's wrapped table, written out
      // longhand so every intermediate stays a local.
      //
      // This used to live in a helper that returned its three components via
      // `this._lx/_lz/_lw`. That cost an allocation PER STORE: V8 no longer
      // keeps mutable heap numbers, so every write of a double to an object
      // field boxes a fresh HeapNumber. Nine stores a sample at 3000 petals is
      // ~430 KB of garbage per frame and a scavenge every few frames. Measured
      // in isolation the typed/local form is also 2.2x faster.
      const nn = L.n;
      const mask = nn - 1;
      const u = (x - (L.ox + dtc * L.avx)) * L.invCell;
      const v = (z - (L.oz + dtc * L.avz)) * L.invCell;
      const fi = Math.floor(u);
      const fj = Math.floor(v);
      let a = u - fi;
      let b = v - fj;
      // Smoothstep fade: six flops that remove the derivative discontinuity at
      // cell edges, which otherwise shows as faint grid-aligned banding in
      // anything that integrates the field (petal trails, most obviously).
      a = a * a * (3 - 2 * a);
      b = b * b * (3 - 2 * b);

      // Bitwise AND wraps negatives correctly (-1 & 63 === 63). World
      // coordinates stay far inside int32 range, so this is safe.
      const i0 = fi & mask;
      const j0 = fj & mask;
      const i1 = (i0 + 1) & mask;
      const j1 = (j0 + 1) & mask;

      const d = L.data;
      const r0 = j0 * nn;
      const r1 = j1 * nn;
      const p00 = (r0 + i0) * 3;
      const p10 = (r0 + i1) * 3;
      const p01 = (r1 + i0) * 3;
      const p11 = (r1 + i1) * 3;
      const g = L.gainNow;

      let e0 = d[p00] + (d[p10] - d[p00]) * a;
      let e1 = d[p01] + (d[p11] - d[p01]) * a;
      tx += (e0 + (e1 - e0) * b) * g;

      e0 = d[p00 + 1] + (d[p10 + 1] - d[p00 + 1]) * a;
      e1 = d[p01 + 1] + (d[p11 + 1] - d[p01 + 1]) * a;
      tz += (e0 + (e1 - e0) * b) * g;

      e0 = d[p00 + 2] + (d[p10 + 2] - d[p00 + 2]) * a;
      e1 = d[p01 + 2] + (d[p11 + 2] - d[p01 + 2]) * a;
      ty += (e0 + (e1 - e0) * b) * L.wGainNow;
    }

    o.x = (w.direction.x + tx * chop) * speed;
    o.y = ty * chop * speed;
    o.z = (w.direction.y + tz * chop) * speed;
    return o;
  }

  /**
   * Same as sample(), with the boundary-layer shear profile applied. Use this
   * for anything whose height matters - canopy branches, airborne petals, rain.
   */
  _sample3Impl(x, y, z, t, out) {
    const o = this._sampleImpl(x, z, t, out);
    const s = this._shearImpl(y);
    o.x *= s;
    o.z *= s;
    // Vertical eddies are far less height-dependent than the horizontal mean;
    // scaling w by the same factor would kill all lift near the ground.
    o.y *= 0.45 + 0.55 * s;
    return o;
  }

  /**
   * Log-law wind profile, normalised to 1.0 at 2 m: 0.63 at knee height, ~1.5 at
   * canopy height, saturating at 2.5 so cloud advection stays sane.
   *
   * It returns exactly 0 at y = 0 because that is the no-slip condition - pass a
   * grass blade's mid-height or tip, not the height of its root, or the blade
   * will sit perfectly still.
   */
  _shearImpl(y) {
    const yy = y > 0 ? y : 0;
    const s = Math.log((yy + ROUGHNESS_Z0) / ROUGHNESS_Z0) * INV_LOG_REF;
    return s > 2.5 ? 2.5 : s;
  }

  /**
   * Coherent gust multiplier, roughly 0..2 with a mean of 1.
   *
   * This is the function that has to be shared. Grass, tree sway and petal
   * spawn all read it, so a front sweeping the field bends the grass, then the
   * branches, then releases blossom - in that order, in the right places.
   */
  _gustImpl(x, z, t) {
    const dtc = clamp(t - this._time, -0.5, 0.5);
    // Table fetches are written out inline rather than through a helper: a
    // non-inlined call returning a double boxes the result, and this runs once
    // per petal per frame. Everything below stays in locals.
    const M = PROFILE_N - 1;
    const lat0 = this._lateral;
    let g = 0;

    // --- travelling bands ---------------------------------------------------
    const bands = this._bands;
    for (let i = 0; i < 2; i++) {
      const B = bands[i];

      // Bending the crests by a slow lateral profile is what stops the bands
      // reading as ruled lines. Real gust fronts arrive as ragged curves.
      const lp = ((x * B.nx + z * B.nz) * B.latInv + B.latPhase) * PROFILE_N;
      const lfi = Math.floor(lp);
      let la = lp - lfi;
      la = la * la * (3 - 2 * la);
      const li0 = lfi & M;
      const lv0 = lat0[li0];
      const wob = B.wobble * (lv0 + (lat0[(li0 + 1) & M] - lv0) * la);

      const prof = B.profile;
      const bp = ((x * B.dx + z * B.dz + wob) * B.invLen - (B.travel + dtc * B.rate)) * PROFILE_N;
      const bfi = Math.floor(bp);
      let ba = bp - bfi;
      ba = ba * ba * (3 - 2 * ba);
      const bi0 = bfi & M;
      const bv0 = prof[bi0];
      g += bv0 + (prof[(bi0 + 1) & M] - bv0) * ba;
    }
    g *= 0.7 * this._bandGain; // 0.7 restores unit RMS after summing two fields

    // --- discrete fronts ----------------------------------------------------
    const fronts = this._fronts;
    const n = this._frontCount;
    for (let i = 0; i < n; i++) {
      const f = fronts[i];
      // q = metres behind the leading edge. Reject on the un-bent coordinate
      // first (widened by the wobble bound) so distant fronts cost two mults
      // and a compare, not a table fetch.
      let q = f.pos + dtc * f.speed - (x * f.dx + z * f.dz);
      if (q <= -f.wobble || q >= f.cutoff + f.wobble) continue;

      const lp = ((z * f.dx - x * f.dz) * f.latInv + f.latPhase) * PROFILE_N;
      const lfi = Math.floor(lp);
      let la = lp - lfi;
      la = la * la * (3 - 2 * la);
      const li0 = lfi & M;
      const lv0 = lat0[li0];
      q -= f.wobble * (lv0 + (lat0[(li0 + 1) & M] - lv0) * la);
      if (q <= 0 || q >= f.cutoff) continue;
      let shape;
      if (q < f.rise) {
        const s = q * f.invRise;
        shape = s * s * (3 - 2 * s);        // steep, smooth onset
      } else {
        const s = 1 - (q - f.rise) * f.invFall;
        shape = s * s * s;                  // long cubic tail, exactly 0 at cutoff
      }
      g += f.amp * shape;
    }

    g = 1 + this._state.wind.gustiness * g;

    // Soft saturation instead of clamp(). Both branches are C1-continuous with
    // the identity at the seam (value and slope match at 0.5 and 1.5), so a
    // violent gust compresses smoothly toward the 0..2 contract instead of
    // flat-topping - hard clipping is visible as grass "sticking" at full bend.
    if (g > 1.5) return 2 - 0.25 / (g - 1);
    if (g < 0.5) return 0.25 / (1 - g);
    return g;
  }

}
