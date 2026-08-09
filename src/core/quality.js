/**
 * quality.js - quality tiers, adaptive resolution, and performance instrumentation.
 *
 * Owns `state.quality.*` and `state.perf.*` (see core/state.js).
 *
 * Three responsibilities, deliberately kept in one file because they share the
 * same per-frame timing data:
 *
 *   1. TIERS - four presets (low/medium/high/ultra) that write every quality
 *      field, plus a GPU sniff at startup so the first frame is already sane.
 *      Tier changes emit EVENTS.QUALITY_CHANGED; main.js fans that out to every
 *      system's onQualityChange(quality).
 *
 *   2. ADAPTIVE RESOLUTION - moves `state.quality.resolutionScale` inside
 *      [minResolutionScale, maxResolutionScale] to hold the tier's target
 *      framerate. Deliberately NOT emitted as QUALITY_CHANGED: a tier change
 *      makes systems rebuild geometry and render targets, and doing that in the
 *      middle of a framerate dip is how you turn a dip into a death spiral.
 *      Consumers of the render scale should either poll
 *      `state.quality.resolutionScale` each frame (cheap, one float compare) or
 *      subscribe via `quality.onResolutionChange(fn)`.
 *
 *   3. INSTRUMENTATION - fps / raw / smoothed frame time, renderer.info counters,
 *      and a `registerCost(label, ms)` hook so any system can publish its own
 *      cost for the HUD breakdown.
 *
 * Hot-path rule: beginFrame()/endFrame()/registerCost() allocate nothing after
 * the first call for a given label. Everything is hoisted.
 */

import { QUALITY, EVENTS } from './state.js';
import { RollingAverage, clamp, lerp } from './math.js';

// ---------------------------------------------------------------------------
// Tier presets
// ---------------------------------------------------------------------------

export const TIER_ORDER = Object.freeze([QUALITY.LOW, QUALITY.MEDIUM, QUALITY.HIGH, QUALITY.ULTRA]);

/**
 * Meta fields (label/targetFps/renderScale/minScale/pixelBudget) are consumed by
 * this manager. Everything else is copied verbatim into `state.quality`.
 *
 * Numbers are tuned for the project's reference GPU (AMD Radeon 780M, 1080p):
 * HIGH is the 16.6 ms budget target, LOW must hold 60 fps with room to spare,
 * ULTRA is allowed to sit around 30-40 fps on that machine.
 */
export const TIER_PRESETS = Object.freeze({
  [QUALITY.LOW]: Object.freeze({
    label: 'Low',
    targetFps: 60,
    renderScale: 0.8,
    minScale: 0.55,
    // Pixel count (post-scale) we expect this tier to afford on a mid iGPU.
    pixelBudget: 1.35e6,
    shadows: true,
    shadowMapSize: 1024,
    cascadeCount: 1,
    cloudSteps: 20,
    cloudLightSteps: 3,
    cloudResolution: 0.25,
    grassDensity: 0.35,
    grassDistance: 70,
    petalCount: 600,
    ao: false,
    godRays: false,
    depthOfField: false,
    motionBlur: false,
    antialias: 'fxaa',
  }),
  [QUALITY.MEDIUM]: Object.freeze({
    label: 'Medium',
    targetFps: 60,
    renderScale: 0.9,
    minScale: 0.6,
    pixelBudget: 1.75e6,
    shadows: true,
    shadowMapSize: 2048,
    cascadeCount: 2,
    cloudSteps: 32,
    cloudLightSteps: 4,
    cloudResolution: 0.5,
    grassDensity: 0.62,
    grassDistance: 105,
    petalCount: 1400,
    ao: false,
    godRays: true,
    depthOfField: false,
    motionBlur: false,
    antialias: 'smaa',
  }),
  [QUALITY.HIGH]: Object.freeze({
    label: 'High',
    targetFps: 60,
    renderScale: 1.0,
    minScale: 0.65,
    pixelBudget: 2.15e6,
    shadows: true,
    shadowMapSize: 2048,
    cascadeCount: 3,
    cloudSteps: 21,
    cloudLightSteps: 4,
    cloudResolution: 0.27,
    grassDensity: 1.0,
    grassDistance: 140,
    petalCount: 3000,
    ao: true,
    godRays: true,
    depthOfField: false,
    motionBlur: false,
    antialias: 'smaa',
  }),
  [QUALITY.ULTRA]: Object.freeze({
    label: 'Ultra',
    targetFps: 40,
    renderScale: 1.0,
    minScale: 0.8,
    pixelBudget: 2.8e6,
    shadows: true,
    shadowMapSize: 4096,
    cascadeCount: 4,
    cloudSteps: 52,
    cloudLightSteps: 6,
    cloudResolution: 0.55,
    grassDensity: 1.6,
    grassDistance: 210,
    petalCount: 8000,
    ao: true,
    godRays: true,
    depthOfField: true,
    motionBlur: true,
    antialias: 'smaa',
  }),
});

/** Fields copied preset -> state.quality. Kept explicit so nothing leaks. */
const APPLIED_KEYS = Object.freeze([
  'shadows', 'shadowMapSize', 'cascadeCount',
  'cloudSteps', 'cloudLightSteps', 'cloudResolution',
  'grassDensity', 'grassDistance', 'petalCount',
  'ao', 'godRays', 'depthOfField', 'motionBlur', 'antialias',
]);

/** Discrete render-scale ladder. Coarse enough to be stable, fine enough to hide. */
const SCALE_LADDER = Object.freeze([0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]);

/**
 * Plausible display periods in ms (144/120/100/90/75/60/50 Hz). The raw frame-time
 * floor is snapped onto this set so a slow drift can never invent a fake refresh
 * rate. Deliberately capped at 20 ms: a browser stuck at half rate on a 60 Hz panel
 * looks exactly like a 30 Hz panel, and the half-rate case is the one worth fixing.
 */
const REFRESH_PERIODS = Object.freeze([6.944, 8.333, 10.0, 11.111, 13.333, 16.667, 20.0]);

function snapToRefresh(ms) {
  let best = REFRESH_PERIODS[0];
  let bestErr = Infinity;
  for (let i = 0; i < REFRESH_PERIODS.length; i++) {
    const err = Math.abs(REFRESH_PERIODS[i] - ms);
    if (err < bestErr) { bestErr = err; best = REFRESH_PERIODS[i]; }
  }
  return best;
}

function normalizeTier(tier) {
  if (typeof tier !== 'string') return null;
  const t = tier.trim().toLowerCase();
  return TIER_ORDER.indexOf(t) >= 0 ? t : null;
}

/** Snap an arbitrary scale onto the ladder (nearest rung, ties round down). */
function ladderIndexFor(scale) {
  let best = 0;
  let bestErr = Infinity;
  for (let i = 0; i < SCALE_LADDER.length; i++) {
    const err = Math.abs(SCALE_LADDER[i] - scale);
    if (err < bestErr - 1e-6) { bestErr = err; best = i; }
  }
  return best;
}

// ---------------------------------------------------------------------------
// GPU detection
// ---------------------------------------------------------------------------

/**
 * Pull whatever identity strings the browser will give us.
 * WEBGL_debug_renderer_info is the good one; Chrome also returns a useful
 * (ANGLE-decorated) string from plain GL_RENDERER now, and Firefox with
 * privacy.resistFingerprinting returns nothing at all - hence the fallbacks.
 */
function readGPUStrings(renderer) {
  const out = { vendor: '', renderer: '', maxTexture: 0, maxSamples: 0, webgl2: false, hasContext: false };
  let gl = null;
  try {
    gl = renderer && typeof renderer.getContext === 'function' ? renderer.getContext() : null;
  } catch (err) {
    gl = null;
  }
  if (!gl) return out;
  out.hasContext = true;

  try {
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    if (ext) {
      out.renderer = String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) || '');
      out.vendor = String(gl.getParameter(ext.UNMASKED_VENDOR_WEBGL) || '');
    }
  } catch (err) {
    /* extension blocked by privacy settings - fall through */
  }
  try {
    if (!out.renderer) out.renderer = String(gl.getParameter(gl.RENDERER) || '');
    if (!out.vendor) out.vendor = String(gl.getParameter(gl.VENDOR) || '');
    out.maxTexture = gl.getParameter(gl.MAX_TEXTURE_SIZE) | 0;
    out.webgl2 = typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext;
    if (out.webgl2) out.maxSamples = gl.getParameter(gl.MAX_SAMPLES) | 0;
  } catch (err) {
    /* nothing else we can learn */
  }
  return out;
}

/** First integer following `re` in the string, or -1. */
function numAfter(str, re) {
  const m = str.match(re);
  if (!m) return -1;
  for (let i = 1; i < m.length; i++) {
    if (m[i] !== undefined && m[i] !== '') {
      const n = parseInt(m[i], 10);
      if (!Number.isNaN(n)) return n;
    }
  }
  return -1;
}

/**
 * Map a GPU description onto a starting tier.
 *
 * Rule from the art/perf brief: an INTEGRATED part (Radeon iGPU, Iris, UHD)
 * lands on MEDIUM or HIGH and never ULTRA - those chips have no VRAM and this
 * scene is fill-rate bound. Only a known-strong discrete part earns ULTRA.
 * Anything we cannot identify gets MEDIUM: adaptive resolution and the settings
 * panel can move from there, and a wrong guess upward is far more painful than
 * a wrong guess downward.
 */
export function classifyGPU(info) {
  // ANGLE decorates names with "(R)" / "(TM)" - strip them and collapse runs of
  // whitespace so "Intel(R) Iris(R) Xe Graphics" matches a plain "iris xe" probe.
  const s = `${info.renderer} ${info.vendor}`
    .toLowerCase()
    .replace(/\((?:r|tm|c)\)/g, ' ')
    .replace(/\s+/g, ' ');
  const mk = (tier, reason) => ({ tier, reason });

  if (!s.trim()) return mk(QUALITY.MEDIUM, 'gpu identity hidden');

  // --- Software rasterisers: nothing will save these ------------------------
  if (/swiftshader|llvmpipe|softpipe|basic render|microsoft basic|mesa offscreen|virgl|paravirtual/.test(s)) {
    return mk(QUALITY.LOW, 'software / virtualised renderer');
  }

  // --- Phones & tablets ------------------------------------------------------
  if (/adreno|mali[- ]|powervr|videocore|xclipse|immortalis/.test(s)) {
    // Recent flagship Adreno/Immortalis can take medium; everything else low.
    const adreno = numAfter(s, /adreno[^\d]*(\d{3})/);
    if (/immortalis/.test(s) || adreno >= 730) return mk(QUALITY.MEDIUM, 'recent mobile GPU');
    return mk(QUALITY.LOW, 'mobile GPU');
  }

  // --- Apple silicon ---------------------------------------------------------
  if (/apple m\d/.test(s)) {
    if (/\bm\d\s*(max|ultra)\b/.test(s)) return mk(QUALITY.ULTRA, 'Apple M-series Max/Ultra');
    return mk(QUALITY.HIGH, 'Apple silicon');
  }
  if (/apple gpu|apple a\d{1,2} gpu/.test(s)) return mk(QUALITY.MEDIUM, 'Apple mobile GPU');

  // --- NVIDIA ----------------------------------------------------------------
  if (/nvidia|geforce|quadro|rtx|gtx/.test(s)) {
    if (/\bmx\d{3}\b/.test(s)) return mk(QUALITY.MEDIUM, 'NVIDIA MX (entry mobile)');
    const rtx = numAfter(s, /rtx\s*(\d{4})/);
    if (rtx > 0) {
      const model = rtx % 1000;
      // A mobile x050/x060 is a 35-75 W part; this scene is fill-rate bound, so
      // do not hand it the ultra preset just because the name looks modern.
      const laptop = /laptop|max-q|mobile/.test(s);
      if (model >= 60 && !(laptop && model < 70)) return mk(QUALITY.ULTRA, 'NVIDIA RTX x060+');
      return mk(QUALITY.HIGH, laptop ? 'NVIDIA RTX mobile (entry)' : 'NVIDIA RTX x050');
    }
    const gtx = numAfter(s, /gtx\s*(\d{3,4})/);
    if (gtx > 0) {
      if (gtx >= 1060) return mk(QUALITY.HIGH, 'GeForce GTX 10/16-series');
      return mk(QUALITY.MEDIUM, 'older GeForce GTX');
    }
    return mk(QUALITY.HIGH, 'unrecognised NVIDIA part');
  }

  // --- AMD -------------------------------------------------------------------
  if (/amd|radeon|ati /.test(s)) {
    const rx = numAfter(s, /(?:radeon\s*)?rx\s*(\d{3,4})/);
    if (rx > 0) {
      const series = rx >= 1000 ? Math.floor(rx / 1000) : 0;
      if (series >= 6) return rx % 1000 >= 500 ? mk(QUALITY.ULTRA, 'Radeon RX 6000+') : mk(QUALITY.HIGH, 'entry Radeon RX 6000+');
      if (series === 5) return mk(QUALITY.HIGH, 'Radeon RX 5000');
      return mk(QUALITY.MEDIUM, 'older Radeon RX');
    }
    // RDNA2/3 integrated parts are named like "Radeon 780M Graphics".
    const igpu = numAfter(s, /\b(\d{3})m\b/);
    if (igpu > 0) return mk(QUALITY.HIGH, 'Radeon RDNA integrated');
    if (/vega/.test(s)) {
      const vega = numAfter(s, /vega\s*(\d{1,2})\b/);
      // "Vega 56/64" are discrete; "Vega 3..11" are the old integrated parts.
      if (vega >= 20) return mk(QUALITY.HIGH, 'Radeon Vega discrete');
      return mk(QUALITY.MEDIUM, 'Radeon Vega integrated');
    }
    if (/graphics/.test(s)) return mk(QUALITY.MEDIUM, 'Radeon integrated');
    return mk(QUALITY.MEDIUM, 'unrecognised AMD part');
  }

  // --- Intel -----------------------------------------------------------------
  if (/intel/.test(s)) {
    const arc = numAfter(s, /\barc\b[^\d]*a?(\d{3})/);
    if (arc > 0) return arc >= 580 ? mk(QUALITY.HIGH, 'Intel Arc') : mk(QUALITY.MEDIUM, 'entry Intel Arc');
    if (/iris xe|xe graphics|arc graphics/.test(s)) return mk(QUALITY.HIGH, 'Intel Xe integrated');
    if (/iris/.test(s)) return mk(QUALITY.MEDIUM, 'Intel Iris integrated');
    if (/uhd graphics/.test(s)) return mk(QUALITY.MEDIUM, 'Intel UHD integrated');
    if (/hd graphics/.test(s)) {
      // Skylake-era parts are 3-digit (HD 520/530/620); the 4-digit families
      // (HD 3000/4000/5500) are 2012-2015 silicon and cannot run this scene.
      const hd = numAfter(s, /hd graphics\s*(\d{3,4})/);
      return hd >= 500 && hd < 1000 ? mk(QUALITY.MEDIUM, 'Intel HD 500-series') : mk(QUALITY.LOW, 'legacy Intel HD');
    }
    return mk(QUALITY.MEDIUM, 'unrecognised Intel part');
  }

  return mk(QUALITY.MEDIUM, 'unknown GPU');
}

/** Coarse secondary signals, used only to nudge an unconfident guess. */
function systemPressure() {
  const cores = typeof navigator !== 'undefined' && navigator.hardwareConcurrency
    ? navigator.hardwareConcurrency : 0;
  const mem = typeof navigator !== 'undefined' && navigator.deviceMemory
    ? navigator.deviceMemory : 0;
  const mobileUA = typeof navigator !== 'undefined' &&
    (navigator.userAgentData?.mobile === true || /android|iphone|ipad|ipod|mobile/i.test(navigator.userAgent || ''));
  return { cores, mem, mobileUA };
}

// ---------------------------------------------------------------------------
// QualityManager
// ---------------------------------------------------------------------------

export class QualityManager {
  /**
   * Constructed by main.js *before* every other system, so the tier chosen here
   * is already in `state.quality` when those systems build their geometry.
   * Nothing is listening on the bus yet, so the initial apply does not emit.
   */
  constructor({ renderer, state, bus, canvas = null }) {
    this.renderer = renderer || null;
    this.state = state;
    this.bus = bus;
    this.canvas = canvas || renderer?.domElement || null;

    // -- Frame timing ---------------------------------------------------------
    // `avgFast` drives the adaptive controller (short window, reacts in ~0.3 s).
    // `avgSlow` drives the HUD readout (long window, no jitter).
    this._avgFast = new RollingAverage(20);
    this._avgSlow = new RollingAverage(90);
    this._avgCpu = new RollingAverage(45);
    this._frameStart = 0;
    this._lastStamp = -1;
    this._rafDelta = -1;
    this._frameIndex = 0;
    this._spikeStreak = 0;
    // Running estimate of the display's vsync period (raw, then snapped).
    this._floorMs = 16.667;
    this._vsyncMs = 16.667;

    // -- Adaptive resolution --------------------------------------------------
    this._warmup = 60;              // frames to ignore entirely at startup
    this._overStreak = 0;
    this._underStreak = 0;
    this._lastChangeAt = 0;
    this._cooldownMs = 900;
    this._upBackoff = 1;            // grows when an up-step immediately regresses
    this._lastStepWasUp = false;
    this._resListeners = new Set();
    this._ladderIndex = SCALE_LADDER.length - 1;
    this._minIndex = 0;
    this._maxIndex = SCALE_LADDER.length - 1;
    // Highest rung not yet proven too expensive. Learned instantly from a failed
    // up-step, forgotten slowly - this is what stops the controller pacing back
    // and forth across a cost boundary.
    this._learnedMaxIndex = SCALE_LADDER.length - 1;
    this._ceilingLearnedAt = 0;
    this._ceilingFailures = 0;
    this._lastSeenScale = -1;
    this._lastSeenAdaptive = null;
    this._adaptiveOverride = null;  // sticky user preference, survives tier changes
    this._viewW = 0;
    this._viewH = 0;
    this._lastPixels = 0;

    // -- Cost breakdown -------------------------------------------------------
    this._costMap = new Map();      // label -> record
    this._costList = [];            // same records, index-iterable (no iterator alloc)
    this._breakdown = [];           // reused public view for the HUD

    this._target = 1000 / 60;       // cached target frame time, refreshed on tier change

    // -- renderer.info --------------------------------------------------------
    // The post stack issues several renderer.render() calls per frame and three
    // resets info at the top of each one, so we take ownership of the reset.
    this._info = this.renderer?.info || null;
    this._infoAutoResetWas = this._info ? this._info.autoReset : null;
    if (this._info) this._info.autoReset = false;

    // -- Detect, then apply ---------------------------------------------------
    this.gpu = readGPUStrings(this.renderer);
    this.detected = this._detectTier();
    const forced = this._readTierOverride();
    const startTier = forced || this.detected.tier;

    this._apply(startTier, false);
    this._initPerfState();
    this._auditPresetCoverage();

    if (typeof console !== 'undefined' && console.info) {
      console.info(
        `[quality] ${this.gpu.renderer || 'unknown GPU'} -> ${startTier}` +
        `${forced ? ' (forced)' : ` (${this.detected.reason})`}`
      );
    }

    // -- Listeners ------------------------------------------------------------
    this._onResize = (e) => this._handleResize(e?.width | 0, e?.height | 0);
    this._offResize = this.bus?.on ? this.bus.on(EVENTS.RESIZE, this._onResize) : null;

    this._onVisibility = () => {
      if (typeof document !== 'undefined' && !document.hidden) this._resetTiming(60);
    };
    this._onContextLost = () => this._resetTiming(120);
    this._onContextRestored = () => this._resetTiming(120);

    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', this._onVisibility, false);
    }
    if (this.canvas) {
      this.canvas.addEventListener('webglcontextlost', this._onContextLost, false);
      this.canvas.addEventListener('webglcontextrestored', this._onContextRestored, false);
    }
  }

  // -------------------------------------------------------------------------
  // Public tier API (CONTRACTS §6)
  // -------------------------------------------------------------------------

  get tier() {
    return this.state.quality.tier;
  }

  get preset() {
    return TIER_PRESETS[this.state.quality.tier] || TIER_PRESETS[QUALITY.HIGH];
  }

  get resolutionScale() {
    return this.state.quality.resolutionScale;
  }

  /** Human-readable tier name for the settings panel. */
  get tierLabel() {
    return this.preset.label;
  }

  /**
   * Switch tier. Writes every field of state.quality, then emits
   * EVENTS.QUALITY_CHANGED with the state.quality object itself (main.js hands
   * that straight to every system's onQualityChange).
   * @returns {boolean} true if anything changed.
   */
  set(tier, opts) {
    const key = normalizeTier(tier);
    if (!key) {
      console.warn(`[quality] unknown tier "${tier}" - ignored`);
      return false;
    }
    // Tolerate set(tier, null) / set(tier, true) from UI code without throwing.
    const o = opts && typeof opts === 'object' ? opts : null;
    const force = o ? !!o.force : false;
    const persist = o ? o.persist !== false : true;
    if (!force && key === this.state.quality.tier) return false;
    this._apply(key, true);
    if (persist) this._persistTier(key);
    return true;
  }

  /** Step one tier up (+1) or down (-1). Returns the resulting tier. */
  cycle(dir = 1) {
    const i = TIER_ORDER.indexOf(this.state.quality.tier);
    const next = TIER_ORDER[clamp(i + Math.sign(dir), 0, TIER_ORDER.length - 1)];
    this.set(next);
    return this.state.quality.tier;
  }

  /** Enable/disable adaptive resolution. Sticky across tier changes. */
  setAdaptive(enabled) {
    const v = !!enabled;
    this._adaptiveOverride = v;
    this.state.quality.adaptive = v;
    this._lastSeenAdaptive = v;
    if (v) this._resetTiming(30);
    return v;
  }

  /**
   * Manually pin the render scale (HUD slider). Snaps to the ladder so the
   * adaptive controller and the user share one coordinate system.
   */
  setResolutionScale(scale) {
    const q = this.state.quality;
    const clamped = clamp(scale, SCALE_LADDER[this._minIndex], SCALE_LADDER[this._maxIndex]);
    this._ladderIndex = clamp(ladderIndexFor(clamped), this._minIndex, this._maxIndex);
    this._commitScale(SCALE_LADDER[this._ladderIndex]);
    // The player just told us what they want; discard our learned pessimism.
    this._learnedMaxIndex = this._maxIndex;
    this._ceilingFailures = 0;
    this._upBackoff = 1;
    this._resetTiming(30);
    return q.resolutionScale;
  }

  /**
   * Subscribe to render-scale changes. Preferred over polling for systems that
   * must reallocate render targets. Returns an unsubscribe function.
   */
  onResolutionChange(fn) {
    if (typeof fn !== 'function') return () => {};
    this._resListeners.add(fn);
    return () => this._resListeners.delete(fn);
  }

  // -------------------------------------------------------------------------
  // Per-frame instrumentation
  // -------------------------------------------------------------------------

  /** Called by main.js at the top of the frame with the rAF timestamp. */
  beginFrame(now) {
    const stamp = typeof now === 'number' && now > 0 ? now : performance.now();

    // rAF-to-rAF interval is the honest frame time: it includes vsync wait and
    // whatever the GPU was still chewing on. `now` is the same value main.js
    // hands to endFrame(), so we can never derive duration from the argument - 
    // we measure the CPU span ourselves below.
    this._rafDelta = this._lastStamp > 0 ? stamp - this._lastStamp : -1;
    this._lastStamp = stamp;
    this._frameStart = performance.now();
    this._frameIndex++;

    if (this._info) {
      // Re-assert every frame: the post stack calls renderer.render() several
      // times and three resets info at the top of each one, so a library pass
      // that flips autoReset back on would silently reduce our counters to
      // "whatever the last fullscreen triangle did".
      this._info.autoReset = false;
      this._info.reset();
    }

    // Mark all cost records unreported; endFrame() decays whatever stayed silent.
    const list = this._costList;
    for (let i = 0; i < list.length; i++) list[i].reported = false;
  }

  /** Called by main.js after the render. The `now` argument is intentionally unused. */
  endFrame() {
    const perf = this.state.perf;
    const cpuMs = performance.now() - this._frameStart;
    this._avgCpu.push(cpuMs);
    perf.cpuMs = this._avgCpu.average;

    if (this._rafDelta <= 0) {
      // First frame of the session, or the first after a reset. There is no
      // interval to measure yet, and the CPU span is NOT a frame time - feeding
      // it in drags the vsync-floor estimate below the real refresh period,
      // after which the controller thinks it can never do better and stops
      // restoring resolution forever. Publish the cheap counters and bail.
      this._readRendererInfo(perf);
      this._updateCosts();
      this._detectExternalEdits();
      if (this._warmup > 0) this._warmup--;
      return;
    }

    const frameMs = this._rafDelta;

    // --- Spike rejection ----------------------------------------------------
    // A GC pause, a shader compile or a stalled main thread must not move the
    // adaptive controller. But three bad frames in a row are not a spike, they
    // are a regime change: drop the accumulated history and start believing the
    // new numbers, rather than filtering out a real regression forever.
    const fastAvg = this._avgFast.count > 0 ? this._avgFast.average : frameMs;
    const isSpike = frameMs > Math.max(50, fastAvg * 2.5);
    let useForControl = true;
    if (isSpike) {
      this._spikeStreak++;
      if (this._spikeStreak >= 3) {
        this._avgFast.reset();
        this._spikeStreak = 0;
        this._overStreak = 0;
        this._underStreak = 0;
      } else {
        useForControl = false;
      }
    } else {
      this._spikeStreak = 0;
    }

    if (useForControl) this._avgFast.push(frameMs);
    // A tab-switch or an alert() produces a multi-second delta. Letting that into
    // the display average poisons the readout for the next 90 frames.
    if (frameMs < 250) this._avgSlow.push(frameMs);

    // --- Vsync floor estimate ------------------------------------------------
    // Falls fast toward any quick frame, creeps back up glacially, so it converges
    // on the FLOOR of observed frame times rather than their mean - that floor is
    // the display period, and "avg == floor" is how we know we are vsync-limited
    // rather than GPU-limited. Snapped to a real refresh period so the slow creep
    // can never drift into a value no monitor has.
    if (!isSpike) {
      this._floorMs = frameMs < this._floorMs
        ? lerp(this._floorMs, frameMs, 0.25)
        : lerp(this._floorMs, frameMs, 0.0006);
      this._floorMs = clamp(this._floorMs, 3.5, 40);
      this._vsyncMs = snapToRefresh(this._floorMs);
    }

    // --- Publish ------------------------------------------------------------
    const slow = this._avgSlow.average;
    perf.frameMs = frameMs;
    perf.avgFrameMs = slow;
    perf.fps = slow > 0.0001 ? 1000 / slow : 0;
    perf.vsyncMs = this._vsyncMs;
    // A frame whose CPU span already fills the budget will not be rescued by
    // fewer pixels; the HUD can surface this as "lower the tier, not the res".
    perf.cpuBound = perf.cpuMs > this._target * 0.85;

    this._readRendererInfo(perf);
    this._updateCosts();
    this._detectExternalEdits();

    if (this._warmup > 0) {
      this._warmup--;
      // Startup frames are dominated by shader compiles and texture uploads;
      // they say nothing about steady-state cost.
      this._overStreak = 0;
      this._underStreak = 0;
      return;
    }

    this._updateAdaptive(useForControl ? this._avgFast.average : fastAvg);
  }

  // -------------------------------------------------------------------------
  // Cost breakdown (CONTRACTS §6: quality.registerCost)
  // -------------------------------------------------------------------------

  /**
   * Publish a per-system cost in milliseconds. Call once per frame per label;
   * labels that stop reporting decay to zero and are eventually dropped.
   * Allocation-free after the first call for a given label.
   */
  registerCost(label, ms) {
    if (typeof label !== 'string' || typeof ms !== 'number' || !(ms >= 0) || !isFinite(ms)) return;
    let rec = this._costMap.get(label);
    if (rec === undefined) rec = this._createCost(label);
    rec.raw = ms;
    rec.reported = true;
    rec.silentFrames = 0;
  }

  /** Convenience timer pair: quality.beginCost('grass') ... quality.endCost('grass'). */
  beginCost(label) {
    if (typeof label !== 'string') return;
    let rec = this._costMap.get(label);
    if (rec === undefined) rec = this._createCost(label);
    rec.start = performance.now();
  }

  endCost(label) {
    if (typeof label !== 'string') return 0;
    const rec = this._costMap.get(label);
    if (rec === undefined || rec.start === 0) return 0;
    const ms = performance.now() - rec.start;
    rec.start = 0;
    rec.raw = ms;
    rec.reported = true;
    rec.silentFrames = 0;
    return ms;
  }

  /**
   * Sorted view of the cost table for the HUD: [{label, ms, peak, share}, ...],
   * heaviest first. Reuses its objects - do not hold references across calls.
   * Not a hot path (the HUD refreshes a few times a second), but still no-alloc
   * once the label set is stable.
   */
  getCostBreakdown() {
    const out = this._breakdown;
    const list = this._costList;
    while (out.length < list.length) out.push({ label: '', ms: 0, peak: 0, share: 0 });
    out.length = list.length;
    const frame = Math.max(this.state.perf.avgFrameMs, 0.001);
    for (let i = 0; i < list.length; i++) {
      const rec = list[i];
      const slot = out[i];
      slot.label = rec.label;
      slot.ms = rec.ms;
      slot.peak = rec.peak;
      slot.share = rec.ms / frame;
    }
    out.sort(descendingMs);
    return out;
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------

  /** renderer.info -> state.perf. Counters accumulated since our beginFrame(). */
  _readRendererInfo(perf) {
    const info = this._info;
    if (!info) return;
    const r = info.render;
    perf.drawCalls = r.calls;
    perf.triangles = r.triangles;
    perf.points = r.points;
    perf.lines = r.lines;
    const m = info.memory;
    if (m) {
      perf.geometries = m.geometries;
      perf.textures = m.textures;
    }
    perf.programs = info.programs ? info.programs.length : 0;
  }

  _createCost(label) {
    const rec = {
      label,
      raw: 0,
      ms: 0,
      peak: 0,
      start: 0,
      reported: false,
      silentFrames: 0,
      avg: new RollingAverage(30),
    };
    this._costMap.set(label, rec);
    this._costList.push(rec);
    this.state.perf.costs[label] = 0;
    return rec;
  }

  _updateCosts() {
    const list = this._costList;
    const perf = this.state.perf;
    const costs = perf.costs;
    let accounted = 0;
    for (let i = list.length - 1; i >= 0; i--) {
      const rec = list[i];
      if (rec.reported) {
        rec.ms = rec.avg.push(rec.raw);
        // Peak decays ~1%/frame so a one-off spike fades instead of sticking.
        rec.peak = rec.raw > rec.peak ? rec.raw : rec.peak * 0.99;
      } else {
        rec.silentFrames++;
        rec.ms = rec.avg.push(0);
        rec.peak *= 0.99;
        if (rec.silentFrames > 240) {
          // The system stopped reporting (disposed, or disabled by the tier).
          this._costMap.delete(rec.label);
          list.splice(i, 1);
          delete costs[rec.label];
          continue;
        }
      }
      costs[rec.label] = rec.ms;
      accounted += rec.ms;
    }
    // Lets the HUD show an honest "other" slice instead of implying the labelled
    // systems account for the whole frame.
    perf.accountedMs = accounted;
    perf.unaccountedMs = Math.max(0, perf.cpuMs - accounted);
  }

  _initPerfState() {
    const perf = this.state.perf;
    if (!perf.costs) perf.costs = {};
    perf.cpuMs = 0;
    perf.vsyncMs = this._vsyncMs;
    perf.targetMs = this._target;
    perf.accountedMs = 0;
    perf.unaccountedMs = 0;
    perf.cpuBound = false;
    perf.points = 0;
    perf.lines = 0;
    perf.geometries = 0;
    perf.textures = 0;
    perf.programs = 0;
    perf.gpu = this.gpu.renderer || 'unknown';
    perf.tier = this.state.quality.tier;
  }

  /** The frame time we are trying to hold, in ms. Cached - read every frame. */
  get _targetMs() {
    return this._target;
  }

  /**
   * Contract drift alarm: state.quality is shared with 19 other modules. If a
   * field is added there and no preset writes it, the tiers silently stop being
   * complete and LOW quietly keeps a HIGH setting. One-time dev check.
   */
  _auditPresetCoverage() {
    const known = new Set(APPLIED_KEYS);
    known.add('tier');
    known.add('resolutionScale');
    known.add('adaptive');
    known.add('minResolutionScale');
    known.add('maxResolutionScale');
    known.add('targetFps');
    const missing = [];
    for (const key of Object.keys(this.state.quality)) {
      if (!known.has(key)) missing.push(key);
    }
    if (missing.length) {
      console.warn('[quality] state.quality fields no tier preset writes:', missing.join(', '));
    }
  }

  _detectTier() {
    const base = classifyGPU(this.gpu);
    const sys = systemPressure();
    let idx = TIER_ORDER.indexOf(base.tier);
    let reason = base.reason;

    // Secondary signals only nudge, and only in the safe direction, except for
    // the one confident upgrade case (many cores + plenty of RAM + an unknown
    // GPU almost always means a desktop, not a netbook).
    if (sys.mobileUA && idx > 1) {
      idx = 1;
      reason += ' (mobile UA)';
    }
    if (sys.cores > 0 && sys.cores <= 4 && idx > 1) {
      idx = 1;
      reason += ' (few CPU cores)';
    }
    if (sys.mem > 0 && sys.mem <= 4 && idx > 0) {
      idx = Math.min(idx, 1);
      reason += ' (low system memory)';
    }
    // A GL_MAX_TEXTURE_SIZE below 8192 means an old or heavily constrained part.
    if (this.gpu.maxTexture > 0 && this.gpu.maxTexture < 8192 && idx > 0) {
      idx -= 1;
      reason += ' (small max texture)';
    }
    // Only trust this when we actually got a context - "we could not ask" must
    // never be read as "the answer is no".
    if (this.gpu.hasContext && !this.gpu.webgl2 && idx > 0) {
      idx = 0;
      reason += ' (no WebGL2)';
    }

    return { tier: TIER_ORDER[clamp(idx, 0, TIER_ORDER.length - 1)], reason, gpu: this.gpu.renderer };
  }

  /** `?quality=low` / `?q=ultra` beats a stored preference beats detection. */
  _readTierOverride() {
    let forced = null;
    try {
      if (typeof location !== 'undefined' && location.search) {
        const params = new URLSearchParams(location.search);
        forced = normalizeTier(params.get('quality') || params.get('q') || '');
        const adaptiveParam = params.get('adaptive');
        if (adaptiveParam !== null) this._adaptiveOverride = adaptiveParam !== '0' && adaptiveParam !== 'false';
      }
    } catch (err) {
      /* exotic environment without URL parsing - ignore */
    }
    if (forced) return forced;
    try {
      if (typeof localStorage !== 'undefined') {
        return normalizeTier(localStorage.getItem('sakura.quality') || '');
      }
    } catch (err) {
      /* storage blocked (private mode / file://) - detection stands */
    }
    return null;
  }

  _persistTier(tier) {
    try {
      if (typeof localStorage !== 'undefined') localStorage.setItem('sakura.quality', tier);
    } catch (err) {
      /* storage blocked - the tier still applies for this session */
    }
  }

  /** Write a whole preset into state.quality and (optionally) announce it. */
  _apply(tierKey, emit) {
    const preset = TIER_PRESETS[tierKey];
    const q = this.state.quality;

    q.tier = tierKey;
    for (let i = 0; i < APPLIED_KEYS.length; i++) {
      const k = APPLIED_KEYS[i];
      q[k] = preset[k];
    }

    // Adaptive is a user preference, not a tier property - a tier change must
    // not silently re-enable something the player turned off.
    q.adaptive = this._adaptiveOverride === null ? true : this._adaptiveOverride;
    this._lastSeenAdaptive = q.adaptive;

    // Resolution bounds come from the tier; the ladder indices are derived so
    // the controller can only ever produce a legal value.
    q.minResolutionScale = preset.minScale;
    q.maxResolutionScale = preset.renderScale;
    q.targetFps = preset.targetFps;
    this._target = 1000 / (preset.targetFps || 60);
    this._minIndex = ladderIndexFor(preset.minScale);
    this._maxIndex = Math.max(this._minIndex, ladderIndexFor(preset.renderScale));
    // A different tier is a different cost curve; nothing we learned still holds.
    this._learnedMaxIndex = this._maxIndex;
    this._ceilingLearnedAt = 0;
    this._ceilingFailures = 0;

    // Start where the screen's pixel count says we can afford to, not blindly at
    // the ceiling: on a 4K panel the ceiling is four times the reference cost and
    // the first second would be a slideshow. Adaptive climbs back if there is room.
    const start = this._startIndexFor(preset);
    this._ladderIndex = clamp(start, this._minIndex, this._maxIndex);
    this._commitScale(SCALE_LADDER[this._ladderIndex]);

    this.state.perf.tier = tierKey;
    this.state.perf.targetMs = this._target;

    // A tier change reallocates shadow maps, rebuilds grass, recompiles shaders.
    // Timing is meaningless until that settles.
    this._resetTiming(45);
    this._lastChangeAt = this._lastStamp > 0 ? this._lastStamp : 0;
    this._upBackoff = 1;
    this._lastStepWasUp = false;

    if (emit && this.bus) this.bus.emit(EVENTS.QUALITY_CHANGED, this.state.quality);
  }

  /** Ladder index whose pixel count is closest to (but not above) the tier budget. */
  _startIndexFor(preset) {
    const w = this._viewW || (typeof window !== 'undefined' ? window.innerWidth : 1920);
    const h = this._viewH || (typeof window !== 'undefined' ? window.innerHeight : 1080);
    const dpr = typeof window !== 'undefined' ? Math.min(window.devicePixelRatio || 1, 2) : 1;
    const pixels = Math.max(1, w * h * dpr * dpr);
    this._lastPixels = pixels;
    if (pixels <= preset.pixelBudget) return this._maxIndex;
    // Cost scales with pixel count, which scales with scale^2.
    const wanted = Math.sqrt(preset.pixelBudget / pixels);
    let idx = this._maxIndex;
    while (idx > this._minIndex && SCALE_LADDER[idx] > wanted) idx--;
    return idx;
  }

  _commitScale(scale) {
    const q = this.state.quality;
    if (Math.abs(q.resolutionScale - scale) < 1e-4) {
      this._lastSeenScale = q.resolutionScale;
      return false;
    }
    q.resolutionScale = scale;
    this._lastSeenScale = scale;
    for (const fn of this._resListeners) {
      try {
        fn(scale, q);
      } catch (err) {
        console.error('[quality] resolution listener threw:', err);
      }
    }
    return true;
  }

  /**
   * The HUD writes state.quality.adaptive / resolutionScale directly (it is a
   * settings panel, not a system). Notice that and adopt it instead of fighting.
   */
  _detectExternalEdits() {
    const q = this.state.quality;
    if (q.adaptive !== this._lastSeenAdaptive) {
      this._adaptiveOverride = !!q.adaptive;
      this._lastSeenAdaptive = q.adaptive;
      this._resetTiming(30);
    }
    if (Math.abs(q.resolutionScale - this._lastSeenScale) > 1e-4) {
      const clamped = clamp(q.resolutionScale, SCALE_LADDER[this._minIndex], SCALE_LADDER[this._maxIndex]);
      this._ladderIndex = clamp(ladderIndexFor(clamped), this._minIndex, this._maxIndex);
      q.resolutionScale = SCALE_LADDER[this._ladderIndex];
      this._lastSeenScale = q.resolutionScale;
      this._resetTiming(30);
    }
  }

  _handleResize(w, h) {
    if (w > 0) this._viewW = w;
    if (h > 0) this._viewH = h;
    // Reallocating render targets costs several frames of stalls.
    this._resetTiming(30);

    if (!this.state.quality.adaptive) return;
    const dpr = typeof window !== 'undefined' ? Math.min(window.devicePixelRatio || 1, 2) : 1;
    const pixels = Math.max(1, this._viewW * this._viewH * dpr * dpr);
    // Only ever step *down* on a resize; letting it step up would undo a
    // hard-won adaptive decision every time the window is nudged.
    if (pixels > this._lastPixels * 1.1) {
      const idx = this._startIndexFor(this.preset);
      if (idx < this._ladderIndex) {
        this._ladderIndex = idx;
        this._commitScale(SCALE_LADDER[idx]);
      }
    }
    this._lastPixels = pixels;
  }

  /** Forget accumulated timing after any event that invalidates it. */
  _resetTiming(warmupFrames) {
    this._warmup = Math.max(this._warmup, warmupFrames | 0);
    this._avgFast.reset();
    this._overStreak = 0;
    this._underStreak = 0;
    this._spikeStreak = 0;
    this._lastStamp = -1;
    this._rafDelta = -1;
  }

  // -------------------------------------------------------------------------
  // Adaptive resolution controller
  // -------------------------------------------------------------------------

  /**
   * Hysteresis model:
   *   - Two thresholds with a dead band between them, so a frame time sitting
   *     near the target cannot toggle anything.
   *   - Streak counters: a decision needs N *consecutive-ish* samples agreeing.
   *     Down is confirmed fast (~0.2 s), up slowly (~1.5 s and rising) because
   *     the cost of a wrong up-step is a visible stutter and the cost of a wrong
   *     down-step is a slightly softer frame.
   *   - Cooldown after every change, longer after an up-step.
   *   - Learned ceiling + exponential backoff: if an up-step is followed by a
   *     down-step we sit on a cost boundary, so that rung is marked too expensive
   *     and the next attempt needs proportionally longer stability. Without this
   *     the controller paces across the boundary forever - the classic visible
   *     "breathing" that makes dynamic resolution obvious and hateful.
   *   - CPU veto: if our own JS span already exceeds the whole budget, pixels
   *     are not the bottleneck and cutting them only costs sharpness.
   *
   * What this controller deliberately does NOT do is judge a step by whether the
   * frame time improved. Under vsync every frame is quantised to a multiple of
   * the refresh period, so a step from 32 ms to 17 ms of real GPU work shows up
   * as no change at all (33.3 ms both times) - "did that help?" is unanswerable
   * without a GPU timer query, and acting on the answer produces exactly the
   * oscillation the rest of this design exists to prevent.
   */
  _updateAdaptive(avg) {
    const q = this.state.quality;
    if (!q.adaptive) return;
    if (typeof document !== 'undefined' && document.hidden) return;
    if (this._avgFast.count < 12) return;

    const now = this._lastStamp;
    const target = this._targetMs;

    // Up cannot be judged against the target alone: on a vsync-locked display a
    // healthy frame reads exactly 16.7 ms and would never look "fast enough".
    // The real signal is "we are pinned to the display's own floor".
    const upThreshold = Math.max(target * 0.85, this._vsyncMs * 1.045);
    const downThreshold = target * 1.12;
    // Sitting on the refresh period means the display is the limit, not us.
    // Cutting pixels here would only make a perfect frame blurrier.
    const displayLimited = avg <= this._vsyncMs * 1.06;

    if (avg > downThreshold) {
      this._overStreak++;
      this._underStreak = 0;
    } else if (avg < upThreshold) {
      this._underStreak++;
      this._overStreak = 0;
    } else {
      // Dead band: bleed the counters instead of resetting, so a noisy signal
      // that mostly sits over budget still eventually acts.
      if (this._overStreak > 0) this._overStreak--;
      if (this._underStreak > 0) this._underStreak--;
    }

    if (now - this._lastChangeAt < this._cooldownMs) return;

    // --- Down ---------------------------------------------------------------
    // 12 samples ≈ 0.2 s at 60 fps, ≈ 0.4 s at 30 fps - fast enough to feel
    // responsive, slow enough that a two-frame hiccup never triggers it.
    if (this._overStreak >= 12 && this._ladderIndex > this._minIndex) {
      if (displayLimited) return;
      // Fewer pixels cannot fix a frame whose JS span alone already exceeds the
      // whole budget - that is a direct measurement, unlike "did the last step
      // help?", which vsync quantisation makes unanswerable (a 17 ms frame and a
      // 32 ms frame both read as exactly 33.3 ms, so real progress looks like
      // none). Threshold is the full budget, not a fraction of it: when the CPU
      // costs 15 ms of a 16.6 ms frame, trimming GPU work still helps.
      if (this.state.perf.cpuMs > target) return;
      // Single rungs by default, for the same quantisation reason: a
      // "proportional" step is guesswork that overshoots, and overshooting costs
      // a visible bounce back up - worse than converging a second slower. Two
      // rungs only past 2.5x target, where we have missed at least three
      // refreshes and the overrun is unambiguous.
      const stepSize = avg > target * 2.5 ? 2 : 1;
      this._stepTo(Math.max(this._ladderIndex - stepSize, this._minIndex), now, false);
      return;
    }

    // --- Up -----------------------------------------------------------------
    // A rung that failed once is not forbidden forever - the scene's cost swings
    // with time of day and weather, so hand one rung back after a long stretch of
    // healthy frames and let the controller re-probe it.
    //
    // Under vsync a healthy frame reads exactly the refresh period whether we
    // have 1 ms or 8 ms of headroom, so there is no way to know the workload got
    // cheaper except by trying. Each failed probe therefore pushes the next one
    // further out (30 s -> 60 s -> 90 s): a scene that genuinely sits on a
    // boundary stops probing, while one that really did get cheaper is found
    // within a minute and a half and resets the interval.
    // An up-step that survived its whole cooldown while staying under budget
    // proves the learned ceiling wrong - the workload really did get cheaper
    // (a storm passed, the sun set, the player walked out of the grass). Drop
    // the ceiling entirely so the normal up-streak can climb rung by rung
    // instead of crawling back one rung per relax interval.
    if (this._lastStepWasUp &&
        this._underStreak > 0 &&
        now - this._lastChangeAt >= this._cooldownMs) {
      this._lastStepWasUp = false;
      this._learnedMaxIndex = this._maxIndex;
      this._ceilingFailures = 0;
    }

    const relaxMs = 30000 * Math.min(Math.max(this._ceilingFailures, 1), 3);
    if (this._learnedMaxIndex < this._maxIndex &&
        this._underStreak > 0 &&
        now - this._ceilingLearnedAt > relaxMs) {
      this._learnedMaxIndex++;
      this._ceilingLearnedAt = now;
    }

    const upNeeded = Math.min(90 * this._upBackoff, 600);
    const upCeiling = Math.min(this._maxIndex, this._learnedMaxIndex);
    if (this._underStreak >= upNeeded && this._ladderIndex < upCeiling) {
      this._stepTo(this._ladderIndex + 1, now, true);
    }
  }

  _stepTo(index, now, isUp) {
    if (index === this._ladderIndex) return;
    // An up-step immediately undone by a down-step means we sit right on a cost
    // boundary. Record the rung that failed so we stop climbing into it, and make
    // each subsequent attempt require longer proof.
    if (!isUp && this._lastStepWasUp) {
      this._upBackoff = Math.min(this._upBackoff * 2, 8);
      this._learnedMaxIndex = clamp(this._ladderIndex - 1, this._minIndex, this._maxIndex);
      this._ceilingLearnedAt = now;
      this._ceilingFailures++;
    } else if (isUp) {
      this._upBackoff = Math.max(1, this._upBackoff * 0.9);
    }

    this._ladderIndex = index;
    this._commitScale(SCALE_LADDER[index]);
    this._lastChangeAt = now;
    this._lastStepWasUp = isUp;
    // Recovering resolution is optional, so be lazy about it; relieving pressure
    // is urgent, so re-evaluate sooner.
    this._cooldownMs = isUp ? 1600 : 600;
    this._overStreak = 0;
    this._underStreak = 0;
    // The next few frames pay for target reallocation - do not judge them.
    this._avgFast.reset();
  }

  // -------------------------------------------------------------------------

  dispose() {
    this._offResize?.();
    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', this._onVisibility, false);
    }
    if (this.canvas) {
      this.canvas.removeEventListener('webglcontextlost', this._onContextLost, false);
      this.canvas.removeEventListener('webglcontextrestored', this._onContextRestored, false);
    }
    if (this._info && this._infoAutoResetWas !== null) this._info.autoReset = this._infoAutoResetWas;
    this._resListeners.clear();
    this._costMap.clear();
    this._costList.length = 0;
    this._breakdown.length = 0;
  }
}

/** Hoisted comparator - a closure here would allocate on every HUD refresh. */
function descendingMs(a, b) {
  return b.ms - a.ms;
}
