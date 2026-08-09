/**
 * daynight.js - the clock and the lighting curves of Sakura Realm.
 *
 * Two responsibilities, nothing else:
 *
 *  1. THE CLOCK. Owns `state.time.timeOfDay` (0..24). Advances it from
 *     `state.time.daySpeed`, honours `state.time.paused`, and supports both
 *     instant scrubbing (HUD slider) and short eased transitions between named
 *     presets. Nobody else may write `timeOfDay`.
 *
 *  2. THE LIGHTING CURVES. A set of keyframed tracks over the 24h cycle giving
 *     sun colour temperature, sun intensity, moon intensity/colour, ambient
 *     intensity and a target camera exposure - published on `this.lighting`
 *     for the engine / celestial / atmosphere / post to consume.
 *
 * Design notes that matter:
 *
 *  - Keys are NOT authored at absolute clock hours. They are authored as offsets
 *    from four solar anchors (sunrise, solar noon, sunset, solar midnight) which
 *    are themselves derived from a real solar-position model (latitude +
 *    declination). Change the latitude or the day of year and every keyframe,
 *    every preset and every twilight window slides with the sun instead of
 *    silently desynchronising. Default location is Kyoto in early April - the
 *    sun this scene was art-directed under.
 *
 *  - Interpolation is monotone cubic Hermite (Fritsch-Carlson), cyclic over 24h.
 *    Catmull-Rom was the obvious choice and is wrong here: golden-hour and
 *    blue-hour keys sit 0.2h apart between keys 3h apart, and Catmull-Rom rings
 *    badly across that spacing - negative sun intensity just after sunset,
 *    colour-temperature overshoot into magenta. Monotone Hermite is C1, never
 *    overshoots a key, and flattens naturally at extrema (noon peak, night
 *    plateau). That is exactly the behaviour these curves need.
 *
 *  - Colour is interpolated in MIRED (10^6 / K), not in Kelvin and definitely
 *    not in RGB. Mired is the space colour temperature is perceptually even in;
 *    a straight Kelvin lerp from 1800K to 5900K spends most of its travel in a
 *    range the eye reads as identical, then snaps.
 *
 *  - Auto-exposure adapts in LOG2 space with asymmetric rates (the eye and every
 *    real camera stop down faster than they open up) and a hard stops-per-second
 *    slew limit, so a cut from noon to midnight is always a ramp, never a jump.
 */

import { Color, LinearSRGBColorSpace } from 'three';
import {
  clamp,
  clamp01,
  lerp,
  mod,
  smoothstep,
  damp,
  kelvinToRGB,
  Easing,
  DEG2RAD,
  RAD2DEG,
} from '../core/math.js';

// ---------------------------------------------------------------------------
// Module-scope scratch. Nothing in update() allocates.
// ---------------------------------------------------------------------------

const _rgb = { r: 1, g: 1, b: 1 };
const _keyScratch = [];

const HOURS = 24;
/** Earth rotates 15 degrees of hour angle per hour. */
const DEG_PER_HOUR = 15;
/**
 * Standard sunrise/sunset altitude: -0.833 deg accounts for the solar radius
 * (0.267) plus mean atmospheric refraction at the horizon (0.566).
 */
const HORIZON_ALTITUDE = -0.833 * DEG2RAD;
const AXIAL_TILT = 23.44 * DEG2RAD;

/**
 * How much of the luminance a warm colour loses gets handed back. Blackbody RGB
 * at 1800K carries roughly a third the luminance of 5900K; if the light colour
 * keeps all of that loss the intensity curve is dimming twice. 0 = keep the full
 * physical dimming, 1 = fully decouple chroma from brightness. 0.35 leaves a
 * believable residual dip at the horizon.
 */
const LUMA_COMPENSATION = 0.35;
const LUMA_COMPENSATION_MAX = 1.6;

/** Moon colour is desaturated toward neutral more aggressively than the sun. */
const MOON_SATURATION = 0.6;
/**
 * Purkinje shift: at scotopic levels rod vision peaks ~507nm, so night reads
 * blue-silver even though moonlight is physically ~4100K reflected sunlight.
 * Gated by moon colour temperature - a moon sitting on the horizon is genuinely
 * amber, and pushing that toward blue at the same time just makes mud.
 */
const PURKINJE_COLOR = { r: 0.62, g: 0.78, b: 1.0 };
const PURKINJE_STRENGTH = 0.62;
const PURKINJE_KELVIN_LO = 3000;
const PURKINJE_KELVIN_HI = 5500;

/**
 * Exposure adaptation. Light adaptation (scene got brighter, exposure must come
 * down) is fast; dark adaptation is slow. Values are exponential-decay lambdas
 * in log2 space, plus an absolute slew ceiling in stops/second.
 */
const ADAPT_LAMBDA_BRIGHTEN = 1.9;
const ADAPT_LAMBDA_DARKEN = 0.62;
const ADAPT_MAX_STOPS_PER_SEC = 1.35;
/** adapt:'fast' multiplier and how quickly it relaxes back to the normal rate. */
const ADAPT_FAST_BOOST = 14;
const ADAPT_BOOST_DECAY = 2.0;

/** sRGB transfer function inverse. Local so this file depends on no three API. */
const srgbToLinear = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));

// ---------------------------------------------------------------------------
// Keyframe tables
//
// Every key is [anchor, offsetHours, value].
//   'R' sunrise · 'N' solar noon · 'S' sunset · 'M' solar midnight
//
// Morning and evening are deliberately NOT mirror images. Afternoon air carries
// more haze and heat shimmer than dawn air, so the evening side runs a little
// warmer, a little dimmer and a little longer. Perfect symmetry is one of the
// loudest tells that a day/night cycle was generated rather than observed.
// ---------------------------------------------------------------------------

/** three.js DirectionalLight intensity for the sun key light. */
const SUN_INTENSITY_KEYS = [
  ['M', -3.00, 0.000],
  ['M', 0.00, 0.000],
  ['M', 3.00, 0.000],
  ['R', -0.30, 0.000],
  ['R', 0.00, 0.050],
  ['R', 0.18, 0.300],
  ['R', 0.45, 0.850],
  ['R', 0.85, 1.550],
  ['R', 1.50, 2.250],
  ['R', 2.80, 2.950],
  ['N', -1.50, 3.300],
  ['N', 0.00, 3.450],
  ['N', 1.50, 3.320],
  ['S', -2.80, 2.880],
  ['S', -1.50, 2.180],
  ['S', -0.85, 1.480],
  ['S', -0.45, 0.800],
  ['S', -0.18, 0.280],
  ['S', 0.00, 0.045],
  ['S', 0.30, 0.000],
];

/** Sun colour temperature in Kelvin. Interpolated in mired - see notes above. */
const SUN_KELVIN_KEYS = [
  // Night value only exists to keep the curve from bulging between the two
  // horizon keys; the sun contributes nothing here.
  ['M', 0.00, 1650],
  ['R', -0.50, 1700],
  ['R', 0.00, 1780],
  ['R', 0.20, 2180],
  ['R', 0.50, 2760],
  ['R', 1.00, 3520],
  ['R', 1.80, 4400],
  ['R', 3.20, 5300],
  ['N', -1.20, 5750],
  ['N', 0.00, 5900],
  ['N', 1.20, 5700],
  ['S', -3.20, 5100],
  ['S', -1.80, 4200],
  ['S', -1.00, 3300],
  ['S', -0.50, 2580],
  ['S', -0.20, 2060],
  ['S', 0.00, 1680],
  ['S', 0.50, 1600],
];

/**
 * Chroma strength: 1 keeps the raw blackbody, 0 collapses to white. The
 * blackbody curve below ~2200K is a fully saturated orange that reads as a
 * lava lamp on a landscape. Pulling saturation down at the horizon - where the
 * blackbody is most extreme - gives the soft amber the art direction asks for
 * while keeping midday honest.
 */
const SUN_SATURATION_KEYS = [
  ['M', 0.00, 0.70],
  ['R', -0.50, 0.70],
  ['R', 0.00, 0.72],
  ['R', 0.60, 0.78],
  ['R', 1.60, 0.84],
  ['N', 0.00, 0.90],
  ['S', -1.60, 0.83],
  ['S', -0.60, 0.76],
  ['S', 0.00, 0.70],
  ['S', 0.50, 0.68],
];

/** Moon key-light intensity at full moon, before phase weighting. */
const MOON_INTENSITY_KEYS = [
  ['M', -2.50, 0.145],
  ['M', 0.00, 0.165],
  ['M', 2.50, 0.145],
  ['R', -1.80, 0.100],
  ['R', -0.80, 0.035],
  ['R', -0.20, 0.000],
  ['N', 0.00, 0.000],
  ['S', 0.20, 0.000],
  ['S', 0.80, 0.035],
  ['S', 1.80, 0.100],
];

/** The moon reddens at the horizon exactly like the sun does, for the same reason. */
const MOON_KELVIN_KEYS = [
  ['M', -2.00, 6600],
  ['M', 0.00, 7000],
  ['M', 2.00, 6600],
  ['R', -2.40, 5200],
  ['R', -1.20, 3400],
  ['R', -0.40, 2500],
  ['N', 0.00, 4500],
  ['S', 0.40, 2500],
  ['S', 1.20, 3400],
  ['S', 2.40, 5200],
];

/** Sky/IBL ambient multiplier. Peaks at noon, carries the whole blue hour alone. */
const AMBIENT_INTENSITY_KEYS = [
  ['M', -3.00, 0.105],
  ['M', 0.00, 0.095],
  ['M', 3.00, 0.105],
  ['R', -1.30, 0.135],
  ['R', -0.75, 0.245],
  ['R', -0.35, 0.400],
  ['R', 0.00, 0.560],
  ['R', 0.60, 0.710],
  ['R', 1.80, 0.860],
  ['N', 0.00, 1.000],
  ['S', -1.80, 0.885],
  ['S', -0.60, 0.735],
  ['S', 0.00, 0.575],
  ['S', 0.35, 0.415],
  ['S', 0.75, 0.255],
  ['S', 1.35, 0.150],
];

/**
 * Target tone-mapping exposure. This is deliberately NOT a full compensation
 * curve: it recovers roughly 1.8 stops between noon and midnight while the
 * scene itself loses about 4.3, so night still reads as night.
 */
const EXPOSURE_KEYS = [
  ['M', -3.00, 2.70],
  ['M', 0.00, 2.75],
  ['M', 3.00, 2.70],
  ['R', -1.20, 2.45],
  ['R', -0.70, 2.05],
  ['R', -0.30, 1.70],
  ['R', 0.00, 1.42],
  ['R', 0.40, 1.20],
  ['R', 1.20, 1.00],
  ['R', 2.50, 0.88],
  ['N', 0.00, 0.80],
  ['S', -2.50, 0.87],
  ['S', -1.20, 0.98],
  ['S', -0.40, 1.18],
  ['S', 0.00, 1.40],
  ['S', 0.35, 1.72],
  ['S', 0.80, 2.10],
  ['S', 1.60, 2.50],
];

/**
 * Named moments, anchored the same way the curves are. `dawn` sits just below
 * the horizon because that is when the sky is lit and the land is not - the
 * single best-looking minute of the cycle.
 */
const PRESET_SPECS = {
  // Sun ~4 deg below the horizon: sky lit, land not. The best minute of the day.
  dawn: ['R', -0.28],
  // Low enough that shadows still rake across the field.
  morning: ['R', 2.20],
  noon: ['N', 0.00],
  // Disc at ~6 deg. Photographic golden hour is roughly -4..+6 deg of altitude;
  // an offset any larger lands the sun too high and the light goes ordinary.
  goldenHour: ['S', -0.55],
  // sunsetHour is already the refracted horizon crossing, so a small negative
  // offset puts the disc straddling the skyline rather than under it.
  sunset: ['S', -0.03],
  // Heart of the blue hour, sun at ~-6 deg. Past -8 it is just dusk.
  blueHour: ['S', 0.42],
  night: ['S', 2.80],
  midnight: ['M', 0.00],
};

/** Order used by nextPreset/previousPreset fallbacks and by the HUD. */
export const PRESET_ORDER = Object.freeze([
  'dawn',
  'morning',
  'noon',
  'goldenHour',
  'sunset',
  'blueHour',
  'night',
  'midnight',
]);

const PRESET_LABELS = Object.freeze({
  dawn: 'Dawn',
  morning: 'Morning',
  noon: 'Noon',
  goldenHour: 'Golden Hour',
  sunset: 'Sunset',
  blueHour: 'Blue Hour',
  night: 'Night',
  midnight: 'Midnight',
});

// ---------------------------------------------------------------------------
// CyclicHermiteTrack
// ---------------------------------------------------------------------------

/**
 * A cyclic, monotone cubic Hermite curve over an arbitrary period.
 *
 * Tangents use the Fritsch-Carlson filter: zero at every local extremum, and
 * clamped to 3x the smaller neighbouring secant elsewhere. That is the standard
 * proof-backed condition for "this spline never leaves the interval spanned by
 * its keys", which is what stops sun intensity from dipping negative between a
 * 0.045 key and a 0.0 key eighteen minutes apart.
 */
export class CyclicHermiteTrack {
  /**
   * @param {string} name  for diagnostics only
   * @param {Array<[number, number]>} keys  [hour, value] pairs, any order
   * @param {number} period
   */
  constructor(name, keys, period = HOURS) {
    this.name = name;
    this.period = period;
    this.count = 0;
    this.times = new Float64Array(0);
    this.values = new Float64Array(0);
    this.tangents = new Float64Array(0);
    this._last = 0;
    this.setKeys(keys);
  }

  setKeys(keys) {
    const period = this.period;
    // Sort a scratch copy by wrapped time; dedupe keys that land on top of each
    // other (which happens when an anchor offset pushes past a neighbour).
    _keyScratch.length = 0;
    for (let i = 0; i < keys.length; i++) {
      _keyScratch.push([mod(keys[i][0], period), keys[i][1]]);
    }
    _keyScratch.sort((a, b) => a[0] - b[0]);

    const times = [];
    const values = [];
    for (let i = 0; i < _keyScratch.length; i++) {
      const t = _keyScratch[i][0];
      if (times.length && t - times[times.length - 1] < 1e-4) {
        // Collapsed key: average rather than discard, so authored intent survives.
        values[values.length - 1] = (values[values.length - 1] + _keyScratch[i][1]) * 0.5;
        continue;
      }
      times.push(t);
      values.push(_keyScratch[i][1]);
    }
    _keyScratch.length = 0;

    const n = times.length;
    if (n === 0) throw new Error(`[DayNightCycle] track "${this.name}" has no keys`);

    this.count = n;
    this.times = Float64Array.from(times);
    this.values = Float64Array.from(values);
    this.tangents = new Float64Array(n);
    this._last = 0;
    this._computeTangents();
    return this;
  }

  _computeTangents() {
    const n = this.count;
    const T = this.times;
    const V = this.values;
    const M = this.tangents;
    if (n === 1) {
      M[0] = 0;
      return;
    }

    // Secant slopes, cyclic. secant[i] is the slope from key i to key i+1.
    const secant = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      let dt = T[j] - T[i];
      if (dt <= 0) dt += this.period;
      secant[i] = (V[j] - V[i]) / dt;
    }

    for (let i = 0; i < n; i++) {
      const p = (i - 1 + n) % n;
      const dPrev = secant[p];
      const dNext = secant[i];
      if (dPrev * dNext <= 0) {
        // Local extremum (or a flat run): a zero tangent is the only value that
        // cannot overshoot, and it gives peaks/plateaus their natural shape.
        M[i] = 0;
        continue;
      }
      let m = (dPrev + dNext) * 0.5;
      // Fritsch-Carlson monotonicity bound.
      const limit = 3 * Math.min(Math.abs(dPrev), Math.abs(dNext));
      if (Math.abs(m) > limit) m = Math.sign(m) * limit;
      M[i] = m;
    }
  }

  /** Index of the segment containing wrapped time t. Cached-then-bisect. */
  _segment(t) {
    const T = this.times;
    const n = this.count;
    if (n === 1) return 0;

    const i = this._last;
    const t0 = T[i];
    const t1 = i === n - 1 ? T[0] + this.period : T[i + 1];
    // The wrap segment also owns times before the first key.
    const tt = i === n - 1 && t < T[0] ? t + this.period : t;
    if (tt >= t0 && tt < t1) return i;

    if (t >= T[n - 1] || t < T[0]) {
      this._last = n - 1;
      return n - 1;
    }
    let lo = 0;
    let hi = n - 1;
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if (T[mid] <= t) lo = mid;
      else hi = mid;
    }
    this._last = lo;
    return lo;
  }

  /** Evaluate at time t (any real value; wrapped internally). */
  eval(t) {
    const n = this.count;
    const V = this.values;
    if (n === 1) return V[0];

    const T = this.times;
    const tw = mod(t, this.period);
    const i = this._segment(tw);
    const j = (i + 1) % n;

    const t0 = T[i];
    let h = T[j] - t0;
    if (h <= 0) h += this.period;
    let x = tw - t0;
    if (x < 0) x += this.period;

    const s = h > 0 ? clamp01(x / h) : 0;
    const s2 = s * s;
    const s3 = s2 * s;

    // Hermite basis.
    const h00 = 2 * s3 - 3 * s2 + 1;
    const h10 = s3 - 2 * s2 + s;
    const h01 = -2 * s3 + 3 * s2;
    const h11 = s3 - s2;

    const M = this.tangents;
    return h00 * V[i] + h10 * h * M[i] + h01 * V[j] + h11 * h * M[j];
  }
}

// ---------------------------------------------------------------------------
// DayNightCycle
// ---------------------------------------------------------------------------

export class DayNightCycle {
  constructor(ctx) {
    this.ctx = ctx;
    this.state = ctx?.state ?? null;
    this.renderer = ctx?.renderer ?? null;
    this.post = null;

    // -- Site + season. Kyoto, first week of April: the sun sakura is lit by.
    this.latitudeDeg = 35.0;
    this.dayOfYear = 96;
    /**
     * Clock hour of solar noon. Slightly past 12 because the site sits east of
     * its timezone meridian - a free half-degree of realism that also keeps
     * "midnight" from landing exactly on the 0/24 wrap seam.
     */
    this.solarNoonHour = 12.15;

    // Derived by _rebuild(); declared here so the shape never changes.
    this.declinationDeg = 0;
    this.sunriseHour = 6;
    this.sunsetHour = 18;
    this.solarMidnightHour = 0.15;
    this.dayLengthHours = 12;
    this._sinLat = 0;
    this._cosLat = 1;
    this._sinDec = 0;
    this._cosDec = 1;

    // -- Tracks. Mired tracks are built from the Kelvin tables at compile time.
    this.tracks = {
      sunIntensity: null,
      sunMired: null,
      sunSaturation: null,
      moonIntensity: null,
      moonMired: null,
      ambientIntensity: null,
      exposure: null,
    };

    // -- Published lighting. Stable object identity: hold a reference to it.
    this.lighting = {
      /** Linear-space sun light colour, chroma decoupled from intensity. */
      sunColor: new Color(1, 1, 1),
      /** DirectionalLight intensity after weather occlusion. */
      sunIntensity: 0,
      /** Same, ignoring weather - for consumers doing their own cloud occlusion. */
      sunIntensityClear: 0,
      sunKelvin: 5900,

      moonColor: new Color(0.7, 0.81, 1),
      moonIntensity: 0,
      moonIntensityClear: 0,
      moonKelvin: 7000,

      ambientIntensity: 1,
      ambientIntensityClear: 1,

      /**
       * Dominant key light, sun and moon crossfaded by relative intensity. For
       * rigs that can only afford one shadow-casting directional light.
       */
      keyColor: new Color(1, 1, 1),
      keyIntensity: 0,
      /** Which body the key currently is - aim the light at that one. */
      keyIsMoon: false,

      /** Tone-mapping multiplier actually in use this frame (smoothed). */
      exposure: 1,
      /** Where the adaptation is heading, after weather. */
      exposureTarget: 1,
      /** Same, straight off the curve with no weather term. */
      exposureTargetClear: 1,
      /** log2 of the live exposure - the space adaptation is integrated in. */
      exposureEV: 0,
    };

    /** Derived scalars other systems key off. Stable object identity. */
    this.factors = {
      /** Sun altitude in radians (negative below horizon) and degrees. */
      elevation: 0,
      elevationDeg: 0,
      /** Azimuth in radians, clockwise from north. */
      azimuth: 0,
      /** 1 in full daylight, 0 once the sun is properly down. */
      day: 0,
      /** 1 in true night (past nautical twilight). */
      night: 1,
      /** Whatever is neither day nor night. */
      twilight: 0,
      /** Peaks while the sun sits between +0.5 and +4 degrees of altitude. */
      golden: 0,
      /** Peaks while the sun sits between -6.5 and -4 degrees of altitude. */
      blue: 0,
      /** Clear-sky star visibility; multiply by your own cloud term. */
      stars: 0,
      /** Phase-weighted moon illumination, 0..1. */
      moonIllumination: 1,
      /** True while the sun disc is geometrically above the horizon. */
      sunUp: false,
      /** In-game hours elapsed per real second this frame (signed). */
      hoursPerSecond: 0,
    };

    // -- Clock internals.
    this._effectiveSpeed = 0;
    /**
     * The clock eases into a new speed instead of snapping, so dragging the
     * time-lapse slider does not jolt the sun. 0 disables the ramp entirely.
     */
    this.speedSmoothing = 9;
    /**
     * Pausing decelerates much harder than a speed change. At 6 in-game hours
     * per second the gentle ramp would coast most of an hour past the moment
     * the viewer asked to freeze, which reads as the pause button not working.
     */
    this.pauseSmoothing = 26;
    this.dayCount = 0;

    this._scrub = {
      active: false,
      from: 0,
      delta: 0,
      elapsed: 0,
      duration: 0,
      ease: Easing.inOutSine,
    };

    // -- Exposure adaptation internals.
    this._ev = 0;
    this._adaptBoost = 0;
    this._flashAdapt = 0;
    /** Set false if another system wants to own renderer.toneMappingExposure. */
    this.driveRendererExposure = true;
    /** Stop down slightly during a lightning flash, like a real camera would. */
    this.reactToLightning = true;
    /** Overcast/storm response strength, 0 disables weather coupling entirely. */
    this.weatherResponse = 1;

    // -- Evaluation throttling (see onQualityChange).
    this._evalStride = 1;
    this._maxHourStep = 0.004;
    this._framesSinceEval = 0;
    this._lastEvalHour = Number.NaN;
    this._forceEval = true;
    this._initialised = false;

    // -- Caches for the string getters the HUD polls.
    this._clockString = '00:00';
    this._clockMinute = -1;
    this._phaseName = 'night';
    this._presetCache = null;

    this._listeners = new Set();
    /** Reused notification payload - listeners must not retain it. */
    this._changePayload = { hour: 0, reason: 'init' };

    this._rebuild();

    if (this.state) {
      const h = mod(this.state.time.timeOfDay, HOURS);
      this.state.time.timeOfDay = h;
      this._effectiveSpeed = this.state.time.paused ? 0 : this.state.time.daySpeed;
      // Populate factors + lighting immediately: sibling systems legitimately
      // read `dayNight.lighting` from their own init()/link(), which both run
      // before the first update(). Nobody should ever see the default values.
      this._updateFactors(h);
      this._evaluateTracks(h, this.state);
      this._applyWeather(this.state);
      this.lighting.exposure = this.lighting.exposureTarget;
      this._ev = Math.log2(Math.max(0.02, this.lighting.exposureTarget));
      this.lighting.exposureEV = this._ev;
    }
  }

  link(systems) {
    this.post = systems?.post ?? null;
  }

  // -------------------------------------------------------------------------
  // Solar geometry
  // -------------------------------------------------------------------------

  /**
   * Rebuild declination, the anchor hours, and every track. Cheap enough to call
   * whenever the site or season changes; never called per frame.
   */
  _rebuild() {
    // Cooper's approximation: declination as the projection of the axial tilt
    // onto the orbit, zero at the equinoxes (day 81 ~ March 22).
    const dec = AXIAL_TILT * Math.sin((2 * Math.PI * (this.dayOfYear - 81)) / 365.24);
    this.declinationDeg = dec * RAD2DEG;

    const lat = this.latitudeDeg * DEG2RAD;
    this._sinLat = Math.sin(lat);
    this._cosLat = Math.cos(lat);
    this._sinDec = Math.sin(dec);
    this._cosDec = Math.cos(dec);

    // Hour angle at which the disc crosses the refracted horizon.
    const denom = this._cosLat * this._cosDec;
    let halfDay;
    if (Math.abs(denom) < 1e-6) {
      halfDay = 6; // Degenerate (pole); fall back to a nominal 12h day.
    } else {
      const cosH0 = (Math.sin(HORIZON_ALTITUDE) - this._sinLat * this._sinDec) / denom;
      if (cosH0 <= -1) halfDay = 12; // Midnight sun.
      else if (cosH0 >= 1) halfDay = 0; // Polar night.
      else halfDay = (Math.acos(cosH0) * RAD2DEG) / DEG_PER_HOUR;
    }

    this.dayLengthHours = halfDay * 2;
    this.sunriseHour = mod(this.solarNoonHour - halfDay, HOURS);
    this.sunsetHour = mod(this.solarNoonHour + halfDay, HOURS);
    this.solarMidnightHour = mod(this.solarNoonHour + 12, HOURS);

    this._buildTracks();
    this._presetCache = null;
    this._forceEval = true;
  }

  /** Resolve an ['R'|'N'|'S'|'M', offset] anchor pair to an absolute clock hour. */
  _anchorHour(base, offset) {
    let a;
    switch (base) {
      case 'R': a = this.sunriseHour; break;
      case 'S': a = this.sunsetHour; break;
      case 'N': a = this.solarNoonHour; break;
      case 'M': a = this.solarMidnightHour; break;
      default: a = 0; break;
    }
    return mod(a + offset, HOURS);
  }

  _compileKeys(specs, transform) {
    const out = new Array(specs.length);
    for (let i = 0; i < specs.length; i++) {
      const s = specs[i];
      const v = transform ? transform(s[2]) : s[2];
      out[i] = [this._anchorHour(s[0], s[1]), v];
    }
    return out;
  }

  _buildTracks() {
    const toMired = (k) => 1e6 / k;
    const t = this.tracks;
    const set = (key, name, specs, transform) => {
      const keys = this._compileKeys(specs, transform);
      if (t[key]) t[key].setKeys(keys);
      else t[key] = new CyclicHermiteTrack(name, keys);
    };

    set('sunIntensity', 'sunIntensity', SUN_INTENSITY_KEYS);
    set('sunMired', 'sunMired', SUN_KELVIN_KEYS, toMired);
    set('sunSaturation', 'sunSaturation', SUN_SATURATION_KEYS);
    set('moonIntensity', 'moonIntensity', MOON_INTENSITY_KEYS);
    set('moonMired', 'moonMired', MOON_KELVIN_KEYS, toMired);
    set('ambientIntensity', 'ambientIntensity', AMBIENT_INTENSITY_KEYS);
    set('exposure', 'exposure', EXPOSURE_KEYS);
  }

  /** Sun altitude in radians at an arbitrary clock hour. */
  getSunElevation(hour = this.state ? this.state.time.timeOfDay : 12) {
    const H = (mod(hour, HOURS) - this.solarNoonHour) * DEG_PER_HOUR * DEG2RAD;
    const s = this._sinLat * this._sinDec + this._cosLat * this._cosDec * Math.cos(H);
    return Math.asin(clamp(s, -1, 1));
  }

  /** Sun azimuth in radians, clockwise from north (+X east, -Z north). */
  getSunAzimuth(hour = this.state ? this.state.time.timeOfDay : 12) {
    const h = mod(hour, HOURS);
    const H = (h - this.solarNoonHour) * DEG_PER_HOUR * DEG2RAD;
    const elev = this.getSunElevation(h);
    const cosElev = Math.cos(elev);
    const denom = cosElev * this._cosLat;
    if (Math.abs(denom) < 1e-6) return Math.PI;
    const cosAz = clamp((this._sinDec - Math.sin(elev) * this._sinLat) / denom, -1, 1);
    const az = Math.acos(cosAz);
    // Hour angle positive means afternoon, i.e. the sun has swung west.
    return Math.sin(H) > 0 ? 2 * Math.PI - az : az;
  }

  /**
   * Unit vector pointing from the world toward the sun at `hour`.
   * @param {{x:number,y:number,z:number}} out
   */
  getSunDirection(out, hour = this.state ? this.state.time.timeOfDay : 12) {
    const elev = this.getSunElevation(hour);
    const az = this.getSunAzimuth(hour);
    const c = Math.cos(elev);
    out.x = Math.sin(az) * c;
    out.y = Math.sin(elev);
    out.z = -Math.cos(az) * c;
    return out;
  }

  // -------------------------------------------------------------------------
  // Clock API
  // -------------------------------------------------------------------------

  getTimeOfDay() {
    return this.state ? this.state.time.timeOfDay : 0;
  }

  /**
   * Jump the clock. This is what the HUD slider calls.
   * @param {number} hour  any real number; wrapped into 0..24
   * @param {{adapt?: 'snap'|'fast'|'natural'}} [opts]
   *   'snap'    exposure follows instantly (correct while dragging a slider)
   *   'fast'    exposure ramps over a few tenths of a second
   *   'natural' exposure adapts at the normal, slow, cinematic rate
   */
  setTimeOfDay(hour, opts) {
    const adapt = opts && opts.adapt ? opts.adapt : 'snap';
    const h = mod(hour, HOURS);
    if (this.state) this.state.time.timeOfDay = h;
    this._scrub.active = false;
    this._forceEval = true;
    this._applyAdaptMode(adapt);
    this._emitChange('set');
    return h;
  }

  /**
   * Animated scrub to an hour. Time still moves forward through every
   * intervening moment, so the transition reads as a time-lapse rather than a cut.
   * @param {number} hour
   * @param {number} duration seconds; <= 0 falls through to setTimeOfDay
   * @param {{direction?: 'forward'|'backward'|'shortest', ease?: Function, adapt?: string}} [opts]
   */
  scrubTo(hour, duration = 1.4, opts) {
    if (!this.state) return;
    if (!(duration > 0)) {
      this.setTimeOfDay(hour, opts);
      return;
    }
    const from = mod(this.state.time.timeOfDay, HOURS);
    const target = mod(hour, HOURS);
    const direction = (opts && opts.direction) || 'forward';

    let delta;
    if (direction === 'shortest') {
      delta = mod(target - from + 12, HOURS) - 12;
    } else if (direction === 'backward') {
      delta = -mod(from - target, HOURS);
    } else {
      delta = mod(target - from, HOURS);
    }
    // A no-op scrub would divide by a zero span; treat it as a full cycle only
    // if the caller explicitly asked for movement, otherwise just snap.
    if (Math.abs(delta) < 1e-4) {
      this.setTimeOfDay(target, opts);
      return;
    }

    const s = this._scrub;
    s.active = true;
    s.from = from;
    s.delta = delta;
    s.elapsed = 0;
    s.duration = duration;
    s.ease = (opts && opts.ease) || Easing.inOutSine;
    if (opts && opts.adapt) this._applyAdaptMode(opts.adapt);
    this._emitChange('scrub');
  }

  /** True while an animated scrub is running. */
  get scrubbing() {
    return this._scrub.active;
  }

  cancelScrub() {
    this._scrub.active = false;
  }

  setPaused(paused) {
    if (this.state) this.state.time.paused = !!paused;
    return !!paused;
  }

  togglePause() {
    return this.setPaused(!(this.state && this.state.time.paused));
  }

  get paused() {
    return !!(this.state && this.state.time.paused);
  }

  /** In-game hours per real second. Negative runs the day backwards. */
  setDaySpeed(hoursPerSecond) {
    if (this.state) this.state.time.daySpeed = hoursPerSecond;
    return hoursPerSecond;
  }

  /** Site latitude in degrees; rebuilds every anchor, keyframe and preset. */
  setLatitude(deg) {
    this.latitudeDeg = clamp(deg, -66, 66);
    this._rebuild();
  }

  /** Day of year 1..365; drives declination, hence day length and sun height. */
  setDayOfYear(day) {
    this.dayOfYear = mod(Math.round(day), 365.24);
    this._rebuild();
  }

  // -------------------------------------------------------------------------
  // Presets
  // -------------------------------------------------------------------------

  /** Clock hour of a named preset, or NaN if the name is unknown. */
  getPresetHour(name) {
    const spec = PRESET_SPECS[name];
    if (!spec) return Number.NaN;
    return this._anchorHour(spec[0], spec[1]);
  }

  /**
   * Cached [{ key, label, hour }] for the HUD. Rebuilt only when the anchors
   * move, so calling this per frame is free after the first call.
   */
  getPresets() {
    if (!this._presetCache) {
      this._presetCache = PRESET_ORDER.map((key) => ({
        key,
        label: PRESET_LABELS[key],
        hour: this.getPresetHour(key),
      }));
    }
    return this._presetCache;
  }

  /**
   * Jump to a named moment.
   * @param {string} name  one of PRESET_ORDER
   * @param {{transition?: number, adapt?: string, direction?: string}} [opts]
   */
  applyPreset(name, opts) {
    const hour = this.getPresetHour(name);
    if (Number.isNaN(hour)) {
      console.warn(`[DayNightCycle] unknown preset "${name}"`);
      return Number.NaN;
    }
    const transition = opts && opts.transition !== undefined ? opts.transition : 0;
    const adapt = (opts && opts.adapt) || 'fast';
    if (transition > 0) this.scrubTo(hour, transition, { ...opts, adapt });
    else this.setTimeOfDay(hour, { adapt });
    return hour;
  }

  /** Key of the preset nearest the current time (cyclic distance). */
  getCurrentPresetKey() {
    const now = this.getTimeOfDay();
    let best = PRESET_ORDER[0];
    let bestDist = Infinity;
    for (let i = 0; i < PRESET_ORDER.length; i++) {
      const h = this.getPresetHour(PRESET_ORDER[i]);
      const d = Math.abs(mod(h - now + 12, HOURS) - 12);
      if (d < bestDist) {
        bestDist = d;
        best = PRESET_ORDER[i];
      }
    }
    return best;
  }

  /**
   * Advance to the next preset in clock order (dir 1) or the previous one
   * (dir -1). Chosen by time rather than by array index, so it still does the
   * obvious thing after the user has scrubbed to an arbitrary hour.
   */
  cyclePreset(dir = 1, opts) {
    const now = this.getTimeOfDay();
    let bestKey = null;
    let bestGap = Infinity;
    for (let i = 0; i < PRESET_ORDER.length; i++) {
      const key = PRESET_ORDER[i];
      const h = this.getPresetHour(key);
      // Ignore anything within a couple of minutes of now, or "next" would
      // return the preset we are already standing on.
      let gap = dir >= 0 ? mod(h - now, HOURS) : mod(now - h, HOURS);
      if (gap < 0.04) gap += HOURS;
      if (gap < bestGap) {
        bestGap = gap;
        bestKey = key;
      }
    }
    return this.applyPreset(bestKey, opts);
  }

  nextPreset(opts) {
    return this.cyclePreset(1, opts);
  }

  previousPreset(opts) {
    return this.cyclePreset(-1, opts);
  }

  // -------------------------------------------------------------------------
  // Change notification (local; the shared bus has no time event to reuse)
  // -------------------------------------------------------------------------

  /** @param {(payload:{hour:number, reason:string}) => void} fn */
  onTimeChanged(fn) {
    this._listeners.add(fn);
    return () => this._listeners.delete(fn);
  }

  _emitChange(reason) {
    if (!this._listeners.size) return;
    // Reused payload: a HUD slider drag calls setTimeOfDay every frame, and a
    // fresh object per call is exactly the kind of drip-feed that lands a GC
    // pause in the middle of a camera move.
    const p = this._changePayload;
    p.hour = this.getTimeOfDay();
    p.reason = reason;
    for (const fn of this._listeners) {
      try {
        fn(p);
      } catch (err) {
        console.error('[DayNightCycle] time listener threw:', err);
      }
    }
  }

  // -------------------------------------------------------------------------
  // Frame update
  // -------------------------------------------------------------------------

  update(dt, state) {
    if (!state) return;
    const time = state.time;

    // --- 1. Advance the clock -------------------------------------------------
    const prev = mod(time.timeOfDay, HOURS);
    let hour;
    const scrub = this._scrub;

    // A scrub deliberately ignores `paused`: the viewer asked to be taken to a
    // specific moment, and freezing halfway there is never what they meant.
    if (scrub.active) {
      scrub.elapsed += dt;
      const s = clamp01(scrub.duration > 0 ? scrub.elapsed / scrub.duration : 1);
      hour = mod(scrub.from + scrub.delta * scrub.ease(s), HOURS);
      if (s >= 1) {
        scrub.active = false;
        // Rejoin the normal clock exactly at the target, not one frame past it.
        hour = mod(scrub.from + scrub.delta, HOURS);
      }
      // A scrub is authoritative, so the speed ramp restarts from rest afterward.
      this._effectiveSpeed = 0;
    } else {
      const wanted = time.paused ? 0 : time.daySpeed;
      const lambda = time.paused ? this.pauseSmoothing : this.speedSmoothing;
      if (lambda > 0) {
        this._effectiveSpeed = damp(this._effectiveSpeed, wanted, lambda, dt);
        // Exponential decay never reaches zero; snap once the residual drift is
        // below a second of in-game time per real minute.
        if (Math.abs(this._effectiveSpeed - wanted) < 1e-5) this._effectiveSpeed = wanted;
      } else {
        this._effectiveSpeed = wanted;
      }
      hour = mod(prev + this._effectiveSpeed * dt, HOURS);
    }

    time.timeOfDay = hour;

    // Signed hours/second actually observed, used to speed up adaptation during
    // a time-lapse. Shortest-arc so the midnight wrap does not read as -24h/s.
    const dHour = mod(hour - prev + 12, HOURS) - 12;
    this.factors.hoursPerSecond = dt > 0 ? dHour / dt : 0;

    // Day rollover, direction-aware. Compared against this frame's own starting
    // hour rather than a cached one, so a scrub that runs through midnight still
    // counts the day while a discrete jump across it correctly does not.
    if (dHour > 0 && hour < prev) this.dayCount++;
    else if (dHour < 0 && hour > prev) this.dayCount--;

    // --- 2. Solar geometry and derived factors --------------------------------
    this._updateFactors(hour);

    // --- 3. Evaluate the lighting tracks --------------------------------------
    // Throttled on lower tiers: at the default day speed the curves move by
    // ~0.0003h per frame, so evaluating every 3rd frame is bit-identical to the
    // eye. The hour-delta guard means a fast time-lapse or a scrub still gets
    // per-frame evaluation, so throttling can never cause visible stepping.
    this._framesSinceEval++;
    const hourMoved = Number.isNaN(this._lastEvalHour)
      ? Infinity
      : Math.abs(mod(hour - this._lastEvalHour + 12, HOURS) - 12);
    if (
      this._forceEval ||
      this._framesSinceEval >= this._evalStride ||
      hourMoved > this._maxHourStep
    ) {
      this._evaluateTracks(hour, state);
      this._framesSinceEval = 0;
      this._lastEvalHour = hour;
      this._forceEval = false;
    }

    // --- 4. Weather response (one frame stale by design) ----------------------
    // main.js runs weather AFTER this system, so we read last frame's cloud
    // cover. At the rate weather transitions move, that is unobservable.
    this._applyWeather(state);

    // --- 5. Exposure adaptation ----------------------------------------------
    this._adaptExposure(dt, state);

    // --- 6. Publish exposure --------------------------------------------------
    const exposure = this.lighting.exposure;
    if (this.post && typeof this.post.setExposure === 'function') {
      this.post.setExposure(exposure);
    } else if (this.driveRendererExposure && this.renderer) {
      this.renderer.toneMappingExposure = exposure;
    }
  }

  /**
   * Solar geometry -> the derived 0..1 factors other systems key off. All of the
   * thresholds are real twilight definitions in degrees of solar altitude, not
   * clock hours, so they stay correct at any latitude or time of year.
   */
  _updateFactors(hour) {
    const f = this.factors;
    const elev = this.getSunElevation(hour);
    const elevDeg = elev * RAD2DEG;
    f.elevation = elev;
    f.elevationDeg = elevDeg;
    f.azimuth = this.getSunAzimuth(hour);
    f.sunUp = elevDeg > 0;

    f.day = smoothstep(-2.5, 4.0, elevDeg);
    f.night = 1 - smoothstep(-13.0, -4.0, elevDeg);
    f.twilight = clamp01(1 - f.day - f.night);
    // Golden hour: disc low but up. Blue hour: disc down but the sky still lit.
    f.golden = smoothstep(-4.0, 0.5, elevDeg) * (1 - smoothstep(4.0, 9.0, elevDeg));
    f.blue = smoothstep(-9.0, -6.5, elevDeg) * (1 - smoothstep(-4.0, -0.5, elevDeg));
    f.stars = 1 - smoothstep(-14.0, -3.0, elevDeg);
  }

  _evaluateTracks(hour, state) {
    const t = this.tracks;
    const L = this.lighting;

    L.sunIntensityClear = Math.max(0, t.sunIntensity.eval(hour));
    L.ambientIntensityClear = Math.max(0, t.ambientIntensity.eval(hour));

    const sunMired = Math.max(20, t.sunMired.eval(hour));
    L.sunKelvin = 1e6 / sunMired;
    this._composeLightColor(L.sunColor, L.sunKelvin, clamp01(t.sunSaturation.eval(hour)), 0);

    // Moon: phase-weighted. Illumination fraction is the classic half-cosine;
    // the exponent is the opposition surge - a half moon gives far less than
    // half the light of a full one, because the lunar surface backscatters.
    const phase = state.moon ? state.moon.phase : 0.5;
    const illum = clamp01((1 - Math.cos(2 * Math.PI * phase)) * 0.5);
    // Floor keeps starlight/airglow alive at new moon instead of going pitch black.
    const illumWeighted = 0.12 + 0.88 * Math.pow(illum, 2.6);
    this.factors.moonIllumination = illumWeighted;

    L.moonIntensityClear = Math.max(0, t.moonIntensity.eval(hour)) * illumWeighted;
    const moonMired = Math.max(20, t.moonMired.eval(hour));
    L.moonKelvin = 1e6 / moonMired;
    const purkinje =
      PURKINJE_STRENGTH *
      this.factors.night *
      smoothstep(PURKINJE_KELVIN_LO, PURKINJE_KELVIN_HI, L.moonKelvin);
    this._composeLightColor(L.moonColor, L.moonKelvin, MOON_SATURATION, purkinje);

    L.exposureTargetClear = Math.max(0.05, t.exposure.eval(hour));
  }

  /**
   * Collapse sun + moon into a single dominant key light.
   *
   * A one-directional-light rig (which is what the LOW tier will want, and what
   * a single shadow cascade set can afford) needs to know which body is lighting
   * the scene and with what colour. Crossfading by relative intensity rather
   * than switching on a threshold means the handover through twilight - where
   * both are near zero - is continuous instead of a pop.
   */
  _updateKeyLight() {
    const L = this.lighting;
    const total = L.sunIntensity + L.moonIntensity;
    const t = total > 1e-6 ? L.moonIntensity / total : 0;
    L.keyIntensity = total;
    L.keyIsMoon = t > 0.5;
    L.keyColor.copy(L.sunColor).lerp(L.moonColor, t);
  }

  /**
   * Blackbody -> art-directed linear light colour.
   *
   * kelvinToRGB returns display-referred (sRGB-encoded) values with a max
   * channel of 1. Three steps from there: linearise, pull the chroma toward
   * neutral by `saturation`, then hand back part of the luminance the warm end
   * lost so the intensity curve is not dimming the light twice.
   */
  _composeLightColor(target, kelvin, saturation, purkinje) {
    kelvinToRGB(kelvin, _rgb);
    let r = srgbToLinear(_rgb.r);
    let g = srgbToLinear(_rgb.g);
    let b = srgbToLinear(_rgb.b);

    // Toward white. Doing this in linear space keeps it an energy mix rather
    // than a gamma-space wash.
    r = 1 + (r - 1) * saturation;
    g = 1 + (g - 1) * saturation;
    b = 1 + (b - 1) * saturation;

    if (purkinje > 0) {
      r = lerp(r, r * PURKINJE_COLOR.r, purkinje);
      g = lerp(g, g * PURKINJE_COLOR.g, purkinje);
      b = lerp(b, b * PURKINJE_COLOR.b, purkinje);
    }

    const luma = Math.max(1e-4, 0.2126 * r + 0.7152 * g + 0.0722 * b);
    const scale = clamp(Math.pow(luma, -LUMA_COMPENSATION), 1, LUMA_COMPENSATION_MAX);

    target.setRGB(r * scale, g * scale, b * scale, LinearSRGBColorSpace);
  }

  _applyWeather(state) {
    const L = this.lighting;
    const w = state.weather;
    const c = state.clouds;
    const k = clamp01(this.weatherResponse);

    if (!k || (!w && !c)) {
      L.sunIntensity = L.sunIntensityClear;
      L.moonIntensity = L.moonIntensityClear;
      L.ambientIntensity = L.ambientIntensityClear;
      L.exposureTarget = L.exposureTargetClear;
      this._updateKeyLight();
      return;
    }

    const coverage = c ? clamp01(c.coverage) : 0;
    const storminess = c ? clamp01(c.storminess) : 0;
    // Falling rain is itself an extinction medium - a downpour under thin cloud
    // is still noticeably darker than the same cloud dry.
    const rain = w ? clamp01(w.rainIntensity) : 0;

    // Thin cover scatters light without hiding the disc; only a properly
    // overcast sky removes the key light, and it removes almost all of it.
    const occl = clamp01(smoothstep(0.42, 0.95, coverage) + 0.35 * storminess + 0.2 * rain) * k;
    const gloom = clamp01(0.6 * storminess + 0.4 * rain) * k;

    L.sunIntensity = L.sunIntensityClear * (1 - 0.94 * occl);
    L.moonIntensity = L.moonIntensityClear * (1 - 0.9 * occl);
    // An overcast sky is a huge soft box: dimmer overall, but far more uniform,
    // so ambient falls much less than the direct term does.
    L.ambientIntensity = L.ambientIntensityClear * (1 - 0.3 * occl) * (1 - 0.35 * gloom);

    // Partial compensation only - overcast must still read as a darker day.
    L.exposureTarget = L.exposureTargetClear * (1 + 0.5 * occl + 0.35 * gloom);

    this._updateKeyLight();
  }

  _adaptExposure(dt, state) {
    const L = this.lighting;

    // Lightning: attack fast, release slow, and never more than a third of a
    // stop. A real camera cannot track a 100ms flash - the flash is supposed to
    // blow out - but the operator's iris does register a sustained storm.
    if (this.reactToLightning && state.weather) {
      const strike = clamp01(state.weather.lightning);
      this._flashAdapt =
        strike > this._flashAdapt
          ? damp(this._flashAdapt, strike, 22, dt)
          : damp(this._flashAdapt, strike, 1.6, dt);
    } else if (this._flashAdapt > 0) {
      this._flashAdapt = damp(this._flashAdapt, 0, 1.6, dt);
    }

    const target = Math.max(0.02, L.exposureTarget * (1 - 0.2 * this._flashAdapt));
    const targetEV = Math.log2(target);

    if (!this._initialised) {
      this._ev = targetEV;
      this._initialised = true;
    } else {
      // Boost decays back to 1 so a "fast" jump settles into the normal rate.
      if (this._adaptBoost > 0) {
        this._adaptBoost = damp(this._adaptBoost, 0, ADAPT_BOOST_DECAY, dt);
        if (this._adaptBoost < 0.02) this._adaptBoost = 0;
      }
      // During a time-lapse the world is moving through hours per second; the
      // adaptation has to keep up or the whole sequence sits blown out.
      const lapse = Math.min(3, Math.abs(this.factors.hoursPerSecond) * 0.35);
      const brightening = targetEV < this._ev; // lower EV = stopping down
      const lambda =
        (brightening ? ADAPT_LAMBDA_BRIGHTEN : ADAPT_LAMBDA_DARKEN) *
        (1 + this._adaptBoost + lapse);

      let next = damp(this._ev, targetEV, lambda, dt);
      // Hard slew limit: guarantees a ramp even if lambda is boosted hard.
      const maxStep = ADAPT_MAX_STOPS_PER_SEC * (1 + this._adaptBoost + lapse) * dt;
      const delta = next - this._ev;
      if (delta > maxStep) next = this._ev + maxStep;
      else if (delta < -maxStep) next = this._ev - maxStep;
      this._ev = next;
    }

    L.exposureEV = this._ev;
    L.exposure = Math.pow(2, this._ev);
  }

  _applyAdaptMode(mode) {
    if (mode === 'snap') {
      // Snap needs a target, and the tracks have not been re-evaluated yet at
      // the new hour - do it now so the snap lands on the right value.
      if (this.state) {
        const h = mod(this.state.time.timeOfDay, HOURS);
        this._updateFactors(h);
        this._evaluateTracks(h, this.state);
        this._applyWeather(this.state);
        this._lastEvalHour = h;
        this._framesSinceEval = 0;
        this._forceEval = false;
      }
      this._ev = Math.log2(Math.max(0.02, this.lighting.exposureTarget));
      this.lighting.exposureEV = this._ev;
      this.lighting.exposure = Math.pow(2, this._ev);
      this._adaptBoost = 0;
      this._initialised = true;
    } else if (mode === 'fast') {
      this._adaptBoost = ADAPT_FAST_BOOST;
    } else {
      this._adaptBoost = 0;
    }
  }

  // -------------------------------------------------------------------------
  // Readouts for the HUD
  // -------------------------------------------------------------------------

  /** "18:24". Rebuilt only when the displayed minute changes - no per-frame GC. */
  getClockString() {
    const h = mod(this.getTimeOfDay(), HOURS);
    const totalMinutes = Math.floor(h * 60);
    if (totalMinutes !== this._clockMinute) {
      this._clockMinute = totalMinutes;
      const hh = Math.floor(totalMinutes / 60) % 24;
      const mm = totalMinutes % 60;
      this._clockString = `${hh < 10 ? '0' : ''}${hh}:${mm < 10 ? '0' : ''}${mm}`;
    }
    return this._clockString;
  }

  /**
   * Human-readable phase, driven by sun altitude rather than by clock hour so
   * it stays correct at any latitude or season.
   */
  getPhaseName() {
    const f = this.factors;
    const e = f.elevationDeg;
    const rising = mod(this.getTimeOfDay() - this.solarNoonHour + HOURS, HOURS) > 12;
    let name;
    if (e < -14) name = 'Night';
    else if (e < -6.5) name = rising ? 'Astronomical Dawn' : 'Astronomical Dusk';
    else if (e < -0.83) name = 'Blue Hour';
    else if (e < 5) name = rising ? 'Sunrise' : 'Sunset';
    else if (e < 12) name = 'Golden Hour';
    else if (e > 45) name = 'Midday';
    else name = rising ? 'Morning' : 'Afternoon';
    this._phaseName = name;
    return name;
  }

  /** Last computed phase name, without recomputing it. */
  get phaseName() {
    return this._phaseName;
  }

  /**
   * Sample every curve at an arbitrary hour without disturbing the live state - 
   * for HUD curve previews and offline tuning. Pass `out` to avoid allocating.
   */
  sampleAt(hour, out) {
    const o = out || {
      sunColor: new Color(),
      moonColor: new Color(),
      sunIntensity: 0,
      sunKelvin: 0,
      moonIntensity: 0,
      moonKelvin: 0,
      ambientIntensity: 0,
      exposure: 0,
      elevationDeg: 0,
    };
    const t = this.tracks;
    const h = mod(hour, HOURS);
    o.sunIntensity = Math.max(0, t.sunIntensity.eval(h));
    o.sunKelvin = 1e6 / Math.max(20, t.sunMired.eval(h));
    o.moonIntensity = Math.max(0, t.moonIntensity.eval(h));
    o.moonKelvin = 1e6 / Math.max(20, t.moonMired.eval(h));
    o.ambientIntensity = Math.max(0, t.ambientIntensity.eval(h));
    o.exposure = Math.max(0.05, t.exposure.eval(h));
    o.elevationDeg = this.getSunElevation(h) * RAD2DEG;
    if (o.sunColor) {
      this._composeLightColor(o.sunColor, o.sunKelvin, clamp01(t.sunSaturation.eval(h)), 0);
    }
    if (o.moonColor) {
      const night = 1 - smoothstep(-13, -4, o.elevationDeg);
      this._composeLightColor(o.moonColor, o.moonKelvin, MOON_SATURATION, PURKINJE_STRENGTH * night);
    }
    return o;
  }

  // Convenience accessors so consumers do not have to reach into `.lighting`.
  get exposure() {
    return this.lighting.exposure;
  }
  get sunColor() {
    return this.lighting.sunColor;
  }
  get sunIntensity() {
    return this.lighting.sunIntensity;
  }
  get moonColor() {
    return this.lighting.moonColor;
  }
  get moonIntensity() {
    return this.lighting.moonIntensity;
  }
  get ambientIntensity() {
    return this.lighting.ambientIntensity;
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  onQualityChange(quality) {
    // The only real cost here is track evaluation, and it is small - but the
    // contract is that LOW does less work, and this genuinely does less work.
    switch (quality && quality.tier) {
      case 'low':
        this._evalStride = 4;
        this._maxHourStep = 0.01;
        this.reactToLightning = false;
        break;
      case 'medium':
        this._evalStride = 2;
        this._maxHourStep = 0.006;
        this.reactToLightning = true;
        break;
      default:
        this._evalStride = 1;
        this._maxHourStep = 0.004;
        this.reactToLightning = true;
        break;
    }
    this._forceEval = true;
  }

  dispose() {
    this._listeners.clear();
    this.post = null;
    this._presetCache = null;
  }
}
