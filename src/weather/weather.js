/**
 * weather.js - the world's weather brain.
 *
 * Owns `state.weather.*` and drives `state.clouds.*`. Three things live here:
 *
 *   1. A parameter blender. Every weather type is a fully specified vector of
 *      ~30 numbers; the "current weather" is always a point somewhere between
 *      two of those vectors. Nothing in this file ever snaps - a change of
 *      weather is a slow continuous walk through parameter space, and a change
 *      requested mid-walk re-anchors from wherever we currently are.
 *
 *   2. An autonomous Markov drift. Weather picks its own successor with
 *      plausible weights, biased by time of day (radiation fog wants dawn,
 *      convective storms want mid-afternoon), and sits there for a dwell time
 *      measured in minutes before moving on.
 *
 *   3. The physical consequences: surface wetness integrates while it rains and
 *      evaporates slowly afterwards at a rate that depends on sun, wind,
 *      humidity and air temperature; lightning fires as a time-varying Poisson
 *      process with a multi-stroke flicker envelope.
 *
 * Everything in update() is scalar arithmetic over preallocated Float32Arrays.
 * The only allocation is the payload object of a lightning strike, which happens
 * a handful of times per minute at worst.
 */

import { Color, Vector3 } from 'three';
import { WEATHER, EVENTS } from '../core/state.js';
import {
  TAU,
  clamp,
  clamp01,
  lerp,
  smoothstep,
  smootherstep,
  makeRNG,
} from '../core/math.js';

// ---------------------------------------------------------------------------
// Parameter vector layout
// ---------------------------------------------------------------------------
// One flat Float32Array per weather preset. Indices are explicit constants so
// the blend loop stays monomorphic - no string keys in the hot path.

const P_COVERAGE      = 0;   // cloud coverage 0..1
const P_DENSITY       = 1;   // cloud optical density
const P_ALTITUDE      = 2;   // cloud base, world units
const P_THICKNESS     = 3;   // cloud slab thickness
const P_CLOUD_SPEED   = 4;   // advection multiplier
const P_ABSORPTION    = 5;   // Beer's-law absorption
const P_EROSION       = 6;   // detail-noise erosion of cloud edges
const P_STORMINESS    = 7;   // 0..1 anvil darkening

const P_FOG_DENSITY   = 8;   // distance-haze extinction coefficient
const P_FOG_R         = 9;   // fog chroma FILTER (normalised to luminance 1)
const P_FOG_G         = 10;
const P_FOG_B         = 11;
const P_FOG_TINT      = 12;  // how strongly the filter overrides the sky colour
const P_FOG_LUMA      = 13;  // brightness multiplier on the fog colour
const P_FOG_DESAT     = 14;  // pull toward luminance 0..1

const P_MIST          = 15;  // ground mist layer strength 0..1
const P_MIST_HEIGHT   = 16;  // metres above ground the mist reaches
const P_HORIZON_LIFT  = 17;  // 0..1 how far the visible horizon is raised

const P_WIND          = 18;  // base wind speed
const P_GUSTINESS     = 19;
const P_TURBULENCE    = 20;

const P_RAIN          = 21;  // precipitation rate 0..1 (before cold->snow split)
const P_PETAL         = 22;  // petal spawn multiplier
const P_LIGHTNING     = 23;  // mean strikes per second

const P_SUN_DIM       = 24;  // 0..1 fraction of direct sun the cloud deck eats
const P_AMBIENT       = 25;  // ambient multiplier hint
const P_TURBIDITY     = 26;  // atmospheric turbidity hint
const P_HUMIDITY      = 27;  // 0..1 relative humidity
const P_TEMP_OFFSET    = 28; // degrees C offset from the seasonal baseline
const P_DESAT         = 29;  // scene desaturation hint 0..1

const PARAM_COUNT = 30;

/**
 * Authored presets. These are the whole art direction of the weather system - 
 * read them as a lookbook, not as config. Values were tuned against each other
 * so the ladder from clear -> partlyCloudy -> overcast -> rain -> storm reads as
 * a single continuous worsening, and so fog / petalStorm sit off to the side as
 * distinct, recognisable events rather than "overcast but different".
 *
 * `fogTint` is authored as an sRGB hex and compiled into a luminance-normalised
 * chroma filter, so it tints the *sky's own* colour rather than replacing it.
 * That is what keeps storm fog dark at noon and near-black at midnight instead
 * of glowing grey against a black sky.
 */
const PRESETS = {
  [WEATHER.CLEAR]: {
    cloudCoverage: 0.07, cloudDensity: 0.55, cloudAltitude: 1500, cloudThickness: 380,
    cloudSpeed: 0.70, cloudAbsorption: 0.60, cloudErosion: 0.55, storminess: 0.0,
    fogDensity: 0.0011, fogTint: '#aebfd2', fogTintWeight: 0.22, fogLuma: 1.06, fogDesat: 0.00,
    mist: 0.03, mistHeight: 6, horizonLift: 0.00,
    wind: 2.3, gustiness: 0.26, turbulence: 0.28,
    rain: 0.0, petalBoost: 1.00, lightningRate: 0.0,
    sunDim: 0.02, ambient: 1.00, turbidity: 2.1, humidity: 0.38, tempOffset: 1.2, desaturation: 0.00,
    dwellMin: 165, dwellMax: 340, arrive: 55, depart: 22,
  },

  [WEATHER.PARTLY_CLOUDY]: {
    // Matches state.js defaults so frame zero never jumps.
    cloudCoverage: 0.42, cloudDensity: 0.62, cloudAltitude: 950, cloudThickness: 700,
    cloudSpeed: 1.00, cloudAbsorption: 0.85, cloudErosion: 0.38, storminess: 0.04,
    fogDensity: 0.0022, fogTint: '#b6c2ce', fogTintWeight: 0.34, fogLuma: 1.00, fogDesat: 0.05,
    mist: 0.05, mistHeight: 8, horizonLift: 0.05,
    wind: 3.2, gustiness: 0.34, turbulence: 0.40,
    rain: 0.0, petalBoost: 1.15, lightningRate: 0.0,
    sunDim: 0.12, ambient: 1.05, turbidity: 2.5, humidity: 0.50, tempOffset: 0.4, desaturation: 0.04,
    dwellMin: 150, dwellMax: 310, arrive: 45, depart: 20,
  },

  [WEATHER.OVERCAST]: {
    cloudCoverage: 0.90, cloudDensity: 0.74, cloudAltitude: 620, cloudThickness: 920,
    cloudSpeed: 1.20, cloudAbsorption: 0.96, cloudErosion: 0.22, storminess: 0.30,
    fogDensity: 0.0042, fogTint: '#9aa2a9', fogTintWeight: 0.62, fogLuma: 0.86, fogDesat: 0.30,
    mist: 0.10, mistHeight: 12, horizonLift: 0.18,
    wind: 4.3, gustiness: 0.40, turbulence: 0.46,
    rain: 0.0, petalBoost: 0.90, lightningRate: 0.0,
    sunDim: 0.74, ambient: 0.92, turbidity: 3.4, humidity: 0.72, tempOffset: -1.6, desaturation: 0.30,
    dwellMin: 130, dwellMax: 270, arrive: 45, depart: 30,
  },

  [WEATHER.RAIN]: {
    cloudCoverage: 0.97, cloudDensity: 0.84, cloudAltitude: 470, cloudThickness: 1020,
    cloudSpeed: 1.45, cloudAbsorption: 1.06, cloudErosion: 0.16, storminess: 0.52,
    fogDensity: 0.0078, fogTint: '#8d959c', fogTintWeight: 0.78, fogLuma: 0.72, fogDesat: 0.48,
    mist: 0.12, mistHeight: 14, horizonLift: 0.30,
    wind: 5.4, gustiness: 0.52, turbulence: 0.58,
    // Wet petals are heavy: fewer of them get airborne, and they do not travel.
    rain: 0.60, petalBoost: 0.60, lightningRate: 0.010,
    sunDim: 0.88, ambient: 0.80, turbidity: 4.2, humidity: 0.95, tempOffset: -2.6, desaturation: 0.48,
    dwellMin: 95, dwellMax: 210, arrive: 34, depart: 30,
  },

  [WEATHER.STORM]: {
    cloudCoverage: 1.00, cloudDensity: 0.97, cloudAltitude: 330, cloudThickness: 1280,
    cloudSpeed: 2.15, cloudAbsorption: 1.28, cloudErosion: 0.12, storminess: 1.00,
    fogDensity: 0.0112, fogTint: '#6b737c', fogTintWeight: 0.88, fogLuma: 0.54, fogDesat: 0.62,
    mist: 0.16, mistHeight: 18, horizonLift: 0.42,
    wind: 9.6, gustiness: 0.88, turbulence: 0.86,
    // Gusts tear blossom off the tree even though the rain weighs it down.
    rain: 1.00, petalBoost: 1.70, lightningRate: 0.085,
    sunDim: 0.96, ambient: 0.66, turbidity: 5.2, humidity: 0.99, tempOffset: -3.4, desaturation: 0.62,
    dwellMin: 70, dwellMax: 150, arrive: 26, depart: 40,
  },

  [WEATHER.PETAL_STORM]: {
    // A dry wind event, not a rain event: broken high cloud, hard gusts, and the
    // whole canopy stripping into the air at once.
    cloudCoverage: 0.52, cloudDensity: 0.60, cloudAltitude: 1150, cloudThickness: 620,
    cloudSpeed: 1.90, cloudAbsorption: 0.78, cloudErosion: 0.46, storminess: 0.14,
    fogDensity: 0.0030, fogTint: '#c0b2b4', fogTintWeight: 0.38, fogLuma: 1.02, fogDesat: 0.00,
    mist: 0.05, mistHeight: 8, horizonLift: 0.06,
    wind: 8.4, gustiness: 0.96, turbulence: 0.92,
    rain: 0.0, petalBoost: 9.00, lightningRate: 0.0,
    sunDim: 0.22, ambient: 1.08, turbidity: 2.8, humidity: 0.42, tempOffset: 0.6, desaturation: 0.00,
    dwellMin: 50, dwellMax: 100, arrive: 20, depart: 14,
  },

  [WEATHER.FOG]: {
    // Thin high overcast above, and everything below 90 m filled with cloud.
    cloudCoverage: 0.60, cloudDensity: 0.50, cloudAltitude: 700, cloudThickness: 480,
    cloudSpeed: 0.45, cloudAbsorption: 0.70, cloudErosion: 0.50, storminess: 0.06,
    fogDensity: 0.0180, fogTint: '#b4b9bd', fogTintWeight: 0.85, fogLuma: 0.92, fogDesat: 0.55,
    mist: 1.00, mistHeight: 95, horizonLift: 1.00,
    // Fog cannot survive wind - calm air is part of what makes it read as fog.
    wind: 1.00, gustiness: 0.14, turbulence: 0.16,
    rain: 0.0, petalBoost: 0.70, lightningRate: 0.0,
    sunDim: 0.58, ambient: 1.00, turbidity: 3.0, humidity: 0.99, tempOffset: -2.8, desaturation: 0.55,
    dwellMin: 130, dwellMax: 250, arrive: 70, depart: 55,
  },
};

/** Display names, so the HUD does not have to invent them. */
export const WEATHER_LABELS = Object.freeze({
  [WEATHER.CLEAR]: 'Clear',
  [WEATHER.PARTLY_CLOUDY]: 'Partly Cloudy',
  [WEATHER.OVERCAST]: 'Overcast',
  [WEATHER.RAIN]: 'Rain',
  [WEATHER.STORM]: 'Storm',
  [WEATHER.PETAL_STORM]: 'Petal Storm',
  [WEATHER.FOG]: 'Fog',
});

/** Presentation order for HUD cycling - roughly "getting worse", fog last. */
export const WEATHER_ORDER = Object.freeze([
  WEATHER.CLEAR,
  WEATHER.PARTLY_CLOUDY,
  WEATHER.OVERCAST,
  WEATHER.RAIN,
  WEATHER.STORM,
  WEATHER.PETAL_STORM,
  WEATHER.FOG,
]);

/**
 * Markov successor weights. Read "clear is most likely to cloud over, and can
 * occasionally fog in overnight, but will never jump straight to a storm."
 * Self-transitions are deliberately absent.
 */
const MARKOV = Object.freeze({
  [WEATHER.CLEAR]: [
    [WEATHER.PARTLY_CLOUDY, 0.68], [WEATHER.FOG, 0.12],
    [WEATHER.PETAL_STORM, 0.12], [WEATHER.OVERCAST, 0.08],
  ],
  [WEATHER.PARTLY_CLOUDY]: [
    [WEATHER.CLEAR, 0.32], [WEATHER.OVERCAST, 0.34],
    [WEATHER.PETAL_STORM, 0.14], [WEATHER.RAIN, 0.12], [WEATHER.FOG, 0.08],
  ],
  [WEATHER.OVERCAST]: [
    [WEATHER.PARTLY_CLOUDY, 0.40], [WEATHER.RAIN, 0.34],
    [WEATHER.FOG, 0.14], [WEATHER.STORM, 0.12],
  ],
  [WEATHER.RAIN]: [
    [WEATHER.OVERCAST, 0.50], [WEATHER.STORM, 0.24],
    [WEATHER.PARTLY_CLOUDY, 0.16], [WEATHER.FOG, 0.10],
  ],
  // Storms always decay through rain or a sullen overcast; they never just stop.
  [WEATHER.STORM]: [
    [WEATHER.RAIN, 0.62], [WEATHER.OVERCAST, 0.34], [WEATHER.PARTLY_CLOUDY, 0.04],
  ],
  [WEATHER.PETAL_STORM]: [
    [WEATHER.PARTLY_CLOUDY, 0.55], [WEATHER.CLEAR, 0.30], [WEATHER.OVERCAST, 0.15],
  ],
  [WEATHER.FOG]: [
    [WEATHER.OVERCAST, 0.40], [WEATHER.PARTLY_CLOUDY, 0.34],
    [WEATHER.CLEAR, 0.20], [WEATHER.RAIN, 0.06],
  ],
});

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/** Extra smoothing applied on top of the eased crossfade, in 1/seconds. Removes
 *  the velocity kink when a transition is interrupted by another. */
const BLEND_LAMBDA = 3.5;

/**
 * Not every parameter should cross over on the same schedule. Weather does not
 * arrive as one uniform dissolve - the gust front runs ahead of the squall line,
 * the sky darkens next, and the rain only starts once the deck is already
 * overhead. Blending all three in lockstep is the single biggest tell that a
 * weather system is a lerp between two config blocks.
 *
 * PHASE_LEAD - wind, reaching full value at 70% of the transition.
 * PHASE_MAIN - cloud, fog and light: the plain eased curve.
 * PHASE_LAG - precipitation and lightning, held off until 35% in.
 */
const PHASE_MAIN = 0;
const PHASE_LEAD = 1;
const PHASE_LAG = 2;
const PHASE_LEAD_END = 0.70;
const PHASE_LAG_START = 0.35;

/** index -> phase. Built once; read once per parameter per frame. */
const PHASE = new Uint8Array(PARAM_COUNT); // PHASE_MAIN everywhere by default
PHASE[P_WIND] = PHASE_LEAD;
PHASE[P_GUSTINESS] = PHASE_LEAD;
PHASE[P_TURBULENCE] = PHASE_LEAD;
PHASE[P_RAIN] = PHASE_LAG;
PHASE[P_LIGHTNING] = PHASE_LAG;

/** Seconds of autonomous-drift suspension after a manual HUD weather pick. */
const MANUAL_HOLD = 240;

/** Wetting/drying rates, in 1/seconds. Wetting is fast, drying is glacial. */
const WET_LAMBDA_BASE = 0.020;
const WET_LAMBDA_RAIN = 0.075;
const DRY_LAMBDA_BASE = 0.010;

/** Seasonal air-temperature baseline (spring) and diurnal swing, in Celsius. */
const TEMP_MEAN = 11.2;
const TEMP_SWING = 5.6;
/** Hour of the daily temperature maximum; the minimum lands 12 h earlier. */
const TEMP_PEAK_HOUR = 15.5;

/** Maximum simultaneously decaying lightning sub-flashes. */
const MAX_FLICKERS = 12;
/** Minimum seconds between two independent strikes. */
const STRIKE_REFRACTORY = 1.1;

const FLASH_TINT = new Color(0.86, 0.92, 1.0);

const WEATHER_SEED = 0x5a17e9;

// ---------------------------------------------------------------------------
// Preset compilation
// ---------------------------------------------------------------------------

const _compileColor = new Color();

function compilePreset(p, name) {
  const v = new Float32Array(PARAM_COUNT);

  v[P_COVERAGE] = p.cloudCoverage;
  v[P_DENSITY] = p.cloudDensity;
  v[P_ALTITUDE] = p.cloudAltitude;
  v[P_THICKNESS] = p.cloudThickness;
  v[P_CLOUD_SPEED] = p.cloudSpeed;
  v[P_ABSORPTION] = p.cloudAbsorption;
  v[P_EROSION] = p.cloudErosion;
  v[P_STORMINESS] = p.storminess;

  v[P_FOG_DENSITY] = p.fogDensity;

  // Author in sRGB, store as a luminance-normalised linear chroma filter. three's
  // ColorManagement does the sRGB -> linear conversion in setStyle for us.
  _compileColor.set(p.fogTint);
  const lum = Math.max(
    1e-4,
    0.2126 * _compileColor.r + 0.7152 * _compileColor.g + 0.0722 * _compileColor.b
  );
  v[P_FOG_R] = _compileColor.r / lum;
  v[P_FOG_G] = _compileColor.g / lum;
  v[P_FOG_B] = _compileColor.b / lum;
  v[P_FOG_TINT] = p.fogTintWeight;
  v[P_FOG_LUMA] = p.fogLuma;
  v[P_FOG_DESAT] = p.fogDesat;

  v[P_MIST] = p.mist;
  v[P_MIST_HEIGHT] = p.mistHeight;
  v[P_HORIZON_LIFT] = p.horizonLift;

  v[P_WIND] = p.wind;
  v[P_GUSTINESS] = p.gustiness;
  v[P_TURBULENCE] = p.turbulence;

  v[P_RAIN] = p.rain;
  v[P_PETAL] = p.petalBoost;
  v[P_LIGHTNING] = p.lightningRate;

  v[P_SUN_DIM] = p.sunDim;
  v[P_AMBIENT] = p.ambient;
  v[P_TURBIDITY] = p.turbidity;
  v[P_HUMIDITY] = p.humidity;
  v[P_TEMP_OFFSET] = p.tempOffset;
  v[P_DESAT] = p.desaturation;

  for (let i = 0; i < PARAM_COUNT; i++) {
    if (!Number.isFinite(v[i])) {
      throw new Error(`[WeatherSystem] preset "${name}" left parameter ${i} undefined`);
    }
  }
  return v;
}

/** Compiled vectors + the scheduling metadata, keyed by WEATHER.* value. */
const COMPILED = (() => {
  const out = Object.create(null);
  for (const key of Object.keys(PRESETS)) {
    const p = PRESETS[key];
    out[key] = {
      vec: compilePreset(p, key),
      dwellMin: p.dwellMin,
      dwellMax: p.dwellMax,
      arrive: p.arrive,
      depart: p.depart,
    };
  }
  return out;
})();

const VALID_TYPES = new Set(Object.keys(COMPILED));

// ---------------------------------------------------------------------------
// Time-of-day bias for the Markov walk
// ---------------------------------------------------------------------------

/**
 * Smooth 24-hour-wrapping lobe: 1 at `center`, 0 at `halfWidth` hours away,
 * measured the short way around the clock. Built this way rather than from a
 * pair of smoothsteps because the naive version has a discontinuity at midnight
 * - the sky would visibly change its mind about fog as the clock rolled over.
 */
function hourLobe(hour, center, halfWidth) {
  let dh = Math.abs(hour - center);
  if (dh > 12) dh = 24 - dh;
  return 1 - smoothstep(0, halfWidth, dh);
}

/**
 * Multiplier on a candidate's base weight. This is what stops the world feeling
 * like a shuffled deck: radiation fog forms in the calm hours either side of
 * dawn and burns off by mid-morning, convective storms build through the
 * afternoon, and the sky tends to clear overnight as the ground radiates heat.
 */
function timeOfDayBias(type, hour, humidity) {
  switch (type) {
    case WEATHER.FOG: {
      // Radiation fog peaks just before sunrise; a weaker lobe covers evening
      // valley mist. Saturated air after rain makes both far more likely.
      const dawn = hourLobe(hour, 5.0, 3.4);
      const dusk = hourLobe(hour, 21.0, 3.0);
      return (0.18 + 4.2 * dawn + 1.5 * dusk) * (0.45 + humidity);
    }
    case WEATHER.STORM:
      // Convective storms need a day's worth of surface heating first.
      return 0.35 + 2.1 * hourLobe(hour, 16.0, 5.5);
    case WEATHER.PETAL_STORM: {
      // The golden hours - this event exists to be looked at.
      const morning = hourLobe(hour, 7.5, 2.6);
      const evening = hourLobe(hour, 17.0, 2.8);
      return 0.65 + 1.15 * (morning > evening ? morning : evening);
    }
    case WEATHER.CLEAR:
      // Nights clear as the ground radiates its heat away.
      return 1.4 - 0.5 * hourLobe(hour, 12.5, 7.5);
    default:
      return 1;
  }
}

// ---------------------------------------------------------------------------

export class WeatherSystem {
  constructor(ctx) {
    this.ctx = ctx;
    this.bus = ctx.bus;

    const state = ctx.state;
    const startType = VALID_TYPES.has(state.weather.type)
      ? state.weather.type
      : WEATHER.PARTLY_CLOUDY;

    // -- blend state -------------------------------------------------------
    this._from = new Float32Array(PARAM_COUNT); // re-anchored on every request
    this._to = new Float32Array(PARAM_COUNT);   // the preset we are heading for
    this._cur = new Float32Array(PARAM_COUNT);  // damped output, what we ship

    this._from.set(COMPILED[startType].vec);
    this._to.set(COMPILED[startType].vec);
    this._cur.set(COMPILED[startType].vec);

    this._fromType = startType;
    this._toType = startType;
    this._prevType = startType;
    this._committedType = startType;
    this._committedTarget = startType;

    this._t = 1;             // raw transition progress 0..1
    this._duration = 0;      // seconds for the current transition
    this._eased = 1;
    this._phase = new Float64Array(3); // scratch for the lead/main/lag curves

    // -- autonomy ----------------------------------------------------------
    this.auto = true;
    this._rng = makeRNG(WEATHER_SEED);
    this._dwell = this._rollDwell(startType);
    this._holdRemaining = 0;

    // -- climate integrators ----------------------------------------------
    this.temperature = TEMP_MEAN;
    this.humidity = COMPILED[startType].vec[P_HUMIDITY];
    this.wetness = 0;
    this.puddles = 0;
    this.snowFraction = 0;
    this._dayIndex = 0;
    this._lastHour = state.time.timeOfDay;
    // Rolled once per in-game day rather than hashed per frame: math.js's hash1
    // is built for small coordinate-sized integers and goes non-uniform once the
    // input passes 2^53 during its internal multiply, which quietly turned "a
    // rare cold morning" into "every morning".
    this._dayAnomaly = this._rollDayAnomaly();

    // -- lightning ---------------------------------------------------------
    // Sub-flash ages are integrated per frame rather than compared against a
    // global clock: a Float32 clock loses sub-millisecond resolution after an
    // hour of runtime, which is the same order as the flash attack time.
    this._strikeDebt = 0;
    this._strikeThreshold = this._rollExponential();
    this._sinceStrike = 999;
    this._fCount = 0;
    this._fAge = new Float32Array(MAX_FLICKERS);
    this._fAmp = new Float32Array(MAX_FLICKERS);
    this._fAttack = new Float32Array(MAX_FLICKERS);
    this._fDecay = new Float32Array(MAX_FLICKERS);
    this._fNorm = new Float32Array(MAX_FLICKERS);
    this.lightning = 0;
    this.lastStrike = null;

    // -- wind hand-off -----------------------------------------------------
    // state.wind is owned by weather/wind.js, so we publish targets rather than
    // claiming the field. link() looks for an explicit setter; if the wind
    // author did not provide one we write the base values instead, which is
    // safe because we run before wind every frame - if wind maintains its own
    // base it simply overwrites us and the two never oscillate.
    this.windTarget = { strength: 0, gustiness: 0, turbulence: 0 };
    this._windSetter = null;
    this._gustSurge = 0;

    // -- published hints (documented on _publish) --------------------------
    this.windStrength = 0;
    this.windGustiness = 0;
    this.windTurbulence = 0;
    this.sunDim = 0;
    this.ambientBoost = 1;
    this.turbidity = 2.4;
    this.desaturation = 0;
    this.mist = 0;
    this.mistHeight = 0;
    this.horizonLift = 0;
    this.visibility = 3000;

    // -- quality scalars ---------------------------------------------------
    this._rainScale = 1;
    this._petalExcess = 1;
    this._petalGlobal = 1;
    this._erosionScale = 1;
    this._maxFlickers = 4;
    this.onQualityChange(ctx.state.quality);

    // Frame zero must already be consistent: any system constructed after us,
    // or reading state before the first update(), must see a fully settled
    // preset rather than the raw defaults from state.js.
    this._publish(state, 0);
  }

  link(systems) {
    const wind = systems?.wind;
    // Only an unambiguously named hook is safe to call blind - a generic name
    // could belong to a method with an entirely different signature.
    if (wind && typeof wind.setWeatherWind === 'function') {
      this._windSetter = wind.setWeatherWind.bind(wind);
    }
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /** Human-readable name of what the sky is currently doing. */
  get label() {
    return WEATHER_LABELS[this._toType] || this._toType;
  }

  /** 0..1 eased progress of the transition currently in flight. */
  get transitionProgress() {
    return this._eased;
  }

  /**
   * Request a weather change.
   * @param {string} type one of WEATHER.*
   * @param {number} [transitionSeconds] omit for a physically plausible default
   * @param {boolean} [manual=true] manual picks suspend autonomous drift
   * @returns {boolean} false if the type was unknown or already the target
   */
  setWeather(type, transitionSeconds, manual = true) {
    if (!VALID_TYPES.has(type)) {
      console.warn(`[WeatherSystem] unknown weather type "${type}"`);
      return false;
    }
    if (type === this._toType) return false;

    // Re-anchor from wherever the blend actually is right now. Without this,
    // interrupting a transition would snap back to the previous preset.
    this._from.set(this._cur);
    this._to.set(COMPILED[type].vec);

    const from = this._toType;
    this._prevType = this._fromType;
    this._fromType = from;
    this._toType = type;
    this._t = 0;
    this._eased = 0;
    this._duration =
      transitionSeconds !== undefined && transitionSeconds !== null && transitionSeconds >= 0
        ? transitionSeconds
        : this._defaultDuration(from, type);

    if (manual) this._holdRemaining = MANUAL_HOLD;

    const state = this.ctx.state;
    state.weather.type = from;
    state.weather.target = type;
    state.weather.transition = 0;
    this._committedType = from;
    this._committedTarget = type;

    this.bus?.emit(EVENTS.WEATHER_CHANGED, {
      from,
      to: type,
      duration: this._duration,
      manual,
    });
    return true;
  }

  /** Step through WEATHER_ORDER - convenient for a HUD keybind. */
  cycleWeather(direction = 1, transitionSeconds) {
    const i = WEATHER_ORDER.indexOf(this._toType);
    const n = WEATHER_ORDER.length;
    const next = WEATHER_ORDER[(((i + direction) % n) + n) % n];
    return this.setWeather(next, transitionSeconds, true);
  }

  /** Enable/disable autonomous drift. Enabling rolls a fresh dwell. */
  setAuto(enabled) {
    this.auto = !!enabled;
    if (this.auto) {
      this._holdRemaining = 0;
      this._dwell = this._rollDwell(this._toType);
    }
  }

  /**
   * Fire a lightning strike now. Used by the storm scheduler, and available to
   * the HUD as a debug poke.
   * @param {number} [scale=1] multiplier on flash brightness
   */
  triggerLightning(scale = 1) {
    this._fireStrike(this.ctx.state, scale);
  }

  /**
   * LOW must genuinely cost less, so the dials that cost other systems real
   * milliseconds all come down: fewer raindrops, far fewer airborne petals,
   * less cloud detail noise, shorter lightning flicker trains.
   * `_petalExcess` thins the *boost* above 1, `_petalGlobal` scales everything - 
   * both are needed, otherwise LOW would spawn more petals in the rain (where
   * the preset multiplier is below 1) than HIGH does.
   */
  onQualityChange(quality) {
    switch (quality?.tier) {
      case 'low':
        this._rainScale = 0.50; this._petalExcess = 0.35; this._petalGlobal = 0.75;
        this._erosionScale = 0.55; this._maxFlickers = 2;
        break;
      case 'medium':
        this._rainScale = 0.75; this._petalExcess = 0.60; this._petalGlobal = 0.90;
        this._erosionScale = 0.80; this._maxFlickers = 3;
        break;
      case 'ultra':
        this._rainScale = 1.00; this._petalExcess = 1.15; this._petalGlobal = 1.00;
        this._erosionScale = 1.15; this._maxFlickers = 5;
        break;
      default:
        this._rainScale = 1.00; this._petalExcess = 1.00; this._petalGlobal = 1.00;
        this._erosionScale = 1.00; this._maxFlickers = 4;
        break;
    }
  }

  // -------------------------------------------------------------------------
  // Frame
  // -------------------------------------------------------------------------

  update(dt, state) {
    const d = dt > 0 ? (dt < 0.1 ? dt : 0.1) : 0;

    this._absorbExternalRequest(state);
    this._advanceTransition(d, state);
    this._blendParams(d);
    this._evolve(d, state);
    this._publish(state, d);
  }

  dispose() {
    this._windSetter = null;
    this.lastStrike = null;
    this._fCount = 0;
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------

  /**
   * If something wrote state.weather.type/target behind our back (a HUD that
   * pokes state directly rather than calling setWeather), honour it as a
   * request instead of silently overwriting it next frame.
   */
  _absorbExternalRequest(state) {
    const w = state.weather;
    if (w.target !== this._committedTarget && VALID_TYPES.has(w.target)) {
      this.setWeather(w.target, undefined, true);
      return;
    }
    if (w.type !== this._committedType && VALID_TYPES.has(w.type)) {
      this.setWeather(w.type, undefined, true);
    }
  }

  _advanceTransition(d, state) {
    if (this._t >= 1) return;
    this._t = this._duration > 0 ? Math.min(1, this._t + d / this._duration) : 1;
    if (this._t >= 1) {
      // Settled: the "current" type finally becomes the one we were heading for,
      // and the dwell clock for autonomous drift starts here, not at request time.
      this._t = 1;
      this._fromType = this._toType;
      this._from.set(this._to);
      state.weather.type = this._toType;
      this._committedType = this._toType;
      this._dwell = this._rollDwell(this._toType);
    }
  }

  _blendParams(d) {
    const from = this._from;
    const to = this._to;
    const cur = this._cur;

    // smootherstep has zero first AND second derivative at both ends, so weather
    // eases in and out of a change without a visible start/stop.
    const t = this._t;
    const e = t >= 1 ? 1 : smootherstep(0, 1, t);
    this._eased = e;

    // Three crossfade curves, evaluated once for the whole vector. All three
    // reach exactly 1 at t = 1, so every phase still lands together.
    const ph = this._phase;
    ph[PHASE_MAIN] = e;
    ph[PHASE_LEAD] = t >= PHASE_LEAD_END ? 1 : smootherstep(0, PHASE_LEAD_END, t);
    ph[PHASE_LAG] = t <= PHASE_LAG_START ? 0 : smootherstep(PHASE_LAG_START, 1, t);

    // One exp() for the whole vector. The damping layer exists purely to erase
    // the velocity discontinuity created when setWeather() re-anchors mid-blend.
    const k = d > 0 ? 1 - Math.exp(-BLEND_LAMBDA * d) : 0;

    for (let i = 0; i < PARAM_COUNT; i++) {
      const target = from[i] + (to[i] - from[i]) * ph[PHASE[i]];
      const diff = target - cur[i];
      // Exponential damping approaches but never reaches its target, which
      // leaves parameters like rainIntensity at ~1e-18 forever. Downstream that
      // reads as "it is still raining, very slightly" and quietly breaks any
      // logic gated on precipitation stopping. Snap the negligible tail.
      cur[i] = Math.abs(diff) < 1e-5 * (1 + Math.abs(target)) ? target : cur[i] + diff * k;
    }
  }

  /**
   * Autonomous drift, air temperature, wetness integration, lightning schedule.
   */
  _evolve(d, state) {
    const cur = this._cur;
    const hour = state.time.timeOfDay;

    // -- climate -----------------------------------------------------------
    // Detect the midnight wrap and re-roll the day's warm/cold anomaly.
    // timeOfDay only ever decreases by wrapping past 24.
    if (hour < this._lastHour - 6) {
      this._dayIndex++;
      this._dayAnomaly = this._rollDayAnomaly();
    }
    this._lastHour = hour;

    // Cloud cover flattens the diurnal swing - that is why overcast nights are mild.
    const swing = TEMP_SWING * (1 - 0.55 * cur[P_COVERAGE]);
    const diurnal = Math.cos(((hour - TEMP_PEAK_HOUR) / 24) * TAU) * swing;
    this.temperature = TEMP_MEAN + this._dayAnomaly + cur[P_TEMP_OFFSET] + diurnal;

    // Standing water raises local humidity as it evaporates.
    this.humidity = clamp01(cur[P_HUMIDITY] + this.wetness * 0.08);

    // -- precipitation phase ----------------------------------------------
    // Sleet band between +3 C and freezing. In this spring climate it only ever
    // fires on an anomalously cold pre-dawn, which is exactly when it should.
    this.snowFraction = smoothstep(3.0, 0.2, this.temperature);

    // -- wetness -----------------------------------------------------------
    // Deliberately driven by the UNSCALED precipitation rate: a low quality tier
    // should draw fewer raindrops, not make the ground less wet.
    const fall = cur[P_RAIN];
    const precipWet = fall * (1 - this.snowFraction) + fall * this.snowFraction * 0.35;
    // Saturated air alone leaves surfaces damp with dew - fog mornings glisten.
    const dewWet = smoothstep(0.90, 1.0, this.humidity) * 0.30;
    // Gate then saturate, rather than adding a constant to any non-zero rate.
    // A constant offset puts a step at precip == 0 that the asymptotic blend can
    // never climb back down from, and drizzle would wet a surface as fast as a
    // downpour. This way light rain darkens the ground, only a storm soaks it.
    const rainWet = smoothstep(0.004, 0.05, precipWet) * clamp01(0.25 + 0.75 * precipWet);
    const wetTarget = clamp01(Math.max(dewWet, rainWet));

    if (wetTarget > this.wetness) {
      const lam = WET_LAMBDA_BASE + WET_LAMBDA_RAIN * precipWet;
      this.wetness += (wetTarget - this.wetness) * (1 - Math.exp(-lam * d));
    } else {
      // Evaporation: sun does most of the work, wind helps, humidity fights it,
      // and cold air holds almost no vapour. Night overcast barely dries at all.
      const sunV = state.sun?.visibility ?? 0;
      const lam =
        DRY_LAMBDA_BASE *
        (0.25 + 1.50 * sunV) *
        (0.55 + 0.10 * cur[P_WIND]) *
        (1.15 - this.humidity) *
        (0.50 + 0.06 * clamp(this.temperature, 0, 30));
      this.wetness += (wetTarget - this.wetness) * (1 - Math.exp(-Math.max(lam, 1e-5) * d));
    }
    this.wetness = clamp01(this.wetness);

    // Puddles lag wetness in both directions: they need a soaked surface to form
    // and they outlive the rain by a long way.
    const puddleTarget = smoothstep(0.55, 1.0, this.wetness);
    const puddleLam = puddleTarget > this.puddles ? 0.020 + 0.030 * precipWet : 0.0035;
    this.puddles += (puddleTarget - this.puddles) * (1 - Math.exp(-puddleLam * d));
    this.puddles = clamp01(this.puddles);

    // -- autonomous drift --------------------------------------------------
    if (this._holdRemaining > 0) this._holdRemaining = Math.max(0, this._holdRemaining - d);
    if (this.auto && this._holdRemaining <= 0 && this._t >= 1) {
      this._dwell -= d;
      if (this._dwell <= 0) {
        const next = this._pickNext(this._toType, hour);
        if (next) this.setWeather(next, undefined, false);
        else this._dwell = this._rollDwell(this._toType);
      }
    }

    // -- lightning ---------------------------------------------------------
    this._updateLightning(d, state);

    // Cold downburst outflow from the last strike, bleeding back to ambient.
    this._gustSurge *= Math.exp(-d / 2.2);
  }

  /**
   * Time-varying Poisson process. Accumulating rate*dt against an Exp(1) draw is
   * exactly correct even as the rate ramps up during a transition into a storm,
   * which a naive countdown timer is not.
   */
  _updateLightning(d, state) {
    const rate = this._cur[P_LIGHTNING];
    this._sinceStrike += d;

    if (rate > 1e-5) {
      this._strikeDebt += rate * d;
      if (this._strikeDebt >= this._strikeThreshold && this._sinceStrike >= STRIKE_REFRACTORY) {
        this._strikeDebt = 0;
        this._strikeThreshold = this._rollExponential();
        this._fireStrike(state, 1);
      }
    } else {
      // Bleed the accumulator away so a long clear spell does not bank a strike
      // that fires the instant a storm arrives.
      this._strikeDebt *= Math.exp(-d * 0.25);
    }

    // Evaluate every live sub-flash. Bounded by MAX_FLICKERS by construction,
    // so this loop can never run away no matter how active the storm is.
    let level = 0;
    for (let i = 0; i < this._fCount; i++) {
      const age = (this._fAge[i] += d);
      if (age < 0) continue; // scheduled to begin a few tens of ms from now
      if (age > this._fDecay[i] * 7) {
        // Swap-remove; order is irrelevant because the strokes combine with max.
        const last = this._fCount - 1;
        this._fAge[i] = this._fAge[last];
        this._fAmp[i] = this._fAmp[last];
        this._fAttack[i] = this._fAttack[last];
        this._fDecay[i] = this._fDecay[last];
        this._fNorm[i] = this._fNorm[last];
        this._fCount--;
        i--;
        continue;
      }
      // Double-exponential: a few-millisecond rise into an exponential falloff.
      // Normalised so the peak of each stroke equals its amplitude exactly.
      const v =
        (Math.exp(-age / this._fDecay[i]) - Math.exp(-age / this._fAttack[i])) *
        this._fNorm[i] *
        this._fAmp[i];
      if (v > level) level = v;
    }
    this.lightning = level > 1 ? 1 : level;
  }

  /**
   * Schedule one strike: a bright leader, a handful of dimmer return strokes
   * over the next few hundred milliseconds, then a slow afterglow if it was
   * close. Distance sets brightness, thunder delay and how smeared it reads.
   */
  _fireStrike(state, scale) {
    const rng = this._rng;
    this._sinceStrike = 0;

    // rng()*rng() is biased toward 0, so most strikes are comfortably distant
    // and the rare close one lands with real weight.
    const distance = lerp(220, 2600, clamp01(rng() * rng() + 0.05 * rng()));
    const distAmp = clamp(1.05 - 0.78 * smoothstep(220, 2600, distance), 0.2, 1.0);
    const storm = clamp01(0.35 + 0.65 * this._cur[P_STORMINESS]);
    const amp = clamp01(distAmp * storm * scale * (0.78 + 0.28 * rng()));

    // Distance smears the flash: overhead is a hard-edged stutter, horizon is a
    // soft pulse. Same reason distant thunder rumbles instead of cracking.
    const smear = 1 + 1.9 * smoothstep(400, 2600, distance);
    const budget = Math.min(this._maxFlickers, MAX_FLICKERS);
    const strokes = 1 + Math.floor(rng() * rng() * budget);

    let delay = 0;
    for (let s = 0; s < strokes && this._fCount < MAX_FLICKERS; s++) {
      // First stroke is the leader; later ones are dimmer, with the occasional
      // bright restrike that makes real lightning feel unpredictable.
      const isRestrike = s > 0 && rng() < 0.22;
      const a = s === 0 ? amp : amp * (isRestrike ? lerp(0.7, 1.0, rng()) : lerp(0.28, 0.62, rng()));
      const attack = lerp(0.004, 0.012, rng()) * smear;
      const decay = (s === 0 ? lerp(0.10, 0.20, rng()) : lerp(0.045, 0.11, rng())) * smear;
      this._pushFlicker(delay, a, attack, decay);
      delay += lerp(0.035, 0.125, rng()) * smear;
    }
    // Near strikes leave a slow glow burning in the cloud base behind the strokes.
    if (distance < 900 && this._fCount < MAX_FLICKERS) {
      this._pushFlicker(0.02, amp * 0.16, 0.05, 0.55);
    }

    // Cold outflow arrives with the strike.
    this._gustSurge = Math.min(0.30, this._gustSurge + 0.10 + 0.14 * amp);

    const az = rng() * TAU;
    const px = state.player?.position?.x ?? 0;
    const pz = state.player?.position?.z ?? 0;
    const base = this._cur[P_ALTITUDE];
    // Rare event - an allocation here costs nothing and lets listeners keep the
    // reference safely, which a shared scratch vector would not.
    const position = new Vector3(
      px + Math.sin(az) * distance,
      base + this._cur[P_THICKNESS] * 0.25,
      pz + Math.cos(az) * distance
    );

    const payload = {
      position,
      distance,
      intensity: amp,
      strokes,
      /** Seconds until the thunder should be heard. Speed of sound at ~15 C. */
      thunderDelay: distance / 340,
    };
    this.lastStrike = payload;
    this.bus?.emit(EVENTS.LIGHTNING_STRIKE, payload);
  }

  /** @param {number} delay seconds from now until this stroke begins. */
  _pushFlicker(delay, amp, attack, decay) {
    const i = this._fCount;
    if (i >= MAX_FLICKERS) return;
    // Peak of exp(-x/d) - exp(-x/a) occurs at x* = ln(d/a) / (1/a - 1/d).
    const xPeak = Math.log(decay / attack) / (1 / attack - 1 / decay);
    const peak = Math.exp(-xPeak / decay) - Math.exp(-xPeak / attack);
    this._fAge[i] = -delay;
    this._fAmp[i] = amp;
    this._fAttack[i] = attack;
    this._fDecay[i] = decay;
    this._fNorm[i] = peak > 1e-4 ? 1 / peak : 1;
    this._fCount = i + 1;
  }

  /**
   * Write the blended parameters out to world state, plus the published hints
   * that sibling systems may read off this instance:
   *
   *   windTarget / windStrength / windGustiness / windTurbulence  -> wind.js
   *   sunDim / ambientBoost / turbidity / desaturation            -> lighting, sky
   *   mist / mistHeight / horizonLift / visibility                -> atmosfx.js
   *   temperature / humidity / puddles / snowFraction             -> anyone
   */
  _publish(state, d) {
    const cur = this._cur;
    const w = state.weather;
    const c = state.clouds;

    // -- clouds ------------------------------------------------------------
    c.coverage = clamp01(cur[P_COVERAGE]);
    c.density = cur[P_DENSITY];
    c.altitude = cur[P_ALTITUDE];
    c.thickness = cur[P_THICKNESS];
    c.speed = cur[P_CLOUD_SPEED];
    c.absorption = cur[P_ABSORPTION];
    c.erosion = cur[P_EROSION] * this._erosionScale;
    c.storminess = clamp01(cur[P_STORMINESS]);

    // -- weather scalars ---------------------------------------------------
    // These three are the canonical mirror of our internal machine - restoring
    // them each frame self-heals anything that poked state directly with a value
    // we could not interpret.
    w.type = this._committedType;
    w.target = this._committedTarget;
    w.transition = this._eased;

    const rainRaw = cur[P_RAIN] * this._rainScale;
    w.rainIntensity = clamp01(rainRaw * (1 - this.snowFraction));
    w.snowIntensity = clamp01(rainRaw * this.snowFraction);
    w.wetness = this.wetness;
    w.lightning = this.lightning;
    // Thin the *excess* above 1 so a low tier tames a petal storm without also
    // rewriting the below-1 multipliers that make rain suppress petals.
    const pb = cur[P_PETAL];
    w.petalBoost = Math.max(
      0,
      (pb > 1 ? 1 + (pb - 1) * this._petalExcess : pb) * this._petalGlobal
    );

    const fogDensity = Math.max(0, cur[P_FOG_DENSITY]);
    w.fogDensity = fogDensity;
    // Koschmieder, 5% contrast threshold - a rough metres figure for anyone who
    // wants to place a far plane or fade distant scatter.
    this.visibility = 3.0 / Math.max(fogDensity, 1e-6);

    this._composeFogColor(state, w.fogColor);

    // -- wind hand-off -----------------------------------------------------
    const surge = this._gustSurge;
    this.windStrength = cur[P_WIND] * (1 + surge);
    this.windGustiness = clamp01(cur[P_GUSTINESS] + surge * 0.35);
    this.windTurbulence = cur[P_TURBULENCE] * (1 + surge * 0.5);
    this.windTarget.strength = this.windStrength;
    this.windTarget.gustiness = this.windGustiness;
    this.windTarget.turbulence = this.windTurbulence;
    if (this._windSetter) {
      this._windSetter(this.windStrength, this.windGustiness, this.windTurbulence, d);
    } else {
      // No agreed setter: write the base values. See the note in the constructor
      // - wind runs after us, so it wins any disagreement and nothing flickers.
      state.wind.strength = this.windStrength;
      state.wind.gustiness = this.windGustiness;
      state.wind.turbulence = this.windTurbulence;
    }

    // -- hints -------------------------------------------------------------
    this.sunDim = clamp01(cur[P_SUN_DIM]);
    this.ambientBoost = cur[P_AMBIENT];
    this.turbidity = cur[P_TURBIDITY];
    this.desaturation = clamp01(cur[P_DESAT]);
    this.mist = clamp01(cur[P_MIST]);
    this.mistHeight = cur[P_MIST_HEIGHT];
    this.horizonLift = clamp01(cur[P_HORIZON_LIFT]);
  }

  /**
   * Fog colour is not a constant. It is the sky's own horizon colour, filtered
   * through the weather's chroma, desaturated, dimmed, floored by moonlight so
   * night fog does not go pure black, and lifted by lightning.
   *
   * Doing it as a filter rather than a mix is what makes a storm read as dark at
   * noon and darker still at midnight; a straight mix toward a grey constant
   * would make storm fog glow brighter than the night sky behind it.
   */
  _composeFogColor(state, out) {
    const cur = this._cur;
    const sky = state.sky;

    // Bias a quarter toward the zenith: pure horizon colour is too warm/bright
    // to read as the colour of the air overhead.
    let r = lerp(sky.horizonColor.r, sky.zenithColor.r, 0.25);
    let g = lerp(sky.horizonColor.g, sky.zenithColor.g, 0.25);
    let b = lerp(sky.horizonColor.b, sky.zenithColor.b, 0.25);

    const tint = cur[P_FOG_TINT];
    r *= lerp(1, cur[P_FOG_R], tint);
    g *= lerp(1, cur[P_FOG_G], tint);
    b *= lerp(1, cur[P_FOG_B], tint);

    const desat = cur[P_FOG_DESAT];
    if (desat > 0) {
      const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      r = lerp(r, l, desat);
      g = lerp(g, l, desat);
      b = lerp(b, l, desat);
    }

    const luma = cur[P_FOG_LUMA];
    r *= luma; g *= luma; b *= luma;

    // Skyglow / moonlight floor. Thick cloud smothers it, so the floor scales
    // down with coverage - a socked-in night really is pitch black.
    const moonV = clamp01(state.moon?.visibility ?? 0);
    const floor = (0.0018 + 0.0095 * moonV) * (1 - 0.7 * clamp01(cur[P_COVERAGE]));
    const mc = state.moon?.color;
    if (mc) {
      r += floor * mc.r; g += floor * mc.g; b += floor * mc.b;
    } else {
      r += floor; g += floor; b += floor;
    }

    // Lightning lights the air itself - the fog is the flash, more than the bolt.
    const flash = this.lightning;
    if (flash > 0) {
      const f = flash * 0.42;
      r += f * FLASH_TINT.r; g += f * FLASH_TINT.g; b += f * FLASH_TINT.b;
    }

    out.setRGB(
      r > 0 ? (r < 4 ? r : 4) : 0,
      g > 0 ? (g < 4 ? g : 4) : 0,
      b > 0 ? (b < 4 ? b : 4) : 0
    );
  }

  // -- scheduling helpers ---------------------------------------------------

  /**
   * How much warmer or colder than the seasonal mean this particular day is.
   * The tails of this are the only thing that can push a pre-dawn shower below
   * freezing, so its width sets how rare spring sleet is - roughly one morning
   * in seven, and never more than a light sleet even then.
   */
  _rollDayAnomaly() {
    return (this._rng() - 0.5) * 6.5;
  }

  _rollDwell(type) {
    const c = COMPILED[type];
    return c.dwellMin + this._rng() * (c.dwellMax - c.dwellMin);
  }

  /** Exp(1) sample, for the Poisson strike process. */
  _rollExponential() {
    return -Math.log(1 - this._rng() * 0.999999);
  }

  /**
   * How long the change should take. Arriving weather has its own pace (a squall
   * line lands in half a minute, fog takes over a minute to form) and departing
   * weather has to clear out first, so both ends contribute.
   */
  _defaultDuration(from, to) {
    const a = COMPILED[to].arrive;
    const dpt = COMPILED[from]?.depart ?? 25;
    const base = 0.6 * a + 0.5 * dpt;
    return base * (0.85 + 0.30 * this._rng());
  }

  /**
   * Weighted successor pick, biased by hour and humidity, with an immediate
   * back-and-forth penalty so the sky does not oscillate A-B-A-B.
   */
  _pickNext(from, hour) {
    const table = MARKOV[from];
    if (!table) return null;
    const hum = this.humidity;

    let total = 0;
    for (let i = 0; i < table.length; i++) {
      const type = table[i][0];
      let w = table[i][1] * timeOfDayBias(type, hour, hum);
      if (type === this._prevType) w *= 0.45;
      total += w;
    }
    if (total <= 0) return null;

    let r = this._rng() * total;
    for (let i = 0; i < table.length; i++) {
      const type = table[i][0];
      let w = table[i][1] * timeOfDayBias(type, hour, hum);
      if (type === this._prevType) w *= 0.45;
      r -= w;
      if (r <= 0) return type;
    }
    return table[table.length - 1][0];
  }
}
