/**
 * state.js - the single mutable source of truth for the whole world.
 *
 * CONTRACT: systems READ freely from this object every frame. Each field has
 * exactly ONE owner system that writes it (marked `@owner` below). If you need
 * to influence a field you do not own, go through the owner's API - never write
 * it directly, or two systems will fight over it and the result will flicker.
 *
 * The object identity is stable for the lifetime of the app: hold a reference,
 * never re-read it from a getter. Nested objects are also stable - mutate their
 * fields in place, never reassign the container (`state.wind = {...}` breaks
 * every system holding a reference to the old object).
 */

import { Vector2, Vector3, Color } from 'three';

export const WEATHER = Object.freeze({
  CLEAR: 'clear',
  PARTLY_CLOUDY: 'partlyCloudy',
  OVERCAST: 'overcast',
  RAIN: 'rain',
  STORM: 'storm',
  PETAL_STORM: 'petalStorm',
  FOG: 'fog',
});

export const QUALITY = Object.freeze({
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
  ULTRA: 'ultra',
});

export function createWorldState() {
  return {
    // -----------------------------------------------------------------------
    // Time  @owner core/engine.js (delta/elapsed/frame) + sky/daynight.js (timeOfDay)
    // -----------------------------------------------------------------------
    time: {
      /** Seconds since start, unscaled by pause. */
      elapsed: 0,
      /** Seconds since last frame, already clamped to avoid tab-switch explosions. */
      delta: 0,
      /** Monotonic frame counter - useful for temporal jitter / TAA sample index. */
      frame: 0,
      /** Hour of day in 0..24 (13.5 === 13:30). Drives everything celestial. */
      timeOfDay: 11.2,
      /** In-game hours per real second. 0 freezes the sun. */
      daySpeed: 0.02,
      /** When true, timeOfDay stops advancing but rendering continues. */
      paused: false,
    },

    // -----------------------------------------------------------------------
    // Sun  @owner sky/celestial.js
    // -----------------------------------------------------------------------
    sun: {
      /** Unit vector pointing FROM the world TOWARD the sun. */
      direction: new Vector3(0, 1, 0),
      /** World-space position on the sky sphere, for lens flare / god ray origin. */
      position: new Vector3(0, 1000, 0),
      /** Linear-space light colour, already atmospherically tinted. */
      color: new Color(1, 1, 1),
      /** Physical-ish irradiance multiplier for the directional light. */
      intensity: 3.0,
      /** Elevation in radians. Negative = below horizon. */
      elevation: 0.6,
      azimuth: 0.0,
      /** 0 when fully below horizon, 1 when comfortably up. Cheap "is it day" test. */
      visibility: 1.0,
    },

    // -----------------------------------------------------------------------
    // Moon  @owner sky/celestial.js
    // -----------------------------------------------------------------------
    moon: {
      direction: new Vector3(0, -1, 0),
      position: new Vector3(0, -1000, 0),
      color: new Color(0.55, 0.68, 1.0),
      intensity: 0.0,
      elevation: -0.6,
      azimuth: Math.PI,
      visibility: 0.0,
      /** 0 = new moon, 0.5 = full, 1 = new again. */
      phase: 0.5,
    },

    // -----------------------------------------------------------------------
    // Sky / atmosphere  @owner sky/atmosphere.js
    // -----------------------------------------------------------------------
    sky: {
      turbidity: 1.15,
      rayleigh: 2.10,
      mieCoefficient: 0.005,
      mieDirectionalG: 0.8,
      /** Ambient/IBL colours other systems tint against. Updated per frame. */
      zenithColor: new Color(0.18, 0.36, 0.72),
      horizonColor: new Color(0.72, 0.80, 0.92),
      groundColor: new Color(0.22, 0.24, 0.20),
      /** Overall scene ambient intensity, follows sun elevation. */
      ambientIntensity: 1.0,
      /** Star visibility 0..1 - fades in at dusk, killed by cloud cover. */
      starIntensity: 0.0,
    },

    // -----------------------------------------------------------------------
    // Clouds  @owner sky/clouds.js, driven by weather/weather.js
    // -----------------------------------------------------------------------
    clouds: {
      /** 0 = clear sky, 1 = fully socked in. The main artistic dial. */
      coverage: 0.07,
      /** Optical density - how dark the interiors get. */
      density: 0.55,
      /** Base altitude and vertical extent in world units. */
      altitude: 1500,
      thickness: 380,
      /** Advection speed multiplier; direction comes from wind. */
      speed: 1.0,
      /** Beer's-law absorption. Higher = moodier, more contrast. */
      absorption: 0.85,
      /** Detail-noise erosion strength - wispy edges. */
      erosion: 0.35,
      /** 0..1 storm-cloud darkening, raised by the weather system. */
      storminess: 0.0,
    },

    // -----------------------------------------------------------------------
    // Weather  @owner weather/weather.js
    // -----------------------------------------------------------------------
    weather: {
      /** Current WEATHER.* value. */
      type: WEATHER.CLEAR,
      /** What we are blending toward, and how far along (0..1). */
      target: WEATHER.CLEAR,
      transition: 1.0,
      /** Surface wetness 0..1 - darkens albedo and raises specular on ground/bark. */
      wetness: 0.0,
      rainIntensity: 0.0,
      snowIntensity: 0.0,
      /** Ground-level fog/mist density. */
      fogDensity: 0.0011,
      fogColor: new Color(0.72, 0.78, 0.86),
      /** Set to 1 on a strike, decays; lighting reads this for the flash. */
      lightning: 0.0,
      /** Extra petal spawn multiplier during petalStorm. */
      petalBoost: 1.0,
    },

    // -----------------------------------------------------------------------
    // Wind  @owner weather/wind.js
    // -----------------------------------------------------------------------
    wind: {
      /** Normalized horizontal heading. */
      direction: new Vector2(1, 0.35).normalize(),
      /** Base speed in m/s-ish units. Grass and branches scale off this. */
      strength: 3.2,
      /** 0..1 how much the gust envelope swings. */
      gustiness: 0.35,
      /** Current gust envelope value, already smoothed. Multiply strength by this. */
      gust: 1.0,
      /** High-frequency turbulence amount for leaf flutter. */
      turbulence: 0.4,
      /**
       * Sample the wind vector at a world position and time.
       * @owner weather/wind.js installs the real implementation.
       * @param {number} x @param {number} z @param {number} t
       * @param {{x:number,y:number,z:number}} out
       * @returns {{x:number,y:number,z:number}} world-space wind velocity
       */
      sample: (x, z, t, out = { x: 0, y: 0, z: 0 }) => {
        out.x = 0; out.y = 0; out.z = 0;
        return out;
      },
    },

    // -----------------------------------------------------------------------
    // Player  @owner player/controller.js
    // -----------------------------------------------------------------------
    player: {
      position: new Vector3(-28, 1.9, 0),
      velocity: new Vector3(),
      /** Horizontal look direction, for wind-relative audio and motion blur. */
      forward: new Vector3(1, 0, 0),
      /** 'walk' | 'fly' */
      mode: 'walk',
      grounded: true,
      sprinting: false,
      /** Eye height above ground when walking. */
      eyeHeight: 1.7,
      /** Ground height under the player, written by world/terrain.js. */
      groundHeight: 0,
    },

    // -----------------------------------------------------------------------
    // Quality  @owner core/quality.js
    // -----------------------------------------------------------------------
    quality: {
      tier: QUALITY.HIGH,
      /** Render scale 0.5..1. Adaptive resolution drives this down under load. */
      resolutionScale: 1.0,
      /** Whether adaptive resolution is allowed to move resolutionScale. */
      adaptive: true,
      shadows: true,
      shadowMapSize: 2048,
      cascadeCount: 3,
      /** Volumetric cloud raymarch steps; 0 disables volumetrics entirely. */
      cloudSteps: 48,
      cloudLightSteps: 6,
      /** Fraction of full res the clouds render at (0.25 / 0.5 / 1). */
      cloudResolution: 0.5,
      /** Grass blades per chunk and how far chunks stream out. */
      grassDensity: 1.0,
      grassDistance: 140,
      /** Max simultaneously simulated falling petals. */
      petalCount: 3000,
      ao: true,
      godRays: true,
      depthOfField: false,
      motionBlur: false,
      antialias: 'smaa',
    },

    // -----------------------------------------------------------------------
    // Perf  @owner core/quality.js
    // -----------------------------------------------------------------------
    perf: {
      fps: 60,
      frameMs: 16.7,
      /** Smoothed frame time the adaptive controller actually reacts to. */
      avgFrameMs: 16.7,
      drawCalls: 0,
      triangles: 0,
    },

    // -----------------------------------------------------------------------
    // Debug flags - toggled from the HUD, read by anyone who cares.
    // -----------------------------------------------------------------------
    debug: {
      wireframe: false,
      showStats: false,
      freezeFrustum: false,
      hideUI: false,
    },
  };
}

/**
 * Tiny synchronous event bus. Used for cross-system events that are too rare or
 * too discrete to poll for (lightning strikes, weather changes, quality changes).
 * Prefer reading state for anything continuous.
 */
export class EventBus {
  constructor() {
    this._handlers = new Map();
  }
  on(event, fn) {
    if (!this._handlers.has(event)) this._handlers.set(event, new Set());
    this._handlers.get(event).add(fn);
    return () => this.off(event, fn);
  }
  once(event, fn) {
    const off = this.on(event, (...args) => {
      off();
      fn(...args);
    });
    return off;
  }
  off(event, fn) {
    this._handlers.get(event)?.delete(fn);
  }
  emit(event, payload) {
    const set = this._handlers.get(event);
    if (!set) return;
    for (const fn of set) {
      try {
        fn(payload);
      } catch (err) {
        console.error(`[EventBus] handler for "${event}" threw:`, err);
      }
    }
  }
  clear() {
    this._handlers.clear();
  }
}

/** Well-known event names, so systems do not typo string literals at each other. */
export const EVENTS = Object.freeze({
  QUALITY_CHANGED: 'quality:changed',
  WEATHER_CHANGED: 'weather:changed',
  LIGHTNING_STRIKE: 'weather:lightning',
  PLAYER_MODE_CHANGED: 'player:mode',
  RESIZE: 'engine:resize',
  ASSETS_READY: 'assets:ready',
  LOAD_PROGRESS: 'assets:progress',
});
