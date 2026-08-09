/**
 * engine.js - renderer, scene graph, camera, lighting rig and cascaded shadows.
 *
 * This module owns the four things every other system depends on and nobody
 * else is allowed to touch:
 *
 *   1. The WebGL2 renderer  (context attributes, colour management, tone
 *      mapping, exposure, context-loss recovery).
 *   2. The scene graph + camera (depth-precision policy, render layers, resize).
 *   3. The light rig         (sun, moon, sky ambient, lightning flash), driven
 *      entirely from `state.sun` / `state.moon` / `state.sky` / `state.weather`.
 *   4. A hand-rolled 3-cascade shadow-map system.
 *
 * ---------------------------------------------------------------------------
 * WHY THE CSM IS BUILT THE WAY IT IS
 * ---------------------------------------------------------------------------
 * The constraint that shapes everything here lives inside `WebGLLights.setup()`
 * in three r180:
 *
 *     state.directionalShadowMap[ directionalLength ] = shadowMap;   // light index
 *     ...
 *     state.directionalShadowMap.length = numDirectionalShadows;     // truncated
 *
 * The shadow uniform arrays are written at the *light* index but truncated to
 * the *shadow* count, so shadow-casting directional lights MUST be the first N
 * directional lights in scene-traversal order or those arrays come out full of
 * holes. Everything below follows from that:
 *
 *   - Every light lives in one Group, added to the scene before any other
 *     system exists, so traversal order is fixed and known.
 *   - Directional order is: [cascade 0..N-1] [sun] [moon] [lightning].
 *   - The cascade lights are pure *shadow carriers*: their colour is black, so
 *     they contribute exactly zero radiance. Their only job is to make three
 *     allocate, render and upload N shadow maps and N shadow matrices.
 *   - Materials are patched (through a wrapped `onBeforeCompile`) so that the
 *     sun and moon are modulated by a single cascade-selected shadow lookup and
 *     the black carriers are compiled out of the light loop entirely.
 *
 * The failure mode if the patch ever misses a material is therefore *safe*:
 * that material renders with fully correct lighting and simply no shadows. It
 * can never double-count light or take shadows from the wrong direction.
 */

import * as THREE from 'three';
import { EVENTS, QUALITY } from './state.js';
import { clamp, clamp01, damp, lerp, smoothstep, makeRNG, TAU } from './math.js';

// ---------------------------------------------------------------------------
// Render layers. Objects live on layer 0 by default; ENABLE an extra layer on
// the object (`obj.layers.enable(LAYERS.GODRAY_OCCLUDER)`) rather than SETTING
// it, or the object vanishes from the main pass.
// ---------------------------------------------------------------------------
export const LAYERS = Object.freeze({
  /** Everything. Never disable this on the main camera. */
  DEFAULT: 0,
  /** Sky dome, sun/moon discs, stars. Depth-write off, drawn first. */
  SKY: 1,
  /** Silhouette casters for the god-ray occlusion pass (tree, terrain, hills). */
  GODRAY_OCCLUDER: 2,
  /** Transparent FX that must be excluded from depth-dependent passes. */
  TRANSPARENT_FX: 3,
  /** Editor-ish helpers. Off on the main camera unless explicitly enabled. */
  DEBUG: 9,
});

/** Engine-local bus events - `EVENTS` in state.js has no slot for these. */
export const ENGINE_EVENTS = Object.freeze({
  CONTEXT_LOST: 'engine:contextlost',
  CONTEXT_RESTORED: 'engine:contextrestored',
});

// ---------------------------------------------------------------------------
// Tuning. Everything the look depends on lives here rather than hiding as
// magic numbers three hundred lines down.
// ---------------------------------------------------------------------------

/**
 * Near/far. We deliberately do NOT use `logarithmicDepthBuffer`: on WebGL2
 * three implements it by writing `gl_FragDepth` from every fragment shader,
 * which disables early-Z rejection. On a 780M - where fill rate is the entire
 * budget - losing early-Z across a grass field costs far more than z-fighting
 * ever would, and it also fights the postprocessing depth buffer.
 *
 * A tuned near plane is enough. Depth resolution for a standard projection is
 * roughly  dz ~= z^2 * (1/near - 1/far) / 2^24 :
 *
 *     z =   10 m  ->  0.03 mm
 *     z =  100 m  ->  3.0  mm
 *     z = 1000 m  ->  0.30 m
 *     z = 4000 m  ->  4.8  m
 *
 * 3 mm at 100 m is far tighter than anything in this scene needs, and 4.8 m at
 * the 4 km view distance lands on a treeline that is already 95 % haze. The far
 * plane is pushed out to 8 km purely to give the sky dome somewhere to live.
 */
const CAMERA_NEAR = 0.2;
const CAMERA_FAR = 8000;

/** 22 mm on a 35 mm gauge: ~77 deg horizontal, ~48 deg vertical at 16:9.
 *  Driving FOV from a focal length gives Hor+ behaviour across aspect ratios
 *  for free, which is both physically sane and what players expect. */
const FILM_GAUGE = 35;
const FOCAL_LENGTH = 22;

/** Device-pixel-ratio ceiling. QualityManager scales further via resolutionScale. */
const MAX_PIXEL_RATIO = 1.5;

/** Cascade split distribution: 0 = uniform, 1 = logarithmic. */
const SPLIT_LAMBDA = 0.72;
/** Fraction of a cascade's depth range spent cross-fading into the next one. */
const CASCADE_BLEND = 0.1;
/** Ortho extent is grown slightly past the fitted sphere so PCF taps never
 *  wander off the map at the frustum corners. */
const CASCADE_MARGIN = 1.02;
/**
 * Tallest thing that can cast into a cascade from outside it - the sakura tree
 * is ~18 m, terrain ridges are the real ceiling. The distance the shadow camera
 * has to be pulled back is this height divided by the sun's elevation, so a low
 * sun automatically buys the long stand-off its long shadows need while noon
 * pays for almost nothing.
 */
const CASTER_HEIGHT = 60;
const CASTER_BACKOFF_MIN = 50;
const CASTER_BACKOFF_MAX = 420;

/**
 * Per-tier shadow behaviour. `maps` scales shadowMapSize per cascade, `cadence`
 * is the refresh interval in frames per cascade.
 *
 * Shadow distance is deliberately close to `state.quality.grassDistance`: past
 * the grass there is only terrain and a hazed-out treeline, and a sphere-fitted
 * far cascade grows with the square of its distance. Every metre of shadow
 * distance past the grass costs texel density everywhere inside it.
 */
const SHADOW_TIERS = {
  [QUALITY.LOW]: { distance: 100, taps: 4, maps: [1, 0.6, 0.5], cadence: [1, 3, 5] },
  [QUALITY.MEDIUM]: { distance: 140, taps: 6, maps: [1, 1, 0.75], cadence: [1, 2, 4] },
  [QUALITY.HIGH]: { distance: 190, taps: 9, maps: [1, 1, 0.75], cadence: [1, 1, 2] },
  [QUALITY.ULTRA]: { distance: 280, taps: 16, maps: [1, 1, 1], cadence: [1, 1, 2] },
};

/** Base PCF kernel radius in shadow texels, per cascade. */
const PCF_TEXEL_RADIUS = [2.6, 2.1, 1.6];

/** Auto-exposure. Partial adaptation: night stays night, it just stays legible. */
const EXPOSURE_REF_LUMINANCE = 1.6;
const EXPOSURE_ADAPT = 0.45;
const EXPOSURE_MIN = 0.55;
const EXPOSURE_MAX = 3.8;
const EXPOSURE_LAMBDA = 0.75;

/** Hemisphere gain relative to `state.sky.ambientIntensity`. */
const AMBIENT_GAIN = 0.92;
/** Deep-night floor, so a moonless midnight reads as blue rather than as a
 *  crash. Roughly four stops below noon once auto-exposure has settled. */
const STARLIGHT_FLOOR = 0.022;
/** How much of the key light bounces off the ground back into the ambient term. */
const GROUND_BOUNCE = 0.05;
/** Peak irradiance of a lightning flash, in the same units as sun intensity. */
const LIGHTNING_PEAK = 22;

// ---------------------------------------------------------------------------
// Module-scope scratch. Nothing in the per-frame path allocates.
// ---------------------------------------------------------------------------
const _vec = new THREE.Vector3();
const _center = new THREE.Vector3();
const _eye = new THREE.Vector3();
const _fwd = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0);
const _origin = new THREE.Vector3(0, 0, 0);
const _lightRot = new THREE.Matrix4();
const _lightRotInv = new THREE.Matrix4();
const _colA = new THREE.Color();
const _colB = new THREE.Color();
const _colSky = new THREE.Color();
const _colGround = new THREE.Color();

const luminance = (c) => c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722;

// GLSL fragments used by the material patch. Kept as escaped strings (not
// literal indented template text) so the surgery below is insensitive to how
// this file happens to be indented.
const TAB = '\t';
const T2 = '\t\t';

// ===========================================================================
// Engine
// ===========================================================================

export class Engine {
  /**
   * @param {{canvas: HTMLCanvasElement, state: object, bus: import('./state.js').EventBus}} opts
   */
  constructor({ canvas, state, bus }) {
    this.canvas = canvas;
    this.state = state;
    this.bus = bus;
    this.supported = false;
    this.contextLost = false;
    this.reason = '';

    // -- WebGL2 acquisition -------------------------------------------------
    // Grab the context ourselves so a missing WebGL2 reports cleanly instead of
    // surfacing as a three.js exception, and so the attributes are exactly what
    // this scene wants rather than three's defaults.
    let creationError = '';
    const onCreationError = (e) => {
      creationError = e.statusMessage || '';
    };
    canvas.addEventListener('webglcontextcreationerror', onCreationError, false);

    let gl = null;
    try {
      gl = canvas.getContext('webgl2', {
        alpha: false, // opaque canvas: no per-pixel blend against the page
        depth: true,
        stencil: false, // nothing uses stencil; saves depth-buffer bandwidth
        antialias: false, // MSAA is pure fill-rate cost; post does SMAA instead
        premultipliedAlpha: true,
        preserveDrawingBuffer: false,
        powerPreference: 'high-performance',
        failIfMajorPerformanceCaveat: false,
        // Skips a compositor sync per present - a real latency win on Windows,
        // and nothing here ever reads the drawing buffer back.
        desynchronized: true,
      });
    } catch (err) {
      creationError = String((err && err.message) || err);
    }
    canvas.removeEventListener('webglcontextcreationerror', onCreationError, false);

    if (!gl) {
      this.reason = creationError || 'WebGL2 context could not be created.';
      console.error('[Engine] WebGL2 unavailable.', this.reason);
      return;
    }

    try {
      this.renderer = new THREE.WebGLRenderer({ canvas, context: gl });
    } catch (err) {
      this.reason = String((err && err.message) || err);
      console.error('[Engine] WebGLRenderer construction failed:', err);
      return;
    }
    this.supported = true;

    // -- Renderer defaults --------------------------------------------------
    // NOTE: post/pipeline.js may legitimately move tone mapping into its own
    // final pass and switch the renderer to LinearSRGBColorSpace. These are the
    // standalone-correct defaults; whatever post establishes afterwards wins and
    // is never clobbered (see `_reapplyRendererState`).
    const renderer = this.renderer;
    renderer.debug.checkShaderErrors =
      typeof import.meta !== 'undefined' && import.meta.env ? !!import.meta.env.DEV : true;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.setClearColor(0x0a0d13, 1);
    renderer.info.autoReset = true;

    // BasicShadowMap on purpose: we never call three's `getShadow()` - the CSM
    // runs its own rotated-disc PCF - so the cheapest variant keeps three's
    // unused 17-tap kernel out of every compiled shader.
    renderer.shadowMap.enabled = !!state.quality.shadows;
    renderer.shadowMap.type = THREE.BasicShadowMap;
    renderer.shadowMap.autoUpdate = true; // per-cascade scheduling is ours

    this.maxPixelRatio = MAX_PIXEL_RATIO;
    this.exposure = 1.0;
    this.autoExposure = true;
    this._exposureSettled = false;

    // -- Scene --------------------------------------------------------------
    this.scene = new THREE.Scene();
    this.scene.name = 'SakuraRealm';
    this.scene.background = null; // sky/atmosphere.js owns the background
    this.scene.fog = null; // weather/atmosfx.js owns fog

    // -- Camera -------------------------------------------------------------
    this.camera = new THREE.PerspectiveCamera(45, 1, CAMERA_NEAR, CAMERA_FAR);
    this.camera.name = 'MainCamera';
    this.camera.filmGauge = FILM_GAUGE;
    this.camera.setFocalLength(FOCAL_LENGTH);
    this.camera.position.set(-28, 1.9, 0);
    this.camera.lookAt(0, 6.2, 0);
    this.camera.layers.enable(LAYERS.SKY);
    this.camera.layers.enable(LAYERS.GODRAY_OCCLUDER);
    this.camera.layers.enable(LAYERS.TRANSPARENT_FX);
    this._appliedFov = this.camera.fov;

    // -- Light rig ----------------------------------------------------------
    // Added before any other system exists, so the cascade carriers are
    // guaranteed to be directional lights 0..N-1 in traversal order.
    this.lightGroup = new THREE.Group();
    this.lightGroup.name = 'LightRig';
    this.scene.add(this.lightGroup);

    this._cascades = [];
    for (let i = 0; i < 3; i++) {
      const light = new THREE.DirectionalLight(0xffffff, 1);
      light.name = `CsmCascade${i}`;
      light.color.setRGB(0, 0, 0); // carries shadows only, never radiance
      light.castShadow = true;
      light.shadow.autoUpdate = false; // we schedule refreshes ourselves
      light.shadow.needsUpdate = true;
      light.shadow.intensity = 1; // strength is applied in our own shader
      light.target.name = `CsmCascade${i}Target`;
      this.lightGroup.add(light, light.target);
      this._cascades.push({
        light,
        shadow: light.shadow,
        mapSize: 1024,
        radius: 1,
        texelWorld: 1,
      });
    }

    this.sunLight = new THREE.DirectionalLight(0xffffff, 0);
    this.sunLight.name = 'SunLight';
    this.sunLight.castShadow = false;
    this.lightGroup.add(this.sunLight, this.sunLight.target);

    this.moonLight = new THREE.DirectionalLight(0xffffff, 0);
    this.moonLight.name = 'MoonLight';
    this.moonLight.castShadow = false;
    this.lightGroup.add(this.moonLight, this.moonLight.target);

    this.flashLight = new THREE.DirectionalLight(0xffffff, 0);
    this.flashLight.name = 'LightningLight';
    this.flashLight.castShadow = false;
    this.lightGroup.add(this.flashLight, this.flashLight.target);

    this.hemiLight = new THREE.HemisphereLight(0xffffff, 0xffffff, 1);
    this.hemiLight.name = 'SkyAmbient';
    this.lightGroup.add(this.hemiLight);

    // -- CSM bookkeeping ----------------------------------------------------
    this._csmVersion = 0;
    this._csmTaps = 9;
    this._cascadeCount = 3;
    this._shadowDistance = 220;
    this._cadence = [1, 1, 2];
    this._shadowsActive = true;
    this._shadowStrength = -1; // < 0 means "snap on the first frame"
    this._softness = 1;
    this._forceRefit = true;
    this._keyIsMoon = false;
    this._zUpFallback = false;

    this._csmUniforms = {
      /** xyz = cascade far distances in view space, w = active cascade count. */
      uCsmSplits: { value: new THREE.Vector4(20, 60, 220, 3) },
      /** x = fade start, y = fade end, z = strength, w = blend fraction. */
      uCsmFade: { value: new THREE.Vector4(180, 220, 1, CASCADE_BLEND) },
      /** xyz = PCF kernel radius in texels, per cascade. */
      uCsmRadius: { value: new THREE.Vector4(2.6, 2.1, 1.6, 0) },
    };

    /** Materials whose `onBeforeCompile` we have wrapped. */
    this._patched = new Set();
    this._patchScanFrame = -1000;
    this._foreignShadowWarned = false;
    this._onMaterialDisposed = (event) => {
      const material = event.target;
      this._patched.delete(material);
      material.removeEventListener('dispose', this._onMaterialDisposed);
    };

    this._lastLightFrame = -1;
    this._lastShadowFrame = -1;
    this._shadowFrameCounter = 0;

    this._flashRng = makeRNG(0x5a4b3c);
    this._flashDir = new THREE.Vector3(0.3, 0.9, 0.2).normalize();
    this._flashFrame = -1;
    this._prevLightning = 0;

    this._keyDir = new THREE.Vector3(0, 1, 0);
    this._keyColor = new THREE.Color(1, 1, 1);
    this._keyIntensity = 0;

    // Built from the live ShaderChunk, so it inherits whatever this exact three
    // build ships rather than a copy that can silently drift out of date.
    this._csmChunk = buildLightsChunk();
    this.onQualityChange(state.quality); // also builds the GLSL for this tier

    // -- Frame hook ---------------------------------------------------------
    // main.js does not put the Engine in its system list, so the per-frame hook
    // has to come from the renderer. `scene.onBeforeRender` fires once per
    // `renderer.render(scene, camera)` - after `scene.updateMatrixWorld()` and,
    // crucially, *before* `shadowMap.render()`, which is exactly the window the
    // cascades must be placed in. The frame guard keeps it to one update even
    // when the post stack renders the scene more than once.
    this.scene.onBeforeRender = (rendererArg, scene, camera) => {
      this._onBeforeSceneRender(camera);
    };

    // -- Context loss -------------------------------------------------------
    this._onContextLost = (event) => {
      event.preventDefault(); // opt in to a restore instead of a dead canvas
      this.contextLost = true;
      console.warn('[Engine] WebGL context lost.');
      this.bus.emit(ENGINE_EVENTS.CONTEXT_LOST, null);
    };
    this._onContextRestored = () => {
      this.contextLost = false;
      console.warn('[Engine] WebGL context restored - rebuilding GPU state.');
      this._reapplyRendererState();
      this.bus.emit(ENGINE_EVENTS.CONTEXT_RESTORED, null);
    };
    canvas.addEventListener('webglcontextlost', this._onContextLost, false);
    canvas.addEventListener('webglcontextrestored', this._onContextRestored, false);

    this._offQuality = bus.on(EVENTS.QUALITY_CHANGED, (q) =>
      this.onQualityChange(q || state.quality)
    );
    this._offLightning = bus.on(EVENTS.LIGHTNING_STRIKE, () => this._pickFlashDirection());
  }

  // =========================================================================
  // Public surface
  // =========================================================================

  /**
   * Runs the light rig and the cascade fit. Safe to call by hand; the renderer
   * hook calls it anyway and both paths are guarded on `state.time.frame`.
   */
  update(dt, state) {
    if (!this.supported) return;
    this._updateLights(dt, state);
    this._updateShadows(this.camera);
  }

  resize(width, height) {
    if (!this.supported) return;
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));

    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    const scale = clamp(this.state.quality.resolutionScale || 1, 0.4, 1);
    this.renderer.setPixelRatio(Math.min(dpr, this.maxPixelRatio) * scale);
    this.renderer.setSize(w, h, false);

    // If the stylesheet has not sized the canvas, fall back to explicit pixels
    // rather than rendering into a zero-sized element.
    if (this.canvas.clientWidth < 1 || this.canvas.clientHeight < 1) {
      this.canvas.style.width = `${w}px`;
      this.canvas.style.height = `${h}px`;
    }

    // Preserve any FOV offset another system dialled in (sprint kick, aim zoom)
    // while still re-deriving the base FOV from the fixed focal length.
    const offset = this.camera.fov - this._appliedFov;
    this.camera.aspect = w / h;
    this.camera.setFocalLength(FOCAL_LENGTH);
    this._appliedFov = this.camera.fov;
    this.camera.fov = this._appliedFov + offset;
    this.camera.updateProjectionMatrix();

    this._forceRefit = true;
  }

  onQualityChange(quality) {
    if (!this.supported) return;
    const q = quality || this.state.quality;
    const tier = SHADOW_TIERS[q.tier] || SHADOW_TIERS[QUALITY.HIGH];

    const wantShadows = !!q.shadows;
    // Zero cascades is a first-class configuration: it removes the carrier
    // lights from the light list AND makes the shader patch a pure no-op.
    const count = wantShadows ? clamp(Math.round(q.cascadeCount || 3), 1, 3) : 0;
    const base = clamp(Math.round(q.shadowMapSize || 2048), 256, 4096);

    const structureChanged =
      this.renderer.shadowMap.enabled !== wantShadows ||
      this._cascadeCount !== count ||
      this._csmTaps !== tier.taps;

    this.renderer.shadowMap.enabled = wantShadows;
    this._cascadeCount = count;
    this._csmTaps = tier.taps;
    this._shadowDistance = tier.distance;
    this._cadence = tier.cadence;

    for (let i = 0; i < this._cascades.length; i++) {
      const c = this._cascades[i];
      const active = i < count;
      // Toggling `visible` drops the light from the light list entirely, which
      // is what keeps unused cascades out of every compiled shader.
      c.light.visible = active;
      if (!active) {
        c.shadow.needsUpdate = false;
        continue;
      }
      // Round to a multiple of 128 - some drivers are noticeably happier with
      // aligned render-target dimensions, and it keeps the derived sizes tidy
      // (2048 -> 1536, 1024 -> 640/768) instead of landing on 614.
      const size = clamp(Math.round((base * tier.maps[i]) / 128) * 128, 256, 4096);
      if (size !== c.mapSize) {
        c.mapSize = size;
        c.shadow.mapSize.set(size, size);
        // three only allocates `shadow.map` when it is null, so a resize means
        // explicitly dropping the old target.
        if (c.shadow.map) {
          c.shadow.map.dispose();
          c.shadow.map = null;
        }
      }
      c.shadow.needsUpdate = true;
    }

    this._csmUniforms.uCsmSplits.value.w = count;
    this._forceRefit = true;

    // Always rebuilt - this is also the path that produces them for the very
    // first time, when nothing has "changed" yet. String building only.
    this._csmDefines = buildCsmDefines(count, tier.taps);
    this._csmPars = buildCsmPars(count, tier.taps);

    if (structureChanged) {
      this._csmVersion++;
      // The patch changes shader source but none of the parameters three
      // hashes, so materials have to be told their program is stale. The
      // wrapped `customProgramCacheKey` carries `_csmVersion`, which is what
      // makes the rebuilt key actually miss the program cache.
      for (const material of this._patched) material.needsUpdate = true;
    }
  }

  /**
   * Runs `fn` with a temporary camera layer mask, then restores it.
   * Allocation-free; for sky-only or occluder-only passes in post.
   * @param {number} mask raw 32-bit layer mask, e.g. `Engine.layerMask(LAYERS.SKY)`
   */
  withCameraLayers(mask, fn) {
    const prev = this.camera.layers.mask;
    this.camera.layers.mask = mask >>> 0;
    try {
      return fn(this.camera);
    } finally {
      this.camera.layers.mask = prev;
    }
  }

  /** Builds a layer mask containing exactly the listed layers.
   *  Uses a rest argument, so cache the result rather than calling it per frame. */
  static layerMask(...layers) {
    let m = 0;
    for (let i = 0; i < layers.length; i++) m |= 1 << layers[i];
    return m >>> 0;
  }

  /** Shows or hides everything on LAYERS.DEBUG. Off by default. */
  setDebugLayerVisible(visible) {
    if (visible) this.camera.layers.enable(LAYERS.DEBUG);
    else this.camera.layers.disable(LAYERS.DEBUG);
  }

  /** Number of shadow cascades currently rendering. 0 when shadows are off. */
  get cascadeCount() {
    return this.renderer && this.renderer.shadowMap.enabled ? this._cascadeCount : 0;
  }

  dispose() {
    if (this._offQuality) this._offQuality();
    if (this._offLightning) this._offLightning();
    if (this.canvas) {
      this.canvas.removeEventListener('webglcontextlost', this._onContextLost, false);
      this.canvas.removeEventListener('webglcontextrestored', this._onContextRestored, false);
    }
    if (this.scene) this.scene.onBeforeRender = () => {};

    for (const c of this._cascades || []) {
      if (c.shadow.map) {
        c.shadow.map.dispose();
        c.shadow.map = null;
      }
      c.light.dispose();
    }
    this.sunLight?.dispose();
    this.moonLight?.dispose();
    this.flashLight?.dispose();
    this.hemiLight?.dispose();
    if (this.lightGroup && this.scene) this.scene.remove(this.lightGroup);

    if (this._patched) {
      for (const material of this._patched) {
        material.removeEventListener('dispose', this._onMaterialDisposed);
      }
      this._patched.clear();
    }
    this.renderer?.dispose();
  }

  // =========================================================================
  // Frame hook
  // =========================================================================

  _onBeforeSceneRender(camera) {
    const state = this.state;
    const frame = state.time.frame;

    if (this._lastLightFrame !== frame) {
      this._lastLightFrame = frame;
      this._updateLights(state.time.delta, state);
      this._scanForNewMaterials(frame);
    }
    if (this._lastShadowFrame !== frame) {
      this._lastShadowFrame = frame;
      this._updateShadows(camera || this.camera);
    }
  }

  /**
   * Called after a context restore. Deliberately touches ONLY the things that
   * actually die with the GL context. Renderer settings such as `toneMapping`
   * and `outputColorSpace` are plain JS properties that survive context loss,
   * and post/pipeline.js may own them by now - resetting them here would
   * silently break the post stack after every driver hiccup.
   */
  _reapplyRendererState() {
    for (const c of this._cascades) {
      c.shadow.map = null; // GPU side is gone; let three allocate a fresh one
      c.shadow.needsUpdate = true;
    }
    this._forceRefit = true;
    this._exposureSettled = false;
  }

  // =========================================================================
  // LEAF 3 - the light rig
  // =========================================================================

  _updateLights(dt, state) {
    const step = clamp(dt || 0, 0, 0.1);
    const sun = state.sun;
    const moon = state.moon;
    const sky = state.sky;
    const weather = state.weather;

    // A body below the horizon cannot light the ground, whatever intensity the
    // celestial system reports.
    //
    // The fade window starts ABOVE the horizon on purpose. Physically it stands
    // in for atmospheric extinction, which really does cost the sun several
    // stops over its last few degrees. Practically it means this rig degrades
    // gracefully even if the celestial module clips its intensity hard at
    // elevation zero: without it, that clip would take the key light AND its
    // shadows from full strength to nothing in a single frame.
    const sunHorizon = smoothstep(-0.015, 0.06, sun.direction.y);
    const moonHorizon = smoothstep(-0.015, 0.06, moon.direction.y);

    // Thick cloud converts direct sun into skylight. Without this a storm keeps
    // razor-sharp sun shadows, which is the single most obvious "fake weather"
    // tell there is. Shadow softness and strength follow the same curve below.
    const coverage = clamp01(state.clouds ? state.clouds.coverage : 0);
    const overcast = smoothstep(0.42, 0.96, coverage);
    const directTransmit = lerp(1, 0.3, overcast);

    const sunI = Math.max(0, sun.intensity) * sunHorizon * directTransmit;
    const moonI = Math.max(0, moon.intensity) * moonHorizon * directTransmit;

    this.sunLight.color.copy(sun.color);
    this.sunLight.intensity = sunI;
    this._aimDirectional(this.sunLight, sun.direction);

    this.moonLight.color.copy(moon.color);
    this.moonLight.intensity = moonI;
    this._aimDirectional(this.moonLight, moon.direction);

    // --- which body owns the shadows --------------------------------------
    const sunWeight = luminance(sun.color) * sunI;
    const moonWeight = luminance(moon.color) * moonI;
    // Hysteresis: the challenger has to be clearly ahead, so the choice cannot
    // chatter through the minutes at dusk when the two are comparable.
    let wantMoon = this._keyIsMoon;
    if (this._keyIsMoon) {
      if (sunWeight > moonWeight * 1.3) wantMoon = false;
    } else if (moonWeight > sunWeight * 1.3) {
      wantMoon = true;
    }
    // Swapping the key body rotates every shadow in the world by up to 180
    // degrees in one frame. That is only invisible if nothing is casting at the
    // moment it happens, so the swap is *deferred*: shadow strength is driven
    // to zero first and the handover commits only once it has actually faded
    // out (see below). The whole exchange takes about half a second, during
    // which the sun is already below the horizon and contributing nothing.
    const handover = wantMoon !== this._keyIsMoon;

    const keySrc = this._keyIsMoon ? moon : sun;
    const keyWeight = this._keyIsMoon ? moonWeight : sunWeight;
    this._keyDir.copy(keySrc.direction);
    if (this._keyDir.lengthSq() < 1e-8) this._keyDir.set(0, 1, 0);
    else this._keyDir.normalize();
    this._keyColor.copy(keySrc.color);
    this._keyIntensity = this._keyIsMoon ? moonI : sunI;

    // --- lightning ---------------------------------------------------------
    const lightning = clamp01(weather ? weather.lightning : 0);
    // The weather system may fire the bus event, or may just spike the scalar.
    // Catch a rising edge either way; the frame guard stops a strike that does
    // both from picking two different directions.
    if (lightning > this._prevLightning + 0.25 && this._prevLightning < 0.2) {
      this._pickFlashDirection();
    }
    this._prevLightning = lightning;

    // A strike is a very short, very bright event. Squaring the envelope keeps
    // the decay tail from reading as a slow fluorescent flicker.
    const flash = lightning * lightning;
    this.flashLight.intensity = flash * LIGHTNING_PEAK;
    this.flashLight.color.setRGB(0.86, 0.92, 1.0);
    this._aimDirectional(this.flashLight, this._flashDir);

    // --- sky ambient -------------------------------------------------------
    // Upper-hemisphere irradiance is dominated by the whole band from zenith to
    // horizon, so weight the two rather than taking the zenith alone - a pure
    // zenith term makes golden hour far too blue.
    _colSky.copy(sky.zenithColor).multiplyScalar(0.55);
    _colA.copy(sky.horizonColor).multiplyScalar(0.45);
    _colSky.add(_colA);

    // Lower hemisphere: the sky's ground colour plus a little of whatever the
    // key light is currently doing to the field. This is what puts warmth under
    // the tree canopy at golden hour instead of flat grey.
    _colGround.copy(sky.groundColor);
    _colB.copy(this._keyColor).multiplyScalar(this._keyIntensity * GROUND_BOUNCE);
    _colGround.add(_colB);

    let ambient = Math.max(sky.ambientIntensity, 0) * AMBIENT_GAIN;
    ambient *= lerp(1, 1.35, overcast); // overcast scatters the lost sun back down
    ambient = Math.max(ambient, STARLIGHT_FLOOR);
    const flashAmbient = flash * LIGHTNING_PEAK * 0.16;

    _colSky.multiplyScalar(ambient);
    _colGround.multiplyScalar(ambient);
    // Metering luminance is captured BEFORE the flash is added. A strike must
    // not drag auto-exposure down, or every thunderclap is followed by a second
    // of the whole scene visibly pumping back up to where it was.
    const meterLum = luminance(_colSky) * 0.6 + luminance(_colGround) * 0.4;
    if (flashAmbient > 0) {
      // A strike lights the entire cloud deck, not just one direction.
      _colA.setRGB(0.8, 0.87, 1.0).multiplyScalar(flashAmbient);
      _colSky.add(_colA);
      _colGround.add(_colA);
    }
    // HemisphereLight multiplies colour by intensity; folding everything into
    // the colours keeps the two from drifting apart in two places.
    this.hemiLight.intensity = 1;
    this.hemiLight.color.copy(_colSky);
    this.hemiLight.groundColor.copy(_colGround);

    // --- shadow strength ---------------------------------------------------
    // Shadows are visible only in proportion to how far the key light beats the
    // ambient it competes with. This one term is what makes the sun/moon
    // handover invisible (both are dim then), keeps moonlight shadows present
    // but gentle, and stops overcast from carrying hard-edged shadows.
    //
    // This one DOES include the lightning term: a strike floods the scene from
    // the cloud deck and genuinely flattens shadow contrast while it lasts.
    const ambLum = luminance(_colSky) * 0.6 + luminance(_colGround) * 0.4;
    const keyContribution = keyWeight * 0.3;
    const contrast = keyContribution / (keyContribution + ambLum + 1e-4);
    let target = smoothstep(0.05, 0.28, contrast);
    target *= lerp(1, 0.62, overcast);
    target = clamp01(target);
    if (handover) target = 0; // fade out before rotating the whole shadow world

    // Short temporal filter (~0.1 s). Not for looks - the inputs come from four
    // other systems and any of them can step discontinuously (a weather state
    // change, an intensity clipped at the horizon). This turns a one-frame step
    // into a fade short enough that shadows never visibly outlive the light
    // that was casting them.
    if (this._shadowStrength < 0) this._shadowStrength = target;
    else this._shadowStrength = damp(this._shadowStrength, target, 10, step);
    const strength = this._shadowStrength;

    // Commit a deferred handover only once nothing is left to pop.
    if (handover && strength < 0.05) {
      this._keyIsMoon = wantMoon;
      this._forceRefit = true;
    }
    this._csmUniforms.uCsmFade.value.z = strength;
    this._shadowsActive = this.renderer.shadowMap.enabled && strength > 0.02;
    this._softness = lerp(1, 1.75, overcast);

    // Lights are moved after `scene.updateMatrixWorld()` has already run this
    // frame, so the rig has to refresh its own world matrices before three
    // reads them for the shadow pass.
    this.lightGroup.updateMatrixWorld(true);

    this._updateExposure(step, sunI, moonI, sun, moon, meterLum);
  }

  /** Points a directional light along `dir` (unit, pointing toward the body). */
  _aimDirectional(light, dir) {
    // Only the position-minus-target difference matters for a directional
    // light; a fixed 1 km stand-off keeps the numbers well conditioned.
    light.position.set(dir.x * 1000, dir.y * 1000, dir.z * 1000);
    light.target.position.set(0, 0, 0);
  }

  _pickFlashDirection() {
    if (this._flashFrame === this.state.time.frame) return;
    this._flashFrame = this.state.time.frame;
    const rng = this._flashRng;
    const azimuth = rng() * TAU;
    // Strikes read best coming from high up and off to one side.
    const elevation = lerp(0.62, 1.25, rng());
    const c = Math.cos(elevation);
    this._flashDir
      .set(Math.cos(azimuth) * c, Math.sin(elevation), Math.sin(azimuth) * c)
      .normalize();
  }

  _updateExposure(dt, sunI, moonI, sun, moon, ambLum) {
    if (!this.autoExposure) {
      this.renderer.toneMappingExposure = this.exposure;
      return;
    }
    // Estimate the irradiance a landscape of mixed slopes actually receives.
    // 0.3 is the mean cos-weighted N.L over such a surface.
    const keyLum =
      luminance(sun.color) * sunI * clamp01(sun.direction.y * 1.6) +
      luminance(moon.color) * moonI * clamp01(moon.direction.y * 1.6);
    const sceneLum = Math.max(keyLum * 0.3 + ambLum, 1e-4);

    // PARTIAL adaptation. Full auto-exposure would make midnight look like
    // noon; the fractional exponent keeps roughly three of the eight stops
    // between them, which is what "your eyes adjusted" actually looks like.
    // Lightning is deliberately excluded from the metering - the eye does not
    // adapt in 200 ms, and letting a flash blow out is the entire point.
    const target = clamp(
      Math.pow(EXPOSURE_REF_LUMINANCE / sceneLum, EXPOSURE_ADAPT),
      EXPOSURE_MIN,
      EXPOSURE_MAX
    );

    if (!this._exposureSettled) {
      this.exposure = target;
      this._exposureSettled = true;
    } else {
      this.exposure = damp(this.exposure, target, EXPOSURE_LAMBDA, dt);
    }
    this.renderer.toneMappingExposure = this.exposure;
  }

  // =========================================================================
  // LEAF 4 - cascaded shadow maps
  // =========================================================================

  _updateShadows(camera) {
    const count = this._cascadeCount;
    if (!this.renderer.shadowMap.enabled || count === 0) return;

    const frameIndex = this._shadowFrameCounter++;

    if (!this._shadowsActive) {
      // Nothing would be visible anyway: park every cascade and skip the depth
      // passes entirely. This reclaims most of the shadow budget at night and
      // in heavy overcast, which is exactly when the clouds want it back.
      for (let i = 0; i < count; i++) this._cascades[i].shadow.needsUpdate = false;
      this._forceRefit = true; // refit from scratch when they come back
      return;
    }
    if (this.state.debug && this.state.debug.freezeFrustum) return;

    camera.updateMatrixWorld();
    const proj = camera.projectionMatrix.elements;
    // Recover the frustum half-angles straight from the projection matrix so a
    // FOV change made by anyone else is picked up automatically.
    const tanV = 1 / proj[5];
    const tanH = 1 / proj[0];
    const a2 = tanH * tanH + tanV * tanV;

    const near = camera.near;
    const far = Math.min(this._shadowDistance, camera.far);

    // Practical split scheme (Zhang et al. 2006): blend the logarithmic
    // distribution, which is optimal for a perspective frustum, with the
    // uniform one, which stops the first cascade collapsing to a few
    // centimetres and wasting a whole 2k map on the player's shoes.
    const splits = this._csmUniforms.uCsmSplits.value;
    for (let i = 1; i <= count; i++) {
      const f = i / count;
      const d = lerp(near + (far - near) * f, near * Math.pow(far / near, f), SPLIT_LAMBDA);
      if (i === 1) splits.x = d;
      else if (i === 2) splits.y = d;
      else splits.z = d;
    }
    // Collapse unused slots onto the last real split so the shader's comparison
    // chain still terminates correctly at 1 or 2 cascades.
    if (count < 2) splits.y = splits.x;
    if (count < 3) splits.z = splits.y;

    const lastSplit = splits.z;
    const fade = this._csmUniforms.uCsmFade.value;
    fade.x = lastSplit * 0.82; // shadows begin dissolving
    fade.y = lastSplit; // fully gone
    fade.w = CASCADE_BLEND;

    // Light-space basis, built as a pure rotation about the origin so the texel
    // grid we snap to does not itself move with the cascade centre. That
    // circular dependency is the classic cause of shadow-edge shimmer.
    this._chooseShadowUp(this._keyDir);
    _lightRot.lookAt(this._keyDir, _origin, _up);
    _lightRot.setPosition(0, 0, 0);
    _lightRotInv.copy(_lightRot).invert();

    camera.getWorldDirection(_fwd);

    const radiusUni = this._csmUniforms.uCsmRadius.value;
    // A smaller tap budget needs a tighter kernel or the dither turns to noise.
    const tapScale = Math.sqrt(this._csmTaps / 9);

    for (let i = 0; i < count; i++) {
      const c = this._cascades[i];
      // Each cascade starts slightly *before* its split so it fully covers the
      // band in which the previous cascade cross-fades into it.
      const sliceNear =
        i === 0 ? near : (i === 1 ? splits.x : splits.y) * (1 - CASCADE_BLEND);
      const sliceFar = i === 0 ? splits.x : i === 1 ? splits.y : splits.z;

      // Tightest sphere around the slice's eight corners. A sphere is invariant
      // under camera rotation, so the cascade's *size* never changes as the
      // player looks around - only its centre moves, and that we snap.
      let cz = ((sliceNear + sliceFar) * (a2 + 1)) / 2;
      let radius;
      if (cz >= sliceFar) {
        // Centre would land past the far plane: the far rectangle's
        // circumcircle is then the minimal enclosing sphere.
        cz = sliceFar;
        radius = sliceFar * Math.sqrt(a2);
      } else {
        const dz = cz - sliceNear;
        radius = Math.sqrt(sliceNear * sliceNear * a2 + dz * dz);
      }
      radius *= CASCADE_MARGIN;
      c.radius = radius;

      const texel = (2 * radius) / c.mapSize;
      c.texelWorld = texel;

      // Distant cascades are refreshed at a lower rate, staggered so two of
      // them never land on the same frame.
      const cadence = Math.max(1, this._cadence[i] | 0);
      const due = this._forceRefit || !c.shadow.map || (frameIndex + i) % cadence === 0;

      if (!due) {
        // Leave the light - and therefore `shadow.matrix` - exactly where it
        // was, so the stale map stays consistent with the coordinates the
        // vertex shader generates. Freezing the transform is what makes
        // reduced-rate cascades *correct* rather than merely cheap.
        c.shadow.needsUpdate = false;
      } else {
        c.shadow.needsUpdate = true;

        // Cascade centre: on the view axis, cz metres ahead of the camera.
        _center.copy(camera.position).addScaledVector(_fwd, cz);

        // Snap to the light-space texel grid. Without this the shadow edges
        // crawl by a fraction of a texel on every frame the camera moves,
        // which reads as the whole world quietly boiling.
        _vec.copy(_center).applyMatrix4(_lightRotInv);
        _vec.x = Math.floor(_vec.x / texel) * texel;
        _vec.y = Math.floor(_vec.y / texel) * texel;
        _vec.z = Math.floor(_vec.z / texel) * texel;
        _center.copy(_vec).applyMatrix4(_lightRot);

        // Stand-off: far enough back that anything CASTER_HEIGHT metres above
        // the cascade still projects into it. Divided by the light's elevation,
        // so a sunset gets a long stand-off and noon gets a short one.
        const back =
          radius +
          clamp(
            CASTER_HEIGHT / Math.max(this._keyDir.y, 0.14),
            CASTER_BACKOFF_MIN,
            CASTER_BACKOFF_MAX
          );
        _eye.copy(_center).addScaledVector(this._keyDir, back);

        c.light.position.copy(_eye);
        c.light.target.position.copy(_center);

        const cam = c.shadow.camera;
        cam.up.copy(_up);
        cam.left = -radius;
        cam.right = radius;
        cam.top = radius;
        cam.bottom = -radius;
        cam.near = 1;
        cam.far = back + radius + 40;
        // three only calls this when it first allocates the map, so changing
        // the extents means updating the projection ourselves.
        cam.updateProjectionMatrix();

        // Bias policy: lean on the NORMAL offset and keep the depth offset
        // small. Normal offset pushes the receiver along its own normal, which
        // removes acne without sliding the shadow away from the contact point;
        // depth offset does slide it, and a large one is exactly what
        // peter-panning looks like. Both scale with texel size, so the
        // behaviour is identical at every cascade and every map resolution.
        c.shadow.normalBias = texel * 1.75 + 0.01;
        // Depth bias is in NDC units. Depth is linear across an orthographic
        // range, so a world offset converts by dividing through that range.
        // three's shadow maps are 32-bit RGBA-packed, so quantisation
        // contributes nothing here - this term only mops up slope aliasing the
        // normal offset misses.
        c.shadow.bias = -(texel * 0.8 + 0.008) / (cam.far - cam.near);
      }

      const r = PCF_TEXEL_RADIUS[i] * this._softness * tapScale;
      if (i === 0) radiusUni.x = r;
      else if (i === 1) radiusUni.y = r;
      else radiusUni.z = r;
    }

    // Park any cascade this tier does not use.
    for (let i = count; i < this._cascades.length; i++) {
      this._cascades[i].shadow.needsUpdate = false;
    }

    this._forceRefit = false;
    this.lightGroup.updateMatrixWorld(true);
  }

  /**
   * Picks the shadow camera's up vector. A `lookAt` degenerates when the view
   * direction is parallel to up, which happens every time the sun crosses the
   * zenith. Hysteresis keeps it from flapping between the two bases.
   */
  _chooseShadowUp(dir) {
    const vertical = Math.abs(dir.y);
    if (this._zUpFallback) {
      if (vertical < 0.985) this._zUpFallback = false;
    } else if (vertical > 0.995) {
      this._zUpFallback = true;
    }
    if (this._zUpFallback) _up.set(0, 0, 1);
    else _up.set(0, 1, 0);
  }

  // =========================================================================
  // Material patching
  // =========================================================================

  /**
   * Walks the scene occasionally and wraps every material's `onBeforeCompile`
   * so the CSM shadow term reaches it. CONTRACTS mandates one material per
   * visual family, so in practice this settles within the first second and
   * costs a few microseconds per scan afterwards.
   */
  _scanForNewMaterials(frame) {
    if (!this.renderer.shadowMap.enabled || !this._csmChunk) return;
    // Scan often while the world streams in, rarely once it is stable.
    const interval = frame < 240 ? 4 : 20;
    if (frame - this._patchScanFrame < interval) return;
    this._patchScanFrame = frame;

    let foreignShadowCasters = 0;
    this.scene.traverse((object) => {
      if (object.isLight) {
        if (object.isDirectionalLight && object.castShadow && object.parent !== this.lightGroup) {
          foreignShadowCasters++;
        }
        return;
      }
      const material = object.material;
      if (!material) return;
      if (Array.isArray(material)) {
        for (let i = 0; i < material.length; i++) this._patchMaterial(material[i]);
      } else {
        this._patchMaterial(material);
      }
    });

    if (foreignShadowCasters > 0 && !this._foreignShadowWarned) {
      this._foreignShadowWarned = true;
      console.warn(
        '[Engine] Another system added a shadow-casting DirectionalLight. three ' +
          'indexes its shadow uniform arrays by LIGHT order but truncates them to ' +
          'the SHADOW count, so the CSM cascades must stay the first directional ' +
          'lights - that light will corrupt the shadow arrays. Set castShadow = false.'
      );
    }
  }

  _patchMaterial(material) {
    if (!material || material.isRawShaderMaterial) return;

    if (this._patched.has(material)) {
      // Another author may have assigned onBeforeCompile after we wrapped it.
      if (material.onBeforeCompile === material.userData.__csmHook) return;
      this._patched.delete(material);
    }

    // Defend against a system that reassigns onBeforeCompile every frame: we
    // would otherwise re-wrap (and force a recompile) forever.
    const attempts = (material.userData.__csmWraps || 0) + 1;
    material.userData.__csmWraps = attempts;
    if (attempts > 4) {
      if (attempts === 5) {
        console.warn(
          `[Engine] "${material.name || material.type}" keeps replacing ` +
            'onBeforeCompile; giving up on CSM for it (it will render unshadowed).'
        );
      }
      return;
    }

    const previous = material.onBeforeCompile;
    const engine = this;
    const hook = function (shader, renderer) {
      if (previous) previous.call(this, shader, renderer);
      engine._injectCsm(shader);
    };
    material.onBeforeCompile = hook;
    material.userData.__csmHook = hook;

    // The patch changes shader source but none of the parameters three hashes,
    // so the cache key has to carry the CSM configuration or a quality change
    // would silently hand back the previously compiled program.
    if (!material.userData.__csmKeyWrapped) {
      material.userData.__csmKeyWrapped = true;
      const prevKey = material.customProgramCacheKey;
      material.customProgramCacheKey = function () {
        const base = prevKey ? prevKey.call(this) : '';
        return `${base}|csm${engine._csmVersion}`;
      };
      material.addEventListener('dispose', this._onMaterialDisposed);
    }

    this._patched.add(material);
    material.needsUpdate = true;
  }

  /** Applies the CSM surgery to one shader. No-ops safely on unlit shaders. */
  _injectCsm(shader) {
    const src = shader.fragmentShader;
    if (typeof src !== 'string') return;
    if (!this._csmChunk || !this._csmPars || !this._csmDefines) return;
    if (src.indexOf('#include <lights_fragment_begin>') === -1) return;
    // A hand-written lit shader that pulls in the light loop but not the shadow
    // declarations has nowhere to hang the helpers; leave it completely alone
    // rather than emit GLSL that will not compile.
    if (src.indexOf('#include <shadowmap_pars_fragment>') === -1) return;

    const out = src
      .replace('#include <lights_fragment_begin>', this._csmChunk)
      .replace(
        '#include <shadowmap_pars_fragment>',
        `#include <shadowmap_pars_fragment>\n${this._csmPars}`
      );

    shader.fragmentShader = `${this._csmDefines}\n${out}`;
    shader.uniforms.uCsmSplits = this._csmUniforms.uCsmSplits;
    shader.uniforms.uCsmFade = this._csmUniforms.uCsmFade;
    shader.uniforms.uCsmRadius = this._csmUniforms.uCsmRadius;
  }
}

// ===========================================================================
// GLSL generation
// ===========================================================================

function buildCsmDefines(cascades, taps) {
  return (
    `#define CSM_CASCADES ${cascades}\n` +
    // Directional lights [0, N) are the black cascade carriers; [N, N+2) are
    // the sun and the moon, which the CSM shadows; N+2 onwards (the lightning
    // flash, and anything a future system adds) is left unshadowed.
    `#define CSM_SHADOW_END ${cascades + 2}\n` +
    `#define CSM_TAPS ${taps}`
  );
}

/**
 * PCF kernel + per-cascade wrappers + the cascade selection chain.
 *
 * Sampler arrays cannot be indexed dynamically in GLSL ES 3.00, so the three
 * cascade lookups are emitted as separate functions with literal indices rather
 * than a loop.
 */
function buildCsmPars(cascades, taps) {
  // Vogel disc: a golden-angle spiral is near-optimally distributed for *any*
  // sample count, so the tables for 4 / 6 / 9 / 16 taps are all prefixes of one
  // sequence and the kernel keeps the same shape across every quality tier.
  const golden = Math.PI * (3 - Math.sqrt(5));
  // Emit `a - b` / `a + b` with the sign folded into the operator rather than
  // letting negative literals produce `- -0.5`, which is legal but noisy.
  const term = (coef, operand) =>
    `${coef < 0 ? '-' : '+'} ${Math.abs(coef).toFixed(5)} * ${operand}`;
  let tapCode = '';
  for (let i = 0; i < taps; i++) {
    const r = Math.sqrt((i + 0.5) / taps);
    const theta = i * golden;
    const px = Math.cos(theta) * r;
    const py = Math.sin(theta) * r;
    tapCode +=
      `${T2}sum += texture2DCompare( map, uv + vec2( ` +
      `${px.toFixed(5)} * rot.x ${term(-py, 'rot.y')}, ` +
      `${px.toFixed(5)} * rot.y ${term(py, 'rot.x')} ) * texel, z );\n`;
  }

  let cascadeFns = '';
  for (let i = 0; i < cascades; i++) {
    const swizzle = 'xyz'[i];
    cascadeFns +=
      `${TAB}float csmCascade${i}( const in vec2 rot ) {\n` +
      `${T2}return csmSample( directionalShadowMap[ ${i} ],\n` +
      `${T2}\tdirectionalLightShadows[ ${i} ].shadowMapSize,\n` +
      `${T2}\tvDirectionalShadowCoord[ ${i} ],\n` +
      `${T2}\tdirectionalLightShadows[ ${i} ].shadowBias,\n` +
      `${T2}\tuCsmRadius.${swizzle}, rot );\n` +
      `${TAB}}\n\n`;
  }

  // Selection chain, generated for the active cascade count so LOW carries no
  // dead branches. Blending happens on the near side of each split, and each
  // cascade was fitted to cover that band (see `sliceNear` above), so the
  // second lookup is always inside its own map.
  const split = ['uCsmSplits.x', 'uCsmSplits.y', 'uCsmSplits.z'];
  let chain = '';
  for (let i = 0; i < cascades; i++) {
    const last = i === cascades - 1;
    if (last && i === 0) {
      chain += `${T2}s = csmCascade0( rot );\n`;
    } else if (last) {
      chain += `${T2}} else {\n${T2}\ts = csmCascade${i}( rot );\n${T2}}\n`;
    } else {
      const open = i === 0 ? `${T2}if` : `${T2}} else if`;
      chain +=
        `${open} ( viewDepth < ${split[i]} ) {\n` +
        `${T2}\ts = csmCascade${i}( rot );\n` +
        `${T2}\tfloat t${i} = smoothstep( ${split[i]} * ( 1.0 - uCsmFade.w ), ${split[i]}, viewDepth );\n` +
        `${T2}\tif ( t${i} > 0.0 ) s = mix( s, csmCascade${i + 1}( rot ), t${i} );\n`;
    }
  }

  return `
#if defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS >= CSM_CASCADES ) && ( CSM_CASCADES > 0 )

${TAB}uniform vec4 uCsmSplits;
${TAB}uniform vec4 uCsmFade;
${TAB}uniform vec4 uCsmRadius;

${TAB}/* Rotated-disc PCF. three's shadow maps are RGBA-packed depth with NEAREST
${TAB}   filtering, so every tap costs an unpack and there is no hardware PCF to
${TAB}   lean on. A golden-angle disc with a per-pixel rotation gets a far
${TAB}   smoother penumbra out of ${taps} taps than any fixed box kernel of the
${TAB}   same cost, and it degenerates gracefully as the tap budget shrinks. */
${TAB}float csmSample( sampler2D map, vec2 mapSize, vec4 coord, float bias, float radius, const in vec2 rot ) {

${T2}vec3 c = coord.xyz / coord.w;
${T2}float z = c.z + bias;
${T2}if ( z > 1.0 || z < 0.0 ) return 1.0;

${T2}/* Outside this cascade's map: unshadowed. The 2% ortho margin means valid
${T2}   fragments never reach here, so this only catches the sub-texel sliver
${T2}   that texel snapping can push past the edge. */
${T2}vec2 e = abs( c.xy - 0.5 );
${T2}if ( max( e.x, e.y ) > 0.5 ) return 1.0;

${T2}vec2 uv = c.xy;
${T2}vec2 texel = radius / mapSize;
${T2}float sum = 0.0;
${tapCode}${T2}return sum * ( 1.0 / float( CSM_TAPS ) );

${TAB}}

${cascadeFns}${TAB}float csmGetShadow( const in float viewDepth ) {

${T2}/* Interleaved gradient noise: a spatially smooth per-pixel phase that is
${T2}   fixed in screen space, so the dither never crawls the way a
${T2}   time-varying rotation would. */
${T2}float ign = fract( 52.9829189 * fract( dot( gl_FragCoord.xy, vec2( 0.06711056, 0.00583715 ) ) ) );
${T2}float ang = ign * 6.2831853;
${T2}vec2 rot = vec2( cos( ang ), sin( ang ) );

${T2}float s = 1.0;
${chain}
${T2}/* Dissolve the last cascade into full light before it runs out of map, so
${T2}   there is no hard line drawn across the field at the shadow distance. */
${T2}float fade = 1.0 - smoothstep( uCsmFade.x, uCsmFade.y, viewDepth );
${T2}return mix( 1.0, s, uCsmFade.z * fade );

${TAB}}

#endif
`;
}

/**
 * Rewrites `lights_fragment_begin`'s directional-light loop:
 *
 *   - directional lights below CSM_CASCADES (the black shadow carriers) are
 *     compiled out completely, which also removes three's own `getShadow()`
 *     call and its texture taps;
 *   - the sun and the moon are modulated by one cascade-selected lookup that is
 *     evaluated once per fragment, before the loop.
 *
 * The surgery is done on the live `THREE.ShaderChunk` rather than on a copied
 * literal, so it inherits whatever this exact three build ships. Every step is
 * asserted; if any marker moves in a future version the whole patch backs out
 * and the scene renders correctly but unshadowed.
 *
 * @returns {string|null} the rewritten chunk, or null if it could not be built.
 */
function buildLightsChunk() {
  const chunk = THREE.ShaderChunk.lights_fragment_begin;
  const startMarker = '#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )';
  const endMarker = '#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )';

  if (typeof chunk !== 'string') return null;
  const start = chunk.indexOf(startMarker);
  const end = chunk.indexOf(endMarker);
  if (start === -1 || end === -1 || end <= start) {
    console.warn('[Engine] lights_fragment_begin has an unexpected shape; CSM disabled.');
    return null;
  }

  // Operate only on the directional-light region: `RE_Direct(` also appears in
  // the point and spot blocks, and we must not touch those.
  let region = chunk.slice(start, end);

  // 1. Evaluate the cascade shadow once, before the unrolled loop.
  const guarded =
    `${startMarker}\n` +
    `${TAB}float csmShadowFactor = 1.0;\n` +
    `${TAB}#if defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS >= CSM_CASCADES ) && ( CSM_CASCADES > 0 )\n` +
    `${TAB}if ( receiveShadow ) csmShadowFactor = csmGetShadow( - geometryPosition.z );\n` +
    `${TAB}#endif`;
  region = region.replace(startMarker, guarded);

  // 2. Open a per-index guard so the carrier lights emit no code at all.
  //    `UNROLLED_LOOP_INDEX` is substituted with a literal by three's loop
  //    unroller, so this resolves entirely in the preprocessor.
  const infoLine = 'directionalLight = directionalLights[ i ];';
  if (region.indexOf(infoLine) === -1) {
    console.warn('[Engine] directional light loop not recognised; CSM disabled.');
    return null;
  }
  region = region.replace(
    infoLine,
    `#if ( UNROLLED_LOOP_INDEX >= CSM_CASCADES )\n${T2}${infoLine}`
  );

  // 3. Replace three's per-light shadow lookup with our cascade result. The
  //    original block only exists for indices < NUM_DIR_LIGHT_SHADOWS - exactly
  //    the carriers, which no longer emit code - so this both removes the
  //    redundant taps and re-targets the shadow onto the sun and moon.
  const shadowBlock =
    /#if defined\( USE_SHADOWMAP \) && \( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS \)[\s\S]*?#endif/;
  if (!shadowBlock.test(region)) {
    console.warn('[Engine] directional shadow block not recognised; CSM disabled.');
    return null;
  }
  region = region.replace(
    shadowBlock,
    `#if ( UNROLLED_LOOP_INDEX < CSM_SHADOW_END )\n` +
      `${T2}directLight.color *= csmShadowFactor;\n` +
      `${T2}#endif`
  );

  // 4. Close the guard after the shading call, just inside the loop body.
  const closer = new RegExp(`(RE_Direct\\([^;]*;)\\n${TAB}\\}`);
  if (!closer.test(region)) {
    console.warn('[Engine] directional RE_Direct call not recognised; CSM disabled.');
    return null;
  }
  region = region.replace(closer, `$1\n${T2}#endif\n${TAB}}`);

  return chunk.slice(0, start) + region + chunk.slice(end);
}
