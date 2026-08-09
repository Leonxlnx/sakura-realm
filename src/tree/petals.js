/**
 * tree/petals.js - falling sakura petals: aerodynamics, rendering, lifecycle.
 *
 * A petal does not drop, it *flutters*. The whole system is built around that:
 *
 *  - Forces come from the thin-plate cross-flow model. The aerodynamic force on a
 *    flat plate acts along the plate NORMAL with magnitude ~ |u_n|·u_n, where u_n
 *    is the component of air-relative velocity along that normal. Because that
 *    force is not aligned with the velocity, lift falls out for free: a tilted
 *    petal darts sideways, an edge-on petal knifes down and accelerates, a
 *    face-on petal stalls and hangs. The quadratic drag is integrated with its
 *    closed-form solution u' = u / (1 + k|u|dt), which is unconditionally stable - 
 *    explicit Euler rings and then explodes at these coefficients and a 20 fps dt.
 *
 *  - Orientation is a rocking + tumbling model whose RATE scales with air-relative
 *    speed (f ~ St·U/L, the Strouhal scaling that governs real fluttering plates).
 *    True St ~ 0.2 puts a 6 cm petal at ~4 Hz, which reads as buzzy on screen, so
 *    we keep the scaling law and choose the constant for the look. The trajectory
 *    is still emergent, never scripted: the zig-zag, the stall, the sideways glide
 *    and the terminal speed all come out of the force model.
 *
 *  - Turbulence is an ABC (Arnold-Beltrami-Childress) flow: exactly
 *    divergence-free - a genuine curl field - for 12 sines instead of the 12
 *    simplex evaluations that differentiating noise into a curl would cost. It is
 *    evaluated in the frame drifting with the mean wind (Taylor's frozen
 *    turbulence hypothesis), so eddies travel downwind instead of being nailed to
 *    the world and strobing as petals blow through them.
 *
 * Everything lives in flat typed arrays; nothing allocates after init().
 */

import * as THREE from 'three';
import { EVENTS, WEATHER, QUALITY } from '../core/state.js';
import { TAU, clamp, clamp01, smoothstep, makeRNG, noise } from '../core/math.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GRAVITY = 9.81;

/**
 * Broadside terminal velocity, m/s - the single tuned number in the force model.
 * The drag coefficient follows from it (mg = ½ρC_dAv² ⇒ k = g / v_term²) and is
 * independent of petal size, because mass and area both scale with area. A
 * fluttering petal averages ~2.6x this since it spends much of each cycle
 * edge-on, which puts the on-screen descent near 0.7 m/s: slow enough to read as
 * sakura, fast enough not to look like it is falling through syrup.
 */
const TERMINAL_FALL = 0.28;
const DRAG_N = GRAVITY / (TERMINAL_FALL * TERMINAL_FALL);
const DRAG_T_RATIO = 0.04;         // skin friction along the plate vs. broadside

const FLUTTER_HZ = 0.8;            // rocking frequency at nominal descent speed
const FLUTTER_HZ_MAX = 3.4;        // clamp so gust-blown petals do not strobe
const REF_LENGTH = 0.065;          // petal length the flutter constant is quoted at

// Petal shape, in units of petal length. y = 0 at the stem, 1 at the tip.
const HALF_W = 0.38;
const HALF_W_NORM = 0.7654;        // peak of the raw profile, folded out
const NOTCH_DEPTH = 0.135;         // the sakura cleft
const NOTCH_WIDTH = 0.17;
const TEX_X_SPAN = HALF_W * 2 * 1.18;
const TEX_Y_SPAN = 1.2;
const TEX_Y_MIN = -0.09;
const PIVOT_Y = 0.55;              // local origin at the centroid: the point physics moves
const MESH_OVERSHOOT = 1.1;        // mesh sits just outside the alpha silhouette
const MESH_NOTCH = 0.78;

const FREE = 0, FALLING = 1, SLIDING = 2, RESTING = 3;

const STRIDE = 16;                 // floats per instance
const WIND_STRIDE = 6;             // frames between wind re-samples per petal
const GROUND_STRIDE = 8;           // frames between terrain re-samples per petal
const SORT_BUCKETS = 192;
const FADE_OUT = 2.6;              // seconds a spent ground petal takes to vanish
const FADE_IN = 0.35;

const TIER = {
  [QUALITY.LOW]:    { dist: 55,  simFar: 22, sort: false, detail: 0, oct: 1, linger: 7,  lodPx: 1e9, aniso: 1 },
  [QUALITY.MEDIUM]: { dist: 78,  simFar: 34, sort: false, detail: 1, oct: 2, linger: 11, lodPx: 30,  aniso: 2 },
  [QUALITY.HIGH]:   { dist: 105, simFar: 50, sort: true,  detail: 1, oct: 2, linger: 16, lodPx: 22,  aniso: 4 },
  [QUALITY.ULTRA]:  { dist: 140, simFar: 72, sort: true,  detail: 1, oct: 2, linger: 24, lodPx: 16,  aniso: 8 },
};

// ---------------------------------------------------------------------------
// Module scratch - never allocate in update()
// ---------------------------------------------------------------------------

const _v3a = new THREE.Vector3();
const _v3b = new THREE.Vector3();
const _mat4 = new THREE.Matrix4();
const _frustum = new THREE.Frustum();
const _windOut = { x: 0, y: 0, z: 0 };

/**
 * Parabolic sine approximation (Bhaskara-style, with the classic 0.225
 * refinement); max error ~0.1%. Only used where accuracy is irrelevant but count
 * is not: the turbulence field costs up to 12 of these per petal per frame.
 */
function fastSin(x) {
  x -= TAU * Math.floor(x * (1 / TAU) + 0.5);   // fold into [-PI, PI]
  const y = 1.2732395447351628 * x - 0.4052847345693511 * x * (x < 0 ? -x : x);
  return 0.225 * (y * (y < 0 ? -y : y) - y) + y;
}
const fastCos = (x) => fastSin(x + 1.5707963267948966);

/** Half-width of the petal silhouette at length-fraction y. */
function petalHalfWidth(y) {
  const t = y < 0 ? 0 : y > 1 ? 1 : y;
  return (HALF_W * Math.pow(t, 0.45) * (1 - 0.3 * smoothstep(0.55, 1, t))) / HALF_W_NORM;
}

/** Upper edge of the petal at width-offset x - this carves the tip cleft. */
function petalTipY(x) {
  const q = x / NOTCH_WIDTH;
  return 1 - NOTCH_DEPTH * Math.exp(-q * q);
}

// ---------------------------------------------------------------------------

export class PetalSystem {
  constructor(ctx) {
    this.ctx = ctx;
    this.state = ctx.state;
    this.bus = ctx.bus;
    this.scene = ctx.scene;
    this.rng = makeRNG(0x5a4b7a);
    this.ready = false;

    this.terrain = null;
    this.tree = null;
    this.spawnPoints = null;
    this.spawnCount = 0;
    this._spawnRetry = 0;

    this.capacity = 0;
    this.hardCapacity = 0;
    this.maxDist = 105;
    this.simFarDist = 50;
    this.sortEnabled = true;
    this.turbOctaves = 2;
    this.restLinger = 16;
    this.lodPixels = 22;
    this.emitBase = 60;

    this._viewportH = 1080;
    this._sizeAtten = 900;
    this._camX = 0; this._camY = 2; this._camZ = 0;
    this._emitAccum = 0;
    this._burstLeft = 0;
    this._burstRate = 0;
    this._burstCooldown = 0;
    this._liftBoost = 0;
    this._prevBoost = 1;
    this._windFallback = false;
    this._driftX = 0; this._driftZ = 0;
    this._turbPhase = 0;
    this._stealCursor = 0;
    this._offWeather = null;

    // One reusable update-range record per stream: three mutates and then clears
    // the array it is pushed into, so re-pushing the same object never allocates.
    this._rangeNear = { start: 0, count: 0 };
    this._rangeFar = { start: 0, count: 0 };
    this._planes = new Float32Array(24);
  }

  // -------------------------------------------------------------------------
  // Init
  // -------------------------------------------------------------------------

  async init() {
    const q = this.state.quality;
    // Sized once for the largest plausible tier so a quality change never
    // reallocates a live buffer; only the active limit moves at runtime.
    this.hardCapacity = clamp(Math.ceil((q.petalCount || 3000) * 1.6), 1200, 9000);
    this._allocArrays(this.hardCapacity);

    this.texture = this._makePetalTexture();
    this.material = this._makeMaterial(this.texture);
    this.nearMesh = this._makeStream(buildPetalGeometry(5, 7), this.hardCapacity, 11);
    this.farMesh = this._makeStream(buildPetalGeometry(2, 2), this.hardCapacity, 10);
    this.scene.add(this.farMesh.mesh, this.nearMesh.mesh);

    this._applyQuality(q);
    this.ready = true;
  }

  link(systems) {
    this.terrain = systems.terrain || null;
    this.tree = systems.tree || null;
    this._refreshSpawnPoints();
    // A petal storm has to arrive as a shock, not as a slowly rising rate.
    this._offWeather = this.bus.on(EVENTS.WEATHER_CHANGED, () => {
      const w = this.state.weather;
      if (w.target === WEATHER.PETAL_STORM || w.type === WEATHER.PETAL_STORM) this._triggerBurst();
    });
  }

  _allocArrays(n) {
    const f = () => new Float32Array(n);
    this.px = f(); this.py = f(); this.pz = f();          // position
    this.vx = f(); this.vy = f(); this.vz = f();          // velocity
    this.nx = f(); this.ny = f(); this.nz = f();          // face normal (unit)
    this.sx = f(); this.sy = f(); this.sz = f();          // tumble axis = long axis
    this.wx = f(); this.wy = f(); this.wz = f();          // cached mean wind
    this.rnx = f(); this.rny = f(); this.rnz = f();       // target normal while settling
    this.phase = f();     // flutter oscillator phase
    this.amp = f();       // rocking amplitude, radians
    this.mix = f();       // 0 = rocking flutter, 1 = continuous tumble
    this.drift = f();     // slow precession of the rocking centre, rad/s
    this.dragK = f();     // per-petal normal drag / mass
    this.gh = f();        // cached terrain height beneath the petal
    this.sizeL = f(); this.sizeW = f();
    this.tint = f();
    this.curl = f();      // cup/bend amount and sign
    this.flexPhase = f();
    this.age = f();       // seconds in the current phase
    this.linger = f();    // ground dwell time before fading
    this.liftAt = f();    // ground-level wind speed that peels this petal up
    this.yOffset = f();   // resting height jitter, so the carpet layers
    this.phaseOf = new Uint8Array(n);

    this._free = new Int32Array(n);
    this._freeCount = 0;
    this._candSlot = new Int32Array(n);
    this._candKey = new Int32Array(n);
    this._candDist = new Float32Array(n);
    this._candLod = new Uint8Array(n);
    this._order = new Int32Array(n);
    this._bucket = new Uint32Array(SORT_BUCKETS);

    this._rebuildFreeList();
  }

  _rebuildFreeList() {
    let c = 0;
    // Descending, so the allocator hands out low slots first and live petals stay
    // packed near the front of every sweep.
    for (let i = this.hardCapacity - 1; i >= 0; i--) {
      if (i < this.capacity && this.phaseOf[i] === FREE) this._free[c++] = i;
    }
    this._freeCount = c;
  }

  _makeStream(geo, cap, renderOrder) {
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(geo.position, 3));
    g.setAttribute('uv', new THREE.BufferAttribute(geo.uv, 2));
    g.setIndex(new THREE.BufferAttribute(geo.index, 1));

    // Interleaved: one contiguous upload feeds every instanced attribute.
    const data = new Float32Array(cap * STRIDE);
    const ib = new THREE.InstancedInterleavedBuffer(data, STRIDE, 1);
    ib.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('aOffset', new THREE.InterleavedBufferAttribute(ib, 3, 0));
    g.setAttribute('aAxisY', new THREE.InterleavedBufferAttribute(ib, 3, 3));
    g.setAttribute('aNormal', new THREE.InterleavedBufferAttribute(ib, 3, 6));
    g.setAttribute('aParams', new THREE.InterleavedBufferAttribute(ib, 4, 9));
    g.setAttribute('aShape', new THREE.InterleavedBufferAttribute(ib, 3, 13));
    g.instanceCount = 0;

    const mesh = new THREE.Mesh(g, this.material);
    mesh.frustumCulled = false;     // culled per petal on the CPU instead
    mesh.matrixAutoUpdate = false;  // instance offsets are already world space
    mesh.renderOrder = renderOrder;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    mesh.visible = false;
    return { mesh, geometry: g, buffer: ib, data, count: 0 };
  }

  // -------------------------------------------------------------------------
  // Procedural petal atlas
  //   R = albedo detail (veins, mottling)   G = thickness (drives translucency)
  //   B = deep-pink mask (throat blush, rim)  A = silhouette
  // Authored as linear data, not colour, so it is never sRGB-decoded on sample.
  // -------------------------------------------------------------------------

  _makePetalTexture() {
    const gen = () => {
      const S = 128;
      const data = new Uint8Array(S * S * 4);
      const feather = 2.6 / S;                  // ~2.6 texels of edge softness
      const rn = makeRNG(0x9e21);
      const ox = rn() * 50, oy = rn() * 50;

      for (let j = 0; j < S; j++) {
        const v = (j + 0.5) / S;
        const y = TEX_Y_MIN + v * TEX_Y_SPAN;
        for (let i = 0; i < S; i++) {
          const u = (i + 0.5) / S;
          const x = (u - 0.5) * TEX_X_SPAN;
          const ax = x < 0 ? -x : x;

          // silhouette: width, tip cleft and stem end, each feathered
          const w = petalHalfWidth(y);
          const alpha =
            clamp01((w - ax) / feather) *
            clamp01((petalTipY(x) - y) / feather) *
            clamp01((y + 0.035) / (feather * 1.6));

          // veins radiate from the stem, so work in petal-polar coordinates
          const ang = Math.atan2(x, y + 0.14);
          const along = clamp01(y / 1.05);
          let vein = 0.5 + 0.5 * Math.cos(ang * 26 + along * 1.4);
          vein = vein * vein * vein * vein * vein;      // thin, sharp lines
          const veinFade = along * (1 - 0.55 * along);  // fade out at both ends
          const mottle = noise.fbm2D(x * 34 + ox, y * 22 + oy, 3) * 0.5 + 0.5;
          const edgeT = clamp01((ax - w * 0.82) / (w * 0.2 + 1e-4));

          // crumpled edges catch light; veins sit slightly in shadow
          const detail = 0.8 + 0.14 * mottle - 0.16 * vein * veinFade + 0.1 * edgeT * edgeT;

          // thickness: fat along the midrib and base, tissue-thin at the rim
          const across = clamp01(ax / (w + 1e-4));
          const thick = clamp01(
            0.14 + 0.86 * (1 - Math.pow(across, 1.45)) * (1 - 0.5 * along) + 0.1 * vein * veinFade
          );

          // pigment: sakura are deepest at the throat, near-white at the tip
          const blush = Math.pow(clamp01(1 - along * 1.15), 1.7);
          const pink = clamp01(blush * 0.95 + 0.3 * edgeT * edgeT * along + 0.1 * vein * veinFade);

          const o = (j * S + i) * 4;
          data[o] = clamp01(detail) * 255 + 0.5;
          data[o + 1] = thick * 255 + 0.5;
          data[o + 2] = pink * 255 + 0.5;
          data[o + 3] = alpha * 255 + 0.5;
        }
      }

      const tex = new THREE.DataTexture(data, S, S, THREE.RGBAFormat);
      tex.colorSpace = THREE.NoColorSpace;
      tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
      tex.magFilter = THREE.LinearFilter;
      tex.minFilter = THREE.LinearMipmapLinearFilter;
      tex.generateMipmaps = true;
      tex.needsUpdate = true;
      return tex;
    };
    return this.ctx.textures?.get ? this.ctx.textures.get('petal.atlas', gen) : gen();
  }

  // -------------------------------------------------------------------------
  // Material
  // -------------------------------------------------------------------------

  _makeMaterial(map) {
    const uniforms = THREE.UniformsUtils.merge([THREE.UniformsLib.fog]);
    uniforms.uMap = { value: map };
    uniforms.uTime = { value: 0 };
    uniforms.uSizeAtten = { value: 900 };
    uniforms.uMinPixels = { value: 1.7 };
    uniforms.uDetail = { value: 1 };
    uniforms.uKeyDir = { value: new THREE.Vector3(0, 1, 0) };
    uniforms.uKeyColor = { value: new THREE.Color(1, 1, 1) };
    uniforms.uSkyUp = { value: new THREE.Color(0.18, 0.36, 0.72) };
    uniforms.uSkyDown = { value: new THREE.Color(0.22, 0.24, 0.2) };
    uniforms.uAmbient = { value: 1 };
    // Pale, desaturated, slightly warm - sakura is closer to white than to pink.
    uniforms.uPale = { value: new THREE.Color('#f4e7e3') };
    uniforms.uDeep = { value: new THREE.Color('#e0b7c1') };
    uniforms.uAged = { value: new THREE.Color('#c3ac9e') };
    uniforms.uCanopy = { value: new THREE.Vector4(0, 6, 0, 5) };
    uniforms.uCanopyShade = { value: 0.55 };
    uniforms.uWetness = { value: 0 };

    const mat = new THREE.ShaderMaterial({
      uniforms,
      vertexShader: PETAL_VERT,
      fragmentShader: PETAL_FRAG,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      fog: true,
      dithering: true,   // pale, smooth petal gradients band badly without this
      blending: THREE.NormalBlending,
    });
    this.u = mat.uniforms;
    return mat;
  }

  // -------------------------------------------------------------------------
  // Quality
  // -------------------------------------------------------------------------

  onQualityChange(quality) {
    if (this.ready) this._applyQuality(quality);
  }

  _applyQuality(q) {
    const s = TIER[q.tier] || TIER[QUALITY.HIGH];
    const prev = this.capacity;
    this.capacity = Math.min(q.petalCount | 0, this.hardCapacity);
    this.maxDist = s.dist;
    this.simFarDist = s.simFar;
    this.sortEnabled = s.sort;
    this.turbOctaves = s.oct;
    this.restLinger = s.linger;
    this.lodPixels = s.lodPx;
    this.u.uDetail.value = s.detail;
    this.u.uMinPixels.value = q.tier === QUALITY.LOW ? 2.1 : 1.7;

    const maxAniso = this.ctx.renderer?.capabilities?.getMaxAnisotropy?.() ?? 1;
    const aniso = Math.min(s.aniso, maxAniso);
    if (this.texture && this.texture.anisotropy !== aniso) {
      this.texture.anisotropy = aniso;
      this.texture.needsUpdate = true;
    }

    // Petals the new tier can no longer afford fade out instead of vanishing - 
    // dropping several hundred petals in one frame is extremely visible.
    for (let i = this.capacity; i < prev; i++) {
      if (this.phaseOf[i] !== FREE) {
        this.phaseOf[i] = RESTING;
        this.linger[i] = 0;
        this.age[i] = 0;
      }
    }
    // Emission so the steady-state population lands just under capacity:
    // rate * (airborne life + ground dwell) ~= capacity.
    this.emitBase = (this.capacity / (this.restLinger + 15)) * 0.88;
    this._rebuildFreeList();
  }

  resize(width, height) {
    this._viewportH = height;
  }

  // -------------------------------------------------------------------------
  // Frame
  // -------------------------------------------------------------------------

  update(dt, state) {
    if (!this.ready || dt <= 0) return;
    if (!this.spawnPoints) {
      this._spawnRetry -= dt;
      if (this._spawnRetry <= 0) { this._spawnRetry = 1.5; this._refreshSpawnPoints(); }
    }
    const cam = this.ctx.camera;
    cam.getWorldPosition(_v3a);
    this._camX = _v3a.x; this._camY = _v3a.y; this._camZ = _v3a.z;

    this._syncUniforms(state);
    this._updateEmission(dt, state);
    this._simulate(dt, state);
    this._buildInstances(state);
  }

  _syncUniforms(state) {
    const u = this.u;
    u.uTime.value = state.time.elapsed;

    // One dominant key light, blended from sun and moon by irradiance. Evaluating
    // both would double the translucency cost for a term only ever visible once,
    // and blending keeps dawn/dusk continuous instead of snapping between them.
    const sun = state.sun, moon = state.moon;
    const si = Math.max(sun.intensity * sun.visibility, 0);
    const mi = Math.max(moon.intensity * moon.visibility, 0);
    _v3b.copy(sun.direction).multiplyScalar(si).addScaledVector(moon.direction, mi);
    if (si + mi > 1e-4 && _v3b.lengthSq() > 1e-8) u.uKeyDir.value.copy(_v3b).normalize();
    else u.uKeyDir.value.set(0, 1, 0);
    u.uKeyColor.value.setRGB(
      sun.color.r * si + moon.color.r * mi,
      sun.color.g * si + moon.color.g * mi,
      sun.color.b * si + moon.color.b * mi
    );

    u.uSkyUp.value.copy(state.sky.zenithColor).lerp(state.sky.horizonColor, 0.35);
    u.uSkyDown.value.copy(state.sky.groundColor);
    u.uAmbient.value = state.sky.ambientIntensity;
    u.uWetness.value = state.weather.wetness;

    if (this.tree?.canopyCenter) {
      const c = this.tree.canopyCenter;
      u.uCanopy.value.set(c.x, c.y, c.z, Math.max(this.tree.canopyRadius || 5, 0.5));
    }
    // Overcast skies cast no crisp canopy shadow to fall out of.
    u.uCanopyShade.value = 0.62 * clamp01(1 - state.clouds.coverage * 0.55);

    // Pixels per world unit at 1 m: the min-screen-size clamp needs it to stop
    // distant petals sizzling into sub-pixel noise.
    const cam = this.ctx.camera;
    const h = this._viewportH * (state.quality.resolutionScale || 1);
    const fov = ((cam && cam.fov) || 55) * Math.PI / 180;
    this._sizeAtten = h / (2 * Math.tan(fov * 0.5));
    u.uSizeAtten.value = this._sizeAtten;
  }

  // -------------------------------------------------------------------------
  // Emission
  // -------------------------------------------------------------------------

  _refreshSpawnPoints() {
    const pts = this.tree?.getBlossomSpawnPoints?.();
    if (pts && pts.length >= 3) {
      this.spawnPoints = pts;
      this.spawnCount = Math.floor(pts.length / 3);
    }
  }

  _triggerBurst() {
    if (this._burstCooldown > 0) return;
    this._burstCooldown = 6;
    this._burstLeft = Math.floor(this.capacity * 0.34);
    this._burstRate = this._burstLeft / 0.85;   // drained over ~0.85 s
    this._liftBoost = 1;                        // and it tears the ground layer up
  }

  _updateEmission(dt, state) {
    const w = state.wind;
    const boost = state.weather.petalBoost || 1;
    if (this._burstCooldown > 0) this._burstCooldown -= dt;
    // Fallback trigger: a weather system that raises petalBoost without emitting
    // WEATHER_CHANGED still gets its burst.
    if (boost > this._prevBoost + 0.35 && boost > 1.25) this._triggerBurst();
    this._prevBoost = boost;
    if (this._liftBoost > 0) this._liftBoost = Math.max(0, this._liftBoost - dt * 0.4);

    // Gusts strip petals loose; still air lets go of only a trickle.
    const gust = clamp01((w.gust * w.strength) / 9);
    this._emitAccum += this.emitBase * (0.5 + 0.75 * gust * gust) * boost * dt;

    let budget = Math.min(Math.floor(this._emitAccum), 90);
    this._emitAccum -= budget;
    if (this._burstLeft > 0) {
      const n = Math.min(this._burstLeft, Math.ceil(this._burstRate * dt));
      budget += n;
      this._burstLeft -= n;
    }
    if (budget > 0) this._spawn(budget, state, boost > 1.25 || this._burstLeft > 0);
  }

  /** Pull a slot, stealing from the ground layer before starving the air. */
  _alloc() {
    if (this._freeCount > 0) return this._free[--this._freeCount];
    const cap = this.capacity;
    if (cap === 0) return -1;
    // Bounded scan: recycle a well-aged resting petal rather than fail to spawn.
    for (let n = 0; n < 96; n++) {
      const i = this._stealCursor;
      this._stealCursor = this._stealCursor + 1 >= cap ? 0 : this._stealCursor + 1;
      if (this.phaseOf[i] === RESTING && this.age[i] > this.linger[i] * 0.4) return i;
    }
    return -1;
  }

  _spawn(count, state, storm) {
    const rng = this.rng;
    const pts = this.spawnPoints;
    const n = this.spawnCount;
    const wind = state.wind;
    const wdx = wind.direction.x, wdz = wind.direction.y;
    const windSpeed = wind.strength * wind.gust;
    const camAhead = storm ? 0.55 : 0;
    const ccx = this.tree?.canopyCenter?.x ?? 0;
    const ccy = this.tree?.canopyCenter?.y ?? 7;
    const ccz = this.tree?.canopyCenter?.z ?? 0;
    const camX = this._camX, camY = this._camY, camZ = this._camZ;

    for (let k = 0; k < count; k++) {
      const i = this._alloc();
      if (i < 0) return;
      let x, y, z;

      if (camAhead > 0 && rng() < camAhead) {
        // Storm petals also stream in from upwind of the viewer, so the world is
        // not empty the moment you walk away from the tree.
        const spread = this.maxDist * 0.55;
        x = camX - wdx * spread * rng.range(0.35, 1) + rng.gaussian(0, spread * 0.4);
        z = camZ - wdz * spread * rng.range(0.35, 1) + rng.gaussian(0, spread * 0.4);
        y = camY + rng.range(1.5, 14);
      } else if (pts && n > 0) {
        // Windward blossoms shed first: three tries, then take what we get.
        let idx = 0;
        for (let t = 0; t < 3; t++) {
          idx = (rng() * n) | 0;
          const o = idx * 3;
          const cx = pts[o] - ccx, cz = pts[o + 2] - ccz;
          const inv = 1 / (Math.sqrt(cx * cx + cz * cz) + 1e-3);
          if ((cx * wdx + cz * wdz) * inv > rng.range(-0.9, 0.35)) break;
        }
        const o = idx * 3;
        x = pts[o] + rng.gaussian(0, 0.06);
        y = pts[o + 1] + rng.gaussian(0, 0.06);
        z = pts[o + 2] + rng.gaussian(0, 0.06);
      } else {
        // Tree not ready: shed from a plausible canopy shell so we are never empty.
        const r = (this.tree?.canopyRadius ?? 5) * Math.cbrt(rng.range(0.35, 1));
        const th = rng() * TAU;
        const ph = Math.acos(rng.range(-0.55, 1));
        const sp = Math.sin(ph);
        x = ccx + sp * Math.cos(th) * r;
        y = ccy + Math.cos(ph) * r * 0.75;
        z = ccz + sp * Math.sin(th) * r;
      }
      this._initPetal(i, x, y, z, storm ? 1 : 0, windSpeed, state);
    }
  }

  _initPetal(i, x, y, z, storm, windSpeed, state) {
    const rng = this.rng;
    this.px[i] = x; this.py[i] = y; this.pz[i] = z;

    // Detachment flick: the branch springs back as the petal lets go.
    const kick = 0.25 + storm * 1.6 + windSpeed * 0.12;
    this.vx[i] = rng.gaussian(0, kick);
    this.vy[i] = rng.range(-0.15, 0.35) + storm * rng.range(0, 1.2);
    this.vz[i] = rng.gaussian(0, kick);

    const a = rng() * TAU;                 // tumble axis: horizontal, random heading
    this.sx[i] = Math.cos(a); this.sy[i] = 0; this.sz[i] = Math.sin(a);
    const b = rng() * TAU;                 // face normal starts perpendicular to it
    const cb = Math.cos(b), sb = Math.sin(b);
    this.nx[i] = -this.sz[i] * cb; this.ny[i] = sb; this.nz[i] = this.sx[i] * cb;
    this._normalize(i);

    this.phase[i] = rng() * TAU;
    this.amp[i] = rng.range(0.55, 1.15);
    // ~30% autorotate instead of rocking; that is where the full face-on to
    // edge-on flips come from, and edge-on petals nearly vanishing is the cue.
    this.mix[i] = rng() < 0.3 ? rng.range(0.6, 1) : rng.range(0, 0.22);
    this.drift[i] = clamp(rng.gaussian(0, 0.3), -0.7, 0.7);
    this.dragK[i] = DRAG_N * rng.range(0.82, 1.2);

    const len = rng.range(0.052, 0.088);
    this.sizeL[i] = len;
    this.sizeW[i] = len * rng.range(0.86, 1.06);
    this.tint[i] = clamp01(rng.gaussian(0.5, 0.26));
    this.curl[i] = rng.range(0.35, 1) * (rng() < 0.5 ? -1 : 1);
    this.flexPhase[i] = rng() * TAU;
    this.age[i] = 0;
    this.linger[i] = this.restLinger * rng.range(0.55, 1.45);
    this.liftAt[i] = rng.range(1.6, 4.4);
    this.yOffset[i] = rng.range(0.004, 0.016);
    this.gh[i] = this._groundAt(x, z);

    const w = this._sampleWind(x, z, state);
    this.wx[i] = w.x; this.wy[i] = w.y; this.wz[i] = w.z;
    this.phaseOf[i] = FALLING;
  }

  _normalize(i) {
    const nx = this.nx[i], ny = this.ny[i], nz = this.nz[i];
    const inv = 1 / (Math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9);
    this.nx[i] = nx * inv; this.ny[i] = ny * inv; this.nz[i] = nz * inv;
  }

  _groundAt(x, z) {
    const t = this.terrain;
    if (t && typeof t.getHeight === 'function') {
      const h = t.getHeight(x, z);
      return h === h && h !== Infinity && h !== -Infinity ? h : 0;
    }
    return 0;
  }

  _sampleWind(x, z, state) {
    const w = state.wind;
    w.sample(x, z, state.time.elapsed, _windOut);
    if (this._windFallback) {
      const s = w.strength * w.gust;
      _windOut.x = w.direction.x * s;
      _windOut.y = 0;
      _windOut.z = w.direction.y * s;
    }
    return _windOut;
  }

  // -------------------------------------------------------------------------
  // Simulation
  // -------------------------------------------------------------------------

  _simulate(dt, state) {
    const wind = state.wind;
    const t = state.time.elapsed;
    const frame = state.time.frame;

    // If wind.sample is still the no-op stub, synthesise the mean flow from the
    // published direction and strength so petals never hang dead in a gale.
    wind.sample(0, 0, t, _windOut);
    const baseSpeed = wind.strength * wind.gust;
    this._windFallback =
      Math.abs(_windOut.x) + Math.abs(_windOut.y) + Math.abs(_windOut.z) < 1e-5 && baseSpeed > 0.05;

    // Frozen-turbulence frame: eddies are carried along by the mean wind.
    this._driftX -= wind.direction.x * baseSpeed * dt;
    this._driftZ -= wind.direction.y * baseSpeed * dt;
    this._turbPhase += dt;
    const tp = this._turbPhase;
    const dX = this._driftX, dZ = this._driftZ;

    // ~15% turbulence intensity in wind, plus a little thermal drift in dead calm.
    const turbAmp = (0.45 + 0.9 * wind.turbulence) * (0.06 + 0.1 * baseSpeed);
    const k1 = 0.45, k2 = k1 * 2.37;    // ~14 m and ~6 m eddies
    const oct2 = this.turbOctaves > 1;

    const player = state.player;
    const pvx = player.velocity.x, pvz = player.velocity.z;
    const kickOn =
      player.mode === 'walk' && player.grounded && pvx * pvx + pvz * pvz > 1.2;
    const plx = player.position.x, plz = player.position.z;
    const ply = player.position.y - player.eyeHeight;

    const camX = this._camX, camZ = this._camZ;
    const farSim = this.simFarDist * this.simFarDist;
    const liftEase = 1 - 0.55 * this._liftBoost;
    const wetGrip = (1 + state.weather.wetness * 1.6) * liftEase;

    const n = this.hardCapacity;
    for (let i = 0; i < n; i++) {
      const ph = this.phaseOf[i];
      if (ph === FREE) continue;

      const x = this.px[i], y = this.py[i], z = this.pz[i];
      const hAbove = y - this.gh[i];

      // ---- staggered terrain + wind refresh -------------------------------
      if (ph !== RESTING && (hAbove < 2 || (i + frame) % GROUND_STRIDE === 0)) {
        this.gh[i] = this._groundAt(x, z);
      }
      if ((i + frame) % WIND_STRIDE === 0) {
        const w = this._sampleWind(x, z, state);
        // Boundary layer: the last couple of metres of air are dragged by the ground.
        const bl = 0.4 + 0.6 * smoothstep(0, 2.5, hAbove);
        const l = 1 - Math.exp(-14 * dt * WIND_STRIDE);
        this.wx[i] += (w.x * bl - this.wx[i]) * l;
        this.wy[i] += (w.y * bl - this.wy[i]) * l;
        this.wz[i] += (w.z * bl - this.wz[i]) * l;
      }

      // ---- the ground layer -----------------------------------------------
      if (ph === RESTING) {
        this.age[i] += dt;
        const spent = this.age[i] > this.linger[i];
        if (spent) {
          if (this.age[i] > this.linger[i] + FADE_OUT) this._release(i);
        } else if (Math.hypot(this.wx[i], this.wz[i]) > this.liftAt[i] * wetGrip) {
          this._lift(i, this.wx[i], this.wz[i], 0.5 + 0.5 * this._liftBoost);
        } else if (kickOn) {
          const ddx = x - plx, ddz = z - plz, ddy = y - ply;
          if (ddx * ddx + ddz * ddz < 0.49 && ddy > -0.4 && ddy < 1.2) {
            this._lift(i, pvx * 0.55, pvz * 0.55, 0.8);
          }
        }
        continue;
      }

      // ---- distance LOD: far petals integrate at half rate, doubled dt ------
      let h = dt;
      const ddx = x - camX, ddz = z - camZ;
      if (ddx * ddx + ddz * ddz > farSim) {
        if (((i + frame) & 1) === 1) continue;
        h = dt * 2;
      }
      this.age[i] += h;

      // ---- air velocity: mean wind + divergence-free ABC turbulence --------
      const ex = x + dX, ez = z + dZ;
      let ax = fastSin(k1 * y + tp * 0.7) + fastCos(k1 * ez + tp * 0.53);
      let ay = fastSin(k1 * ez + tp * 0.61) + fastCos(k1 * ex + tp * 0.83);
      let az = fastSin(k1 * ex + tp * 0.47) + fastCos(k1 * y + tp * 0.67);
      if (oct2) {
        ax += 0.45 * (fastSin(k2 * y - tp * 1.3) + fastCos(k2 * ez + tp * 1.1));
        ay += 0.45 * (fastSin(k2 * ez - tp * 1.2) + fastCos(k2 * ex + tp * 1.4));
        az += 0.45 * (fastSin(k2 * ex - tp * 1.5) + fastCos(k2 * y + tp * 1.05));
      }
      // No flow through the ground: fade the vertical eddy component out near it.
      const vFade = smoothstep(0, 1.6, hAbove);
      const wxa = this.wx[i] + ax * turbAmp;
      const wya = this.wy[i] + ay * turbAmp * vFade;
      const wza = this.wz[i] + az * turbAmp;

      let ux = this.vx[i] - wxa, uy = this.vy[i] - wya, uz = this.vz[i] - wza;
      const speed = Math.sqrt(ux * ux + uy * uy + uz * uz);

      // ---- orientation ----------------------------------------------------
      if (ph === FALLING) {
        const rate = Math.min(
          (TAU * FLUTTER_HZ * Math.max(speed, 0.08) * REF_LENGTH) /
            (TERMINAL_FALL * this.sizeL[i]),
          TAU * FLUTTER_HZ_MAX
        );
        const m = this.mix[i];
        const dTheta =
          (m * rate * 0.6 + (1 - m) * this.amp[i] * rate * fastCos(this.phase[i]) + this.drift[i]) * h;
        this.phase[i] += rate * h;
        if (this.phase[i] > TAU) this.phase[i] -= TAU;

        // Rodrigues about a perpendicular axis collapses to a 2D rotation.
        const c = Math.cos(dTheta), s = Math.sin(dTheta);
        const sxi = this.sx[i], syi = this.sy[i], szi = this.sz[i];
        const nxi = this.nx[i], nyi = this.ny[i], nzi = this.nz[i];
        this.nx[i] = nxi * c + (syi * nzi - szi * nyi) * s;
        this.ny[i] = nyi * c + (szi * nxi - sxi * nzi) * s;
        this.nz[i] = nzi * c + (sxi * nyi - syi * nxi) * s;

        // The tumble axis swings horizontal and across the drift - exactly what a
        // falling leaf does. Damped, so a gust turns petals instead of snapping them.
        const hs = Math.sqrt(ux * ux + uz * uz);
        if (hs > 0.25) {
          const tx = -uz / hs, tz = ux / hs;
          const sgn = sxi * tx + szi * tz < 0 ? -1 : 1;
          const l = 1 - Math.exp(-0.9 * h);
          let nsx = sxi + (tx * sgn - sxi) * l;
          let nsy = syi - syi * l;
          let nsz = szi + (tz * sgn - szi) * l;
          const inv = 1 / (Math.sqrt(nsx * nsx + nsy * nsy + nsz * nsz) + 1e-9);
          this.sx[i] = nsx * inv; this.sy[i] = nsy * inv; this.sz[i] = nsz * inv;
        }
        // Re-orthogonalise the normal against the (possibly moved) axis.
        const d = this.nx[i] * this.sx[i] + this.ny[i] * this.sy[i] + this.nz[i] * this.sz[i];
        this.nx[i] -= this.sx[i] * d; this.ny[i] -= this.sy[i] * d; this.nz[i] -= this.sz[i] * d;
        this._normalize(i);
      } else {
        // Settling: flatten onto the cached ground normal over ~0.4 s.
        const l = 1 - Math.exp(-9 * h);
        this.nx[i] += (this.rnx[i] - this.nx[i]) * l;
        this.ny[i] += (this.rny[i] - this.ny[i]) * l;
        this.nz[i] += (this.rnz[i] - this.nz[i]) * l;
        this._normalize(i);
      }

      // ---- forces: gravity kick, exact quadratic drag, gravity kick --------
      let vxi = this.vx[i], vyi = this.vy[i], vzi = this.vz[i];
      const gk = GRAVITY * 0.5 * h;
      vyi -= gk;

      ux = vxi - wxa; uy = vyi - wya; uz = vzi - wza;
      const nxi = this.nx[i], nyi = this.ny[i], nzi = this.nz[i];
      const un = ux * nxi + uy * nyi + uz * nzi;
      const tx = ux - un * nxi, ty = uy - un * nyi, tz = uz - un * nzi;
      const tmag = Math.sqrt(tx * tx + ty * ty + tz * tz);
      const kn = this.dragK[i];
      // u' = u / (1 + k|u|dt) is the closed-form solution of du/dt = -k·u|u|.
      const unN = un / (1 + kn * (un < 0 ? -un : un) * h);
      const ts = 1 / (1 + kn * DRAG_T_RATIO * tmag * h);
      vxi = wxa + unN * nxi + tx * ts;
      vyi = wya + unN * nyi + ty * ts;
      vzi = wza + unN * nzi + tz * ts;
      vyi -= gk;

      if (ph === SLIDING) {
        const f = Math.exp(-6.5 * h);      // skid friction, no lift, pinned down
        vxi *= f; vzi *= f; vyi = 0;
        this.px[i] = x + vxi * h;
        this.pz[i] = z + vzi * h;
        this.py[i] = this.gh[i] + this.yOffset[i];
        if (this.age[i] > 0.55 || vxi * vxi + vzi * vzi < 0.0025) {
          this.vx[i] = vxi; this.vz[i] = vzi;
          this._settle(i);
        } else {
          this.vx[i] = vxi; this.vy[i] = 0; this.vz[i] = vzi;
        }
        continue;
      }

      this.px[i] = x + vxi * h;
      this.py[i] = y + vyi * h;
      this.pz[i] = z + vzi * h;
      this.vx[i] = vxi; this.vy[i] = vyi; this.vz[i] = vzi;

      const rest = this.gh[i] + this.yOffset[i];
      if (this.py[i] <= rest) {
        this.py[i] = rest;
        this.phaseOf[i] = SLIDING;
        this.age[i] = 0;
        this._beginSlide(i);
      } else if (this.py[i] > this.gh[i] + 300) {
        this._release(i);                  // updraught runaway guard
      }
    }
  }

  /** Cache this petal's target lie-flat orientation from the ground normal. */
  _beginSlide(i) {
    let gx = 0, gy = 1, gz = 0;
    const t = this.terrain;
    if (t && typeof t.getNormal === 'function') {
      const nrm = t.getNormal(this.px[i], this.pz[i], _v3b);
      if (nrm && nrm.y === nrm.y) { gx = nrm.x; gy = nrm.y; gz = nrm.z; }
    }
    // Petals do not lie perfectly flat on grass - tilt each one a little, or the
    // carpet reads as a decal sheet.
    const r = this.rng;
    gx += r.gaussian(0, 0.16); gy += r.gaussian(0, 0.05); gz += r.gaussian(0, 0.16);
    if (gy < 0.35) gy = 0.35;
    const inv = 1 / (Math.sqrt(gx * gx + gy * gy + gz * gz) + 1e-9);
    this.rnx[i] = gx * inv; this.rny[i] = gy * inv; this.rnz[i] = gz * inv;
    this.vy[i] = 0;
  }

  /** Lock the long axis into the ground plane, aligned with the slide direction. */
  _settle(i) {
    let ax = this.vx[i], az = this.vz[i];
    if (ax * ax + az * az < 1e-5) { const a = this.rng() * TAU; ax = Math.cos(a); az = Math.sin(a); }
    const nx = this.rnx[i], ny = this.rny[i], nz = this.rnz[i];
    this.nx[i] = nx; this.ny[i] = ny; this.nz[i] = nz;
    let sx = ax, sy = 0, sz = az;
    const d = sx * nx + sy * ny + sz * nz;
    sx -= nx * d; sy -= ny * d; sz -= nz * d;
    const inv = 1 / (Math.sqrt(sx * sx + sy * sy + sz * sz) + 1e-9);
    this.sx[i] = sx * inv; this.sy[i] = sy * inv; this.sz[i] = sz * inv;
    this.vx[i] = 0; this.vy[i] = 0; this.vz[i] = 0;
    this.phaseOf[i] = RESTING;
    this.age[i] = 0;
  }

  /** Peel a settled petal back into the air. */
  _lift(i, wx, wz, power) {
    const r = this.rng;
    this.phaseOf[i] = FALLING;
    this.age[i] = FADE_IN;                 // already visible; do not fade it back in
    this.linger[i] = this.restLinger * r.range(0.55, 1.45);
    this.vx[i] = wx * r.range(0.35, 0.8);
    this.vz[i] = wz * r.range(0.35, 0.8);
    this.vy[i] = r.range(0.5, 2.2) * power;
    this.py[i] += 0.02;
    this.phase[i] = r() * TAU;
    this.mix[i] = r() < 0.5 ? r.range(0.5, 1) : r.range(0, 0.3);
    const a = r() * TAU;
    this.sx[i] = Math.cos(a); this.sy[i] = 0; this.sz[i] = Math.sin(a);
    const d = this.nx[i] * this.sx[i] + this.ny[i] * this.sy[i] + this.nz[i] * this.sz[i];
    this.nx[i] -= this.sx[i] * d; this.ny[i] -= this.sy[i] * d; this.nz[i] -= this.sz[i] * d;
    this._normalize(i);
  }

  _release(i) {
    this.phaseOf[i] = FREE;
    if (i < this.capacity && this._freeCount < this._free.length) {
      this._free[this._freeCount++] = i;
    }
  }

  // -------------------------------------------------------------------------
  // Instance build: cull, LOD split, depth sort, upload
  // -------------------------------------------------------------------------

  _buildInstances(state) {
    const cam = this.ctx.camera;
    const camX = this._camX, camY = this._camY, camZ = this._camZ;

    if (!state.debug.freezeFrustum) {
      _mat4.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
      _frustum.setFromProjectionMatrix(_mat4);
      const pl = this._planes;
      for (let k = 0; k < 6; k++) {
        const p = _frustum.planes[k];
        pl[k * 4] = p.normal.x; pl[k * 4 + 1] = p.normal.y;
        pl[k * 4 + 2] = p.normal.z; pl[k * 4 + 3] = p.constant;
      }
    }
    const pl = this._planes;
    const maxDist = this.maxDist;
    const maxD2 = maxDist * maxDist;
    const fadeStart = maxDist * 0.72;
    const lodPixels = this.lodPixels;
    const atten = this._sizeAtten;
    const invMax = (SORT_BUCKETS - 1) / maxDist;
    const n = this.hardCapacity;

    // -- gather visible candidates ----------------------------------------
    let cand = 0;
    for (let i = 0; i < n; i++) {
      if (this.phaseOf[i] === FREE) continue;
      const x = this.px[i], y = this.py[i], z = this.pz[i];
      const ddx = x - camX, ddy = y - camY, ddz = z - camZ;
      const d2 = ddx * ddx + ddy * ddy + ddz * ddz;
      if (d2 > maxD2) continue;

      const r = this.sizeL[i] * 0.75;
      let vis = true;
      for (let k = 0; k < 6; k++) {
        const o = k * 4;
        if (pl[o] * x + pl[o + 1] * y + pl[o + 2] * z + pl[o + 3] < -r) { vis = false; break; }
      }
      if (!vis) continue;

      const dist = Math.sqrt(d2);
      this._candSlot[cand] = i;
      this._candDist[cand] = dist;
      this._candLod[cand] = (this.sizeL[i] * atten) / (dist < 0.05 ? 0.05 : dist) > lodPixels ? 1 : 0;
      // Descending distance key: bucket 0 is the far plane, so ascending bucket
      // order is back to front.
      const b = (SORT_BUCKETS - 1) - ((dist * invMax) | 0);
      this._candKey[cand] = b < 0 ? 0 : b;
      cand++;
    }

    // -- counting sort so overlapping petals composite in the right order ---
    const order = this._order;
    if (this.sortEnabled && cand > 1) {
      const counts = this._bucket;
      counts.fill(0);
      for (let c = 0; c < cand; c++) counts[this._candKey[c]]++;
      let sum = 0;
      for (let b = 0; b < SORT_BUCKETS; b++) { const v = counts[b]; counts[b] = sum; sum += v; }
      for (let c = 0; c < cand; c++) order[counts[this._candKey[c]]++] = c;
    } else {
      for (let c = 0; c < cand; c++) order[c] = c;
    }

    // -- write instance data ----------------------------------------------
    const nd = this.nearMesh.data, fd = this.farMesh.data;
    let nc = 0, fc = 0;
    for (let c = 0; c < cand; c++) {
      const ci = order[c];
      const i = this._candSlot[ci];
      const dist = this._candDist[ci];
      const ph = this.phaseOf[i];

      // Fade in at the near plane and out into the atmosphere. A hard cut at
      // either end pops, and a petal spawning at full opacity in front of the
      // camera reads as a glitch.
      let alpha = smoothstep(0.12, 0.5, dist) * (1 - smoothstep(fadeStart, maxDist, dist));
      const age = this.age[i];
      if (age < FADE_IN) alpha *= age / FADE_IN;
      let aged = 0;
      if (ph === RESTING) {
        aged = clamp01(age / (this.linger[i] + 3));
        const over = age - this.linger[i];
        if (over > 0) alpha *= 1 - clamp01(over / FADE_OUT);
      }
      if (alpha <= 0.004) continue;

      const isNear = this._candLod[ci] === 1;
      const data = isNear ? nd : fd;
      const o = (isNear ? nc++ : fc++) * STRIDE;
      const shrink = 1 - aged * 0.12;      // petals shrivel slightly as they dry
      data[o] = this.px[i]; data[o + 1] = this.py[i]; data[o + 2] = this.pz[i];
      data[o + 3] = this.sx[i]; data[o + 4] = this.sy[i]; data[o + 5] = this.sz[i];
      data[o + 6] = this.nx[i]; data[o + 7] = this.ny[i]; data[o + 8] = this.nz[i];
      data[o + 9] = this.sizeL[i] * shrink;
      data[o + 10] = this.sizeW[i] * shrink;
      data[o + 11] = alpha;
      data[o + 12] = this.tint[i];
      data[o + 13] = this.curl[i];
      data[o + 14] = aged;
      data[o + 15] = this.flexPhase[i];
    }

    this._upload(this.nearMesh, nc, this._rangeNear);
    this._upload(this.farMesh, fc, this._rangeFar);
  }

  _upload(stream, count, range) {
    stream.count = count;
    stream.geometry.instanceCount = count;
    stream.mesh.visible = count > 0;
    if (count === 0) return;
    const buf = stream.buffer;
    buf.updateRanges.length = 0;
    range.start = 0;
    range.count = count * STRIDE;
    buf.updateRanges.push(range);
    buf.needsUpdate = true;
  }

  // -------------------------------------------------------------------------

  dispose() {
    this._offWeather?.();
    this._offWeather = null;
    for (const s of [this.nearMesh, this.farMesh]) {
      if (!s) continue;
      this.scene.remove(s.mesh);
      s.geometry.dispose();
    }
    this.material?.dispose();
    // The atlas belongs to the TextureFactory cache whenever one is present.
    if (!this.ctx.textures?.get) this.texture?.dispose();
    this.ready = false;
  }
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/**
 * Builds a petal card. The mesh HUGS the silhouette instead of being a quad with
 * a petal-shaped alpha: on a fill-rate-bound iGPU ~40% of a quad's pixels are
 * fully transparent and still cost a blend. The alpha map then only has to soften
 * the last couple of texels of the edge.
 *
 * cols = rows = 2 degenerates to the far-LOD quad - two triangles, because a
 * 5-pixel petal made of 48 triangles is pure quad-overshading.
 */
function buildPetalGeometry(cols, rows) {
  const simple = cols === 2 && rows === 2;
  const position = new Float32Array(cols * rows * 3);
  const uv = new Float32Array(cols * rows * 2);
  const index = new Uint16Array((cols - 1) * (rows - 1) * 6);

  let p = 0, q = 0;
  for (let r = 0; r < rows; r++) {
    const tv = simple ? r : Math.pow(r / (rows - 1), 0.92);
    for (let c = 0; c < cols; c++) {
      const tu = (c / (cols - 1)) * 2 - 1;
      let x, y;
      if (simple) {
        x = tu * TEX_X_SPAN * 0.5;
        y = TEX_Y_MIN + r * TEX_Y_SPAN;
      } else {
        // A width floor keeps the stem end from collapsing into a degenerate fan.
        const w = Math.max(petalHalfWidth(tv), HALF_W * 0.15) * MESH_OVERSHOOT;
        x = tu * w;
        const notch = (1 - petalTipY(x)) * MESH_NOTCH;
        y = r === 0 ? -0.05 : tv * (1 - notch * smoothstep(0.25, 1, tv));
      }
      position[p++] = x; position[p++] = y - PIVOT_Y; position[p++] = 0;
      uv[q++] = x / TEX_X_SPAN + 0.5;
      uv[q++] = (y - TEX_Y_MIN) / TEX_Y_SPAN;
    }
  }
  let t = 0;
  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const a = r * cols + c, b = a + 1, d = a + cols, e = d + 1;
      index[t++] = a; index[t++] = d; index[t++] = b;
      index[t++] = b; index[t++] = d; index[t++] = e;
    }
  }
  return { position, uv, index };
}

// ---------------------------------------------------------------------------
// Shaders - GLSL1 style, three transpiles for WebGL2
// ---------------------------------------------------------------------------

const PETAL_VERT = /* glsl */ `
#include <common>
#include <fog_pars_vertex>

attribute vec3 aOffset;
attribute vec3 aAxisY;
attribute vec3 aNormal;
attribute vec4 aParams;   // length, width, alpha, tint
attribute vec3 aShape;    // curl, age, flex phase

uniform float uTime;
uniform float uSizeAtten;
uniform float uMinPixels;
uniform float uDetail;

varying vec2 vUv;
varying vec3 vNormalW;
varying vec3 vWorldPos;
varying vec3 vShade;      // alpha, tint, age

const float PIVOT_Y = ${PIVOT_Y.toFixed(4)};

void main() {
  vec3 axisY = aAxisY;              // petal long axis == tumble axis
  vec3 axisZ = aNormal;             // face normal
  vec3 axisX = cross(axisY, axisZ);

  // Sub-pixel petals are pure aliasing. Grow them to a floor of uMinPixels and
  // pay the energy back through alpha, so the far field dissolves into the
  // atmosphere instead of sizzling. (The standard particle min-screen-size trick.)
  float depth = max(-(viewMatrix * vec4(aOffset, 1.0)).z, 0.05);
  float px = aParams.x * uSizeAtten / depth;
  float grow = clamp(uMinPixels / max(px, 1e-4), 1.0, 4.0);
  float sl = aParams.x * grow;
  float sw = aParams.y * grow;

  float u = position.x;
  float v = position.y;
  float vv = v + PIVOT_Y;

  // Cup across the petal plus a gentle lengthwise bend, breathing in the airflow.
  // Real petals are never flat; a flat card reads as a sticker no matter how good
  // the lighting is.
  float curl = aShape.x * (1.0 + uDetail * 0.38 * sin(uTime * (4.1 + fract(aShape.z) * 2.4) + aShape.z));
  float cupA = curl * 1.15;
  float bendA = curl * 0.2;
  float z = -cupA * u * u * (0.35 + 0.65 * vv) + bendA * vv * vv;
  float dzdu = -2.0 * cupA * u * (0.35 + 0.65 * vv);
  float dzdv = -cupA * u * u * 0.65 + 2.0 * bendA * vv;

  // Analytic normal of the deformed surface, in the anisotropically scaled frame.
  vec3 nLocal = normalize(vec3(-dzdu * sl / max(sw, 1e-4), -dzdv, 1.0));

  vec3 world = aOffset + axisX * (u * sw) + axisY * (v * sl) + axisZ * (z * sl);
  vNormalW = normalize(axisX * nLocal.x + axisY * nLocal.y + axisZ * nLocal.z);
  vWorldPos = world;
  vUv = uv;
  vShade = vec3(aParams.z / (grow * grow), aParams.w, aShape.y);

  vec4 mvPosition = viewMatrix * vec4(world, 1.0);
  #include <fog_vertex>
  gl_Position = projectionMatrix * mvPosition;
}
`;

const PETAL_FRAG = /* glsl */ `
#include <common>
#include <fog_pars_fragment>

uniform sampler2D uMap;
uniform vec3 uKeyDir;
uniform vec3 uKeyColor;
uniform vec3 uSkyUp;
uniform vec3 uSkyDown;
uniform float uAmbient;
uniform vec3 uPale;
uniform vec3 uDeep;
uniform vec3 uAged;
uniform vec4 uCanopy;
uniform float uCanopyShade;
uniform float uWetness;

varying vec2 vUv;
varying vec3 vNormalW;
varying vec3 vWorldPos;
varying vec3 vShade;

/**
 * Analytic canopy shadow: how much blossom mass sits between this petal and the
 * key light. Ten ALU instead of a shadow-map tap, and it is what makes petals
 * visibly brighten as they drift out from under the tree.
 */
float canopyShade(vec3 p, vec3 L) {
  vec3 d = uCanopy.xyz - p;
  float t = dot(d, L);
  float dist = length(d - L * max(t, 0.0));
  float inside = 1.0 - smoothstep(uCanopy.w * 0.45, uCanopy.w * 1.05, dist);
  return 1.0 - inside * uCanopyShade * step(0.0, t);
}

void main() {
  vec4 tex = texture2D(uMap, vUv);
  float alpha = tex.a * vShade.x;
  if (alpha < 0.008) discard;

  float facing = gl_FrontFacing ? 1.0 : 0.0;
  vec3 N = normalize(vNormalW) * (facing * 2.0 - 1.0);
  vec3 V = normalize(cameraPosition - vWorldPos);
  vec3 L = uKeyDir;
  float thickness = tex.g;

  // --- albedo ------------------------------------------------------------
  vec3 albedo = mix(uPale, uDeep, tex.b * vShade.y);
  albedo *= 0.70 + 0.42 * tex.r;
  // The underside of a petal is paler and less saturated than the face.
  float luma = dot(albedo, vec3(0.2126, 0.7152, 0.0722));
  albedo = mix(mix(albedo, vec3(luma), 0.30) * 1.05, albedo, facing);
  albedo = mix(albedo, uAged, vShade.z * 0.7);
  albedo *= 1.0 - 0.20 * uWetness;

  float shade = canopyShade(vWorldPos, L);

  // --- direct: wrapped diffuse, because a 0.1 mm plate has no hard terminator
  float wrapD = clamp((dot(N, L) + 0.45) / 1.45, 0.0, 1.0);
  vec3 direct = albedo * (1.0 / PI) * uKeyColor * wrapD * shade;

  // --- transmission: the whole reason petals are beautiful ----------------
  // Frostbite's approximate translucency: light leaving the far face toward the
  // eye, strongest when the key is behind the petal and the petal is thin.
  float lt = clamp(dot(V, -(L + N * 0.14)), 0.0, 1.0);
  lt = pow(lt, 3.5) * (1.0 - thickness * 0.8);
  vec3 transmit = albedo * uKeyColor * lt * 1.35 * shade;

  // --- specular: a faint waxy sheen, much stronger when wet ---------------
  vec3 H = normalize(L + V);
  float spec = pow(max(dot(N, H), 0.0), mix(26.0, 110.0, uWetness)) * mix(0.025, 0.14, uWetness);

  // --- ambient: a thin two-sided plate is lit from BOTH hemispheres --------
  float hemi = 0.5 + 0.5 * N.y;
  vec3 front = mix(uSkyDown, uSkyUp, hemi);
  vec3 back = mix(uSkyDown, uSkyUp, 1.0 - hemi);
  vec3 indirect = albedo * (1.0 / PI) * (front + back * (1.0 - thickness) * 0.7) * uAmbient;

  // Grazing sheen picks the petal edge out against the sky.
  float fres = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 4.0);
  indirect += uSkyUp * fres * 0.06 * uAmbient;

  gl_FragColor = vec4(direct + transmit + uKeyColor * spec * shade + indirect, alpha);

  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  #include <fog_fragment>
  #include <dithering_fragment>
}
`;
