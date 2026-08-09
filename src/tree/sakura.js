/**
 * tree/sakura.js - the hero asset: one veteran Prunus x yedoensis at the origin.
 *
 * `branches.js` grows the skeleton and sweeps the wood into one welded mesh;
 * `blossoms.js` turns the twig cloud into 122 400 instanced flowers, 102 800 of
 * them drawn at HIGH. This file is the conductor: it chooses the tree, stands
 * it on the terrain, dresses the wood in cherry bark, drives everything with a
 * physical wind model, and publishes the interface the rest of the world reads.
 *
 * WHY SEED 0x178f4d88 - the macro structure is hand-authored inside
 * branches.js (eleven primary limbs off a short bole, grown against an explicit
 * crown envelope), but the seed still decides gnarl, bud placement and which
 * shoots survive crowding.
 *
 * MEASURE THE TREE BEFORE YOU BELIEVE THIS BLOCK. It has been wrong twice: once
 * describing a tree that had not existed for some time, and once describing an
 * upright vase after the brief had asked three times for a broad dome. There is
 * a headless harness for exactly this - `generateSkeleton`, `collectTwigSites`
 * and `generateBlossomInstances` are all pure and all run under plain node with
 * no GL context. Re-measured as of the crown-envelope rebuild:
 *
 *   height        9.6 m
 *   crown         12.2 m across (the envelope; the flower cloud's 95th
 *                 percentile reaches 11.7 m) by 7.7 m of vertical extent - 
 *                 1.58 wider than tall, widest band at 4.8 m, which is 0.38
 *                 of the way up a crown whose floor is 1.9 m
 *   crown floor   1.9 m: the skirt hangs to head height
 *   bole          0.96 m at breast height, ~2.2 m across the root buttresses
 *   branches      15 200, carrying 5 060 m of wood - 4 650 m of it flowering - 
 *                 on 145 000 twig sites
 *   blossom       122 400 cards, 102 800 of them drawn at HIGH, 482 m² of quad
 *   coverage      71 % of the crown's own outline is opaque, measured by
 *                 rasterising the flower cloud orthographically at 12 mm and
 *                 dividing by a per-row hull trimmed to its 3rd and 97th
 *                 percentile; the sky that remains is at the margin and in the
 *                 top third, which is where the reference has it
 *   build         about 730 ms once, at init: 480 ms skeleton, 35 ms twig
 *                 walk, 190 ms blossom placement, 25 ms mesh
 *
 * The old seed audit that used to be quoted here is no longer meaningful: it
 * scored 224 candidates on a silhouette this tree no longer has, and its two
 * strongest terms ("no gap in any 0.5 m band", "mass sits mid-crown") were both
 * written against a tall crown. The shape is now set by the ENVELOPE in
 * branches.js rather than by which seed happened to sprawl the right way, which
 * is the point of having one: sweeping this seed across a dozen values moves the
 * crown's width by under 4 % and its height by under 6 %, where it used to move
 * them by 30 %. The seed is kept because it is what every capture in
 * /captures was taken with, not because it is load-bearing any more.
 *
 * WIND - the bend of the whole crown is a real second-order system (spring,
 * damper, turbulent forcing) driven by `wind.getGustAt` sampled at the canopy
 * centre. That is the same travelling-front field the grass reads, so a gust
 * crosses the field, bends the grass and reaches the tree in the right order
 * and at the right moment. Because the crown is a spring rather than a
 * follower, it overshoots a front and rings down afterwards, which is most of
 * what separates a tree from a flag. Above it sit branches.js's four
 * hierarchical oscillators, each amplitude tracking the wind through its own
 * first-order filter: a limb takes a second to wind up, a twig responds
 * instantly, and the levels lag each other without a scripted phase offset.
 */

import * as THREE from 'three';
import { QUALITY } from '../core/state.js';
import { LAYERS } from '../core/engine.js';
import { clamp, clamp01, lerp, smoothstep, damp, createNoise, TAU } from '../core/math.js';
import {
  generateSkeleton,
  buildBranchGeometry,
  collectTwigSites,
  createSakuraWindUniforms,
  SAKURA_WIND_GLSL,
  LOD_TIERS,
} from './branches.js';
import { BlossomField } from './blossoms.js';

// ===========================================================================
// Tuning
// ===========================================================================

/** See the header. Do not change casually - the silhouette is the project. */
const TREE_SEED = 0x178f4d88;
const BLOSSOM_SEED = 0x5a11a9;

/**
 * Twig sampling pitch, in metres of arc along the flowering wood.
 *
 * 22 mm was a number that did nothing. `collectTwigSites` used to visit
 * skeleton POINTS and reject any that fell inside the pitch, so it could never
 * place sites closer than `LEVELS[].segLen` - 92 mm apart on depth-3 wood, four
 * times what was being asked for. The parameter has only been live since
 * branches.js started interpolating along the segments.
 *
 * 34 mm is a real choice, and it is set by the flower rather than by the mesh:
 * a card is about 68 mm across, so sites at 34 mm overlap by half and the wood
 * is clothed continuously instead of in beads. This is the dial that buys
 * canopy coverage EFFICIENTLY - a site is a new place, whereas the cluster
 * multiplier only stacks more flowers on the places that already have them,
 * where the cards fall on top of each other and buy nothing.
 *
 * Note what it now means. With 3 600 m of flowering wood the walk produces
 * 136 500 sites and blossoms.js accepts fewer than one umbel per site, so the
 * pitch is no longer "where a flower goes" - it is the resolution at which the
 * wood is SAMPLED, and the Bernoulli acceptance inside `generateBlossomInstances`
 * turns that into a stochastic scatter along the twig. Coarsening it to 50 mm
 * would save about 60 ms of build time and visibly quantise the flowers onto a
 * comb; it is the one place in this file where the extra build time is bought
 * back directly in how continuous the canopy looks.
 */
const TWIG_SPACING = 0.034;
const TWIG_MAX_RADIUS = 0.048;

/**
 * Crown fundamental. A 11 m broadleaf sways at roughly 0.35-0.45 Hz; 2.5 rad/s
 * is 0.40 Hz. Damping ratio for a tree in leaf is measured at 0.1-0.25 - low
 * enough that it visibly rings after a gust passes, which is the point.
 */
const CROWN_OMEGA = 2.45;
const CROWN_ZETA = 0.21;

/**
 * Crown drag. Force goes as speed squared, but foliage streamlines as it
 * loads, so the response saturates - this is why doubling a storm's wind speed
 * moves a tree by 17% rather than 300%. The constant is fixed by requiring
 * ~1 m of crown travel at a storm's 14 m/s canopy wind, which works out at
 * 5 cm two metres up the bole: the trunk barely moves, exactly as it should.
 */
const BEND_K = 0.0021;
const BEND_SAT = 0.020;
/** Drag at a full storm; everything downstream is expressed as a fraction of it. */
const DRIVE_REF = 35;

/** Natural frequencies of the four oscillator levels, in rad/s of the shader
 *  clock. Held CONSTANT: retuning them at runtime would jump every phase by
 *  t*delta-f, which reads as the whole canopy snapping. The global clock RATE
 *  carries the "everything moves faster in a gale" instead, and that is
 *  continuous by construction. */
const SWAY_FREQ = [1.05, 1.95, 3.50, 6.20];
/** Response time constants per level. Limbs wind up slowly, twigs do not. */
const SWAY_LAMBDA = [0.95, 1.7, 3.2, 6.5];
/**
 * Amplitude in metres of travel: [still-air floor, gain per unit of drive].
 * These are the peak excursions of the most compliant point at each level - 
 * branches.js normalises its sway weights to 1 there - so they can be read
 * straight off a photograph rather than tuned by eye.
 */
const SWAY_AMP = [
  [0.060, 0.340],
  [0.035, 0.200],
  [0.018, 0.115],
  [0.008, 0.065],
];

/**
 * Distance thinning of the blossom instance count. Deliberately paired with
 * blossoms.js's alpha-cutoff relaxation window (10-65 m): that relaxation
 * gives each surviving card more coverage as the mip chain eats its silhouette,
 * and this removes cards over roughly the same span, so the canopy's total
 * projected coverage stays flat instead of swelling into a blob.
 *
 * TIGHTENED (30/100/0.45 -> 22/80/0.32) to help pay for a canopy that carries
 * 55 % more instances than it used to. The hero framing is at 20 m, which is
 * inside LOD_NEAR, so nothing about the shot this asset exists for is thinned;
 * everything past it is. At 80 m the crown is about 150 px across and a third
 * of the flowers reproduce it exactly.
 */
const LOD_NEAR = 22;
const LOD_FAR = 80;
const LOD_MIN_FRACTION = 0.32;
/** Instance count is quantised to this so the count never churns frame to frame. */
const LOD_QUANTUM = 512;

/** Petal spawn points refreshed per frame. 1449 points cycle in ~5 frames. */
const SPAWN_SLICE = 288;

/**
 * Ceiling on each oscillator level as a fraction of the level above it.
 *
 * The four levels chase their targets through filters an order of magnitude
 * apart (SWAY_LAMBDA 0.95 -> 6.5), so for roughly the first second of a gust
 * front the fast levels are already near their storm amplitude while the slow
 * ones are still winding up: at t = 0.3 s after a step to a storm, level 2
 * reaches 0.222 m against level 1's 0.195 m. That is exactly the branchlet
 * out-travelling the branch that carries it, and it reads as jelly. In STEADY
 * state the worst ratio any pair reaches is 0.807 (level 2 over level 1, at
 * full drive with turbulence and storminess both at 1), so a 0.9 ceiling is
 * inert once the tree has settled and only ever bites during the transient.
 */
const AMP_HIERARCHY = 0.9;

// ===========================================================================
// Module scratch - nothing in update() allocates
// ===========================================================================

const _camPos = new THREE.Vector3();

/**
 * Finite-or-fallback.
 *
 * Everything below `_syncWind`'s inputs feeds DAMPED ACCUMULATORS that are read
 * back the next frame (`_bendX/_bendZ/_bendVX/_bendVZ`, `_amp[]`, `_clock`,
 * `_storm`, `_noiseT`), and `damp(NaN, x)` is NaN forever. Those accumulators
 * become `uSakuraBend` / `uSakuraAmp` / `uSakuraTime`, and `sakuraBend` turns a
 * NaN uniform into a NaN position for every vertex of the wood mesh AND all
 * 67 000 blossom cards - so a single bad frame in `state.wind.strength` (or
 * gust, direction, turbulence, storminess, rainIntensity) takes the hero asset
 * off the screen permanently, until reload. Sanitising on the way in, plus the
 * accumulator guards below, closes every one of those paths.
 */
const fin = (v, fallback) => (Number.isFinite(v) ? v : fallback);

// ===========================================================================

export class SakuraTree {
  constructor(ctx) {
    this.ctx = ctx;
    this.scene = ctx.scene;
    this.state = ctx.state;

    this.root = new THREE.Group();
    this.root.name = 'sakura';
    this.root.matrixAutoUpdate = true;

    // --- published interface (CONTRACTS section 6) --------------------------
    /** Trunk base, world space. */
    this.position = new THREE.Vector3(0, 0, 0);
    this.canopyCenter = new THREE.Vector3(0, 6, 0);
    this.canopyRadius = 5;
    /** Total height of the tree above its base, in metres. */
    this.canopyHeight = 11;

    this.ready = false;
    this.skeleton = null;
    this.wood = null;
    this.blossoms = null;

    this.terrain = null;
    this.windSystem = null;

    // --- wind state --------------------------------------------------------
    this.windU = createSakuraWindUniforms();
    this._clock = 0;
    this._bendX = 0; this._bendZ = 0;
    this._bendVX = 0; this._bendVZ = 0;
    this._amp = new Float32Array(4);
    this._drive = 0;
    this._storm = 0;
    this._gustNoise = createNoise(TREE_SEED ^ 0x77ab);
    this._noiseT = 0;

    // --- petal spawn cache -------------------------------------------------
    this._spawnWorld = new Float32Array(0);
    this._spawnCount = 0;
    this._spawnRest = null;
    this._spawnW = null;      // 4 sway weights per point
    this._spawnCos = null;    // cos/sin of the two phase sets, 8 per point
    this._spawnSin = null;
    this._spawnCos2 = null;
    this._spawnSin2 = null;
    this._spawnCursor = 0;
    this._trigS = new Float32Array(4);
    this._trigC = new Float32Array(4);
    this._trigS2 = new Float32Array(4);
    this._trigC2 = new Float32Array(4);

    this._lodFraction = 1;
    this._lastInstanceCount = -1;
    this._wireframe = false;
    this._tier = ctx.state.quality.tier || QUALITY.HIGH;

    /** 0 when the moss bake failed, so the moss term costs nothing and shows
     *  nothing rather than painting the tree with a 1x1 stand-in. */
    this._mossGain = 1;
    /** 1x1 stand-ins created only when a bake failed; ours to dispose. */
    this._fallbackTex = null;
  }

  // =========================================================================
  // Build
  // =========================================================================

  async init() {
    const state = this.state;
    this._tier = state.quality.tier || QUALITY.HIGH;

    this.skeleton = generateSkeleton({ seed: TREE_SEED });
    const skel = this.skeleton;

    this.canopyHeight = Math.max(skel.height, 1);
    this.canopyRadius = Math.max(skel.canopyRadius, 1);
    // The shader's bend profile is pinned at y = 0 and normalised by height, so
    // this uniform has to be the tree's OWN height, not a constant.
    this.windU.uSakuraExtra.value.x = 1 / this.canopyHeight;
    this.windU.uSakuraFreq.value.set(SWAY_FREQ[0], SWAY_FREQ[1], SWAY_FREQ[2], SWAY_FREQ[3]);

    // --- wood --------------------------------------------------------------
    this.barkMaterial = this._makeBarkMaterial();
    this.barkDepthMaterial = this._makeBarkDepthMaterial();
    const geo = this._buildWoodGeometry(this._lodKey(this._tier));
    const wood = new THREE.Mesh(geo, this.barkMaterial);
    wood.name = 'sakura.wood';
    wood.castShadow = true;
    wood.receiveShadow = true;
    wood.customDepthMaterial = this.barkDepthMaterial;
    wood.matrixAutoUpdate = false;
    wood.layers.enable(LAYERS.GODRAY_OCCLUDER);
    this.wood = wood;
    this.root.add(wood);

    // --- blossom ------------------------------------------------------------
    const sites = collectTwigSites(skel, {
      minDepth: 2,
      spacing: TWIG_SPACING,
      maxRadius: TWIG_MAX_RADIUS,
    });
    this.blossoms = new BlossomField(this.ctx, {
      windUniforms: this.windU,
      seed: BLOSSOM_SEED,
    }).build(skel, sites);
    this.blossoms.setQuality(this._lodKey(this._tier));
    this.blossoms.mesh.layers.enable(LAYERS.GODRAY_OCCLUDER);
    this.root.add(this.blossoms.mesh);

    this.scene.add(this.root);
    // Stand the tree up BEFORE the spawn cache is primed: `_refreshSpawn` adds
    // `root.position` to every point, so priming first published 1 530 petal
    // spawn points sitting at y = 0 instead of on the terrain. link() refreshes
    // them and petals.js links after us, so nothing consumed the stale values - 
    // but getBlossomSpawnPoints() is a published interface and must not hand
    // out wrong world positions to anything that asks between init and link.
    this._placeOnTerrain();
    this._buildSpawnCache(this.blossoms.spawn);
    this.ready = true;
  }

  /**
   * Grabs siblings. Ordering in main.js guarantees terrain exists here and that
   * petals.js links AFTER us, so the spawn points it caches are already in
   * their final world position.
   */
  link(systems) {
    this.terrain = systems.terrain || null;
    this.windSystem = systems.wind || null;
    this._placeOnTerrain();

    // The tree is the scene's one real shaft occluder. atmosfx re-tags the root
    // on every sweep, so children added later are picked up automatically; the
    // explicit layer bits above cover a build where atmosfx is not present.
    const fx = systems.atmosfx;
    if (fx && typeof fx.markOccluder === 'function') {
      try {
        fx.markOccluder(this.root);
      } catch (err) {
        console.warn('[SakuraTree] markOccluder failed:', err);
      }
    }
    // A pipeline that wants occluders handed to it directly gets them; the
    // sun-body overload of registerGodRayLight is deliberately NOT used here - 
    // passing a tree to it would register the canopy as a light source.
    const post = systems.post;
    if (post && typeof post.registerGodRayOccluder === 'function') {
      try {
        post.registerGodRayOccluder(this.root);
      } catch (err) {
        console.warn('[SakuraTree] registerGodRayOccluder failed:', err);
      }
    }

    // Prime the wind uniforms so the first rendered frame is not a rest pose
    // snapping into motion.
    this._syncWind(1 / 60, this.state, true);
    this._refreshSpawn(this._spawnCount);
  }

  /**
   * Wood mesh for a tier, with the cull volume widened past what branches.js
   * allows for. Its own +1.2 m covers ordinary sway; a storm gust adds crown
   * bend overshoot on top of that, and a hero asset popping out at the edge of
   * the screen is the most expensive kind of artefact there is.
   */
  _buildWoodGeometry(tier) {
    const geo = buildBranchGeometry(this.skeleton, tier);
    if (geo.boundingSphere) geo.boundingSphere.radius += 1.6;
    return geo;
  }

  /** Stand the trunk base on the ground and republish every world-space field. */
  _placeOnTerrain() {
    let y = 0;
    // Reachable from init() as well as link(): terrain.init() runs before ours
    // in main.js, so the tree never spends a frame hovering at y = 0.
    const t = this.terrain || this.ctx.systems?.terrain || null;
    if (t && typeof t.getHeight === 'function') {
      const h = t.getHeight(0, 0);
      if (Number.isFinite(h)) y = h;
    }
    this.root.position.set(0, y, 0);
    this.root.updateMatrixWorld(true);
    this.position.set(0, y, 0);
    if (this.skeleton) {
      this.canopyCenter.copy(this.skeleton.canopyCenter).add(this.root.position);
    }
  }

  // =========================================================================
  // Bark
  // =========================================================================

  /** Every map the bark needs, with a per-suite guard: a texture bake that
   *  throws must cost us that one map, not the whole tree. */
  _textureSuite() {
    const tex = this.ctx.textures;
    const out = { albedo: null, normal: null, orm: null, moss: null, noise: null };
    if (!tex) return out;
    const grab = (label, fn) => {
      try {
        return fn();
      } catch (err) {
        console.warn(`[SakuraTree] ${label} unavailable:`, err);
        return null;
      }
    };
    const bark = grab('bark suite', () => (tex.barkSet ? tex.barkSet() : null));
    out.albedo = bark ? bark.albedo : grab('barkAlbedo', () => tex.get?.('barkAlbedo'));
    out.normal = bark ? bark.normal : grab('barkNormal', () => tex.get?.('barkNormal'));
    out.orm = bark ? bark.orm : grab('barkORM', () => tex.get?.('barkORM'));
    out.moss = grab('moss suite', () =>
      (tex.mossSet ? tex.mossSet().albedo : tex.get?.('mossAlbedo')));
    out.noise = grab('utility noise', () => tex.get?.('noise2d'));
    return out;
  }

  /**
   * 1x1 stand-in for a map whose bake threw.
   *
   * The per-suite try/catch above is only worth having if a missing map costs
   * us that map and nothing else - and a NULL sampler uniform does the
   * opposite. three binds its empty texture, which reads (0,0,0,1), so a
   * missing ORM would drive `roughnessFactor` to the 0.045 floor and `barkAO`
   * to zero: a mirror-smooth trunk with no indirect light on it at all. A
   * neutral texel makes the corresponding term an identity instead.
   */
  _neutralTex(r, g, b, a) {
    const t = new THREE.DataTexture(
      new Uint8Array([r, g, b, a]), 1, 1, THREE.RGBAFormat, THREE.UnsignedByteType
    );
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.needsUpdate = true;
    (this._fallbackTex || (this._fallbackTex = [])).push(t);
    return t;
  }

  _makeBarkMaterial() {
    const t = this._textureSuite();
    this._barkTex = t;
    // AO 1, roughness 1, metal 0, cavity 0 - every ORM-driven term becomes a
    // no-op rather than a black hole.
    const ormTex = t.orm || this._neutralTex(255, 255, 0, 0);
    // Mid grey: the moss patch field lands mid-range instead of collapsing.
    const noiseTex = t.noise || this._neutralTex(128, 128, 128, 255);
    this._mossGain = t.moss ? 1 : 0;
    const mossTex = t.moss || this._neutralTex(255, 255, 255, 0);

    const mat = new THREE.MeshStandardMaterial({
      map: t.albedo || null,
      normalMap: t.normal || null,
      // Roughness and occlusion come out of a SINGLE hand-rolled ORM fetch
      // below rather than three's roughnessMap + aoMap, which would sample the
      // same texture twice and still not expose the alpha channel the bark
      // suite stores its wetness affinity in.
      roughness: 1.0,
      metalness: 0.0,
      color: 0xffffff,
      fog: true,
      dithering: true,
    });
    mat.name = 'sakura.bark';
    if (mat.normalMap) mat.normalScale.set(1.0, 1.0);

    const u = {
      uBarkOrm: { value: ormTex },
      uBarkMoss: { value: mossTex },
      uBarkNoise: { value: noiseTex },
      /** x amount, y low edge, z high edge, w texture scale */
      uMossParams: { value: new THREE.Vector4(0.85 * this._mossGain, 0.16, 0.44, 1.9) },
      /**
       * Multiplicative tint pushing first-year shoots toward the warm
       * mahogany-red of new cherry wood. Kept as a ratio rather than a colour
       * so it rides whatever the bark albedo is doing locally instead of
       * flattening the twigs to one hue. Authored in linear space - it is a
       * gain, not a colour, so it must not be sRGB-decoded.
       */
      uBarkYoung: { value: new THREE.Vector3(1.42, 1.12, 0.98) },
      /** x wetness, y darkening, z gloss */
      uBarkWet: { value: new THREE.Vector3(0, 0.52, 0.13) },
      /** Grazing sky reflection - the satin sheen cherry bark actually has. */
      uBarkSheen: { value: new THREE.Color(0, 0, 0) },
      uBarkSheenAmt: { value: 0.05 },
      /** Normal-map fade window in metres, for specular anti-aliasing. */
      uBarkFade: { value: new THREE.Vector2(16, 80) },
      /**
       * Linear-space gain on the bark albedo. Warm, slightly red, and below 1
       * overall so the bole reads as dark wood against a bright sky instead of
       * the pale concrete grey it was.
       */
      uBarkTint: { value: new THREE.Vector3(0.76, 0.58, 0.47) },
    };
    this.barkU = u;

    mat.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, u, this.windU);
      shader.vertexShader = patchBarkVertex(shader.vertexShader);
      shader.fragmentShader = patchBarkFragment(shader.fragmentShader);
    };
    mat.customProgramCacheKey = () => 'sakuraBark';
    return mat;
  }

  /**
   * A depth material that bends EXACTLY the way the bark does. Without this the
   * tree's shadow stands perfectly still while the tree thrashes - the engine's
   * own docs call it out, and it is instantly visible on the ground under a
   * cherry, where the shadow is most of what you are looking at.
   */
  _makeBarkDepthMaterial() {
    const mat = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking });
    mat.name = 'sakura.bark.depth';
    mat.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, this.windU);
      shader.vertexShader = patchBarkVertex(shader.vertexShader, true);
    };
    mat.customProgramCacheKey = () => 'sakuraBarkDepth';
    return mat;
  }

  // =========================================================================
  // Petal spawn points  (CONTRACTS section 6)
  // =========================================================================

  /**
   * Caches everything needed to move the spawn points with the canopy.
   *
   * The oscillator's phase terms are constant per point, so their sines and
   * cosines are baked here and the per-frame update becomes an angle-sum
   * expansion: zero transcendentals per point per frame, which is what makes
   * animating 1500 spawn points on the CPU free rather than merely cheap.
   */
  _buildSpawnCache(spawn) {
    const n = spawn ? spawn.count : 0;
    this._spawnCount = n;
    this._spawnWorld = new Float32Array(n * 3);
    if (n === 0) return;

    this._spawnRest = new Float32Array(n * 3);
    this._spawnW = new Float32Array(n * 4);
    this._spawnCos = new Float32Array(n * 4);
    this._spawnSin = new Float32Array(n * 4);
    this._spawnCos2 = new Float32Array(n * 4);
    this._spawnSin2 = new Float32Array(n * 4);

    for (let i = 0; i < n; i++) {
      const o3 = i * 3, o4 = i * 4;
      this._spawnRest[o3] = spawn.position[o3];
      this._spawnRest[o3 + 1] = spawn.position[o3 + 1];
      this._spawnRest[o3 + 2] = spawn.position[o3 + 2];

      // Weights and phases are packed exactly the way sakuraOsc unpacks them:
      // w = (swA.x, swA.z, swB.x, swB.z), ph = (swA.y, swA.w, swB.y, swB.w).
      const w0 = spawn.swayA[o4], p0 = spawn.swayA[o4 + 1];
      const w1 = spawn.swayA[o4 + 2], p1 = spawn.swayA[o4 + 3];
      const w2 = spawn.swayB[o4], p2 = spawn.swayB[o4 + 1];
      const w3 = spawn.swayB[o4 + 2], p3 = spawn.swayB[o4 + 3];
      this._spawnW[o4] = w0; this._spawnW[o4 + 1] = w1;
      this._spawnW[o4 + 2] = w2; this._spawnW[o4 + 3] = w3;
      const ph = [p0 * TAU, p1 * TAU, p2 * TAU, p3 * TAU];
      for (let k = 0; k < 4; k++) {
        this._spawnCos[o4 + k] = Math.cos(ph[k]);
        this._spawnSin[o4 + k] = Math.sin(ph[k]);
        const b = ph[k] * 1.31 + 1.7;
        this._spawnCos2[o4 + k] = Math.cos(b);
        this._spawnSin2[o4 + k] = Math.sin(b);
      }
    }
    this._refreshSpawn(n);
  }

  /**
   * World-space positions of live blossoms on the outer canopy.
   * Stable identity: the array is allocated once and mutated in place, so
   * petals.js can cache the reference.
   * @returns {Float32Array} xyz triplets
   */
  getBlossomSpawnPoints() {
    return this._spawnWorld;
  }

  /** Advance `count` spawn points from the rolling cursor. Allocation-free. */
  _refreshSpawn(count) {
    const n = this._spawnCount;
    if (!n) return;

    const amp = this._amp;
    const t = this._clock;
    const dirX = this.windU.uSakuraDir.value.x;
    const dirZ = this.windU.uSakuraDir.value.y;
    const perpX = -dirZ, perpZ = dirX;
    const bendX = this._bendX, bendZ = this._bendZ;
    const invH = this.windU.uSakuraExtra.value.x;
    const ox = this.root.position.x, oy = this.root.position.y, oz = this.root.position.z;

    // Per-level clock terms, shared by every point this frame.
    for (let k = 0; k < 4; k++) {
      const a = t * SWAY_FREQ[k];
      this._trigS[k] = Math.sin(a);
      this._trigC[k] = Math.cos(a);
      const b = t * SWAY_FREQ[k] * 0.607;
      this._trigS2[k] = Math.sin(b);
      this._trigC2[k] = Math.cos(b);
    }
    const tS = this._trigS, tC = this._trigC, tS2 = this._trigS2, tC2 = this._trigC2;

    const rest = this._spawnRest, W = this._spawnW;
    const cP = this._spawnCos, sP = this._spawnSin;
    const cQ = this._spawnCos2, sQ = this._spawnSin2;
    const out = this._spawnWorld;

    let i = this._spawnCursor;
    const total = Math.min(count, n);
    for (let k = 0; k < total; k++) {
      const o3 = i * 3, o4 = i * 4;
      const px = rest[o3], py = rest[o3 + 1], pz = rest[o3 + 2];

      // --- sakuraBend, verbatim from the GLSL ------------------------------
      const len = Math.sqrt(px * px + py * py + pz * pz);
      let bx = px, by = py, bz = pz;
      if (len > 1e-4) {
        let f = py * invH + 1;
        f *= f;
        f = f * f - f;
        const qx = px + bendX * f;
        const qz = pz + bendZ * f;
        const ql = Math.sqrt(qx * qx + py * py + qz * qz) || 1;
        const s = len / ql;
        bx = qx * s; by = py * s; bz = qz * s;
      }

      // --- sakuraOsc, expanded by angle-sum so no trig runs per point ------
      //   along = dot(a, sin(t*f + ph))          a = amp * weight
      //   side  = dot(a, cos(t*f*0.607 + ph*1.31 + 1.7)) * 0.55
      //   lift  = (a.z*c.z + a.w*s.w) * 0.42
      let along = 0, side = 0, liftC = 0, liftS = 0;
      for (let m = 0; m < 4; m++) {
        const a = amp[m] * W[o4 + m];
        const sm = tS[m] * cP[o4 + m] + tC[m] * sP[o4 + m];
        const cm = tC2[m] * cQ[o4 + m] - tS2[m] * sQ[o4 + m];
        along += a * sm;
        side += a * cm;
        if (m === 2) liftC = a * cm;
        else if (m === 3) liftS = a * sm;
      }
      side *= 0.55;
      const lift = (liftC + liftS) * 0.42;

      out[o3] = ox + bx + dirX * along + perpX * side;
      out[o3 + 1] = oy + by + lift;
      out[o3 + 2] = oz + bz + dirZ * along + perpZ * side;

      i++;
      if (i >= n) i = 0;
    }
    this._spawnCursor = i;
  }

  // =========================================================================
  // Frame
  // =========================================================================

  update(dt, state) {
    if (!this.ready || dt <= 0) return;
    this._syncWind(dt, state, false);

    const cam = this.ctx.camera;
    if (cam) {
      cam.getWorldPosition(_camPos);
      this._updateLod(_camPos);
      this.blossoms.update(state, cam);
    }
    this._syncBark(state);
    this._refreshSpawn(SPAWN_SLICE);

    if (state.debug.wireframe !== this._wireframe) {
      this._wireframe = state.debug.wireframe;
      if (this.barkMaterial) this.barkMaterial.wireframe = this._wireframe;
      if (this.blossoms?.material) this.blossoms.material.wireframe = this._wireframe;
    }
  }

  /**
   * The wind model. Everything here is a physical quantity or a filter over
   * one; there are no scripted animations.
   */
  _syncWind(dt, state, snap) {
    const w = state.wind;
    const u = this.windU;
    const step = clamp(fin(dt, 1 / 60), 1 / 1000, 1 / 20);

    // --- the shared gust field --------------------------------------------
    // Sampled at the canopy centre so the tree and the grass field are reading
    // the SAME travelling front at the same world position. That coherence is
    // the entire reason wind.js exposes this function.
    let gust = fin(w.gust, 1);
    const gustAt = this.windSystem?.getGustAt || w.getGustAt;
    if (typeof gustAt === 'function') {
      const g = gustAt(this.canopyCenter.x, this.canopyCenter.z, state.time.elapsed);
      if (Number.isFinite(g)) gust = g;
    }
    // Log-law shear: the crown sits in faster air than the reference height.
    let shear = 1.35;
    const shearAt = this.windSystem?.shearAt || w.shearAt;
    if (typeof shearAt === 'function') {
      const s = shearAt(this.canopyCenter.y - this.position.y);
      if (Number.isFinite(s) && s > 0) shear = s;
    }
    const speed = Math.max(0, fin(w.strength, 0)) * Math.max(0, gust) * shear;

    // --- drive: saturating quadratic drag ---------------------------------
    const q = speed * speed;
    const drag = q / (1 + q * BEND_SAT);
    this._drive = drag;
    // Normalised so 1.0 is "a proper storm"; used for every amplitude below.
    const f = clamp(drag / DRIVE_REF, 0, 2.0);

    const turb = clamp01(fin(w.turbulence, 0));
    const storm = clamp01(
      Math.max(fin(state.clouds.storminess, 0), fin(state.weather.rainIntensity, 0) * 0.8)
    );
    if (!Number.isFinite(this._storm)) this._storm = storm;
    this._storm = damp(this._storm, storm, 2.0, step);

    // --- crown bend: a real second-order system ---------------------------
    // Renormalised rather than trusted: state documents this as a unit heading,
    // but a zero or NaN vector here would put NaN into uSakuraDir and every
    // vertex on the tree would be discarded by the rasteriser.
    let dirX = fin(w.direction.x, 1), dirZ = fin(w.direction.y, 0);
    const dirLen = Math.hypot(dirX, dirZ);
    if (dirLen > 1e-6) { dirX /= dirLen; dirZ /= dirLen; } else { dirX = 1; dirZ = 0; }
    u.uSakuraDir.value.set(dirX, dirZ);

    const targetX = dirX * drag * BEND_K;
    const targetZ = dirZ * drag * BEND_K;

    // Turbulent buffeting: a broadband force, not a displacement, so the crown
    // filters it through its own dynamics. This is what makes a storm thrash
    // instead of merely leaning further over.
    if (!Number.isFinite(this._noiseT)) this._noiseT = 0;
    this._noiseT += step * (0.55 + 1.5 * this._storm);
    const buffet = (0.15 + 0.75 * this._storm) * (0.35 + 0.65 * turb);
    // Two decorrelated lanes through the same field: same statistics, no
    // shared value at t = 0, and no second noise instance to build.
    const nx = this._gustNoise.noise2D(this._noiseT, 11.3);
    const nz = this._gustNoise.noise2D(this._noiseT + 57.1, 91.7);
    const fx = targetX + nx * buffet * drag * BEND_K;
    const fz = targetZ + nz * buffet * drag * BEND_K;

    if (snap) {
      this._bendX = targetX; this._bendZ = targetZ;
      this._bendVX = 0; this._bendVZ = 0;
    } else {
      // Semi-implicit Euler. omega*dt <= 0.13 at the clamped step, so this is
      // unconditionally stable here and conserves the ringing that explicit
      // Euler would pump up.
      const k = CROWN_OMEGA * CROWN_OMEGA;
      const c = 2 * CROWN_ZETA * CROWN_OMEGA;
      this._bendVX += ((fx - this._bendX) * k - this._bendVX * c) * step;
      this._bendVZ += ((fz - this._bendZ) * k - this._bendVZ * c) * step;
      this._bendX += this._bendVX * step;
      this._bendZ += this._bendVZ * step;
    }
    // Last line of defence: every input above is finite by construction now, so
    // this can only fire if the accumulators were poisoned before the guards
    // existed (a state field mutated mid-frame, a restored session). Snapping
    // to the steady-state target costs one frame of ringing; leaving NaN in
    // uSakuraBend costs the tree for the rest of the session.
    if (!Number.isFinite(this._bendX) || !Number.isFinite(this._bendZ) ||
        !Number.isFinite(this._bendVX) || !Number.isFinite(this._bendVZ)) {
      this._bendX = targetX; this._bendZ = targetZ;
      this._bendVX = 0; this._bendVZ = 0;
    }
    u.uSakuraBend.value.set(this._bendX, this._bendZ);

    // --- oscillator amplitudes, one first-order filter per level -----------
    // Turbulence and storminess shake the FAST levels hardest - a squall makes
    // the twigs chatter long before it moves a limb - but the boost is kept
    // small enough that the hierarchy stays strictly decreasing. A branchlet
    // travelling further than the branch carrying it is the classic failure of
    // hand-tuned tree wind, and it instantly reads as jelly.
    const turbGain = 0.85 + 0.30 * turb;
    for (let i = 0; i < 4; i++) {
      let target = SWAY_AMP[i][0] + SWAY_AMP[i][1] * f;
      if (i >= 2) target *= turbGain * (1 + 0.22 * this._storm * (i - 1));
      let a = snap ? target : damp(this._amp[i], target, SWAY_LAMBDA[i], step);
      if (!Number.isFinite(a)) a = target;
      // Steady state already satisfies this (worst pair ratio 0.807); the
      // ceiling exists for the ~0.8 s after a gust front, when the fast filters
      // have arrived and the slow ones have not. See AMP_HIERARCHY.
      if (i > 0) a = Math.min(a, this._amp[i - 1] * AMP_HIERARCHY);
      this._amp[i] = a;
    }
    u.uSakuraAmp.value.set(this._amp[0], this._amp[1], this._amp[2], this._amp[3]);

    // --- clock -------------------------------------------------------------
    // Rate, not frequency: integrating a changing rate keeps every phase
    // continuous, whereas retuning uSakuraFreq would jump them all at once.
    const rate = 0.85 + 0.34 * Math.min(f, 2.0) + 0.12 * this._storm;
    if (!Number.isFinite(this._clock)) this._clock = 0;
    this._clock += step * rate;
    u.uSakuraTime.value = this._clock;

    // Blossom flutter, in metres. Tiny by design: it is the highest-frequency
    // motion on the tree and reads as shimmer, not as displacement.
    const flutter = clamp(0.0035 + 0.021 * f * turbGain, 0, 0.055);
    u.uSakuraExtra.value.y = this._storm;
    u.uSakuraExtra.value.z = flutter;
  }

  /** Bark uniforms that follow the weather and the sky. */
  _syncBark(state) {
    const u = this.barkU;
    if (!u) return;
    const wet = clamp01(fin(state.weather.wetness, 0));
    u.uBarkWet.value.set(wet, 0.52, 0.13);

    // The sheen is a stand-in for the missing environment reflection, so it
    // has to be sky radiance - otherwise the trunk glows at midnight.
    const sky = state.sky;
    const amb = Math.max(fin(sky.ambientIntensity, 1), 0);
    u.uBarkSheen.value.copy(sky.horizonColor).lerp(sky.zenithColor, 0.35).multiplyScalar(amb);
    // Wet wood is a mirror by comparison with dry wood.
    u.uBarkSheenAmt.value = 0.045 + 0.30 * wet;

    // Moss is a wet-climate feature; it darkens and greens up in the rain.
    // Multiplied by the gain so a failed moss bake shows no moss at all rather
    // than smearing the 1x1 stand-in over the bole.
    u.uMossParams.value.x = (0.72 + 0.22 * wet) * this._mossGain;
  }

  /**
   * Distance LOD on the blossom instance count.
   *
   * At 100 m every flower is well under a pixel, so half of them can go with no
   * visible change - and because the instance order is a shuffle, dropping a
   * suffix is an even thinning rather than a bald patch. What replaces the lost
   * coverage is the alpha-cutoff relaxation in blossoms.js over the same span,
   * not a size boost; the two are tuned against each other precisely so the
   * canopy neither dissolves nor swells. Quantised so the count is not rewritten
   * every frame.
   */
  _updateLod(camPos) {
    const b = this.blossoms;
    if (!b || !b.geometry) return;
    const d = camPos.distanceTo(this.canopyCenter);
    // A non-finite camera position is one frame of someone else's bug; latching
    // it here would be permanent. `smoothstep` propagates NaN, `Math.round(NaN)`
    // is NaN, and `geometry.instanceCount = NaN` draws nothing - and because
    // `NaN === NaN` is false the dead-band below can never settle again, so the
    // canopy would stay gone for the rest of the session even after the camera
    // recovered. This is only reachable on the very first frame (the dead band
    // rejects a NaN target afterwards), which is exactly when it would be worst.
    if (!Number.isFinite(d)) return;
    this._lodFraction = lerp(1, LOD_MIN_FRACTION, smoothstep(LOD_NEAR, LOD_FAR, d));

    const raw = b.activeCount * this._lodFraction;
    // Quantising alone is NOT enough: Math.round flips at exactly (k+0.5)Q, so
    // a camera parked on a rounding boundary toggles a whole quantum - 512
    // flowers - on and off every single frame, which shimmers right across the
    // canopy. Re-quantise only once the unquantised target has drifted 0.75 of
    // a quantum from the count actually in use, giving a 1.5-quantum dead band
    // that ordinary camera jitter cannot cross.
    let count = this._lastInstanceCount;
    if (count < 0 || Math.abs(raw - count) > LOD_QUANTUM * 0.75) {
      count = Math.min(
        b.activeCount,
        Math.max(LOD_QUANTUM, Math.round(raw / LOD_QUANTUM) * LOD_QUANTUM)
      );
    }
    if (count === this._lastInstanceCount) return;
    this._lastInstanceCount = count;
    b.geometry.instanceCount = count;
    // Card size is NOT boosted here - see the note on LOD_NEAR. The tier boost
    // set in blossoms.setQuality() stays untouched.
  }

  // =========================================================================
  // Quality
  // =========================================================================

  _lodKey(tier) {
    return LOD_TIERS[tier] ? tier : 'high';
  }

  onQualityChange(quality) {
    if (!this.ready) return;
    const tier = this._lodKey(quality?.tier || QUALITY.HIGH);
    if (tier === this._tier) return;
    this._tier = tier;

    // The wood is one welded mesh, so a tier change is a rebuild - 40-90 ms,
    // once, on an explicit user action. Swapping the attribute set in place is
    // not an option: ring counts and decimation strides both change.
    const next = this._buildWoodGeometry(tier);
    const old = this.wood.geometry;
    this.wood.geometry = next;
    old.dispose();

    this.blossoms.setQuality(tier);
    this._lastInstanceCount = -1;
  }

  // =========================================================================

  dispose() {
    this.ready = false;
    if (this.root.parent) this.root.parent.remove(this.root);
    if (this.wood) {
      this.wood.geometry?.dispose();
      this.wood.customDepthMaterial = undefined;
    }
    this.barkMaterial?.dispose();
    this.barkDepthMaterial?.dispose();
    this.blossoms?.dispose();
    this.wood = null;
    this.blossoms = null;
    this.skeleton = null;
    // Textures belong to the TextureFactory cache; disposing them here would
    // pull them out from under every other system that shares them. The 1x1
    // stand-ins are the exception - we made them, so we free them.
    if (this._fallbackTex) {
      for (let i = 0; i < this._fallbackTex.length; i++) this._fallbackTex[i].dispose();
      this._fallbackTex = null;
    }
    this._barkTex = null;
    this._spawnRest = null;
    this._spawnW = null;
    this._spawnCos = null;
    this._spawnSin = null;
    this._spawnCos2 = null;
    this._spawnSin2 = null;
  }
}

// ===========================================================================
// Bark shader surgery
// ===========================================================================

const BARK_VERT_PARS = /* glsl */ `
attribute vec4 aSwayA;
attribute vec4 aSwayB;
attribute vec2 aBark;      // x = bark age (1 = this year's shoot), y = baked AO

varying vec2 vBarkUv;
varying vec3 vBarkData;    // age, baked AO, height up the tree in metres
`;

/**
 * The bend is a POSITION field, so a finite difference along an axis is exactly
 * the rotation it applies to that axis - two extra evaluations carry the whole
 * tangent frame with the wood. Skipping this is why so many wind-animated trees
 * shade as though they were still standing upright in a gale.
 *
 * The oscillator is a pure translation for one weight set, so it cancels out of
 * the difference and costs nothing here.
 */
function barkVertBody(lit) {
  return /* glsl */ `
${lit ? `vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif` : ''}

vec3 barkBase = sakuraBend( position );
${lit ? `
vec3 barkDN = sakuraBend( position + objectNormal * 0.06 ) - barkBase;
objectNormal = normalize( barkDN );
#ifdef USE_TANGENT
	vec3 barkDT = sakuraBend( position + objectTangent * 0.06 ) - barkBase;
	objectTangent = normalize( barkDT );
#endif

// Per-branch tile offset. branches.js starts every branch's V coordinate at
// zero, so without this the SAME centimetre of bark appears at the base of all
// four hundred branches - the kind of repetition the eye picks out instantly
// even when it cannot say why. The sway phases are already a stable per-branch
// hash (constant along a branch, different for each one), so the offset is
// free. V only: U repeats an integer number of tiles to close the seam, and
// shifting it would tear that.
float barkPhase = aSwayA.y * 3.1 + aSwayA.w * 5.7 + aSwayB.w * 9.3;
vBarkUv = vec2( uv.x, uv.y + barkPhase );
vBarkData = vec3( aBark.x, aBark.y, position.y );
` : ''}
vec3 barkPos = barkBase + sakuraOsc( aSwayA, aSwayB );
`;
}

const BARK_FRAG_PARS = /* glsl */ `
uniform sampler2D uBarkOrm;
uniform sampler2D uBarkMoss;
uniform sampler2D uBarkNoise;
uniform vec4 uMossParams;    // amount, low edge, high edge, moss texture scale
uniform vec3 uBarkYoung;     // multiplicative tint of first-year shoot wood
uniform vec3 uBarkWet;       // wetness, albedo darkening, gloss target
uniform vec3 uBarkSheen;
uniform float uBarkSheenAmt;
uniform vec2 uBarkFade;
uniform vec3 uBarkTint;

varying vec2 vBarkUv;
varying vec3 vBarkData;
`;

/**
 * One ORM fetch feeds roughness, occlusion AND the wetness mask - the bark
 * suite packs its cavity/wetness affinity in alpha, which three's roughnessMap
 * path throws away.
 *
 * Moss placement is not a noise threshold, it is the three things that decide
 * where moss actually grows: moisture (low on the bole and inside crotches),
 * shade (upward faces under the canopy hold water; undersides dry), and time
 * (this year's glossy shoots have not been colonised yet). The bark suite's
 * per-vertex `aBark` already carries the last two.
 */
const BARK_MAP = /* glsl */ `
// Sampled through vBarkUv rather than three's vMapUv so albedo, normal, ORM
// and moss all share the per-branch tile offset. Split them and the fissures
// in the colour map stop lining up with the relief in the normal map, which
// looks worse than the repetition it was meant to hide.
#ifdef USE_MAP
	diffuseColor *= texture2D( map, vBarkUv );
#endif

// Cherry bark is a warm, fairly DARK grey-brown with a purple-red undertone - 
// not the pale blue-grey the generated albedo alone produces, which made the
// trunk read as weathered concrete. Applied as a gain on the sampled albedo so
// the fissure and lenticel detail survives; the value is authored in linear
// space because it is a multiplier, not a colour.
//
// The lift on the low end keeps the deepest fissures from crushing to black
// once the tint has darkened everything.
diffuseColor.rgb = diffuseColor.rgb * uBarkTint + uBarkTint * 0.035;

vec4 barkOrm = texture2D( uBarkOrm, vBarkUv );

// First-year shoots are glossy mahogany-red, not grey-brown bole. One mix and
// the tree stops looking like a single material stretched over every diameter.
diffuseColor.rgb *= mix( vec3( 1.0 ), uBarkYoung, vBarkData.x );

// World-space normal, recovered by multiplying from the LEFT - that is the
// transpose of the view rotation, i.e. its inverse. One mat4 multiply instead
// of carrying another varying.
//
// This deliberately uses the interpolated GEOMETRIC normal rather than the
// normal-mapped one, for two reasons: the mapped normal does not exist yet at
// this point in the shader, and moss accumulates by BRANCH orientation, not by
// which wall of a 3 mm fissure happens to tilt upward.
#ifdef FLAT_SHADED
	vec3 barkWN = vec3( 0.0, 1.0, 0.0 );
#else
	vec3 barkWN = normalize( ( vec4( normalize( vNormal ), 0.0 ) * viewMatrix ).xyz );
#endif

float mUp     = clamp( barkWN.y * 0.5 + 0.5, 0.0, 1.0 );
float mDamp   = 1.0 - smoothstep( 1.2, 6.0, vBarkData.z );
float mOld    = 1.0 - vBarkData.x;
float mCrotch = 1.0 - vBarkData.y;
// Metre-scale patchiness, sampled in bark UV space so it does not tile with
// the moss texture itself and cannot beat against it.
float mPatch = texture2D( uBarkNoise, vBarkUv * 0.11 ).r;
float mRaw = mPatch * ( 0.22 + 0.78 * mUp ) * mDamp * mOld * ( 0.42 + 0.95 * mCrotch );
float moss = smoothstep( uMossParams.y, uMossParams.z, mRaw ) * uMossParams.x;
// Moss cannot grow in a fissure that sheds water; the ORM alpha IS that cavity
// signal, so this costs nothing extra.
moss *= 1.0 - 0.55 * barkOrm.a;

vec4 mossTex = texture2D( uBarkMoss, vBarkUv * uMossParams.w );
// The moss suite thins its own cushions out in alpha - use it, so the edges
// dissolve into bark instead of stopping on a contour line.
float mossK = clamp( moss * ( 0.35 + 0.85 * mossTex.a ), 0.0, 1.0 );
diffuseColor.rgb = mix( diffuseColor.rgb, mossTex.rgb, mossK );

// Rain soaks into the fissures first and the plates last.
float barkWet = uBarkWet.x * ( 0.30 + 0.70 * barkOrm.a ) * ( 1.0 - 0.5 * mossK );
diffuseColor.rgb *= mix( 1.0, uBarkWet.y, barkWet );
`;

const BARK_ROUGHNESS = /* glsl */ `
float roughnessFactor = roughness * barkOrm.g;
// Young wood has not weathered yet: it is smooth, waxy and catches a highlight
// along its length. That specular streak on the twigs is the single strongest
// cue that the thin wood is a different age from the bole.
roughnessFactor = mix( roughnessFactor, 0.30, vBarkData.x * 0.85 );
roughnessFactor = mix( roughnessFactor, 0.94, mossK * 0.85 );
roughnessFactor = mix( roughnessFactor, uBarkWet.z, barkWet );

// Specular anti-aliasing. Past a dozen metres a bark texel is smaller than a
// pixel, and the mip chain averages the normals but NOT the BRDF lobe they
// imply - so the lost variance has to come back as roughness or the trunk
// crawls with sparkle. Same window fades the normal map itself, below.
float barkFar = smoothstep( uBarkFade.x, uBarkFade.y, length( vViewPosition ) );
roughnessFactor = mix( roughnessFactor, min( roughnessFactor + 0.22, 1.0 ), barkFar );
roughnessFactor = clamp( roughnessFactor, 0.045, 1.0 );
`;

const BARK_NORMAL_MAPS = /* glsl */ `
#ifdef USE_NORMALMAP_TANGENTSPACE
	vec3 mapN = texture2D( normalMap, vBarkUv ).xyz * 2.0 - 1.0;
	// Young shoots are nearly smooth; the bole's fissure relief on a 5 mm twig
	// is both wrong and a source of specular sparkle at that texel density.
	mapN.xy *= normalScale * ( 1.0 - 0.65 * barkFar ) * ( 1.0 - 0.70 * vBarkData.x );
	normal = normalize( tbn * mapN );
#endif
`;

const BARK_AO = /* glsl */ `
// Two occlusion terms that measure different things: the texture's is the
// cavity inside a fissure, the vertex one is how much wood is piled up around
// this point - which is what actually darkens the inside of a fork.
float barkAO = barkOrm.r * mix( 1.0, vBarkData.y, 0.85 );
reflectedLight.indirectDiffuse *= barkAO;
`;

/**
 * Grazing sky reflection. There is no environment map in this scene, so the
 * one thing physically-based shading is missing on bark is the sheet of sky
 * that slides along a trunk at a glancing angle. Ten instructions buy it back,
 * and multiplying by wetness is what makes rain read on the wood at all.
 */
const BARK_SHEEN = /* glsl */ `
{
  vec3 sV = normalize( vViewPosition );
  float sF = pow( 1.0 - clamp( dot( normal, sV ), 0.0, 1.0 ), 4.5 );
  float sG = uBarkSheenAmt * ( 1.0 - 0.65 * roughnessFactor ) * ( 1.0 - 0.7 * mossK );
  outgoingLight += uBarkSheen * sF * sG * barkAO;
}
`;

function patchBarkVertex(src, depthOnly = false) {
  if (typeof src !== 'string' || src.indexOf('aSwayA') !== -1) return src;
  const pars = depthOnly
    ? 'attribute vec4 aSwayA;\nattribute vec4 aSwayB;\n'
    : BARK_VERT_PARS;
  let out = src.replace('#include <common>', `#include <common>\n${SAKURA_WIND_GLSL}\n${pars}`);
  if (depthOnly) {
    out = out.replace(
      '#include <begin_vertex>',
      `${barkVertBody(false)}\n\tvec3 transformed = barkPos;`
    );
  } else {
    out = out.replace('#include <beginnormal_vertex>', barkVertBody(true));
    out = out.replace('#include <begin_vertex>', 'vec3 transformed = barkPos;');
  }
  return out;
}

function patchBarkFragment(src) {
  if (typeof src !== 'string' || src.indexOf('uBarkOrm') !== -1) return src;
  let out = src.replace('#include <common>', `#include <common>\n${BARK_FRAG_PARS}`);
  out = out.replace('#include <map_fragment>', BARK_MAP);
  out = out.replace('#include <roughnessmap_fragment>', BARK_ROUGHNESS);
  out = out.replace('#include <normal_fragment_maps>', BARK_NORMAL_MAPS);
  out = out.replace('#include <aomap_fragment>', BARK_AO);
  // Before opaque_fragment, which is where outgoingLight is consumed, and
  // deliberately NOT inside lights_fragment_begin - the engine's cascaded
  // shadow patch owns that chunk.
  out = out.replace('#include <opaque_fragment>', `${BARK_SHEEN}\n\t#include <opaque_fragment>`);
  return out;
}
