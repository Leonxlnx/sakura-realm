/**
 * world/scatter.js - everything growing BETWEEN the grass blades, plus horizon depth.
 *
 * `world/grass.js` renders one species: a dense sward of blades. A real meadow is
 * never one species, and a field of a single repeated silhouette is the loudest
 * remaining "this is procedural" tell in the scene. This module supplies the rest of
 * the plant community:
 *
 *   cover   low creeping ground cover - clover, plantain, vetch. Fills the gaps at
 *           blade level so the soil never reads as a bare floor.
 *   bent    fine feathery bent-grass tufts. Soft, hair-thin, breaks the hard edges
 *           of the sward.
 *   broad   broad-leafed weeds and dock. The only wide silhouette out there, which
 *           is exactly why it does so much work.
 *   stalk   seed-head stalks and grass flowers standing ABOVE the sward line. These
 *           catch backlight and are what makes a field read as *tall*.
 *   bloom   sparse wildflowers in muted colours - chalk white, pale yellow, faded
 *           lilac. Never a primary; the field is quiet, not a paint chart.
 *   reed    tall plumed reed stands, restricted to damp hollows.
 *
 * Each species has its OWN density field, so they form drifts, colonies and stands
 * instead of an even sprinkle. Seed-heads thin out where the ground is damp, dock
 * only stands where it is; the reeds and the clover almost never meet. Everything
 * else in here is unchanged in intent: weathered rocks, fallen wood, a drift of
 * fallen sakura petals, and a two-ring procedural treeline on the horizon.
 *
 * Placement discipline (all of it load-bearing):
 *  - Pure function of world position via `cellSeed`/`makeRNG` - the world is
 *    identical every run and stable across streaming. No `Math.random()`.
 *  - Streamed in 32 m chunks around the camera, time-budgeted, allocation-free per
 *    frame. Chunk *generation* allocates; the frame loop never does.
 *  - Every instance carries its own fade window and dissolves by scaling to zero
 *    about its own pivot, so props sink into the ground instead of popping.
 *  - Real bounding volumes: exact geometry spheres, plus a per-frame world sphere
 *    unioned from the packed chunk extents.
 *
 * Performance notes for the 780M:
 *  - One InstancedMesh per visual family; chunks own contiguous slices of the
 *    instance buffers, re-packed per GROUP only when that group's chunks change.
 *  - Three materials for twelve meshes: stone, plant, petal. Twelve draw calls.
 *  - Plant silhouettes live in the atlas alpha, not in geometry - a card is 4-8
 *    vertices whatever it is a picture of. Nineteen species tiles share one 1024²
 *    atlas, so a new plant costs a tile, not a mesh and not a draw call.
 *  - Chunk generation is resumable at row granularity, so a dense ground-cover
 *    chunk cannot blow the streaming budget the way an atomic build did.
 *  - A group's baseRadius is the reference fade reach of its largest instance,
 *    or a deliberate cap below it (rockL, branch, stalk, reed are capped - their
 *    biggest instances fade over a shorter window rather than being streamed a
 *    further 20-40 m to fade somewhere nobody is looking).
 *
 * Measured on the reference machine, HIGH, camera at ground level (before the
 * fill-rate trim to `cover` and `bent`, which removes roughly a fifth of the
 * plant instances and about a quarter of the near-field card area):
 *   12 draw calls · 6.5k instances · 148k vertices · update() 0.005 ms CPU ·
 *   worst streaming frame 2.1 ms while walking · atlas bake ~180 ms at load.
 * Plus 2 draw calls for the treeline rings and 1 for the haze band.
 */

import * as THREE from 'three';
import {
  makeRNG, createNoise, clamp, clamp01, damp, lerp, smoothstep, TAU,
} from '../core/math.js';

// ---------------------------------------------------------------------------
// Module-scope scratch. Nothing below allocates inside update().
// ---------------------------------------------------------------------------

const UP = new THREE.Vector3(0, 1, 0);
const _pos = new THREE.Vector3();
const _axis = new THREE.Vector3();
const _scl = new THREE.Vector3();
const _qa = new THREE.Quaternion();
const _qb = new THREE.Quaternion();
const _qc = new THREE.Quaternion();
const _nrm = new THREE.Vector3();
const _mat = new THREE.Matrix4();
const _c0 = new THREE.Color();
const _c1 = new THREE.Color();
const _c2 = new THREE.Color();

/** Chunk edge length in metres. Every prop cell size divides this exactly. */
const CHUNK = 32;
/** Upper bound on instances any single group may emit for one chunk. */
const SCRATCH = 448;
/** Seed for every spatial decision this module makes. */
const WORLD_SEED = 0x5a4b12;

/**
 * Eight offsets on a 1 km ring, used to find the height of the *far* ground so the
 * treeline can sit on it. The terrain's largest octave is ±34 m over a ~1.5 km
 * wavelength, so a horizon pinned to y = 0 would visibly float when the player is in
 * a basin and sink when they are on a ridge.
 */
const HORIZON_RING = (() => {
  const a = new Float64Array(16);
  for (let i = 0; i < 8; i++) {
    a[i * 2] = Math.cos((i / 8) * TAU) * 1000;
    a[i * 2 + 1] = Math.sin((i / 8) * TAU) * 1000;
  }
  return a;
})();

// ---------------------------------------------------------------------------
// Flora atlas layout
// ---------------------------------------------------------------------------
//
// One 1024² atlas on a 16 x 16 grid of 64 px cells. A species claims a RECTANGLE
// of cells sized to the aspect of the card it will be drawn on - a 1.7 m seed-head
// stalk gets 128 x 512, a clover clump gets 128 x 128 - so no tile is ever
// squashed onto its quad, and no texel is spent on empty margin.
//
// Variants of one species sit at a constant stride from the base tile, which is
// what lets a single per-instance uv offset pick between them: two crossed cards
// baked to variants 0 and 1 become variants 2 and 3 for the cost of one float.

const ATLAS_SIZE = 1024;
/** Atlas grid cell, pixels. */
const ACELL = 64;
/** Texels trimmed off each tile edge when computing UVs. */
const AINSET = 5;
/** Transparent gutter kept around drawn content, pixels. Must exceed AINSET. */
const ADRAW = 7;

/** [cellX, cellY, cellW, cellH] of each species' FIRST variant, canvas space (y down). */
const TILE = {
  stalk: [0, 0, 2, 8],    // 128 x 512 - four variants striding +2 cells in u
  reed: [8, 0, 3, 8],     // 192 x 512 - two variants striding +3 cells in u
  cover: [14, 0, 2, 2],   // 128 x 128 - four variants striding +2 cells in v (down)
  bent: [0, 8, 4, 4],     // 256 x 256 - four variants striding +4 cells in u
  bloom: [0, 12, 4, 4],   // 256 x 256 - three variants striding +4 cells in u
  broad: [12, 12, 2, 4],  // 128 x 256 - two variants striding +2 cells in u
};

/** Per-instance uv shift of one variant step, in uv units. */
const STEP_U = (n) => (n * ACELL) / ATLAS_SIZE;

/**
 * UV rect of a tile. The canvas is authored top-down and uploaded bottom-up, so
 * uv.y = 0 - which is the *base* of every plant card - must land on the BOTTOM edge
 * of the tile in canvas space. Getting this inverted silently plants every species
 * upside down, and upside-down foliage still looks vaguely plausible in a still.
 */
function tileUV(cx, cy, cw, ch) {
  const s = 1 / ATLAS_SIZE;
  return [
    (cx * ACELL + AINSET) * s,
    1 - ((cy + ch) * ACELL - AINSET) * s,
    (cw * ACELL - AINSET * 2) * s,
    (ch * ACELL - AINSET * 2) * s,
  ];
}

/** Canvas-space drawing rect of a tile, inset by the mip gutter. */
function tileRect(cx, cy, cw, ch) {
  return {
    x: cx * ACELL + ADRAW,
    y: cy * ACELL + ADRAW,
    w: cw * ACELL - ADRAW * 2,
    h: ch * ACELL - ADRAW * 2,
  };
}

/** UV rect of variant `k`, offset from the species base by (du, dv) per step. */
function variantUV(spec, k, du, dv) {
  const base = tileUV(TILE[spec][0], TILE[spec][1], TILE[spec][2], TILE[spec][3]);
  return [base[0] + k * du, base[1] + k * dv, base[2], base[3]];
}

const UV_STALK = (k) => variantUV('stalk', k, STEP_U(2), 0);
const UV_REED = (k) => variantUV('reed', k, STEP_U(3), 0);
const UV_BENT = (k) => variantUV('bent', k, STEP_U(4), 0);
const UV_BLOOM = (k) => variantUV('bloom', k, STEP_U(4), 0);
const UV_BROAD = (k) => variantUV('broad', k, STEP_U(2), 0);
// Cover variants march DOWN the canvas, which is downward in flipped v as well.
const UV_COVER = (k) => variantUV('cover', k, 0, -STEP_U(2));

// ---------------------------------------------------------------------------
// Palette - quiet, low-saturation, slightly melancholy. Nothing here is allowed
// to be a primary; the pinks live in the tree, not in the meadow.
// ---------------------------------------------------------------------------

const PAL = {
  stemDark: '#5c6a43',
  stem: '#6b7a4e',
  stemPale: '#7e8a5c',
  leafDeep: '#59683f',
  leaf: '#6b7a4d',
  leafPale: '#828d5f',
  leafGrey: '#6d7659',
  dry: '#9a9370',
  dryPale: '#b3aa88',
  dryDeep: '#82794f',
  husk: '#c2b795',
  huskPale: '#d6cdb4',
  seed: '#a89b78',
  rust: '#8a7550',
  white: '#e9e6da',
  whiteHi: '#f3f0e7',
  yellow: '#dcd19b',
  yellowDeep: '#c3b479',
  lilac: '#a99cb6',
  lilacHi: '#bcb2c8',
  lilacDeep: '#8d8399',
  pollen: '#c6b276',
};

/** Murmur3 finalizer over three integers - a pure, allocation-free cell hash. */
function cellSeed(ix, iz, salt) {
  let h = Math.imul(ix | 0, 0x27d4eb2d) ^ Math.imul(iz | 0, 0x165667b1) ^ Math.imul(salt | 0, 0x9e3779b1);
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return (h ^ (h >>> 16)) >>> 0;
}
const cellHash01 = (ix, iz, salt) => cellSeed(ix, iz, salt) / 4294967296;

/**
 * NaN/Infinity firewall. `clamp`/`clamp01` are comparison-based and pass NaN
 * straight through, so a single bad frame from a sibling would otherwise reach
 * either an instance buffer (where it is cached and regenerates bit-identically
 * for ever) or a piece of lagged state like `_lag` / `_horizonY` (where the
 * damping term keeps it NaN for the rest of the session).
 */
const finite = (v, fallback) => (Number.isFinite(v) ? v : fallback);

/**
 * Clamps a Color in place to a finite, non-negative range. Cheaper than guarding
 * every sibling colour individually and it catches the ones that arrive through
 * a lerp/multiply chain, where a single NaN component would otherwise reach a
 * uniform and turn the whole treeline into a NaN-coloured hole.
 */
function scrubColor(c, hi = 8) {
  c.r = clamp(finite(c.r, 0), 0, hi);
  c.g = clamp(finite(c.g, 0), 0, hi);
  c.b = clamp(finite(c.b, 0), 0, hi);
  return c;
}

/**
 * One reusable mulberry32, seeded per placement cell.
 *
 * `makeRNG()` builds a closure plus five method closures on every call, and
 * placement called it once per accepted instance - six objects per plant, a few
 * thousand per chunk. That was the single largest allocation source in the
 * streaming path and the origin of the GC outliers during a walk. Reseeding
 * resets the generator's entire state, so the emitted sequence is bit-identical
 * to `makeRNG(seed)` and the world does not move.
 */
function makeSeedableRNG() {
  let a = 0;
  const rand = () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  rand.seed = (s) => { a = s >>> 0; };
  rand.range = (min, max) => min + rand() * (max - min);
  rand.int = (min, max) => Math.floor(min + rand() * (max - min + 1));
  rand.sign = () => (rand() < 0.5 ? -1 : 1);
  rand.gaussian = (mean = 0, stdev = 1) => {
    const u = 1 - rand();
    const v = rand();
    return mean + stdev * Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v);
  };
  return rand;
}
/** Placement RNG. Only ever live inside one synchronous `_emit`. */
const PLACE_RNG = makeSeedableRNG();

/**
 * Welds a non-indexed geometry (PolyhedronGeometry is non-indexed) down to unique
 * positions. 80 flat-shaded triangles then cost 42 vertices instead of 240 - three
 * reconstructs the facets from screen-space derivatives when `flatShading` is on.
 */
function weldPositions(geo, tol = 1e-4) {
  const src = geo.attributes.position.array;
  const inv = 1 / tol;
  const map = new Map();
  const out = [];
  const index = new Array(src.length / 3);
  for (let i = 0, v = 0; i < src.length; i += 3, v++) {
    const key =
      `${Math.round(src[i] * inv)},${Math.round(src[i + 1] * inv)},${Math.round(src[i + 2] * inv)}`;
    let id = map.get(key);
    if (id === undefined) {
      id = out.length / 3;
      map.set(key, id);
      out.push(src[i], src[i + 1], src[i + 2]);
    }
    index[v] = id;
  }
  const welded = new THREE.BufferGeometry();
  welded.setAttribute('position', new THREE.Float32BufferAttribute(out, 3));
  welded.setIndex(index);
  return welded;
}

function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.clearRect(0, 0, w, h);
  return g;
}

/**
 * Bleeds colour outward into fully transparent texels. Without this, mip generation
 * averages transparent black into every silhouette edge and alpha-tested foliage
 * grows a dark halo at distance - the most common "this is a game asset" tell.
 * Works on the raw ImageData because a canvas backing store is premultiplied and
 * would throw the dilated colour away on the next read.
 *
 * A 1024² atlas is mostly empty air, so the 3x3 gather is gated on a coarse
 * occupancy mask: only blocks that hold, or touch, opaque texels are visited. That
 * roughly halves a bake that would otherwise be the slowest thing in init().
 */
function dilateAlpha(ctx2d, w, h, passes) {
  const img = ctx2d.getImageData(0, 0, w, h);
  const dst = img.data;
  const px = w * h;

  // `known` marks texels that carry usable colour: originally opaque, OR filled by
  // an earlier pass. Dilation only ever writes RGB - alpha must stay untouched or
  // the silhouette grows - so without this mask every pass reads the same original
  // opaque set, rewrites the same one-texel ring, and the bleed never widens no
  // matter how many passes are asked for. One texel protects mip 0 and nothing
  // beyond it, which is precisely where alpha-tested foliage starts to halo.
  const known = new Uint8Array(px);
  for (let p = 0, i = 3; p < px; p++, i += 4) known[p] = dst[i] > 6 ? 1 : 0;

  // Coarse occupancy so the 3x3 gather only visits blocks that hold, or touch,
  // known texels. A foliage atlas is mostly empty air; this roughly halves a bake
  // that would otherwise be the slowest thing in init().
  const BS = 16;
  const bw = Math.ceil(w / BS);
  const bh = Math.ceil(h / BS);
  const solid = new Uint8Array(bw * bh);
  const visit = new Uint8Array(bw * bh);
  const srcKnown = new Uint8Array(px);

  for (let pass = 0; pass < passes; pass++) {
    const src = dst.slice();
    srcKnown.set(known);

    solid.fill(0);
    for (let by = 0; by < bh; by++) {
      const y1 = Math.min(h, (by + 1) * BS);
      for (let bx = 0; bx < bw; bx++) {
        const x1 = Math.min(w, (bx + 1) * BS);
        let hit = 0;
        scan:
        for (let y = by * BS; y < y1; y++) {
          const row = y * w;
          for (let x = bx * BS; x < x1; x++) {
            if (srcKnown[row + x]) { hit = 1; break scan; }
          }
        }
        solid[by * bw + bx] = hit;
      }
    }
    visit.fill(0);
    let any = 0;
    for (let by = 0; by < bh; by++) {
      for (let bx = 0; bx < bw; bx++) {
        if (!solid[by * bw + bx]) continue;
        for (let j = -1; j <= 1; j++) {
          const ny = by + j;
          if (ny < 0 || ny >= bh) continue;
          for (let i = -1; i <= 1; i++) {
            const nx = bx + i;
            if (nx < 0 || nx >= bw) continue;
            visit[ny * bw + nx] = 1;
            any = 1;
          }
        }
      }
    }
    if (!any) break;   // nothing opaque at all: further passes cannot do anything

    for (let by = 0; by < bh; by++) {
      const yEnd = Math.min(h, (by + 1) * BS);
      for (let bx = 0; bx < bw; bx++) {
        if (!visit[by * bw + bx]) continue;
        const xEnd = Math.min(w, (bx + 1) * BS);
        for (let y = by * BS; y < yEnd; y++) {
          const row = y * w;
          for (let x = bx * BS; x < xEnd; x++) {
            const p = row + x;
            if (srcKnown[p]) continue;
            let r = 0, g = 0, b = 0, n = 0;
            const y0 = y > 0 ? y - 1 : y;
            const y1 = y < h - 1 ? y + 1 : y;
            const x0 = x > 0 ? x - 1 : x;
            const x1 = x < w - 1 ? x + 1 : x;
            for (let ny = y0; ny <= y1; ny++) {
              const nrow = ny * w;
              for (let nx = x0; nx <= x1; nx++) {
                const q = nrow + nx;
                if (srcKnown[q]) { const j = q * 4; r += src[j]; g += src[j + 1]; b += src[j + 2]; n++; }
              }
            }
            if (n) {
              const i = p * 4;
              dst[i] = r / n; dst[i + 1] = g / n; dst[i + 2] = b / n;
              known[p] = 1;
            }
          }
        }
      }
    }
  }
  return img;
}

/** ImageData -> DataTexture, flipping rows so v = 0 is the bottom of the drawing. */
function textureFromImageData(img, aniso) {
  const { width: w, height: h, data } = img;
  const flipped = new Uint8Array(w * h * 4);
  const stride = w * 4;
  for (let y = 0; y < h; y++) {
    flipped.set(data.subarray(y * stride, y * stride + stride), (h - 1 - y) * stride);
  }
  const tex = new THREE.DataTexture(flipped, w, h, THREE.RGBAFormat);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = aniso;
  tex.needsUpdate = true;
  return tex;
}

// ---------------------------------------------------------------------------
// Canvas primitive: the tapered ribbon
// ---------------------------------------------------------------------------
//
// Every organic shape in the flora atlas - blade, stem, leaf, petal, awn - is this
// one primitive: a quadratic Bezier spine swept by a width profile. A round-capped
// `stroke()` cannot taper, and a grass blade that ends in a 3 px dot instead of a
// hairline is the single most obvious tell in hand-drawn 2D foliage.

const RIB_MAX = 33;
const RIB_L = new Float64Array(RIB_MAX * 2);
const RIB_R = new Float64Array(RIB_MAX * 2);

/** Blade profile: full width at the base, a point at the tip. */
const wBlade = (w, k = 0.75) => (t) => w * Math.pow(1 - t, k);
/** Leaf profile: pinched at both ends, widest a little before halfway. */
const wLeaf = (w, bias = 0.68, sharp = 0.85) =>
  (t) => w * Math.pow(Math.sin(Math.PI * Math.pow(t, bias)), sharp);
/** Constant width, for stems that carry something heavy at the top. */
const wStem = (a, b) => (t) => lerp(a, b, t);

function ribbonPath(g, x0, y0, cx, cy, x1, y1, wFn, steps) {
  const n = Math.min(RIB_MAX - 1, steps);
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const s = 1 - t;
    const x = s * s * x0 + 2 * s * t * cx + t * t * x1;
    const y = s * s * y0 + 2 * s * t * cy + t * t * y1;
    let dx = 2 * s * (cx - x0) + 2 * t * (x1 - cx);
    let dy = 2 * s * (cy - y0) + 2 * t * (y1 - cy);
    const d = Math.hypot(dx, dy);
    if (d > 1e-6) { dx /= d; dy /= d; } else { dx = 0; dy = -1; }
    const hw = wFn(t) * 0.5;
    RIB_L[i * 2] = x - dy * hw; RIB_L[i * 2 + 1] = y + dx * hw;
    RIB_R[i * 2] = x + dy * hw; RIB_R[i * 2 + 1] = y - dx * hw;
  }
  g.beginPath();
  g.moveTo(RIB_L[0], RIB_L[1]);
  for (let i = 1; i <= n; i++) g.lineTo(RIB_L[i * 2], RIB_L[i * 2 + 1]);
  for (let i = n; i >= 0; i--) g.lineTo(RIB_R[i * 2], RIB_R[i * 2 + 1]);
  g.closePath();
}

function ribbon(g, x0, y0, cx, cy, x1, y1, wFn, fill, steps = 12) {
  ribbonPath(g, x0, y0, cx, cy, x1, y1, wFn, steps);
  g.fillStyle = fill;
  g.fill();
}

// ===========================================================================

export class Scatter {
  constructor(ctx) {
    this.ctx = ctx;
    this.scene = ctx.scene;
    this.camera = ctx.camera;
    this.state = ctx.state;

    this.terrain = null;
    this.tree = null;
    this.wind = null;

    this.root = new THREE.Group();
    this.root.name = 'scatter';
    this.root.matrixAutoUpdate = false; // identity forever; the shaders rely on it

    this.horizon = new THREE.Group();
    this.horizon.name = 'scatter.horizon';

    /** Density fields sampled at placement time. */
    this.noise = createNoise(WORLD_SEED);

    this.chunks = new Map();
    /** Active chunks in ascending distance order - packing consumes nearest first. */
    this.sorted = [];
    this.queue = [];
    this._taskPool = [];
    this.groups = [];

    this._camCX = 1e9;
    this._camCZ = 1e9;
    this._firstFill = true;
    /** Round-robin start for the per-frame packing budget, so no group starves. */
    this._packCursor = 0;
    this._densityScale = 1;
    this._plantScale = 1;

    // Chunk-generation scratch, reused for every chunk and every group.
    this._sMat = new Float32Array(SCRATCH * 16);
    this._sCol = new Float32Array(SCRATCH * 3);
    this._sExt = [new Float32Array(SCRATCH * 4), new Float32Array(SCRATCH * 4)];
    /** Per-chunk instance extent, accumulated during generation for the bounds. */
    this._bLo = 0;
    this._bHi = 0;
    this._bRad = 0;
    /**
     * Resume cursor for a chunk that ran out of frame budget half way through.
     * One preallocated object: only one chunk is ever in flight, because the drain
     * loop is strictly sequential and the generation scratch is shared.
     */
    this._rs = { group: null, chunk: null, j: 0, n: 0, lo: 0, hi: 0, rad: 0 };

    this._treePos = new THREE.Vector3(0, 0, 0);
    this._canopyR = 9;
    this._driftR = 9 * 3.1 + 7;

    // Horizon ring lag: the near ring trails the player, so the two rings show real
    // parallax against each other, clamped so it can never drift off centre.
    this._lag = new THREE.Vector2(0, 0);
    this._prevCam = new THREE.Vector2(0, 0);
    this._hasPrevCam = false;
    this._horizonY = 0;
    this._horizonYInit = false;

    this._ownedTextures = [];
    this._disposables = [];
  }

  // -------------------------------------------------------------------------
  // Build
  // -------------------------------------------------------------------------

  async init() {
    const caps = this.ctx.renderer?.capabilities;
    const aniso = Math.min(4, caps?.getMaxAnisotropy?.() ?? 1);

    const floraAtlas = this._texture('scatter.meadowAtlas', () => this._makeFloraAtlas(aniso));
    const petalTex = this._texture('scatter.groundPetal', () => this._makePetalTexture(aniso));
    const treeAtlas = this._texture('scatter.treelineAtlas', () => this._makeTreelineAtlas(aniso));

    this._buildMaterials(floraAtlas, petalTex);
    this._buildGroups();
    this._buildHorizon(treeAtlas);

    this.scene.add(this.root);
    this.scene.add(this.horizon);

    this._applyQuality(this.state.quality);
  }

  link(systems) {
    this.terrain = systems.terrain || null;
    this.tree = systems.tree || null;
    this.wind = systems.wind || null;
    const t = this.tree;
    if (t?.position?.isVector3) this._treePos.copy(t.position);
    if (typeof t?.canopyRadius === 'number' && t.canopyRadius > 0.5) this._canopyR = t.canopyRadius;
    this._driftR = this._canopyR * 3.1 + 7;
  }

  _texture(name, gen) {
    const factory = this.ctx.textures;
    if (factory && typeof factory.get === 'function') {
      const t = factory.get(name, gen);
      if (t) return t; // the factory owns disposal
    }
    const t = gen();
    this._ownedTextures.push(t);
    return t;
  }

  // -------------------------------------------------------------------------
  // Materials
  // -------------------------------------------------------------------------

  _buildMaterials(floraAtlas, petalTex) {
    // Shared uniform objects - one write per frame drives every patched material.
    this.u = {
      time: { value: 0 },
      windDir: { value: new THREE.Vector2(1, 0) },
      windStr: { value: 0.5 },
      gust: { value: 1 },
      turb: { value: 0.4 },
      snow: { value: 0 },
      snowColor: { value: new THREE.Color(0.85, 0.88, 0.92) },
      sunDir: { value: new THREE.Vector3(0, 1, 0) },
      sunCol: { value: new THREE.Color(0, 0, 0) },
      // The VIEW camera position. The visible materials can read three's own
      // `cameraPosition`, but a shadow pass binds the *light's* camera to that
      // uniform, so the depth material below needs its own copy or every prop
      // would fade by its distance from the sun.
      camPos: { value: new THREE.Vector3() },
    };

    // --- Rocks and fallen wood share one material: pure vertex colour, no texture
    //     fetch, facets from screen-space derivatives. The cheapest thing that still
    //     reads as weathered stone at 0.1-2.7 m.
    const ground = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.93,
      metalness: 0.0,
      flatShading: true,
      dithering: true,
    });
    ground.customProgramCacheKey = () => 'scatter-ground';
    ground.onBeforeCompile = (sh) => {
      sh.uniforms.uSnow = this.u.snow;
      sh.uniforms.uSnowColor = this.u.snowColor;
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', `#include <common>
attribute vec2 aFade;
varying float vUpY;`)
        .replace('#include <begin_vertex>', `#include <begin_vertex>
{
  vec3 sOrigin = ( modelMatrix * instanceMatrix * vec4( 0.0, 0.0, 0.0, 1.0 ) ).xyz;
  transformed *= 1.0 - smoothstep( aFade.x, aFade.y, distance( cameraPosition, sOrigin ) );
  vec3 nw = ( modelMatrix * instanceMatrix * vec4( normal, 0.0 ) ).xyz;
  vUpY = nw.y * inversesqrt( max( dot( nw, nw ), 1e-6 ) );
}`);
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', `#include <common>
uniform float uSnow;
uniform vec3 uSnowColor;
varying float vUpY;`)
        .replace('#include <color_fragment>', `#include <color_fragment>
float sSnow = uSnow * smoothstep( 0.30, 0.74, vUpY );
diffuseColor.rgb = mix( diffuseColor.rgb, uSnowColor, sSnow );`)
        .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
roughnessFactor = mix( roughnessFactor, 0.74, uSnow * smoothstep( 0.30, 0.74, vUpY ) );`);
    };
    this.matGround = ground;
    this._disposables.push(ground);

    // --- Shadow casters have to fade on the SAME curve as the visible instance.
    //     three's shared MeshDepthMaterial knows nothing about aFade, so a rock
    //     that has already scaled to zero at 97 m went on casting a full-size
    //     shadow onto open ground for the rest of the 190 m shadow distance - 
    //     a hard dark blob with nothing above it, at exactly the range where the
    //     field is most open. Only rockM / rockL / branch cast, and they all use
    //     matGround, so one depth material covers every caster in the module.
    const groundDepth = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking });
    groundDepth.customProgramCacheKey = () => 'scatter-ground-depth';
    groundDepth.onBeforeCompile = (sh) => {
      sh.uniforms.uCamPos = this.u.camPos;
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', `#include <common>
attribute vec2 aFade;
uniform vec3 uCamPos;`)
        .replace('#include <begin_vertex>', `#include <begin_vertex>
#ifdef USE_INSTANCING
{
  vec3 sOrigin = ( modelMatrix * instanceMatrix * vec4( 0.0, 0.0, 0.0, 1.0 ) ).xyz;
  transformed *= 1.0 - smoothstep( aFade.x, aFade.y, distance( uCamPos, sOrigin ) );
}
#endif`);
    };
    this.matGroundDepth = groundDepth;
    this._disposables.push(groundDepth);

    // --- Plants: Lambert rather than Standard. These are thousands of small
    //     alpha-tested cards; the PBR specular lobe would never be resolvable and
    //     the BRDF is pure fill-rate on an iGPU. Lambert still gives us shadow
    //     reception, scene fog and tonemapping for free, which a hand-rolled
    //     ShaderMaterial would have to reimplement (and would have to own the four
    //     fog uniforms or throw inside refreshFogUniforms).
    const flora = new THREE.MeshLambertMaterial({
      map: floraAtlas,
      alphaTest: 0.42,
      side: THREE.DoubleSide,
      vertexColors: true,
      transparent: false,
      dithering: true,
    });
    flora.customProgramCacheKey = () => 'scatter-plant';
    flora.onBeforeCompile = (sh) => {
      sh.uniforms.uTime = this.u.time;
      sh.uniforms.uWindDir = this.u.windDir;
      sh.uniforms.uWindStr = this.u.windStr;
      sh.uniforms.uGust = this.u.gust;
      sh.uniforms.uTurb = this.u.turb;
      sh.uniforms.uSunDir = this.u.sunDir;
      sh.uniforms.uSunCol = this.u.sunCol;
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', `#include <common>
uniform float uTime;
uniform vec2 uWindDir;
uniform float uWindStr;
uniform float uGust;
uniform float uTurb;
attribute vec4 aAtlas;  // uv rect of this card in the flora atlas: origin.xy, span.zw
attribute vec2 aLeaf;   // x translucency, y tip-flutter weight
attribute vec4 aInst;   // phase, wind compliance, atlas u shift, atlas v shift
attribute vec2 aFade;
varying float vTrans;
vec3 sakuraWind = vec3( 0.0 );
vec3 sOrigin = vec3( 0.0 );
float sFade = 1.0;`)
        .replace('#include <uv_vertex>', `#include <uv_vertex>
#ifdef USE_MAP
  vMapUv = aAtlas.xy + aInst.zw + uv * aAtlas.zw;
#endif
vTrans = aLeaf.x;`)
        .replace('#include <begin_vertex>', `#include <begin_vertex>
sOrigin = ( modelMatrix * instanceMatrix * vec4( 0.0, 0.0, 0.0, 1.0 ) ).xyz;
sFade = 1.0 - smoothstep( aFade.x, aFade.y, distance( cameraPosition, sOrigin ) );
transformed *= sFade;`)
        .replace('#include <project_vertex>', `
vec4 sPos = vec4( transformed, 1.0 );
#ifdef USE_INSTANCING
  sPos = instanceMatrix * sPos;
#endif
sPos = modelMatrix * sPos;
{
  // uGust is sampled from wind.getGustAt() at the listener, exactly as the grass
  // does, so a front that flattens the sward flattens these at the same instant.
  // The travelling term on top of it is local detail only: neighbours must not
  // move in lockstep at arm's length, or a metre of meadow reads as one object.
  float travel = dot( sOrigin.xz, uWindDir ) * 0.105;
  float w1 = sin( travel - uTime * 1.42 + aInst.x );
  float w2 = sin( travel * 2.63 - uTime * 3.15 + aInst.x * 1.7 );
  float w3 = sin( uTime * 9.10 + aInst.x * 5.3 );
  float w4 = sin( uTime * 6.35 + aInst.x * 3.1 + 1.9 );
  // Column 1 of the instance matrix is the local Y basis, so its length is the
  // plant's height in metres. Sway has to be proportional to it: an absolute
  // displacement would whip a 12 cm clover leaf as far as a 1.8 m reed.
  float stemLen = length( instanceMatrix[ 1 ].xyz );
  float drive = uWindStr * uGust * aInst.y * sFade * stemLen;
  float bend = ( 0.34 + 0.38 * w1 + 0.17 * w2 ) * drive * 0.24;
  float flut = ( 0.085 * w3 + 0.060 * w4 ) * drive * uTurb * aLeaf.y;
  float lateral = ( bend + flut ) * uv.y * uv.y;
  // Cross-wind flutter turns pure in-line swaying into a shallow figure of eight,
  // which is what stops a stand of seed-heads looking like a row of windscreen
  // wipers when a gust is square on. Named sway rather than cross: a local
  // variable that shadows a GLSL built-in is legal but is exactly the sort of
  // thing an ANGLE/D3D translation layer has been known to reject.
  float sway = ( 0.055 * w4 - 0.042 * w3 + 0.06 * w2 ) * drive * uTurb * aLeaf.y
             * uv.y * uv.y;
  // Cap the lean at ~33 deg: past that the quadratic drop below stops approximating
  // arc length and a storm gust would visibly stretch the stem.
  lateral = clamp( lateral, -0.55 * stemLen, 0.55 * stemLen );
  sway = clamp( sway, -0.22 * stemLen, 0.22 * stemLen );
  sakuraWind = vec3( uWindDir.x, 0.0, uWindDir.y ) * lateral
             + vec3( -uWindDir.y, 0.0, uWindDir.x ) * sway;
  // Arc length is conserved: a tip pushed sideways by d drops by about d^2 / 2L.
  sakuraWind.y = -( lateral * lateral + sway * sway ) / max( 2.0 * stemLen, 0.05 );
  sPos.xyz += sakuraWind;
}
vec4 mvPosition = viewMatrix * sPos;
gl_Position = projectionMatrix * mvPosition;`)
        .replace('#include <worldpos_vertex>', `#include <worldpos_vertex>
#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
  worldPosition.xyz += sakuraWind;   // keep shadow lookups on the bent geometry
#endif`);
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', `#include <common>
uniform vec3 uSunDir;
uniform vec3 uSunCol;
varying float vTrans;`)
        // Mip-loss compensation on the alpha cutoff. A seed-head culm is ~4 of
        // its tile's 128 texels wide; by mip 3 the box filter has averaged it
        // down to roughly a quarter alpha, and a fixed 0.42 cutoff then erases
        // the stalk while the denser head survives - floating seed-heads at
        // exactly the 20-60 m the composition leans on hardest. Relaxing the
        // cutoff with the texture LOD lets the silhouette fatten slightly
        // instead of dissolving, which is also what holds the apparent density
        // of the field constant with distance. Capped at LOD 4: past that a tap
        // starts reaching across the atlas gutter into the neighbouring tile.
        .replace('#include <alphatest_fragment>', `
#if defined( USE_ALPHATEST ) && defined( USE_MAP )
{
  vec2 duv = max( abs( dFdx( vMapUv ) ), abs( dFdy( vMapUv ) ) ) * ${ATLAS_SIZE}.0;
  float lod = clamp( 0.5 * log2( max( dot( duv, duv ), 1e-8 ) ), 0.0, 4.0 );
  if ( diffuseColor.a < alphaTest * exp2( -0.30 * lod ) ) discard;
}
#else
  #include <alphatest_fragment>
#endif`)
        // Foliage cards shaded off their true plane normal flicker and go black on
        // the far side. Keep about half the lateral variation - that is what makes
        // one side of a tuft darker than the other - and bias the rest toward the
        // sky so nothing can ever face away from all the light in the scene.
        .replace('#include <normal_fragment_begin>', `#include <normal_fragment_begin>
normal = normalize( vec3( normal.x * 0.58, abs( normal.y ) * 0.70 + 0.58, normal.z * 0.58 ) );`)
        .replace('#include <lights_fragment_end>', `#include <lights_fragment_end>
{
  // Forward-scattered sunlight through a thin blade. Peaks when the sun is
  // directly behind the plant, which is precisely when a real meadow lights up
  // and the seed-heads turn into little filaments of fire. Cheaper and steadier
  // than a two-sided BRDF, and it is most of why these read as alive.
  vec3 sunV = ( viewMatrix * vec4( uSunDir, 0.0 ) ).xyz;
  float fwd = max( dot( normalize( -vViewPosition ), normalize( sunV ) ), 0.0 );
  float glow = fwd * fwd * fwd * 0.86 + 0.14;
  reflectedLight.directDiffuse += diffuseColor.rgb * uSunCol * ( glow * vTrans );
}`);
    };
    this.matFlora = flora;
    this._disposables.push(flora);

    // --- Fallen petals: flat, no wind, tinted per instance.
    //     `vertexColors` is load-bearing, not decoration: three's color_fragment
    //     only multiplies diffuseColor by vColor under USE_COLOR, which comes
    //     from this flag alone. Without it the vertex stage still folds
    //     instanceColor into vColor and the fragment stage never reads it, so
    //     every per-instance tint in the group - the 22 % browned petals and the
    //     whole value jitter - is computed, uploaded, and silently discarded.
    //     The petal geometry carries a matching all-ones `color` attribute; with
    //     USE_COLOR defined and no such attribute, WebGL feeds the shader the
    //     default generic vertex attribute (0,0,0) and every petal goes black.
    const petals = new THREE.MeshLambertMaterial({
      map: petalTex,
      alphaTest: 0.45,
      side: THREE.DoubleSide,
      vertexColors: true,
      transparent: false,
      dithering: true,
    });
    petals.customProgramCacheKey = () => 'scatter-petal';
    petals.onBeforeCompile = (sh) => {
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', `#include <common>
attribute vec2 aFade;`)
        .replace('#include <begin_vertex>', `#include <begin_vertex>
{
  vec3 sOrigin = ( modelMatrix * instanceMatrix * vec4( 0.0, 0.0, 0.0, 1.0 ) ).xyz;
  transformed *= 1.0 - smoothstep( aFade.x, aFade.y, distance( cameraPosition, sOrigin ) );
}`);
    };
    this.matPetal = petals;
    this._disposables.push(petals);
  }

  // -------------------------------------------------------------------------
  // Instanced groups
  // -------------------------------------------------------------------------

  _buildGroups() {
    // `sink` is a fraction of the instance's *width*; a rock is roughly half as tall
    // as it is wide, so 0.13 buries about a quarter of its height.
    //
    // `cell` MUST divide CHUNK exactly (32 / integer) or the placement lattice
    // slips relative to the chunk grid and props duplicate along chunk seams.
    this.groups = [
      this._group({
        id: 'rockS', kind: 'rock', salt: 1013, cell: 4, baseRadius: 44,
        density: 0.55, maxPerChunk: 40, fill: 0.62,
        geo: this._makeRockGeometry(11, 1, { flat: 0.55, cuts: 5, chip: 0.9, bed: 0 }),
        size: [0.11, 0.34], sizePow: 1.9, sink: 0.16, align: 0.55, tilt: 0.34,
        fadeK: [22, 60], shadow: false, field: 'rock',
      }),
      this._group({
        id: 'rockM', kind: 'rock', salt: 2027, cell: 8, baseRadius: 98,
        density: 0.45, maxPerChunk: 14, fill: 0.62,
        geo: this._makeRockGeometry(23, 2, { flat: 0.62, cuts: 7, chip: 1.0, bed: 0.15 }),
        size: [0.38, 1.15], sizePow: 1.9, sink: 0.13, align: 0.72, tilt: 0.22,
        fadeK: [26, 62], shadow: true, field: 'rock',
      }),
      this._group({
        id: 'rockL', kind: 'rock', salt: 3041, cell: 32, baseRadius: 152,
        density: 0.30, maxPerChunk: 2, fill: 0.62,
        geo: this._makeRockGeometry(37, 2, { flat: 0.80, cuts: 9, chip: 0.8, bed: 0.42 }),
        size: [1.25, 2.70], sizePow: 1.9, sink: 0.11, align: 0.82, tilt: 0.14,
        fadeK: [34, 58], shadow: true, field: 'rock',
      }),
      this._group({
        id: 'branch', kind: 'stick', salt: 4051, cell: 8, baseRadius: 96,
        density: 0.20, maxPerChunk: 10, fill: 0.62,
        geo: this._makeStickGeometry(53, 'branch'),
        size: [0.90, 2.30], sizePow: 1.4, sink: 0, align: 0.95, tilt: 0.10,
        fadeK: [22, 40], shadow: true, field: 'stick', slopeMin: 0.78,
      }),
      this._group({
        id: 'twig', kind: 'stick', salt: 5077, cell: 4, baseRadius: 44,
        density: 0.22, maxPerChunk: 26, fill: 0.62,
        geo: this._makeStickGeometry(67, 'twig'),
        size: [0.28, 0.62], sizePow: 1.4, sink: 0, align: 0.95, tilt: 0.16,
        fadeK: [18, 40], shadow: false, field: 'stick', slopeMin: 0.78,
      }),

      // --- the meadow itself -------------------------------------------------
      // Radii and fade windows are staggered on purpose: the tall silhouettes
      // survive furthest, ground cover gives up first. That is both the cheapest
      // and the most truthful LOD there is - at 20 m you genuinely cannot resolve
      // a clover leaf, but you can absolutely see a seed-head against the sky.
      // `cover` and `bent` are the two fill-rate items in this module and the
      // reason it was over budget on the 780M. Card area per square metre of
      // ground, integrated over each group's own reach, came out at roughly
      // 0.36 (cover) and 0.45 (bent) against 0.15 for the seed-heads and under
      // 0.03 for everything else - i.e. these two alone laid very nearly one
      // extra full-coverage alpha-tested layer over the near field, on top of
      // the sward. Neither is a silhouette the eye can resolve: cover is a
      // 20 cm mat and bent is hair-thin. So both are thinned and both give up
      // earlier, and the tall species - the ones the brief actually leans on - 
      // are untouched.
      this._group({
        id: 'cover', kind: 'plant', salt: 6091, cell: 1, baseRadius: 10,
        density: 0.58, maxPerChunk: 340, fill: 0.58, lowScale: 0.30,
        geo: this._makeCoverGeometry(),
        size: [0.12, 0.28], sizePow: 1.15, sink: 0, align: 0.85, tilt: 0.10,
        fadeK: [4.5, 18], shadow: false, field: 'cover',
        stiff: [0.30, 0.58], slopeMin: 0.78, trample: [0.30, 0.62],
      }),
      this._group({
        id: 'bent', kind: 'plant', salt: 6217, cell: 1.6, baseRadius: 24,
        density: 0.62, maxPerChunk: 170, fill: 0.55, lowScale: 0.55,
        geo: this._makeBentGeometry(),
        size: [0.30, 0.78], sizePow: 1.25, sink: 0, align: 0.45, tilt: 0.06,
        fadeK: [7, 21], shadow: false, field: 'bent',
        stiff: [0.72, 1.10], slopeMin: 0.82, trample: [0.25, 0.60], variants: 2,
        variantStep: STEP_U(4), variantFreq: 0.021,
      }),
      this._group({
        id: 'broad', kind: 'plant', salt: 6337, cell: 2, baseRadius: 36,
        density: 0.50, maxPerChunk: 130, fill: 0.40, lowScale: 0.55,
        geo: this._makeBroadGeometry(),
        size: [0.22, 0.62], sizePow: 1.35, sink: 0, align: 0.78, tilt: 0.08,
        fadeK: [11, 38], shadow: false, field: 'broad',
        stiff: [0.34, 0.60], slopeMin: 0.82, trample: [0.35, 0.70],
      }),
      this._group({
        id: 'stalk', kind: 'plant', salt: 6449, cell: 32 / 23, baseRadius: 66,
        density: 0.50, maxPerChunk: 270, fill: 0.45, lowScale: 0.62,
        geo: this._makeStalkGeometry(),
        size: [0.92, 1.72], sizePow: 1.5, sink: 0, align: 0.24, tilt: 0.05,
        fadeK: [22, 34], shadow: false, field: 'stalk',
        stiff: [1.05, 1.50], slopeMin: 0.80, trample: [0.20, 0.62], variants: 2,
        variantStep: STEP_U(4), variantFreq: 0.013,
      }),
      this._group({
        id: 'bloom', kind: 'plant', salt: 6571, cell: 2, baseRadius: 27,
        density: 0.50, maxPerChunk: 130, fill: 0.35, lowScale: 0.50,
        geo: this._makeBloomGeometry(),
        size: [0.20, 0.52], sizePow: 1.2, sink: 0, align: 0.42, tilt: 0.07,
        fadeK: [9, 32], shadow: false, field: 'bloom',
        stiff: [0.62, 0.98], slopeMin: 0.84, trample: [0.30, 0.66], variants: 3,
        variantStep: STEP_U(4), variantFreq: 0.030,
      }),
      this._group({
        id: 'reed', kind: 'plant', salt: 6689, cell: 32 / 9, baseRadius: 70,
        density: 0.38, maxPerChunk: 34, fill: 0.32, lowScale: 0.70,
        geo: this._makeReedGeometry(),
        size: [1.05, 2.10], sizePow: 1.5, sink: 0, align: 0.20, tilt: 0.05,
        fadeK: [26, 30], shadow: false, field: 'reed',
        stiff: [1.20, 1.60], slopeMin: 0.86, trample: [0.20, 0.70],
      }),

      this._group({
        id: 'petal', kind: 'petal', salt: 7103, cell: 1.6, baseRadius: 32,
        density: 0.55, maxPerChunk: 200, fill: 0.62,
        geo: this._makePetalGeometry(),
        size: [0.028, 0.050], sizePow: 1, sink: 0, align: 1.0, tilt: 0.14,
        fadeK: [16, 300], shadow: false, field: 'petal', slopeMin: 0.90,
      }),
    ];
  }

  _group(def) {
    // Capacity is sized for the widest tier this module will ever run at, so the
    // instance buffers are allocated exactly once, at load. The widest reach is
    // grassDistance 196 (scale 1.4) on ULTRA (tierRadius 1.12), and packing pulls
    // from every chunk whose CENTRE is within radius + 0.75 chunks.
    const reach = def.baseRadius * 1.4 * 1.12 + CHUNK * 0.75;
    const span = reach / CHUNK;
    // Number of chunk CENTRES inside `reach` of an arbitrary point. It has to be
    // the real worst case for a lattice, pi*(span + halfDiagonal)^2, not the disc
    // area: the naive pi*span^2 + 2*span + 1 under-counts by ~15 % (59 against 70
    // for the seed-heads), and packing handles an overflow by dropping the
    // FARTHEST chunks - which on ULTRA, where the reach is 1.57x the reference,
    // would draw a hard circular edge across the field instead of a fade.
    // The whole correction costs 0.4 MB of instance buffer.
    const chunkCount = Math.ceil(Math.PI * (span + Math.SQRT1_2) * (span + Math.SQRT1_2));
    const capacity = Math.min(
      24000,
      Math.ceil(chunkCount * def.maxPerChunk * (def.fill ?? 0.6)) + def.maxPerChunk);

    const material =
      def.kind === 'plant' ? this.matFlora : def.kind === 'petal' ? this.matPetal : this.matGround;

    const mesh = new THREE.InstancedMesh(def.geo, material, capacity);
    mesh.name = `scatter.${def.id}`;
    mesh.count = 0;
    mesh.visible = false;
    // The packed set always encircles the camera - its bounding sphere contains the
    // near plane by construction, so a frustum test can only ever return "visible".
    // Leaving it on would pay for a sphere transform per mesh per shadow cascade to
    // learn nothing. A REAL world-space sphere is still maintained in _pack() so
    // anything that reads mesh.boundingSphere (raycasts, editors, future passes)
    // gets the truth rather than a stale unit sphere at the origin.
    mesh.frustumCulled = false;
    mesh.matrixAutoUpdate = false;
    mesh.receiveShadow = true;
    mesh.castShadow = false;
    mesh.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 0);
    // Casters get the aFade-aware depth material, or their shadow outlives them.
    if (def.shadow && this.matGroundDepth) mesh.customDepthMaterial = this.matGroundDepth;
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3).fill(1), 3);
    mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);

    const attrs = [{ name: 'aFade', size: 2 }];
    if (def.kind === 'plant') attrs.push({ name: 'aInst', size: 4 });
    for (const a of attrs) {
      a.attr = new THREE.InstancedBufferAttribute(new Float32Array(capacity * a.size), a.size);
      a.attr.setUsage(THREE.DynamicDrawUsage);
      def.geo.setAttribute(a.name, a.attr);
    }

    this.root.add(mesh);
    this._disposables.push(def.geo);

    // Local half-extent of one unit-scaled instance, used to grow the world bounds
    // by the real geometry rather than by a guess.
    const bs = def.geo.boundingSphere;
    const localR = bs ? bs.center.length() + bs.radius : 1;

    return Object.assign(def, {
      mesh, capacity, attrs, localR,
      /** Set when this group's packed set is stale; cleared by _pack(). */
      dirty: true,
      radius: def.baseRadius,
      radiusK: 1,
      densityNow: def.density,
      wantsShadow: def.shadow,
      cellsPerChunk: Math.round(CHUNK / def.cell),
    });
  }

  // -------------------------------------------------------------------------
  // Geometry: rocks
  // -------------------------------------------------------------------------

  /**
   * A weathered rock is not a noisy sphere. Real ones are convex blocks broken by a
   * handful of flat fracture planes, rounded by weathering, dark where they meet the
   * soil and pale where lichen has taken the up-facing surfaces. So: anisotropic base
   * -> low-frequency lumps -> half-space plane cuts (the fractures) -> fine relief ->
   * bottom flatten -> bake cavity AO, contact darkening, lichen and iron stain into
   * vertex colours. No texture fetch at runtime, no UV seam to hide.
   */
  _makeRockGeometry(seed, detail, opt) {
    const rng = makeRNG(seed * 7919 + WORLD_SEED);
    const nz = createNoise(seed * 131 + 7);

    let geo = new THREE.IcosahedronGeometry(1, detail);
    geo.deleteAttribute('uv');
    geo.deleteAttribute('normal');
    geo = weldPositions(geo);

    const pos = geo.attributes.position;
    const arr = pos.array;
    const n = pos.count;

    const sx = rng.range(0.80, 1.30);
    const sy = rng.range(0.52, 0.88) * (1 - opt.flat * 0.35);
    const sz = rng.range(0.80, 1.30);
    const ox = rng.range(-40, 40), oy = rng.range(-40, 40), oz = rng.range(-40, 40);

    // Fracture planes, biased away from vertical so the crown stays rounded and the
    // flanks get the flat faces - which is what frost shattering actually produces.
    const cuts = [];
    for (let k = 0; k < opt.cuts; k++) {
      const a = rng() * TAU;
      const ny = rng.range(-0.55, 0.35);
      const rxz = Math.sqrt(Math.max(0, 1 - ny * ny));
      cuts.push(Math.cos(a) * rxz, ny, Math.sin(a) * rxz, rng.range(0.50, 0.86));
    }
    const beds = opt.bed > 0 ? 2 + Math.floor(rng() * 2) : 0;

    for (let i = 0; i < n; i++) {
      const j = i * 3;
      const ux = arr[j], uy = arr[j + 1], uz = arr[j + 2];

      let px = ux * sx, py = uy * sy, pz = uz * sz;

      const lump = nz.fbm3D(ux * 1.25 + ox, uy * 1.25 + oy, uz * 1.25 + oz, 3, 2.15, 0.52);
      const k1 = 1 + lump * 0.30 * opt.chip;
      px *= k1; py *= k1; pz *= k1;

      for (let c = 0; c < cuts.length; c += 4) {
        const d = px * cuts[c] + py * cuts[c + 1] + pz * cuts[c + 2] - cuts[c + 3];
        if (d > 0) { px -= cuts[c] * d; py -= cuts[c + 1] * d; pz -= cuts[c + 2] * d; }
      }
      for (let b = 0; b < beds; b++) {          // sedimentary bedding for the slab
        const level = -0.4 + (b + 0.5) * (0.9 / beds);
        const d = Math.abs(py - level);
        if (d < 0.055) {
          const t = 1 - d / 0.055;
          px *= 1 - 0.10 * t * opt.bed;
          pz *= 1 - 0.10 * t * opt.bed;
        }
      }

      // Fine relief applied radially, so it survives the plane cuts as surface
      // roughness instead of re-rounding the facets we just cut.
      const inv = 1 / Math.max(1e-5, Math.sqrt(px * px + py * py + pz * pz));
      const fine = nz.fbm3D(ux * 6.4 + 3.1, uy * 6.4 - 2.7, uz * 6.4 + 8.3, 2, 2.4, 0.5) * 0.030
        + nz.noise3D(ux * 15.0, uy * 15.0, uz * 15.0) * 0.012;
      px += px * inv * fine; py += py * inv * fine; pz += pz * inv * fine;

      arr[j] = px; arr[j + 1] = py; arr[j + 2] = pz;
    }

    // Flatten the base so the rock has a real footprint to sit on.
    let minY = Infinity, maxY = -Infinity, maxXZ = 0;
    for (let i = 0; i < n; i++) {
      const y = arr[i * 3 + 1];
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const flatY = minY + (maxY - minY) * 0.16 * (0.6 + opt.flat);
    for (let i = 0; i < n; i++) {
      const j = i * 3 + 1;
      if (arr[j] < flatY) arr[j] = flatY + (arr[j] - flatY) * 0.18;
    }

    // Normalise so instance scale reads as "footprint width in metres", and move the
    // pivot to the contact plane so sinking and ground alignment are trivial.
    minY = Infinity; maxY = -Infinity;
    for (let i = 0; i < n; i++) {
      const j = i * 3;
      const y = arr[j + 1];
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      const r = Math.hypot(arr[j], arr[j + 2]);
      if (r > maxXZ) maxXZ = r;
    }
    const k = 0.5 / Math.max(1e-4, maxXZ);
    for (let i = 0; i < n * 3; i++) arr[i] *= k;
    minY *= k; maxY *= k;
    for (let i = 0; i < n; i++) arr[i * 3 + 1] -= minY;
    const height = maxY - minY;

    pos.needsUpdate = true;
    geo.computeVertexNormals();
    geo.computeBoundingSphere();
    geo.computeBoundingBox();

    // --- vertex colours
    const nrm = geo.attributes.normal.array;
    const col = new Float32Array(n * 3);
    const warm = new THREE.Color('#7d766b');
    const cool = new THREE.Color('#666b6e');
    const lichen = new THREE.Color('#98a08c');
    const stain = new THREE.Color('#84714f');

    let rMean = 0;
    for (let i = 0; i < n; i++) {
      rMean += Math.hypot(arr[i * 3], arr[i * 3 + 1] - height * 0.5, arr[i * 3 + 2]);
    }
    rMean /= n;

    for (let i = 0; i < n; i++) {
      const j = i * 3;
      const x = arr[j], y = arr[j + 1], z = arr[j + 2];
      const ny = nrm[j + 1];

      const tone = nz.fbm3D(x * 2.6 + 21, y * 2.6, z * 2.6 - 13, 2, 2.2, 0.5) * 0.5 + 0.5;
      _c0.copy(cool).lerp(warm, clamp01(tone * 1.25 - 0.12));

      // Cavity: vertices inside the mean radius are pits, and pits hold shadow.
      const r = Math.hypot(x, y - height * 0.5, z);
      _c0.multiplyScalar(lerp(0.66, 1.06, clamp01((r - rMean * 0.86) / (rMean * 0.30))));
      // Contact darkening at the soil line - sells "sitting in" over "sitting on".
      // Kept tight (a third of the height) so it reads as occlusion, not as dirt.
      _c0.multiplyScalar(lerp(0.50, 1.0, smoothstep(0, height * 0.32, y)));

      const lich = smoothstep(0.20, 0.68, ny)
        * smoothstep(0.46, 0.66, nz.fbm3D(x * 3.4 - 5, y * 3.4 + 9, z * 3.4 + 2, 2, 2.3, 0.5) * 0.5 + 0.5);
      _c0.lerp(lichen, lich * 0.52);

      // Iron staining runs downward off the shoulders.
      const st = smoothstep(0.58, 0.90, nz.fbm3D(x * 2.1, y * 5.2 + 30, z * 2.1, 2, 2.0, 0.5) * 0.5 + 0.5)
        * clamp01(0.75 - ny) * 0.30;
      _c0.lerp(stain, st);

      col[j] = _c0.r; col[j + 1] = _c0.g; col[j + 2] = _c0.b;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    return geo;
  }

  // -------------------------------------------------------------------------
  // Geometry: fallen wood
  // -------------------------------------------------------------------------

  _makeStickGeometry(seed, kind) {
    const rng = makeRNG(seed * 6151 + WORLD_SEED);
    const P = [], I = [], C = [];
    const bark = new THREE.Color('#6a6055');
    const heart = new THREE.Color('#4a3d31');
    const bleach = new THREE.Color('#8c8474');

    const limbs = kind === 'twig' ? 3 : 1 + (rng() < 0.55 ? 1 : 0);
    for (let l = 0; l < limbs; l++) {
      const yaw = kind === 'twig' ? rng() * TAU : (l === 0 ? 0 : rng.range(0.5, 1.1) * rng.sign());
      // Thickness is a fraction of the instance's length, because the whole prop is
      // uniformly scaled. 0.024 puts a 2.3 m limb at ~11 cm and a 0.9 m one at
      // ~4 cm, which is where real fallen wood sits.
      const thick = kind === 'twig' ? 0.015 : 0.024;
      this._appendLimb(P, I, C, {
        yaw,
        len: l === 0 ? 1.0 : rng.range(0.28, 0.50),
        r0: thick * rng.range(0.80, 1.25) * (l === 0 ? 1 : 0.55),
        start: l === 0 ? 0 : rng.range(0.20, 0.70),
        taper: rng.range(0.20, 0.42),
        bend: rng.range(0.05, 0.20) * rng.sign(),
        lift: kind === 'twig' ? rng.range(0, 0.05) : rng.range(0, 0.035),
        grain: rng.range(0, 6.283),
        bark, heart, bleach,
      });
    }

    // Centre the footprint and put the pivot on the ground contact.
    let minX = Infinity, maxX = -Infinity, minY = Infinity, minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < P.length; i += 3) {
      minX = Math.min(minX, P[i]); maxX = Math.max(maxX, P[i]);
      minY = Math.min(minY, P[i + 1]);
      minZ = Math.min(minZ, P[i + 2]); maxZ = Math.max(maxZ, P[i + 2]);
    }
    const cx = (minX + maxX) * 0.5, cz = (minZ + maxZ) * 0.5;
    const s = 1 / Math.max(1e-4, Math.max(maxX - minX, maxZ - minZ)); // scale == length in metres
    for (let i = 0; i < P.length; i += 3) {
      P[i] = (P[i] - cx) * s;
      P[i + 1] = (P[i + 1] - minY) * s;
      P[i + 2] = (P[i + 2] - cz) * s;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(P, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(C, 3));
    geo.setIndex(I);
    geo.computeVertexNormals();
    geo.computeBoundingSphere();
    geo.computeBoundingBox();
    return geo;
  }

  _appendLimb(P, I, C, o) {
    const RAD = 6, SEG = 7;
    const base = P.length / 3;
    const dx = Math.cos(o.yaw), dz = Math.sin(o.yaw);
    const nx = -dz, nz = dx;      // horizontal perpendicular; the limb lies flat

    for (let s = 0; s <= SEG; s++) {
      const t = s / SEG;
      const along = o.start + t * o.len;
      const swerve = Math.sin(t * Math.PI) * o.bend * o.len;
      const r = o.r0 * lerp(1, o.taper, t * t * 0.7 + t * 0.3);
      const cxp = dx * along + nx * swerve;
      const czp = dz * along + nz * swerve;
      const cyp = r + Math.sin(t * Math.PI) * o.lift;   // underside rests on y = lift

      for (let a = 0; a < RAD; a++) {
        const th = (a / RAD) * TAU;
        const ca = Math.cos(th), sa = Math.sin(th);
        const rr = r * (0.86 + 0.28 * (0.5 + 0.5 * Math.cos(th * 3 + s * 0.9 + o.grain)));
        P.push(cxp + nx * ca * rr, cyp + sa * rr, czp + nz * ca * rr);

        // Bleached and dry on top, dark and damp underneath, with longitudinal
        // stripes where the bark has come away.
        const up = sa * 0.5 + 0.5;
        const strip = 0.5 + 0.5 * Math.sin(th * 2.0 + along * 9.0 + o.grain);
        _c0.copy(strip < 0.30 ? o.heart : o.bark);
        _c0.lerp(o.bleach, clamp01(up - 0.35) * 0.55);
        _c0.multiplyScalar(lerp(0.34, 1.0, up));
        C.push(_c0.r, _c0.g, _c0.b);
      }
    }
    for (let s = 0; s < SEG; s++) {
      for (let a = 0; a < RAD; a++) {
        const a2 = (a + 1) % RAD;
        const i0 = base + s * RAD + a, i1 = base + s * RAD + a2;
        const i2 = base + (s + 1) * RAD + a, i3 = base + (s + 1) * RAD + a2;
        I.push(i0, i2, i1, i1, i2, i3);
      }
    }
    // End caps - a hollow tube reads as a bug the moment the sun is behind it.
    for (let e = 0; e < 2; e++) {
      const s = e === 0 ? 0 : SEG;
      const centre = P.length / 3;
      let ax = 0, ay = 0, az = 0;
      for (let a = 0; a < RAD; a++) {
        const q = (base + s * RAD + a) * 3;
        ax += P[q]; ay += P[q + 1]; az += P[q + 2];
      }
      P.push(ax / RAD, ay / RAD, az / RAD);
      _c0.copy(o.heart).multiplyScalar(1.15);
      C.push(_c0.r, _c0.g, _c0.b);
      for (let a = 0; a < RAD; a++) {
        const a2 = (a + 1) % RAD;
        const i0 = base + s * RAD + a, i1 = base + s * RAD + a2;
        if (e === 0) I.push(centre, i1, i0); else I.push(centre, i0, i1);
      }
    }
  }

  // -------------------------------------------------------------------------
  // Geometry: plants
  // -------------------------------------------------------------------------

  /**
   * Builds one plant from a list of cards.
   *
   * Local space is normalised so the plant is exactly 1 unit tall, which means the
   * length of the instance matrix's Y column IS the plant's height in metres - the
   * wind shader reads it to keep bend proportional without a second attribute.
   *
   * Cards are plain rectangular strips. The silhouette lives entirely in the atlas
   * alpha, so a dock leaf and a seed-head cost the same six vertices, and adding a
   * species costs a tile rather than a mesh.
   *
   * card = {
   *   leaf     false: a vertical blade card whose width runs along `yaw` and which
   *                   leans out of its own plane. true: a leaf card whose SPINE runs
   *                   along `yaw` and whose width runs across it.
   *   yaw      radians about +Y
   *   seg      spine segments (1..4)
   *   w        card width in local units - must match the tile aspect or the
   *            drawing comes out stretched
   *   yBase    height the card starts at
   *   tipY     height the card ends at
   *   reach    horizontal displacement of the tip
   *   ctrlX/Y  quadratic control point, gives the arch
   *   atlas    uv rect [u0, v0, du, dv]
   *   mirror   flips uv.x - two leaves from one tile
   *   ao/aoH   baked base occlusion and the height it fades over
   *   value    per-card shade multiplier; crossed cards must not match exactly
   *   trans    translucency 0..1 for the backlight term
   *   flut     tip-flutter weight
   * }
   */
  _makePlantGeometry(cards) {
    const P = [], N = [], U = [], C = [], A = [], LF = [], I = [];

    for (const card of cards) {
      const seg = Math.max(1, card.seg | 0);
      const dx = Math.cos(card.yaw), dz = Math.sin(card.yaw);
      const px = -dz, pz = dx;
      const leaf = !!card.leaf;
      // A leaf grows out along its yaw and is wide across it; a blade card is wide
      // along its yaw and leans out across it.
      const wx = leaf ? px : dx, wz = leaf ? pz : dz;
      const ox = leaf ? dx : px, oz = leaf ? dz : pz;
      const base = P.length / 3;

      const x0 = 0, y0 = card.yBase;
      const x1 = card.reach, y1 = card.tipY;
      const cxs = card.ctrlX, cys = card.ctrlY;

      for (let s = 0; s <= seg; s++) {
        const v = s / seg;
        const t = 1 - v;
        const r = t * t * x0 + 2 * t * v * cxs + v * v * x1;
        const y = t * t * y0 + 2 * t * v * cys + v * v * y1;
        let drx = 2 * t * (cxs - x0) + 2 * v * (x1 - cxs);
        let dry = 2 * t * (cys - y0) + 2 * v * (y1 - cys);
        const dl = Math.hypot(drx, dry);
        if (dl > 1e-6) { drx /= dl; dry /= dl; } else { drx = 0; dry = 1; }

        // Face normal: perpendicular to both the width axis and the spine tangent.
        // The cross product is taken width x tangent, NOT tangent x width - the
        // other handedness points a flattening leaf tip at the ground, and since
        // the up-bias below is added blind, the leaf would then be lit from under
        // the soil.
        let nx, ny, nz;
        if (leaf) {
          const tx = dx * drx, ty = dry, tz = dz * drx;
          nx = -ty * pz;
          ny = tx * pz - tz * px;
          nz = ty * px;
        } else {
          nx = px; ny = 0; nz = pz;
        }
        // Tilt every normal toward the sky before it is ever lit. Half the reason a
        // card of foliage looks like a sticker is that its true plane normal makes
        // it flip from lit to black as the sun crosses it.
        const nl0 = Math.hypot(nx, ny, nz) || 1;
        ny += 0.55 * nl0;
        const nl = 1 / (Math.hypot(nx, ny, nz) || 1);
        nx *= nl; ny *= nl; nz *= nl;

        const ao = lerp(card.ao, 1, smoothstep(0, card.aoH, y)) * card.value;

        for (let e = 0; e < 2; e++) {
          const side = (e - 0.5) * card.w;
          P.push(ox * r + wx * side, y, oz * r + wz * side);
          N.push(nx, ny, nz);
          U.push(card.mirror ? 1 - e : e, v);
          C.push(ao, ao, ao);
          A.push(card.atlas[0], card.atlas[1], card.atlas[2], card.atlas[3]);
          LF.push(card.trans, card.flut);
        }
      }
      for (let s = 0; s < seg; s++) {
        const i0 = base + s * 2;
        I.push(i0, i0 + 2, i0 + 1, i0 + 1, i0 + 2, i0 + 3);
      }
    }

    // Normalise the whole plant to exactly one unit tall. Uniform, so tile aspects
    // survive, and it means every species' instance scale is "height in metres".
    let top = 0;
    for (let i = 1; i < P.length; i += 3) if (P[i] > top) top = P[i];
    if (top > 1e-4 && Math.abs(top - 1) > 1e-4) {
      const k = 1 / top;
      for (let i = 0; i < P.length; i++) P[i] *= k;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(P, 3));
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(N, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(U, 2));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(C, 3));
    geo.setAttribute('aAtlas', new THREE.Float32BufferAttribute(A, 4));
    geo.setAttribute('aLeaf', new THREE.Float32BufferAttribute(LF, 2));
    geo.setIndex(I);
    geo.computeBoundingSphere();
    geo.computeBoundingBox();
    return geo;
  }

  /**
   * Ground cover: four small leaves splayed low and wide from a common crown.
   * Deliberately much broader than it is tall - real cover spreads as a MAT, and
   * one 50 cm mat closes far more of the gap between blades than four upright
   * 12 cm sprigs would for the same four cards.
   */
  _makeCoverGeometry() {
    const cards = [];
    for (let k = 0; k < 4; k++) {
      const reach = 0.62 + 0.12 * ((k + 1) % 2);
      const tipY = 0.86 + 0.14 * (k % 2);
      cards.push({
        leaf: true, seg: 2, yaw: 0.55 + k * 1.5708 + (k % 2) * 0.28,
        // The tile is square, so the card must be as wide as the leaf is long or
        // the drawing comes out stretched along the arch.
        w: Math.hypot(reach, tipY - 0.04),
        yBase: 0.04, tipY, reach, ctrlX: reach * 0.24, ctrlY: 0.80,
        atlas: UV_COVER(k), mirror: k === 1 || k === 2,
        ao: 0.52, aoH: 0.55, value: 0.90 + 0.10 * (k % 2),
        trans: 0.55, flut: 0.45,
      });
    }
    return this._makePlantGeometry(cards);
  }

  /** Fine bent-grass: three crossed fans, splaying outward at the tips. */
  _makeBentGeometry() {
    const cards = [];
    for (let k = 0; k < 3; k++) {
      const lean = (k === 1 ? -0.11 : 0.09) * (k === 2 ? -1 : 1);
      cards.push({
        leaf: false, seg: 2, yaw: 0.35 + k * 1.047,
        w: 1.0, yBase: 0, tipY: 1.0,
        reach: lean, ctrlX: lean * 0.18, ctrlY: 0.56,
        atlas: UV_BENT(k), mirror: k === 1,
        ao: 0.38, aoH: 0.48, value: k === 0 ? 1.0 : 0.90,
        trans: 0.88, flut: 1.0,
      });
    }
    return this._makePlantGeometry(cards);
  }

  /** Dock / broad-leaf rosette: four leaves of different length and droop. */
  _makeBroadGeometry() {
    const spec = [
      { yaw: 0.30, reach: 0.60, tipY: 0.98, v: 0, mir: false, w: 0.58 },
      { yaw: 1.95, reach: 0.52, tipY: 0.78, v: 1, mir: true, w: 0.50 },
      { yaw: 3.42, reach: 0.66, tipY: 0.64, v: 0, mir: true, w: 0.52 },
      { yaw: 4.92, reach: 0.44, tipY: 0.90, v: 1, mir: false, w: 0.50 },
    ];
    return this._makePlantGeometry(spec.map((s, k) => ({
      leaf: true, seg: 2, yaw: s.yaw, w: s.w, yBase: 0.05, tipY: s.tipY,
      reach: s.reach, ctrlX: s.reach * 0.22, ctrlY: s.tipY * 0.80,
      atlas: UV_BROAD(s.v), mirror: s.mir,
      ao: 0.48, aoH: 0.34, value: 0.90 + 0.10 * (k % 2),
      trans: 0.62, flut: 0.32,
    })));
  }

  /** Seed-head stalks: two narrow crossed cards, four segments so the arc is real. */
  _makeStalkGeometry() {
    const cards = [];
    for (let k = 0; k < 2; k++) {
      const lean = k === 0 ? 0.055 : -0.045;
      cards.push({
        leaf: false, seg: 3, yaw: 0.28 + k * 1.571,
        w: 0.25, yBase: 0, tipY: 1.0,
        reach: lean, ctrlX: lean * 0.12, ctrlY: 0.52,
        atlas: UV_STALK(k), mirror: k === 1,
        ao: 0.42, aoH: 0.30, value: k === 0 ? 1.0 : 0.92,
        trans: 1.0, flut: 1.0,
      });
    }
    return this._makePlantGeometry(cards);
  }

  /** Wildflower sprig: one tile, two crossed cards, mirrored so they differ. */
  _makeBloomGeometry() {
    const cards = [];
    for (let k = 0; k < 2; k++) {
      const lean = k === 0 ? 0.07 : -0.06;
      cards.push({
        leaf: false, seg: 2, yaw: 0.42 + k * 1.571,
        w: 1.0, yBase: 0, tipY: 1.0,
        reach: lean, ctrlX: lean * 0.2, ctrlY: 0.55,
        atlas: UV_BLOOM(0), mirror: k === 1,
        ao: 0.44, aoH: 0.42, value: k === 0 ? 1.0 : 0.91,
        trans: 0.92, flut: 0.85,
      });
    }
    return this._makePlantGeometry(cards);
  }

  /** Reed stand: two tall crossed cards. */
  _makeReedGeometry() {
    const cards = [];
    for (let k = 0; k < 2; k++) {
      const lean = k === 0 ? 0.085 : -0.070;
      cards.push({
        leaf: false, seg: 3, yaw: 0.22 + k * 1.571,
        w: 0.375, yBase: 0, tipY: 1.0,
        reach: lean, ctrlX: lean * 0.15, ctrlY: 0.54,
        atlas: UV_REED(k), mirror: k === 1,
        ao: 0.36, aoH: 0.34, value: k === 0 ? 1.0 : 0.93,
        trans: 0.95, flut: 1.0,
      });
    }
    return this._makePlantGeometry(cards);
  }

  _makePetalGeometry() {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(
      [-0.5, 0, -0.5, 0.5, 0, -0.5, -0.5, 0, 0.5, 0.5, 0, 0.5], 3));
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(
      [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 0, 1, 1, 1], 2));
    // Identity vertex colour. matPetal needs `vertexColors` on for instanceColor
    // to reach the fragment shader at all, and USE_COLOR without a `color`
    // attribute reads the generic attribute default (0,0,0) - black petals.
    geo.setAttribute('color', new THREE.Float32BufferAttribute(
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 3));
    geo.setIndex([0, 2, 1, 1, 2, 3]);
    geo.computeBoundingSphere();
    geo.computeBoundingBox();
    return geo;
  }

  // -------------------------------------------------------------------------
  // Procedural textures: flora atlas
  // -------------------------------------------------------------------------

  _makeFloraAtlas(aniso) {
    const g = makeCanvas(ATLAS_SIZE, ATLAS_SIZE);
    g.lineCap = 'round';
    g.lineJoin = 'round';

    // Every tile is drawn inside a hard clip of its own rect. The per-species
    // extents below are tuned to fit, but a stray gaussian tail that leaked a dock
    // leaf onto the wildflower tile would be invisible in code review and glaring
    // on screen - and there is no cheaper insurance than one clip() per tile.
    const T = TILE;
    const draw = (fn, rng, R, k) => {
      g.save();
      g.beginPath();
      g.rect(R.x, R.y, R.w, R.h);
      g.clip();
      fn.call(this, g, rng, R, k);
      g.restore();
    };

    for (let k = 0; k < 4; k++) {
      draw(this._drawStalkTile, makeRNG(20011 + k * 733), tileRect(T.stalk[0] + k * 2, T.stalk[1], 2, 8), k);
    }
    for (let k = 0; k < 2; k++) {
      draw(this._drawReedTile, makeRNG(31013 + k * 911), tileRect(T.reed[0] + k * 3, T.reed[1], 3, 8), k);
    }
    for (let k = 0; k < 4; k++) {
      draw(this._drawCoverTile, makeRNG(41017 + k * 617), tileRect(T.cover[0], T.cover[1] + k * 2, 2, 2), k);
    }
    for (let k = 0; k < 4; k++) {
      draw(this._drawBentTile, makeRNG(51019 + k * 887), tileRect(T.bent[0] + k * 4, T.bent[1], 4, 4), k);
    }
    for (let k = 0; k < 3; k++) {
      draw(this._drawBloomTile, makeRNG(61031 + k * 977), tileRect(T.bloom[0] + k * 4, T.bloom[1], 4, 4), k);
    }
    for (let k = 0; k < 2; k++) {
      draw(this._drawBroadTile, makeRNG(71039 + k * 691), tileRect(T.broad[0] + k * 2, T.broad[1], 2, 4), k);
    }

    return textureFromImageData(dilateAlpha(g, ATLAS_SIZE, ATLAS_SIZE, 3), aniso);
  }

  /**
   * A seed-head stalk. Four real inflorescence architectures, because "grass with a
   * fuzzy bit on top" repeated across a field is exactly the sameness we are here to
   * break: an open panicle, a compact spike, a nodding oat-like head, and a soft
   * bloomed head. Each gets one or two sheath leaves low on the culm - a bare stick
   * with a head on it reads as a lollipop.
   */
  _drawStalkTile(g, rng, R, variant) {
    const bx = R.x + R.w * 0.5;
    const by = R.y + R.h;
    // The head owns the top of the tile and the culm runs up into it. Splitting the
    // tile this way is what keeps every decoration inside its own rect no matter how
    // far a gaussian tail wanders - a spikelet drawn past the tile edge lands on the
    // neighbouring species.
    const headFrac = variant === 0 ? 0.46 : variant === 2 ? 0.42 : 0.34;
    const headTop = R.y + R.h * rng.range(0.035, 0.075);
    const headBase = R.y + R.h * headFrac;
    // The culm ends at the head apex for the panicle types (they carry the axis all
    // the way up) and at the head base for the two solid heads.
    const tipY = variant === 1 || variant === 3 ? headBase : headTop;
    const tipX = bx + rng.range(-0.10, 0.10) * R.w;
    const stemCX = bx + rng.range(-0.07, 0.07) * R.w;
    const stemCY = by - (by - tipY) * 0.55;

    // --- sheath leaves, low on the culm. A bare stick with a head on it is a
    //     lollipop; two blades at the base is all it takes to stop that.
    const leaves = 2 + (rng() < 0.5 ? 1 : 0);
    for (let i = 0; i < leaves; i++) {
      const dir = rng.sign();
      const lh = R.h * rng.range(0.18, 0.38);
      const y0 = by - R.h * rng.range(0.01, 0.12);
      const lx = bx + dir * R.w * rng.range(0.16, 0.40);
      ribbon(g, bx, y0,
        bx + dir * R.w * 0.13, y0 - lh * 0.72,
        lx, y0 - lh * rng.range(0.55, 0.95),
        wBlade(rng.range(4.0, 6.6), 0.85),
        rng() < 0.4 ? PAL.leafDeep : PAL.leaf, 10);
    }

    // --- culm
    const stemCol = variant === 3 ? PAL.dryDeep : (rng() < 0.4 ? PAL.stemDark : PAL.stem);
    ribbon(g, bx, by, stemCX, stemCY, tipX, tipY, wStem(rng.range(3.4, 4.4), 1.5), stemCol, 16);
    // A node or two: real culms are jointed, and the joint catches a highlight.
    for (let i = 0; i < 2; i++) {
      const t = 0.28 + i * 0.30;
      const s = 1 - t;
      const nx = s * s * bx + 2 * s * t * stemCX + t * t * tipX;
      const ny = s * s * by + 2 * s * t * stemCY + t * t * tipY;
      g.fillStyle = PAL.stemPale;
      g.beginPath();
      g.ellipse(nx, ny, 2.6, 1.7, 0, 0, TAU);
      g.fill();
    }

    // Point along the culm, used to hang the head off it.
    const at = (t) => {
      const s = 1 - t;
      return [s * s * bx + 2 * s * t * stemCX + t * t * tipX,
      s * s * by + 2 * s * t * stemCY + t * t * tipY];
    };
    // Half-width the head may occupy without touching the tile edge.
    const room = R.w * 0.44;

    if (variant === 0) {
      // Open panicle - bent-grass / Agrostis. Whorls of hair-fine branchlets, each
      // ending in two or three spikelets, widest at the bottom of the head. Almost
      // transparent, which is exactly what makes a backlit meadow shimmer.
      const whorls = 7;
      for (let wI = 0; wI < whorls; wI++) {
        const f = wI / (whorls - 1);
        const ny = lerp(headBase, headTop, f);
        // Solve back onto the culm rather than trusting a t: the culm is a curve.
        const t = clamp01((by - ny) / Math.max(1, by - tipY));
        const nx = at(t)[0];
        const branches = 3 + rng.int(0, 3);
        const reach = (room - Math.abs(nx - bx)) * lerp(0.92, 0.24, Math.pow(f, 0.75));
        for (let b = 0; b < branches; b++) {
          const dir = rng.sign();
          const rr = reach * rng.range(0.45, 1.0);
          const bxE = nx + dir * rr;
          const byE = ny - (ny - headTop) * rng.range(0.25, 0.75) - 2;
          ribbon(g, nx, ny, nx + dir * rr * 0.35, ny - (ny - byE) * 0.55, bxE, byE,
            wStem(1.5, 0.6), PAL.dryDeep, 8);
          const spikes = 1 + rng.int(0, 2);
          for (let s = 0; s < spikes; s++) {
            const sx = bxE + rng.range(-3, 3);
            const sy = byE - s * rng.range(3.5, 6.0);
            g.save();
            g.translate(sx, sy);
            g.rotate(rng.range(-0.9, 0.9) + (dir > 0 ? 0.5 : -0.5));
            g.fillStyle = rng() < 0.45 ? PAL.huskPale : PAL.husk;
            g.beginPath();
            g.ellipse(0, 0, 1.5, rng.range(3.2, 5.0), 0, 0, TAU);
            g.fill();
            g.restore();
          }
        }
      }
    } else if (variant === 1) {
      // Compact spike - Timothy. A dense cylinder of tiny florets; the silhouette is
      // the whole point, so build it from hundreds of short strokes rather than a
      // solid shape, or it reads as a drawn rectangle.
      const ax = tipX, ay = headBase;
      const tx2 = tipX + rng.range(-4, 4), ty2 = headTop;
      const n = 330;
      const maxRad = Math.min(R.w * 0.115, room - Math.abs(tipX - bx) - 8);
      for (let i = 0; i < n; i++) {
        const t = rng();
        const cx = lerp(ax, tx2, t);
        const cy = lerp(ay, ty2, t);
        const rad = maxRad * Math.pow(Math.sin(Math.PI * clamp01(t * 0.92 + 0.06)), 0.42);
        const off = clamp(rng.gaussian(0, rad * 0.46), -rad, rad);
        const a = -Math.PI * 0.5 + (off > 0 ? 1 : -1) * rng.range(0.55, 1.15);
        const len = rng.range(3.5, 7.0);
        const shade = rng();
        g.strokeStyle = shade < 0.30 ? PAL.dryDeep : shade < 0.72 ? PAL.dry : PAL.dryPale;
        g.globalAlpha = rng.range(0.55, 1.0);
        g.lineWidth = rng.range(1.1, 2.1);
        g.beginPath();
        g.moveTo(cx + off * 0.4, cy);
        g.lineTo(cx + off + Math.cos(a) * len, cy + Math.sin(a) * len);
        g.stroke();
      }
      g.globalAlpha = 1;
    } else if (variant === 2) {
      // Nodding oat-like panicle. Pendulous spikelets with long awns hang off short
      // pedicels; the awns are what still read at 20 m once the husk is a pixel.
      const heads = 6 + rng.int(0, 3);
      for (let i = 0; i < heads; i++) {
        const ny = lerp(headBase, headTop + 8, (i + rng.range(0.1, 0.9)) / heads);
        const t = clamp01((by - ny) / Math.max(1, by - tipY));
        const nx = at(t)[0];
        const dir = rng.sign();
        const room2 = Math.max(6, room - Math.abs(nx - bx) - 14);
        const px = nx + dir * room2 * rng.range(0.35, 1.0);
        const py = ny + Math.min(R.h * 0.02, (headBase - ny) * 0.3 + 2);
        ribbon(g, nx, ny, nx + dir * room2 * 0.4, ny - 3, px, py, wStem(1.4, 0.8), PAL.dryDeep, 6);
        const len = R.h * rng.range(0.028, 0.048);
        ribbon(g, px, py, px + dir * 2.5, py + len * 0.5, px + dir * rng.range(-1, 3), py + len,
          wLeaf(rng.range(6.0, 8.5), 0.6, 0.7), rng() < 0.4 ? PAL.huskPale : PAL.husk, 10);
        for (let a = 0; a < 2; a++) {
          const ax2 = px + rng.range(-2, 2);
          ribbon(g, ax2, py + len * 0.25,
            ax2 + dir * 5, py - len * 0.2,
            ax2 + dir * Math.min(12, room2 * 0.5), py - len * rng.range(0.4, 0.9),
            wStem(1.0, 0.35), PAL.seed, 8);
        }
      }
    } else {
      // Soft bloomed head - Yorkshire fog. A woolly ovoid with the faintest warm
      // cast, which is the one place in the meadow allowed to answer the blossom.
      const ax = tipX, ay = headBase;
      const tx2 = tipX + rng.range(-3, 3), ty2 = headTop;
      const n = 420;
      const maxRad = Math.min(R.w * 0.20, room - Math.abs(tipX - bx) - 8);
      for (let i = 0; i < n; i++) {
        const t = Math.pow(rng(), 0.85);
        const cx = lerp(ax, tx2, t);
        const cy = lerp(ay, ty2, t);
        const rad = maxRad * Math.pow(Math.sin(Math.PI * clamp01(t * 0.9 + 0.08)), 0.55);
        const off = clamp(rng.gaussian(0, rad * 0.5), -rad, rad);
        const a = rng.range(-2.6, -0.5);
        const len = rng.range(3.0, 6.5);
        const shade = rng();
        g.strokeStyle = shade < 0.22 ? '#c9b7b4' : shade < 0.62 ? PAL.huskPale : '#e0d8c6';
        g.globalAlpha = rng.range(0.30, 0.78);
        g.lineWidth = rng.range(1.0, 1.9);
        g.beginPath();
        g.moveTo(cx + off * 0.35, cy);
        g.lineTo(cx + off + Math.cos(a) * len, cy + Math.sin(a) * len);
        g.stroke();
      }
      g.globalAlpha = 1;
    }
  }

  /**
   * Pampas-style reed stand: a fan of long tapered leaves plus one or two feathered
   * plumes. Variant 1 is the drier, more open one so a wet hollow is never uniform.
   */
  _drawReedTile(g, rng, R, variant) {
    const bx = R.x + R.w * 0.5;
    const by = R.y + R.h;
    const dry = variant === 1;

    // Leaves fan out from a common crown. The tall ones stay near vertical - a
    // 2 m reed leaning 45 degrees would run straight out of its own tile, and the
    // clip would shear it off mid-blade.
    const blades = dry ? 9 + rng.int(0, 3) : 12 + rng.int(0, 4);
    const tops = [];
    for (let i = 0; i < blades; i++) {
      const h = R.h * rng.range(dry ? 0.42 : 0.50, dry ? 0.92 : 0.99);
      const dir = rng.sign();
      // Lean is inversely proportional to height, so every tip lands inside 0.40 R.w.
      const spread = rng.range(0.04, (dry ? 0.44 : 0.34)) * lerp(1.0, 0.35, h / R.h);
      const tipX = bx + dir * spread * R.w;
      const tipY = by - h;
      const cx = bx + dir * spread * R.w * 0.22;
      const cy = by - h * rng.range(0.55, 0.80);
      const shade = rng();
      const col = dry
        ? (shade < 0.34 ? PAL.dry : shade < 0.7 ? PAL.leafGrey : PAL.dryDeep)
        : (shade < 0.32 ? PAL.leafDeep : shade < 0.72 ? PAL.leaf : PAL.leafPale);
      ribbon(g, bx + rng.range(-0.04, 0.04) * R.w, by, cx, cy, tipX, tipY,
        wBlade(rng.range(5.0, 9.0), 0.80), col, 14);
      // Only culms that top out with room above them are candidates to carry a
      // plume; the plume grows DOWNWARD from its anchor so it needs the height.
      if (h > R.h * 0.55 && h < R.h * 0.80) tops.push(tipX, tipY);
    }

    // --- plumes on the tallest culms
    const plumes = dry ? 1 : 2;
    const loY = R.y + 2, hiY = R.y + R.h - 2;
    const loX = R.x + 2, hiX = R.x + R.w - 2;
    for (let p = 0; p < plumes && tops.length; p++) {
      const idx = Math.floor(rng() * (tops.length / 2)) * 2;
      const px = clamp(tops[idx] + rng.range(-4, 4), bx - R.w * 0.24, bx + R.w * 0.24);
      const py = clamp(tops[idx + 1] + R.h * rng.range(0.0, 0.05), R.y + R.h * 0.10, hiY);
      const ph = R.h * rng.range(0.13, 0.21);
      const hairs = 210;
      const spreadMax = R.w * 0.105;
      for (let k = 0; k < hairs; k++) {
        const t = Math.pow(rng(), 0.8);
        const spread = Math.sin(Math.PI * clamp01(t * 0.88 + 0.1)) * spreadMax + 2.5;
        const hx = px + clamp(rng.gaussian(0, spread * 0.45), -spread, spread);
        const hy = py + ph * t;
        const a = rng.range(-1.5, 1.5) + (hx > px ? 0.55 : -0.55);
        const len = rng.range(5, 14);
        g.globalAlpha = rng.range(0.28, 0.82);
        g.lineWidth = rng.range(1.0, 1.8);
        const shade = rng();
        g.strokeStyle = shade < 0.42 ? PAL.huskPale : shade < 0.8 ? PAL.husk : PAL.seed;
        g.beginPath();
        g.moveTo(clamp(hx, loX, hiX), clamp(hy, loY, hiY));
        g.lineTo(clamp(hx + Math.cos(a) * len * 0.6, loX, hiX),
          clamp(hy + Math.sin(a) * len, loY, hiY));
        g.stroke();
      }
      g.globalAlpha = 1;
    }
  }

  /**
   * Low ground cover, the layer that decides whether the soil between blades reads
   * as a meadow floor or as a texture. Four different plants, drawn small: a clover
   * trefoil, a leaf rosette, a trailing vetch and a mixed shoot clump.
   */
  _drawCoverTile(g, rng, R, variant) {
    const bx = R.x + R.w * 0.5;
    const by = R.y + R.h;

    if (variant === 0) {
      // White clover. Three heart-shaped leaflets, each carrying the pale crescent.
      const stalks = 3;
      for (let s = 0; s < stalks; s++) {
        const px = bx + rng.range(-0.14, 0.14) * R.w;
        const ph = R.h * rng.range(0.34, 0.58);
        const ty = by - ph;
        ribbon(g, bx, by, bx + (px - bx) * 0.3, by - ph * 0.6, px, ty, wStem(2.6, 1.8), PAL.stem, 8);
        const rad = R.h * rng.range(0.11, 0.15);
        for (let k = 0; k < 3; k++) {
          const a = -Math.PI * 0.5 + k * 2.0944 + rng.range(-0.25, 0.25);
          const cx = px + Math.cos(a) * rad * 0.85;
          const cy = ty + Math.sin(a) * rad * 0.85;
          const dx = Math.cos(a), dy = Math.sin(a);
          const nx = -dy, ny = dx;
          g.fillStyle = rng() < 0.4 ? PAL.leafDeep : PAL.leaf;
          g.beginPath();
          g.moveTo(px, ty);
          g.arc(cx + nx * rad * 0.42, cy + ny * rad * 0.42, rad * 0.52, 0, TAU);
          g.fill();
          g.beginPath();
          g.moveTo(px, ty);
          g.arc(cx - nx * rad * 0.42, cy - ny * rad * 0.42, rad * 0.52, 0, TAU);
          g.fill();
          g.beginPath();
          g.moveTo(px, ty);
          g.lineTo(cx + nx * rad * 0.7, cy + ny * rad * 0.7);
          g.lineTo(cx - nx * rad * 0.7, cy - ny * rad * 0.7);
          g.closePath();
          g.fill();
          // The pale chevron. It is the single detail that says "clover".
          g.save();
          g.globalAlpha = 0.55;
          g.strokeStyle = PAL.leafPale;
          g.lineWidth = rad * 0.20;
          g.beginPath();
          g.arc(cx, cy, rad * 0.46, a + 2.1, a + 4.2);
          g.stroke();
          g.restore();
        }
      }
    } else if (variant === 1) {
      // Basal rosette - ribwort / plantain. Ovate leaves radiating almost flat.
      const n = 5 + rng.int(0, 2);
      for (let i = 0; i < n; i++) {
        const a = -Math.PI * 0.5 + (i / n - 0.5) * 1.9 + rng.range(-0.12, 0.12);
        const len = R.h * rng.range(0.36, 0.52);
        const tx = bx + Math.cos(a) * len;
        const ty = by + Math.sin(a) * len;
        ribbon(g, bx, by, bx + Math.cos(a) * len * 0.42, by + Math.sin(a) * len * 0.62,
          tx, ty, wLeaf(rng.range(9, 15), 0.72, 0.9),
          rng() < 0.35 ? PAL.leafDeep : PAL.leaf, 12);
        g.save();
        g.globalAlpha = 0.4;
        ribbon(g, bx, by, bx + Math.cos(a) * len * 0.42, by + Math.sin(a) * len * 0.62,
          tx, ty, wStem(1.6, 0.5), PAL.leafPale, 10);
        g.restore();
      }
    } else if (variant === 2) {
      // Vetch: a trailing stem with paired leaflets and a tendril.
      const dir = rng.sign();
      const tx = bx + dir * R.w * rng.range(0.16, 0.26);
      const ty = by - R.h * rng.range(0.62, 0.88);
      ribbon(g, bx, by, bx + dir * R.w * 0.08, by - R.h * 0.5, tx, ty,
        wStem(2.2, 1.2), PAL.stem, 12);
      const pairs = 4;
      for (let i = 0; i < pairs; i++) {
        const t = 0.20 + 0.68 * (i / (pairs - 1));
        const s = 1 - t;
        const nx = s * s * bx + 2 * s * t * (bx + dir * R.w * 0.08) + t * t * tx;
        const ny = s * s * by + 2 * s * t * (by - R.h * 0.5) + t * t * ty;
        const lr = R.h * rng.range(0.09, 0.14) * (1 - t * 0.35);
        for (let sgn = -1; sgn <= 1; sgn += 2) {
          ribbon(g, nx, ny, nx + sgn * lr * 0.55, ny - lr * 0.30,
            nx + sgn * lr, ny - lr * rng.range(0.2, 0.55),
            wLeaf(rng.range(6, 9), 0.66, 0.9), rng() < 0.4 ? PAL.leafDeep : PAL.leaf, 8);
        }
      }
      ribbon(g, tx, ty, tx + dir * 8, ty - 8, tx + dir * rng.range(2, 8), ty - R.h * 0.12,
        wStem(1.2, 0.5), PAL.stemPale, 8);
    } else {
      // Mixed clump: short shoots plus a couple of round basal leaves. This is the
      // filler that keeps the cover layer from reading as three repeated stickers.
      const n = 7 + rng.int(0, 5);
      for (let i = 0; i < n; i++) {
        const dir = rng.sign();
        const h = R.h * rng.range(0.30, 0.86);
        const tx = bx + dir * rng.range(0.05, 0.32) * R.w;
        ribbon(g, bx + rng.range(-0.10, 0.10) * R.w, by,
          bx + dir * R.w * 0.06, by - h * 0.62, tx, by - h,
          wBlade(rng.range(2.6, 4.2), 0.8),
          rng() < 0.3 ? PAL.leafGrey : rng() < 0.6 ? PAL.leaf : PAL.leafDeep, 9);
      }
      for (let i = 0; i < 2; i++) {
        const dir = i ? 1 : -1;
        const len = R.h * rng.range(0.22, 0.34);
        ribbon(g, bx, by, bx + dir * len * 0.5, by - len * 0.30,
          bx + dir * len, by - len * rng.range(0.1, 0.4),
          wLeaf(rng.range(10, 14), 0.7, 0.9), PAL.leafDeep, 10);
      }
    }
  }

  /**
   * Fine bent-grass. A fan of hair-thin tapered blades from a common crown; the
   * long ones flop right over, which is what separates soft meadow grass from the
   * stiff sward the grass system already draws.
   */
  _drawBentTile(g, rng, R, variant) {
    const bx = R.x + R.w * 0.5;
    const by = R.y + R.h;

    const cfg = [
      { n: 74, lean: 0.85, hi: 0.98, cols: [PAL.leaf, PAL.leafPale, PAL.leafDeep], seeds: 0.10 },
      { n: 62, lean: 1.30, hi: 0.92, cols: [PAL.leafGrey, PAL.leaf, PAL.dry], seeds: 0.34 },
      { n: 86, lean: 0.62, hi: 0.72, cols: ['#5f7050', PAL.leafDeep, PAL.leaf], seeds: 0.05 },
      { n: 58, lean: 1.15, hi: 0.88, cols: [PAL.dry, PAL.dryPale, PAL.leafGrey], seeds: 0.42 },
    ][variant];

    // Half-width the fan may occupy. Everything below is derived from it rather
    // than from R.h, because a blade's LENGTH and its LEAN trade off: a culm that
    // has flopped 80 degrees over cannot also still be reaching for the top of the
    // tile. Sampling the angle first and then capping the length by how far that
    // angle is allowed to travel gives a fan that fills the tile and never leaves it.
    const room = R.w * 0.44;
    for (let i = 0; i < cfg.n; i++) {
      const theta = clamp(rng.gaussian(0, cfg.lean * 0.42), -1.45, 1.45);
      const sn = Math.abs(Math.sin(theta));
      const sx = bx + clamp(rng.gaussian(0, R.w * 0.040), -R.w * 0.10, R.w * 0.10);
      const lMax = Math.min(R.h * cfg.hi, (room - Math.abs(sx - bx)) / Math.max(0.16, sn));
      const L = lMax * rng.range(0.30, 1.0);
      // Chord shortening: an arched blade of arc length L spans less than L.
      const chord = 0.88;
      const tipX = sx + Math.sin(theta) * L * chord;
      const tipY = by - Math.cos(theta) * L * chord;
      // The control point bulges the arc upward, so the blade rises before it falls.
      const cx = sx + Math.sin(theta) * L * 0.26;
      const cy = by - Math.cos(theta) * L * 0.74 - L * 0.10;
      const col = cfg.cols[rng.int(0, cfg.cols.length - 1)];
      g.globalAlpha = rng.range(0.72, 1.0);
      ribbon(g, sx, by, cx, cy, tipX, tipY, wBlade(rng.range(1.9, 3.4), 0.62), col, 12);
      g.globalAlpha = 1;
      if (rng() < cfg.seeds) {
        g.save();
        g.translate(tipX, tipY);
        g.rotate(theta + rng.range(-0.5, 0.5));
        g.fillStyle = rng() < 0.5 ? PAL.husk : PAL.seed;
        g.beginPath();
        g.ellipse(0, 0, 1.4, rng.range(3.0, 5.2), 0, 0, TAU);
        g.fill();
        g.restore();
      }
    }
  }

  /**
   * Wildflower sprig. One muted hue per tile - chalk white, pale yellow, faded
   * lilac. Colonies of one colour is how wildflowers actually grow, and it is also
   * the only way a flower stays quiet: a mixed confetti of hues would fight the
   * blossom, which is the one thing in this scene allowed to be pink.
   */
  _drawBloomTile(g, rng, R, variant) {
    const bx = R.x + R.w * 0.5;
    const by = R.y + R.h;

    // --- foliage first, so the flowers sit in front of it
    const leaves = 4 + rng.int(0, 3);
    for (let i = 0; i < leaves; i++) {
      const dir = rng.sign();
      const h = R.h * rng.range(0.22, 0.52);
      ribbon(g, bx + rng.range(-0.08, 0.08) * R.w, by,
        bx + dir * R.w * 0.10, by - h * 0.62,
        bx + dir * R.w * rng.range(0.10, 0.34), by - h,
        wBlade(rng.range(3.2, 5.6), 0.8),
        rng() < 0.4 ? PAL.leafDeep : PAL.leaf, 10);
    }

    const heads = variant === 2 ? 3 + rng.int(0, 2) : 4 + rng.int(0, 3);
    for (let i = 0; i < heads; i++) {
      const dir = rng.sign();
      // Head height is capped so the outermost floret ring still clears the tile
      // top; the same reason the horizontal offset stops well short of the edge.
      const h = R.h * rng.range(0.44, 0.84);
      const hx = bx + dir * R.w * rng.range(0.02, 0.28);
      const hy = by - h;
      ribbon(g, bx + rng.range(-0.06, 0.06) * R.w, by,
        bx + dir * R.w * 0.07, by - h * 0.55, hx, hy,
        wStem(2.6, 1.5), rng() < 0.4 ? PAL.stemDark : PAL.stem, 12);

      if (variant === 0) {
        // Chalk-white composite. Narrow ray florets round an ochre disc.
        const rad = R.w * rng.range(0.055, 0.085);
        const petals = 10 + rng.int(0, 4);
        const a0 = rng() * TAU;
        for (let p = 0; p < petals; p++) {
          const a = a0 + (p / petals) * TAU + rng.range(-0.08, 0.08);
          const rr = rad * rng.range(0.82, 1.12);
          ribbon(g, hx, hy, hx + Math.cos(a) * rr * 0.45, hy + Math.sin(a) * rr * 0.45,
            hx + Math.cos(a) * rr, hy + Math.sin(a) * rr,
            wLeaf(rad * 0.36, 0.6, 0.75), rng() < 0.35 ? PAL.whiteHi : PAL.white, 6);
        }
        g.fillStyle = PAL.pollen;
        g.beginPath();
        g.arc(hx, hy, rad * 0.30, 0, TAU);
        g.fill();
      } else if (variant === 1) {
        // Pale yellow cup - five broad rounded petals, faintly overlapping.
        const rad = R.w * rng.range(0.050, 0.078);
        const a0 = rng() * TAU;
        for (let p = 0; p < 5; p++) {
          const a = a0 + (p / 5) * TAU;
          g.save();
          g.translate(hx + Math.cos(a) * rad * 0.52, hy + Math.sin(a) * rad * 0.52);
          g.rotate(a);
          g.fillStyle = p % 2 ? PAL.yellow : PAL.yellowDeep;
          g.beginPath();
          g.ellipse(0, 0, rad * 0.58, rad * 0.44, 0, 0, TAU);
          g.fill();
          g.restore();
        }
        g.fillStyle = '#a8924c';
        g.beginPath();
        g.arc(hx, hy, rad * 0.24, 0, TAU);
        g.fill();
      } else {
        // Faded lilac pincushion - scabious. Dozens of tiny florets, denser at the
        // rim, so the head has a soft edge instead of a drawn circle.
        const rad = R.w * rng.range(0.058, 0.086);
        const n = 34 + rng.int(0, 16);
        for (let p = 0; p < n; p++) {
          const a = rng() * TAU;
          const rr = rad * Math.pow(rng(), 0.42);
          const fr = rng.range(1.6, 3.2) * (1 - rr / rad * 0.35);
          g.fillStyle = rng() < 0.34 ? PAL.lilacHi : rng() < 0.75 ? PAL.lilac : PAL.lilacDeep;
          g.beginPath();
          g.arc(hx + Math.cos(a) * rr, hy + Math.sin(a) * rr * 0.86, fr, 0, TAU);
          g.fill();
        }
        // A ring of longer outer florets pushes the silhouette out into a star.
        const outer = 7 + rng.int(0, 4);
        for (let p = 0; p < outer; p++) {
          const a = (p / outer) * TAU + rng.range(-0.2, 0.2);
          ribbon(g, hx + Math.cos(a) * rad * 0.6, hy + Math.sin(a) * rad * 0.6,
            hx + Math.cos(a) * rad * 1.0, hy + Math.sin(a) * rad * 0.95,
            hx + Math.cos(a) * rad * 1.34, hy + Math.sin(a) * rad * 1.22,
            wLeaf(rad * 0.30, 0.6, 0.8), PAL.lilacHi, 5);
        }
      }
    }

    // A couple of unopened buds. Every real stand has some.
    for (let i = 0; i < 2; i++) {
      const dir = rng.sign();
      const h = R.h * rng.range(0.34, 0.62);
      const hx = bx + dir * R.w * rng.range(0.10, 0.30);
      const hy = by - h;
      ribbon(g, bx, by, bx + dir * R.w * 0.08, by - h * 0.55, hx, hy,
        wStem(2.2, 1.3), PAL.stemDark, 8);
      g.fillStyle = variant === 2 ? PAL.lilacDeep : variant === 1 ? PAL.yellowDeep : PAL.leafGrey;
      g.beginPath();
      g.ellipse(hx, hy, R.w * 0.020, R.w * 0.030, rng.range(-0.4, 0.4), 0, TAU);
      g.fill();
    }
  }

  /**
   * A single broad leaf, rising from the bottom of the tile so the card's base sits
   * at the plant's crown. Dock and ribwort: the two wide silhouettes in an English
   * meadow, and the reason the field stops reading as one repeated blade.
   */
  _drawBroadTile(g, rng, R, variant) {
    const bx = R.x + R.w * 0.5;
    const by = R.y + R.h;
    const dir = rng.sign();
    // The blade is nearly as wide as the tile, so the spine has almost no room to
    // wander: 0.5*width + spine offset + ripple must stay under half the tile.
    const tipX = bx + dir * R.w * rng.range(0.03, 0.08);
    const tipY = R.y + R.h * rng.range(0.03, 0.10);
    const cX = bx + dir * R.w * rng.range(0.04, 0.10);
    const cY = by - R.h * rng.range(0.48, 0.62);

    // The margin. Docks are wavy-edged; the ripple is small but it is most of what
    // stops a leaf reading as a vector shape.
    const phase = rng() * TAU;
    const ripple = variant === 0 ? 0.070 : 0.030;
    const wide = R.w * (variant === 0 ? rng.range(0.60, 0.70) : rng.range(0.34, 0.44));
    const shape = variant === 0
      ? wLeaf(wide, 0.60, 0.72)
      : wLeaf(wide, 0.74, 0.95);
    const wFn = (t) => shape(t) * (1 + ripple * Math.sin(t * 15.0 + phase))
      // a nibbled notch: asymmetry is free realism and reads even at 10 m
      * (1 - 0.30 * Math.exp(-Math.pow((t - 0.62) * 14, 2)));

    // Base fill, then tonal blotches clipped inside it. Canvas fills flat, so the
    // clip is the only way to get a leaf that is not one solid colour.
    ribbonPath(g, bx, by, cX, cY, tipX, tipY, wFn, 26);
    g.save();
    g.clip();
    g.fillStyle = variant === 0 ? PAL.leaf : PAL.leafGrey;
    g.fillRect(R.x, R.y, R.w, R.h);
    for (let i = 0; i < 8; i++) {
      g.globalAlpha = rng.range(0.10, 0.26);
      g.fillStyle = rng() < 0.5 ? PAL.leafDeep : PAL.leafPale;
      g.beginPath();
      g.ellipse(bx + rng.range(-0.5, 0.5) * R.w, by - rng() * R.h,
        R.w * rng.range(0.20, 0.55), R.h * rng.range(0.08, 0.22),
        rng() * TAU, 0, TAU);
      g.fill();
    }
    // Autumn rust creeping in from the tip and the margin.
    g.globalAlpha = rng.range(0.16, 0.34);
    g.fillStyle = PAL.rust;
    g.beginPath();
    g.ellipse(tipX, tipY + R.h * 0.06, R.w * 0.55, R.h * 0.20, 0, 0, TAU);
    g.fill();
    g.globalAlpha = 1;
    g.restore();

    // Midrib and veins, drawn after the blotches so they stay crisp. Dock's veins
    // branch out to the margin; ribwort's are the parallel ribs it is named for,
    // and getting those two the same way round loses half the difference between
    // the two silhouettes.
    ribbon(g, bx, by, cX, cY, tipX, tipY, wStem(4.0, 1.0), PAL.leafPale, 22);
    g.save();
    g.globalAlpha = 0.42;
    if (variant === 0) {
      const veins = 7;
      for (let i = 0; i < veins; i++) {
        const t = 0.14 + 0.76 * (i / (veins - 1));
        const s = 1 - t;
        const mx = s * s * bx + 2 * s * t * cX + t * t * tipX;
        const my = s * s * by + 2 * s * t * cY + t * t * tipY;
        const hw = wFn(t) * 0.5;
        for (let sgn = -1; sgn <= 1; sgn += 2) {
          ribbon(g, mx, my, mx + sgn * hw * 0.6, my - hw * 0.35,
            mx + sgn * hw * 0.92, my - hw * rng.range(0.5, 0.9),
            wStem(1.6, 0.4), PAL.leafPale, 6);
        }
      }
    } else {
      // Two ribs each side, tracking the blade's own taper so they converge on the
      // tip exactly the way the outline does.
      for (let i = 1; i <= 2; i++) {
        const f = i / 2.6;
        for (let sgn = -1; sgn <= 1; sgn += 2) {
          ribbon(g,
            bx + sgn * wFn(0.05) * 0.5 * f, by - R.h * 0.02,
            cX + sgn * wFn(0.5) * 0.5 * f, cY,
            tipX + sgn * wFn(0.92) * 0.5 * f, tipY + R.h * 0.02,
            wStem(1.5, 0.5), PAL.leafPale, 16);
        }
      }
    }
    g.restore();

    // Petiole: the leaf has to attach to something or the card floats.
    ribbon(g, bx, by, bx, by - R.h * 0.03, bx + dir * 2, by - R.h * 0.07,
      wStem(5.5, 4.0), PAL.stem, 4);
  }

  _makePetalTexture(aniso) {
    const S = 128;
    const g = makeCanvas(S, S);
    const grad = g.createRadialGradient(S * 0.5, S * 0.62, 2, S * 0.5, S * 0.62, S * 0.52);
    grad.addColorStop(0, '#f7ebec');
    grad.addColorStop(0.55, '#f0dade');
    grad.addColorStop(1, '#e2c5cb');
    g.fillStyle = grad;
    g.beginPath();
    g.moveTo(64, 118);
    g.bezierCurveTo(26, 104, 14, 62, 40, 26);
    g.bezierCurveTo(50, 16, 58, 20, 64, 34);   // the notch that says "sakura"
    g.bezierCurveTo(70, 20, 78, 16, 88, 26);
    g.bezierCurveTo(114, 62, 102, 104, 64, 118);
    g.closePath();
    g.fill();
    g.strokeStyle = 'rgba(198,164,171,0.5)';
    g.lineWidth = 1.1;
    for (let i = -2; i <= 2; i++) {
      g.beginPath();
      g.moveTo(64, 114);
      g.quadraticCurveTo(64 + i * 13, 70, 64 + i * 20, 34 + Math.abs(i) * 7);
      g.stroke();
    }
    return textureFromImageData(dilateAlpha(g, S, S, 3), aniso);
  }

  // -------------------------------------------------------------------------
  // Procedural textures: treeline atlas
  // -------------------------------------------------------------------------

  _makeTreelineAtlas(aniso) {
    const W = 1024, H = 512, TW = 512, TH = 256;
    const g = makeCanvas(W, H);
    for (let i = 0; i < 4; i++) {
      this._drawTreelineTile(g, makeRNG(4400 + i * 977), (i % 2) * TW, Math.floor(i / 2) * TH, TW, TH);
    }
    return textureFromImageData(dilateAlpha(g, W, H, 2), aniso);
  }

  _drawTreelineTile(g, rng, ox, oy, TW, TH) {
    const gutter = 6;
    const left = ox + gutter, right = ox + TW - gutter, w = right - left;
    const ground = oy + TH * 0.88;
    const usable = ground - oy - gutter;

    // Three depth layers. Farther layers are lighter (haze) and shorter - within a
    // single card that is what turns a flat silhouette into a forest with depth.
    const layers = [
      { n: 22, hMin: 0.24, hMax: 0.40, v: 168 },
      { n: 15, hMin: 0.38, hMax: 0.64, v: 126 },
      { n: 10, hMin: 0.55, hMax: 0.95, v: 88 },
    ];
    for (const L of layers) {
      for (let i = 0; i < L.n; i++) {
        const x = left + ((i + rng.range(0.15, 0.85)) / L.n) * w;
        // Trees thin toward the card edges, so the seam between overlapping cards
        // never lands on a hard vertical cut through solid canopy.
        const edge = smoothstep(0, w * 0.16, Math.min(x - left, right - x));
        if (rng() > 0.35 + 0.65 * edge) continue;
        const h = usable * rng.range(L.hMin, L.hMax) * lerp(0.55, 1.0, edge);
        const v = Math.round(clamp(L.v + rng.gaussian(0, 16), 60, 230));
        const tw = h * rng.range(0.42, 0.72);
        const base = ground + rng.range(-2, 3);
        if (rng() < 0.62) this._drawConifer(g, rng, x, base, h, tw * 0.8, v);
        else this._drawBroadleaf(g, rng, x, base, h, tw, v);
      }
    }

    // Understory: a solid strip along the card foot whose top edge is uneven and
    // tapers at the card edges, so neighbouring cards merge into one ragged line
    // rather than a row of identical rectangles.
    g.fillStyle = 'rgb(74,74,74)';
    g.beginPath();
    g.moveTo(left, oy + TH);
    const steps = 42;
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const x = left + t * w;
      const taper = smoothstep(0, 0.15, t) * smoothstep(0, 0.15, 1 - t);
      const bump = (0.5 + 0.5 * Math.sin(t * 17.0)) * TH * 0.035
        + (0.5 + 0.5 * Math.sin(t * 41.3 + 2.1)) * TH * 0.018;
      g.lineTo(x, ground + TH * 0.02 - (bump + TH * 0.03) * taper);
    }
    g.lineTo(right, oy + TH);
    g.closePath();
    g.fill();

    // Leaf-scale speckle along the canopy edges. Vector silhouettes read as vector
    // silhouettes; this is what breaks that. One ImageData read, not one per dot.
    const img = g.getImageData(ox, oy, TW, TH);
    const src = img.data;
    for (let i = 0; i < 1100; i++) {
      const px = Math.round(rng.range(gutter, TW - gutter));
      const py = Math.round(rng.range(gutter, gutter + usable));
      const a = src[(py * TW + px) * 4 + 3];
      if (a < 40 || a > 225) continue;      // only the fringe, not the interior
      const v = Math.round(clamp(90 + rng.gaussian(0, 40), 50, 220));
      g.fillStyle = `rgba(${v},${v},${v},${rng.range(0.4, 0.95).toFixed(2)})`;
      g.beginPath();
      g.arc(ox + px, oy + py, rng.range(0.8, 2.2), 0, TAU);
      g.fill();
    }
  }

  _drawConifer(g, rng, x, base, h, w, v) {
    g.fillStyle = `rgb(${v},${v},${v})`;
    g.fillRect(x - w * 0.035, base - h * 0.30, w * 0.07, h * 0.30);
    const tiers = 6 + Math.floor(rng() * 4);
    for (let t = 0; t < tiers; t++) {
      const f = t / (tiers - 1);
      const ty = base - h * (0.10 + 0.90 * f);
      const tw = w * 0.5 * (1 - f) * rng.range(0.85, 1.12) + w * 0.03;
      const th = h * 0.11;
      g.beginPath();
      g.moveTo(x, ty - th);
      g.lineTo(x - tw, ty + th * 0.42);
      g.lineTo(x - tw * 0.34, ty + th * 0.24);
      g.lineTo(x, ty + th * 0.55);
      g.lineTo(x + tw * 0.34, ty + th * 0.24);
      g.lineTo(x + tw, ty + th * 0.42);
      g.closePath();
      g.fill();
    }
  }

  _drawBroadleaf(g, rng, x, base, h, w, v) {
    g.fillStyle = `rgb(${v},${v},${v})`;
    g.fillRect(x - w * 0.04, base - h * 0.50, w * 0.08, h * 0.50);
    const blobs = 6 + Math.floor(rng() * 5);
    for (let i = 0; i < blobs; i++) {
      const a = rng() * TAU, r = Math.sqrt(rng());
      g.beginPath();
      g.arc(x + Math.cos(a) * r * w * 0.48, base - h * 0.70 + Math.sin(a) * r * h * 0.24,
        w * (0.24 + 0.24 * rng()), 0, TAU);
      g.fill();
    }
  }

  // -------------------------------------------------------------------------
  // Horizon
  // -------------------------------------------------------------------------

  _buildHorizon(atlas) {
    this.hu = {
      uAtlas: { value: atlas },
      uCanopy: { value: new THREE.Color(0.06, 0.075, 0.06) },
      uCanopyWarm: { value: new THREE.Color(0.075, 0.070, 0.050) },
      uFogColor: { value: new THREE.Color(0.72, 0.78, 0.86) },
      uHazeColor: { value: new THREE.Color(0.80, 0.84, 0.90) },
      uSunColor: { value: new THREE.Color(1, 1, 1) },
      uSunDir: { value: new THREE.Vector3(1, 0, 0) },
      uSunVis: { value: 1 },
      uFogDensity: { value: 0.0022 },
      uAerial: { value: 0.55 },
      uMaxExt: { value: 0.93 },
      uGroundHaze: { value: 0.45 },
      uAlphaTest: { value: 0.38 },
    };

    const mat = new THREE.ShaderMaterial({
      uniforms: this.hu,
      side: THREE.DoubleSide,
      transparent: false,
      depthWrite: true,
      fog: false,
      vertexShader: /* glsl */`
        attribute vec4 aData;   // x: radius, y: value, z: height 0..1, w: hue mix
        uniform vec3 uSunDir;
        varying vec2 vUvA;
        varying float vDist;
        varying float vVal;
        varying float vH;
        varying float vHue;
        varying float vSun;
        void main() {
          vUvA = uv;
          vDist = aData.x;
          vVal = aData.y;
          vH = aData.z;
          vHue = aData.w;
          vec3 radial = vec3( position.x, 0.0, position.z );
          vSun = clamp( dot( radial / max( length( radial ), 1e-3 ), uSunDir ), 0.0, 1.0 );
          gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4( position, 1.0 );
        }`,
      fragmentShader: /* glsl */`
        uniform sampler2D uAtlas;
        uniform vec3 uCanopy;
        uniform vec3 uCanopyWarm;
        uniform vec3 uFogColor;
        uniform vec3 uHazeColor;
        uniform vec3 uSunColor;
        uniform float uSunVis;
        uniform float uFogDensity;
        uniform float uAerial;
        uniform float uMaxExt;
        uniform float uGroundHaze;
        uniform float uAlphaTest;
        varying vec2 vUvA;
        varying float vDist;
        varying float vVal;
        varying float vH;
        varying float vHue;
        varying float vSun;
        void main() {
          vec4 t = texture2D( uAtlas, vUvA );
          if ( t.a < uAlphaTest ) discard;

          // The atlas stores relative luminance, not colour - the tint is lit here,
          // so the treeline tracks time of day and weather for free.
          vec3 base = mix( uCanopy, uCanopyWarm, vHue ) * ( 0.34 + 1.25 * t.r ) * vVal;
          base *= mix( 0.58, 1.0, vH );                     // the forest floor is darker
          base += uSunColor * ( uSunVis * 0.13 * vSun * smoothstep( 0.42, 1.0, vH ) * t.r );

          // Aerial perspective: the same exponential-squared falloff the scene fog
          // uses, scaled down because a real treeline stays faintly readable long
          // after a raw FogExp2 term would have erased it.
          float od = vDist * uFogDensity * uAerial;
          float ext = 1.0 - exp( -od * od );
          ext = mix( ext, 1.0, uGroundHaze * ( 1.0 - smoothstep( 0.0, 0.5, vH ) ) );
          ext = min( ext, uMaxExt );

          gl_FragColor = vec4( mix( base, mix( uFogColor, uHazeColor, vSun * uSunVis ), ext ), 1.0 );
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    });
    this.matHorizon = mat;
    this._disposables.push(mat);

    this.treeNear = this._makeTreeRing(makeRNG(31337), mat, -7, [
      { r: 520, w: 108, h: 22 },
      { r: 645, w: 130, h: 25 },
      { r: 770, w: 150, h: 27 },
    ]);
    this.treeFar = this._makeTreeRing(makeRNG(80021), mat, -9, [
      { r: 980, w: 190, h: 27 },
      { r: 1250, w: 235, h: 30 },
      { r: 1560, w: 285, h: 33 },
    ]);

    // Haze band: guarantees there is never raw sky under the treeline if the terrain
    // clipmap stops short of it. Coloured as fully extincted fog, so when it is not
    // needed it is invisible rather than wrong.
    const bandGeo = new THREE.CylinderGeometry(2100, 2100, 150, 40, 1, true);
    bandGeo.translate(0, -70, 0);            // spans y = +5 .. -145
    const bp = bandGeo.attributes.position;
    const bc = new Float32Array(bp.count * 3);
    for (let i = 0; i < bp.count; i++) {
      const t = clamp01((bp.getY(i) + 145) / 150);
      const v = lerp(0.80, 1.0, t * t);
      bc[i * 3] = v; bc[i * 3 + 1] = v; bc[i * 3 + 2] = v;
    }
    bandGeo.setAttribute('color', new THREE.BufferAttribute(bc, 3));
    const bandMat = new THREE.MeshBasicMaterial({
      color: 0xffffff, vertexColors: true, side: THREE.BackSide, fog: false, depthWrite: true,
    });
    this.band = new THREE.Mesh(bandGeo, bandMat);
    this.band.frustumCulled = false;
    this.bandMat = bandMat;
    this._disposables.push(bandGeo, bandMat);

    this.horizon.add(this.band, this.treeFar, this.treeNear);
  }

  /**
   * One merged, pre-oriented card ring. The rings are re-centred on the camera every
   * frame, so every card already faces the viewer - no billboard maths at all, and
   * the whole horizon costs two draw calls of ~300 triangles each.
   */
  _makeTreeRing(rng, mat, baseY, layers) {
    const P = [], U = [], D = [], I = [];
    const shape = createNoise((rng() * 1e6) | 0);
    const layerEnd = [];

    for (let li = 0; li < layers.length; li++) {
      const L = layers[li];
      const count = Math.max(8, Math.ceil((TAU * L.r) / (L.w * 0.62)));
      for (let i = 0; i < count; i++) {
        const theta = ((i + rng.range(-0.34, 0.34)) / count) * TAU;
        const ca = Math.cos(theta), sa = Math.sin(theta);

        // Landform modulation: bays, headlands, ridges, the occasional clearing.
        // Without it this is a ring of trees; with it, it is a coastline of forest.
        // `lf` is shared by every layer so they bulge together as one landmass;
        // `lf2` is decorrelated per layer so a clearing in front reveals trees
        // behind it instead of punching a corridor straight through to the sky.
        const lf = shape.fbm2D(ca * 1.6, sa * 1.6, 3, 2.1, 0.55);
        const lf2 = shape.fbm2D(ca * 4.3 + 11 + li * 37.1, sa * 4.3 - 7 - li * 21.7, 2, 2.0, 0.5);
        if (lf2 < -0.42) continue;

        const r = L.r * (1 + lf * 0.19 + rng.range(-0.035, 0.035));
        const h = L.h * (0.72 + 0.55 * clamp01(lf * 0.9 + lf2 * 0.5 + 0.6)) * rng.range(0.88, 1.14);
        const hw = L.w * 0.5 * rng.range(0.85, 1.18);
        const by = baseY + lf * 4.5 + rng.range(-1.6, 1.6);

        const cx = ca * r, cz = sa * r;
        const tx = -sa, tz = ca;                       // ring tangent == card right

        const tile = Math.floor(rng() * 4);
        const u0 = (tile % 2) * 0.5 + 0.006;
        const v0 = Math.floor(tile / 2) * 0.5 + 0.004;
        const u1 = u0 + 0.488, v1 = v0 + 0.492;
        const flip = rng() < 0.5;
        const uL = flip ? u1 : u0, uR = flip ? u0 : u1;

        const val = rng.range(0.86, 1.16);
        const hue = clamp01(0.5 + shape.noise2D(ca * 2.7, sa * 2.7) * 0.6);

        const b = P.length / 3;
        P.push(cx - tx * hw, by, cz - tz * hw);
        P.push(cx + tx * hw, by, cz + tz * hw);
        P.push(cx - tx * hw, by + h, cz - tz * hw);
        P.push(cx + tx * hw, by + h, cz + tz * hw);
        U.push(uL, v0, uR, v0, uL, v1, uR, v1);
        D.push(r, val, 0, hue, r, val, 0, hue, r, val, 1, hue, r, val, 1, hue);
        I.push(b, b + 1, b + 2, b + 2, b + 1, b + 3);
      }
      layerEnd.push(I.length);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(P, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(U, 2));
    geo.setAttribute('aData', new THREE.Float32BufferAttribute(D, 4));
    geo.setIndex(I);
    geo.computeBoundingSphere();

    const mesh = new THREE.Mesh(geo, mat);
    mesh.frustumCulled = false;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    mesh.userData.layerEnd = layerEnd;
    this._disposables.push(geo);
    return mesh;
  }

  // -------------------------------------------------------------------------
  // Quality
  // -------------------------------------------------------------------------

  onQualityChange(quality) {
    this._applyQuality(quality);
    // Drop every chunk and re-stream: densities and radii both moved, and the
    // time-budgeted queue refills without a visible hitch.
    for (const t of this.queue) this._recycle(t);
    this.queue.length = 0;
    this.chunks.clear();
    this.sorted.length = 0;
    this._rs.group = null;
    this._rs.chunk = null;
    this._camCX = 1e9;
    this._firstFill = true;
    for (const g of this.groups) { g.mesh.count = 0; g.mesh.visible = false; }
  }

  _applyQuality(quality) {
    const q = (quality && typeof quality === 'object' && quality.tier) ? quality : this.state.quality;
    const tier = q?.tier ?? 'high';
    // Streaming distance derives from the grass distance so scatter and grass always
    // end at the same place - a rock floating past the end of the grass is the
    // fastest way to make an endless field look like a demo.
    const scale = clamp((q?.grassDistance ?? 140) / 140, 0.45, 1.4);
    const tierRadius = tier === 'low' ? 0.70 : tier === 'medium' ? 0.86 : tier === 'ultra' ? 1.12 : 1.0;
    const tierDensity = tier === 'low' ? 0.42 : tier === 'medium' ? 0.68 : tier === 'ultra' ? 1.25 : 1.0;
    // The vegetation dial the user actually moves is grassDensity; the plant layer
    // has to answer it or "thin grass" leaves a meadow of floating wildflowers.
    const veg = clamp(q?.grassDensity ?? 1, 0.30, 1.6);

    this._densityScale = tierDensity;
    this._plantScale = tierDensity * lerp(1, veg, 0.7);

    for (const g of this.groups) {
      g.radius = g.baseRadius * scale * tierRadius;
      g.radiusK = scale * tierRadius;
      const s = g.kind === 'plant' ? this._plantScale : this._densityScale;
      g.densityNow = clamp01(g.density * s * (tier === 'low' ? (g.lowScale ?? 1) : 1));
      g.mesh.castShadow = !!(g.wantsShadow && q?.shadows && tier !== 'low');
      g.mesh.receiveShadow = !!q?.shadows;
    }

    // LOW drops the outermost treeline layer. The horizon still reads, because the
    // landform modulation means the remaining layers are not a flat wall.
    const keep = tier === 'low' ? 2 : 3;
    for (const ring of [this.treeNear, this.treeFar]) {
      if (!ring) continue;
      const ends = ring.userData.layerEnd;
      ring.geometry.setDrawRange(0, ends[Math.min(keep, ends.length) - 1]);
    }
    if (this.hu) this.hu.uAerial.value = tier === 'low' ? 0.62 : 0.55;
  }

  // -------------------------------------------------------------------------
  // Frame
  // -------------------------------------------------------------------------

  update(dt, state) {
    if (!this.groups.length) return;

    const cam = this.camera.position;
    // A NaN camera would make every chunk key NaN (they all collide under
    // SameValueZero), stop eviction dead, and re-queue the whole world every
    // frame - so the streaming grid is driven off sanitised copies. The uniform
    // and horizon paths guard their own reads.
    const camX = finite(cam.x, 0), camZ = finite(cam.z, 0);
    const cx = Math.floor(camX / CHUNK);
    const cz = Math.floor(camZ / CHUNK);
    if (cx !== this._camCX || cz !== this._camCZ) {
      this._camCX = cx;
      this._camCZ = cz;
      this._refreshChunks(camX, camZ);
    }

    this._drainQueue();
    // Per-group, not global: a streaming burst finishes one chunk of one group per
    // frame, and re-packing all twelve would re-upload every instance buffer in the
    // module - around half a megabyte a frame - to move one chunk of one species.
    //
    // Bounded per frame as well, because `_refreshChunks` marks ALL twelve stale
    // on every chunk boundary crossed (the chunk order moved, so each group's
    // nearest-first packing has to be redone). Doing all twelve at once memcpys
    // and re-uploads every instance buffer in the module - ~2.6 MB - in the same
    // frame the generation budget has already been spent, which is the streaming
    // spike. Deferring is always SAFE: a group simply keeps the previous frame's
    // instances, and every one of those is still inside its own fade window
    // (eviction happens a full two chunks beyond the packing limit, and the
    // packing limit is itself beyond every instance's fade end).
    const groups = this.groups;
    let packBudget = this._firstFill ? groups.length : 3;
    for (let i = 0; i < groups.length && packBudget > 0; i++) {
      const g = groups[(this._packCursor + i) % groups.length];
      if (!g.dirty) continue;
      g.dirty = false;
      this._pack(g);
      packBudget--;
    }
    this._packCursor = (this._packCursor + 1) % groups.length;

    this._updateUniforms(state, cam);
    this._updateHorizon(dt, state, cam);
  }

  _task(chunk, group, dist) {
    const t = this._taskPool.pop();
    if (t) { t.chunk = chunk; t.group = group; t.dist = dist; return t; }
    return { chunk, group, dist };
  }

  _recycle(t) {
    t.chunk = null;
    t.group = null;
    if (this._taskPool.length < 1024) this._taskPool.push(t);
  }

  _refreshChunks(camX, camZ) {
    let maxRadius = 0;
    for (const g of this.groups) maxRadius = Math.max(maxRadius, g.radius);
    const evict = maxRadius + CHUNK * 2;
    const span = Math.ceil(evict / CHUNK);

    for (const t of this.queue) this._recycle(t);
    this.queue.length = 0;
    this.sorted.length = 0;
    // Abandon any half-built chunk: it may have just been evicted, and if it has
    // not, it is about to be re-queued below and would otherwise be built twice.
    this._rs.group = null;
    this._rs.chunk = null;

    // `.values()`, not the default entry iterator: destructuring `[key, ch]`
    // allocates a fresh two-element array per entry, and there are ~450 live
    // chunks at ULTRA. The key is carried on the chunk instead.
    for (const ch of this.chunks.values()) {
      ch.dist = Math.hypot(ch.wx - camX, ch.wz - camZ);
      if (ch.dist > evict) this.chunks.delete(ch.key);
    }

    for (let j = -span; j <= span; j++) {
      for (let i = -span; i <= span; i++) {
        const gx = this._camCX + i, gz = this._camCZ + j;
        const wx = (gx + 0.5) * CHUNK, wz = (gz + 0.5) * CHUNK;
        const dist = Math.hypot(wx - camX, wz - camZ);
        if (dist > evict) continue;
        const key = (gx + 0x8000) * 0x10000 + (gz + 0x8000);
        let ch = this.chunks.get(key);
        if (!ch) {
          ch = { key, cx: gx, cz: gz, wx, wz, dist, data: Object.create(null) };
          this.chunks.set(key, ch);
        } else {
          ch.dist = dist;
        }
        this.sorted.push(ch);
        for (const g of this.groups) {
          if (dist > g.radius + CHUNK * 0.75) continue;
          if (ch.data[g.id] !== undefined) continue;
          this.queue.push(this._task(ch, g, dist));
        }
      }
    }

    // Nearest first for packing; the build queue is popped from the end, so it is
    // sorted the other way round.
    this.sorted.sort(byDistAsc);
    this.queue.sort(byDistDesc);
    // The chunk ORDER changed, so every group has to be re-packed once even if none
    // of its chunk data did: packing consumes nearest-first and an overflow drops
    // the farthest, so a stale order silently keeps the wrong instances resident.
    for (const g of this.groups) g.dirty = true;
  }

  _drainQueue() {
    const rs = this._rs;
    if (!this.queue.length && !rs.group) { this._firstFill = false; return; }
    // The first fill lands on the frame the loading screen hides, so it can afford a
    // fat budget; steady-state streaming has to stay invisible.
    //
    // The deadline is enforced INSIDE _generate, per row of cells, not just between
    // chunks. A single dense ground-cover chunk is ~1000 cells and costs more than
    // the whole steady-state budget on its own, so a between-chunks-only check gave
    // a measured 4 ms hitch on every chunk boundary crossed - once every 32 m of
    // walking, which is exactly often enough to feel.
    const budget = this._firstFill ? 6.0 : 1.1;
    const deadline = performance.now() + budget;

    if (rs.group) {
      if (!this._generate(rs.group, rs.chunk, deadline)) return;
      rs.group.dirty = true;
      rs.group = null;
      rs.chunk = null;
    }

    while (this.queue.length) {
      const t = this.queue.pop();
      rs.group = t.group;
      rs.chunk = t.chunk;
      rs.j = 0; rs.n = 0;
      rs.lo = Infinity; rs.hi = -Infinity; rs.rad = 0;
      this._recycle(t);
      if (!this._generate(rs.group, rs.chunk, deadline)) return;
      rs.group.dirty = true;
      rs.group = null;
      rs.chunk = null;
      if (performance.now() > deadline) break;
    }
    if (!this.queue.length) this._firstFill = false;
  }

  // -------------------------------------------------------------------------
  // Placement
  // -------------------------------------------------------------------------

  _height(x, z) {
    const t = this.terrain;
    return t && t.getHeight ? t.getHeight(x, z) : 0;
  }

  _normal(x, z) {
    const t = this.terrain;
    if (t && t.getNormal) {
      const r = t.getNormal(x, z, _nrm);
      if (r && r !== _nrm) _nrm.set(r.x, r.y, r.z);
      if (_nrm.lengthSq() < 1e-6) _nrm.set(0, 1, 0); else _nrm.normalize();
    } else {
      _nrm.set(0, 1, 0);
    }
    return _nrm;
  }

  /**
   * Height AND normal in three height taps instead of five, by forward-differencing
   * off the sample we already had to take. `terrain.getNormal` is a central
   * difference, so the accurate version costs four extra taps per plant - at ~0.7 us
   * each, over a few hundred plants a chunk, that is most of the streaming budget
   * for a 35 cm bias in where the slope is measured. Rocks and fallen wood, which
   * visibly lie against the ground, still get the exact normal.
   */
  _ground(x, z) {
    const t = this.terrain;
    if (!t || !t.getHeight) { _nrm.set(0, 1, 0); return 0; }
    const e = 0.7, inv = 1 / e;
    const h0 = t.getHeight(x, z);
    const nx = -(t.getHeight(x + e, z) - h0) * inv;
    const nz = -(t.getHeight(x, z + e) - h0) * inv;
    const k = 1 / Math.sqrt(nx * nx + 1 + nz * nz);
    _nrm.set(nx * k, k, nz * k);
    return h0;
  }

  // --- shared environment fields ------------------------------------------
  // Two cheap scalars that everything else is expressed in terms of, so the
  // species correlate the way a real plant community does: dock and reed answer
  // damp ground, seed-heads and bent-grass prefer the dry rises, and nothing at
  // all likes the stony patches the boulders come from.

  _stonyAt(x, z) {
    const nz = this.noise;
    return clamp01(nz.fbm2D(x * 0.0105, z * 0.0105, 3, 2.1, 0.55) * 0.5 + 0.5);
  }

  _moistAt(x, z) {
    const nz = this.noise;
    return clamp01(
      nz.noise2D(x * 0.0062 + 311, z * 0.0062 - 77) * 0.5 + 0.5
      + nz.noise2D(x * 0.0190 - 12, z * 0.0190 + 43) * 0.17);
  }

  /** Two-octave patch field - the thing that turns a sprinkle into a drift. */
  _patchAt(x, z, f, ox, oz) {
    const nz = this.noise;
    return clamp01(
      nz.noise2D(x * f + ox, z * f + oz) * 0.5 + 0.5
      + nz.noise2D(x * f * 2.7 + ox * 0.3, z * f * 2.7 + oz * 0.3) * 0.18);
  }

  /** How walked-on this spot is: 0 at the trunk, 1 out past the canopy edge. */
  _trampleAt(x, z) {
    const d = Math.hypot(x - this._treePos.x, z - this._treePos.z);
    return smoothstep(this._canopyR * 0.16, this._canopyR * 0.62, d);
  }

  /**
   * 0..1 acceptance weight for a group at a world position.
   *
   * Every species gets its own field, and the fields are deliberately correlated
   * through `stony` and `moist` rather than being six independent noises. That is
   * what makes the meadow read as terrain-driven - a damp hollow full of reeds and
   * dock, a dry stony rise with seed-heads and nothing underfoot - instead of six
   * unrelated confetti layers stacked on the same ground.
   */
  _densityAt(group, x, z) {
    const field = group.field;

    if (field === 'rock') return smoothstep(0.34, 0.80, this._stonyAt(x, z));

    if (field === 'stick') {
      const nz = this.noise;
      const d = Math.hypot(x - this._treePos.x, z - this._treePos.z);
      const near = Math.exp(-d / Math.max(6, this._canopyR * 2.4)); // limbs drop near the tree
      const woody = clamp01(nz.fbm2D(x * 0.021 + 40, z * 0.021 - 17, 2, 2.0, 0.5) * 0.5 + 0.5);
      return clamp01(smoothstep(0.30, 0.72, woody) * 0.85 + near * 1.6);
    }

    if (field === 'petal') {
      // A soft pool under the canopy streaked downwind, plus a thin scatter
      // everywhere, so the drift has no visible edge.
      const nz = this.noise;
      const dx = x - this._treePos.x, dz = z - this._treePos.z;
      const pool = Math.pow(clamp01(1 - Math.hypot(dx, dz) / this._driftR), 1.7);
      const ca = 0.86, sa = 0.51;   // fixed prevailing drift axis: stable at bake time
      const drift = clamp01(
        nz.fbm2D((dx * ca - dz * sa) * 0.018, (dx * sa + dz * ca) * 0.115, 2, 2.0, 0.5) * 0.5 + 0.5);
      return clamp01((pool * 0.92 + 0.09) * smoothstep(0.30, 0.82, drift) * 1.5);
    }

    // --- plants
    const trample = lerp(group.trample[0], 1, this._trampleAt(x, z));
    let w;
    switch (field) {
      case 'cover': {
        // Creeping cover follows damp shade and fills in wherever the sward thins.
        const p = this._patchAt(x, z, 0.058, 501, -233);
        const m = this._moistAt(x, z);
        w = smoothstep(0.28, 0.70, p) * (0.34 + 0.66 * m);
        break;
      }
      case 'bent': {
        // The most widespread species out there: broad drifts, only really absent
        // from the stoniest ground.
        const p = this._patchAt(x, z, 0.034, -91, 617);
        w = smoothstep(0.16, 0.58, p) * (0.42 + 0.58 * (1 - this._stonyAt(x, z)));
        break;
      }
      case 'broad': {
        // Dock stands: tight, fertile, damp. Rare, and never alone.
        const p = this._patchAt(x, z, 0.026, 733, 155);
        w = smoothstep(0.50, 0.86, p) * (0.22 + 0.78 * this._moistAt(x, z));
        break;
      }
      case 'stalk': {
        // Seed-heads: wide, soft drifts that thin out where the ground gets wet.
        const p = this._patchAt(x, z, 0.019, -407, -61);
        w = smoothstep(0.20, 0.64, p) * (0.50 + 0.50 * (1 - this._moistAt(x, z)));
        break;
      }
      case 'bloom': {
        // Wildflowers grow in colonies, not as a sprinkle. Tight threshold, and a
        // hard preference for the thin unshaded soil the grass does worst on.
        const p = this._patchAt(x, z, 0.044, 977, 313);
        w = smoothstep(0.55, 0.88, p) * (0.30 + 0.70 * this._stonyAt(x, z) * 1.1);
        break;
      }
      default: {
        // Reeds: damp hollows only, and even there in discrete stands.
        const m = this._moistAt(x, z);
        const p = this._patchAt(x, z, 0.028, 61, -829);
        w = smoothstep(0.46, 0.80, m) * smoothstep(0.22, 0.66, p);
        break;
      }
    }
    return clamp01(w) * trample;
  }

  /**
   * Fills one chunk's slice of the scratch buffers. Resumable: it stops on a row
   * boundary once `deadline` passes and returns false, leaving the cursor in
   * `this._rs` for the next frame to pick up. At least one row always runs, so it
   * can never fail to make progress.
   *
   * @returns {boolean} true when the chunk is finished and committed.
   */
  _generate(group, chunk, deadline) {
    const cell = group.cell;
    const per = group.cellsPerChunk;
    const baseI = chunk.cx * per, baseJ = chunk.cz * per;
    const maxP = group.densityNow;
    const sMat = this._sMat, sCol = this._sCol, sExt = this._sExt;
    const limit = Math.min(group.maxPerChunk, SCRATCH);
    const kind = group.kind;
    const plant = kind === 'plant';
    const slopeMin = group.slopeMin ?? 0;
    const rs = this._rs;

    const j0 = rs.j;
    this._bLo = rs.lo;
    this._bHi = rs.hi;
    this._bRad = rs.rad;

    let n = rs.n;
    outer:
    for (let j = j0; j < per; j++) {
      if (j > j0 && performance.now() > deadline) {
        rs.j = j; rs.n = n;
        rs.lo = this._bLo; rs.hi = this._bHi; rs.rad = this._bRad;
        return false;
      }
      for (let i = 0; i < per; i++) {
        if (n >= limit) break outer;
        const ix = baseI + i, iz = baseJ + j;
        const u = cellHash01(ix, iz, group.salt);
        if (u > maxP) continue;                       // cheap reject before any noise
        const ccx = (ix + 0.5) * cell, ccz = (iz + 0.5) * cell;
        if (u > maxP * this._densityAt(group, ccx, ccz)) continue;

        const rng = PLACE_RNG;
        rng.seed(cellSeed(ix, iz, group.salt ^ 0x5bf03635));
        const x = ccx + (rng() - 0.5) * cell * 0.80;
        const z = ccz + (rng() - 0.5) * cell * 0.80;

        if (Math.hypot(x - this._treePos.x, z - this._treePos.z) < 1.15) continue; // trunk base

        // Plants get the cheap forward-difference normal; anything that lies flat
        // against the soil gets the exact central difference.
        let y, nrm;
        if (plant || kind === 'petal') {
          y = this._ground(x, z);
          nrm = _nrm;
        } else {
          nrm = this._normal(x, z);
          y = this._height(x, z);
        }
        // A non-finite ground sample is the one failure this module cannot walk
        // off: it would go straight into the chunk's cached matrix slice, and
        // regeneration is bit-identical, so that instance would be NaN for the
        // rest of the session. Two compares per accepted cell to make it
        // impossible.
        const slope = nrm.y;
        if (!(y > -1e6 && y < 1e6) || !(slope >= -1 && slope <= 1)) continue;
        if (slope < slopeMin) continue;
        // Boulders prefer broken ground; scattered evenly over a perfect lawn they
        // look dropped in. Slope is a free proxy for "here the soil is thin".
        if (kind === 'rock' && slope > 0.995 && rng() < 0.35) continue;

        this._emit(group, rng, n, x, y, z, nrm, sMat, sCol, sExt);
        n++;
      }
    }

    chunk.data[group.id] = n === 0 ? EMPTY_CHUNK_DATA : {
      count: n,
      mat: sMat.slice(0, n * 16),
      col: sCol.slice(0, n * 3),
      ext: group.attrs.map((a, k) => sExt[k].slice(0, n * a.size)),
      // Real extents, measured, not assumed: the packed mesh's bounding sphere is
      // the union of these. A guessed sphere either culls the field away or draws
      // it when nothing is on screen, and both failures are silent.
      yLo: this._bLo,
      yHi: this._bHi,
      rad: this._bRad,
    };
    return true;
  }

  /** Composes one instance into the chunk scratch buffers. */
  _emit(group, rng, n, x, y, z, nrm, sMat, sCol, sExt) {
    const kind = group.kind;
    let size, align = group.align, phase = 0, stiff = 1, uShift = 0, vShift = 0;

    if (kind === 'rock') {
      // Skewed small: a field of same-sized rocks is the classic procedural tell.
      size = lerp(group.size[0], group.size[1], Math.pow(rng(), group.sizePow));
      _scl.set(size * rng.range(0.88, 1.14), size * rng.range(0.62, 1.05), size * rng.range(0.88, 1.14));
      y -= size * group.sink * rng.range(0.7, 1.3);
    } else if (kind === 'stick') {
      size = lerp(group.size[0], group.size[1], Math.pow(rng(), group.sizePow));
      _scl.set(size, size * rng.range(0.75, 1.05), size);
      y += 0.004;
    } else if (kind === 'plant') {
      size = lerp(group.size[0], group.size[1], Math.pow(rng(), group.sizePow));
      // Trodden ground grows shorter plants, not just fewer of them.
      size *= lerp(group.trample[1], 1, this._trampleAt(x, z));
      // Near-uniform: the tile aspect is baked into the card width, so pulling x/z
      // far from y would visibly stretch the drawing.
      _scl.set(size * rng.range(0.90, 1.12), size, size * rng.range(0.90, 1.12));
      phase = rng() * TAU * 3;
      stiff = lerp(group.stiff[0], group.stiff[1], rng()) * rng.range(0.86, 1.16);
      // Tall plants lean with the ground less than short ones do.
      align = group.align * lerp(1.0, 0.55, clamp01((size - 0.25) / 1.4));
      if (group.variants > 1) {
        // A low-frequency field picks the variant, so one species holds a whole
        // drift and the neighbouring drift is a different one. Per-instance random
        // choice reads as noise; this reads as a plant community.
        const f = group.variantFreq;
        const s = this.noise.noise2D(x * f + group.salt * 0.017, z * f - group.salt * 0.011)
          + rng.range(-0.22, 0.22);
        const k = clamp(Math.floor((s * 0.5 + 0.5) * group.variants), 0, group.variants - 1);
        uShift = k * group.variantStep;
      }
      y -= 0.015;
    } else {
      size = rng.range(group.size[0], group.size[1]);
      _scl.set(size * rng.range(0.9, 1.1), 1, size * rng.range(0.9, 1.1));
      y += 0.010 + rng() * 0.006;   // clear of the ground plane without floating
    }

    // --- orientation: partial alignment to the ground normal, then yaw, then tilt
    _qa.setFromUnitVectors(UP, nrm);
    _qb.identity().slerp(_qa, align);
    _qc.setFromAxisAngle(UP, rng() * TAU);
    _qb.multiply(_qc);
    if (group.tilt > 0) {
      const ta = rng() * TAU;
      _axis.set(Math.cos(ta), 0, Math.sin(ta));
      _qa.setFromAxisAngle(_axis, rng.gaussian(0, group.tilt * 0.45));
      _qb.multiply(_qa);
    }

    _pos.set(x, y, z);
    _mat.compose(_pos, _qb, _scl);
    sMat.set(_mat.elements, n * 16);

    // --- per-instance tint
    if (kind === 'rock') {
      // A regional term over a ~160 m wavelength on top of the per-instance jitter:
      // stone in one part of the field shares a cast, which is how real geology
      // looks. Pure per-instance randomness reads as confetti.
      const region = this.noise.noise2D(x * 0.0062, z * 0.0062);
      const v = rng.range(0.80, 1.16) * (1 + region * 0.10);
      _c0.setRGB(
        v * rng.range(0.96, 1.05) * (1 + region * 0.055),
        v,
        v * rng.range(0.93, 1.04) * (1 - region * 0.045));
    } else if (kind === 'stick') {
      const v = rng.range(0.78, 1.12);
      _c0.setRGB(v * rng.range(1.00, 1.09), v, v * rng.range(0.86, 0.98));
    } else if (kind === 'plant') {
      this._plantTint(group, rng, x, z, _c0);
    } else if (rng() < 0.22) {
      _c0.setRGB(0.78, 0.70, 0.62).multiplyScalar(rng.range(0.85, 1.05));   // browned petal
    } else {
      _c0.setRGB(1.00, 0.985, 0.99).multiplyScalar(rng.range(0.82, 1.06));
    }
    sCol[n * 3] = _c0.r; sCol[n * 3 + 1] = _c0.g; sCol[n * 3 + 2] = _c0.b;

    // --- per-instance fade window: big props survive to the group radius, small ones
    //     give up long before they turn into sub-pixel shimmer.
    //
    //     Scaled by radiusK - how far this tier streams relative to the reference
    //     tier - so the fade tracks the quality dial instead of being a constant.
    //     Without it LOW streams chunks it then fades to nothing inside, and ULTRA
    //     pays to stream chunks whose contents died at the HIGH distance. Most
    //     groups set baseRadius to the reference fade reach of their largest
    //     instance so the two end in the same place; the four that cap it below
    //     that (rockL, branch, stalk, reed) fade their biggest instances over a
    //     shorter window instead, which is still a fade and never a pop.
    const fadeEnd = Math.min(
      group.radius, (group.fadeK[0] + size * group.fadeK[1]) * group.radiusK);
    sExt[0][n * 2] = fadeEnd * 0.80;
    sExt[0][n * 2 + 1] = fadeEnd;

    if (kind === 'plant') {
      const o = n * 4;
      sExt[1][o] = phase;
      sExt[1][o + 1] = stiff;
      sExt[1][o + 2] = uShift;
      sExt[1][o + 3] = vShift;
    }

    // --- accumulate the real extent of this chunk's slice. `localR` is the
    //     geometry's own bounding radius about its pivot, so this is a measured
    //     half-extent, not a guess; _pack() adds the half-chunk that an instance
    //     can sit from its chunk centre.
    const half = group.localR * Math.max(_scl.x, _scl.y, _scl.z);
    if (y - half < this._bLo) this._bLo = y - half;
    if (y + half > this._bHi) this._bHi = y + half;
    if (half > this._bRad) this._bRad = half;
  }

  /**
   * Per-instance plant tint. Every species carries a slow regional term so a drift
   * shares a cast, exactly like the rocks - a meadow where each plant is
   * independently randomised reads as static, not as variety.
   */
  _plantTint(group, rng, x, z, out) {
    const region = this.noise.noise2D(x * 0.0074 + group.salt * 0.003, z * 0.0074);
    switch (group.field) {
      case 'cover': {
        const v = rng.range(0.84, 1.06) * (1 + region * 0.06);
        out.setRGB(v * rng.range(0.94, 1.04), v * (1 + 0.03 * region), v * rng.range(0.88, 0.98));
        break;
      }
      case 'bent': {
        // Bleaches toward khaki on the dry rises, stays green in the hollows. The
        // only species that pays for a second field lookup here, because it is the
        // one that covers enough ground for the gradient to be visible.
        const v = rng.range(0.82, 1.10) * (1 + region * 0.07);
        const k = clamp01(0.62 - this._moistAt(x, z) * 0.55 + region * 0.14);
        out.setRGB(v * lerp(1.0, 1.16, k), v * lerp(1.0, 1.05, k), v * lerp(1.0, 0.80, k));
        break;
      }
      case 'broad': {
        const v = rng.range(0.80, 1.06) * (1 + region * 0.05);
        const rust = rng() < 0.18 ? rng.range(0.15, 0.45) : 0;
        out.setRGB(v * lerp(1.0, 1.22, rust), v * lerp(1.0, 0.96, rust), v * lerp(1.0, 0.72, rust));
        break;
      }
      case 'stalk': {
        const v = rng.range(0.86, 1.10) * (1 + region * 0.06);
        out.setRGB(v * rng.range(1.00, 1.10), v, v * rng.range(0.84, 0.96));
        break;
      }
      case 'bloom': {
        // The texture carries the hue; the tint only varies its value, or the
        // colonies stop reading as one species.
        const v = rng.range(0.90, 1.08);
        out.setRGB(v, v * rng.range(0.985, 1.01), v * rng.range(0.97, 1.02));
        break;
      }
      default: {
        const v = rng.range(0.86, 1.12) * (1 + region * 0.05);
        out.setRGB(v * rng.range(1.02, 1.12), v, v * rng.range(0.80, 0.92));
        break;
      }
    }
  }

  _pack(group) {
    const mesh = group.mesh;
    const mArr = mesh.instanceMatrix.array;
    const cArr = mesh.instanceColor.array;
    const cap = group.capacity;
    const limit = group.radius + CHUNK * 0.75;
    const chunks = this.sorted;   // nearest first, so an overflow drops the farthest
    let n = 0;

    let bx0 = Infinity, bx1 = -Infinity, bz0 = Infinity, bz1 = -Infinity;
    let by0 = Infinity, by1 = -Infinity;

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      if (chunk.dist > limit) break;
      const d = chunk.data[group.id];
      if (!d || d.count === 0) continue;
      if (n + d.count > cap) break;
      mArr.set(d.mat, n * 16);
      cArr.set(d.col, n * 3);
      for (let k = 0; k < group.attrs.length; k++) {
        group.attrs[k].attr.array.set(d.ext[k], n * group.attrs[k].size);
      }
      n += d.count;

      const pad = CHUNK * 0.5 + d.rad;
      if (chunk.wx - pad < bx0) bx0 = chunk.wx - pad;
      if (chunk.wx + pad > bx1) bx1 = chunk.wx + pad;
      if (chunk.wz - pad < bz0) bz0 = chunk.wz - pad;
      if (chunk.wz + pad > bz1) bz1 = chunk.wz + pad;
      if (d.yLo < by0) by0 = d.yLo;
      if (d.yHi > by1) by1 = d.yHi;
    }

    mesh.count = n;
    mesh.visible = n > 0;
    if (n === 0) {
      mesh.boundingSphere.center.set(0, 0, 0);
      mesh.boundingSphere.radius = 0;
      return;
    }

    // Real world-space bounds, unioned from the chunk extents actually packed.
    // O(chunks), not O(instances), and exact enough to be safe to cull against.
    const bs = mesh.boundingSphere;
    bs.center.set((bx0 + bx1) * 0.5, (by0 + by1) * 0.5, (bz0 + bz1) * 0.5);
    bs.radius = 0.5 * Math.hypot(bx1 - bx0, by1 - by0, bz1 - bz0);

    uploadPrefix(mesh.instanceMatrix, n * 16);
    uploadPrefix(mesh.instanceColor, n * 3);
    for (let k = 0; k < group.attrs.length; k++) {
      uploadPrefix(group.attrs[k].attr, n * group.attrs[k].size);
    }
  }

  // -------------------------------------------------------------------------
  // Per-frame uniforms
  // -------------------------------------------------------------------------

  _updateUniforms(state, cam) {
    const u = this.u;
    // Everything read out of `state` below goes through `finite()`. clamp() is
    // comparison-based and propagates NaN, and a NaN in uWindStr or uWindDir
    // takes EVERY plant in the module off screen for as long as it lasts.
    u.time.value = finite(state.time.elapsed, u.time.value);
    u.camPos.value.set(finite(cam.x, 0), finite(cam.y, 0), finite(cam.z, 0));

    // Re-normalised rather than copied: `state.wind.direction` is documented as a
    // unit heading, so this is free in the good case and a firewall otherwise.
    const wdx = finite(state.wind.direction.x, 1);
    const wdy = finite(state.wind.direction.y, 0);
    const wdl = Math.hypot(wdx, wdy);
    if (wdl > 1e-4) u.windDir.value.set(wdx / wdl, wdy / wdl);
    else u.windDir.value.set(1, 0);
    // Normalised, so the shader's bend constants stay meaningful whatever units the
    // wind system settles on.
    u.windStr.value = clamp(finite(state.wind.strength, 0) / 6, 0, 1.8);

    // The SAME call the grass makes, at the same point, so a gust front that lays
    // the sward over lays the seed-heads over with it. Reading `state.wind.gust`
    // instead would be a frame-damped scalar and the two layers would drift apart
    // during a front, which is instantly readable as two systems rather than one
    // field of wind.
    const gustAt = this.wind?.getGustAt || state.wind.getGustAt;
    const gust = gustAt ? gustAt(u.camPos.value.x, u.camPos.value.z, u.time.value) : state.wind.gust;
    u.gust.value = clamp(finite(gust, 1), 0.2, 2.4);
    u.turb.value = clamp01(finite(state.wind.turbulence, 0.4));

    const w = state.weather;
    const snow = clamp01(finite(w.snowIntensity, 0) * 1.15);
    u.snow.value = snow;
    scrubColor(u.snowColor.value.copy(state.sky.horizonColor).lerp(WHITE, 0.55)
      .multiplyScalar(0.92), 2);

    // Backlight drive for the plant translucency term. Scaled by 1/PI to sit in the
    // same units as three's Lambert diffuse, so it reads as a fraction of the direct
    // term rather than an arbitrary bloom.
    const sun = state.sun;
    const vis = clamp01(finite(sun.visibility, 1));
    const wet = clamp01(finite(w.wetness, 0));
    u.sunDir.value.set(
      finite(sun.direction.x, 0), finite(sun.direction.y, 1), finite(sun.direction.z, 0));
    // Written as a positive test so a NaN length falls into the reset branch too.
    if (!(u.sunDir.value.lengthSq() > 1e-8)) u.sunDir.value.set(0, 1, 0);
    const sunK = clamp(finite(sun.intensity, 0), 0, 8) * vis * 0.16 * lerp(1, 0.55, wet);
    u.sunCol.value.setRGB(
      finite(sun.color.r, 1) * sunK, finite(sun.color.g, 1) * sunK, finite(sun.color.b, 1) * sunK);

    // Wet stone darkens and gets much glossier; foliage and petals only darken.
    this.matGround.roughness = lerp(0.93, 0.44, wet);
    this.matGround.color.setScalar(lerp(1, 0.60, wet));
    this.matFlora.color.setScalar(lerp(1, 0.80, wet));
    this.matPetal.color.setScalar(lerp(1, 0.66, wet) * lerp(1, 1.12, snow));
  }

  _updateHorizon(dt, state, cam) {
    const hu = this.hu;
    if (!hu) return;

    // Prefer whatever the scene actually renders with, so the treeline can never
    // disagree with the terrain in front of it.
    const fog = this.scene.fog;
    let density = state.weather.fogDensity;
    if (fog && fog.isFogExp2) {
      density = fog.density;
      hu.uFogColor.value.copy(fog.color);
    } else {
      hu.uFogColor.value.copy(state.weather.fogColor);
    }
    density = clamp(finite(density, 0.0022), 0, 0.5);
    // At the skyline, fog and sky are the same thing - and `sky.horizonColor` is the
    // only one of the two guaranteed to track time of day, so let it dominate.
    // Without this the treeline sits as a dark band against a stale bright haze at
    // night, which is the single most obvious way to break a horizon.
    hu.uFogColor.value.lerp(state.sky.horizonColor, 0.70);
    scrubColor(hu.uFogColor.value, 4);
    hu.uFogDensity.value = density;
    // Heavy fog is allowed to erase the treeline entirely; clear air never is.
    hu.uMaxExt.value = lerp(0.93, 1.0, smoothstep(0.004, 0.013, density));
    hu.uGroundHaze.value = lerp(0.34, 0.86, clamp01(density / 0.010));

    const sun = state.sun, moon = state.moon;
    const vis = clamp01(finite(sun.visibility, 1));
    const sunPower = clamp(finite(sun.intensity, 3) / 3, 0, 1.5);
    hu.uSunVis.value = vis;
    hu.uSunColor.value.setRGB(
      finite(sun.color.r, 1), finite(sun.color.g, 1), finite(sun.color.b, 1));
    _axis.set(finite(sun.direction.x, 1), 0, finite(sun.direction.z, 0));
    if (!(_axis.lengthSq() > 1e-6)) _axis.set(1, 0, 0); else _axis.normalize();
    hu.uSunDir.value.copy(_axis);

    // Forward-scattered haze on the sun side - the reason a distant treeline looks
    // washed out toward the sun and inky away from it.
    hu.uHazeColor.value.copy(hu.uFogColor.value)
      .lerp(_c1.copy(hu.uSunColor.value).multiplyScalar(Math.max(0.2, sunPower)), 0.34 * vis)
      .multiplyScalar(lerp(1.0, 1.16, vis));
    scrubColor(hu.uHazeColor.value, 4);

    // Canopy lighting, entirely state driven: sky ambient + a warm sun wash + a cold
    // moon wash. It goes near-black at night with no special case.
    const amb = clamp(finite(state.sky.ambientIntensity, 1), 0, 2.2);
    _c0.setRGB(0.055, 0.070, 0.052).multiplyScalar(0.16 + 0.62 * amb);
    _c1.copy(state.sky.zenithColor).multiplyScalar(0.030 * amb);
    _c2.copy(hu.uSunColor.value).multiplyScalar(0.055 * vis * sunPower);
    _c0.add(_c1).add(_c2);
    _c1.copy(moon.color).multiplyScalar(0.020 * clamp01(finite(moon.visibility, 0)) * (1 - vis));
    _c0.add(_c1);
    scrubColor(_c0, 2);

    // a strike silhouettes it hard
    const flash = 1 + clamp01(finite(state.weather.lightning, 0)) * 0.9;
    hu.uCanopy.value.copy(_c0).multiplyScalar(flash);
    hu.uCanopyWarm.value.copy(_c0).multiply(WARM_SHIFT).multiplyScalar(flash);

    this.bandMat.color.copy(hu.uFogColor.value).multiplyScalar(0.97);

    // Placement: the far ring rides with the camera; the near ring trails it by a
    // bounded amount, which is real parallax between the two without ever losing
    // centring however far the player walks.
    // `_lag`, `_prevCam` and `_horizonY` are all integrators: a single NaN frame
    // from the camera or the terrain sticks in them for the rest of the session
    // (`NaN - x` is NaN, `damp(NaN, t)` is NaN, and `NaN > 25600` is false so the
    // existing clamp never fires). Sanitise the inputs and check the accumulator.
    const cx = finite(cam.x, 0), cz = finite(cam.z, 0);
    if (!this._hasPrevCam) { this._prevCam.set(cx, cz); this._hasPrevCam = true; }
    this._lag.x -= (cx - this._prevCam.x) * 0.20;
    this._lag.y -= (cz - this._prevCam.y) * 0.20;
    this._prevCam.set(cx, cz);
    const lag2 = this._lag.lengthSq();
    if (!Number.isFinite(lag2)) this._lag.set(0, 0);
    else if (lag2 > 25600) this._lag.setLength(160);

    // Height: average the ground over a 1 km ring, which is exactly the far-field
    // level the treeline should be standing on. Eight cheap height samples, and
    // because the terrain's big octave has a ~1.5 km wavelength the average drifts
    // smoothly as the player walks - it reads as real landform, not as bobbing.
    if (this.terrain) {
      let sum = 0;
      for (let i = 0; i < 8; i++) {
        sum += this._height(cx + HORIZON_RING[i * 2], cz + HORIZON_RING[i * 2 + 1]);
      }
      const target = sum * 0.125;
      if (Number.isFinite(target)) {
        const sdt = clamp(finite(dt, 0), 0, 0.25);
        this._horizonY = this._horizonYInit ? damp(this._horizonY, target, 3.0, sdt) : target;
        this._horizonYInit = Number.isFinite(this._horizonY);
        if (!this._horizonYInit) this._horizonY = target;
      }
    }
    const hy = finite(this._horizonY, 0);

    this.treeFar.position.set(cx, hy, cz);
    this.treeNear.position.set(cx + this._lag.x, hy, cz + this._lag.y);
    this.band.position.set(cx, hy, cz);
  }

  // -------------------------------------------------------------------------

  dispose() {
    this.root.removeFromParent();
    this.horizon.removeFromParent();
    for (const g of this.groups) g.mesh.dispose?.();
    for (const d of this._disposables) d.dispose?.();
    for (const t of this._ownedTextures) t.dispose?.();
    this._disposables.length = 0;
    this._ownedTextures.length = 0;
    this.chunks.clear();
    this.sorted.length = 0;
    this.queue.length = 0;
    this._taskPool.length = 0;
    this.groups.length = 0;
    this._rs.group = null;
    this._rs.chunk = null;
  }
}

// ---------------------------------------------------------------------------
// Small shared helpers
// ---------------------------------------------------------------------------

/** Shared immutable marker so empty chunks cost one object for the whole world. */
const EMPTY_CHUNK_DATA = Object.freeze({
  count: 0, mat: null, col: null, ext: null, yLo: 0, yHi: 0, rad: 0,
});
const WHITE = new THREE.Color(1, 1, 1);
const WARM_SHIFT = new THREE.Color(1.20, 1.02, 0.80);

const byDistAsc = (a, b) => a.dist - b.dist;
const byDistDesc = (a, b) => b.dist - a.dist;

/** Uploads only the used prefix of an instance buffer. */
function uploadPrefix(attr, count) {
  if (attr.clearUpdateRanges) {
    attr.clearUpdateRanges();
    attr.addUpdateRange(0, count);
  }
  attr.needsUpdate = true;
}
