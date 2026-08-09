/**
 * world/terrain.js - the ground.
 *
 * Three things live here:
 *
 *  1. An analytic heightfield.  `getHeight(x,z)` / `getNormal(x,z,out)` are
 *     synchronous, allocation-free and cheap enough to be hammered every frame
 *     by the player controller and by grass placement.  Ten simplex taps,
 *     hand-unrolled, ~0.7 us a call.  No tables, no caches to invalidate.
 *
 *  2. A world-aligned quadtree clipmap.  Chunks sit on a fixed world lattice - 
 *     they are NOT re-snapped to the camera - so a chunk's vertex data stays
 *     valid forever once built; walking away and back is a cache hit, not a
 *     rebuild.  LOD transitions use CDLOD-style vertex geomorphing: every
 *     vertex slides onto its parent-grid position as it approaches the level's
 *     range limit, and because the parent grid's heights come from the same
 *     analytic field, a fully-morphed chunk is bit-identical to the chunk that
 *     replaces it.  No popping.  A short skirt covers the residual seam where
 *     the quadtree hands a two-level jump to a neighbour.
 *
 *  3. A splat-blended ground material patched into MeshStandardMaterial, so it
 *     keeps three's shadows, lights and tonemapping while adding height-blended
 *     earth / thatch / moss / petal-litter layers, cloud shadows, wetness,
 *     puddles and aerial perspective.
 *
 *     THE GROUND IS EARTH, NOT VEGETATION. This is worth stating because the
 *     material used to carry a "grass" layer with a green albedo, and the
 *     result read as a mottled blue-green bog: two systems were both drawing
 *     the sward, and the one with no silhouette won the argument on every pixel
 *     between the blades. world/grass.js draws the meadow. What this material
 *     draws is the substrate the meadow grows out of - warm dark umber where it
 *     is damp and shaded, dry pale grey-brown on the crusted ridges, with dead
 *     thatch, root debris and grit lying on it. Moss is a minority layer
 *     confined to hollows and shade; there is no living green in the palette at
 *     all, and there must not be.
 *
 *     The one exception is deliberate and is the whole of LEAF 3: past the
 *     reach of the blade field there is nothing but this material, so beyond
 *     that distance the albedo converges on the colour the sward itself reads
 *     as - sampled from grass.js's own uniforms, not guessed. See
 *     `_updateMeadowPalette` and `uMeadowFade`.
 *
 *     Five scales of variation, which is the minimum an open field needs before
 *     it stops reading as a tiled plane:
 *       2 m    near detail - grain, crumb, thatch strands, grit, petal flakes
 *       32 m   mid detail - the same texture, quarter-turned, so the two
 *                             copies can never correlate
 *       128 m  meso field - patch-scale coverage; also domain-warps the two
 *                             detail lookups, which is what actually kills the
 *                             2 m repeat rather than merely disguising it
 *       1024 m macro field - which stretches are dry, lush, mossy, mottled
 *       ~kilometre - the damp/dry axis (`gDry`) swings the earth
 *                             albedo by a factor of 3.7 in linear luminance
 *                             across the whole field, which is by far the
 *                             strongest broad drift and the one that actually
 *                             hides the tile at a distance
 *     Coverage is competed between layers by slope, local cavity, regional
 *     altitude and both noise scales, so nothing has uniform coverage anywhere.
 *
 *     ATMOSPHERICS.  If weather/atmosfx.js is present it owns aerial
 *     perspective for the whole scene and this material simply hands it the fog
 *     flag. Otherwise the fallback below integrates two exponentially
 *     stratified media - a low mist and a tall haze - analytically along the
 *     view ray, in linear space, AFTER lighting. Extinction is Beer-Lambert in
 *     the true optical depth, with sigma taken straight from
 *     `state.weather.fogDensity` (weather.js derives visibility = 3/sigma from
 *     the same number, which is what fixes the units). At the default 0.0022/m
 *     that is 0.6% wash at three metres and 18% at a hundred - depth cue, not
 *     a grey veil - reaching the horizon at a kilometre and a half.
 *
 * Budget notes for the 780M, measured rather than estimated: the selection is
 * 81 / 81 / 87 / 201 chunks at LOW / MEDIUM / HIGH / ULTRA from the spawn point
 * - 156k triangles at HIGH - of which three's per-object frustum cull typically
 * submits a third, so terrain draw calls land around 30. (ULTRA's count triples
 * because range0 * 2^5 exceeds ROOT_SIZE, so every root in the 3x3 window
 * refines instead of half of them; that is a real tier cost, not a bug.) The
 * terrain casts no shadows - gentle relief gains almost nothing from
 * self-shadowing and the shadow pass is the scarcest millisecond in the frame.
 * The fragment shader takes six texture fetches at most, two of which are
 * branched away beyond the detail-fade distance - which is where most of the
 * screen is - and the two macro taps hit a 64 kB texture that never leaves
 * cache.
 */

import * as THREE from 'three';
import {
  createNoise, hash2, hash3, clamp, clamp01, lerp, smoothstep, smootherstep, damp, mod,
} from '../core/math.js';

// ---------------------------------------------------------------------------
// Heightfield.  Amplitudes are metres, frequencies are 1 / wavelength.
// Tuned so mean slope is ~5 deg and max ~24 deg: rolling, walkable, never lumpy.
// ---------------------------------------------------------------------------

const TERRAIN_SEED = 20240517;

const WARP_F = 1 / 2600, WARP_A = 210;   // domain warp - stops fbm reading as round blobs
const ENV_F = 1 / 5400;                  // relief envelope - plains between the rolling stretches
const BIG_F = 1 / 1450, BIG_A = 34;      // the horizon silhouette
const BIG_F2 = BIG_F * 2.03, BIG_F3 = BIG_F * 4.1209;
const MID_F = 1 / 265, MID_A = 4.6;      // hillside shoulders
const MID_F2 = MID_F * 2.11;
const SML_F = 1 / 62, SML_A = 1.05;      // the scale you actually walk over
const MIC_F = 1 / 23, MIC_A = 0.24;      // just enough that the ground is not a NURBS patch

/**
 * Cavity (height minus the smooth trend) spans about +/- 1.3 m, so this maps it
 * into a byte with headroom. Every threshold downstream is in these units.
 */
const CAVITY_SCALE = 2.6;

// Flattened clearing under the sakura, metres.
const PAD_IN = 21, PAD_OUT = 92, PAD_RISE = 0.9;
const PAD_LIMIT = PAD_OUT * 1.35, PAD_LIMIT_SQ = PAD_LIMIT * PAD_LIMIT;

// Quadtree. Root cells are world-aligned; a 3x3 window of them is drawn and a
// 5x5 window is kept warm, which guarantees at least ROOT_SIZE metres of ground
// in every direction no matter where the camera sits inside its root cell.
const ROOT_SIZE = 4096;
const PREFETCH = 1.45;                   // build children this far before they are needed
const MORPH_START = 0.62;                // where geomorphing begins, as a fraction of the level
                                         // range. Must exceed 0.5 or a chunk would start
                                         // morphing while its finer neighbour is still crisp.
/**
 * Texture-space origin snap, metres. EVERY tile size that reads `tp` must
 * divide this exactly, or the pattern jumps when the snap moves - 256 / 4 = 64
 * tiles and 256 / 32 = 8 tiles, both whole, so it never does.
 *
 * What it buys: the near detail UV is `tp / 4`, so at 100 km from the origin an
 * unsnapped UV would be 25000, where a float32 ulp is half a texel of a 256 px
 * texture and the detail visibly stair-steps. Snapping caps the magnitude and
 * the world stays walkable indefinitely. 256 rather than the 2048 this used to
 * be simply because it costs nothing and leaves three more bits of headroom;
 * the snap fires every 256 m of walking and is invisible when it does.
 *
 * Note this is NOT a fix for mip selection - the derivative is differenced from
 * the interpolated world position, so its precision is set by the magnitude of
 * `vWorldPos`, which no snap downstream of the interpolator can change. At six
 * kilometres out that is a 0.5 mm ulp against a per-pixel ground footprint of
 * several millimetres even at the very bottom of the frame, so mip choice is
 * accurate to well under a tenth of a level regardless.
 */
const UV_SNAP = 256;

// Detail tiling, metres per repeat. Near and mid must divide UV_SNAP.
//
// 2 m, and it must STAY 2: core/textures.js sizes every feature in the near
// detail map against `TERRAIN_TILE_M = 2.0` (published back on the texture as
// `userData.tileMeters`), so changing this here silently rescales 11 mm grain
// and 40 mm gravel by the same factor. The tile went 4 -> 2 in an earlier pass
// as an attempt to shrink the round blobs; that was the wrong lever - shrinking
// a tile full of domes only shrinks the domes - and the blobs were finally
// removed by replacing the map's CONTENT (no inverted-F1 cones anywhere in the
// relief) rather than its scale. 2 m is kept because it is what the feature
// sizes are now authored against.
const TILE_NEAR = 2;
const TILE_MID = 32;
const TILE_MESO = 128;      // patch scale - sampled from raw world coords, no snap needed
const TILE_MACRO = 1024;    // region scale
/**
 * Domain-warp amplitude for the detail lookups, metres.
 *
 * The number is picked from a measurement, not by eye. Warping UVs distorts
 * their derivatives, and past a point that wrecks mip selection or folds the
 * mapping outright. Sampling the actual warp field on its 1 m texel lattice and
 * taking the determinant of d(warped)/d(world) over 160k points:
 *
 *   2.6 m  det 0.70 .. 1.31   p90 drift 0.25 tiles
 *   3.4 m  det 0.62 .. 1.40   p90 drift 0.33 tiles
 *   6.5 m  det 0.30 .. 1.81   p90 drift 0.62 tiles
 *  10.0 m  det -0.02 .. 2.29  FOLDS
 *
 * 3.4 costs at worst a third of a mip level - invisible - and slides the 4 m
 * lattice by a third of a tile at p90, so two patches 128 m apart are commonly
 * two thirds of a tile out of phase with each other. That is what breaks the
 * repeat. 6.5 would break it harder and visibly blur in patches; 10 tears.
 */
const DETAIL_WARP = 3.4;

/**
 * Split of the weather's extinction coefficient between the two media of the
 * fallback aerial model. They sum to 1 at ground level so `visibility = 3/sigma`,
 * the relation weather.js publishes, still holds at eye height.
 */
const HAZE_SHARE = 0.55;
const MIST_SHARE = 0.45;

/**
 * LEAF 3 - where the ground stops being soil and starts being meadow.
 *
 * world/grass.js draws a finite disc of blades: `uFade = (radius * 0.55,
 * radius)`, radius at most 72 m on foot, and its altitude clipmap multiplies
 * that by up to 8 as the player climbs. Past the reach there is nothing but
 * this material, so if the two do not agree on a colour the edge of the field
 * is a circle centred on the camera. From the air it is the most obvious
 * artefact in the scene.
 *
 * The blend therefore has to be finished by the time the last blade is gone - 
 * `MEADOW_END_FRAC` of grass's own live reach - and in any case by
 * MEADOW_MAX_END, because the window is deliberately NOT allowed to follow the
 * altitude clipmap out to 576 m. From two hundred metres up nobody resolves
 * bare earth between blades; what they see is the aggregate colour of the
 * field, and holding the transition at a walking-scale distance is what makes
 * the ground read as continuous meadow from the air while staying honest soil
 * underfoot.
 */
const MEADOW_START_FRAC = 0.40;
const MEADOW_END_FRAC = 1.0;
const MEADOW_MAX_END = 90;
/** Used only if world/grass.js is absent or has not published a fade yet. */
const MEADOW_FALLBACK_REACH = 72;

/**
 * Blade palette in world/grass.js, restated ONLY as the fallback for a scene
 * built without it. `_updateMeadowPalette` reads the live uniforms whenever the
 * system exists, so a retune over there cannot silently reopen the ring.
 */
const GRASS_PALETTE_FALLBACK = { base: '#3d4a30', tip: '#78855a', dry: '#9c9370' };

/**
 * Weights the blade palette is combined with, and the factor that turns "blade
 * albedo" into "what a field of them reads as from a distance".
 *
 * A blade is a near-vertical surface: under a high sun it catches a fraction of
 * the irradiance a flat ground fragment does, and the sward shadows itself
 * besides. Flat ground painted with the raw blade albedo therefore comes out
 * visibly brighter than the grass it is supposed to continue. 0.72 is the
 * factor that puts the two level under the default noon key; it is an art
 * number, but it is an art number applied to sampled colours rather than a
 * guessed colour.
 */
const MEADOW_MIX = { base: 0.30, tip: 0.52, dry: 0.18, gain: 0.72 };
const MEADOW_DRY_MIX = { base: 0.14, tip: 0.30, dry: 0.56, gain: 0.76 };

/**
 * RMS surface slope the FALLBACK micro-relief bake solves for, degrees. Matches
 * TERRAIN_SLOPE_RMS_DEG in core/textures.js, which owns the map that actually
 * ships, so a scene running on the fallback has the same relief rather than a
 * different one. Multiplied by uNormalScale (0.62) in the shader.
 */
const FALLBACK_SLOPE_RMS_DEG = 11;

/**
 * Per-tier LOD and shading budget. `maxChunks` is the ceiling on live chunk
 * geometries: once it is reached, a new chunk recycles the least recently used
 * one instead of allocating, which is what keeps streaming free of GC pauses.
 *
 * The numbers are the MEASURED steady-state working set plus ~12% headroom, not
 * estimates. Simulated at each tier from the spawn point and then walking 300 m
 * at sprint speed, the set peaks at 289 / 297 / 337 / 381 chunks. The old values
 * (200 / 250 / 300 / 320) sat below every one of them, which does not cap
 * anything - `_acquireGeometry` simply falls through to a fresh allocation when
 * the recycler finds nothing evictable - but it does make the recycler steal
 * geometry from chunks that are still wanted. MEDIUM did that 49 times in a
 * one-minute walk, and every steal is a full rebuild of ground that was already
 * correct. A ceiling that is genuinely above the working set costs 1-5 MB more
 * and turns the recycler back into the emergency valve it is meant to be.
 *
 * `meso` is ON at every tier, LOW included, and that is a deliberate reversal.
 *
 * It looks like the obvious thing to cut: it is a second tap of the macro
 * texture on every ground pixel. But look at what the `#else` branch in
 * FRAG_SURFACE actually does when it is off - `tMes = tMac` - and two things go
 * at once, not one:
 *
 *   1. The 3.4 m domain warp disappears entirely (`tp +=` lives inside
 *      `#ifdef TERRAIN_MESO`), so the 2 m near tile becomes a rigid lattice out
 *      to the detail fade. That warp is not a nicety; it is the ONLY thing in
 *      the file that puts neighbouring patches out of phase with each other,
 *      and the file header says so.
 *   2. Every splat driver collapses onto the 1024 m region field, so there is
 *      no patch-scale variation at all - the ground's coverage is constant over
 *      hundreds of metres. "Uniform ground with a rigid repeat on it" is the
 *      exact read this whole material was rewritten to remove.
 *
 * And LOW is the tier the target 780M actually settles on (core/textures.js's
 * own size policy says as much where it resolves the live tier). So the tier
 * most likely to be seen was the only one shipping the artefact.
 *
 * The cost is small and bounded, which is why the trade is wrong in the other
 * direction: terrainMacro is 128^2 RGBA = 64 kB with mips, sampled at 1/1024
 * and 1/128 of world, i.e. an almost-zero screen-space derivative and
 * anisotropy 2 - it is resident in L2 and never leaves it. One extra fetch plus
 * two madds over ~0.9 Mpix of ground at 1080p is on the order of 0.01-0.05 ms
 * on this part. LOW is still far cheaper than the tiers above it: 18 cells
 * against 32, a 55 m detail fade against 210, no puddles, no wet sheen, no
 * snow, no cloud detail tap and the cheap aerial path.
 */
const TIERS = {
  low: { depth: 6, cells: 18, range0: 96, detailFade: 55, aniso: 2, budgetMs: 0.8, maxChunks: 325, puddles: false, cloudDetail: false, snow: false, cheapAerial: true, meso: true, wetSheen: false },
  medium: { depth: 6, cells: 24, range0: 108, detailFade: 85, aniso: 4, budgetMs: 1.0, maxChunks: 335, puddles: true, cloudDetail: false, snow: true, cheapAerial: false, meso: true, wetSheen: true },
  high: { depth: 6, cells: 28, range0: 120, detailFade: 130, aniso: 8, budgetMs: 1.3, maxChunks: 380, puddles: true, cloudDetail: true, snow: true, cheapAerial: false, meso: true, wetSheen: true },
  ultra: { depth: 6, cells: 32, range0: 135, detailFade: 210, aniso: 16, budgetMs: 1.6, maxChunks: 430, puddles: true, cloudDetail: true, snow: true, cheapAerial: false, meso: true, wetSheen: true },
};

// ---------------------------------------------------------------------------
// Module scratch - nothing in the hot path allocates.
// ---------------------------------------------------------------------------

const _scratchNormal = new THREE.Vector3(0, 1, 0);
const _v3 = new THREE.Vector3();
const _col = new THREE.Color();
const _col2 = new THREE.Color();
// Blade palette scratch. Only touched by _updateMeadowPalette, which runs on
// link() and on a quality change, never per frame.
const _gBase = new THREE.Color();
const _gTip = new THREE.Color();
const _gDry = new THREE.Color();

const _byPriority = (a, b) => b.priority - a.priority;   // descending, so pop() is the nearest

// ---------------------------------------------------------------------------
// Tileable procedural noise - used only to bake textures.
//
// core/math.js owns the simplex and worley the world is built from, but neither
// can be made to wrap, and a ground texture that does not tile seamlessly is
// worse than no texture at all. These put a *periodic* lattice on top of
// math.js's hash primitives. The terrain itself never touches them.
// ---------------------------------------------------------------------------

/**
 * `hash2`/`hash3` in core/math.js document themselves as returning 0..1 but
 * actually return exactly uniform on [0, 0.5): the final `* 1274126177` is a
 * float64 multiply, so the product's top bits never survive the ToInt32 and the
 * sign bit is always clear. Measured over 200k samples the distribution is
 * flat, correct and half-scale - min 0.00000, max 0.50000, mean 0.2500, decile
 * counts 20/20/20/20/20/0/0/0/0/0.
 *
 * That file is shared and not ours to edit, so the correction lives here, where
 * it is exactly right: the distribution IS uniform, just on the wrong interval,
 * so one multiply restores it with no change to its spectral character or its
 * determinism.
 *
 * This is not cosmetic. Unscaled, an fbm over these lattices came out with mean
 * 0.23 and sd 0.05 instead of 0.5 and 0.10 - a field with a fifth of its
 * intended dynamic range, which is why the macro coverage masks below produced
 * near-uniform ground with no patchiness at all no matter how the splat weights
 * were tuned. And in the Worley lattice it confined every feature point to the
 * lower-left quarter of its cell, which is a regular grid wearing a noise
 * function's name - the "regularly spaced blobs" look, exactly.
 */
const HASH_SCALE = 2;
const phash2 = (x, y) => hash2(x, y) * HASH_SCALE;
const phash3 = (x, y, z) => hash3(x, y, z) * HASH_SCALE;

/** Periodic value noise. `k` = lattice cells across the [0,1) tile. */
function pvalue(x, y, k, seed) {
  const xi = Math.floor(x), yi = Math.floor(y);
  const xf = x - xi, yf = y - yi;
  const u = xf * xf * xf * (xf * (xf * 6 - 15) + 10);   // quintic: hides the lattice
  const v = yf * yf * yf * (yf * (yf * 6 - 15) + 10);
  const s = seed * 1319 + 7;
  const x0 = mod(xi, k) + s, x1 = mod(xi + 1, k) + s;
  const y0 = mod(yi, k) - s, y1 = mod(yi + 1, k) - s;
  const a = phash2(x0, y0), b = phash2(x1, y0);
  const c = phash2(x0, y1), d = phash2(x1, y1);
  const top = a + (b - a) * u;
  return top + (c + (d - c) * u - top) * v;
}

/** Periodic fbm over `pvalue`. `k` cells at octave 0, doubling per octave. */
function pfbm(s, t, k, octaves, seed, gain = 0.5) {
  let sum = 0, amp = 0.5, norm = 0, f = 1;
  for (let o = 0; o < octaves; o++) {
    sum += amp * pvalue(s * k * f, t * k * f, k * f, seed + o * 41);
    norm += amp;
    f *= 2;
    amp *= gain;
  }
  return sum / norm;
}

/**
 * Periodic cellular aggregate. `out.id` is a FLAT per-fragment value, `out.wall`
 * the narrowed F2-F1 crease between fragments, `out.d1` the F1 distance in cell
 * units for callers that want the fragment's extent.
 *
 * Not `1 - pworley()`. Inverted F1 is a radial cone centred on every feature
 * point, so it produces one identical round dome per lattice cell however hard
 * the points are jittered - bubble wrap, and the exact thing the user is
 * looking at when they call the ground "moor-like". Soil crumb is a mosaic of
 * flat-ish fragments at slightly different heights separated by thin voids, and
 * that is what these two channels are. The cone is not computed at all.
 */
function pcell(s, t, k, seed, out) {
  const x = s * k, y = t * k;
  const xi = Math.floor(x), yi = Math.floor(y);
  const xf = x - xi, yf = y - yi;
  let d1 = 8, d2 = 8, bestX = 0, bestY = 0;
  for (let j = -1; j <= 1; j++) {
    const cy = mod(yi + j, k);
    for (let i = -1; i <= 1; i++) {
      const cx = mod(xi + i, k);
      const dx = i + phash3(cx, cy, seed) - xf;
      const dy = j + phash3(cx, cy, seed + 71) - yf;
      const d = dx * dx + dy * dy;
      if (d < d1) { d2 = d1; d1 = d; bestX = cx; bestY = cy; }
      else if (d < d2) { d2 = d; }
    }
  }
  out.id = phash3(bestX, bestY, seed + 149);
  out.d1 = Math.sqrt(d1);
  // F2 - F1, cubed. The raw difference is a broad linear ramp and leaves a
  // paved-mosaic look; the void between two crumbs is a slot, not a valley.
  const w = 1 - clamp01((Math.sqrt(d2) - out.d1) * 1.9);
  out.wall = w * w * w;
  return out;
}

/** Periodic value noise with independent periods per axis. */
function pvalue2(x, y, kx, ky, seed) {
  const xi = Math.floor(x), yi = Math.floor(y);
  const xf = x - xi, yf = y - yi;
  const u = xf * xf * xf * (xf * (xf * 6 - 15) + 10);
  const v = yf * yf * yf * (yf * (yf * 6 - 15) + 10);
  const sd = seed * 1319 + 7;
  const x0 = mod(xi, kx) + sd, x1 = mod(xi + 1, kx) + sd;
  const y0 = mod(yi, ky) - sd, y1 = mod(yi + 1, ky) - sd;
  const a = phash2(x0, y0), b = phash2(x1, y0);
  const c = phash2(x0, y1), d = phash2(x1, y1);
  const top = a + (b - a) * u;
  return top + (c + (d - c) * u - top) * v;
}

/** Integer shears. Only integers keep the unit torus mapped onto itself. */
const FIBRE_SHEAR = [0, 1, -1, 2];

/**
 * Tangled dead-stem thatch without a sprite rasteriser.
 *
 * No noise field produces a strand: fbm gives blobs, Worley gives cells, and an
 * anisotropic field gives an axis-aligned corduroy that is instantly readable
 * as synthetic. What DOES produce strands is an anisotropic field seen through
 * an integer shear - the level sets of a field that varies fast across and slow
 * along become long thin curves at slope -n - and integer shears are the only
 * ones that preserve periodicity, so the tile still wraps exactly. Four of them
 * at four different slopes cross into a mat.
 *
 * `kAlong` sets strand length, `kAcross` their thickness.
 */
function pfibre(s, t, kAlong, kAcross, seed) {
  let sum = 0;
  for (let i = 0; i < 4; i++) {
    const n = FIBRE_SHEAR[i];
    const v = t + n * s;
    // Sharpened: a strand has an edge, and the raw field is a smooth swell.
    const f = pvalue2(s * kAlong, v * kAcross, kAlong, kAcross, seed + i * 37);
    sum += clamp01((f - 0.52) * 3.4);
  }
  return clamp01(sum * 0.62);
}

/** RMS central-difference gradient of a wrapping field - used to solve for slope. */
function gradRMSWrap(field, size) {
  let acc = 0;
  for (let y = 0; y < size; y++) {
    const yc = y * size, ym = mod(y - 1, size) * size, yp = mod(y + 1, size) * size;
    for (let x = 0; x < size; x++) {
      const dx = (field[yc + mod(x + 1, size)] - field[yc + mod(x - 1, size)]) * 0.5;
      const dy = (field[yp + x] - field[ym + x]) * 0.5;
      acc += dx * dx + dy * dy;
    }
  }
  return Math.sqrt(acc / (size * size));
}

/** Subtract the field's own low frequencies, so no landmark can repeat. */
function highpassWrap(field, size, radius, keep = 0.0) {
  const lo = new Float32Array(size * size);
  blurWrap(field, lo, size, radius);
  const k = 1 - keep;
  for (let i = 0; i < field.length; i++) field[i] = clamp01(field[i] + k * (0.5 - lo[i]));
}

/**
 * Shift and scale a field onto an exact mean and standard deviation, in place.
 *
 * Not decoration. `terrainHeightBlend` admits a layer only if it comes within
 * `sharpness` of the leader, so a channel's spread IS how decisively it competes
 * - and a raw `pfbm` over this lattice measures mean 0.46, sd 0.098, which is a
 * fifth of the dynamic range the splat weights above were tuned against. Hand
 * gains and biases were how that was patched before, and they are wrong the
 * moment an octave count or a seed changes: measured, the four macro channels
 * came out at means 0.38 / 0.54 / 0.35 / 0.57 against a single intended 0.50.
 * Solving for the moment rather than guessing at it is the only version of this
 * that survives a retune.
 */
function retargetWrap(field, mean, sd) {
  const n = field.length;
  let m = 0;
  for (let i = 0; i < n; i++) m += field[i];
  m /= n;
  let v = 0;
  for (let i = 0; i < n; i++) { const d = field[i] - m; v += d * d; }
  const s = Math.sqrt(v / n);
  const k = s > 1e-6 ? sd / s : 0;
  for (let i = 0; i < n; i++) field[i] = clamp01((field[i] - m) * k + mean);
  return field;
}

/** Wrapping separable box blur - turns a height field into cavity AO. */
function blurWrap(src, dst, size, radius) {
  const tmp = new Float32Array(size * size);
  const inv = 1 / (radius * 2 + 1);
  for (let y = 0; y < size; y++) {
    const row = y * size;
    let acc = 0;
    for (let i = -radius; i <= radius; i++) acc += src[row + mod(i, size)];
    for (let x = 0; x < size; x++) {
      tmp[row + x] = acc * inv;
      acc += src[row + mod(x + radius + 1, size)] - src[row + mod(x - radius, size)];
    }
  }
  for (let x = 0; x < size; x++) {
    let acc = 0;
    for (let i = -radius; i <= radius; i++) acc += tmp[mod(i, size) * size + x];
    for (let y = 0; y < size; y++) {
      dst[y * size + x] = acc * inv;
      acc += tmp[mod(y + radius + 1, size) * size + x] - tmp[mod(y - radius, size) * size + x];
    }
  }
}

// ---------------------------------------------------------------------------
// GLSL
// ---------------------------------------------------------------------------

const VERT_PARS = /* glsl */`
attribute vec3 aMorphOffset;
attribute vec3 aMorphNormal;
attribute float aCavity;
uniform vec2 uMorphRange;
varying vec3 vWorldPos;
varying vec3 vWorldNormal;
varying float vCavity;
varying float vViewDist;
`;

const VERT_NORMAL = /* glsl */`
// CDLOD geomorph. The factor comes from the *unmorphed* world position so two
// chunks sharing an edge always agree on it, and smoothstep gives it a zero
// derivative at both ends so the shading never creases as it engages.
vec3 tWorld0 = ( modelMatrix * vec4( position, 1.0 ) ).xyz;
float tRawDist = distance( tWorld0, cameraPosition );
float tMorph = clamp( ( tRawDist - uMorphRange.x ) / max( uMorphRange.y - uMorphRange.x, 1e-3 ), 0.0, 1.0 );
tMorph = tMorph * tMorph * ( 3.0 - 2.0 * tMorph );
vec3 objectNormal = normalize( mix( normal, aMorphNormal, tMorph ) );
`;

const VERT_POS = /* glsl */`
vec3 transformed = position + aMorphOffset * tMorph;
`;

const VERT_WORLD = /* glsl */`
vec4 tWorldPos4 = modelMatrix * vec4( transformed, 1.0 );
vWorldPos = tWorldPos4.xyz;
vWorldNormal = normalize( mat3( modelMatrix ) * objectNormal );
vCavity = aCavity * 2.0 - 1.0;
vViewDist = distance( tWorldPos4.xyz, cameraPosition );
`;

const FRAG_PARS = /* glsl */`
uniform sampler2D tDetail;
uniform sampler2D tNormalAO;
uniform sampler2D tMacro;
uniform sampler2D tCloud;
uniform vec2 uUVOrigin;
uniform vec4 uTiling;          // 1/tile: x near, y mid, z macro, w meso
uniform vec2 uDetailFade;
uniform vec2 uDetailWarp;      // x warp amplitude in metres, y meso lookup offset
uniform vec2 uHeightBand;      // x centre altitude, y 1/half relief range
// Earth, damp and dry. There is no green in this pair on purpose - see the
// file header. tDetail.r is the bare-soil structure channel.
uniform vec3 uColEarth;
uniform vec3 uColEarthDry;
// Dead thatch and stem litter, damp and sun-bleached. tDetail.g is two layers
// of stamped fibre, so this layer really is strands, not a green wash.
uniform vec3 uColThatch;
uniform vec3 uColThatchDry;
uniform vec3 uColMoss;
uniform vec3 uColPetal;
// LEAF 3: what the sward reads as at distance, sampled from world/grass.js.
uniform vec3 uColMeadow;
uniform vec3 uColMeadowDry;
uniform vec2 uMeadowFade;      // x blend start, y blend end, metres
uniform vec3 uTintCool;        // linear multipliers, not colours - see _buildUniforms
uniform vec3 uTintWarm;
uniform vec4 uLayerRough;
uniform float uNormalScale;
uniform float uWetness;
uniform float uPuddle;
uniform float uRipple;
uniform float uSnow;
uniform float uTime;
uniform vec3 uSunDir;
uniform vec3 uSkyColor;
uniform vec4 uCloudParams;     // x sun-shadow strength, y fallback coverage threshold,
                               // z shadow reference altitude, w 1/world scale
uniform float uCloudSky;       // how much a cloud also blocks *sky* light
uniform vec4 uCloudDrift;      // xy base drift, zw detail drift (fallback field only)
uniform vec4 uPetalLitter;     // treeX, treeZ, canopy radius, amount
varying vec3 vWorldPos;
varying vec3 vWorldNormal;
varying float vCavity;
varying float vViewDist;

// Height-aware splat blend. Linear weights make every transition a 50/50 mush;
// letting each layer's own height field compete inside a narrow band lets soil
// come up through the gaps in the thatch instead of averaging with it.
vec4 terrainHeightBlend( vec4 w, vec4 h, float sharpness ) {
  vec4 a = w + h * 0.5;
  float m = max( max( a.x, a.y ), max( a.z, a.w ) ) - sharpness;
  vec4 b = max( a - m, 0.0 );
  return b / max( b.x + b.y + b.z + b.w, 1e-4 );
}

// One expanding ring per lattice cell. Pure ALU, because the 780M has far more
// of that than bandwidth, and a scrolled normal map reads as chop, not rain.
// The lattice wraps every 64 cells so the hash keeps its precision out at the
// far edge of a six-kilometre world.
vec2 terrainRipple( vec2 p, float t, float cell, float seedOff ) {
  vec2 g = p * cell + seedOff;
  vec2 gi = mod( floor( g ), 64.0 );
  vec2 gf = fract( g ) - 0.5;
  float phase = fract( sin( dot( gi, vec2( 12.9898, 78.233 ) ) ) * 43758.5453 );
  float age = fract( t + phase );
  float r = length( gf ) + 1e-4;
  float d = r - age * 0.55;
  return ( gf / r ) * exp( -60.0 * d * d ) * ( 1.0 - age ) * ( 1.0 - smoothstep( 0.30, 0.5, r ) );
}
`;

const FRAG_SURFACE = /* glsl */`
// ---- Sakura Realm ground -------------------------------------------------
vec3 gGeoN = normalize( vWorldNormal );
// sin(slope) rather than 1 - N.y: on terrain this gentle the latter never
// leaves the first two percent of its range and every threshold on it is dead.
float gSlope = sqrt( max( 0.0, 1.0 - gGeoN.y * gGeoN.y ) );
float gCav = vCavity;
// Regional altitude: -1 in the deepest hollow of the world, +1 on the highest
// ridge. Cavity is *local* concavity and says nothing about where in the
// landscape you are; ridges being drier than valley floors is the other half.
float gAlt = clamp( ( vWorldPos.y - uHeightBand.x ) * uHeightBand.y, -1.0, 1.0 );

// The two low-frequency fields read RAW world coordinates. Six kilometres out
// that is a UV of 5.9 and 47 respectively - a thousand times the headroom the
// 2 m detail tile has - so they need no origin snap, and skipping it is not
// merely cheaper: no snap means no constraint that the macro tile divide it,
// and no risk of the whole distant ground re-tinting as the snap steps.
vec4 tMac = texture2D( tMacro, vWorldPos.xz * uTiling.z );
#ifdef TERRAIN_MESO
  // Same texture at an eighth the size, quarter-turned and offset. At 8:1 a
  // rotated copy can never line up with the original, so the patch scale stays
  // uncorrelated with the region scale instead of reading as one self-similar
  // pattern seen twice.
  vec4 tMes = texture2D( tMacro, vec2( -vWorldPos.z, vWorldPos.x ) * uTiling.w + uDetailWarp.y );
#else
  vec4 tMes = tMac;
#endif

vec2 tp = vWorldPos.xz - uUVOrigin;
#ifdef TERRAIN_MESO
  // Domain warp, not a scroll. A scroll moves the whole lattice together and
  // the repeat survives; a warp puts neighbouring patches out of phase with
  // each other, and that is what the eye reads as "not tiled".
  //
  // b and a rather than r and g because r and g are the two channels driving
  // the soil/grass competition, which is the strongest visual signal in the
  // frame - tying the warp to them would put a warp extremum on every patch
  // boundary and the two would rhyme. b and a still feed moss and mottle, but
  // far more weakly.
  tp += ( tMes.ba - 0.5 ) * uDetailWarp.x;
#endif

// The mid-scale copy is the same texture turned a right angle: a quarter turn
// keeps the lattice aligned to the UV snap, so the snap stays invisible while
// the correlation with the near copy is destroyed.
vec4 tDm = texture2D( tDetail, vec2( tp.y, -tp.x ) * uTiling.y );

// Chunk levels that live entirely past the detail-fade distance compile the
// near fetches, the branch and the wet-surface work away completely. Those
// levels cover most of the screen, so this is the single biggest fill-rate
// saving in the file - and because the first far level starts exactly where
// the fade has already reached zero, there is no seam between the two paths.
#ifdef TERRAIN_FAR
  const float tFade = 0.0;
  vec4 tDn = tDm;
  vec4 tNA = vec4( 0.5, 0.5, 1.0, 1.0 );
#else
  float tFade = 1.0 - smoothstep( uDetailFade.x, uDetailFade.y, vViewDist );
  vec4 tDn = tDm;
  vec4 tNA = vec4( 0.5, 0.5, 1.0, 1.0 );
  if ( tFade > 0.003 ) {
    tDn = texture2D( tDetail, tp * uTiling.x );
    tNA = texture2D( tNormalAO, tp * uTiling.x );
  }
#endif
vec4 tLay = mix( tDm, tDn, ( 0.45 + 0.35 * tMac.a ) * tFade );

// Everything that exists ONLY on the near path - puddles, the grazing wet
// sheen, and the specular they add - has to be identically zero by the distance
// at which the first TERRAIN_FAR chunk level starts, or there is a hard ring in
// the rain where wet ground turns dry across one LOD seam. _makeMaterial marks
// a level FAR when its nearest possible distance ( _ranges[level-1] ) is >=
// uDetailFade.y, so keying these terms to tFade - which is exactly zero at
// uDetailFade.y - is the only fade that is guaranteed continuous on every tier.
// The window this replaces ( 0.8 .. 1.6 * detailFade ) outlived the boundary by
// 37% on MEDIUM ( first far level 108 m, fade ended 136 m ) and 34% on ULTRA
// ( 270 m vs 336 m ); the wet sheen had no distance fade at all and simply
// stopped existing at the seam on all three tiers that carry it. The remap
// holds full strength out to ~0.81 * detailFade so the near field is unchanged.
float tNearK = smoothstep( 0.02, 0.35, tFade );

// Macro sets the region's character; meso breaks it into the patches you cross
// in half a minute. Summing two independent fields widens the distribution
// rather than averaging it flat, and that widening is exactly the difference
// between organic patchiness and uniform coverage with a gentle tint on top.
//
// The #else path (tMes = tMac) is a compile-only fallback and no tier selects
// it any more - see the note on meso in TIERS. It is not "the same at one
// scale": measured over 20 m patches, the drivers' local standard deviation
// falls from 0.077-0.121 to 0.021-0.035, i.e. the ground stops varying at all
// on the scale a walking player can see, and the domain warp below vanishes
// with it.
vec4 tK = clamp(
  tMac * vec4( 0.62, 0.62, 0.55, 0.55 ) + tMes * vec4( 0.58, 0.58, 0.62, 0.45 )
    - vec4( 0.10, 0.10, 0.14, 0.00 ), 0.0, 1.0 );
float kDry = tK.r, kLush = tK.g, kMoss = tK.b, kMot = tK.a;

// Slope response. The thresholds are set against the heightfield this file
// actually generates rather than guessed: sampled over four kilometres,
// sin(slope) is 0.050 at p25, 0.078 at the median, 0.172 at p95 and 0.328 at
// its absolute maximum. A band from 0.15 to 0.335 therefore puts bare earth
// only on the steepest few percent of the world, which is where erosion really
// does strip a grassland, and leaves every walkable slope grassed.
float gSlopeK = smoothstep( 0.15, 0.335, gSlope );

// LAYER SEMANTICS - read the file header before retuning these.
//   x  bare earth   (tDetail.r: grain, crumb mosaic, grit)
//   y  dead thatch  (tDetail.g: two layers of stamped stem litter)
//   z  moss         (tDetail.b: cushions, minority layer)
//   w  fallen petals(tDetail.a: stamped flakes)
// Nothing here is living green. The previous balance ran a green "grass" layer
// at 56% of the ground and it fought world/grass.js for the same surface; what
// came out was the flat blue-green the user reads as a bog. Earth leads now,
// thatch supports it, and the sward is left to the system that has silhouettes
// to draw it with.
//
// The weights are deliberately kept within about half a unit of each other:
// terrainHeightBlend only lets a layer through if it comes within "sharpness"
// of the leader, so a layer sitting a full unit above the others suppresses the
// rest completely no matter what the noise says.
//
// Simulated through this exact blend against the real baked detail and macro
// bytes (60k samples, level ground, cavity and altitude drawn from their real
// ranges): earth 58.5% / thatch 38.2% / moss 3.3% of the surface, each visible
// (share > 0.15) at 80% / 60% / 8% of samples. Moss being present on one sample
// in twelve is the point - it is a hollow-and-shade layer, and the balance it
// replaces had a green layer on 66% of the ground.
float wEarth  = 0.55 + gSlopeK * 1.35 + kDry * 0.85 - kLush * 0.30 + max( gAlt, 0.0 ) * 0.24;
float wThatch = 0.42 + kLush * 0.70 - gSlopeK * 0.55 + max( gCav, 0.0 ) * 0.18 - kDry * 0.14;
// Negative constant on purpose: on mean ground moss loses the blend outright
// and contributes exactly nothing. It has to be *earned* by a hollow, a shaded
// drainage line or a damp macro patch, which is where moss actually is.
float wMoss   = -0.04 + kMoss * 1.10 - gSlopeK * 1.00 - gCav * 0.52 - max( gAlt, 0.0 ) * 0.36;

float rTree = length( vWorldPos.xz - uPetalLitter.xy );
float pNear = smoothstep( uPetalLitter.z * 3.2, uPetalLitter.z * 0.5, rTree );
float pDrift = smoothstep( 0.15, -0.35, gCav ) * smoothstep( uPetalLitter.z * 9.0, uPetalLitter.z * 1.5, rTree );
// Litter blows into drifts. Letting the meso field thin and thicken it stops
// the canopy laying down an even pink ring, which is the tell every fallen-
// petal decal in every game has.
//
// tLay.a is deliberately NOT a factor here: it is this layer's HEIGHT field in
// terrainHeightBlend below, and folding it into the weight as well applied the
// same sparse blob mask twice, which put the product just under the blend's
// admission threshold everywhere - measured peak petal coverage 0.3%, i.e.
// uColPetal and the whole fourth splat channel were dead weight. Applied once,
// in the height slot where it belongs, and scaled so a full-litter flake comes
// out level with the earth/thatch leader while the gaps between flakes stay
// well under it. That calibration depends on the A channel peaking near 0.7 and
// essentially never reaching 1.0; both bakes are measured against it (0.741 in
// the local fallback, 0.687 in the map core/textures.js ships), and a mask that
// did reach 1.0 would hand petals 100% of the ground under the canopy - the
// flat pink decal this layer exists to avoid.
float wPetal = uPetalLitter.w * ( pNear + 0.45 * pDrift )
             * ( 0.42 + 0.92 * kMot ) * ( 1.0 - smoothstep( 0.13, 0.32, gSlope ) ) * 1.5;

vec4 tB = terrainHeightBlend( max( vec4( wEarth, wThatch, wMoss, wPetal ), 0.0 ), tLay, 0.22 );

// How dry this patch of ground is, 0 = saturated hollow, 1 = crusted ridge.
//
// This single axis carries most of the "that is soil" read, because it is the
// axis real earth actually varies along: damp umber in the shade of the sward
// and in every dip, pale grey-brown where sun and wind have crusted it. It is
// also a factor of 3.7 in linear luminance between the two ends, which makes it
// by far the strongest broad tonal drift on the ground - and broad tonal drift
// is what hides a 2 m detail tile across a kilometre of field.
//
// Four independent causes, so no one of them can stamp a pattern: the macro
// dryness field, regional altitude (ridges shed, valleys hold), local cavity,
// and the tree's own shade, which keeps the ground under a canopy damp all day.
// The canopy term is gated on uPetalLitter.w - which update() zeroes when there
// is no tree in the scene - so a treeless world does not get an unexplained
// damp patch at the origin. The gate saturates at 1 for every real litter value
// (the range is 0.34 .. 0.72), so it is a presence test, not a modulation.
float gDry = clamp( kDry * 0.90 + max( gAlt, 0.0 ) * 0.40
                  - smoothstep( 0.05, -0.40, gCav ) * 0.45
                  - pNear * min( 1.0, uPetalLitter.w * 4.0 ) * 0.22, 0.0, 1.0 );

vec3 cEarth = mix( uColEarth, uColEarthDry, gDry );
// Thatch bleaches faster than soil does and keeps a little of the mottle field
// so two neighbouring patches of litter are never the same straw.
vec3 cThatch = mix( uColThatch, uColThatchDry, clamp( gDry * 0.85 + kMot * 0.15, 0.0, 1.0 ) );

vec3 tAlb = tB.x * cEarth + tB.y * cThatch + tB.z * uColMoss + tB.w * uColPetal;

// ---- LEAF 3: converge on the meadow past the reach of the blade field -----
// Earth up close, where the blades part and you genuinely see the ground they
// grow out of; the colour of the sward itself out where there are no blades
// left to draw it. Without this the edge of world/grass.js's disc is a circle
// of bare brown around the camera, and from the air it is unmissable.
//
// tB.w holds the blend back over fallen petals: litter under the tree is a
// landmark and must not turn green because the player stepped back from it.
float gMead = smoothstep( uMeadowFade.x, uMeadowFade.y, vViewDist ) * ( 1.0 - tB.w * 0.75 );
vec3 cMeadow = mix( uColMeadow, uColMeadowDry,
                    clamp( kDry * 0.85 + max( gAlt, 0.0 ) * 0.50, 0.0, 1.0 ) );
tAlb = mix( tAlb, cMeadow, gMead );

// Micro value from the detail texture, weighted toward the near copy where it
// still exists. Applied AFTER the meadow mix so the far field keeps the mid-tap
// texture instead of going flat - a kilometre of one colour is its own tell.
//
// Widened from +/-20% to +/-27%. This is the only place clods, grit and root
// debris get VALUE contrast rather than merely hue, and the ground needs more
// of it than before for two reasons that both pull the same way: the layer
// blend hands the near tap only 45-80% of its weight (the rest is the 32 m
// copy, which is heavily magnified and therefore smooth at arm's length), and
// the micro-relief normal is now at 6.8 effective degrees, so the shading that
// used to carry the crumb structure no longer does.
tAlb *= mix( 1.0, mix( 0.73, 1.27, tLay.r * 0.6 + tLay.b * 0.4 ), 0.35 + 0.5 * tFade );
// ...and a slow drift over hundreds of metres. Widened from +/-8% to about
// +/-15%, and with more of the swing in hue than in value: with the palette now
// warm and the near tile only 2 m across, this and gDry together are what stop
// the eye finding the same square twice.
tAlb *= mix( uTintCool, uTintWarm, kMot );

float tRough = dot( tB, uLayerRough ) * mix( 0.94, 1.06, tLay.r );

#ifdef TERRAIN_SNOW
  float tSnow = uSnow * smoothstep( 0.45, 0.12, gSlope ) * ( 0.55 + 0.45 * tLay.b );
  tAlb = mix( tAlb, vec3( 0.78, 0.81, 0.88 ), tSnow );
  tRough = mix( tRough, 0.55, tSnow );
#endif

// Standing water gathers where the ground dips below its own smooth trend and
// the slope is near zero. vCavity is that residual, computed on the CPU from
// the analytic field, so it is identical at every LOD and cannot shimmer.
float tPud = 0.0;
#if defined( TERRAIN_PUDDLES ) && !defined( TERRAIN_FAR )
  // tNearK, not a window of its own: see the note where it is computed. It
  // reaches zero at exactly the distance the far path takes over.
  tPud = clamp( uPuddle
      * smoothstep( 0.05, 0.32, -gCav )
      * ( 1.0 - smoothstep( 0.035, 0.090, gSlope ) )
      * smoothstep( 0.30, 0.62, kMot )
      * tNearK, 0.0, 1.0 );
#endif

// Wet ground darkens because water fills the pores and light that would have
// scattered straight back out is instead trapped for another bounce. Porous
// soil therefore darkens far more than a moss mat does.
float tPorosity = 0.55 + 0.45 * tB.x;
tAlb *= mix( 1.0, 0.52, uWetness * tPorosity );
tAlb *= mix( 1.0, 0.34, tPud );
tRough = mix( tRough, 0.18, uWetness * 0.85 );
tRough = mix( tRough, 0.035, tPud );

// Roughening with distance is the cheapest specular-aliasing fix there is: the
// normal detail has faded out by then anyway, so nothing visible is lost.
//
// This has to come AFTER the wetness mixes, not before them. Applied first, a
// rain-soaked field was pulled back down to 0.18 at every distance and the
// whole far half of the screen turned into a field of shimmering specular
// fireflies. Wetness holds part of the roughening back so the sheen survives at
// range under a low sun, which is the one shot rain is actually for.
tRough = min( 1.0, tRough + 0.13 * ( 1.0 - tFade ) * ( 1.0 - 0.45 * uWetness ) );

// Tangent frame straight from the geometric normal - this saves a tangent
// attribute on every vertex in the world, and the terrain never comes within
// sixty degrees of vertical so crossing against +Z cannot degenerate.
//
// The frame MUST come out T = +X, B = +Z on level ground. tNormalAO is sampled
// at tp * uTiling.x, i.e. s runs along world +X and t along world +Z, and the
// bake stores -dh/ds in x and -dh/dt in y. cross( +Z, N ) gives T = -X, which
// mirrors the whole micro-relief along one axis: every clod is then lit from
// the wrong side in X and right in Z, which is the "plasticky, subtly wrong"
// ground read, and it sheared the rain ripples the same way. Swapping the
// operands gives T = +X, and B = cross( T, N ) is then +Z rather than -Z.
vec3 tT = normalize( cross( gGeoN, vec3( 0.0, 0.0, 1.0 ) ) );
vec3 tBt = cross( tT, gGeoN );
vec3 tMapN = tNA.xyz * 2.0 - 1.0;
// Water fills the micro-relief long before it pools: rain flattens the surface
// visibly earlier than it puddles it. The meso term is free variety - one madd
// buys relief that is coarse and stony in some patches and soft and matted in
// others, at the one scale the near detail texture cannot vary over.
tMapN.xy *= uNormalScale * tFade * ( 0.65 + 0.72 * kMot )
          * ( 1.0 - tPud * 0.92 ) * ( 1.0 - uWetness * 0.28 );
#if defined( TERRAIN_PUDDLES ) && !defined( TERRAIN_FAR )
  float tRipFade = uRipple * tPud * ( 1.0 - smoothstep( 12.0, 40.0, vViewDist ) );
  if ( tRipFade > 0.002 ) {
    vec2 rp = terrainRipple( vWorldPos.xz, uTime * 1.10, 1.9, 0.0 )
            + terrainRipple( vWorldPos.xz, uTime * 0.83 + 0.37, 3.1, 11.7 ) * 0.7;
    tMapN.xy += rp * tRipFade * 0.9;
  }
#endif
vec3 gNrm = normalize( tT * tMapN.x + tBt * tMapN.y + gGeoN * max( tMapN.z, 0.05 ) );

float gAO = mix( 1.0, tNA.a, 0.75 * tFade );
gAO *= mix( 1.0, 0.80, smoothstep( 0.0, -0.45, gCav ) );

// Cloud shadows - the single strongest realism cue on open ground, because it
// is the only thing in the frame that proves the sky is a volume and not a
// backdrop. Two transmittances come out of here: one for the sun, one for the
// sky, because a cloud overhead dims the ambient too, just far more softly.
float gCloudSun = 1.0;
float gCloudSky = 1.0;
#ifdef TERRAIN_CLOUD_MAP
  // sky/clouds.js raymarches a real transmittance map: R is how much SUN light
  // reaches the reference plane, G how much ZENITH light does. Both are already
  // premultiplied by its own strength and low-sun fade, so all we add is an
  // artistic scale and the murk term.
  //
  // The map is rendered at one reference ground height, so walk our own
  // altitude back down the sun ray to that plane before looking up. Without it
  // a hillside forty metres above the valley wears the shadow of the cloud over
  // the valley, and the error grows without bound as the sun drops.
  vec3 cw = vWorldPos;
  cw.xz -= uSunDir.xz * ( ( vWorldPos.y - uCloudParams.z ) / max( uSunDir.y, 0.15 ) );
  vec4 cp = uCloudMatrix * vec4( cw, 1.0 );
  vec2 cuv = cp.xy / max( cp.w, 1e-4 );
  // The map clamps at its border, which would otherwise smear the edge texel
  // into an infinite streak across everything beyond its extent.
  vec2 cEdge2 = abs( cuv - 0.5 );
  float cEdge = 1.0 - smoothstep( 0.44, 0.499, max( cEdge2.x, cEdge2.y ) );
  vec2 cT = texture2D( tCloudMap, cuv ).rg;
  gCloudSun = mix( 1.0, cT.r, uCloudParams.x * cEdge );
  gCloudSky = mix( 1.0, cT.g, uCloudSky * cEdge );
#else
  // Fallback field: this texture stores COVERAGE, not transmittance, so it has
  // to be inverted. Projecting along the sun onto the deck is the difference
  // between a shadow that sits under its cloud and one that slides off it as
  // the sun drops.
  if ( uCloudParams.x > 0.001 ) {
    vec2 cq = vWorldPos.xz + uSunDir.xz * ( ( uCloudParams.z - vWorldPos.y ) / max( uSunDir.y, 0.15 ) );
    cq *= uCloudParams.w;
    float cov = texture2D( tCloud, cq + uCloudDrift.xy ).r;
    #ifdef TERRAIN_CLOUD_DETAIL
      cov = cov * 0.74 + texture2D( tCloud, cq * 3.17 + uCloudDrift.zw ).g * 0.26;
    #else
      cov = cov * 0.74 + 0.13;
    #endif
    float cShade = uCloudParams.x * smoothstep( uCloudParams.y, uCloudParams.y + 0.26, cov );
    gCloudSun = 1.0 - cShade;
    gCloudSky = 1.0 - cShade * uCloudSky;
  }
#endif

// View vector, shared by both grazing-angle terms - one normalize between them,
// and the whole block compiles out on tiers that have neither.
float gPuddleFresnel = 0.0;
float gWetSheen = 0.0;
#if !defined( TERRAIN_FAR ) && ( defined( TERRAIN_PUDDLES ) || defined( TERRAIN_WET_SHEEN ) )
  vec3 tV = normalize( cameraPosition - vWorldPos );
  float tNdV = 1.0 - clamp( dot( tV, gNrm ), 0.0, 1.0 );
  #ifdef TERRAIN_PUDDLES
    gPuddleFresnel = tPud * ( tNdV * tNdV ) * ( tNdV * tNdV );
  #endif
  #ifdef TERRAIN_WET_SHEEN
    // A wet field is not a mirror, but it does go glossy at grazing angles long
    // before it puddles - the thin film on every blade and clod turns the whole
    // surface into one weak fresnel. Fifth power, unlike the puddle's fourth,
    // so it stays confined to the true grazing band and never lifts the ground
    // you are standing on. tNearK because this term does not exist on the far
    // path at all - without it the sheen ended in a hard ring at the first far
    // level, at full strength, exactly when it is raining.
    gWetSheen = uWetness * ( 1.0 - tPud ) * tNdV * ( tNdV * tNdV ) * ( tNdV * tNdV ) * 0.55 * tNearK;
  #endif
#endif

diffuseColor.rgb *= tAlb;
`;

/**
 * Fallback aerial perspective. Only compiled when weather/atmosfx.js is absent
 * or declines to adopt the material - when it is present it owns the whole
 * scene's atmosphere and this is dead code, which is the correct outcome: one
 * model, one set of numbers, no two systems disagreeing about the same air.
 */
const FRAG_AERIAL_PARS = /* glsl */`
// uGround*, not uAerial*: weather/atmosfx.js publishes uAerialHaze and
// uAerialMist with the same shape but different semantics in .w, and its
// injectUniforms() only fills entries that are still undefined. Sharing a name
// would have let our values silently drive its shader on this one material - 
// the kind of collision that shows up as "the terrain's fog looks slightly
// wrong" and takes an afternoon to find.
uniform vec4 uGroundHaze;      // x sigma  y 1/scale height  z base altitude  w unused
uniform vec4 uGroundMist;      // x sigma  y 1/scale height  z base altitude  w unused
uniform vec3 uFogColor;        // horizon radiance
uniform vec3 uFogZenith;       // zenith radiance
uniform vec3 uMistColor;       // near-ground inscatter, warmer and brighter
uniform vec3 uFogSunColor;
uniform float uMieG;
uniform vec2 uFarFade;

/**
 * Optical depth of an exponentially stratified medium along a ray:
 *
 *   od = INT sigma * e^(-(y-base)/H) ds
 *      = sigma * H * ( e^(-h0/H) - e^(-h1/H) ) / dirY
 *
 * Written as that difference of endpoint densities rather than the algebraically
 * equal sigma*d*(1-e^-x)/x, because in float the second form underflows
 * sigma*d to zero and overflows (1-e^-x)/x as soon as the camera climbs a few
 * scale heights - which is precisely what flight mode does. This form is
 * bounded everywhere and degrades to its own first-order series for a level
 * ray, where the division is singular. Same derivation weather/atmosfx.js uses,
 * deliberately: the two must agree wherever they overlap.
 */
float terrainOpticalDepth( const in float sigma, const in float invH, const in float baseY,
                           const in float camY, const in float dirY, const in float dist ) {
  float h0 = max( camY - baseY, 0.0 );
  float e0 = exp( - invH * h0 );
  float x = invH * dirY * dist;
  if ( abs( x ) < 1e-3 ) return min( sigma * e0 * dist * ( 1.0 - 0.5 * x ), 32.0 );
  float h1 = max( camY + dirY * dist - baseY, 0.0 );
  // dist / x is 1 / ( invH * dirY ); written this way so both branches share x.
  return min( sigma * ( e0 - exp( - invH * h1 ) ) * ( dist / x ), 32.0 );
}
`;

const FRAG_AERIAL = /* glsl */`
{
  vec3 aV = vWorldPos - cameraPosition;
  float aD = max( length( aV ), 1e-3 );
  aV /= aD;

  // Two media. The tall haze is the real aerial perspective - it is what
  // dissolves the treeline into the sky over a kilometre. The low mist is what
  // sits in the grass at dawn and is gone by the time you look up.
  float odHaze = terrainOpticalDepth( uGroundHaze.x, uGroundHaze.y, uGroundHaze.z,
                                      cameraPosition.y, aV.y, aD );
  float odMist = terrainOpticalDepth( uGroundMist.x, uGroundMist.y, uGroundMist.z,
                                      cameraPosition.y, aV.y, aD );
  float od = odHaze + odMist;

  // Ground within a few metres of the camera is a large share of the shaded
  // pixels in a field, and for those the whole thing is a no-op to four decimal
  // places. Bail before the phase function and the three colour mixes.
  if ( od > 0.0025 || aD > uFarFade.x ) {
    // Beer-Lambert in the TRUE optical depth. This used to be three's FogExp2
    // curve, 1 - exp(-(sigma*d)^2): squaring the exponent stops sigma being an
    // extinction coefficient at all, so it could not be reconciled with the
    // visibility weather.js publishes and the falloff had no physical shape to
    // check against. With sigma = 0.0022/m this is 0.6% at three metres, 18% at
    // a hundred, 91% at 1.2 km - near field untouched, horizon dissolved.
    float aExt = 1.0 - exp( - od );
    // The outermost ring of chunks must never meet the sky as a visible edge.
    // At any sane sigma the term above has saturated long before this engages;
    // it exists for the pathological case of near-zero fog.
    aExt = max( aExt, smoothstep( uFarFade.x, uFarFade.y, aD ) );

    // Inscattered radiance has to be the colour of the sky in the view
    // direction, not one flat fog colour, or ground meets sky along a line.
    // sqrt rather than pow( y, 0.62 ): almost every ground fragment looks
    // downward, so the argument is exactly zero, and pow(0, k) is one of the
    // few places the GLSL spec still permits a driver to hand back a NaN.
    vec3 aSky = mix( uFogColor, uFogZenith, sqrt( clamp( aV.y, 0.0, 1.0 ) ) );
    #ifndef TERRAIN_CHEAP_AERIAL
      float aCos = dot( aV, uSunDir );
      float g = uMieG;
      float aHG = ( 1.0 - g * g ) / pow( max( 1.0 + g * g - 2.0 * g * aCos, 1e-4 ), 1.5 );
      aSky += uFogSunColor * min( aHG * 0.22, 2.0 );
    #endif
    // Weight the two inscatter colours by their share of the optical depth, so
    // a long ground-hugging ray stays mist-coloured near the camera instead of
    // taking the zenith's tint all the way down to your boots.
    vec3 aIn = ( odHaze * aSky + odMist * uMistColor ) / max( od, 1e-4 );

    // Applied here - injected after <opaque_fragment>, before
    // <tonemapping_fragment> - so it mixes finished linear radiance toward a
    // physical sky radiance. Any earlier and it would flatten the albedo before
    // the lights ever saw it, which is what turns a scene monochrome.
    gl_FragColor.rgb = mix( gl_FragColor.rgb, aIn, aExt );
  }
}
`;

// ---------------------------------------------------------------------------

export class Terrain {
  constructor(ctx) {
    this.ctx = ctx;
    this.state = ctx.state;

    // Hand the shared TextureFactory the live quality object. main.js builds it
    // as `new TextureFactory({ renderer })` - no state, no quality, no bus - and
    // it is not a member of ctx.systems, so nothing else ever tells it what tier
    // the QualityManager settled on one line earlier. Left alone it pins itself
    // to its 'high' default and bakes HIGH-sized maps on every tier: measured
    // over the five suites a real boot requests, 17.8 MB instead of the 6.8 MB
    // LOW asks for, on a part with no dedicated VRAM.
    //
    // Here rather than in init() because every bake is lazy and this is the
    // earliest hook that is guaranteed to precede all of them: main.js runs all
    // seventeen constructors before the first init(), and no constructor bakes.
    // The call is idempotent, never overrides an explicitly-configured source,
    // and is a no-op once anything has been baked - see `useLiveQuality`.
    ctx.textures?.useLiveQuality?.({ state: ctx.state, quality: ctx.quality });

    this.noise = createNoise(TERRAIN_SEED);
    this._n2 = this.noise.noise2D;   // hoisted: this is called ten times per sample

    // `_sample` leaves the smooth (large + medium) component here so callers
    // get local concavity for free; cavity = height - trend.
    this._mid = 0;
    this._hRef = 0;
    this._hRef = this._sample(0, 0);   // put the tree's clearing at y = 0

    this.group = new THREE.Group();
    this.group.name = 'terrain';
    this.group.matrixAutoUpdate = false;
    ctx.scene.add(this.group);

    this._roots = new Map();
    this._queue = [];
    this._selected = [];
    this._live = [];
    this._geoPool = [];
    /**
     * Meshes are pooled for the same reason geometries are. A chunk build is a
     * hot-path event - streaming while walking runs several a frame - and
     * `new THREE.Mesh()` is not one allocation but about a dozen: position,
     * scale, quaternion, rotation, up, matrix, matrixWorld, modelViewMatrix,
     * normalMatrix, layers, userData. Recycling the wrapper costs one `set` and
     * one `updateMatrix`, which the old path paid anyway.
     */
    this._meshPool = [];
    /**
     * Per-level mesh names, built once. The old `terrain-l${node.level}` was a
     * template literal evaluated inside `_buildNode`, i.e. a string allocation
     * on every chunk build for a debug label nothing reads per frame.
     */
    this._levelNames = [];
    this._geomCount = 0;
    this._ranges = [];
    this._materials = [];
    this._frame = 0;
    this._pruneTimer = 0;
    this._chunkMs = 1.0;          // rolling cost of one chunk build, milliseconds
    this._puddleAmount = 0;
    this._snowCover = 0;
    this._wireframe = false;
    /** Altitude the fallback mist layer sits on; follows the ground, damped. */
    this._mistBase = 0;
    this._mistBaseInit = false;

    // Conservative vertical bounds until init() measures the real ones. Only
    // used to size unbuilt nodes for the LOD distance test.
    this._worldMinY = -70;
    this._worldMaxY = 40;

    this._tier = TIERS[this.state.quality.tier] || TIERS.high;
    /** Guaranteed radius of real ground around the camera, in metres. */
    this.viewDistance = ROOT_SIZE;

    this._uniforms = null;
    this._tex = { detail: null, normalAO: null, macro: null, cloud: null };
    /** Only the textures we baked ourselves - see _texture(). */
    this._ownedTex = [];
    this._cloudMap = null;
    this._cloudMatrix = null;
    this._clouds = null;          // sibling system, re-polled every frame
    this._cloudPushed = false;    // true once someone called setCloudShadowMap()
    /** Frames elapsed with no cloud shadow map adopted; see _checkCloudShadow. */
    this._cloudWaited = 0;
    this._cloudWarned = false;
    this._externalAerial = false;
    this._tree = null;
    this._grass = null;
    /**
     * Damped end of the earth -> meadow blend. Damped rather than snapped
     * because grass.js's altitude clipmap doubles its reach in one frame as the
     * player crosses 18 m, and an instant jump in the window would recolour the
     * whole middle distance on that frame.
     */
    this._meadowEnd = MEADOW_FALLBACK_REACH;
    /** See the use site in update(). Driven by world/grass.js's sward mode. */
    this._meadowNear = 1;
  }

  /**
   * How close to the camera the ground may start reading as meadow rather than
   * soil. 1 keeps the authored far-field blend; smaller values pull it in.
   * Called by world/grass.js when the sward is cut short.
   */
  setMeadowNear(factor) {
    this._meadowNear = clamp(factor, 0.02, 1);
  }

  // =========================================================================
  // LEAF 1 - analytic heightfield
  // =========================================================================

  /**
   * Raw field before the clearing is flattened. Exactly ten simplex taps, all
   * unrolled - the fbm helpers cost ~35% in closure and loop overhead at this
   * call rate. Writes `this._mid` with the large-scale trend.
   */
  _sample(x, z) {
    const n2 = this._n2;

    // Warping the sample coordinates rather than the output is what stops fbm
    // hills from reading as a field of identical round humps.
    const wxf = x * WARP_F, wzf = z * WARP_F;
    const px = x + n2(wxf + 11.3, wzf - 4.7) * WARP_A;
    const pz = z + n2(wxf - 6.1, wzf + 8.9) * WARP_A;

    // Relief envelope, 0.30..1.15. Stretches of near-flat plain between the
    // rolling ones is what makes an endless field read as landscape.
    const env = 0.30 + 0.85 * (0.5 + 0.5 * n2(x * ENV_F + 31.7, z * ENV_F - 19.4));

    let big = (0.5 * n2(px * BIG_F, pz * BIG_F)
      + 0.26 * n2(px * BIG_F2, pz * BIG_F2)
      + 0.1352 * n2(px * BIG_F3, pz * BIG_F3)) * (1 / 0.8952);
    // Broaden the crowns and the valley floors, steepen the flanks: an eroded
    // profile. Monotonic on [-1,1] (derivative 1.30 - 0.90 f^2 > 0) so it can
    // never fold the surface back on itself.
    big = big * (1.30 - 0.30 * big * big);

    let h = big * BIG_A * env;
    h += (0.5 * n2(px * MID_F + 5.5, pz * MID_F - 2.5)
      + 0.25 * n2(px * MID_F2 + 5.5, pz * MID_F2 - 2.5)) * (1 / 0.75) * MID_A * (0.45 + 0.55 * env);

    const mid = h;
    h += n2(px * SML_F - 17.2, pz * SML_F + 4.4) * SML_A;
    h += n2(x * MIC_F + 63.1, z * MIC_F - 27.9) * MIC_A;

    this._mid = mid - this._hRef;
    return h - this._hRef;
  }

  /**
   * Ground height at a world position. Synchronous, allocation-free, ~0.7 us.
   * Leaves `this._mid` set to the smooth trend at the same point.
   */
  getHeight(x, z) {
    const h = this._sample(x, z);
    const rSq = x * x + z * z;
    if (rSq >= PAD_LIMIT_SQ) return h;

    // Perturbing the radius keeps the clearing from reading as a stamped disc.
    // Beyond PAD_LIMIT the perturbation can no longer pull rEff under PAD_OUT,
    // so the branch is exactly continuous at its own edge.
    const r = Math.sqrt(rSq);
    const rEff = r * (1 + 0.17 * this._n2(x * 0.017 + 2.1, z * 0.017 - 5.3));
    const m = smootherstep(PAD_IN, PAD_OUT, rEff);
    const pad = PAD_RISE * (1 - smootherstep(0, PAD_IN * 1.6, rEff));
    this._mid = pad + (this._mid - pad) * m;
    return pad + (h - pad) * m;
  }

  /**
   * Surface normal by central difference. Four height evaluations; epsilon is
   * deliberately wider than the micro octave so walking does not jitter.
   * `out` is optional - omit it and you get a shared scratch vector back, so do
   * not hold on to the result across another call.
   */
  getNormal(x, z, out) {
    const e = 0.6, inv2e = 1 / (2 * e);
    const nx = -(this.getHeight(x + e, z) - this.getHeight(x - e, z)) * inv2e;
    const nz = -(this.getHeight(x, z + e) - this.getHeight(x, z - e)) * inv2e;
    const inv = 1 / Math.sqrt(nx * nx + 1 + nz * nz);
    const v = out || _scratchNormal;
    if (v.set) v.set(nx * inv, inv, nz * inv);
    else { v.x = nx * inv; v.y = inv; v.z = nz * inv; }
    return v;
  }

  /** Slope in radians. */
  getSlope(x, z) {
    const n = this.getNormal(x, z, _v3);
    return Math.acos(clamp(n.y, -1, 1));
  }

  /**
   * Metres below (negative) or above (positive) the local smooth trend. Hollows
   * hold water and leaf litter; scatter and grass can use it the same way the
   * ground shader does.
   */
  getCavity(x, z) {
    return this.getHeight(x, z) - this._mid;
  }

  /**
   * Batch height fill over an axis-aligned grid - for systems that need
   * thousands of samples at once (grass chunk baking) and would rather not pay
   * the per-call overhead. `out` must hold count*count floats; it is returned.
   */
  sampleGrid(x0, z0, step, count, out) {
    let k = 0;
    for (let j = 0; j < count; j++) {
      const z = z0 + j * step;
      for (let i = 0; i < count; i++) out[k++] = this.getHeight(x0 + i * step, z);
    }
    return out;
  }

  // =========================================================================
  // LEAF 3 (bake) - procedural textures
  // =========================================================================

  /**
   * Textures come from the shared TextureFactory when there is one, and it owns
   * their lifetime. When there is not - or it refuses - we bake locally and then
   * we own them, so they are recorded here and freed in dispose(). Without the
   * record, an HMR cycle or a quality rebuild leaked four 256 kB textures every
   * time, because dispose() assumed the factory always held them.
   */
  _texture(name, gen) {
    const f = this.ctx.textures;
    if (f && typeof f.get === 'function') {
      try {
        const t = f.get(name, gen);
        if (t && t.isTexture) return t;
      } catch (e) { /* factory refused; bake locally */ }
    }
    const own = gen();
    this._ownedTex.push(own);
    return own;
  }

  _finishTexture(tex, aniso) {
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.colorSpace = THREE.NoColorSpace;   // every channel is data, not colour
    tex.magFilter = THREE.LinearFilter;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    tex.generateMipmaps = true;
    tex.anisotropy = aniso;
    tex.needsUpdate = true;
    return tex;
  }

  /**
   * Feature-size helper for the fallback bakes. Every frequency below is
   * authored as a physical size in millimetres over TILE_NEAR, so "12 mm grain"
   * stays 12 mm if the resolution or the tile ever changes - getting that wrong
   * by a factor of five is how procedural ground ends up with 330 mm soil clods.
   * Capped at ~3.2 texels a cell: past that the base mip aliases against its own
   * lattice and the ground sparkles.
   */
  _cellsFor(size) {
    const nyq = Math.max(4, Math.round(size / 3.2));
    return (mm) => clamp(Math.round((TILE_NEAR * 1000) / mm), 2, nyq);
  }

  /**
   * FALLBACK detail texture. core/textures.js is authoritative for this map (see
   * FACTORY_OWNED there) and its `_buildTerrainSurface` is what actually ships;
   * this runs only if that built-in throws, or if the Terrain is constructed
   * without a TextureFactory at all.
   *
   * It is still written properly, because "only reachable when something else
   * broke" is not a licence to ship the bug this project spent a day on. The
   * version this replaces built 52% of its micro-relief out of inverted-F1
   * Worley - a radial cone per feature point, i.e. one identical round dome per
   * lattice cell - and drove its cavity AO from the same field, which put a
   * bright disc on the top of every dome. That is the regular carpet of pale
   * blobs in the captures, and shrinking the tile only shrinks the blobs.
   *
   * Nothing below computes a cone. Granularity comes from `pcell`, which returns
   * flat fragments and narrow voids; strands come from `pfibre`; and every
   * channel is high-passed, because a tile repeats visibly only when the eye can
   * find a landmark in it, and landmarks are low-frequency by definition.
   *
   * Channels match the material: R bare earth, G dead thatch, B moss, A petals.
   */
  _bakeDetail(size) {
    const N = size * size;
    const inv = 1 / size;
    const cells = this._cellsFor(size);
    // Octave ladders double, so a two-octave field at base k reaches 2k and that
    // is what has to stay under the Nyquist cap - hence the mid and coarse bases
    // sit well below the fine one rather than above it.
    const kFine = cells(11);
    const kMid = Math.max(3, Math.round(kFine * 0.46));
    const kCoarse = cells(58);
    const kCrumbF = cells(38), kCrumbC = cells(95);
    const kMoss = cells(85), kPetal = cells(26);
    const kMossGrain = Math.max(2, Math.round(kFine * 0.62));
    const kFibLen = cells(90), kFibWide = cells(6);
    const kFibLen2 = cells(38);

    const chR = new Float32Array(N);
    const chG = new Float32Array(N);
    const chB = new Float32Array(N);
    const chA = new Float32Array(N);
    const cF = { id: 0, wall: 0, d1: 0 };
    const cC = { id: 0, wall: 0, d1: 0 };
    const cP = { id: 0, wall: 0, d1: 0 };

    for (let y = 0; y < size; y++) {
      const t = y * inv;
      for (let x = 0; x < size; x++) {
        const s = x * inv;
        const i = y * size + x;

        const gF = pvalue(s * kFine, t * kFine, kFine, 21);
        const gM = pfbm(s, t, kMid, 2, 23, 0.5);
        const gC = pfbm(s, t, kCoarse, 2, 27, 0.52);
        pcell(s, t, kCrumbF, 9, cF);
        pcell(s, t, kCrumbC, 3, cC);
        // Aggregation is patchy in reality: soil is crumbed where roots and
        // worms worked it and smooth where they did not. Uniformly strong
        // structure is the second-order version of a uniform lattice.
        const mat = pfbm(s, t, cells(190), 3, 31, 0.55);
        const crumbK = 0.45 + 1.15 * mat;

        const fib = pfibre(s, t, kFibLen, kFibWide, 401);
        const fibF = pfibre(s, t, kFibLen2, kFibWide, 409);

        // R - bare earth: grain, the fragment mosaic, and the voids between.
        chR[i] = 0.5
          + (gF - 0.5) * 0.26 + (gM - 0.5) * 0.22 + (gC - 0.5) * 0.20
          + (cC.id - 0.5) * 0.46 * crumbK + (cF.id - 0.5) * 0.26 * crumbK
          - cC.wall * 0.34 - cF.wall * 0.24
          - fib * 0.12;

        // G - dead thatch: two crossed strand layers lying in their mat.
        chG[i] = 0.34 + mat * 0.15 + fib * 0.50 + fibF * 0.30 + (gF - 0.5) * 0.16;

        // B - moss: soft cushions where the mat is thick, dominated within each
        // cushion by a fine crumb so it never reads as a tray of cobbles.
        const cush = pfbm(s, t, kMoss, 2, 63, 0.55);
        const patch = pfbm(s, t, cells(160), 3, 67, 0.55);
        chB[i] = 0.5 + (cush - 0.5) * 0.50 * (0.30 + 1.05 * patch)
          + (pvalue(s * kMossGrain, t * kMossGrain, kMossGrain, 65) - 0.5) * 0.24
          + (gF - 0.5) * 0.10;

        // A - fallen petals. A sparse mask, so it is NOT high-passed and NOT
        // recentred on 0.5: doing either lifts its empty background into the
        // splat blend and lays a flat pink wash under the canopy.
        //
        // A flat-topped cap, not the inverted cone the rest of this file goes
        // out of its way to avoid: a petal really is a small rounded object and
        // at 26 mm it is three texels across, so what matters is that it has a
        // top rather than a point.
        //
        // Acceptance is per-FLAKE (the cell hash, offset by a 22 cm drift field)
        // rather than a value multiplier, so litter reads as whole petals lying
        // in drifts rather than as a mask fading in and out across each one. The
        // peak is held near 0.74 to match the distribution core/textures.js
        // produces, which is what terrain's litter weight of 1.5 is calibrated
        // against - a mask that reaches 1.0 turns the ground under the canopy
        // into a solid pink decal.
        const drift = pfbm(s, t, cells(220), 3, 71, 0.58);
        pcell(s, t, kPetal, 101, cP);
        const cap = smoothstep(0.62, 0.28, cP.d1);
        const accept = smoothstep(0.26, 0.48, drift + (cP.id - 0.5) * 0.50);
        chA[i] = clamp01(cap * accept * (0.42 + 0.32 * cP.id));
      }
    }

    const hp = Math.max(2, Math.round(size / 7));
    highpassWrap(chR, size, hp);
    highpassWrap(chG, size, hp);
    highpassWrap(chB, size, hp);
    // The distributions core/textures.js measures for the map that actually
    // ships, so the splat weights above land in the same place on either path.
    retargetWrap(chR, 0.48, 0.18);
    retargetWrap(chG, 0.46, 0.19);
    retargetWrap(chB, 0.47, 0.21);

    const d = new Uint8Array(N * 4);
    for (let i = 0; i < N; i++) {
      const o = i * 4;
      d[o] = Math.round(clamp01(chR[i]) * 255);
      d[o + 1] = Math.round(clamp01(chG[i]) * 255);
      d[o + 2] = Math.round(clamp01(chB[i]) * 255);
      d[o + 3] = Math.round(clamp01(chA[i]) * 255);
    }
    return this._finishTexture(new THREE.DataTexture(d, size, size, THREE.RGBAFormat), 4);
  }

  /**
   * FALLBACK micro-relief normal + cavity AO. Same standing as `_bakeDetail`.
   *
   * Two things here are not negotiable and were both wrong before. First, no
   * cones: the height stack is grain, flat fragments and narrow voids, so the
   * relief has no repeating silhouette. Second, the amplitude is SOLVED rather
   * than dialled - `normal` shows the gradient, so a hand-picked strength drifts
   * every time a field is added and drifts again with resolution. Inverting
   * `slope = atan(k * gradientRMS)` makes the surface land on the same RMS slope
   * whatever the stack contains, which is the whole fix for "strong round bumps
   * make it look like bubble wrap".
   */
  _bakeNormalAO(size) {
    const N = size * size;
    const h = new Float32Array(N);
    const blurred = new Float32Array(N);
    const inv = 1 / size;
    const cells = this._cellsFor(size);
    const kFine = cells(11);
    const kMid = Math.max(3, Math.round(kFine * 0.46));
    const kCoarse = cells(58);
    const kCrumbF = cells(38), kCrumbC = cells(95);
    const kFibLen = cells(90), kFibWide = cells(6);
    const wallF = new Float32Array(N);
    const wallC = new Float32Array(N);
    const cF = { id: 0, wall: 0, d1: 0 };
    const cC = { id: 0, wall: 0, d1: 0 };

    for (let y = 0; y < size; y++) {
      const t = y * inv;
      for (let x = 0; x < size; x++) {
        const s = x * inv;
        const i = y * size + x;
        pcell(s, t, kCrumbF, 9, cF);
        pcell(s, t, kCrumbC, 3, cC);
        wallF[i] = cF.wall;
        wallC[i] = cC.wall;
        // Amplitudes are "how steep", not "how big": the RMS gradient of a
        // band-limited layer goes as amplitude over wavelength, so the coarse
        // layers need more amplitude than intuition suggests merely to survive
        // the division by their own wavelength. Fine grain still leads - real
        // earth is dominated by it at centimetre scale.
        h[i] = 0.5
          + (pvalue(s * kFine, t * kFine, kFine, 211) - 0.5) * 0.22
          + (pfbm(s, t, kMid, 2, 213, 0.5) - 0.5) * 0.20
          + (pfbm(s, t, kCoarse, 2, 217, 0.52) - 0.5) * 0.24
          - cF.wall * 0.30 - cC.wall * 0.34
          + (cF.id - 0.5) * 0.20 + (cC.id - 0.5) * 0.32
          + pfibre(s, t, kFibLen, kFibWide, 401) * 0.30;
      }
    }

    // Solve the encode strength for a fixed RMS surface slope.
    const k = Math.tan(FALLBACK_SLOPE_RMS_DEG * Math.PI / 180)
      / Math.max(gradRMSWrap(h, size), 1e-6);

    blurWrap(h, blurred, size, Math.max(2, Math.round(size / 26)));

    const d = new Uint8Array(N * 4);
    for (let y = 0; y < size; y++) {
      const ym = mod(y - 1, size) * size, yp = mod(y + 1, size) * size, yc = y * size;
      for (let x = 0; x < size; x++) {
        const i = yc + x;
        const dx = (h[yc + mod(x + 1, size)] - h[yc + mod(x - 1, size)]) * 0.5 * k;
        const dy = (h[yp + x] - h[ym + x]) * 0.5 * k;
        const len = Math.sqrt(dx * dx + dy * dy + 1);
        const o = i * 4;
        d[o] = Math.round((-dx / len * 0.5 + 0.5) * 255);
        d[o + 1] = Math.round((-dy / len * 0.5 + 0.5) * 255);
        d[o + 2] = Math.round((1 / len * 0.5 + 0.5) * 255);
        // Cavity AO belongs in the creases between crumbs and nowhere else, so
        // it is driven mostly by the narrowed wall masks and only partly by the
        // local dip. Its mean sits near 1 with thin dark lines - the version
        // this replaces had half the tile lit, which is what made the domes pale.
        const dip = clamp01((blurred[i] - h[i]) * 2.6);
        const occ = clamp01(dip * 0.60 + wallF[i] * 0.58 + wallC[i] * 0.46);
        d[o + 3] = Math.round(clamp01(1 - occ * 0.34) * 255);
      }
    }
    return this._finishTexture(new THREE.DataTexture(d, size, size, THREE.RGBAFormat), 4);
  }

  /**
   * FALLBACK kilometre-scale variation: R dryness, G lushness, B moss affinity,
   * A mottle. Same standing as the two bakes above - core/textures.js owns the
   * map that ships.
   *
   * Two constraints, and they pull against each other:
   *
   *  1. SPREAD. The splat weights above compete within about half a unit of each
   *     other, so the macro field's standard deviation is precisely how
   *     decisively earth, thatch and moss take turns. A raw `pfbm` has sd 0.10,
   *     which produced near-uniform ground no matter how the weights were tuned.
   *     The gains below put the four channels on the same distribution
   *     core/textures.js measures for the map it ships - R 0.46/0.20, G
   *     0.51/0.20, B 0.46/0.19, A 0.50/0.145 - so the weights land in the same
   *     place either way.
   *  2. GRADIENT of B and A, which is a separate constraint and the easy one to
   *     break: the shader domain-warps its near detail lookup by
   *     `(macro.ba - 0.5) * DETAIL_WARP` metres, and DETAIL_WARP = 3.4 was
   *     picked against a measured Jacobian. Gradient is spread times frequency,
   *     so doubling the spread at the old octave depth would have pushed the
   *     warp past the "visibly blurs" row of that table without anyone touching
   *     DETAIL_WARP. Those two channels therefore lose an octave each as they
   *     gain contrast, which trades frequency for spread at constant gradient.
   */
  _bakeMacro(size) {
    const N = size * size;
    const inv = 1 / size;
    const cDry = new Float32Array(N);
    const cLush = new Float32Array(N);
    const cMoss = new Float32Array(N);
    const cMot = new Float32Array(N);
    for (let y = 0; y < size; y++) {
      const t = y * inv;
      for (let x = 0; x < size; x++) {
        const s = x * inv, i = y * size + x;
        cDry[i] = pfbm(s, t, 3, 4, 301, 0.55);
        // Lushness is mildly anti-correlated with dryness rather than
        // independent. Fully independent fields let a patch be both parched and
        // lush, which is where "procedural patchwork" comes from.
        cLush[i] = pfbm(s, t, 4, 4, 331, 0.55) * 0.88 + (1 - cDry[i]) * 0.17;
        // Two octaves and four, not three and five: these are the two channels
        // the shader domain-warps by, and they are about to gain contrast. See
        // constraint 2 above.
        cMoss[i] = pfbm(s, t, 6, 2, 367, 0.5);
        cMot[i] = pfbm(s, t, 2, 4, 401, 0.5);
      }
    }
    retargetWrap(cDry, 0.46, 0.20);
    retargetWrap(cLush, 0.51, 0.20);
    retargetWrap(cMoss, 0.46, 0.19);
    retargetWrap(cMot, 0.50, 0.145);

    const d = new Uint8Array(N * 4);
    for (let i = 0; i < N; i++) {
      const o = i * 4;
      d[o] = Math.round(cDry[i] * 255);
      d[o + 1] = Math.round(cLush[i] * 255);
      d[o + 2] = Math.round(cMoss[i] * 255);
      d[o + 3] = Math.round(cMot[i] * 255);
    }
    return this._finishTexture(new THREE.DataTexture(d, size, size, THREE.RGBAFormat), 1);
  }

  /** Fallback cloud-shadow field, used when sky/clouds.js publishes no map. */
  _bakeCloud(size) {
    const d = new Uint8Array(size * size * 4);
    const inv = 1 / size;
    for (let y = 0; y < size; y++) {
      const t = y * inv;
      for (let x = 0; x < size; x++) {
        const s = x * inv, o = (y * size + x) * 4;
        // Warping the coverage field gives the cast shadows the lobed edges of
        // real cumulus instead of the isotropic smear of plain fbm.
        const wx = (pfbm(s, t, 3, 3, 511) - 0.5) * 0.20;
        const wy = (pfbm(s, t, 3, 3, 547) - 0.5) * 0.20;
        // Four octaves from base 4, not five: at 128 texels the fifth lands at
        // 64 cells, i.e. two texels a cell, which is under Nyquist for the base
        // mip - it contributes sparkle and nothing else, and costs a full pass.
        d[o] = Math.round(clamp01((pfbm(s + wx, t + wy, 4, 4, 577, 0.55) - 0.30) * 1.9) * 255);
        d[o + 1] = Math.round(clamp01(pfbm(s, t, 9, 3, 601, 0.5) * 1.2 - 0.1) * 255);
        d[o + 2] = Math.round(clamp01(pfbm(s, t, 6, 3, 631, 0.5)) * 255);
        d[o + 3] = 255;
      }
    }
    return this._finishTexture(new THREE.DataTexture(d, size, size, THREE.RGBAFormat), 2);
  }

  // =========================================================================
  // LEAF 3 - material
  // =========================================================================

  _buildUniforms() {
    const t = this._tex;
    return {
      tDetail: { value: t.detail },
      tNormalAO: { value: t.normalAO },
      tMacro: { value: t.macro },
      tCloud: { value: t.cloud },
      // Declared unconditionally so the value can be swapped whenever
      // sky/clouds.js reallocates its target; only the DEFINE that reads them
      // costs a recompile, and that only flips when the map appears or goes.
      tCloudMap: { value: null },
      uCloudMatrix: { value: new THREE.Matrix4() },
      uUVOrigin: { value: new THREE.Vector2() },
      uTiling: {
        value: new THREE.Vector4(1 / TILE_NEAR, 1 / TILE_MID, 1 / TILE_MACRO, 1 / TILE_MESO),
      },
      uDetailFade: { value: new THREE.Vector2(this._tier.detailFade * 0.55, this._tier.detailFade) },
      // y is an irrational-ish UV offset for the meso lookup: it moves the
      // rotated copy off the macro copy's own lattice phase as well as its
      // orientation, so no scale of the field can rhyme with another.
      uDetailWarp: { value: new THREE.Vector2(DETAIL_WARP, 0.3819660) },
      uHeightBand: { value: new THREE.Vector2(0, 1 / 50) },
      // Authored in sRGB and converted to linear by three's colour management.
      //
      // THE EARTH PALETTE. Weighted through the splat these average to sRGB
      // (0.37, 0.33, 0.26) before lighting - a mid-dark warm brown with r/b of
      // 1.4. The palette this replaces averaged (0.42, 0.42, 0.33): lighter than
      // the grass standing on it, and neutral to slightly cool, which is exactly
      // the "mottled blue-green bog" read. Every value here is desaturated - 
      // the brief is quiet and melancholy, not a ploughed field in a postcard - 
      // but the hue is unmistakably earth and it is darker than the sward.
      //
      // Damp umber. This is the base note of the whole ground: it is what shows
      // in every hollow, under the canopy and between the thatch, and it wants
      // to be dark. sRGB 0.23/0.19/0.15.
      uColEarth: { value: new THREE.Color('#4a3422') },
      // Sun-crusted soil. Grey-brown rather than orange-brown: a dried meadow
      // crust is silty and pale, and any more chroma here reads as desert.
      uColEarthDry: { value: new THREE.Color('#7d6141') },
      // Dead thatch, shaded and matted.
      uColThatch: { value: new THREE.Color('#6b5a38') },
      // Sun-bleached straw. The one genuinely light value in the palette, and
      // it is a minority of a minority - it needs to be there, because a meadow
      // floor with no pale litter in it looks wet.
      uColThatchDry: { value: new THREE.Color('#9c8757') },
      // Moss: the only green, deliberately dark and desaturated, and confined by
      // wMoss to hollows and shade.
      uColMoss: { value: new THREE.Color('#3d4a33') },
      uColPetal: { value: new THREE.Color('#b7a2a1') },
      // Overwritten by _updateMeadowPalette from world/grass.js's live blade
      // uniforms; these are the values that combination produces from grass.js's
      // documented palette, so a scene with no grass system still matches.
      uColMeadow: { value: new THREE.Color('#636948') },
      uColMeadowDry: { value: new THREE.Color('#787757') },
      uMeadowFade: {
        value: new THREE.Vector2(
          MEADOW_FALLBACK_REACH * MEADOW_START_FRAC,
          MEADOW_FALLBACK_REACH * MEADOW_END_FRAC
        ),
      },
      // Vector3, NOT Color: these are linear multipliers. A Color built from a
      // hex string would be gamma-decoded on the way in and the drift would come
      // out roughly twice as strong as authored.
      uTintCool: { value: new THREE.Vector3(0.86, 0.92, 0.96) },
      uTintWarm: { value: new THREE.Vector3(1.16, 1.07, 0.90) },
      // Per layer: earth, thatch, moss, petals. Bare soil is the roughest thing
      // in the scene; dead straw has a faint cuticle sheen left; a petal is the
      // smoothest of the four.
      uLayerRough: { value: new THREE.Vector4(0.96, 0.90, 0.86, 0.80) },
      // Halved, near enough, alongside dropping the baked map's calibrated RMS
      // slope from 14 degrees to 11 (core/textures.js TERRAIN_SLOPE_RMS_DEG).
      // The two multiply: 14 x 0.85 = 11.9 effective degrees of micro-relief
      // became 11 x 0.62 = 6.8. What is left is grain and the thin voids between
      // crumbs - soil texture - instead of centimetre-scale shading that made
      // the ground read as a moulded surface.
      uNormalScale: { value: 0.62 },
      uWetness: { value: 0 },
      uPuddle: { value: 0 },
      uRipple: { value: 0 },
      uSnow: { value: 0 },
      uTime: { value: 0 },
      uSunDir: { value: new THREE.Vector3(0, 1, 0) },
      uSkyColor: { value: new THREE.Color(0.6, 0.7, 0.85) },
      // --- fallback aerial perspective (see FRAG_AERIAL_PARS) ---
      uGroundHaze: { value: new THREE.Vector4(0.0022 * HAZE_SHARE, 1 / 1400, 0, 0) },
      uGroundMist: { value: new THREE.Vector4(0.0022 * MIST_SHARE, 1 / 5.5, 0, 0) },
      uFogColor: { value: new THREE.Color(0.72, 0.78, 0.86) },
      uFogZenith: { value: new THREE.Color(0.42, 0.56, 0.80) },
      uMistColor: { value: new THREE.Color(0.76, 0.80, 0.86) },
      uFogSunColor: { value: new THREE.Color(0, 0, 0) },
      uMieG: { value: 0.62 },
      uFarFade: { value: new THREE.Vector2(ROOT_SIZE * 0.80, ROOT_SIZE * 0.99) },
      // ----------------------------------------------------------
      uCloudParams: { value: new THREE.Vector4(0, 0.42, 900, 1 / 1150) },
      uCloudSky: { value: 0 },
      uCloudDrift: { value: new THREE.Vector4() },
      uPetalLitter: { value: new THREE.Vector4(0, 0, 12, 0) },
    };
  }

  _makeMaterial(level) {
    const mat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 1.0,
      metalness: 0.0,
      dithering: true,          // the aerial gradient is the one place 8-bit banding shows
      side: THREE.FrontSide,
    });
    // Carried across the rebuild. `update()` only pushes `wireframe` when
    // state.debug.wireframe *changes*, so a rebuild (quality change, or the
    // cloud shadow map appearing) while wireframe was on used to hand back a
    // solid ground and leave the HUD toggle out of sync until it was pressed
    // twice.
    mat.wireframe = this._wireframe;
    // We own the atmospheric fade (height falloff plus directional inscatter),
    // so three's flat-colour fog would only double up on it. If AtmosFX takes
    // the job instead, hand the fog back to three.
    mat.fog = this._externalAerial;
    mat.defines = {};
    if (this._tier.snow) mat.defines.TERRAIN_SNOW = '';
    if (this._tier.puddles) mat.defines.TERRAIN_PUDDLES = '';
    if (this._tier.cloudDetail) mat.defines.TERRAIN_CLOUD_DETAIL = '';
    if (this._tier.cheapAerial) mat.defines.TERRAIN_CHEAP_AERIAL = '';
    if (this._tier.meso) mat.defines.TERRAIN_MESO = '';
    if (this._tier.wetSheen) mat.defines.TERRAIN_WET_SHEEN = '';
    if (this._cloudMap) mat.defines.TERRAIN_CLOUD_MAP = '';
    // A level whose nearest possible vertex is already past the detail fade can
    // never show near detail, wetness or ripples. Compile them out.
    if (level > 0 && this._ranges[level - 1] >= this._tier.detailFade) {
      mat.defines.TERRAIN_FAR = '';
    }

    const range = this._ranges[level];
    const morph = { value: new THREE.Vector2(range * MORPH_START, range) };
    mat.userData.uMorphRange = morph;

    const shared = this._uniforms;
    const useCloudMap = !!this._cloudMap;
    const useAerial = !this._externalAerial;

    mat.onBeforeCompile = (shader) => {
      // The same uniform objects go onto every clone, so one write per frame
      // updates all seven levels. Only the morph range is per-level.
      Object.assign(shader.uniforms, shared);
      shader.uniforms.uMorphRange = morph;

      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\n' + VERT_PARS)
        .replace('#include <beginnormal_vertex>', VERT_NORMAL)
        .replace('#include <begin_vertex>', VERT_POS)
        .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\n' + VERT_WORLD);

      // Only declare what the compiled variant actually reads. An unused
      // sampler is harmless but an unused *declaration* still occupies a
      // texture-unit binding slot on some drivers, and the aerial block's eight
      // uniforms are pure noise in the far commoner atmosfx-owned case.
      const pars = FRAG_PARS
        + (useCloudMap ? '\nuniform sampler2D tCloudMap;\nuniform mat4 uCloudMatrix;\n' : '')
        + (useAerial ? FRAG_AERIAL_PARS : '');

      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', '#include <common>\n' + pars)
        .replace('#include <map_fragment>', FRAG_SURFACE)
        .replace('#include <roughnessmap_fragment>', 'float roughnessFactor = clamp( tRough, 0.03, 1.0 );')
        .replace('#include <normal_fragment_maps>', 'normal = normalize( ( viewMatrix * vec4( gNrm, 0.0 ) ).xyz );')
        .replace('#include <aomap_fragment>', `
          // Cavity AO and the cloud's own sky-occlusion both act on ambient;
          // the sun transmittance acts on the direct term only. Keeping the two
          // separate is what makes a passing cloud read as a moving shadow
          // rather than as a global dimmer.
          reflectedLight.indirectDiffuse *= gAO * gCloudSky;
          reflectedLight.indirectSpecular *= gAO * gCloudSky;
          reflectedLight.directDiffuse *= gCloudSun;
          reflectedLight.directSpecular *= gCloudSun;
          #ifndef TERRAIN_FAR
            // With no env map, water has nothing to mirror - so hand it the sky
            // colour directly. That is most of what sells wet ground.
            reflectedLight.indirectSpecular +=
              uSkyColor * ( gPuddleFresnel * 0.85 + gWetSheen ) * gCloudSky;
          #endif
        `);

      if (useAerial) {
        shader.fragmentShader = shader.fragmentShader
          .replace('#include <opaque_fragment>', '#include <opaque_fragment>\n' + FRAG_AERIAL);
      }
    };

    // three's own cache key covers `defines` and `fog`, which between them
    // already separate every variant this file emits - but only by coincidence,
    // since `useAerial` happens to equal `!mat.fog`. Stating the two decisions
    // explicitly means a future variant that is NOT covered by a define cannot
    // silently collide with a program compiled before a rebuild. Set before
    // atmosfx patches the material, because its wrapper captures whatever is
    // here at patch time and chains onto it.
    const keySuffix = `|terrain${useAerial ? 'A' : '-'}${useCloudMap ? 'C' : '-'}`;
    mat.customProgramCacheKey = () => keySuffix;

    const fx = this.ctx.systems && this.ctx.systems.atmosfx;
    if (this._externalAerial && fx) {
      if (typeof fx.applyToMaterial === 'function') fx.applyToMaterial(mat);
      else if (typeof fx.registerMaterial === 'function') fx.registerMaterial(mat);
    }
    return mat;
  }

  _rebuildMaterials() {
    for (const m of this._materials) m.dispose();
    this._materials.length = 0;
    for (let l = 0; l <= this.depth; l++) this._materials.push(this._makeMaterial(l));
    for (const node of this._live) {
      if (node.mesh) node.mesh.material = this._materials[node.level];
    }
  }

  // =========================================================================
  // LEAF 2 - chunked clipmap
  // =========================================================================

  _configure() {
    const tier = this._tier;
    this.depth = tier.depth;
    this.cells = tier.cells & ~1;             // must be even for the parent-grid collapse
    this._ranges.length = 0;
    for (let l = 0; l <= this.depth; l++) this._ranges.push(tier.range0 * Math.pow(2, l));

    const N = this.cells;
    this._gridW = N + 1;
    this._borderW = N + 5;                    // two-vertex apron: the parent normal needs i +/- 2
    this._vertCount = this._gridW * this._gridW + 4 * this._gridW;
    this._hs = new Float32Array(this._borderW * this._borderW);
    this._cs = new Float32Array(this._borderW * this._borderW);
    this._index = this._buildIndex(N);
  }

  _buildIndex(N) {
    const gw = N + 1;
    const idx = new Uint16Array((N * N * 2 + N * 4 * 2) * 3);
    let o = 0;
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const a = j * gw + i, b = a + 1, c = a + gw, d = c + 1;
        // Anti-diagonal everywhere. It is the same split the parent grid uses,
        // which is what makes a fully-morphed chunk identical to its parent.
        idx[o++] = a; idx[o++] = c; idx[o++] = b;
        idx[o++] = b; idx[o++] = c; idx[o++] = d;
      }
    }
    // Skirt vertices, in edge order: -Z, +Z, -X, +X, gw verts each. Windings
    // face outward so front-face culling keeps them.
    const S = gw * gw;
    for (let k = 0; k < N; k++) {
      const t0 = k, t1 = k + 1, b0 = S + k, b1 = b0 + 1;
      idx[o++] = t0; idx[o++] = t1; idx[o++] = b0;
      idx[o++] = t1; idx[o++] = b1; idx[o++] = b0;
    }
    for (let k = 0; k < N; k++) {
      const t0 = N * gw + k, t1 = t0 + 1, b0 = S + gw + k, b1 = b0 + 1;
      idx[o++] = t0; idx[o++] = b0; idx[o++] = t1;
      idx[o++] = t1; idx[o++] = b0; idx[o++] = b1;
    }
    for (let k = 0; k < N; k++) {
      const t0 = k * gw, t1 = t0 + gw, b0 = S + 2 * gw + k, b1 = b0 + 1;
      idx[o++] = t0; idx[o++] = b0; idx[o++] = t1;
      idx[o++] = t1; idx[o++] = b0; idx[o++] = b1;
    }
    for (let k = 0; k < N; k++) {
      const t0 = k * gw + N, t1 = t0 + gw, b0 = S + 3 * gw + k, b1 = b0 + 1;
      idx[o++] = t0; idx[o++] = t1; idx[o++] = b0;
      idx[o++] = t1; idx[o++] = b1; idx[o++] = b0;
    }
    return new THREE.BufferAttribute(idx, 1);
  }

  _createGeometry() {
    const n = this._vertCount;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    g.setAttribute('normal', new THREE.BufferAttribute(new Int8Array(n * 3), 3, true));
    g.setAttribute('aMorphOffset', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    g.setAttribute('aMorphNormal', new THREE.BufferAttribute(new Int8Array(n * 3), 3, true));
    g.setAttribute('aCavity', new THREE.BufferAttribute(new Uint8Array(n), 1, true));
    g.setIndex(this._index);        // one shared index buffer for every chunk in the world
    g.boundingBox = new THREE.Box3();
    g.boundingSphere = new THREE.Sphere();
    this._geomCount++;
    return g;
  }

  /**
   * Chunk geometry always comes from the pool; only its attribute contents
   * change, which turns a streaming build into a bufferSubData instead of a
   * fresh 30 kB allocation. Past the tier's ceiling we take the geometry back
   * off the least recently drawn chunk rather than growing the heap - that
   * ceiling is the difference between smooth streaming and a GC hitch every
   * few seconds while walking.
   */
  _acquireGeometry() {
    const pooled = this._geoPool.pop();
    if (pooled) return pooled;
    if (this._geomCount < this._tier.maxChunks) return this._createGeometry();
    return this._recycleOldest() || this._createGeometry();
  }

  _recycleOldest() {
    const live = this._live;
    let best = -1, bestUsed = this._frame;
    for (let i = 0; i < live.length; i++) {
      const n = live[i];
      // Anything drawn or traversed this frame is off limits.
      if (n.ready !== 2 || n.lastUsed >= this._frame) continue;
      if (n.lastUsed < bestUsed) { bestUsed = n.lastUsed; best = i; }
    }
    if (best < 0) return null;
    const node = live[best];
    live[best] = live[live.length - 1];
    live.length--;
    const g = node.geom;
    node.geom = null;                    // detach first so it is not pooled twice
    this._retireMesh(node);
    node.ready = 0;
    return g;
  }

  /** Detach a node's mesh and return the wrapper to the pool. */
  _retireMesh(node) {
    const mesh = node.mesh;
    if (!mesh) return;
    this.group.remove(mesh);
    mesh.geometry = null;                // the geometry is pooled separately
    mesh.visible = false;
    node.mesh = null;
    this._meshPool.push(mesh);
  }

  /** Interned debug label, so `_buildNode` never concatenates a string. */
  _levelName(level) {
    let n = this._levelNames[level];
    if (n === undefined) { n = `terrain-l${level}`; this._levelNames[level] = n; }
    return n;
  }

  _edgeVertex(e, k, N, gw) {
    return e === 0 ? k : e === 1 ? N * gw + k : e === 2 ? k * gw : k * gw + N;
  }

  /** Fill a chunk from the analytic field. The only expensive call in here. */
  _buildNode(node) {
    const N = this.cells;
    const gw = this._gridW;
    const W = this._borderW;
    const B = 2;
    const step = node.size / N;
    const half = node.size * 0.5;
    const hs = this._hs, cs = this._cs;

    for (let jj = 0; jj < W; jj++) {
      const wz = node.z + (jj - B) * step;
      const row = jj * W;
      for (let ii = 0; ii < W; ii++) {
        const h = this.getHeight(node.x + (ii - B) * step, wz);
        hs[row + ii] = h;
        cs[row + ii] = h - this._mid;
      }
    }

    const g = node.geom || (node.geom = this._acquireGeometry());
    const pos = g.attributes.position.array;
    const nrm = g.attributes.normal.array;
    const mo = g.attributes.aMorphOffset.array;
    const mn = g.attributes.aMorphNormal.array;
    const cav = g.attributes.aCavity.array;

    const invStep2 = 1 / (2 * step);
    const invStep4 = 1 / (4 * step);
    const invCav = 1 / CAVITY_SCALE;
    let minY = Infinity, maxY = -Infinity;

    for (let j = 0; j <= N; j++) {
      const jb = (j + B) * W;
      const j0b = ((j & ~1) + B) * W;
      const lz = j * step - half;
      for (let i = 0; i <= N; i++) {
        const k = jb + i + B;
        const h = hs[k];
        const p3 = (j * gw + i) * 3;

        pos[p3] = i * step - half;
        pos[p3 + 1] = h;
        pos[p3 + 2] = lz;
        if (h < minY) minY = h;
        if (h > maxY) maxY = h;

        // Geometric normal at this LOD's own spacing. A coarse mesh really is
        // flatter, and pretending otherwise puts a lighting seam at every LOD
        // boundary that no amount of geomorphing can hide.
        let dx = (hs[k + 1] - hs[k - 1]) * invStep2;
        let dz = (hs[k + W] - hs[k - W]) * invStep2;
        let inv = 1 / Math.sqrt(dx * dx + 1 + dz * dz);
        nrm[p3] = Math.round(-dx * inv * 127);
        nrm[p3 + 1] = Math.round(inv * 127);
        nrm[p3 + 2] = Math.round(-dz * inv * 127);

        // Parent-grid target: snap both indices down to even. That position is
        // a vertex of the parent chunk and its height came from the same
        // analytic field, so a fully-morphed chunk equals its parent exactly.
        const i0 = i & ~1;
        const k0 = j0b + i0 + B;
        mo[p3] = (i0 - i) * step;
        mo[p3 + 1] = hs[k0] - h;
        mo[p3 + 2] = ((j & ~1) - j) * step;

        dx = (hs[k0 + 2] - hs[k0 - 2]) * invStep4;
        dz = (hs[k0 + 2 * W] - hs[k0 - 2 * W]) * invStep4;
        inv = 1 / Math.sqrt(dx * dx + 1 + dz * dz);
        mn[p3] = Math.round(-dx * inv * 127);
        mn[p3 + 1] = Math.round(inv * 127);
        mn[p3 + 2] = Math.round(-dz * inv * 127);

        cav[j * gw + i] = Math.round(clamp01(cs[k] * invCav + 0.5) * 255);
      }
    }

    // Skirts. Depth follows the cell size because that is what bounds the
    // height error a coarser neighbour can present at the seam.
    const skirt = 0.45 + 0.09 * step;
    const S = gw * gw;
    for (let e = 0; e < 4; e++) {
      for (let k = 0; k <= N; k++) {
        const src = this._edgeVertex(e, k, N, gw);
        const dst = S + e * gw + k;
        const s3 = src * 3, d3 = dst * 3;
        pos[d3] = pos[s3];
        pos[d3 + 1] = pos[s3 + 1] - skirt;
        pos[d3 + 2] = pos[s3 + 2];
        nrm[d3] = nrm[s3]; nrm[d3 + 1] = nrm[s3 + 1]; nrm[d3 + 2] = nrm[s3 + 2];
        mn[d3] = mn[s3]; mn[d3 + 1] = mn[s3 + 1]; mn[d3 + 2] = mn[s3 + 2];
        // Same morph offset as the rim vertex, so the skirt tracks the surface
        // through the whole transition instead of tearing away from it.
        mo[d3] = mo[s3]; mo[d3 + 1] = mo[s3 + 1]; mo[d3 + 2] = mo[s3 + 2];
        cav[dst] = cav[src];
      }
    }

    g.attributes.position.needsUpdate = true;
    g.attributes.normal.needsUpdate = true;
    g.attributes.aMorphOffset.needsUpdate = true;
    g.attributes.aMorphNormal.needsUpdate = true;
    g.attributes.aCavity.needsUpdate = true;

    node.minY = minY - skirt;
    node.maxY = maxY;
    // Morphing can lift a vertex by up to the local relief, so pad rather than
    // trusting the sampled extremes to bound the animated mesh.
    const pad = (maxY - minY) * 0.25 + step * 0.5;
    g.boundingBox.min.set(-half, minY - skirt - pad, -half);
    g.boundingBox.max.set(half, maxY + pad, half);
    g.boundingBox.getBoundingSphere(g.boundingSphere);

    if (!node.mesh) {
      const mesh = this._meshPool.pop() || new THREE.Mesh(g, this._materials[node.level]);
      mesh.geometry = g;
      mesh.material = this._materials[node.level];
      mesh.castShadow = false;       // gentle relief gains nothing, and the shadow pass is precious
      mesh.receiveShadow = true;
      mesh.matrixAutoUpdate = false;
      mesh.position.set(node.x + half, 0, node.z + half);
      mesh.updateMatrix();           // also raises matrixWorldNeedsUpdate, which a pooled mesh needs
      mesh.visible = false;
      mesh.name = this._levelName(node.level);
      node.mesh = mesh;
      this.group.add(mesh);
    } else {
      node.mesh.geometry = g;
      node.mesh.material = this._materials[node.level];
    }
    node.ready = 2;
    node.lastUsed = this._frame;
    this._live.push(node);
  }

  _makeNode(level, x, z) {
    return {
      level, x, z,
      size: ROOT_SIZE / Math.pow(2, this.depth - level),
      ix: 0, iz: 0, key: 0,      // roots only; declared here so every node shares a shape
      children: null,
      geom: null, mesh: null,
      ready: 0, lastUsed: 0, priority: 0,
      minY: this._worldMinY, maxY: this._worldMaxY,
    };
  }

  _children(node) {
    const h = node.size * 0.5;
    node.children = [
      this._makeNode(node.level - 1, node.x, node.z),
      this._makeNode(node.level - 1, node.x + h, node.z),
      this._makeNode(node.level - 1, node.x, node.z + h),
      this._makeNode(node.level - 1, node.x + h, node.z + h),
    ];
    return node.children;
  }

  _boxDist(node, cx, cy, cz) {
    const s = node.size;
    const dx = Math.max(node.x - cx, 0, cx - (node.x + s));
    const dz = Math.max(node.z - cz, 0, cz - (node.z + s));
    const dy = Math.max(node.minY - cy, 0, cy - node.maxY);
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  _enqueue(node, priority) {
    if (node.ready === 2) return;
    node.priority = priority;
    node.lastUsed = this._frame;
    if (node.ready === 1) return;
    node.ready = 1;
    this._queue.push(node);
  }

  _refine(node, cx, cy, cz) {
    node.lastUsed = this._frame;
    if (node.level > 0) {
      const d = this._boxDist(node, cx, cy, cz);
      const r = this._ranges[node.level - 1];
      if (d < r) {
        const ch = node.children || this._children(node);
        let ready = true;
        for (let i = 0; i < 4; i++) {
          if (ch[i].ready !== 2) { this._enqueue(ch[i], d); ready = false; }
          // Touch the siblings that ARE built. Without this, three ready
          // children sit untouched for however long the fourth takes, and
          // _prune's "crowded" clock (3 s) can evict them out from under a
          // subtree the camera is actively looking at.
          else ch[i].lastUsed = this._frame;
        }
        // Descending only into fully-built children is what keeps streaming
        // hitch-free: a miss shows one frame of coarser ground, never a hole.
        if (ready) {
          for (let i = 0; i < 4; i++) this._refine(ch[i], cx, cy, cz);
          return;
        }
      } else if (d < r * PREFETCH) {
        const ch = node.children || this._children(node);
        for (let i = 0; i < 4; i++) {
          if (ch[i].ready === 0) this._enqueue(ch[i], d * 4);
          // A prefetched chunk that is already built is still WANTED - it is
          // the whole point of prefetching. This line was missing, so a
          // prefetched chunk's lastUsed froze the moment it finished building,
          // _prune evicted it a few seconds later, and the prefetch branch
          // immediately re-queued it because ready had gone back to 0. The
          // result was a permanent build treadmill: measured over ten seconds
          // with the camera STATIONARY, 34 chunk rebuilds a second at HIGH and
          // 32 at LOW - roughly 1.2 MB/s of pointless vertex-buffer traffic and
          // a chunk build inside the frame budget on almost every frame, on the
          // tier that has the least room for it.
          else ch[i].lastUsed = this._frame;
        }
      }
    }
    node.mesh.visible = true;
    this._selected.push(node);
  }

  _select(cx, cy, cz) {
    for (let i = 0; i < this._selected.length; i++) {
      const m = this._selected[i].mesh;
      if (m) m.visible = false;
    }
    this._selected.length = 0;

    const rx = Math.floor(cx / ROOT_SIZE), rz = Math.floor(cz / ROOT_SIZE);
    // The inner 3x3 is drawn; the surrounding ring is kept warm so crossing a
    // four-kilometre root boundary never costs a visible frame.
    for (let j = -2; j <= 2; j++) {
      for (let i = -2; i <= 2; i++) {
        const ix = rx + i, iz = rz + j;
        // Kept inside the small-integer range so the lookup never boxes a
        // double. Two roots can only collide four million metres apart, and
        // the stored indices catch that case anyway.
        const key = ((iz & 1023) << 10) | (ix & 1023);
        let root = this._roots.get(key);
        if (root && (root.ix !== ix || root.iz !== iz)) {
          this._releaseTree(root);
          this._roots.delete(key);
          root = undefined;
        }
        if (!root) {
          root = this._makeNode(this.depth, ix * ROOT_SIZE, iz * ROOT_SIZE);
          root.ix = ix; root.iz = iz; root.key = key;
          this._roots.set(key, root);
        }
        const inner = Math.abs(i) <= 1 && Math.abs(j) <= 1;
        if (root.ready !== 2) { this._enqueue(root, inner ? -1 : 1e6); continue; }
        root.lastUsed = this._frame;
        if (inner) this._refine(root, cx, cy, cz);
      }
    }
  }

  _processQueue(budgetMs) {
    const q = this._queue;
    if (q.length === 0) return;
    q.sort(_byPriority);
    const t0 = performance.now();
    const stale = this._frame - 150;
    let built = 0;
    while (q.length > 0) {
      // Stop *before* a build that would blow the budget rather than after.
      // One chunk costs about a millisecond, so checking afterwards overshoots
      // by a whole chunk every time. Always allow the first one so the queue
      // can never deadlock.
      if (built > 0 && performance.now() - t0 + this._chunkMs > budgetMs) break;
      const node = q.pop();
      if (node.ready === 2) continue;
      // Nothing has asked for this in two and a half seconds - the camera moved
      // on. Drop it rather than spend the budget on ground nobody wants.
      if (node.lastUsed < stale) { node.ready = 0; continue; }
      const s = performance.now();
      this._buildNode(node);
      this._chunkMs += (performance.now() - s - this._chunkMs) * 0.25;
      built++;
    }
  }

  _releaseNode(node) {
    this._retireMesh(node);
    if (node.geom) {
      this._geoPool.push(node.geom);   // _geomCount is the real ceiling, not the pool
      node.geom = null;
    }
    node.ready = 0;
  }

  _releaseTree(node) {
    if (node.children) {
      for (let i = 0; i < 4; i++) this._releaseTree(node.children[i]);
      node.children = null;
    }
    this._releaseNode(node);
  }

  /** Evict chunks nobody has looked at in a while. O(live), runs at 1 Hz. */
  _prune(cx, cz) {
    const live = this._live;
    const idle = this._frame - 900;
    const crowded = live.length > this._tier.maxChunks * 0.85 ? this._frame - 180 : idle;
    let w = 0;
    for (let i = 0; i < live.length; i++) {
      const node = live[i];
      if (node.ready !== 2) continue;      // already released elsewhere; drop the reference
      // Coarse chunks are cheap to keep and expensive to lose: they are the
      // fallback the streaming system leans on when a fine chunk is late.
      const keep = node.level >= this.depth - 1
        || node.lastUsed > (node.level >= 3 ? idle : crowded);
      if (keep) live[w++] = node;
      else this._releaseNode(node);
    }
    live.length = w;

    const rx = Math.floor(cx / ROOT_SIZE), rz = Math.floor(cz / ROOT_SIZE);
    for (const root of this._roots.values()) {
      if (Math.abs(root.ix - rx) > 2 || Math.abs(root.iz - rz) > 2) {
        this._releaseTree(root);
        this._roots.delete(root.key);
      }
    }
  }

  // =========================================================================
  // Lifecycle
  // =========================================================================

  async init() {
    this._configure();

    // Measure the real vertical extent instead of guessing it - these bounds
    // decide the LOD distance of every chunk before it is built.
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < 1600; i++) {
      const a = i * 2.399963;                                   // golden-angle spiral
      const r = Math.sqrt(i / 1600) * 9000;
      const h = this.getHeight(Math.cos(a) * r, Math.sin(a) * r);
      if (h < lo) lo = h;
      if (h > hi) hi = h;
    }
    this._worldMinY = lo * 1.25 - 5;
    this._worldMaxY = hi * 1.25 + 5;
    // Regional altitude band for the splat weights. Taken from the measured
    // relief rather than a magic number, so the ridge/hollow distinction lands
    // in the same place whatever the heightfield constants are retuned to.
    this._altCentre = (lo + hi) * 0.5;
    this._altInvHalf = 2 / Math.max(hi - lo, 1);

    this._tex.detail = this._texture('terrain.detail', () => this._bakeDetail(256));
    this._tex.normalAO = this._texture('terrain.normalAO', () => this._bakeNormalAO(256));
    // Macro and cloud fields are low frequency by construction; 128 is 8 m a
    // texel over their tiles, which is finer than either mask ever varies.
    this._tex.macro = this._texture('terrain.macro', () => this._bakeMacro(128));
    this._tex.cloud = this._texture('terrain.cloudShadow', () => this._bakeCloud(128));
    this._applyAnisotropy();

    this._uniforms = this._buildUniforms();
    this._uniforms.uHeightBand.value.set(this._altCentre, this._altInvHalf);
    // Seed the mist plane at the spawn point's ground height so the very first
    // frame is not rendered with the layer sitting at absolute zero.
    this._mistBase = this.getHeight(this.ctx.camera.position.x, this.ctx.camera.position.z) - 0.2;
    this._mistBaseInit = true;
    this._rebuildMaterials();

    this._preallocatePool();

    // Warm the ground around the spawn point while the loading screen is still
    // up, so the opening frames never pay for a burst of builds.
    const cam = this.ctx.camera.position;
    this._frame = 1;
    const deadline = performance.now() + 700;
    for (let pass = 0; pass < 24; pass++) {
      this._select(cam.x, cam.y, cam.z);
      if (this._queue.length === 0 || performance.now() > deadline) break;
      this._processQueue(70);
    }
  }

  /**
   * Allocate the chunk pool in one run. Interleaving 30 kB allocations with
   * build work is what turns a warm-up into a GC storm; doing it up front costs
   * ~20 ms and means neither the warm-up nor streaming ever allocates again.
   */
  _preallocatePool() {
    const target = Math.round(this._tier.maxChunks * 0.8);
    while (this._geomCount < target) this._geoPool.push(this._createGeometry());
    // Tier went down: hand the surplus back rather than sitting on the memory.
    while (this._geomCount > this._tier.maxChunks && this._geoPool.length > 0) {
      this._geoPool.pop().dispose();
      this._geomCount--;
    }
  }

  /**
   * These textures are usually the shared ones core/textures.js bakes, so this
   * overrides a sampler setting another module chose. Both caps below exist
   * because that override was costing fill rate on the two heaviest fetches in
   * the frame, on the part with the least of it to spare.
   *
   * DETAIL / NORMAL-AO, capped at 8. ULTRA asks for 16, and 16 is 16 texel
   * fetches per tap on the two RGBA maps that every ground fragment inside the
   * detail fade reads - on a tier that already targets 30 fps and whose fade
   * runs out to 210 m. core/textures.js caps its own at 8 for exactly this
   * reason ("the 780M pays real bandwidth beyond that for no visible win") and
   * that reasoning is right; 8 is still full quality at grazing incidence on a
   * 256-512 px map, so LOW/MEDIUM/HIGH (2/4/8) are untouched.
   *
   * MACRO, capped at 2. This one was plainly wrong: the map is sampled from RAW
   * world coordinates at 1/1024 and 1/128, so its screen-space derivative is
   * minute and an anisotropic tap fetches sixteen texels to return the one it
   * already had. HIGH was running it at 4 and ULTRA at 8, on a fetch taken by
   * every ground pixel on screen near AND far - the one fetch that is never
   * branched away. 2 keeps the meso tap from smearing at the horizon (which is
   * the real reason it is not 1) and drops the rest.
   */
  _applyAnisotropy() {
    const caps = this.ctx.renderer && this.ctx.renderer.capabilities;
    const max = caps && caps.getMaxAnisotropy ? caps.getMaxAnisotropy() : 4;
    const a = Math.max(1, Math.min(this._tier.aniso, 8, max));
    for (const key of ['detail', 'normalAO']) {
      const t = this._tex[key];
      if (t && t.anisotropy !== a) { t.anisotropy = a; t.needsUpdate = true; }
    }
    const am = Math.max(1, Math.min(this._tier.aniso >> 1, 2, max));
    const macro = this._tex.macro;
    if (macro && macro.anisotropy !== am) { macro.anisotropy = am; macro.needsUpdate = true; }
  }

  /**
   * LEAF 3 - build the far-field meadow colour out of world/grass.js's own
   * blade palette instead of guessing one.
   *
   * grass.js authors `uColBase` / `uColTip` / `uColDry` as sRGB hex, and three's
   * colour management has already decoded them to LINEAR working space by the
   * time they sit in the uniform - which is exactly where a weighted average of
   * radiances belongs, so the three are combined as they are and written back
   * with `setRGB`, which also writes working space and performs no conversion.
   * Doing this in sRGB bytes would darken the mix by several percent and, worse,
   * would shift its hue toward whichever component happened to be largest.
   *
   * Runs on link() and on a quality change, never per frame.
   */
  _updateMeadowPalette() {
    const u = this._uniforms;
    if (!u) return;
    const gu = this._grass && this._grass.material && this._grass.material.uniforms;
    const pick = (name, hex, out) => {
      const v = gu && gu[name] && gu[name].value;
      if (v && v.isColor) out.copy(v);
      else out.set(hex);
      return out;
    };
    const b = pick('uColBase', GRASS_PALETTE_FALLBACK.base, _gBase);
    const t = pick('uColTip', GRASS_PALETTE_FALLBACK.tip, _gTip);
    const d = pick('uColDry', GRASS_PALETTE_FALLBACK.dry, _gDry);

    const m = MEADOW_MIX;
    u.uColMeadow.value.setRGB(
      (b.r * m.base + t.r * m.tip + d.r * m.dry) * m.gain,
      (b.g * m.base + t.g * m.tip + d.g * m.dry) * m.gain,
      (b.b * m.base + t.b * m.tip + d.b * m.dry) * m.gain
    );
    const k = MEADOW_DRY_MIX;
    u.uColMeadowDry.value.setRGB(
      (b.r * k.base + t.r * k.tip + d.r * k.dry) * k.gain,
      (b.g * k.base + t.g * k.tip + d.g * k.dry) * k.gain,
      (b.b * k.base + t.b * k.tip + d.b * k.dry) * k.gain
    );
  }

  /**
   * The blend window, from grass.js's live fade. Read every frame because its
   * altitude clipmap rewrites `uFade` as the player climbs, and clamped because
   * the window must NOT follow that out to 576 m - see MEADOW_MAX_END.
   */
  _meadowReach() {
    const gu = this._grass && this._grass.material && this._grass.material.uniforms;
    const f = gu && gu.uFade && gu.uFade.value;
    const reach = f && f.y > 1 ? f.y : MEADOW_FALLBACK_REACH;
    return Math.min(reach, MEADOW_MAX_END);
  }

  /**
   * Devtools probe: `SAKURA.systems.terrain.meadowReport()`.
   *
   * LEAF 3 has the same failure mode as the cloud shadow map - it is wired by
   * duck typing off world/grass.js, it degrades to a plausible-looking fallback
   * if a property name drifts, and the symptom (a brown ring around the camera,
   * visible only from the air) is one nobody goes looking for. So the wiring
   * states itself.
   *
   * `sampled` false means the blade palette was NOT read from grass.js and the
   * hard-coded fallback is in use - a retune over there would then silently
   * reopen the ring. `endsBeforeBlades` must be true: the earth -> meadow blend
   * has to be finished by the time the last blade is drawn, never after it.
   */
  meadowReport() {
    const u = this._uniforms;
    const gu = this._grass && this._grass.material && this._grass.material.uniforms;
    const fade = gu && gu.uFade && gu.uFade.value;
    const bladeReach = fade && fade.y > 1 ? fade.y : null;
    const hex = (c) => (c ? `#${c.getHexString()}` : null);
    return {
      // false rather than null everywhere it is a genuine answer; null only
      // where init() has not run yet and the question has no answer at all.
      ready: !!u,
      grassLinked: !!this._grass,
      sampled: !!(gu && gu.uColBase && gu.uColBase.value && gu.uColBase.value.isColor),
      bladeReach,
      meadowEnd: this._meadowEnd,
      blendStart: u ? u.uMeadowFade.value.x : null,
      blendEnd: u ? u.uMeadowFade.value.y : null,
      meadow: hex(u && u.uColMeadow.value),
      meadowDry: hex(u && u.uColMeadowDry.value),
      earth: hex(u && u.uColEarth.value),
      endsBeforeBlades: u ? (!bladeReach || u.uMeadowFade.value.y <= bladeReach + 1e-3) : null,
    };
  }

  link(systems) {
    this._tree = systems.tree || null;
    this._clouds = systems.clouds || null;
    this._grass = systems.grass || null;
    this._updateMeadowPalette();
    const fx = systems.atmosfx || null;

    // AtmosFX owns the scene's aerial perspective if it offers a way in. When it
    // does, three's fog flag goes back on and our own FRAG_AERIAL is not
    // compiled at all - two atmospheres over one ground is the classic way to
    // end up with a scene that is washed out at arm's length.
    const external = !!(fx && (typeof fx.applyToMaterial === 'function'
      || typeof fx.registerMaterial === 'function'));

    const cloudChanged = this._pollCloudShadow();
    if (external !== this._externalAerial) {
      this._externalAerial = external;
      if (this._uniforms) this._rebuildMaterials();
    } else if (cloudChanged && this._uniforms) {
      this._rebuildMaterials();
    }
  }

  /**
   * Read sky/clouds.js's shadow map. It publishes either an explicit
   * `getShadowInfo()` or a texture + projection matrix pair; anything else falls
   * through to the analytic shadow field, which is a real feature, not a
   * placeholder.
   *
   * Called every frame, not once in link(), because clouds.js reallocates its
   * render target whenever the quality tier changes - holding the reference
   * from link() means silently sampling a disposed texture from then on. The
   * uniform value is swapped in place; only a change in PRESENCE alters a
   * define, and that is the only case that returns true.
   *
   * @returns {boolean} true if the material defines need rebuilding
   */
  _pollCloudShadow() {
    if (this._cloudPushed) return false;   // an explicit push wins over polling
    const clouds = this._clouds;
    let map = null, matrix = null;
    if (clouds) {
      const info = typeof clouds.getShadowInfo === 'function' ? clouds.getShadowInfo() : clouds;
      if (info) {
        const t = info.shadowTexture || info.cloudShadowTexture || info.shadowMap || info.map;
        const m = info.shadowMatrix || info.cloudShadowMatrix || info.matrix;
        if (t && t.isTexture && m && m.isMatrix4) { map = t; matrix = m; }
      }
    }
    return this._adoptCloudShadow(map, matrix);
  }

  _adoptCloudShadow(map, matrix) {
    const changed = !!map !== !!this._cloudMap;
    this._cloudMap = map;
    this._cloudMatrix = matrix;
    if (this._uniforms) {
      this._uniforms.tCloudMap.value = map;
      // Referenced, not copied: clouds.js keeps one stable Matrix4 and rewrites
      // its elements, so this stays live for free.
      if (matrix) this._uniforms.uCloudMatrix.value = matrix;
    }
    return changed;
  }

  /**
   * Cloud shadows are the single strongest realism cue on open ground, and they
   * are also the easiest thing in the file to have silently lost: the map
   * arrives by duck typing off a sibling system, and if the property names ever
   * drift the fallback field takes over and the ground still looks plausible.
   * "Plausible" is precisely why nobody notices for a month.
   *
   * So the absence is stated rather than assumed. Three frames of grace would be
   * a race - sky/clouds.js allocates its target in init() and re-allocates it on
   * every quality change - so this waits three seconds of frames before it says
   * anything, and says it once.
   */
  _checkCloudShadow() {
    if (this._cloudMap || this._cloudWarned) return;
    if (++this._cloudWaited < 180) return;
    this._cloudWarned = true;
    const c = this._clouds;
    console.warn('[Terrain] no cloud shadow map adopted after 180 frames; the ground is '
      + 'using the analytic fallback field. sky/clouds.js must expose a Texture on one of '
      + 'shadowTexture / cloudShadowTexture / shadowMap / map and a Matrix4 on one of '
      + `shadowMatrix / cloudShadowMatrix / matrix. Saw: system=${!!c}`
      + `, texture=${!!(c && (c.shadowTexture || c.cloudShadowTexture))}`
      + `, matrix=${!!(c && (c.shadowMatrix || c.cloudShadowMatrix))}.`);
  }

  /**
   * Devtools probe: `SAKURA.systems.terrain.cloudShadowReport()`.
   *
   * Reports which path the material actually compiled, so "are cloud shadows
   * working?" is a question with an answer rather than a squint at the screen.
   * `strength` is the artistic scale on the sun term and `sky` the one on
   * ambient; if `map` is true and `strength` is above zero, a cloud crossing the
   * sun is darkening the ground.
   */
  cloudShadowReport() {
    const u = this._uniforms;
    return {
      map: !!this._cloudMap,
      pushed: this._cloudPushed,
      define: !!(this._materials[0] && this._materials[0].defines
        && 'TERRAIN_CLOUD_MAP' in this._materials[0].defines),
      strength: u ? u.uCloudParams.value.x : 0,
      sky: u ? u.uCloudSky.value : 0,
      referenceY: u ? u.uCloudParams.value.z : 0,
    };
  }

  /** sky/clouds.js may push its shadow map here at any time instead of via link(). */
  setCloudShadowMap(texture, matrix) {
    this._cloudPushed = true;
    const map = texture && texture.isTexture ? texture : null;
    const mat = matrix && matrix.isMatrix4 ? matrix : null;
    if (map === this._cloudMap && mat === this._cloudMatrix) return;
    if (this._adoptCloudShadow(map, mat) && this._uniforms) this._rebuildMaterials();
  }

  onQualityChange(quality) {
    const tier = TIERS[quality.tier] || TIERS.high;
    if (tier === this._tier) return;
    const wasCells = this.cells;
    this._tier = tier;
    this._configure();

    if (wasCells !== this.cells) {
      // The vertex layout changed: every cached chunk and pooled geometry is
      // the wrong shape now.
      for (const root of this._roots.values()) this._releaseTree(root);
      this._roots.clear();
      this._live.length = 0;
      this._selected.length = 0;
      this._queue.length = 0;
      for (const g of this._geoPool) g.dispose();
      this._geoPool.length = 0;
      this._geomCount = 0;
    } else {
      for (const node of this._live) node.lastUsed = this._frame;
    }
    this._preallocatePool();

    this._applyAnisotropy();
    if (this._uniforms) {
      this._uniforms.uDetailFade.value.set(tier.detailFade * 0.55, tier.detailFade);
      // world/grass.js rebuilds its own material state on a quality change, so
      // re-derive the far-field colour from whatever its blade uniforms hold
      // now. `_updateMeadowPalette` documents itself as running here and did
      // not; without it a grass retune applied at runtime left the meadow on
      // the palette captured at link().
      this._updateMeadowPalette();
      this._rebuildMaterials();
    }
  }

  // =========================================================================
  // Per-frame
  // =========================================================================

  update(dt, state) {
    if (!this._uniforms) return;
    this._frame++;

    const cam = this.ctx.camera.position;
    this._select(cam.x, cam.y, cam.z);
    this._processQueue(this._tier.budgetMs);

    this._pruneTimer += dt;
    if (this._pruneTimer > 1.0) { this._pruneTimer = 0; this._prune(cam.x, cam.z); }

    // state.js delegates this field to us. The controller samples getHeight()
    // itself for its own step, but everything downstream can read it off state.
    const p = state.player.position;
    state.player.groundHeight = this.getHeight(p.x, p.z);

    const u = this._uniforms;
    const w = state.weather;

    u.uTime.value = state.time.elapsed;
    u.uUVOrigin.value.set(
      Math.floor(cam.x / UV_SNAP) * UV_SNAP,
      Math.floor(cam.z / UV_SNAP) * UV_SNAP
    );

    u.uWetness.value = w.wetness;
    // Puddles fill in seconds and drain over minutes. The asymmetry is what
    // makes them read as accumulation rather than as a slider.
    const target = clamp01(w.rainIntensity * 1.15) * clamp01(w.wetness * 1.4);
    this._puddleAmount = damp(this._puddleAmount, target,
      target > this._puddleAmount ? 0.35 : 0.08, dt);
    u.uPuddle.value = this._puddleAmount;
    u.uRipple.value = clamp01(w.rainIntensity * 1.6) * 0.55;

    this._snowCover = damp(this._snowCover, clamp01(w.snowIntensity * 1.3), 0.05, dt);
    u.uSnow.value = this._snowCover;

    const sun = state.sun;
    u.uSunDir.value.copy(sun.direction);

    const murk = clamp01(w.fogDensity * 300);
    const fog = this.ctx.scene.fog;

    // -- Aerial perspective ------------------------------------------------
    // sigma is a genuine per-metre extinction coefficient: weather.js derives
    // its own `visibility = 3 / fogDensity` (Koschmieder, 5% contrast) from the
    // same number, which is what pins the units down. We deliberately do NOT
    // read scene.fog.density instead - a FogExp2 density is a different
    // quantity that enters the exponent squared, and treating one as the other
    // is exactly how a scene ends up washed out three metres from the camera.
    // Its COLOUR is still honoured below so ground and sky agree on the horizon.
    const sigma = Math.max(w.fogDensity, 1e-6);

    // The mist layer rides the ground rather than absolute zero, so it stays in
    // the grass when the player walks up a hill instead of draining into it.
    // Damped, because snapping it to a per-frame ground sample would make the
    // whole near-field haze breathe in time with the player's footsteps.
    const mistTarget = state.player.groundHeight - 0.2;
    if (this._mistBaseInit) {
      this._mistBase = damp(this._mistBase, mistTarget, 3.0, dt);
    } else {
      this._mistBase = mistTarget;
      this._mistBaseInit = true;
    }

    // Thicker air is also shallower air: fog is a boundary-layer phenomenon and
    // it stratifies harder as it thickens, while the mist deepens as it fills.
    u.uGroundHaze.value.set(sigma * HAZE_SHARE, 1 / lerp(1400, 420, murk), 0, 0);
    u.uGroundMist.value.set(sigma * MIST_SHARE, 1 / lerp(5.5, 14.0, murk), this._mistBase, 0);

    // The horizon must dissolve into the same colour the sky draws there, or
    // the join between ground and sky becomes a visible line.
    _col.copy(state.sky.horizonColor).lerp(w.fogColor, 0.35 + 0.45 * murk);
    if (fog && fog.color) _col.lerp(fog.color, 0.5);
    u.uFogColor.value.copy(_col);
    // Zenith end of the inscatter gradient. Looking down at ground below you,
    // the air between is lit from above, and it is bluer than the horizon.
    u.uFogZenith.value.copy(state.sky.zenithColor)
      .multiplyScalar(state.sky.ambientIntensity)
      .lerp(_col, 0.30 + 0.55 * murk);
    // Ground mist is brighter than the haze above it - it is a dense scatterer
    // sitting directly under the sky's whole hemisphere - and it warms toward
    // the sun far more readily.
    _col2.copy(_col).multiplyScalar(1.06).lerp(sun.color, 0.10 * sun.visibility);
    u.uMistColor.value.copy(_col2);

    _col2.copy(sun.color).multiplyScalar(0.055 * sun.visibility * clamp01(sun.intensity * 0.3));
    u.uFogSunColor.value.copy(_col2);
    u.uMieG.value = lerp(0.70, 0.42, murk);      // haze scatters more isotropically than clear air

    u.uSkyColor.value.copy(state.sky.zenithColor).lerp(state.sky.horizonColor, 0.55)
      .multiplyScalar(state.sky.ambientIntensity);

    // -- LEAF 3: track the blade field's reach ------------------------------
    this._meadowEnd = damp(this._meadowEnd, this._meadowReach(), 1.2, dt);
    // `_meadowNear` pulls the earth -> meadow blend toward the camera. With a
    // tall sward the blades hide the ground and honest soil under them is right;
    // with a mown one they cannot, and bare earth showing between short blades
    // is the single thing that makes a lawn look broken. world/grass.js sets
    // this from its sward mode. 1 = unchanged, 0 = fully green at the camera.
    const nearPull = this._meadowNear ?? 1;
    u.uMeadowFade.value.set(
      this._meadowEnd * MEADOW_START_FRAC * nearPull,
      this._meadowEnd * MEADOW_END_FRAC * Math.max(nearPull, 0.06)
    );

    // -- Cloud shadows -----------------------------------------------------
    // Re-read the map every frame. clouds.js disposes and reallocates its
    // shadow target on a quality change, so a reference taken once in link()
    // silently becomes a handle to freed memory. Swapping the uniform value is
    // free; only the map appearing or disappearing costs a recompile.
    if (this._pollCloudShadow()) this._rebuildMaterials();
    this._checkCloudShadow();

    const c = state.clouds;
    // A cast shadow seen through thick air has less contrast left - but the
    // aerial term already removes most of that per pixel, so this stays gentle
    // (0.84 at partly-cloudy) rather than the 0.6 it used to be, which was
    // quietly halving the best lighting cue in the scene at the exact weather
    // the scene spends most of its time in.
    const murkFade = 1 - murk * 0.25;
    // How BROKEN the deck is, as opposed to how covered. A solid overcast has no
    // edges to cast: its dimming is uniform and sky/atmosphere.js has already
    // taken it out of the ambient, so adding a second uniform dip here would
    // double-count it. Only a broken deck should modulate ambient locally.
    const broken = smoothstep(0.05, 0.35, c.coverage) * (1 - smoothstep(0.72, 0.98, c.coverage));
    const cp = u.uCloudParams.value;
    if (this._cloudMap) {
      // clouds.js raymarches real transmittance and already folds in its own
      // shadowStrength and low-sun fade, so multiplying by sun.visibility again
      // here would darken the ground twice for the same reason. z carries the
      // map's reference altitude - the ground height it was rendered at, which
      // is this same field, so the two cannot drift apart.
      cp.set(murkFade, cp.y, state.player.groundHeight, cp.w);
      u.uCloudSky.value = 0.34 * murkFade * clamp01(broken * 1.2);
    } else {
      // Analytic field. Strongest under a broken deck, gone at night.
      const strength = clamp01(broken * 0.85) * sun.visibility * murkFade;
      cp.set(strength * 0.72, lerp(0.62, 0.20, c.coverage), c.altitude, 1 / 1150);
      u.uCloudSky.value = 0.30;
    }

    const wind = state.wind;
    // Drift is wrapped into [0,1) - one whole tile of tCloud - so the offset
    // stays seamless and never loses float precision as elapsed time grows.
    const drift = state.time.elapsed * c.speed * wind.strength * 0.0022;
    u.uCloudDrift.value.set(
      mod(-wind.direction.x * drift, 1),
      mod(-wind.direction.y * drift, 1),
      mod(-wind.direction.x * drift * 2.3, 1),
      mod(-wind.direction.y * drift * 2.3, 1)
    );

    // Petal litter follows the tree, and thickens in a storm. With no tree in
    // the scene there is nothing to have shed them, so the amount goes to zero
    // rather than laying an unexplained pink ring around the world origin.
    const litter = u.uPetalLitter.value;
    const tree = this._tree;
    if (tree && tree.position) {
      litter.x = tree.position.x;
      litter.y = tree.position.z;
      litter.z = Math.max(4, tree.canopyRadius || 12);
      // petalBoost runs 0.60 (rain) .. 9.00 (petalStorm) in weather.js, so a
      // linear 0.22/unit mapping pinned the amount at 1.0 for the whole of a
      // petal storm - and at 1.0 the height blend hands petals 100% of the
      // ground under the canopy, which is the flat pink decal this layer exists
      // to avoid. Compressed and capped so the dial stays monotone across the
      // whole range: rain 0.51 -> 19% coverage at the canopy centre, the calm
      // default 0.55 -> 29%, storm 0.61 -> 45%, petalStorm capped at 0.72 ->
      // 77%, which is deep litter with the thatch still coming through it.
      litter.w = clamp(0.55 + (w.petalBoost - 1) * 0.09, 0.34, 0.72);
    } else {
      litter.w = 0;
    }

    if (state.debug.wireframe !== this._wireframe) {
      this._wireframe = state.debug.wireframe;
      for (const m of this._materials) m.wireframe = this._wireframe;
    }
  }

  dispose() {
    for (const root of this._roots.values()) this._releaseTree(root);
    this._roots.clear();
    this._live.length = 0;
    this._selected.length = 0;
    this._queue.length = 0;
    for (const g of this._geoPool) g.dispose();
    this._geoPool.length = 0;
    this._meshPool.length = 0;
    this._geomCount = 0;
    for (const m of this._materials) m.dispose();
    this._materials.length = 0;
    this._index = null;
    // Textures that came from the TextureFactory are the factory's to free; the
    // ones we had to bake ourselves are ours.
    for (const t of this._ownedTex) t.dispose();
    this._ownedTex.length = 0;
    this._tex.detail = this._tex.normalAO = this._tex.macro = this._tex.cloud = null;
    // Stop update() from resurrecting the group after a dispose (HMR tears the
    // module down while the frame loop may still be mid-flight).
    this._uniforms = null;
    if (this.group.parent) this.group.parent.remove(this.group);
  }
}
