/**
 * textures.js - the single place procedural textures are created and cached.
 *
 * ZERO external assets: every byte below is computed on the CPU at runtime.
 * Nothing here runs per frame; everything is generated lazily on first request
 * and memoised for the life of the app, so the cost lands on the loading screen.
 *
 * ---------------------------------------------------------------------------
 * Why the noise here is not just `createNoise()` from core/math.js
 * ---------------------------------------------------------------------------
 * math.js's simplex is the right tool for sampling a field at arbitrary world
 * positions, and it is used below for that. But a *texture* must tile exactly:
 * simplex has no period, so baking it into a repeating map leaves a seam, and a
 * seam on a trunk or on the ground is the single most obvious "this is fake"
 * tell in the whole scene. So the bakers use a periodic gradient lattice built
 * on math.js's own permutation table (`createNoise().perm`) and its `hash3`,
 * which gives byte-exact wrap-around in every axis (verified: |f(x) - f(x+P)|
 * < 1e-14) while reusing the shared randomness. Worley is likewise generated on
 * a wrapped cell grid rather than the unbounded `worley3D` helper.
 *
 * ---------------------------------------------------------------------------
 * Performance strategy (this is a laptop CPU, single threaded, no workers)
 * ---------------------------------------------------------------------------
 *  1. Every fbm octave is evaluated at a resolution matched to its own
 *     frequency (~8 samples per lattice cell) and bilinearly accumulated into
 *     the full-res target with wrap. A 6-octave 1024² field costs 133 ms
 *     instead of 447 ms, and is visually identical because a band-limited
 *     octave carries no information above its own Nyquist rate.
 *  2. Octave frequencies are nudged off exact harmonics (3, 7, 13, 27, ...).
 *     Perlin noise is exactly zero at every lattice point; if octaves share a
 *     common lattice those zeros stack and you get a visible grid.
 *  3. Fields are histogram-normalised to their 1st/99th percentile so tuning a
 *     parameter can never silently wash out or clip a map.
 *  4. Quantisation to 8 bit is Bayer-dithered. Smooth ramps (sky-facing
 *     roughness, macro variation, cloud density) band badly otherwise.
 *  5. Every sprite-like element (lenticels, fallen petals, stars, grass blades,
 *     thatch, stamens) is rasterised into its own *oriented* bounding box,
 *     never by testing every pixel against every element.
 *  6. Both Worley generators prune neighbour cells by a lower bound on the
 *     distance to the cell's own face, which is exact and skips roughly half
 *     the 3×3 (or 3×3×3) search.
 *
 * ---------------------------------------------------------------------------
 * Why the surfaces do not look procedural - the anti-repetition strategy
 * ---------------------------------------------------------------------------
 * A tiled surface gives itself away in two distinct ways and they need
 * different fixes, which is why the first version of this file failed on both.
 *
 * The first is *feature shape*. An inverted-F1 Worley field is a radial cone
 * centred on each feature point, so every feature is the same round dome on a
 * grid of the same pitch, however hard the points are jittered: bubble wrap.
 * The soil, moss and grit maps were 70% that. Granular surfaces are built here
 * instead from `worley2Agg`, which returns a per-cell *flat* value (`id`) and
 * the narrowed cell wall, and reserves the cone (`cells`) for the one surface
 * that really is domed - moss cushions. Everything strand-like (thatch, moss
 * shoots, stamens, twigs, litter) is stamped as a sprite, because no noise
 * field produces a strand.
 *
 * The second is *landmark repetition*. The eye locks onto the largest, highest
 * contrast blob in a tile and then finds it again one tile away. Landmarks are
 * low-frequency by definition, so the detail maps have no business carrying low
 * frequencies at all: source fields go through `highpassField`, the assembled
 * albedo goes through `flattenLowFrequency`, and the broad drift they no longer
 * carry is delegated entirely to `groundMacro`, which tiles ~48× slower.
 *
 * Every feature frequency in the ground, moss, litter and bark bakers is
 * expressed as a physical size divided by that map's world tile (published as
 * `userData.tileMeters`), so "12 mm grain" stays 12 mm when the quality tier
 * changes the resolution. Getting that wrong by a factor of five is the other
 * way procedural ground fails: 110 mm sakura petals, 330 mm soil clods.
 *
 * ---------------------------------------------------------------------------
 * One thing here reaches into another module's ground, deliberately
 * ---------------------------------------------------------------------------
 * `FACTORY_OWNED` makes this file authoritative for world/terrain.js's three
 * ground maps even when terrain.js passes its own generator, because that file
 * documents its local bake as the fallback for a factory that cannot supply
 * them and this one can. Consequences worth knowing before touching it:
 *
 *  - terrain.js's `_bakeDetail` / `_bakeNormalAO` / `_bakeMacro` are now dead
 *    code. Confirm at runtime with `SAKURA.ctx.textures.report().names`: it must
 *    list terrain.detail, terrain.normalAO and terrain.macro.
 *  - Two of terrain.js's constants are calibrated against the *distribution* of
 *    what those maps contain, not merely against their existence, and both are
 *    restated and re-measured here: the splat weights (see `_buildTerrainMacro`)
 *    and the domain-warp Jacobian (see TERRAIN_WARP_GRAD_RMS). A macro map can
 *    break DETAIL_WARP without anyone editing terrain.js, simply by putting the
 *    same contrast at a higher frequency - which the first version of this
 *    suite did, hard enough to fold the mapping at ULTRA.
 *  - The splat balance is genuinely different from the bake it replaces
 *    (soil/grass/moss 25.7/56.2/18.1 against 22.3/65.8/11.9). That is an art
 *    call, made because terrain.js's own author wrote that soil and moss "could
 *    never break the surface" under the old numbers.
 *
 * ---------------------------------------------------------------------------
 * Conventions
 * ---------------------------------------------------------------------------
 *  - Colour maps: `SRGBColorSpace`, authored directly as sRGB bytes.
 *  - Data maps (normal / ORM / noise / masks): `NoColorSpace`, linear bytes.
 *  - ORM packing follows three's own channel expectations so the map can be
 *    assigned to `aoMap`, `roughnessMap` and `metalnessMap` simultaneously:
 *        R = ambient occlusion   G = roughness   B = metalness   A = cavity
 *    (A is a wetness-affinity mask: 1 where water pools. Free, and the wet
 *     look in rain depends on it.)
 *  - Normal maps are OpenGL/three convention (+Y green up the V axis) and are
 *    generated with wrapped Sobel differences, so they tile with the albedo.
 *    Alpha of a normal map carries the source height, useful for blending.
 *  - Alpha cutout cards store a *ramped* alpha (roughly a signed distance
 *    remapped so 0.5 is the true edge) rather than a hard step. Hard alpha
 *    erodes to nothing after two mip levels; a ramp survives. Use
 *    `alphaTest: 0.5` and, if available, `alphaToCoverage`.
 *  - Every tiled surface and every cutout card ships a *hand-built* mip chain
 *    rather than calling glGenerateMipmap: sRGB albedo is filtered in linear
 *    light, normals are averaged as vectors and the variance they lose is added
 *    back to the matching roughness level, and cutout alpha is rescaled to hold
 *    its coverage. See SECTION 2b. Consumers need do nothing - the chain is on
 *    `texture.mipmaps` with `generateMipmaps = false`, which three uploads as-is.
 *  - `userData.tileMeters` on a tiling map says how much world one repeat covers
 *    (a number, or `{u, v}` for bark). Set your repeat from it rather than
 *    guessing; the feature sizes were authored against it.
 */

import * as THREE from 'three';
import {
  clamp,
  clamp01,
  lerp,
  smoothstep,
  hash2,
  hash3,
  makeRNG,
  createNoise,
  fibonacciSphere,
  kelvinToRGB,
  TAU,
  PI,
} from './math.js';
import { EVENTS } from './state.js';

// ===========================================================================
// SECTION 1 - periodic noise kernels
// ===========================================================================

/** 12 edge-midpoint gradients (Perlin's improved set): uniform length, no bias. */
const GRAD3 = new Float32Array([
  1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1, 0,
  1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, -1,
  0, 1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1,
]);

/** 8 unit gradients at 45° steps. Equal length matters: mixing (1,1) with
 *  (1,0) biases the field toward the diagonals and shows as a plaid weave. */
const S2 = Math.SQRT1_2;
const GRAD2 = new Float32Array([
  1, 0, S2, S2, 0, 1, -S2, S2, -1, 0, -S2, -S2, 0, -1, S2, -S2,
]);

/** Permutation table borrowed from math.js so this file adds no new randomness. */
const PERM = createNoise(0x5a17).perm; // Uint8Array(512)
const P12 = new Uint8Array(512);
for (let i = 0; i < 512; i++) P12[i] = PERM[i] % 12;

/**
 * `hash2`/`hash3` in core/math.js document themselves as 0..1 and are used that
 * way throughout this file - but they actually return exactly uniform on
 * [0, 0.5). The final `* 1274126177` is a float64 multiply of an int32, so the
 * product needs 62 bits, float64 keeps 53, and the sign bit of the ToInt32 that
 * follows is therefore always clear. Measured over 200k samples: min 0.00000,
 * max 0.50000, mean 0.2500, decile counts 20/20/20/20/20/0/0/0/0/0 - flat,
 * correct, and on half the interval.
 *
 * core/math.js is shared and not ours to edit, so the correction lives here.
 * One multiply restores the interval with no change to the spectral character
 * or the determinism. world/terrain.js reached the same conclusion
 * independently and applies the same factor to its fallback bakes.
 *
 * This was NOT cosmetic, and it was breaking the single thing this file exists
 * to prevent:
 *
 *  - `worley2Fields` places its feature point at `i + c0 + jitter * hash3(...)`
 *    with `c0 = 0.5 - 0.5*jitter`, i.e. jitter 1 is meant to span the whole
 *    cell. On [0, 0.5) it spanned the lower-LEFT QUADRANT of every cell
 *    (measured offsets 0.002..0.499 in both axes, octile histogram
 *    101/114/102/124/0/0/0/0). One feature point per cell confined to a quarter
 *    of that cell is a regular sub-lattice wearing a noise function's name - 
 *    precisely the "identical blobs on a grid of the same pitch" failure the
 *    header, `worley2Agg` and world/terrain.js all describe. It affected the
 *    shipped terrain detail map, the moss cushions, the bark plates and the
 *    ground clods.
 *  - `worley3` placed its cloud-volume points in the same way, in the lower
 *    corner octant.
 *  - `worley2Agg().id` - the per-fragment FLAT value, returned unnormalised and
 *    consumed as `(id - 0.5)` - came out mean 0.251, max 0.500, so the term was
 *    always negative and carried half its intended spread. The crumb mosaic,
 *    which is the whole reason `id` exists, was contributing half of what the
 *    amplitudes above say it does.
 *  - `value2P` returned [-1, 0) rather than [-1, 1]. Harmless where a
 *    `normalizeField` follows (which is everywhere today) and a trap for the
 *    next caller.
 */
const HASH_SCALE = 2;
const h2 = (x, y) => hash2(x, y) * HASH_SCALE;
const h3 = (x, y, z) => hash3(x, y, z) * HASH_SCALE;

/** Quintic fade - C2 continuous, so second derivatives (and therefore normal
 *  maps derived from the field) stay smooth. */
const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);

/**
 * Lattice gradient lookup. The multiplicative mixing (rather than the classic
 * `perm[perm[x] + y]`) does two things: it decorrelates the field from its own
 * transpose, and it makes `seed` a genuinely different field instead of the
 * same field translated along x - which is what plain additive seeding gives
 * you, and why "different seed, same texture" bugs are so common.
 */
function gi2(x, y, s) {
  return (PERM[(PERM[(x * 3 + s) & 255] * 5 + y * 7 + s * 13 + 29) & 255] & 7) * 2;
}
function gi3(x, y, z, s) {
  const a = PERM[(x * 3 + s) & 255];
  const b = PERM[(a * 5 + y * 7 + s * 13 + 29) & 255];
  return P12[(b * 3 + z * 11 + s * 29 + 131) & 255] * 3;
}

/** Periodic 2D gradient noise. Periods must be integers <= 256. Range ~[-1,1]. */
function perlin2P(x, y, px, py, s) {
  const X = Math.floor(x), Y = Math.floor(y);
  const fx = x - X, fy = y - Y;
  const x0 = ((X % px) + px) % px, x1 = (x0 + 1) % px;
  const y0 = ((Y % py) + py) % py, y1 = (y0 + 1) % py;
  const u = fade(fx), v = fade(fy);
  const fx1 = fx - 1, fy1 = fy - 1;
  const g00 = gi2(x0, y0, s), g10 = gi2(x1, y0, s);
  const g01 = gi2(x0, y1, s), g11 = gi2(x1, y1, s);
  const n00 = GRAD2[g00] * fx + GRAD2[g00 + 1] * fy;
  const n10 = GRAD2[g10] * fx1 + GRAD2[g10 + 1] * fy;
  const n01 = GRAD2[g01] * fx + GRAD2[g01 + 1] * fy1;
  const n11 = GRAD2[g11] * fx1 + GRAD2[g11 + 1] * fy1;
  const a = n00 + u * (n10 - n00);
  const b = n01 + u * (n11 - n01);
  return (a + v * (b - a)) * 1.4142;
}

/** Periodic 3D gradient noise. Range ~[-1,1]. */
function perlin3P(x, y, z, px, py, pz, s) {
  const X = Math.floor(x), Y = Math.floor(y), Z = Math.floor(z);
  const fx = x - X, fy = y - Y, fz = z - Z;
  const x0 = ((X % px) + px) % px, x1 = (x0 + 1) % px;
  const y0 = ((Y % py) + py) % py, y1 = (y0 + 1) % py;
  const z0 = ((Z % pz) + pz) % pz, z1 = (z0 + 1) % pz;
  const u = fade(fx), v = fade(fy), w = fade(fz);
  const fx1 = fx - 1, fy1 = fy - 1, fz1 = fz - 1;
  const g000 = gi3(x0, y0, z0, s), g100 = gi3(x1, y0, z0, s);
  const g010 = gi3(x0, y1, z0, s), g110 = gi3(x1, y1, z0, s);
  const g001 = gi3(x0, y0, z1, s), g101 = gi3(x1, y0, z1, s);
  const g011 = gi3(x0, y1, z1, s), g111 = gi3(x1, y1, z1, s);
  const n000 = GRAD3[g000] * fx + GRAD3[g000 + 1] * fy + GRAD3[g000 + 2] * fz;
  const n100 = GRAD3[g100] * fx1 + GRAD3[g100 + 1] * fy + GRAD3[g100 + 2] * fz;
  const n010 = GRAD3[g010] * fx + GRAD3[g010 + 1] * fy1 + GRAD3[g010 + 2] * fz;
  const n110 = GRAD3[g110] * fx1 + GRAD3[g110 + 1] * fy1 + GRAD3[g110 + 2] * fz;
  const n001 = GRAD3[g001] * fx + GRAD3[g001 + 1] * fy + GRAD3[g001 + 2] * fz1;
  const n101 = GRAD3[g101] * fx1 + GRAD3[g101 + 1] * fy + GRAD3[g101 + 2] * fz1;
  const n011 = GRAD3[g011] * fx + GRAD3[g011 + 1] * fy1 + GRAD3[g011 + 2] * fz1;
  const n111 = GRAD3[g111] * fx1 + GRAD3[g111 + 1] * fy1 + GRAD3[g111 + 2] * fz1;
  const x00 = n000 + u * (n100 - n000), x10 = n010 + u * (n110 - n010);
  const x01 = n001 + u * (n101 - n001), x11 = n011 + u * (n111 - n011);
  const a = x00 + v * (x10 - x00), b = x01 + v * (x11 - x01);
  return (a + w * (b - a)) * 1.1547;
}

/**
 * Periodic 2D *value* noise, range ~[-1,1].
 *
 * Gradient noise is identically zero at every lattice point. At the 3-5 texels
 * per cell that a soil-grain field wants, those zeros are only a few texels
 * apart and read as a faint regular grid through the whole map - the exact
 * failure this file exists to avoid. Value noise has no zero set, costs half as
 * much (four hashes and three lerps, no dot products), and its slightly blockier
 * character is invisible once it is the *finest* octave in the stack.
 */
function value2P(x, y, px, py, s) {
  const X = Math.floor(x), Y = Math.floor(y);
  const u = fade(x - X), v = fade(y - Y);
  const x0 = ((X % px) + px) % px, x1 = (x0 + 1) % px;
  const y0 = ((Y % py) + py) % py, y1 = (y0 + 1) % py;
  const a = h3(x0, y0, s), b = h3(x1, y0, s);
  const c = h3(x0, y1, s), d = h3(x1, y1, s);
  const t = a + (b - a) * u;
  return (t + (c + (d - c) * u - t) * v) * 2 - 1;
}

/**
 * Octave frequency ladder. Rounded to integers (required for periodicity) and
 * pushed off exact multiples of the previous octave so lattice zeros - Perlin
 * is identically zero at every lattice point - do not stack into a grid.
 */
function freqLadder(base, octaves, lacunarity) {
  const out = new Int32Array(octaves);
  let f = Math.max(1, Math.round(base));
  for (let o = 0; o < octaves; o++) {
    let fi = Math.max(1, Math.min(256, Math.round(f)));
    if (o > 0) {
      const prev = out[o - 1];
      // Break harmonic alignment; also never repeat the previous frequency.
      let guard = 0;
      while ((fi % prev === 0 || prev % fi === 0 || fi === prev) && fi < 256 && guard++ < 4) fi++;
    }
    out[o] = fi;
    f *= lacunarity;
  }
  return out;
}

/** Working resolution for one octave: ~`spc` samples per lattice cell. */
function octaveRes(size, freq, spc) {
  return Math.min(size, Math.max(8, Math.ceil(freq * spc)));
}

/**
 * How many octaves may be asked for before the finest one crosses Nyquist.
 *
 * `octaves` is a property of the *look* and the resolution is a property of the
 * tier, so a hand-picked octave count is a Nyquist bug waiting for a quality
 * change. Measured on the maps below: at LOW the ground's 27 mm grain ran two
 * octaves from 74 cells, i.e. 148 cells over 256 texels - 1.7 texels a cell,
 * well past the base mip's Nyquist rate - and the moss fuzz was worse at 1.6.
 * Both showed as salt-and-pepper that sparkles in the base mip and vanishes one
 * level down, which is the worst of both: it costs a full-resolution pass to
 * compute and contributes nothing but aliasing.
 *
 * @param {number} baseSpc texels per lattice cell of the coarsest octave
 *   (`min(w/fx, h/fy)` - the tighter axis is the one that aliases first)
 * @param {number} lacunarity @param {number} wanted the count the caller asked for
 * @param {number} minSpc floor in texels a cell; 3.2 is what this file uses
 *   everywhere else it derives a frequency from a resolution.
 */
function octavesFor(baseSpc, lacunarity, wanted, minSpc = 3.2) {
  let o = 1, s = baseSpc;
  while (o < wanted && s / lacunarity >= minSpc) { s /= lacunarity; o++; }
  return o;
}

/**
 * RMS of the wrapped *central-difference* gradient of a field, per texel.
 *
 * Deliberately the central difference and not `gradientRMS`'s Sobel: this
 * number is compared against the domain-warp Jacobian bound world/terrain.js
 * published, that bound was measured with a central difference, and a Sobel
 * reads exactly twice as large on the same ramp. Getting the operator wrong
 * would silently halve the safety margin.
 */
function centralGradRMS(field, w, h) {
  let acc = 0;
  for (let y = 0; y < h; y++) {
    const r0 = ((((y - 1) % h) + h) % h) * w, r1 = y * w, r2 = ((y + 1) % h) * w;
    for (let x = 0; x < w; x++) {
      const x0 = (((x - 1) % w) + w) % w, x1 = (x + 1) % w;
      const gx = (field[r1 + x1] - field[r1 + x0]) * 0.5;
      const gy = (field[r2 + x] - field[r0 + x]) * 0.5;
      acc += gx * gx + gy * gy;
    }
  }
  return Math.sqrt(acc / (w * h));
}

/**
 * Wrapped bilinear upsample of `src` (rw×rh) accumulated into `dst` (w×h).
 * The half-texel convention (`(i+0.5)/n`) is what keeps the low-res octave
 * phase-aligned with the full-res one; get it wrong and octaves drift apart
 * and the field loses contrast.
 */
function upsampleAdd2(src, rw, rh, dst, w, h, amp) {
  const ux = new Int32Array(w), vx = new Int32Array(w);
  const fxa = new Float32Array(w);
  for (let x = 0; x < w; x++) {
    const sx = ((x + 0.5) / w) * rw - 0.5;
    const x0 = Math.floor(sx);
    fxa[x] = sx - x0;
    ux[x] = ((x0 % rw) + rw) % rw;
    vx[x] = (ux[x] + 1) % rw;
  }
  for (let y = 0; y < h; y++) {
    const sy = ((y + 0.5) / h) * rh - 0.5;
    const y0 = Math.floor(sy);
    const fy = sy - y0;
    const ya = (((y0 % rh) + rh) % rh) * rw;
    const yb = ((((y0 + 1) % rh) + rh) % rh) * rw;
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const xa = ux[x], xb = vx[x], fx = fxa[x];
      const a = src[ya + xa], b = src[ya + xb];
      const c = src[yb + xa], d = src[yb + xb];
      const t = a + (b - a) * fx;
      dst[row + x] += amp * (t + (c + (d - c) * fx - t) * fy);
    }
  }
}

/** Wrapped trilinear upsample-accumulate, cubic volumes only. */
function upsampleAdd3(src, r, dst, n, amp) {
  const ia = new Int32Array(n), ib = new Int32Array(n);
  const fr = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const s = ((i + 0.5) / n) * r - 0.5;
    const i0 = Math.floor(s);
    fr[i] = s - i0;
    ia[i] = ((i0 % r) + r) % r;
    ib[i] = (ia[i] + 1) % r;
  }
  const rr = r * r;
  for (let z = 0; z < n; z++) {
    const za = ia[z] * rr, zb = ib[z] * rr, fz = fr[z];
    for (let y = 0; y < n; y++) {
      const ya = ia[y] * r, yb = ib[y] * r, fy = fr[y];
      const o000 = za + ya, o010 = za + yb, o100 = zb + ya, o110 = zb + yb;
      const row = (z * n + y) * n;
      for (let x = 0; x < n; x++) {
        const xa = ia[x], xb = ib[x], fx = fr[x];
        const c00 = src[o000 + xa] + (src[o000 + xb] - src[o000 + xa]) * fx;
        const c01 = src[o010 + xa] + (src[o010 + xb] - src[o010 + xa]) * fx;
        const c10 = src[o100 + xa] + (src[o100 + xb] - src[o100 + xa]) * fx;
        const c11 = src[o110 + xa] + (src[o110 + xb] - src[o110 + xa]) * fx;
        const c0 = c00 + (c01 - c00) * fy;
        const c1 = c10 + (c11 - c10) * fy;
        dst[row + x] += amp * (c0 + (c1 - c0) * fz);
      }
    }
  }
}

/** Per-octave shaping. 0 = signed fbm, 1 = billow (puffy), 2 = ridged (creases). */
function shapeOctave(n, kind) {
  if (kind === 1) return Math.abs(n) * 2 - 1;
  if (kind === 2) {
    const r = 1 - Math.abs(n);
    return r * r * 2 - 1;
  }
  return n;
}

/**
 * 2D fractal field, anisotropic and tiling.
 * opts: { fx, fy, octaves, lacunarity, gain, seed, kind, spc }
 * `fx`/`fy` are lattice periods across the *whole* tile, so an fy of 2 with an
 * fx of 20 gives features 10× taller than they are wide - which is how bark
 * fissures and grass striations are built.
 */
function fbm2(w, h, opts) {
  const octaves = opts.octaves ?? 4;
  const lac = opts.lacunarity ?? 2;
  const gain = opts.gain ?? 0.5;
  const kind = opts.kind ?? 0;
  const seed = opts.seed ?? 0;
  const spc = opts.spc ?? 8;
  const dst = new Float32Array(w * h);
  const lx = freqLadder(opts.fx, octaves, lac);
  const ly = freqLadder(opts.fy ?? opts.fx, octaves, lac);
  let amp = 1, norm = 0;
  for (let o = 0; o < octaves; o++) { norm += amp; amp *= gain; }
  amp = 1 / norm;
  for (let o = 0; o < octaves; o++) {
    const fx = lx[o], fy = ly[o];
    const rw = octaveRes(w, fx, spc);
    const rh = octaveRes(h, fy, spc);
    const s = seed + o * 101 + 7;
    if (rw === w && rh === h) {
      for (let y = 0; y < h; y++) {
        const vy = ((y + 0.5) / h) * fy;
        const row = y * w;
        for (let x = 0; x < w; x++) {
          dst[row + x] += amp * shapeOctave(perlin2P(((x + 0.5) / w) * fx, vy, fx, fy, s), kind);
        }
      }
    } else {
      const tmp = new Float32Array(rw * rh);
      for (let y = 0; y < rh; y++) {
        const vy = ((y + 0.5) / rh) * fy;
        const row = y * rw;
        for (let x = 0; x < rw; x++) {
          tmp[row + x] = shapeOctave(perlin2P(((x + 0.5) / rw) * fx, vy, fx, fy, s), kind);
        }
      }
      upsampleAdd2(tmp, rw, rh, dst, w, h, amp);
    }
    amp *= gain;
  }
  return dst;
}

/**
 * Tiling fractal *value*-noise field, evaluated at full resolution.
 *
 * This is the grain generator. It is deliberately not routed through the
 * reduced-resolution octave machinery above: the whole point of these octaves is
 * that they sit within a few texels of Nyquist, so there is no lower resolution
 * to evaluate them at. Returned unnormalised in roughly [-1, 1].
 */
function valueFbm2(w, h, opts) {
  const octaves = opts.octaves ?? 2;
  const lac = opts.lacunarity ?? 2;
  const gain = opts.gain ?? 0.5;
  const seed = opts.seed ?? 0;
  const dst = new Float32Array(w * h);
  const lx = freqLadder(opts.fx, octaves, lac);
  const ly = freqLadder(opts.fy ?? opts.fx, octaves, lac);
  let amp = 1, norm = 0;
  for (let o = 0; o < octaves; o++) { norm += amp; amp *= gain; }
  amp = 1 / norm;
  for (let o = 0; o < octaves; o++) {
    const fx = lx[o], fy = ly[o];
    const s = seed + o * 617 + 3;
    const sx = fx / w, sy = fy / h;
    for (let y = 0; y < h; y++) {
      const vy = (y + 0.5) * sy;
      const row = y * w;
      for (let x = 0; x < w; x++) dst[row + x] += amp * value2P((x + 0.5) * sx, vy, fx, fy, s);
    }
    amp *= gain;
  }
  return dst;
}

/**
 * Periodic domain-warp displacement, in *cell* units of a target lattice.
 *
 * Built at a resolution matched to its own (low) frequency and wrap-upsampled,
 * because a warp field is by construction smooth: evaluating it per texel at
 * full resolution is pure waste, and it is evaluated per texel by every caller
 * that uses it. Returned as two Float32Arrays of length w*h.
 */
function warpField2(w, h, fx, fy, seed, ampX, ampY, spc = 6) {
  // Capped below one cell. Every consumer of this field folds the warped sample
  // back into one period with a single compare per axis, which is only valid
  // while the displacement cannot cross a whole cell.
  ampX = Math.min(0.9, Math.abs(ampX)) * Math.sign(ampX || 1);
  ampY = Math.min(0.9, Math.abs(ampY)) * Math.sign(ampY || 1);
  const rw = octaveRes(w, fx, spc);
  const rh = octaveRes(h, fy, spc);
  const lowX = new Float32Array(rw * rh);
  const lowY = new Float32Array(rw * rh);
  for (let y = 0; y < rh; y++) {
    const v = ((y + 0.5) / rh) * fy;
    const row = y * rw;
    for (let x = 0; x < rw; x++) {
      const u = ((x + 0.5) / rw) * fx;
      lowX[row + x] = ampX * perlin2P(u, v, fx, fy, seed + 401);
      lowY[row + x] = ampY * perlin2P(u, v, fx, fy, seed + 823);
    }
  }
  const wx = new Float32Array(w * h);
  const wy = new Float32Array(w * h);
  upsampleAdd2(lowX, rw, rh, wx, w, h, 1);
  upsampleAdd2(lowY, rw, rh, wy, w, h, 1);
  return { wx, wy };
}

/** Cubic 3D fractal field, tiling. opts: { f, octaves, lacunarity, gain, seed, kind, spc } */
function fbm3(n, opts) {
  const octaves = opts.octaves ?? 3;
  const lac = opts.lacunarity ?? 2;
  const gain = opts.gain ?? 0.5;
  const kind = opts.kind ?? 0;
  const seed = opts.seed ?? 0;
  const spc = opts.spc ?? 6;
  const dst = new Float32Array(n * n * n);
  const ladder = freqLadder(opts.f, octaves, lac);
  let amp = 1, norm = 0;
  for (let o = 0; o < octaves; o++) { norm += amp; amp *= gain; }
  amp = 1 / norm;
  for (let o = 0; o < octaves; o++) {
    const f = ladder[o];
    const r = octaveRes(n, f, spc);
    const s = seed + o * 211 + 13;
    const tmp = r === n ? dst : new Float32Array(r * r * r);
    const scale = f / r;
    let i = 0;
    for (let z = 0; z < r; z++) {
      const wz = (z + 0.5) * scale;
      for (let y = 0; y < r; y++) {
        const wy = (y + 0.5) * scale;
        for (let x = 0; x < r; x++) {
          const v = shapeOctave(perlin3P((x + 0.5) * scale, wy, wz, f, f, f, s), kind);
          if (r === n) dst[i++] += amp * v;
          else tmp[i++] = v;
        }
      }
    }
    if (r !== n) upsampleAdd3(tmp, r, dst, n, amp);
    amp *= gain;
  }
  return dst;
}

/**
 * Tiling 2D Worley, anisotropic, computed at an arbitrary working resolution.
 *
 * Returns the raw F1 and (optionally) F2 *distances* rather than a shaped mask.
 * That distinction matters: a distance field is piecewise near-linear, so it
 * upsamples with almost no error, whereas the shaped `1 - d` mask has a cusp at
 * every feature point that bilinear magnification rounds off. Shaping is
 * therefore always applied by the caller, at full resolution, after upsampling.
 *
 * The per-column wrap tables hoist all of the modulo/branch work out of the
 * 9-cell inner loop, which is otherwise the single hottest loop in this file.
 */
function worley2Fields(w, h, fx, fy, seed, jitter, wantF2, warp = 0, extra = null) {
  const wantId = !!(extra && extra.wantId);
  const pts = new Float32Array(fx * fy * 2);
  const cid = wantId ? new Float32Array(fx * fy) : null;
  const c0 = 0.5 - 0.5 * jitter;
  for (let j = 0; j < fy; j++) {
    for (let i = 0; i < fx; i++) {
      const o = (j * fx + i) * 2;
      // h3, not hash3: the interval matters here more than anywhere else in the
      // file - see HASH_SCALE. On [0, 0.5) `c0 + jitter*h` spans a quarter of
      // the cell however large `jitter` is, which is a lattice, not a jitter.
      pts[o] = i + c0 + jitter * h3(i, j, seed);
      pts[o + 1] = j + c0 + jitter * h3(i + 977, j, seed + 31);
      // Per-cell random value. This is what turns a cellular field from a grid
      // of identical round domes into a mosaic of distinct fragments: the cell
      // *interior* gets its own flat tone/height instead of inheriting the
      // radial distance profile that makes worley read as bubble wrap.
      // Consumed as `(id - 0.5)` by every caller, so it must be centred on 0.5
      // and span 0..1 - see HASH_SCALE.
      if (cid) cid[j * fx + i] = h3(i + 313, j + 571, seed + 97);
    }
  }

  // Domain warp. One feature point per cell of a square grid is *always*
  // readable as a square grid, however hard the points are jittered - it was
  // plainly visible as a woven/brick pattern in the soil and moss maps.
  // Displacing the lookup by a noise field at a frequency near the cell pitch
  // destroys the axis alignment while leaving the cell topology intact, which
  // is exactly the organic-but-cellular look soil aggregates and moss cushions
  // have. Tiling survives because the warp field is itself periodic over the
  // tile, so u -> u+1 moves the sample by exactly fx cells.
  let warpX = extra && extra.warpXY ? extra.warpXY.wx : null;
  let warpY = extra && extra.warpXY ? extra.warpXY.wy : null;
  if (!warpX && warp > 0) {
    const wfx = Math.max(2, Math.min(256, Math.round(fx * 0.75)));
    const wfy = Math.max(2, Math.min(256, Math.round(fy * 0.75)));
    const f = warpField2(w, h, wfx, wfy, seed, warp, warp * (fy / fx));
    warpX = f.wx;
    warpY = f.wy;
  }
  // Cells are square in *texture* space, not in cell-index space, so an
  // elongated grid (fx 20 / fy 3) still yields round features rather than
  // features stretched by the grid aspect on top of the intended stretch.
  const ay = fx / fy;

  const sx = fx / w, sy = fy / h;
  const colI = new Int32Array(w * 3);
  const colO = new Float32Array(w * 3);
  const colX = new Float32Array(w);
  for (let x = 0; x < w; x++) {
    const wx = (x + 0.5) * sx;
    colX[x] = wx;
    const cx = Math.floor(wx);
    for (let k = 0; k < 3; k++) {
      let ii = cx + k - 1, off = 0;
      if (ii < 0) { ii += fx; off = -fx; } else if (ii >= fx) { ii -= fx; off = fx; }
      colI[x * 3 + k] = ii;
      colO[x * 3 + k] = off;
    }
  }

  const f1 = new Float32Array(w * h);
  const f2 = wantF2 ? new Float32Array(w * h) : null;
  const id = wantId ? new Float32Array(w * h) : null;
  const rowBase = new Int32Array(3);
  const rowOff = new Float32Array(3);

  if (warpX) {
    // Warped path: the cell index now depends on the displaced position, so the
    // per-column tables no longer apply. This always runs at the reduced
    // working resolution, where the extra wrap arithmetic is immaterial.
    for (let y = 0; y < h; y++) {
      const by = (y + 0.5) * sy;
      const out = y * w;
      for (let x = 0; x < w; x++) {
        const i = out + x;
        // The warp never exceeds one cell (it is amplitude-capped at build
        // time), so folding back into one period is a single compare per axis
        // rather than a floor and a division - which matters, because this is a
        // full-resolution per-texel loop over a megapixel.
        let wx = (x + 0.5) * sx + warpX[i];
        let wy = by + warpY[i];
        if (wx < 0) wx += fx; else if (wx >= fx) wx -= fx;
        if (wy < 0) wy += fy; else if (wy >= fy) wy -= fy;
        // Same belt-and-braces the 3D generator already had: a tiny negative
        // sample can come back out of the fold as exactly `fx`, and the search
        // below indexes the sample's own cell directly. With cx === fx the
        // +1 neighbour wraps to cell 0 instead of cell 1, so one cell is
        // searched twice and its neighbour not at all - a wrong F1/F2 pair
        // rather than a crash, which is exactly the kind of defect that never
        // gets found by looking at the picture.
        if (wx >= fx) wx = 0;
        if (wy >= fy) wy = 0;
        const cx = Math.floor(wx), cy = Math.floor(wy);
        // Same exact branch-and-bound as the 3D generator: the closest point a
        // neighbouring cell can hold lies on that cell's near face, so the
        // squared distance to the face is a valid lower bound on F1. F2 needs
        // the second-best, so a cell is only skipped once *both* candidates are
        // provably beaten - hence the test against d2, not d1.
        const gx = wx - cx, gy = (wy - cy) * ay;
        BND_X[0] = gx * gx; BND_X[1] = 0; BND_X[2] = (1 - gx) * (1 - gx);
        BND_Y[0] = gy * gy; BND_Y[1] = 0; BND_Y[2] = (ay - gy) * (ay - gy);
        let d1 = 1e9, d2 = 1e9, w1 = 0;
        for (let dj = -1; dj <= 1; dj++) {
          const ly = BND_Y[dj + 1];
          if (ly >= d2) continue;
          let jj = cy + dj, oy = 0;
          if (jj < 0) { jj = fy - 1; oy = -fy; } else if (jj >= fy) { jj = 0; oy = fy; }
          const base = jj * fx;
          for (let di = -1; di <= 1; di++) {
            if (ly + BND_X[di + 1] >= d2) continue;
            let ii = cx + di, ox = 0;
            if (ii < 0) { ii = fx - 1; ox = -fx; } else if (ii >= fx) { ii = 0; ox = fx; }
            const o = (base + ii) * 2;
            const ddx = pts[o] + ox - wx;
            const ddy = (pts[o + 1] + oy - wy) * ay;
            const d = ddx * ddx + ddy * ddy;
            if (d < d1) { d2 = d1; d1 = d; w1 = base + ii; } else if (d < d2) d2 = d;
          }
        }
        f1[i] = Math.sqrt(d1);
        if (f2) f2[i] = Math.sqrt(d2);
        if (id) id[i] = cid[w1];
      }
    }
    return { f1, f2, id };
  }

  for (let y = 0; y < h; y++) {
    const wy = (y + 0.5) * sy;
    const cy = Math.floor(wy);
    for (let k = 0; k < 3; k++) {
      let jj = cy + k - 1, off = 0;
      if (jj < 0) { jj += fy; off = -fy; } else if (jj >= fy) { jj -= fy; off = fy; }
      rowBase[k] = jj * fx;
      rowOff[k] = off;
    }
    const out = y * w;
    for (let x = 0; x < w; x++) {
      const wx = colX[x];
      const c3 = x * 3;
      let d1 = 1e9, d2 = 1e9, w1 = 0;
      for (let kj = 0; kj < 3; kj++) {
        const base = rowBase[kj];
        const oy = rowOff[kj];
        for (let ki = 0; ki < 3; ki++) {
          const ci = base + colI[c3 + ki];
          const o = ci * 2;
          const ddx = pts[o] + colO[c3 + ki] - wx;
          const ddy = (pts[o + 1] + oy - wy) * ay;
          const d = ddx * ddx + ddy * ddy;
          if (d < d1) { d2 = d1; d1 = d; w1 = ci; } else if (d < d2) d2 = d;
        }
      }
      f1[out + x] = Math.sqrt(d1);
      if (f2) f2[out + x] = Math.sqrt(d2);
      if (id) id[out + x] = cid[w1];
    }
  }
  return { f1, f2, id };
}

/**
 * Aggregate structure: the cellular field a granular surface actually has.
 *
 * `cells` (inverted F1) is a radial cone centred on every feature point, which
 * is precisely why worley-driven soil and moss read as bubble wrap - every
 * feature is the same round dome. What real crumb structure looks like is a
 * mosaic of *flat-topped* fragments at assorted heights, separated by thin
 * voids. That is `id` (a flat per-fragment value) plus `walls` (the narrowed
 * F2−F1 crease), and only a trace of `cells`.
 *
 * Computed at full resolution so `id` - which is piecewise constant and would
 * be smeared into ramps or stair-stepped by any resampling - is exact. Only the
 * warp, which is smooth by construction, is evaluated cheaply and upsampled.
 *
 * @returns {{id:Float32Array, walls:Float32Array, cells:Float32Array}} 0..1 fields
 */
function worley2Agg(w, h, fx, fy, seed, opts = {}) {
  const jitter = opts.jitter ?? 1;
  const warp = opts.warp ?? 0.55;
  // Warp lattice a little coarser than the cell lattice: fine enough to break
  // the axis alignment, coarse enough that six samples per warp cell is plenty.
  const wfx = Math.max(2, Math.min(256, Math.round(fx * 0.65)));
  const wfy = Math.max(2, Math.min(256, Math.round(fy * 0.65)));
  const warpXY = warp > 0
    ? warpField2(w, h, wfx, wfy, seed + 5, warp, warp * (fy / fx))
    : null;
  const { f1, f2, id } = worley2Fields(w, h, fx, fy, seed, jitter, true, 0, { warpXY, wantId: true });
  const n = w * h;
  const walls = new Float32Array(n);
  const cells = new Float32Array(n);
  for (let i = 0; i < n; i++) { walls[i] = f1[i] - f2[i]; cells[i] = -f1[i]; }
  return { id, walls: normalizeField(walls), cells: normalizeField(cells) };
}

/**
 * Worley evaluated at a resolution matched to its cell count (`spc` samples per
 * cell) and wrap-upsampled. A 14-cell field over 1024² costs 140² samples
 * instead of 1024² - a 50× saving with no visible difference once the caller's
 * shaping curve is applied at full resolution.
 *
 * mode 0 → inverted F1 (bright at feature points: clods, pebbles, cloud puffs)
 * mode 1 → F2-F1     (bright at cell walls: cracks, plate joints, veins)
 *
 * Output is histogram-normalised to 0..1.
 */
function worley2Scaled(w, h, fx, fy, seed, mode = 0, jitter = 1, spc = 10, warp = 0.5) {
  const rw = octaveRes(w, fx, spc);
  const rh = octaveRes(h, fy, spc);
  const { f1, f2 } = worley2Fields(rw, rh, fx, fy, seed, jitter, mode === 1, warp);
  const src = new Float32Array(rw * rh);
  // F2-F1 vanishes exactly on the cell boundary and grows toward the interior,
  // so the *wall* mask is its negation. Getting this backwards gives cracks
  // that run through the middle of every clod instead of between them.
  if (mode === 1) for (let i = 0; i < src.length; i++) src[i] = f1[i] - f2[i];
  else for (let i = 0; i < src.length; i++) src[i] = -f1[i];
  if (rw === w && rh === h) return normalizeField(src);
  const dst = new Float32Array(w * h);
  upsampleAdd2(src, rw, rh, dst, w, h, 1);
  return normalizeField(dst);
}

/**
 * Both shapes from one pass. Bark plates and soil clods each need the cell
 * interior *and* the cell wall from the same cell layout; computing them as two
 * calls doubles the cost and is the kind of duplication that quietly eats a
 * second of load time.
 */
function worley2Pair(w, h, fx, fy, seed, jitter = 1, spc = 10, warp = 0.5) {
  const rw = octaveRes(w, fx, spc);
  const rh = octaveRes(h, fy, spc);
  const { f1, f2 } = worley2Fields(rw, rh, fx, fy, seed, jitter, true, warp);
  const a = new Float32Array(rw * rh);
  const b = new Float32Array(rw * rh);
  for (let i = 0; i < a.length; i++) { a[i] = -f1[i]; b[i] = f1[i] - f2[i]; }
  if (rw === w && rh === h) return { cells: normalizeField(a), walls: normalizeField(b) };
  const ca = new Float32Array(w * h);
  const cb = new Float32Array(w * h);
  upsampleAdd2(a, rw, rh, ca, w, h, 1);
  upsampleAdd2(b, rw, rh, cb, w, h, 1);
  return { cells: normalizeField(ca), walls: normalizeField(cb) };
}

/** Per-axis lower bounds for the 3D Worley branch-and-bound. Module scope so
 *  the ~900k-voxel inner loop allocates nothing. */
const BND_X = new Float64Array(3);
const BND_Y = new Float64Array(3);
const BND_Z = new Float64Array(3);

/** Tiling cubic 3D Worley F1, inverted (1 at feature points). */
function worley3(size, freq, seed, jitter = 1, warp = 0) {
  const f = freq;
  const nc = f * f * f;
  const pts = new Float32Array(nc * 3);
  for (let k = 0; k < f; k++) {
    for (let j = 0; j < f; j++) {
      for (let i = 0; i < f; i++) {
        const o = (k * f * f + j * f + i) * 3;
        const c = 0.5 - 0.5 * jitter;
        // h3, not hash3 - see HASH_SCALE. Unscaled, every cloud feature point
        // sat in the lower corner octant of its cell.
        pts[o] = i + c + jitter * h3(i, j, k + seed);
        pts[o + 1] = j + c + jitter * h3(i + 131, j, k + seed);
        pts[o + 2] = k + c + jitter * h3(i, j + 197, k + seed);
      }
    }
  }
  const dst = new Float32Array(size * size * size);
  const sc = f / size;
  const ff = f * f;
  // Same lattice-breaking warp as the 2D generator. Only worth paying for on
  // the reduced-resolution octaves, which is where it is enabled from.
  const wf = warp > 0 ? Math.max(2, Math.min(256, Math.round(f * 0.75))) : 0;
  let idx = 0;
  for (let z = 0; z < size; z++) {
    const bz = (z + 0.5) * sc;
    for (let y = 0; y < size; y++) {
      const by = (y + 0.5) * sc;
      for (let x = 0; x < size; x++) {
        const bx = (x + 0.5) * sc;
        let wx = bx, wy = by, wz = bz;
        if (wf) {
          const u = (x + 0.5) / size, v = (y + 0.5) / size, t = (z + 0.5) / size;
          wx += warp * perlin3P(u * wf, v * wf, t * wf, wf, wf, wf, seed + 401);
          wy += warp * perlin3P(u * wf, v * wf, t * wf, wf, wf, wf, seed + 823);
          wz += warp * perlin3P(u * wf, v * wf, t * wf, wf, wf, wf, seed + 1259);
          // Fold back into one period. The warp can push the sample outside
          // [0,f), and the ±1 neighbour wrap below only survives a single step
          // out of range; the field is periodic, so this is exact, not a clamp.
          wx -= Math.floor(wx / f) * f;
          wy -= Math.floor(wy / f) * f;
          wz -= Math.floor(wz / f) * f;
          // Floating-point belt and braces: a tiny negative input can come back
          // out of the fold as exactly `f`, and the branch-and-bound below
          // indexes the sample's own cell directly, so it must be in [0, f).
          if (wx >= f) wx = 0; else if (wx < 0) wx += f;
          if (wy >= f) wy = 0; else if (wy < 0) wy += f;
          if (wz >= f) wz = 0; else if (wz < 0) wz += f;
        }
        const cx = Math.floor(wx), cy = Math.floor(wy), cz = Math.floor(wz);
        // Branch-and-bound over the 27 neighbours. The nearest point a cell can
        // possibly hold is on the face of that cell, so the squared distance to
        // the cell's slab is a valid lower bound; once the sample's own cell has
        // supplied a candidate, roughly half of the 26 neighbours are provably
        // further away and never need their point fetched. This is the hottest
        // loop in the whole file - the cloud volumes are ~900k voxels - and the
        // pruning is exact, so the field is bit-identical to the naive version.
        const gx = wx - cx, gy = wy - cy, gz = wz - cz;
        BND_X[0] = gx * gx; BND_X[1] = 0; BND_X[2] = (1 - gx) * (1 - gx);
        BND_Y[0] = gy * gy; BND_Y[1] = 0; BND_Y[2] = (1 - gy) * (1 - gy);
        BND_Z[0] = gz * gz; BND_Z[1] = 0; BND_Z[2] = (1 - gz) * (1 - gz);
        let best;
        {
          const o = ((cz * f + cy) * f + cx) * 3;
          const dx = pts[o] - wx, dy = pts[o + 1] - wy, dz = pts[o + 2] - wz;
          best = dx * dx + dy * dy + dz * dz;
        }
        for (let dk = -1; dk <= 1; dk++) {
          const lz = BND_Z[dk + 1];
          if (lz >= best) continue;
          let kk = cz + dk, oz = 0;
          if (kk < 0) { kk += f; oz = -f; } else if (kk >= f) { kk -= f; oz = f; }
          const bk = kk * ff;
          for (let dj = -1; dj <= 1; dj++) {
            const lzy = lz + BND_Y[dj + 1];
            if (lzy >= best) continue;
            let jj = cy + dj, oy = 0;
            if (jj < 0) { jj += f; oy = -f; } else if (jj >= f) { jj -= f; oy = f; }
            const bj = bk + jj * f;
            for (let di = -1; di <= 1; di++) {
              if (!di && !dj && !dk) continue;         // already seeded `best`
              if (lzy + BND_X[di + 1] >= best) continue;
              let ii = cx + di, ox = 0;
              if (ii < 0) { ii += f; ox = -f; } else if (ii >= f) { ii -= f; ox = f; }
              const o = (bj + ii) * 3;
              const dx = pts[o] + ox - wx;
              const dy = pts[o + 1] + oy - wy;
              const dz = pts[o + 2] + oz - wz;
              const d = dx * dx + dy * dy + dz * dz;
              if (d < best) best = d;
            }
          }
        }
        // Raw inverted distance, unclamped. Clamping here would flatten every
        // cell corner to zero before the caller ever sees the distribution.
        dst[idx++] = -Math.sqrt(best);
      }
    }
  }
  return dst;
}

/**
 * Worley at a reduced resolution when its frequency does not justify full res,
 * then wrap-upsampled. `spc` of 6 keeps the cell cusps crisp; going below ~4
 * visibly rounds them off.
 */
function worley3Scaled(size, freq, seed, spc = 5) {
  const r = octaveRes(size, freq, spc);
  // The lattice-breaking warp costs three 3D noise evaluations per voxel, which
  // is worth paying at reduced resolution but not at full: the octaves that run
  // full-res here are the finest ones, where the grid is least readable anyway,
  // and they carry the smallest fbm weight.
  if (r === size) return normalizeField(worley3(size, freq, seed, 1, 0));
  const low = worley3(r, freq, seed, 1, 0.55);
  const dst = new Float32Array(size * size * size);
  upsampleAdd3(low, r, dst, size, 1);
  return normalizeField(dst);
}

// ===========================================================================
// SECTION 2 - field utilities
// ===========================================================================

/**
 * Remap a field to 0..1 using its own 1st/99th percentile (256-bin histogram).
 * Using min/max instead would let a single outlier voxel crush the whole range.
 */
function normalizeField(field, lo = 0.01, hi = 0.99) {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < field.length; i++) {
    const v = field[i];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  if (!(mx > mn)) {
    field.fill(0.5);
    return field;
  }
  const BINS = 256;
  const hist = new Int32Array(BINS);
  const inv = (BINS - 1) / (mx - mn);
  for (let i = 0; i < field.length; i++) hist[((field[i] - mn) * inv) | 0]++;
  const loCount = field.length * lo, hiCount = field.length * hi;
  let acc = 0, b0 = 0, b1 = BINS - 1;
  for (let b = 0; b < BINS; b++) {
    acc += hist[b];
    if (acc >= loCount) { b0 = b; break; }
  }
  acc = 0;
  for (let b = 0; b < BINS; b++) {
    acc += hist[b];
    if (acc >= hiCount) { b1 = b; break; }
  }
  if (b1 <= b0) { b0 = 0; b1 = BINS - 1; }
  const a = mn + (b0 / (BINS - 1)) * (mx - mn);
  const c = mn + (b1 / (BINS - 1)) * (mx - mn);
  const k = 1 / (c - a);
  for (let i = 0; i < field.length; i++) field[i] = clamp01((field[i] - a) * k);
  return field;
}

/**
 * Separable wrapped box blur, two passes for a near-Gaussian kernel.
 *
 * The wrap indices are tabulated rather than computed with a double modulo per
 * sample. Two `%` and two adds per texel per sweep sounds harmless until you
 * notice this runs eight sweeps over a megapixel per surface set; tabulating
 * them measured a third off the whole ground bake.
 */
function boxBlurWrap(src, w, h, radius, scratch) {
  // A radius at or beyond the half-period would wrap the running sum onto
  // itself and count texels twice.
  const r = Math.min(radius | 0, ((Math.min(w, h) - 1) / 2) | 0);
  // A copy, never `src` itself. Every caller treats the result as a separate
  // buffer - `highpassField` does `field[i] += k * (0.5 - lo[i])` - and
  // returning the input would turn that into a field subtracting itself in
  // place, which is a silently blank map rather than an error.
  if (r < 1) return src.slice();
  const tmp = scratch || new Float32Array(w * h);
  const out = new Float32Array(w * h);
  const inv = 1 / (r * 2 + 1);
  const subX = new Int32Array(w), addX = new Int32Array(w);
  for (let x = 0; x < w; x++) {
    subX[x] = ((x - r) % w + w) % w;
    addX[x] = (x + r + 1) % w;
  }
  const subY = new Int32Array(h), addY = new Int32Array(h);
  for (let y = 0; y < h; y++) {
    subY[y] = (((y - r) % h + h) % h) * w;
    addY[y] = ((y + r + 1) % h) * w;
  }
  for (let pass = 0; pass < 2; pass++) {
    const s = pass === 0 ? src : out;
    for (let y = 0; y < h; y++) {
      const row = y * w;
      let sum = 0;
      for (let k = -r; k <= r; k++) sum += s[row + (((k % w) + w) % w)];
      for (let x = 0; x < w; x++) {
        tmp[row + x] = sum * inv;
        sum += s[row + addX[x]] - s[row + subX[x]];
      }
    }
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let k = -r; k <= r; k++) sum += tmp[(((k % h) + h) % h) * w + x];
      for (let y = 0; y < h; y++) {
        out[y * w + x] = sum * inv;
        sum += tmp[addY[y] + x] - tmp[subY[y] + x];
      }
    }
  }
  return out;
}

/**
 * Box blur for cases where the result is only ever used as a *low-pass
 * reference* - cavity extraction, high-pass, luminance flattening.
 *
 * Blurring at full resolution to produce something whose entire content sits
 * below 1/(2r) cycles is a straight waste of a megapixel of work. Averaging
 * down by k, blurring there, and wrap-upsampling is visually indistinguishable
 * once k is under about a third of the radius, and it is k² cheaper.
 */
function boxBlurWrapFast(src, w, h, radius) {
  const k = Math.max(1, Math.min(8, Math.floor(radius / 3)));
  if (k < 2 || (w % k) !== 0 || (h % k) !== 0) return boxBlurWrap(src, w, h, radius);
  const rw = w / k, rh = h / k;
  const low = new Float32Array(rw * rh);
  const invK2 = 1 / (k * k);
  for (let y = 0; y < h; y++) {
    const dy = ((y / k) | 0) * rw;
    const row = y * w;
    for (let x = 0; x < w; x++) low[dy + ((x / k) | 0)] += src[row + x];
  }
  for (let i = 0; i < low.length; i++) low[i] *= invK2;
  const blurred = boxBlurWrap(low, rw, rh, Math.max(1, Math.round(radius / k)));
  const out = new Float32Array(w * h);
  upsampleAdd2(blurred, rw, rh, out, w, h, 1);
  return out;
}

/**
 * Remove everything below ~1/(2·radius) cycles from a 0..1 field, re-centred on
 * 0.5. This is the single most effective anti-repetition tool a *tiled* map has.
 *
 * A tile only reads as a tile because the eye finds a landmark and then sees it
 * again one tile away. Landmarks are low-frequency: a big dark patch, a pale
 * blotch, a long crack. Detail maps therefore have no business carrying low
 * frequencies at all - that job belongs to the macro map, which tiles dozens of
 * times more slowly and has no landmark of its own at the detail map's scale.
 * Stripping the bottom of the spectrum out of the detail map costs nothing and
 * removes the thing the eye was locking onto.
 */
function highpassField(field, w, h, radius, keep = 0) {
  const k = 1 - clamp01(keep);          // `keep` lets a caller leave some drift in
  const lo = boxBlurWrapFast(field, w, h, radius);
  for (let i = 0; i < field.length; i++) field[i] += k * (0.5 - lo[i]);
  return field;
}

/** 8×8 Bayer matrix, used to dither float→byte and kill ramp banding. */
const BAYER8 = new Uint8Array([
  0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26,
  12, 44, 4, 36, 14, 46, 6, 38, 60, 28, 52, 20, 62, 30, 54, 22,
  3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25,
  15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21,
]);
/** Dither offset in 0..1 byte units for pixel (x,y). */
const dither = (x, y) => (BAYER8[(y & 7) * 8 + (x & 7)] + 0.5) * (1 / 64) - 0.5;

/** Scratch for the star-map diffraction spikes; avoids a closure per sample. */
const SPIKE_X = new Int32Array(4);
const SPIKE_Y = new Int32Array(4);

/** Quantise a 0..1 float to a byte with ordered dithering. */
function q8(v, d) {
  const b = v * 255 + d;
  return b < 0 ? 0 : b > 255 ? 255 : b | 0;
}

/** sRGB byte triplet from a 0xRRGGBB literal, so palettes read like hex. */
function rgb(hex) {
  return [(hex >> 16) & 255, (hex >> 8) & 255, hex & 255];
}

/**
 * Linear→sRGB lookup, indexed by sqrt(v) so the table is dense exactly where
 * the transfer curve is steep. `Math.pow(v, 1/2.4)` measured at 27% of the star
 * map's entire bake time; a sqrt plus a table read is roughly free, and the
 * worst-case error is a ninth of an 8-bit level.
 */
const SRGB_LUT_N = 2048;
const SRGB_LUT = new Float32Array(SRGB_LUT_N + 1);
for (let i = 0; i <= SRGB_LUT_N; i++) {
  const v = (i / SRGB_LUT_N) * (i / SRGB_LUT_N);
  SRGB_LUT[i] = v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
}
const linToSrgb = (v) => (v <= 0 ? 0 : v >= 1 ? 1 : SRGB_LUT[(Math.sqrt(v) * SRGB_LUT_N) | 0]);

/** sRGB byte → linear. 256 entries, so the decode is a single array read. */
const S2L = new Float32Array(256);
for (let i = 0; i < 256; i++) {
  const c = i / 255;
  S2L[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

// ===========================================================================
// SECTION 2b - hand-built mip chains
// ===========================================================================
//
// Why not just let the driver call glGenerateMipmap?
//
//  1. Colour. An sRGB-encoded albedo must be filtered in *linear* light. WebGL2
//     leaves the behaviour of generateMipmap on an SRGB8_ALPHA8 texture up to
//     the implementation, and several drivers box-filter the encoded bytes,
//     which darkens every level of a high-contrast map. Distant tiled ground
//     going muddy is almost always this.
//  2. Normals. Averaging unit normals and renormalising throws away exactly the
//     information that matters: how much the surface wobbled inside the
//     footprint. That lost wobble has to reappear as roughness or the surface
//     specular-aliases - which on this target is very visible, because there is
//     no temporal AA to hide it. The chains below keep the *unnormalised* mean
//     vector, derive a von Mises-Fisher variance from its length, and widen the
//     matching roughness mip by it (Toksvig / Frostbite specular AA).
//  3. Cutouts. Plain alpha averaging shrinks an alpha-tested silhouette a
//     little at every level, so distant blossom and grass quietly evaporate.
//     Rescaling each level's alpha to preserve the level-0 coverage fraction
//     (Castano) fixes it for the price of one histogram per level.

/**
 * Full mip chain for an RGBA8 map, filtered in linear light.
 * @param {Uint8Array} base level-0 bytes (referenced, not copied)
 * @param {object} o `{srgb, alphaWeighted, preserveCoverage, alphaTest}`
 * @returns {Array<{data:Uint8Array,width:number,height:number}>}
 */
function mipChainRGBA(base, w, h, o = {}) {
  const srgb = !!o.srgb;
  const aw = !!o.alphaWeighted;
  const levels = [{ data: base, width: w, height: h }];
  const n0 = w * h;
  let cw = w, ch = h;
  let cur = new Float32Array(n0 * 4);
  for (let i = 0; i < n0; i++) {
    const s = i * 4;
    cur[s] = srgb ? S2L[base[s]] : base[s] * (1 / 255);
    cur[s + 1] = srgb ? S2L[base[s + 1]] : base[s + 1] * (1 / 255);
    cur[s + 2] = srgb ? S2L[base[s + 2]] : base[s + 2] * (1 / 255);
    cur[s + 3] = base[s + 3] * (1 / 255);
  }

  let targetCoverage = -1;
  if (o.preserveCoverage) {
    const at = o.alphaTest ?? 0.5;
    let c = 0;
    for (let i = 0; i < n0; i++) if (cur[i * 4 + 3] >= at) c++;
    targetCoverage = c / n0;
  }

  while (cw > 1 || ch > 1) {
    const nw = Math.max(1, cw >> 1);
    const nh = Math.max(1, ch >> 1);
    const next = new Float32Array(nw * nh * 4);
    for (let y = 0; y < nh; y++) {
      const y0 = Math.min(ch - 1, y * 2) * cw;
      const y1 = Math.min(ch - 1, y * 2 + 1) * cw;
      for (let x = 0; x < nw; x++) {
        const x0 = Math.min(cw - 1, x * 2);
        const x1 = Math.min(cw - 1, x * 2 + 1);
        const a = (y0 + x0) * 4, b = (y0 + x1) * 4, c = (y1 + x0) * 4, d = (y1 + x1) * 4;
        const p = (y * nw + x) * 4;
        const aa = cur[a + 3], ab = cur[b + 3], ac = cur[c + 3], ad = cur[d + 3];
        const sum = aa + ab + ac + ad;
        if (aw && sum > 1e-4) {
          const inv = 1 / sum;
          next[p] = (cur[a] * aa + cur[b] * ab + cur[c] * ac + cur[d] * ad) * inv;
          next[p + 1] = (cur[a + 1] * aa + cur[b + 1] * ab + cur[c + 1] * ac + cur[d + 1] * ad) * inv;
          next[p + 2] = (cur[a + 2] * aa + cur[b + 2] * ab + cur[c + 2] * ac + cur[d + 2] * ad) * inv;
        } else {
          next[p] = (cur[a] + cur[b] + cur[c] + cur[d]) * 0.25;
          next[p + 1] = (cur[a + 1] + cur[b + 1] + cur[c + 1] + cur[d + 1]) * 0.25;
          next[p + 2] = (cur[a + 2] + cur[b + 2] + cur[c + 2] + cur[d + 2]) * 0.25;
        }
        next[p + 3] = sum * 0.25;
      }
    }
    cur = next; cw = nw; ch = nh;

    if (targetCoverage > 0) {
      // Bisect a scale on alpha so this level tests to the same coverage as
      // level 0. Ten iterations lands within 0.1%, and the level is tiny.
      const at = o.alphaTest ?? 0.5;
      const n = cw * ch;
      let lo = 0.25, hi = 8;
      for (let it = 0; it < 10; it++) {
        const mid = (lo + hi) * 0.5;
        let c = 0;
        for (let i = 0; i < n; i++) if (cur[i * 4 + 3] * mid >= at) c++;
        if (c / n < targetCoverage) lo = mid; else hi = mid;
      }
      const s = (lo + hi) * 0.5;
      for (let i = 0; i < n; i++) cur[i * 4 + 3] = clamp01(cur[i * 4 + 3] * s);
    }

    const out = new Uint8Array(cw * ch * 4);
    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) {
        const i = (y * cw + x) * 4;
        const r = cur[i], g = cur[i + 1], b = cur[i + 2];
        const dth = dither(x, y);
        // Most of a star map is exactly black. `q8(0, d)` is 0 for every dither
        // offset, so short-circuiting the transfer curve there is exact, and it
        // is worth more than any micro-optimisation of the encode itself.
        out[i] = r === 0 ? 0 : q8(srgb ? linToSrgb(r) : r, dth);
        out[i + 1] = g === 0 ? 0 : q8(srgb ? linToSrgb(g) : g, dth);
        out[i + 2] = b === 0 ? 0 : q8(srgb ? linToSrgb(b) : b, dth);
        out[i + 3] = q8(cur[i + 3], dth);
      }
    }
    levels.push({ data: out, width: cw, height: ch });
  }
  return levels;
}

/**
 * Normal-map mip chain from a height field, plus the per-level normal variance
 * the roughness chain needs.
 *
 * @returns {{levels:Array, sigma2:Array<Float32Array>, size:Array<[number,number]>}}
 */
function normalMipChain(height, w, h, o = {}) {
  const strength = o.strength ?? 1;
  const k = strength * Math.max(w, h) / 128;
  const flipY = o.flipGreen ? -1 : 1;
  const n0 = w * h;
  // Wrapped differencing is right for a tiling surface and wrong for a
  // ClampToEdge card. On the grass atlas row 0 is the *root* of every blade
  // (height ~0.92) and row H-1 is empty space above the tip (~0.4), so a
  // wrapped Sobel put a row of normals tilted by ~20° straight across the base
  // of all four blades. Clamping instead reproduces what the sampler will
  // actually do outside the card.
  const wrap = o.wrap !== false;
  const wrapX = wrap ? (x) => ((x % w) + w) % w : (x) => (x < 0 ? 0 : x >= w ? w - 1 : x);
  const wrapY = wrap ? (y) => ((y % h) + h) % h : (y) => (y < 0 ? 0 : y >= h ? h - 1 : y);

  // Level 0: unit normals, held as floats so the chain can average vectors.
  let cw = w, ch = h;
  let vec = new Float32Array(n0 * 3);
  let hgt = new Float32Array(n0);
  for (let y = 0; y < h; y++) {
    const y0 = wrapY(y - 1), y1 = wrapY(y + 1);
    const r0 = y0 * w, r1 = y * w, r2 = y1 * w;
    for (let x = 0; x < w; x++) {
      const x0 = wrapX(x - 1), x1 = wrapX(x + 1);
      const h00 = height[r0 + x0], h10 = height[r0 + x], h20 = height[r0 + x1];
      const h01 = height[r1 + x0], h21 = height[r1 + x1];
      const h02 = height[r2 + x0], h12 = height[r2 + x], h22 = height[r2 + x1];
      const gx = (h20 + 2 * h21 + h22 - h00 - 2 * h01 - h02) * 0.25;
      const gy = (h02 + 2 * h12 + h22 - h00 - 2 * h10 - h20) * 0.25;
      let nx = -gx * k, ny = -gy * k * flipY;
      const inv = 1 / Math.sqrt(nx * nx + ny * ny + 1);
      const i3 = (r1 + x) * 3;
      vec[i3] = nx * inv; vec[i3 + 1] = ny * inv; vec[i3 + 2] = inv;
      hgt[r1 + x] = height[r1 + x];
    }
  }

  const levels = [];
  const sigma2 = [];
  const encode = () => {
    const out = new Uint8Array(cw * ch * 4);
    const s2 = new Float32Array(cw * ch);
    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) {
        const i = y * cw + x, i3 = i * 3;
        const mx = vec[i3], my = vec[i3 + 1], mz = vec[i3 + 2];
        const len = Math.sqrt(mx * mx + my * my + mz * mz);
        const invL = len > 1e-6 ? 1 / len : 0;
        const d = dither(x, y);
        const o4 = i * 4;
        out[o4] = q8((mx * invL) * 0.5 + 0.5, d);
        out[o4 + 1] = q8((my * invL) * 0.5 + 0.5, d);
        out[o4 + 2] = q8((mz * invL) * 0.5 + 0.5, d);
        out[o4 + 3] = q8(hgt[i], d);
        // von Mises-Fisher: a shorter mean vector means the normals inside this
        // footprint disagreed more, and that disagreement is variance the BRDF
        // must be told about. Capped, because kappa blows up as |m| -> 0 and a
        // fully mirror-free footprint is already maximally rough.
        const L = len > 0.9995 ? 0.9995 : len;
        const kappa = L * (3 - L * L) / (1 - L * L);
        s2[i] = Math.min(0.25, 1 / (2 * Math.max(kappa, 1e-3)));
      }
    }
    levels.push({ data: out, width: cw, height: ch });
    sigma2.push(s2);
  };

  encode();
  while (cw > 1 || ch > 1) {
    const nw = Math.max(1, cw >> 1);
    const nh = Math.max(1, ch >> 1);
    const nv = new Float32Array(nw * nh * 3);
    const nhg = new Float32Array(nw * nh);
    for (let y = 0; y < nh; y++) {
      const y0 = Math.min(ch - 1, y * 2) * cw;
      const y1 = Math.min(ch - 1, y * 2 + 1) * cw;
      for (let x = 0; x < nw; x++) {
        const x0 = Math.min(cw - 1, x * 2);
        const x1 = Math.min(cw - 1, x * 2 + 1);
        const a = y0 + x0, b = y0 + x1, c = y1 + x0, d = y1 + x1;
        const o3 = (y * nw + x) * 3;
        nv[o3] = (vec[a * 3] + vec[b * 3] + vec[c * 3] + vec[d * 3]) * 0.25;
        nv[o3 + 1] = (vec[a * 3 + 1] + vec[b * 3 + 1] + vec[c * 3 + 1] + vec[d * 3 + 1]) * 0.25;
        nv[o3 + 2] = (vec[a * 3 + 2] + vec[b * 3 + 2] + vec[c * 3 + 2] + vec[d * 3 + 2]) * 0.25;
        nhg[y * nw + x] = (hgt[a] + hgt[b] + hgt[c] + hgt[d]) * 0.25;
      }
    }
    vec = nv; hgt = nhg; cw = nw; ch = nh;
    encode();
  }
  return { levels, sigma2 };
}

/**
 * ORM mip chain whose roughness carries the normal detail each level lost.
 * three's `roughness` is Disney/Burley, so the GGX width is alpha = r²; filtered
 * normal variance adds in alpha² (Frostbite, "Moving Frostbite to PBR", §5.3):
 *   alpha'^2 = alpha^2 + 2*sigma^2   =>   r' = (r^4 + 2*sigma^2)^(1/4)
 */
function ormMipChain(base, w, h, sigma2) {
  const levels = [{ data: base, width: w, height: h }];
  let cw = w, ch = h;
  let cur = new Float32Array(w * h * 4);
  for (let i = 0; i < w * h * 4; i++) cur[i] = base[i] * (1 / 255);
  let lvl = 0;
  while (cw > 1 || ch > 1) {
    const nw = Math.max(1, cw >> 1);
    const nh = Math.max(1, ch >> 1);
    const next = new Float32Array(nw * nh * 4);
    for (let y = 0; y < nh; y++) {
      const y0 = Math.min(ch - 1, y * 2) * cw;
      const y1 = Math.min(ch - 1, y * 2 + 1) * cw;
      for (let x = 0; x < nw; x++) {
        const x0 = Math.min(cw - 1, x * 2);
        const x1 = Math.min(cw - 1, x * 2 + 1);
        const a = (y0 + x0) * 4, b = (y0 + x1) * 4, c = (y1 + x0) * 4, d = (y1 + x1) * 4;
        const p = (y * nw + x) * 4;
        for (let q = 0; q < 4; q++) next[p + q] = (cur[a + q] + cur[b + q] + cur[c + q] + cur[d + q]) * 0.25;
      }
    }
    cur = next; cw = nw; ch = nh; lvl++;
    const s2 = sigma2 && sigma2[lvl];
    const out = new Uint8Array(cw * ch * 4);
    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) {
        const i = y * cw + x;
        const o4 = i * 4;
        let r = cur[o4 + 1];
        if (s2 && i < s2.length) {
          const a2 = r * r * r * r + 2 * s2[i];
          r = clamp01(Math.sqrt(Math.sqrt(a2)));
        }
        const dth = dither(x, y);
        out[o4] = q8(cur[o4], dth);
        out[o4 + 1] = q8(r, dth);
        out[o4 + 2] = q8(cur[o4 + 2], dth);
        out[o4 + 3] = q8(cur[o4 + 3], dth);
      }
    }
    levels.push({ data: out, width: cw, height: ch });
  }
  return levels;
}

// ===========================================================================
// SECTION 3 - void-and-cluster blue noise
// ===========================================================================

/**
 * Void-and-cluster (Ulichney 1993). Produces a rank field whose every
 * threshold slice is itself a blue-noise pattern - which is exactly what a
 * raymarch jitter needs, because the marcher thresholds it implicitly.
 *
 * The energy field is maintained incrementally: adding or removing a point
 * splats a truncated Gaussian instead of rebuilding the field, which turns an
 * O(N²·k) algorithm into roughly O(N²) with a tiny constant. 64² lands in
 * ~50 ms; 128² would be ~16× that, which is why 64² is the shipped size (it is
 * also what most engines tile, and it fits comfortably in texture cache).
 */
function voidAndCluster(size, seed) {
  const N = size * size;
  const binary = new Uint8Array(N);
  const energy = new Float32Array(N);
  const rank = new Int32Array(N).fill(-1);

  // Truncated Gaussian, sigma 1.5 - Ulichney's value. Radius 4 sigma.
  const SIG = 1.5;
  const R = Math.min(Math.floor(size / 2) - 1, 6);
  const K = 2 * R + 1;
  const kern = new Float32Array(K * K);
  for (let j = -R; j <= R; j++) {
    for (let i = -R; i <= R; i++) {
      kern[(j + R) * K + (i + R)] = Math.exp(-(i * i + j * j) / (2 * SIG * SIG));
    }
  }

  // --- row-blocked extreme cache -------------------------------------------
  // The textbook algorithm scans all N texels for its argmax and again for its
  // argmin, N times over, which is O(N²) with a constant of two - about 50 M
  // comparisons at 64² and the reason a second independent pattern was too
  // expensive to want. A splat only perturbs a (2R+1)² neighbourhood, so per
  // row we cache the extremes and recompute only the rows the splat dirtied:
  // ~900 operations an iteration instead of ~8200, for byte-identical output
  // (ties still break toward the lowest index, in the same order).
  const dirty = new Uint8Array(size).fill(1);
  const maxOneV = new Float32Array(size), maxOneI = new Int32Array(size);
  const minZeroV = new Float32Array(size), minZeroI = new Int32Array(size);
  const maxZeroV = new Float32Array(size), maxZeroI = new Int32Array(size);

  const refresh = (b) => {
    let a1 = -Infinity, i1 = -1, a2 = Infinity, i2 = -1, a3 = -Infinity, i3 = -1;
    const base = b * size;
    for (let x = 0; x < size; x++) {
      const i = base + x, e = energy[i];
      if (binary[i] === 1) {
        if (e > a1) { a1 = e; i1 = i; }
      } else {
        if (e < a2) { a2 = e; i2 = i; }
        if (e > a3) { a3 = e; i3 = i; }
      }
    }
    maxOneV[b] = a1; maxOneI[b] = i1;
    minZeroV[b] = a2; minZeroI[b] = i2;
    maxZeroV[b] = a3; maxZeroI[b] = i3;
    dirty[b] = 0;
  };

  const splat = (idx, sgn) => {
    const px = idx % size, py = (idx / size) | 0;
    for (let j = -R; j <= R; j++) {
      const yy = ((py + j) % size + size) % size;
      const row = yy * size;
      const krow = (j + R) * K + R;
      for (let i = -R; i <= R; i++) {
        const xx = ((px + i) % size + size) % size;
        energy[row + xx] += sgn * kern[krow + i];
      }
      dirty[yy] = 1;
    }
  };
  /** Flip binary[i] and keep the cache honest. */
  const setBit = (i, v) => { binary[i] = v; dirty[(i / size) | 0] = 1; };
  const markAllDirty = () => dirty.fill(1);

  /** kind 0: max energy among ones · 1: min among zeros · 2: max among zeros */
  const pick = (kind) => {
    let best = kind === 1 ? Infinity : -Infinity, bi = -1;
    for (let b = 0; b < size; b++) {
      if (dirty[b]) refresh(b);
      const i = kind === 0 ? maxOneI[b] : kind === 1 ? minZeroI[b] : maxZeroI[b];
      if (i < 0) continue;
      const v = kind === 0 ? maxOneV[b] : kind === 1 ? minZeroV[b] : maxZeroV[b];
      if (kind === 1 ? v < best : v > best) { best = v; bi = i; }
    }
    return bi;
  };
  const findTightestCluster = () => pick(0);
  const findLargestVoid = () => pick(1);

  // --- initial pattern: ~10% ones, then swap until stable -------------------
  const rand = makeRNG(seed);
  let ones = Math.max(1, Math.round(N * 0.1));
  {
    let placed = 0;
    while (placed < ones) {
      const i = (rand() * N) | 0;
      if (binary[i] === 0) { setBit(i, 1); splat(i, 1); placed++; }
    }
    // Relaxation: move the worst-clustered point into the largest void until
    // doing so is a no-op. Bounded so a pathological seed cannot spin forever;
    // in practice it converges in a few hundred swaps.
    for (let iter = 0; iter < N * 4; iter++) {
      const c = findTightestCluster();
      setBit(c, 0); splat(c, -1);
      const v = findLargestVoid();
      if (v === c) { setBit(c, 1); splat(c, 1); break; }
      setBit(v, 1); splat(v, 1);
    }
  }

  const initial = binary.slice();

  // --- phase 1: rank the initial ones downward from ones-1 -----------------
  for (let r = ones - 1; r >= 0; r--) {
    const c = findTightestCluster();
    if (c < 0) break;
    setBit(c, 0); splat(c, -1);
    rank[c] = r;
  }

  // --- phase 2: restore, then fill voids upward from ones ------------------
  binary.set(initial);
  energy.fill(0);
  for (let i = 0; i < N; i++) if (binary[i] === 1) splat(i, 1);
  markAllDirty();
  const half = (N + 1) >> 1;
  for (let r = ones; r < half; r++) {
    const v = findLargestVoid();
    if (v < 0) break;
    setBit(v, 1); splat(v, 1);
    rank[v] = r;
  }

  // --- phase 3: invert the sense and keep filling ---------------------------
  // Past the halfway point the interesting structure is the distribution of
  // the remaining *zeros*, so the energy field is rebuilt from them.
  energy.fill(0);
  for (let i = 0; i < N; i++) if (binary[i] === 0) splat(i, 1);
  markAllDirty();
  for (let r = half; r < N; r++) {
    // Tightest cluster of zeros == highest energy among zeros.
    const bi = pick(2);
    if (bi < 0) break;
    setBit(bi, 1); splat(bi, -1);
    rank[bi] = r;
  }

  const out = new Float32Array(N);
  const inv = 1 / N;
  for (let i = 0; i < N; i++) out[i] = (rank[i] < 0 ? 0 : rank[i] + 0.5) * inv;
  return out;
}

// ===========================================================================
// SECTION 4 - sprite rasterisation helpers
// ===========================================================================

/**
 * How much world each tiling surface map covers, in metres.
 *
 * Published on every texture as `userData.tileMeters` so a consumer can set its
 * repeat from the world scale instead of guessing. These are not decoration:
 * every feature frequency in the ground, moss and litter bakers is derived from
 * a physical size divided by the tile, which is the only way to keep "12 mm
 * grain" meaning 12 mm when the texture resolution changes with quality tier.
 */
const GROUND_TILE_M = 2.0;
const MOSS_TILE_M = 1.2;
const LITTER_TILE_M = 0.6;
/** Bark: metres around the trunk (U) by metres along it (V). */
const BARK_TILE_M = { u: 1.1, v: 1.8 };

/**
 * World tile of the terrain near-detail suite, metres.
 *
 * This MUST equal `TILE_NEAR` in world/terrain.js - that file sets the repeat,
 * this one sizes every feature against it. If terrain.js ever changes TILE_NEAR
 * the maps below stay seamless (they are periodic regardless) but every physical
 * size in `_buildTerrainSurface` scales with it, which is exactly how a 12 mm
 * grain silently becomes a 24 mm one. Published on each texture as
 * `userData.tileMeters` so the mismatch is at least inspectable at runtime.
 */
const TERRAIN_TILE_M = 2.0;

/**
 * Reference resolution the sprite DENSITIES in the ground / moss / litter
 * bakers are anchored to, and the helper that converts one.
 *
 * `_buildTerrainSurface` states the defect and fixes it for its own suite with
 * `perSqM`; the three older builders below still had it. A sprite count written
 * as `N / k` is proportional to S², and every sprite's *area* is proportional to
 * S² as well because its size is authored in metres over a fixed world tile - so
 * coverage grows with the square of the resolution and a tier change hands you a
 * different surface rather than a sharper one. Measured on `mossSet`, which is
 * live (tree/sakura.js consumes it for the trunk base): LOW and MEDIUM bake at
 * 256 and got 2521 shoots where HIGH and ULTRA bake at 512 and got 10082, i.e. a
 * quarter of the pile on the two tiers this target actually runs. `_buildGround`
 * had it twice and ULTRA's 1024 ran four times HIGH's thatch.
 *
 * A count over a fixed patch of world is a physical quantity and must not move
 * with the texel grid at all. REF is 512 because that is what the three
 * palettes were authored against, so the HIGH bake is byte-identical and only
 * the tiers that were wrong change.
 */
const SPRITE_REF = 512;
const spriteCount = (divisor) => Math.max(1, Math.round((SPRITE_REF * SPRITE_REF) / divisor));

/**
 * RMS surface slope the terrain micro-relief normal map is calibrated to, in
 * degrees, before world/terrain.js applies its own `uNormalScale` (0.62).
 *
 * This is the single number that decides whether soil reads as soil or as
 * bubble wrap, and it is far smaller than intuition suggests. Bare earth under
 * a metre of meadow is nearly flat: the relief is a few millimetres of crumb
 * and grain over centimetres of run. The map this replaces produced an RMS
 * slope near fifty - round, evenly spaced, strongly shaded domes, visible as a
 * regular carpet through the grass in every capture of the build. Amplitude is
 * not a taste setting here; it is measured (see `gradientRMS`) and solved for,
 * so it stays correct when the height field, the resolution or the tier changes.
 *
 * 11, down from 14. The value that matters is the product with terrain.js's
 * `uNormalScale`, and the two moved together in the same pass: 14 x 0.85 = 11.9
 * effective degrees became 11 x 0.62 = 6.8. The relief that survives is fine
 * grain and the thin voids between crumbs - which is all a millimetre of soil
 * texture at 8 mm a texel can honestly cast - while the centimetre-scale
 * shading that made the ground read as a moulded surface rather than as dirt is
 * gone. Below about 5 degrees the surface goes plastic-smooth at grazing
 * incidence, so this is near the floor, not merely lower.
 */
const TERRAIN_SLOPE_RMS_DEG = 11;

/**
 * The domain-warp budget world/terrain.js published, restated here because this
 * file now owns the map that drives it.
 *
 * terrain.js warps its near detail lookup by `(macro.ba - 0.5) * DETAIL_WARP`
 * metres, sampled at `TILE_MESO`, and it picked DETAIL_WARP = 3.4 from a
 * measurement of the Jacobian determinant of d(warped)/d(world):
 *
 *   2.6 m  det 0.70 .. 1.31        3.4 m  det 0.62 .. 1.40
 *   6.5 m  det 0.30 .. 1.81  (visibly blurs)      10.0 m  det -0.02 .. 2.29 (FOLDS)
 *
 * That bound is NOT a property of 3.4 alone - it is 3.4 times the *gradient* of
 * this map's B and A channels, and the gradient is standard deviation times
 * frequency. Which means a macro map can break another module's calibrated
 * number without touching it, just by putting the same contrast at a higher
 * frequency. Measured on the fallback bake this replaces, the whole warp field
 * has an RMS |d/d(world metre)| of 0.126 and reproduces terrain.js's published
 * 0.62..1.40 exactly; the first version of `_buildTerrainMacro` came out at
 * 0.324 (det 0.09..2.05, i.e. between terrain.js's "visibly blurs" and "folds"
 * rows) and at ULTRA's 256² at 0.422 with a determinant that went *negative*
 * - the mapping folded, which is a hard artefact, not a soft one.
 *
 * So the number below is held as a measured constraint, not assumed: see the
 * solve at the end of `_buildTerrainMacro`.
 */
const TERRAIN_MESO_M = 128;      // world/terrain.js TILE_MESO
const TERRAIN_WARP_M = 3.4;      // world/terrain.js DETAIL_WARP
const TERRAIN_WARP_GRAD_RMS = 0.126;

/**
 * Stamp `count` bent, tapered fibres into a wrapping tile, compositing over.
 *
 * Dead-stem thatch is the single most characteristic thing about the ground
 * under a metre of pampas, and no noise field can produce a strand: fbm gives
 * blobs, worley gives cells, and anisotropic fbm gives a corduroy that is
 * axis-aligned and therefore obviously synthetic. Sprites are the honest
 * answer, and they are cheap here because each is scanned inside its own
 * oriented bounding box rather than against the whole tile.
 *
 * `out` is `{a, h}` plus an OPTIONAL `{r, g, b}`. The terrain suite stores layer
 * heights and takes its palette from world/terrain.js, so it passes no colour
 * buffers at all: dropping them removes three Float32Array(S*S) of transient
 * allocation (3 MB at 256², 12 MB at 512²) and three lerps from the inner write
 * of every one of ~5000 strands. The RNG draws are unchanged either way - the
 * palette pick and the tone jitter still happen - so `a` and `h` come out
 * byte-identical with or without colour. Colours are sRGB 0..255.
 */
function stampFibres(S, count, rand, cfg, out) {
  const A = out.a, H = out.h;
  const R = out.r || null, G = out.g || null, B = out.b || null;
  const pal = cfg.palette;
  const taper = cfg.taper ?? 0.4;
  // The width profile is pow(1 - u², taper) - one Math.pow per texel of every
  // fibre, which measured as most of this function. `taper` is constant for the
  // whole call, so a 129-entry table plus a lerp is exact to within a thousandth
  // and about fifteen times cheaper.
  const TAPN = 128;
  const tap = new Float32Array(TAPN + 1);
  for (let i = 0; i <= TAPN; i++) tap[i] = Math.pow(i / TAPN, taper);
  // Optional density mask. Dead stems are not scattered evenly: they lie in
  // mats where the grass fell and leave patches of open soil between, and
  // uniform-random placement is instantly readable as procedural because real
  // debris has that clumping. `cfg.mask` is a 0..1 field, `maskBias` the floor.
  const mask = cfg.mask || null;
  const maskBias = cfg.maskBias ?? 0;
  const maskGain = cfg.maskGain ?? 1;

  for (let n = 0; n < count; n++) {
    const cx = rand() * S, cy = rand() * S;
    const ang = rand() * TAU;
    if (mask) {
      const mi = (Math.floor(cy) % S) * S + (Math.floor(cx) % S);
      // SEVEN draws, matching the seven the accepted branch makes below:
      // length is `rand() * rand()` and costs two, then width, bend, palette,
      // tone and height. It was six, which is the same off-by-one this comment
      // was written to prevent - tuning `maskBias` resequenced the RNG from the
      // first rejected fibre onward and reshuffled the whole tile rather than
      // just thinning it. (The degenerate-size bail below is already correct at
      // four: `lt` and `hw` have been drawn by the time it fires.)
      if (rand() > maskBias + mask[mi] * maskGain) {
        rand(); rand(); rand(); rand(); rand(); rand(); rand();
        continue;
      }
    }
    const ca = Math.cos(ang), sa = Math.sin(ang);
    // Squared draw: short fragments outnumber long ones, as broken stems do.
    const lt = rand() * rand();
    const half = 0.5 * lerp(cfg.lenMin, cfg.lenMax, lt);
    const hw = lerp(cfg.widMin, cfg.widMax, rand());
    // `half` and `hw` are divisors below. A tier whose resolution rounds a
    // configured length to zero would otherwise produce Infinity -> NaN, and a
    // NaN fails every `<= 0` early-out in the loop, so one degenerate fibre
    // silently poisons the whole tile.
    if (!(half > 0) || !(hw > 0)) { rand(); rand(); rand(); rand(); continue; }
    const bendA = (rand() - 0.5) * (cfg.bend ?? 0.25) * half;
    const c = pal[(rand() * pal.length) | 0];
    const tone = 1 + (rand() - 0.5) * 2 * (cfg.shade ?? 0.2);
    const hBase = lerp(cfg.hMin ?? 0.5, cfg.hMax ?? 1, rand());

    const reach = hw + Math.abs(bendA) + 1.5;
    const ex = Math.ceil(Math.abs(half * ca) + Math.abs(reach * sa)) + 1;
    const ey = Math.ceil(Math.abs(half * sa) + Math.abs(reach * ca)) + 1;
    const px = Math.round(cx), py = Math.round(cy);
    for (let dy = -ey; dy <= ey; dy++) {
      const yy = ((py + dy) % S + S) % S;
      const row = yy * S;
      const oy = py + dy - cy;
      for (let dx = -ex; dx <= ex; dx++) {
        const ox = px + dx - cx;
        const lx = ox * ca + oy * sa;
        if (lx < -half - 1 || lx > half + 1) continue;
        const u = lx / half;
        const prof = 1 - u * u;
        if (prof <= 0) continue;
        const tf = prof * TAPN;
        const ti = tf >= TAPN ? TAPN - 1 : tf | 0;   // prof can reach exactly 1
        const wprof = hw * (tap[ti] + (tap[ti + 1] - tap[ti]) * (tf - ti));
        const ly = (-ox * sa + oy * ca) - bendA * prof;
        const ay = ly < 0 ? -ly : ly;
        const cov = clamp01(wprof - ay + 0.5);
        if (cov <= 0.004) continue;
        // Round cross-section: the strand is a cylinder, so its relief peaks on
        // the centre line and falls to nothing at either edge.
        const rel = wprof > 1e-4 ? clamp(ly / wprof, -1, 1) : 0;
        const round = Math.sqrt(Math.max(0, 1 - rel * rel));
        const i = row + ((px + dx) % S + S) % S;
        const a0 = A[i];
        const na = cov + a0 * (1 - cov);
        // Straight-alpha `over`, NOT `lerp(dst, src, cov)`.
        //
        // These buffers start at zero and every consumer composites them with
        // their own alpha again - `lerp(soil, th.r, th.a)` for colour and
        // `th.h * th.a` for height. Weighting the source by `cov` here as well
        // makes the contribution quadratic in coverage, so the one-to-three
        // texel edge that is most of a strand's area came out as a black
        // outline and the relief ramp was cov² instead of linear. Dividing by
        // the accumulated alpha is the exact unpremultiplied `over` and leaves
        // a partly-covered texel carrying the strand's *full* colour, which is
        // what the consumer's own alpha blend expects.
        const wS = na > 1e-6 ? cov / na : 0;
        A[i] = na;
        if (R) {
          // Round cross-section: a highlight down the centre of the strand and
          // a fall-off to either side. This is what makes thatch read as stems
          // and not as painted streaks.
          const sh = tone * (0.80 + 0.30 * round);
          R[i] = lerp(R[i], c[0] * sh, wS);
          G[i] = lerp(G[i], c[1] * sh, wS);
          B[i] = lerp(B[i], c[2] * sh, wS);
        }
        H[i] = lerp(H[i], hBase * round, wS);
      }
    }
  }
}

/**
 * Stamp part-buried grit on a jittered grid.
 *
 * `bury` is the fraction of the stone's radius that sits below the surface.
 * Leaving it out - which is what a plain `sqrt(1 - e)` dome does - is what turns
 * scattered grit into a floor of marbles: a real stone in soil shows a shallow
 * cap, not a hemisphere. `out` is `{h, m, t}`: protrusion, coverage mask, and a
 * per-stone tone in 0..1.
 */
function stampPebbles(S, gridN, occupancy, rand, cfg, out) {
  const cell = S / gridN;
  const bury = clamp(cfg.bury ?? 0.6, 0, 0.95);
  const invCap = 1 / (1 - bury);
  for (let j = 0; j < gridN; j++) {
    for (let i = 0; i < gridN; i++) {
      const take = rand() < occupancy;
      const jx = rand(), jy = rand(), sr = rand(), ar = rand(), an = rand(), tn = rand();
      if (!take) continue;                       // draws are consumed either way,
      const cx = (i + 0.12 + 0.76 * jx) * cell;  // so occupancy can be tuned
      const cy = (j + 0.12 + 0.76 * jy) * cell;  // without reshuffling the world
      const ra = lerp(cfg.rMin, cfg.rMax, sr * sr);
      const rb = ra * (0.62 + 0.38 * ar);
      // Both are divisors. At the LOW tier a configured 3 mm stone is already
      // under half a texel, and a config that rounded one to zero would make
      // `e` NaN - which passes the `e >= 1` reject and writes NaN into the
      // height field, where a single poisoned texel spreads through the blur
      // and the normal chain into the whole map.
      if (!(ra > 0) || !(rb > 0)) continue;
      const ang = an * PI;
      const ca = Math.cos(ang), sa = Math.sin(ang);
      const bx = Math.ceil(Math.max(ra, rb)) + 2;
      const px = Math.round(cx), py = Math.round(cy);
      const rimK = ra * 0.8;
      const hScale = clamp01(ra / cfg.rMax);
      for (let dy = -bx; dy <= bx; dy++) {
        const yy = ((py + dy) % S + S) % S;
        const row = yy * S;
        for (let dx = -bx; dx <= bx; dx++) {
          const lx = (dx * ca + dy * sa) / ra;
          const ly = (-dx * sa + dy * ca) / rb;
          const e = lx * lx + ly * ly;
          if (e >= 1) continue;
          const v = Math.sqrt(1 - e) - bury;
          if (v <= 0) continue;
          const k = row + ((px + dx) % S + S) % S;
          const m = clamp01(v * rimK);
          if (m > out.m[k]) {
            out.m[k] = m;
            out.t[k] = tn;
          }
          const hv = v * invCap * hScale;
          if (hv > out.h[k]) out.h[k] = hv;
        }
      }
    }
  }
}

/**
 * Stamp fallen-petal flakes into a wrapping tile, compositing by maximum.
 *
 * The litter channel of a ground splat map is a *coverage* field, and coverage
 * fields are where procedural ground most often gives itself away: two
 * inverted-Worley lattices thresholded into discs, which is a grid of identical
 * round dots however hard the points are jittered. A petal is not a disc - it is
 * a spatulate blade with a notched tip lying at a random angle and a random
 * foreshortening - and `petalDist` already knows that shape, because the blossom
 * card, the single-petal card and the ground litter all share it. Stamping the
 * real silhouette costs a few milliseconds and removes the lattice outright.
 *
 * Maximum rather than `over`: this is a height/coverage mask, so a petal lying
 * across another does not accumulate to something twice as tall.
 *
 * `out` is `{h, a}`, two Float32Array(S*S): flake height and coverage.
 */
function stampFlakes(S, count, rand, cfg, out) {
  const H = out.h, A = out.a;
  const notch = cfg.notch ?? 0.13;
  const mask = cfg.mask || null;
  const maskBias = cfg.maskBias ?? 0;
  const maskGain = cfg.maskGain ?? 1;
  for (let n = 0; n < count; n++) {
    const cx = rand() * S, cy = rand() * S;
    const ang = rand() * TAU;
    if (mask) {
      const mi = (Math.floor(cy) % S) * S + (Math.floor(cx) % S);
      // Three draws, matching the three the accepted branch makes below (length,
      // squash, tone). Consuming a different count would resequence the RNG and
      // reshuffle every flake in the tile whenever `maskBias` is tuned, instead
      // of merely thinning them.
      if (rand() > maskBias + mask[mi] * maskGain) { rand(); rand(); rand(); continue; }
    }
    const L = lerp(cfg.lenMin, cfg.lenMax, rand());
    // A petal on the ground is seen at a random tilt, so it is foreshortened
    // across its width far more often than along its length. Without this every
    // flake is the same silhouette rotated, which reads as a stencil.
    const squash = 0.45 + 0.55 * rand();
    const tone = lerp(cfg.hMin ?? 0.6, cfg.hMax ?? 1, rand());
    if (!(L > 1)) continue;
    const W = L * 0.60 * squash;
    if (!(W > 0)) continue;
    const ca = Math.cos(ang), sa = Math.sin(ang);
    const halfL = L * 0.5;
    // petalWidth peaks at ~0.80 of wMax, so 0.82*W bounds the silhouette.
    const reach = W * 0.82 + 1.5;
    const ex = Math.ceil(Math.abs(halfL * ca) + Math.abs(reach * sa)) + 1;
    const ey = Math.ceil(Math.abs(halfL * sa) + Math.abs(reach * ca)) + 1;
    const px = Math.round(cx), py = Math.round(cy);
    const invW = 1 / W;
    for (let dy = -ey; dy <= ey; dy++) {
      const yy = ((py + dy) % S + S) % S;
      const row = yy * S;
      const oy = py + dy - cy;
      for (let dx = -ex; dx <= ex; dx++) {
        const ox = px + dx - cx;
        const lx = ox * ca + oy * sa;
        const t = (lx + halfL) / L;
        if (t <= 0 || t >= 1) continue;
        const ly = -ox * sa + oy * ca;
        const d = petalDist(t, ly, L, W, notch, W * 0.33);
        if (d <= -0.5) continue;
        const cov = clamp01(d + 0.5);
        const i = row + ((px + dx) % S + S) % S;
        if (cov <= A[i]) continue;
        A[i] = cov;
        // A fallen petal cups: its edges lift off the soil while its middle lies
        // flat. That is the whole of its relief and it is what stops a drift of
        // litter reading as a flat pink decal.
        const rel = clamp01(Math.abs(ly) * invW);
        H[i] = tone * (0.74 + 0.34 * rel * rel);
      }
    }
  }
}

/**
 * RMS of the wrapped 3×3 Sobel gradient of a height field, in height units per
 * texel. Exactly the operator `normalFromHeight` differentiates with, so
 * `atan(k * rms)` is the RMS surface slope the finished normal map will carry
 * and a target slope can be solved for rather than dialled in by eye.
 */
function gradientRMS(field, w, h) {
  let acc = 0;
  for (let y = 0; y < h; y++) {
    const r0 = (((y - 1) % h) + h) % h * w, r1 = y * w, r2 = ((y + 1) % h) * w;
    for (let x = 0; x < w; x++) {
      const x0 = (((x - 1) % w) + w) % w, x1 = (x + 1) % w;
      const h00 = field[r0 + x0], h10 = field[r0 + x], h20 = field[r0 + x1];
      const h01 = field[r1 + x0], h21 = field[r1 + x1];
      const h02 = field[r2 + x0], h12 = field[r2 + x], h22 = field[r2 + x1];
      const gx = (h20 + 2 * h21 + h22 - h00 - 2 * h01 - h02) * 0.25;
      const gy = (h02 + 2 * h12 + h22 - h00 - 2 * h10 - h20) * 0.25;
      acc += gx * gx + gy * gy;
    }
  }
  return Math.sqrt(acc / (w * h));
}

/** Mean and standard deviation of a field. Reported, not guessed. */
function fieldStats(field) {
  const n = field.length;
  let m = 0;
  for (let i = 0; i < n; i++) m += field[i];
  m /= n;
  let v = 0;
  for (let i = 0; i < n; i++) { const d = field[i] - m; v += d * d; }
  return { mean: m, sd: Math.sqrt(v / n) };
}

/**
 * Affine-map a field to a target mean and standard deviation, then clamp.
 *
 * `normalizeField` fixes a field's *range*, which says nothing about how the
 * mass inside it is distributed - two maps normalised to 0..1 can have standard
 * deviations a factor of three apart. Every channel below feeds a splat blend
 * whose behaviour depends on exactly that spread, so the distributions are set
 * explicitly and measured afterwards rather than inherited from whatever the
 * noise happened to produce.
 */
function retargetField(field, mean, sd, lo = 0.02, hi = 0.98) {
  const s = fieldStats(field);
  const k = s.sd > 1e-6 ? sd / s.sd : 0;
  for (let i = 0; i < field.length; i++) {
    field[i] = clamp(mean + (field[i] - s.mean) * k, lo, hi);
  }
  return field;
}

/**
 * Divide out the low-frequency luminance of an RGB field.
 *
 * The last line of defence against visible tiling. Even after every source
 * field is high-passed, the *composition* of them can leave a broad bright or
 * dark region - and a broad bright region is exactly the landmark that makes a
 * repeat readable. Normalising against a heavily blurred copy of the map's own
 * luminance guarantees there is none, without touching any detail. `keep` lets
 * a caller leave a trace behind so the surface does not go clinically flat.
 */
function flattenLowFrequency(r, g, b, w, h, radius, keep = 0) {
  const n = w * h;
  const lum = new Float32Array(n);
  for (let i = 0; i < n; i++) lum[i] = 0.2126 * r[i] + 0.7152 * g[i] + 0.0722 * b[i];
  const lo = boxBlurWrapFast(lum, w, h, radius);
  let mean = 0;
  for (let i = 0; i < n; i++) mean += lo[i];
  mean /= n;
  const k = 1 - clamp01(keep);
  for (let i = 0; i < n; i++) {
    const t = lo[i] > 1e-4 ? mean / lo[i] : 1;
    const s = clamp(1 + (t - 1) * k, 0.75, 1.35);
    r[i] *= s; g[i] *= s; b[i] *= s;
  }
}

/**
 * Petal outline as an approximate signed distance (positive inside), in units
 * where `L` is the petal length. Shared by the blossom card, the single-petal
 * card and the ground litter so all three read as the same species.
 *
 * The width curve peaks at t≈0.8 - a sakura petal is spatulate: a narrow claw
 * at the base widening to a broad, almost square tip. The `notch` term cuts a
 * shallow V into that tip, which is the feature that makes a cherry petal
 * instantly readable and which almost every procedural petal misses.
 */
function petalDist(t, py, L, wMax, notch, nw) {
  const w = petalWidth(t, wMax);
  const lateral = w - (py < 0 ? -py : py);
  const g = py / nw;
  const end = 1 - notch * Math.exp(-g * g);
  const along = (end - t) * L;
  const base = t * L * 3 + L * 0.02;
  let d = lateral < along ? lateral : along;
  if (base < d) d = base;
  return d;
}

/**
 * Petal width at parameter t, for the outline and for placing veins.
 *
 * Tabulated because `sqrt(t) * pow(1 - t^6, 0.42)` is evaluated twice per texel
 * - once for the silhouette and once for the vein fan - across every petal of
 * every card and every petal of the ground litter. That was three million
 * `Math.pow` calls and the largest single cost in the blossom bake. A 257-entry
 * table with a lerp is accurate to about a ten-thousandth of the width, which is
 * a hundredth of a texel on the silhouette.
 */
const PW_N = 256;
const PW_TABLE = new Float32Array(PW_N + 1);
for (let i = 0; i <= PW_N; i++) {
  const t = i / PW_N;
  const t3 = t * t * t;
  PW_TABLE[i] = Math.sqrt(t) * Math.pow(1 - t3 * t3, 0.42);
}
function petalWidth(t, wMax) {
  const tc = t < 0 ? 0 : t > 1 ? 1 : t;
  const f = tc * PW_N;
  const i = f >= PW_N ? PW_N - 1 : f | 0;
  return wMax * (PW_TABLE[i] + (PW_TABLE[i + 1] - PW_TABLE[i]) * (f - i));
}

// ===========================================================================
// SECTION 5 - the factory
// ===========================================================================

/** Fallback returned instead of throwing when a generator fails mid-load. */
function makeFallback(name) {
  const t = new THREE.DataTexture(new Uint8Array([128, 128, 128, 255]), 1, 1);
  t.needsUpdate = true;
  t.name = `fallback:${name}`;
  return t;
}

/** Normalise a texture key so 'blue-noise', 'blueNoise' and 'BlueNoise' agree. */
const normKey = (s) => String(s).toLowerCase().replace(/[^a-z0-9]/g, '');

/**
 * Names this factory is authoritative for even when the caller supplies its own
 * generator.
 *
 * `get(name, gen)` normally prefers `gen`, which is right: a caller that hands
 * over a generator is describing a texture the factory has never heard of. But
 * world/terrain.js asks for its three ground maps through this factory and says
 * in its own words that they "come from the shared TextureFactory when there is
 * one, and it owns their lifetime - when there is not, or it refuses, we bake
 * locally". Its local bake is the fallback for a factory that cannot supply
 * them, and this file can: the maps are procedural surface textures, which is
 * precisely what it owns per the ownership map.
 *
 * That matters because the local fallback is where the regular carpet of round
 * pale blobs on the ground came from - two inverted-F1 Worley layers carrying
 * 52% of the micro-relief height, at 8 cm and 17 cm over a 2 m tile, with a
 * cavity-AO channel that then lit the top of every dome. Inverted F1 is a radial
 * cone centred on each feature point, so every feature is the same round dome on
 * a grid of the same pitch. Shrinking the tile (which terrain.js already tried,
 * 4 m -> 2 m) only shrinks the domes.
 *
 * Anything not listed here keeps the old precedence exactly. If a listed
 * built-in throws, the caller's generator still runs - the override can degrade,
 * never break.
 */
const FACTORY_OWNED = new Set(['terraindetail', 'terrainnormalao', 'terrainmacro']);

export class TextureFactory {
  /**
   * @param {object} ctx the shared context object is accepted whole
   * @param {THREE.WebGLRenderer} [ctx.renderer] used only for max anisotropy
   * @param {object} [ctx.state] optional, read for the live quality tier
   * @param {object} [ctx.quality] optional, read for the live quality tier
   * @param {EventBus} [ctx.bus] optional; subscribed for QUALITY_CHANGED
   * @param {number} [ctx.seed] world seed; every texture derives from it
   */
  constructor({ renderer, state, quality, bus, seed } = {}) {
    this.renderer = renderer || null;
    this.seed = (seed ?? 20240401) | 0;

    // Held as LIVE references, not read once. This factory is constructed
    // alongside QualityManager and every bake is lazy, so the tier that matters
    // is the one in effect when a texture is first asked for - not the one that
    // happened to be set during construction. Whichever of these is supplied is
    // authoritative; with neither, `this.tier` is whatever it was last set to.
    this._qualitySrc = quality || null;
    this._stateSrc = state || null;
    // Self-subscribe when handed a bus, so a rebake does not depend on the
    // caller remembering to fan QUALITY_CHANGED out to a non-system object.
    this._offQuality = (bus && typeof bus.on === 'function')
      ? bus.on(EVENTS.QUALITY_CHANGED, (q) => this.onQualityChange(q))
      : null;

    // Anisotropy matters a lot here: ground and bark are almost always viewed
    // at grazing angles, and without it they turn to mush two metres out.
    // Capped at 8 - the 780M pays real bandwidth beyond that for no visible win.
    let maxAniso = 4;
    try {
      maxAniso = renderer?.capabilities?.getMaxAnisotropy?.() ?? 4;
    } catch {
      maxAniso = 4;
    }
    this.maxAnisotropy = Math.min(8, Math.max(1, maxAniso));

    this.tier = quality?.tier || state?.quality?.tier || 'high';
    /**
     * The tier the cached textures were actually built against.
     *
     * Distinct from `tier` on purpose: when a live source is supplied, that
     * source is the same object the quality manager writes, so it has already
     * moved by the time QUALITY_CHANGED is delivered. Comparing against `tier`
     * would then always find "no change" and never rebake - which is exactly
     * what happened the first time this factory was given a live source.
     */
    this._lastBakedTier = null;

    /** name -> THREE.Texture */
    this.cache = new Map();
    /** group key -> object of textures (bark set, ground set, ...) */
    this._groups = new Map();
    /** name -> {generator, options} so a quality change can rebake in place. */
    this._recipes = new Map();

    this.stats = { textures: 0, bytes: 0, ms: 0 };
    /** Set true from devtools (`SAKURA.ctx.textures.verbose = true`) to profile. */
    this.verbose = false;

    this._builtins = this._makeBuiltins();
  }

  /**
   * Adopt the live quality/state objects when they were not available at
   * construction. Idempotent, cheap, and safe to call from anywhere.
   *
   * THIS IS NOT OPTIONAL POLISH - without it the entire size policy below is
   * dead code. `src/main.js` builds this factory as
   * `new TextureFactory({ renderer: engine.renderer })`: no `state`, no
   * `quality`, no `bus`. So `this.tier` falls through to its `'high'` default
   * and stays there forever, `_qualitySrc` and `_stateSrc` are null so `_size()`
   * can never resolve a live tier, and because the factory is not a member of
   * `ctx.systems` main.js's QUALITY_CHANGED fan-out never reaches it either.
   * Every map was therefore baked at HIGH sizes on every tier.
   *
   * Measured, on the five suites a real boot actually requests (terrain, bark,
   * moss, noise2d, blossom): 6.78 MB at LOW and 8.79 MB at MEDIUM against
   * 17.84 MB at HIGH. The 780M this project targets detects as LOW or MEDIUM and
   * has no dedicated VRAM, so it was carrying about 11 MB of texture it had
   * asked not to have - which is precisely the failure the size policy below
   * says in its own comments that it exists to prevent.
   *
   * Only fills sources that are still empty, so an explicitly-constructed
   * factory is never overridden, and only moves `tier` while nothing has been
   * baked yet: after a bake, `tier` describes what the cache actually contains
   * and `onQualityChange` compares against `_lastBakedTier` to decide whether a
   * rebake is warranted. Moving it behind their backs would break both.
   *
   * Deliberately does NOT subscribe to the bus. `rebake()` re-runs every stored
   * recipe synchronously - including generators owned by other systems, some of
   * which (the cloud volumes) cost hundreds of milliseconds and are re-created
   * by their owners on the same event - so auto-wiring it would trade a memory
   * bug for a mid-frame stall and a double bake. Callers that genuinely want a
   * resize at runtime still have `onQualityChange()` / `setQuality()`.
   *
   * @param {{state?:object, quality?:object}} sources
   * @returns {string} the tier the next bake will use
   */
  useLiveQuality(sources = {}) {
    if (sources.quality && !this._qualitySrc) this._qualitySrc = sources.quality;
    if (sources.state && !this._stateSrc) this._stateSrc = sources.state;
    const live = this._qualitySrc?.tier || this._stateSrc?.quality?.tier;
    if (live && this._lastBakedTier === null) this.tier = live;
    return this.tier;
  }

  // -------------------------------------------------------------------------
  // Size policy
  // -------------------------------------------------------------------------

  /** Half-size maps below HIGH. Texture memory and cache pressure both halve. */
  get _half() {
    return this.tier === 'low' || this.tier === 'medium';
  }

  /**
   * Size policy. `tier` is for hypothetical questions only ("what would this be
   * at LOW?"); omit it and the live tier is used and recorded as the one the
   * caches are being built against.
   */
  _size(kind, tier) {
    let t = tier;
    if (!t) {
      // Every bake is lazy, so the tier is resolved here rather than in the
      // constructor: on the target iGPU the quality manager settles on LOW or
      // MEDIUM, and a factory that snapshotted 'high' at construction would
      // bake the whole suite at HIGH sizes anyway - three times the texture
      // memory on a part with no dedicated VRAM.
      const live = this._qualitySrc?.tier || this._stateSrc?.quality?.tier;
      if (live && live !== this.tier) this.tier = live;
      t = this.tier;
      this._lastBakedTier = t;
    }
    const ultra = t === 'ultra';
    const low = t === 'low';
    const half = low || t === 'medium';
    switch (kind) {
      // Bark is the hero asset - the player can put their face on the trunk - 
      // but it is also the most expensive thing in the file, because every
      // field, every mip chain and every per-texel pass scales with its area.
      // 768 over the 1.1 m of trunk circumference it tiles across is 1.4 mm a
      // texel, which at a metre's viewing distance on a 1080p display is about
      // one and a half texels per pixel: with anisotropic filtering on, that is
      // indistinguishable from 1024 and costs 56% as much. ULTRA still gets the
      // full 1024 for anyone who wants to press their nose against it.
      case 'bark': return half ? 512 : ultra ? 1024 : 768;
      // LOW gets 256. This used to read `this._half ? 512 : ultra ? 1024 : 512`,
      // which is 512 on three of the four tiers: the ground suite is the single
      // most expensive bake in the file (measured 442 ms at LOW against 475 ms
      // at HIGH) and it was also the one map that did not shrink for the tier
      // that needs it to. 256 over a 2 m tile is 7.8 mm a texel, the same
      // density the moss and litter maps already run at on LOW, and the grain
      // frequencies follow it automatically because `finest()` derives them
      // from S.
      case 'ground': return low ? 256 : ultra ? 1024 : 512;
      case 'moss': return half ? 256 : 512;
      case 'litter': return half ? 256 : 512;
      case 'macro': return half ? 256 : 512;
      // The terrain near-detail suite covers a fixed 2 m of world, so this is a
      // texel-density choice and not a feature-size one: 256 is 7.8 mm a texel,
      // about the finest an anisotropic tap still resolves on ground seen at
      // grazing incidence from standing height. Every feature frequency in
      // `_buildTerrainSurface` is derived from a physical size, so ULTRA's 512
      // buys genuinely finer grain rather than the same map interpolated. LOW
      // and MEDIUM keep 256: their detail map has faded out entirely by 55-85 m,
      // so the map is only ever seen close, which is where the density is
      // needed. Two RGBA maps: 350 kB with mips at 256, 1.4 MB at 512.
      case 'terrain': return ultra ? 512 : 256;
      // Region/patch variation. One texel is 8 m through terrain's macro tap and
      // 1 m through its meso tap, which already resolves patch boundaries finer
      // than a patch boundary is: the strength of this map is its *distribution*
      // (see `_buildTerrainMacro`), not its resolution, and 256 measured 54 ms
      // more of load time for structure no camera position can reach. ULTRA
      // takes it anyway.
      case 'terrainMacro': return ultra ? 256 : 128;
      case 'blossom': return low ? 256 : 512;
      case 'petal': return low ? 128 : 256;
      case 'grass': return half ? 128 : 256;
      case 'stars': return low ? 1024 : ultra ? 4096 : 2048;
      case 'rain': return half ? 128 : 256;
      case 'ripple': return 128;
      case 'blue': return 64;
      case 'noise2d': return half ? 256 : 512;
      case 'cloudShape': return low ? 48 : t === 'medium' ? 64 : ultra ? 128 : 96;
      case 'cloudDetail': return low ? 24 : ultra ? 48 : 32;
      default: return 512;
    }
  }

  // -------------------------------------------------------------------------
  // Core API
  // -------------------------------------------------------------------------

  /**
   * Memoised texture accessor - the contract entry point.
   * `generatorFn` may be omitted for any built-in name (see `builtinNames()`),
   * which lets sibling systems ask for `textures.get('barkAlbedo')` without
   * knowing how it is made.
   */
  get(name, generatorFn, options) {
    if (this.cache.has(name)) return this.cache.get(name);

    let gen = generatorFn;
    let opts = options;

    // Factory-owned names win over the caller's generator; see FACTORY_OWNED.
    // No `_recipes` entry is stored for these, so `rebake()` falls through to
    // the built-in and rebuilds ours rather than the caller's fallback.
    if (typeof gen === 'function' && FACTORY_OWNED.has(normKey(name))) {
      const owned = this._builtins.get(normKey(name));
      if (owned) {
        const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
        try {
          const tex = owned();
          if (tex && tex.isTexture) {
            if (opts) this._apply(tex, opts);
            // The group builder registers its own members, so only account for
            // the texture once.
            if (!this.cache.has(name)) this._register(name, tex, t0);
            return tex;
          }
        } catch (err) {
          console.error(`[TextureFactory] built-in "${name}" threw; using the caller's generator:`, err);
        }
      }
    }

    if (typeof gen !== 'function') {
      const b = this._builtins.get(normKey(name));
      if (!b) {
        console.warn(`[TextureFactory] "${name}" is not a built-in and no generator was supplied.`);
        const fb = makeFallback(name);
        this.cache.set(name, fb);
        return fb;
      }
      // Built-ins register themselves under their canonical names; look again.
      const tex = b();
      // The contract signature is `get(name, generatorFn, options)`; options
      // were being dropped on this branch, so `get('groundNormal', null, {...})`
      // silently ignored anisotropy, wrap and filter overrides.
      if (opts) this._apply(tex, opts);
      this.cache.set(name, tex);
      return tex;
    }

    const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    let tex;
    try {
      tex = gen();
    } catch (err) {
      console.error(`[TextureFactory] generator for "${name}" threw:`, err);
      tex = makeFallback(name);
    }
    if (!tex || !tex.isTexture) {
      console.warn(`[TextureFactory] generator for "${name}" did not return a Texture.`);
      tex = makeFallback(name);
    }
    if (opts) this._apply(tex, opts);
    this._register(name, tex, t0);
    if (typeof gen === 'function') this._recipes.set(name, { gen, opts });
    return tex;
  }

  has(name) {
    return this.cache.has(name);
  }

  /** Every name `get()` understands without a generator. */
  builtinNames() {
    return Array.from(this._builtins.keys());
  }

  /**
   * Draw into a 2D canvas and wrap it as a texture.
   * `size` may be a number (square) or `{width, height}`.
   *
   * `options.bleed` runs an alpha-dilation pass: canvas compositing zeroes RGB
   * wherever alpha is zero, and bilinear filtering then drags that black into
   * the visible edge as a dark halo. Dilating opaque colour outward fixes it.
   * Anything with a cutout should pass `{ bleed: 4 }`.
   */
  canvas(size, drawFn, options = {}) {
    const w = typeof size === 'number' ? size : size.width;
    const h = typeof size === 'number' ? size : size.height;
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    const g = c.getContext('2d', { willReadFrequently: !!options.bleed });
    drawFn(g, w, h);

    if (options.bleed) {
      const img = g.getImageData(0, 0, w, h);
      this._alphaBleed(img.data, w, h, options.bleed | 0, options.wrap !== false);
      g.putImageData(img, 0, 0);
    }

    const t = new THREE.CanvasTexture(c);
    t.colorSpace = options.srgb === false ? THREE.NoColorSpace : THREE.SRGBColorSpace;
    this._apply(t, options);
    t.needsUpdate = true;
    return t;
  }

  /** Dilate opaque RGB outward into transparent texels. In-place. */
  _alphaBleed(data, w, h, passes, wrap) {
    const filled = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) filled[i] = data[i * 4 + 3] > 0 ? 1 : 0;
    const next = new Uint8Array(w * h);
    for (let p = 0; p < passes; p++) {
      next.set(filled);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const i = y * w + x;
          if (filled[i]) continue;
          let r = 0, gg = 0, b = 0, n = 0;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (!dx && !dy) continue;
              let xx = x + dx, yy = y + dy;
              if (wrap) { xx = (xx + w) % w; yy = (yy + h) % h; }
              else if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
              const j = yy * w + xx;
              if (!filled[j]) continue;
              r += data[j * 4]; gg += data[j * 4 + 1]; b += data[j * 4 + 2]; n++;
            }
          }
          if (n) {
            data[i * 4] = (r / n) | 0;
            data[i * 4 + 1] = (gg / n) | 0;
            data[i * 4 + 2] = (b / n) | 0;
            next[i] = 1;
          }
        }
      }
      filled.set(next);
    }
  }

  /**
   * General tiling 2D noise, RGBA.
   * `opts.channels` is an array of up to 4 descriptors; each is
   *   { type:'fbm'|'ridged'|'billow'|'worley'|'cells'|'white',
   *     fx, fy, octaves, gain, lacunarity, seed, invert }
   * The default packs a genuinely useful utility set:
   *   R low-freq fbm · G high-freq fbm · B worley cells · A worley walls.
   */
  noise2D(size = this._size('noise2d'), opts = {}) {
    const w = opts.width || size;
    const h = opts.height || size;
    const seed = (opts.seed ?? 0) + this.seed;
    const channels = opts.channels || [
      { type: 'fbm', fx: 4, octaves: 4 },
      { type: 'fbm', fx: 17, octaves: 4 },
      { type: 'worley', fx: 8 },
      { type: 'worley', fx: 16, mode: 1 },
    ];
    const data = new Uint8Array(w * h * 4);
    data.fill(255);
    for (let c = 0; c < 4 && c < channels.length; c++) {
      const spec = channels[c];
      if (!spec) continue;
      const field = this._field2(w, h, spec, seed + c * 7919);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const i = y * w + x;
          data[i * 4 + c] = q8(spec.invert ? 1 - field[i] : field[i], dither(x, y));
        }
      }
    }
    const t = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.UnsignedByteType);
    this._apply(t, {
      srgb: false,
      repeat: true,
      mipmaps: opts.mipmaps !== false,
      anisotropy: opts.anisotropy ?? this.maxAnisotropy,
    });
    t.needsUpdate = true;
    return t;
  }

  /** Dispatch one channel descriptor to the right generator, normalised 0..1. */
  _field2(w, h, spec, seed) {
    const fx = Math.max(1, Math.round(spec.fx ?? 8));
    const fy = Math.max(1, Math.round(spec.fy ?? fx));
    const s = (seed + (spec.seed ?? 0)) | 0;
    switch (spec.type) {
      case 'worley':
      case 'cells':
        return worley2Scaled(w, h, fx, fy, s, spec.mode ?? 0, spec.jitter ?? 1, spec.spc ?? 10);
      case 'white': {
        const f = new Float32Array(w * h);
        // h2, not hash2: a "white" channel is meant to fill 0..1 - see HASH_SCALE.
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) f[y * w + x] = h2(x + s, y * 31 + s);
        return f;
      }
      default: {
        const kind = spec.type === 'ridged' ? 2 : spec.type === 'billow' ? 1 : 0;
        const lac = spec.lacunarity ?? 2;
        // The default channel set asks for four octaves from 17 cells, which at
        // the LOW tier's 256² lands its finest at 143 cells - 1.8 texels a cell.
        // That is aliasing, not detail: it sparkles in the base mip and is gone
        // one level down. The caller's count is an upper bound, not a promise.
        const oct = octavesFor(Math.min(w / fx, h / fy), lac, spec.octaves ?? 4);
        const f = fbm2(w, h, {
          fx, fy,
          octaves: oct,
          lacunarity: lac,
          gain: spec.gain ?? 0.5,
          kind,
          seed: s,
        });
        return normalizeField(f);
      }
    }
  }

  /**
   * General tiling 3D noise, RGBA8 `Data3DTexture`.
   * Defaults to the cloud shape volume; pass `opts.channels` for anything else.
   */
  noise3D(size = this._size('cloudShape'), opts = {}) {
    if (!opts.channels) return this._buildCloudShape(size);
    const n = size;
    const data = new Uint8Array(n * n * n * 4);
    data.fill(255);
    for (let c = 0; c < 4 && c < opts.channels.length; c++) {
      const spec = opts.channels[c];
      if (!spec) continue;
      const f = Math.max(1, Math.round(spec.f ?? 8));
      const s = (this.seed + (spec.seed ?? 0) + c * 5501) | 0;
      let field;
      if (spec.type === 'worley' || spec.type === 'cells') {
        field = worley3Scaled(n, f, s, spec.spc ?? 6);
      } else {
        const kind = spec.type === 'ridged' ? 2 : spec.type === 'billow' ? 1 : 0;
        field = normalizeField(fbm3(n, {
          f, octaves: spec.octaves ?? 3, gain: spec.gain ?? 0.5, kind, seed: s,
        }));
      }
      for (let i = 0; i < field.length; i++) {
        data[i * 4 + c] = q8(spec.invert ? 1 - field[i] : field[i], dither(i, i >> 6));
      }
    }
    return this._make3D(data, n, opts);
  }

  // -------------------------------------------------------------------------
  // LEAF 1 - cloud volumes and blue noise
  // -------------------------------------------------------------------------

  /**
   * Perlin-Worley cloud shape volume (Schneider/Guerrilla, Horizon Zero Dawn).
   *   R = perlin remapped by an inverted worley fbm - billowy cumulus base
   *   G/B/A = worley fbms at rising frequency, for progressive edge erosion
   *
   * A cloud shader typically does:
   *   base   = tex.r
   *   detail = tex.g*0.625 + tex.b*0.25 + tex.a*0.125
   *   density = remap(base * coverageShape, detail * erosion, 1, 0, 1)
   */
  _buildCloudShape(n) {
    // Frequencies scale with the volume so LOW does not alias into static.
    const base = Math.max(2, Math.round(n / 24));
    const f0 = base, f1 = base * 2, f2 = base * 4, f3 = base * 5;
    const s = this.seed;

    const w0 = worley3Scaled(n, f0, s + 1);
    const w1 = worley3Scaled(n, f1, s + 2);
    const w2 = worley3Scaled(n, f2, s + 3);
    const w3 = worley3Scaled(n, f3, s + 4);

    // Perlin base: billowy rather than signed, which reads as cumulus mass.
    const perlin = normalizeField(fbm3(n, {
      f: Math.max(2, Math.round(n / 24)),
      octaves: 4, gain: 0.55, lacunarity: 2.02, kind: 1, seed: s + 91,
    }));

    const count = n * n * n;
    const chR = new Float32Array(count);
    const chG = new Float32Array(count);
    const chB = new Float32Array(count);
    const chA = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const a0 = w0[i], a1 = w1[i], a2 = w2[i], a3 = w3[i];
      const wLow = 0.625 * a0 + 0.25 * a1 + 0.125 * a2;
      // Schneider's remap: the perlin field is squeezed into the range left by
      // the worley fbm, which is what gives clouds their cauliflower silhouette
      // instead of the smooth blobs a plain perlin threshold produces.
      const denom = 2 - wLow;
      chR[i] = (perlin[i] - wLow + 1) / (denom > 1e-4 ? denom : 1e-4);
      chG[i] = wLow;
      chB[i] = 0.625 * a1 + 0.25 * a2 + 0.125 * a3;
      chA[i] = 0.625 * a2 + 0.375 * a3;
    }
    // Each channel is stretched to fill 0..1. The remap above only ever spans
    // roughly 0.16..1, and a cloud raymarch samples this texture more than
    // anything else in the frame - throwing away a fifth of the 8-bit range
    // there is precision nobody can afford.
    normalizeField(chR, 0.001, 0.999);
    normalizeField(chG, 0.001, 0.999);
    normalizeField(chB, 0.001, 0.999);
    normalizeField(chA, 0.001, 0.999);

    const data = new Uint8Array(count * 4);
    const nn = n * n;
    for (let i = 0; i < count; i++) {
      const o = i * 4;
      const dx = i % n, dy = ((i / n) | 0) % n, dz = (i / nn) | 0;
      const d = dither(dx + dz, dy + dz);
      data[o] = q8(chR[i], d);
      data[o + 1] = q8(chG[i], d);
      data[o + 2] = q8(chB[i], d);
      data[o + 3] = q8(chA[i], d);
    }
    return this._make3D(data, n, { name: 'cloudShape' });
  }

  /** High-frequency erosion volume: cellular wisps plus one non-cellular curl-ish channel. */
  _buildCloudDetail(n) {
    const base = Math.max(2, Math.round(n / 8));
    const s = this.seed + 300;
    const d0 = worley3Scaled(n, base, s + 1, 5);
    const d1 = worley3Scaled(n, base * 2, s + 2, 5);
    const d2 = worley3Scaled(n, base * 3, s + 3, 4);
    // A ridged perlin channel: worley alone erodes into uniform bubbles, and
    // real cloud edges have stringy, non-cellular filaments too.
    const wisp = normalizeField(fbm3(n, {
      f: base * 2, octaves: 3, gain: 0.55, kind: 2, seed: s + 77, spc: 5,
    }));

    const count = n * n * n;
    const chR = new Float32Array(count);
    const chG = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      chR[i] = 0.625 * d0[i] + 0.25 * d1[i] + 0.125 * d2[i];
      chG[i] = 0.625 * d1[i] + 0.375 * d2[i];
    }
    normalizeField(chR, 0.001, 0.999);
    normalizeField(chG, 0.001, 0.999);

    const data = new Uint8Array(count * 4);
    const nn = n * n;
    for (let i = 0; i < count; i++) {
      const o = i * 4;
      const dx = i % n, dy = ((i / n) | 0) % n, dz = (i / nn) | 0;
      const d = dither(dx + dz, dy + dz);
      data[o] = q8(chR[i], d);
      data[o + 1] = q8(chG[i], d);
      data[o + 2] = q8(wisp[i], d);
      data[o + 3] = q8(d2[i], d);
    }
    return this._make3D(data, n, { name: 'cloudDetail' });
  }

  perlinWorley3D(size) {
    return this.get('cloudShape', () => this._buildCloudShape(size || this._size('cloudShape')));
  }

  cloudDetail3D(size) {
    return this.get('cloudDetail', () => this._buildCloudDetail(size || this._size('cloudDetail')));
  }

  /**
   * Blue-noise mask for raymarch offsets, dithering and stochastic alpha.
   *   R = void-and-cluster rank
   *   G = R advanced by the golden ratio - add `frame * 0.618034` to R yourself
   *       for temporally stable animated noise, or just sample G on odd frames
   *   B = a second, independent pattern (use with R for 2D sample directions)
   *   A = R offset by 2φ
   * Nearest filtering and no mipmaps: blue noise that gets filtered is grey noise.
   */
  blueNoise(size) {
    const n = size || this._size('blue');
    return this.get(`blueNoise${n}`, () => {
      const a = voidAndCluster(n, this.seed ^ 0x9e37);
      const b = voidAndCluster(n, this.seed ^ 0x4f1b);
      const data = new Uint8Array(n * n * 4);
      const PHI = 0.6180339887498949;
      for (let i = 0; i < n * n; i++) {
        const r = a[i];
        data[i * 4] = Math.min(255, (r * 256) | 0);
        data[i * 4 + 1] = Math.min(255, (((r + PHI) % 1) * 256) | 0);
        data[i * 4 + 2] = Math.min(255, (b[i] * 256) | 0);
        data[i * 4 + 3] = Math.min(255, (((r + 2 * PHI) % 1) * 256) | 0);
      }
      const t = new THREE.DataTexture(data, n, n, THREE.RGBAFormat, THREE.UnsignedByteType);
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      t.magFilter = t.minFilter = THREE.NearestFilter;
      t.generateMipmaps = false;
      t.colorSpace = THREE.NoColorSpace;
      t.needsUpdate = true;
      return t;
    });
  }

  /**
   * Tiling 2D curl-noise vector field, RG = normalised curl, B = |curl|,
   * A = the scalar potential. Divergence-free, so anything advected by it
   * swirls instead of piling up - gusts, mist drift, petal eddies.
   */
  curlNoise2D(size = 256, freq = 6) {
    return this.get(`curl2D_${size}_${freq}`, () => {
      const pot = normalizeField(fbm2(size, size, { fx: freq, octaves: 4, gain: 0.5, seed: this.seed + 55 }));
      const data = new Uint8Array(size * size * 4);
      // Curl of a 2D potential is (dP/dy, -dP/dx); central differences with wrap.
      let maxMag = 1e-6;
      const cx = new Float32Array(size * size);
      const cy = new Float32Array(size * size);
      for (let y = 0; y < size; y++) {
        const yn = ((y - 1) + size) % size, yp = (y + 1) % size;
        for (let x = 0; x < size; x++) {
          const xn = ((x - 1) + size) % size, xp = (x + 1) % size;
          const gx = (pot[y * size + xp] - pot[y * size + xn]) * 0.5;
          const gy = (pot[yp * size + x] - pot[yn * size + x]) * 0.5;
          const vx = gy, vy = -gx;
          cx[y * size + x] = vx;
          cy[y * size + x] = vy;
          const m = Math.hypot(vx, vy);
          if (m > maxMag) maxMag = m;
        }
      }
      const inv = 1 / maxMag;
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const i = y * size + x;
          const vx = cx[i] * inv, vy = cy[i] * inv;
          const d = dither(x, y);
          data[i * 4] = q8(vx * 0.5 + 0.5, d);
          data[i * 4 + 1] = q8(vy * 0.5 + 0.5, d);
          data[i * 4 + 2] = q8(Math.hypot(vx, vy), d);
          data[i * 4 + 3] = q8(pot[i], d);
        }
      }
      const t = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
      this._apply(t, { srgb: false, repeat: true, mipmaps: true });
      t.needsUpdate = true;
      return t;
    });
  }

  // -------------------------------------------------------------------------
  // LEAF 4 helper - the reusable height→normal converter
  // -------------------------------------------------------------------------

  /**
   * Seamless tangent-space normal map from a 0..1 height field.
   *
   * Uses a wrapped 3×3 Sobel rather than a bare central difference: the extra
   * vertical taps low-pass the gradient, which measurably reduces the specular
   * sparkle that plain central differences produce on high-frequency fields.
   *
   * `strength` is resolution-independent - the gradient is scaled by
   * `size/128`, so a 512² and a 1024² bake of the same field give the same
   * apparent relief instead of the finer one going flat.
   *
   * The alpha channel carries the source height; parallax, blend masks and
   * puddle logic all want it and it is free here.
   *
   * @returns {Uint8Array} RGBA, ready for a DataTexture.
   */
  normalFromHeight(height, w, h, opts = {}) {
    const strength = opts.strength ?? 1;
    const out = opts.target || new Uint8Array(w * h * 4);
    const k = strength * Math.max(w, h) / 128;
    const flipY = opts.flipGreen ? -1 : 1;
    // `wrap: false` for a ClampToEdge map, so the border difference matches what
    // the sampler does rather than folding the opposite edge in. See
    // `normalMipChain` for why this is not cosmetic on a card atlas.
    const wrap = opts.wrap !== false;
    const wx = wrap ? (x) => ((x % w) + w) % w : (x) => (x < 0 ? 0 : x >= w ? w - 1 : x);
    const wy = wrap ? (y) => ((y % h) + h) % h : (y) => (y < 0 ? 0 : y >= h ? h - 1 : y);
    for (let y = 0; y < h; y++) {
      const y0 = wy(y - 1), y1 = wy(y + 1);
      const r0 = y0 * w, r1 = y * w, r2 = y1 * w;
      for (let x = 0; x < w; x++) {
        const x0 = wx(x - 1), x1 = wx(x + 1);
        const h00 = height[r0 + x0], h10 = height[r0 + x], h20 = height[r0 + x1];
        const h01 = height[r1 + x0], h21 = height[r1 + x1];
        const h02 = height[r2 + x0], h12 = height[r2 + x], h22 = height[r2 + x1];
        // Sobel: dx = right column - left column, dy = bottom row - top row.
        const gx = (h20 + 2 * h21 + h22 - h00 - 2 * h01 - h02) * 0.25;
        const gy = (h02 + 2 * h12 + h22 - h00 - 2 * h10 - h20) * 0.25;
        // Surface (u, v, h) has normal (-dh/du, -dh/dv, 1).
        let nx = -gx * k, ny = -gy * k * flipY, nz = 1;
        const inv = 1 / Math.sqrt(nx * nx + ny * ny + 1);
        nx *= inv; ny *= inv; nz = inv;
        const i = (r1 + x) * 4;
        const d = dither(x, y);
        out[i] = q8(nx * 0.5 + 0.5, d);
        out[i + 1] = q8(ny * 0.5 + 0.5, d);
        out[i + 2] = q8(nz * 0.5 + 0.5, d);
        out[i + 3] = q8(height[r1 + x], d);
      }
    }
    return out;
  }

  /** `normalFromHeight` wrapped straight into a configured DataTexture. */
  normalTexture(height, w, h, opts = {}) {
    const data = this.normalFromHeight(height, w, h, opts);
    const t = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.UnsignedByteType);
    this._apply(t, {
      srgb: false,
      repeat: opts.repeat !== false,
      mipmaps: opts.mipmaps !== false,
      anisotropy: opts.anisotropy ?? this.maxAnisotropy,
    });
    t.needsUpdate = true;
    return t;
  }

  // -------------------------------------------------------------------------
  // LEAF 2 - sakura bark
  // -------------------------------------------------------------------------

  /**
   * Bark suite for Prunus serrulata.
   *
   * The species reads as: a smooth, faintly glossy grey-brown ground; bold
   * *horizontal* lenticel dashes (the single most identifying feature - cherry
   * is the tree with dashes running around the trunk); shallow, wandering
   * vertical fissures rather than the deep plated relief of oak; and patches of
   * sage lichen sitting on the raised plates. Generic brown fbm gets none of
   * that, which is why every field below is a specific structure rather than
   * another octave.
   *
   * @returns {{albedo:THREE.Texture, normal:THREE.Texture, orm:THREE.Texture,
   *            roughness:THREE.Texture, ao:THREE.Texture, height:Float32Array}}
   */
  barkSet() {
    return this._group('bark', () => this._buildBark(this._size('bark')));
  }

  _buildBark(S) {
    const N = S * S;
    const seed = this.seed + 4100;

    // --- vertical fissures -------------------------------------------------
    // Anisotropy comes from the lattice periods: ~7 cells around the trunk
    // against ~2 along it, so features are roughly 4× taller than wide.
    //
    // Straight fissures are the giveaway that noise was stretched rather than
    // grown, so each octave is sampled through a domain warp that makes the
    // creases wander and occasionally merge, as real bark splits do. Each octave
    // is evaluated at its own resolution - ten samples per lattice cell - and
    // wrap-upsampled, warp included: the warp is low frequency by construction,
    // so evaluating it per texel for every octave was five sixths of this
    // function's cost for no information at all. Five octaves now cost a
    // seventh of what four used to.
    const fiss = new Float32Array(N);
    {
      // Four octaves, not five: the fifth sits at ten texels and its 5% weight
      // is swallowed by the fine grain field that overlaps it, so it was a
      // full-resolution pass over a megapixel for nothing visible.
      const FX = [7, 15, 29, 57];
      const FY = [2, 5, 11, 23];
      const AMP = [0.45, 0.28, 0.17, 0.10];
      for (let o = 0; o < FX.length; o++) {
        const fx = FX[o], fy = FY[o];
        const rw = octaveRes(S, fx, 10);
        const rh = octaveRes(S, fy, 10);
        const tmp = new Float32Array(rw * rh);
        const s = seed + o * 37 + 5;
        for (let y = 0; y < rh; y++) {
          const v = (y + 0.5) / rh;
          const row = y * rw;
          for (let x = 0; x < rw; x++) {
            const u = (x + 0.5) / rw;
            // The warp field is itself periodic, so u + w(u,v) advances by
            // exactly 1 when u does and the tile still wraps byte-exactly.
            const wu = u + perlin2P(u * 3, v * 2, 3, 2, seed + 11) * 0.05;
            const wv = v + perlin2P(u * 2, v * 3, 2, 3, seed + 23) * 0.018;
            const r = 1 - Math.abs(perlin2P(wu * fx, wv * fy, fx, fy, s));
            tmp[row + x] = r * r;
          }
        }
        if (rw === S && rh === S) {
          for (let i = 0; i < N; i++) fiss[i] += AMP[o] * tmp[i];
        } else {
          upsampleAdd2(tmp, rw, rh, fiss, S, S, AMP[o]);
        }
      }
      normalizeField(fiss, 0.02, 0.995);
      // Push the distribution toward the low end: bark is mostly smooth plate
      // with a few deep splits, not a uniform corduroy.
      for (let i = 0; i < N; i++) {
        const f = fiss[i];
        fiss[i] = f * f * f * (0.35 + 0.65 * f);
      }
    }

    // --- plates ------------------------------------------------------------
    // Broad relief. One pass yields both the plate interiors and the joints
    // between them; they share a cell layout, so computing them separately
    // would be exactly twice the work for the same field.
    const plates = worley2Pair(S, S, 5, 3, seed + 71, 0.85, 14);
    const plateFill = plates.cells;
    const plate = plates.walls;
    // Re-sharpen the joint after upsampling. The underlying F2-F1 crease is
    // linear either side of the wall so it magnifies cleanly, but it needs a
    // curve to read as a narrow joint rather than a broad soft trough.
    for (let i = 0; i < N; i++) {
      const v = plate[i];
      plate[i] = v * v * v * v;
    }

    // --- lenticels ---------------------------------------------------------
    // Rasterised per dash into its own box; 24×64 cells over a tile intended to
    // cover ~1 m of trunk gives dashes ~15 mm long on ~15 mm rows, which is
    // the real spacing on a mature Somei-Yoshino.
    const lent = new Float32Array(N);
    const lentRim = new Float32Array(N);
    {
      const LU = 24, LV = 64;
      const cw = S / LU, ch = S / LV;
      const rand = makeRNG(seed + 909);
      // Row positions are a cumulative random walk, not `j * ch`.
      //
      // The columns were fixed by walking them at accumulated gaps; the rows
      // were left on a lattice with a ±0.325-row jitter on top, and jittering a
      // lattice preserves its pitch - the same lesson, the other axis. The
      // vertical autocorrelation of the finished albedo showed the residual as
      // local maxima at lags 10, 23 and 35, i.e. a ripple at exactly the
      // S/64 row spacing. Gaps drawn over 0.55..1.45 of the mean and then
      // rescaled to sum to exactly S destroy the pitch while still closing over
      // the tile, so V still wraps byte-exactly and no two rows collide at the
      // seam the way a free-running walk would let them.
      const rowGap = new Float32Array(LV);
      let gapSum = 0;
      for (let j = 0; j < LV; j++) { rowGap[j] = 0.55 + 0.90 * rand(); gapSum += rowGap[j]; }
      const gapK = S / gapSum;
      const rowY = new Float32Array(LV);
      let rowAcc = rand() * S;
      for (let j = 0; j < LV; j++) { rowY[j] = rowAcc; rowAcc += rowGap[j] * gapK; }
      // Dashes are walked along each row at randomly-drawn gaps rather than
      // placed on a lattice of LU columns. Jittering a lattice - which is what
      // the first two drafts did - leaves the lattice *pitch* intact no matter
      // how hard you jitter, and an autocorrelation of the finished map showed
      // a 0.24 spike at exactly the column spacing: a readable vertical grid
      // across the whole trunk. Accumulated random gaps have no pitch at all.
      for (let j = 0; j < LV; j++) {
        // Rows vary in how crowded they are.
        const gapScale = 0.62 + 1.05 * rand() * rand();
        let cxp = rand() * cw * 2;
        while (cxp < S) {
          // Occasionally a few pores have fused into one long bar. This is the
          // feature that reads as "cherry" from three metres and it is the one
          // an even scatter of identical dashes can never produce.
          const fuse = rand() < 0.13 ? 1.6 + 1.6 * rand() : 1;
          const cyp = rowY[j] + 0.30 * (rand() - 0.5) * ch;
          const halfL = cw * (0.16 + 0.30 * rand()) * fuse;
          const halfH = ch * (0.10 + 0.13 * rand());
          // A slight tilt; perfectly horizontal dashes look printed on.
          const tilt = (rand() - 0.5) * 0.16;
          const step = cw * gapScale * (0.45 + 1.5 * rand() * rand());
          // Advance before rasterising: nothing below touches cxp, and doing it
          // here means no path through the body can leave the walk stalled.
          cxp += Math.max(halfL * 2 + cw * 0.12, step);
          const bx = Math.ceil(halfL + Math.abs(tilt) * halfL) + 3;
          const by = Math.ceil(halfH) + 4;
          const px = Math.round(cxp), py = Math.round(cyp);
          for (let dy = -by; dy <= by; dy++) {
            const yy = ((py + dy) % S + S) % S;
            const row = yy * S;
            for (let dx = -bx; dx <= bx; dx++) {
              const xx = ((px + dx) % S + S) % S;
              const lx = dx;
              const ly = dy - lx * tilt;
              const e = (lx / halfL) * (lx / halfL) + (ly / halfH) * (ly / halfH);
              // Superellipse-ish falloff: dashes have blunt ends, not points.
              const core = clamp01(1.25 - e * 1.25);
              const v = core * core;
              const k = row + xx;
              if (v > lent[k]) lent[k] = v;
              // Rim sits just below the dash - a lenticel is a raised pore with
              // a shadow line under it, and that line is what sells the relief.
              const ry = ly - halfH * 0.85;
              const er = (lx / (halfL * 1.05)) * (lx / (halfL * 1.05)) +
                (ry / (halfH * 0.9)) * (ry / (halfH * 0.9));
              const rv = clamp01(1 - er) * 0.9;
              if (rv > lentRim[k]) lentRim[k] = rv;
            }
          }
        }
      }
    }

    // --- fine grain, lichen, tonal drift -----------------------------------
    // spc 6 rather than 8: the grain's finest octave is already inside four
    // texels of Nyquist, so the extra samples buy nothing and cost a full-res
    // pass over a megapixel.
    const grain = normalizeField(fbm2(S, S, { fx: 90, fy: 26, octaves: 2, gain: 0.55, spc: 6, seed: seed + 131 }));
    // Cherry bark bands *horizontally* - around the trunk, not up it - because
    // that is the direction a season's growth runs. U is around the trunk here,
    // so a horizontal band means low frequency in U and high in V. The first
    // draft had fx 3 / fy 2, which is the wrong way round and gave isotropic
    // blotches that could have been any species at all.
    const tonal = normalizeField(fbm2(S, S, { fx: 2, fy: 8, octaves: 3, gain: 0.55, seed: seed + 211 }));
    const tonalBlot = normalizeField(fbm2(S, S, { fx: 4, fy: 3, octaves: 3, seed: seed + 217 }));
    // Crustose lichen on a cherry trunk is 10-40 mm flecks, not the 200 mm
    // splodges the first draft produced - those read as camouflage and, being
    // the largest and highest-contrast feature in the map, were also exactly the
    // landmark that makes a tiled trunk obviously tiled.
    const lichenF = normalizeField(fbm2(S, S, { fx: 22, fy: 16, octaves: 3, gain: 0.5, spc: 6, seed: seed + 313 }));
    const lichenD = normalizeField(fbm2(S, S, { fx: 42, fy: 36, octaves: 2, spc: 6, seed: seed + 317 }));

    // --- height ------------------------------------------------------------
    const height = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      let hgt = 0.5;
      hgt += (plateFill[i] - 0.5) * 0.18;
      hgt -= plate[i] * 0.12;
      // Cherry is a *smooth*-barked species: shallow splits in a broad plate,
      // not the deep plated relief of oak. 0.62 was oak.
      hgt -= fiss[i] * 0.44;
      hgt += lent[i] * 0.10;
      hgt -= lentRim[i] * 0.05;
      hgt += (grain[i] - 0.5) * 0.05;
      height[i] = hgt;
    }
    normalizeField(height, 0.005, 0.995);

    // --- curvature-based occlusion ----------------------------------------
    // h minus a blurred h is a decent cheap curvature signal; concave texels
    // (fissure floors) come out negative and get darkened and made wetter.
    const blurR = Math.max(2, Math.round(S / 128));
    const hBlur = boxBlurWrapFast(height, S, S, blurR);
    const cavity = new Float32Array(N);
    for (let i = 0; i < N; i++) cavity[i] = clamp01((hBlur[i] - height[i]) * 3.2 + 0.12);

    // --- palette (sRGB, deliberately low chroma) ---------------------------
    // Checked in linear reflectance, not by eye: Somei-Yoshino bark is a mid
    // grey-brown around 0.09 linear with a faint purple cast, and the previous
    // set averaged 0.066 - dark enough that the trunk read as a silhouette in
    // anything but direct sun.
    const cDark = rgb(0x322820);
    const cMid = rgb(0x5a4b3f);
    const cLight = rgb(0x7d6c5b);
    const cFiss = rgb(0x231b16);
    const cLent = rgb(0xa89279);
    const cLentHot = rgb(0xc4b094);
    const cLichen = rgb(0x76806e);
    const cLichenPale = rgb(0x929a8a);

    const albedo = new Uint8Array(N * 4);
    const orm = new Uint8Array(N * 4);

    for (let y = 0; y < S; y++) {
      const row = y * S;
      for (let x = 0; x < S; x++) {
        const i = row + x;
        const d = dither(x, y);

        // Growth banding dominates, with a weaker isotropic blotch on top so
        // the bands do not read as printed stripes.
        const t = clamp01(tonal[i] * 0.72 + tonalBlot[i] * 0.28);
        const g = grain[i];
        const f = fiss[i];
        const le = lent[i];
        const lr = lentRim[i];

        // Base: dark → mid → light along the tonal drift, so the trunk keeps
        // large-scale value variation that survives mipping to distance.
        let r, gr, b;
        if (t < 0.5) {
          const k = t * 2;
          r = lerp(cDark[0], cMid[0], k); gr = lerp(cDark[1], cMid[1], k); b = lerp(cDark[2], cMid[2], k);
        } else {
          const k = (t - 0.5) * 2;
          r = lerp(cMid[0], cLight[0], k); gr = lerp(cMid[1], cLight[1], k); b = lerp(cMid[2], cLight[2], k);
        }

        // Grain modulates value only - tinting it would add chroma noise, and
        // low-chroma is the whole art direction.
        const gm = 0.90 + 0.20 * g;
        r *= gm; gr *= gm; b *= gm;

        // Fissures: darken hard and cool very slightly (they are in shadow).
        const fk = clamp01(f * 0.82);
        r = lerp(r, cFiss[0], fk); gr = lerp(gr, cFiss[1], fk); b = lerp(b, cFiss[2], fk);

        // Lenticel: pale tan bar with a dark line beneath it.
        const lk = clamp01(le * 0.92);
        const lentR = lerp(cLent[0], cLentHot[0], g);
        const lentG = lerp(cLent[1], cLentHot[1], g);
        const lentB = lerp(cLent[2], cLentHot[2], g);
        r = lerp(r, lentR, lk); gr = lerp(gr, lentG, lk); b = lerp(b, lentB, lk);
        const rk = clamp01(lr * 0.55) * (1 - lk);
        r = lerp(r, cFiss[0], rk); gr = lerp(gr, cFiss[1], rk); b = lerp(b, cFiss[2], rk);

        // Lichen: only on raised, unfissured plate - it cannot grow in a crack
        // that sheds water, and putting it there is the usual procedural error.
        const exposure = clamp01((height[i] - 0.42) * 2.6) * (1 - clamp01(f * 2.2));
        const lich = smoothstep(0.66, 0.90, lichenF[i]) * exposure * 0.46;
        if (lich > 0.002) {
          const lp = lichenD[i];
          const lr2 = lerp(cLichen[0], cLichenPale[0], lp);
          const lg2 = lerp(cLichen[1], cLichenPale[1], lp);
          const lb2 = lerp(cLichen[2], cLichenPale[2], lp);
          r = lerp(r, lr2, lich); gr = lerp(gr, lg2, lich); b = lerp(b, lb2, lich);
        }

        // Bake a little of the curvature occlusion into albedo. Real contact
        // shadowing in a crack is far darker than any AO term will deliver.
        const ao = clamp01(1 - cavity[i] * 0.55);
        const aoAlb = lerp(1, ao, 0.45);
        r *= aoAlb; gr *= aoAlb; b *= aoAlb;

        const o = i * 4;
        albedo[o] = q8(r / 255, d);
        albedo[o + 1] = q8(gr / 255, d);
        albedo[o + 2] = q8(b / 255, d);
        albedo[o + 3] = 255;

        // Roughness: cherry bark is smooth and slightly waxy on the plates,
        // matte in the splits, and lichen is the roughest thing on the tree.
        let rough = 0.54;
        rough += f * 0.30;
        rough += le * 0.10;
        rough += lich * 0.38;
        rough -= clamp01((height[i] - 0.55) * 1.4) * 0.09;
        rough += (g - 0.5) * 0.05;

        orm[o] = q8(ao, d);
        orm[o + 1] = q8(clamp(rough, 0.26, 0.97), d);
        orm[o + 2] = 0;
        // Cavity → wetness affinity. Rain runs down the fissures and pools in
        // them; the tree shader multiplies state.weather.wetness by this.
        orm[o + 3] = q8(clamp01(cavity[i] * 0.85 + f * 0.35), d);
      }
    }

    const albedoTex = this._dataTexMips(albedo, S, S, { srgb: true, name: 'barkAlbedo' });
    const pbr = this._pbrTextures(height, orm, S, S, {
      strength: 1.15, normalName: 'barkNormal', ormName: 'barkORM',
    });
    // Bark deliberately keeps its low frequencies, unlike the ground maps: the
    // growth banding *is* the species read, and a trunk is normally about one
    // tile tall so the banding never gets a chance to repeat vertically. What
    // was removed instead is the one feature big enough to be a landmark when a
    // branch does repeat the map - the oversized lichen.
    for (const t of [albedoTex, pbr.normal, pbr.orm]) t.userData.tileMeters = BARK_TILE_M;

    this._register('barkAlbedo', albedoTex);
    this._register('barkNormal', pbr.normal);
    this._register('barkORM', pbr.orm);

    return {
      albedo: albedoTex,
      normal: pbr.normal,
      orm: pbr.orm,
      roughness: pbr.orm,
      ao: pbr.orm,
      height,
      size: S,
      tileMeters: BARK_TILE_M,
    };
  }

  // -------------------------------------------------------------------------
  // LEAF 3 - ground
  // -------------------------------------------------------------------------

  /**
   * Soil / bare earth suite.
   *
   * WHAT WENT WRONG BEFORE, because it is the instructive part: the height field
   * was three superimposed inverted-F1 Worley fields (clods at 14 cells, grit at
   * 46, pebble domes at 26). Inverted F1 is a *radial cone centred on every
   * feature point*, so however hard the points are jittered, every feature comes
   * out the same round dome of the same size on a grid of the same pitch. That
   * is bubble wrap, and it was 70% of the height field's amplitude.
   *
   * What aggregated earth actually looks like at centimetre scale is a mosaic of
   * flat-ish fragments at slightly different heights, separated by thin voids,
   * with the visible relief dominated by grain an order of magnitude finer than
   * the fragments - plus, under a grass field, a mat of dead stems. So:
   *
   *   - fragments come from `worley2Agg`'s per-cell `id` (flat plateaus) and
   *     `walls` (narrow creases), never from `cells`;
   *   - the largest amplitude in the height field belongs to the two finest
   *     value-noise octaves, at ~12 mm and ~26 mm;
   *   - dead-stem thatch and a few part-buried pebbles are stamped as real
   *     sprites, because no noise field produces a strand or a stone;
   *   - the colour fields are high-passed, so the detail map carries no
   *     low-frequency landmark for the eye to lock onto one tile away. Broad
   *     tonal drift is `groundMacro`'s job and only its job.
   *
   * @returns {{albedo,normal,orm,roughness,ao,height,size,tileMeters}}
   */
  groundSet() {
    return this._group('ground', () => this._buildGround(this._size('ground')));
  }

  _buildGround(S) {
    const N = S * S;
    const seed = this.seed + 6100;

    // Every frequency below is derived from a physical size. "How many cells
    // across the texture" is meaningless until you say how big the texture is
    // in the world, and being wrong about that by a factor of five is precisely
    // how procedural soil ends up looking like bubble wrap.
    const TILE_M = GROUND_TILE_M;
    const cellsFor = (sizeMM) => clamp(Math.round((TILE_M * 1000) / sizeMM), 2, 250);
    // The finest octave is also capped at ~3.2 texels per cell so the LOW tier
    // does not put the grain past Nyquist and turn it into sparkle.
    const finest = (sizeMM) => Math.min(cellsFor(sizeMM), Math.round(S / 3.2));

    // --- structure ---------------------------------------------------------
    const aggC = worley2Agg(S, S, cellsFor(78), cellsFor(78), seed + 3, { warp: 0.62 });
    const aggF = worley2Agg(S, S, cellsFor(31), cellsFor(31), seed + 9, { warp: 0.52 });
    // Narrow both wall masks. F2−F1 comes back as a broad linear ramp; the void
    // between two soil crumbs is a slot, not a valley.
    const wallC = aggC.walls, wallF = aggF.walls;
    for (let i = 0; i < N; i++) {
      const a = wallC[i], b = wallF[i];
      wallC[i] = a * a * a;
      wallF[i] = b * b * b * b;
    }
    // Soften the plateau steps by a couple of texels. The step sits exactly on
    // the crease between two fragments, which is where a discontinuity belongs,
    // but a one-texel cliff makes a razor-sharp normal that sparkles.
    const idC = boxBlurWrap(aggC.id, S, S, Math.max(1, Math.round(S / 220)));
    const idF = boxBlurWrap(aggF.id, S, S, Math.max(1, Math.round(S / 340)));

    // --- grain: the dominant relief ---------------------------------------
    // Two different *kinds* of noise at neighbouring frequencies. Value noise
    // alone leaves a faint square weave at three texels per cell (it is built on
    // an axis-aligned lattice with no gradient to hide it), and gradient noise
    // alone leaves its zero set. Superimposed at incommensurate frequencies each
    // conceals the other's signature, which neither could do on its own.
    // Octave counts are derived, not written down: at LOW (256²) the second
    // octave of the 27 mm layer landed at 148 cells - 1.7 texels a cell - and
    // sparkled. `octavesFor` drops it there and keeps it at HIGH and ULTRA,
    // where the resolution genuinely carries it.
    const gFine = finest(12);
    const gMidF = Math.max(2, Math.round(gFine * 0.71));
    const gCoarseF = cellsFor(27);
    const g1 = normalizeField(valueFbm2(S, S, { fx: gFine, octaves: 1, seed: seed + 21 }));
    const g1b = normalizeField(fbm2(S, S, { fx: gMidF, octaves: octavesFor(S / gMidF, 2, 2), gain: 0.5, seed: seed + 23 }));
    const g2 = normalizeField(valueFbm2(S, S, { fx: gCoarseF, octaves: octavesFor(S / gCoarseF, 2, 2), gain: 0.5, seed: seed + 27 }));

    // --- dead-stem thatch and part-buried grit ----------------------------
    // Where the mat is thick and where the soil is open. Mid-frequency so it
    // never becomes a landmark, and reused for both thatch layers so twigs lie
    // in the same mats the stems do.
    const mat = normalizeField(fbm2(S, S, { fx: cellsFor(340), fy: cellsFor(520), octaves: 3, gain: 0.55, seed: seed + 31 }));
    const th = { a: new Float32Array(N), r: new Float32Array(N), g: new Float32Array(N), b: new Float32Array(N), h: new Float32Array(N) };
    // Count is a density over the tile's fixed 2 m of world, not a fraction of
    // the texel count - see SPRITE_REF.
    stampFibres(S, spriteCount(130), makeRNG(seed + 401), {
      // 40-150 mm of dead stem, 3-8 mm across.
      lenMin: 0.040 * S / TILE_M, lenMax: 0.150 * S / TILE_M,
      widMin: 0.0030 * S / TILE_M, widMax: 0.0080 * S / TILE_M,
      bend: 0.30, taper: 0.55,
      // Dead grass is *paler* than the soil it lies on - that value contrast is
      // most of what makes thatch visible at all.
      palette: [rgb(0x8d8058), rgb(0xa39263), rgb(0x6f6441), rgb(0xb2a475), rgb(0x585034)],
      shade: 0.22, hMin: 0.55, hMax: 1.0,
      mask: mat, maskBias: 0.06, maskGain: 1.15,
    }, th);
    // A handful of darker, straighter twig fragments on top.
    stampFibres(S, spriteCount(4200), makeRNG(seed + 409), {
      lenMin: 0.060 * S / TILE_M, lenMax: 0.190 * S / TILE_M,
      widMin: 0.0055 * S / TILE_M, widMax: 0.0130 * S / TILE_M,
      bend: 0.10, taper: 0.25,
      palette: [rgb(0x4c4132), rgb(0x362e23), rgb(0x5e4f3a)],
      shade: 0.18, hMin: 0.7, hMax: 1.0,
      mask: mat, maskBias: 0.25, maskGain: 0.8,
    }, th);

    // Under a thick mat of dead stems you see stems, not soil structure, so the
    // crumb relief is modulated down where the mat is dense. Aggregation that is
    // uniformly strong everywhere is the other way procedural soil gives itself
    // away: real ground is aggregated in patches.
    for (let i = 0; i < N; i++) {
      const k = 1 - 0.55 * mat[i];
      wallC[i] *= k; wallF[i] *= k;
    }

    const peb = { h: new Float32Array(N), m: new Float32Array(N), t: new Float32Array(N) };
    // 6-22 mm grit, a fifth of the cells occupied, and mostly buried: a pebble
    // that stands proud of the soil by its own radius is a marble.
    stampPebbles(S, cellsFor(85), 0.20, makeRNG(seed + 555), {
      rMin: 0.0030 * S / TILE_M, rMax: 0.0110 * S / TILE_M, bury: 0.58,
    }, peb);
    // And a very few larger stones, 25-55 mm. These are what the eye actually
    // finds in a patch of soil; grit alone reads as texture rather than as
    // objects lying in the ground.
    stampPebbles(S, cellsFor(420), 0.28, makeRNG(seed + 557), {
      rMin: 0.0125 * S / TILE_M, rMax: 0.0270 * S / TILE_M, bury: 0.66,
    }, peb);

    // --- height ------------------------------------------------------------
    // Amplitudes are not "how big does this feature look" but "how steep is it".
    // A normal map shows the *gradient*, so a 30 mm feature needs three times the
    // amplitude of a 10 mm one to read at the same strength. The first draft got
    // this wrong and buried the aggregate structure completely under the grain.
    const height = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      let h = 0.5;
      h += (g1[i] - 0.5) * 0.19;          // finest grain
      h += (g1b[i] - 0.5) * 0.13;
      h += (g2[i] - 0.5) * 0.17;
      h -= wallF[i] * 0.26;               // crumb voids
      h -= wallC[i] * 0.24;               // clod joints
      h += (idF[i] - 0.5) * 0.13;         // per-fragment flat offsets
      h += (idC[i] - 0.5) * 0.20;
      h += aggC.cells[i] * 0.05;          // a *trace* of bulge, no more
      h += th.h[i] * th.a[i] * 0.30;      // thatch lies on top
      h += peb.h[i] * 0.17;
      height[i] = h;
    }
    normalizeField(height, 0.004, 0.996);

    const hBlur = boxBlurWrapFast(height, S, S, Math.max(2, Math.round(S / 40)));
    const cavity = new Float32Array(N);
    for (let i = 0; i < N; i++) cavity[i] = clamp01((hBlur[i] - height[i]) * 3.0 + 0.15);

    // --- colour ------------------------------------------------------------
    // Mid-frequency only: high-passed at a sixth of the tile, so nothing in
    // here can act as a landmark when the tile repeats. All broad drift is
    // delegated to groundMacro.
    const dryness = normalizeField(highpassField(
      normalizeField(fbm2(S, S, { fx: cellsFor(240), octaves: 3, gain: 0.55, seed: seed + 11 })),
      S, S, Math.max(2, Math.round(S / 7))));
    const humus = normalizeField(highpassField(
      normalizeField(fbm2(S, S, { fx: cellsFor(150), octaves: 3, gain: 0.5, seed: seed + 7 })),
      S, S, Math.max(2, Math.round(S / 6))));

    // Ground under a metre of pampas is dark: shaded thatch, humus and damp
    // mineral soil. The pale dry end exists but should be rare, and it is the
    // macro map's business to decide where - hence the narrow spread here.
    //
    // These are checked against *linear* reflectance, not eyeballed as hex.
    // sRGB is deceptive: #332a20 looks like a reasonable mid brown and is
    // actually 0.024 linear, which is darker than fresh asphalt and turns the
    // whole field into a void the moment the sun goes behind a cloud. Damp soil
    // is 0.05-0.09 linear, dry soil 0.13-0.20, and the mix below lands at ~0.07.
    const cWet = rgb(0x40372b);
    const cEarth = rgb(0x776856);
    const cDry = rgb(0x9c8a74);
    const cSilt = rgb(0x9e8e76);
    const cStone = rgb(0x7c766c);
    const cStoneP = rgb(0x9d988e);
    const cHumus = rgb(0x2e271c);

    const fr = new Float32Array(N), fg = new Float32Array(N), fb = new Float32Array(N);
    const orm = new Uint8Array(N * 4);

    for (let y = 0; y < S; y++) {
      const row = y * S;
      for (let x = 0; x < S; x++) {
        const i = row + x;
        const d = dither(x, y);
        const dr = dryness[i];
        const hgt = height[i];

        // Raised texels dry out first, so micro-relief drives the wet↔dry mix
        // as well as the noise field - that correlation is what makes soil
        // colour look like it belongs to the surface rather than painted on.
        const dryK = clamp01(dr * 0.72 + (hgt - 0.5) * 0.55 + 0.14);
        let r, g, b;
        if (dryK < 0.5) {
          const k = dryK * 2;
          r = lerp(cWet[0], cEarth[0], k); g = lerp(cWet[1], cEarth[1], k); b = lerp(cWet[2], cEarth[2], k);
        } else {
          const k = (dryK - 0.5) * 2;
          r = lerp(cEarth[0], cDry[0], k); g = lerp(cEarth[1], cDry[1], k); b = lerp(cEarth[2], cDry[2], k);
        }

        // Decayed leaf litter worked into the soil.
        const hk = smoothstep(0.66, 0.95, humus[i]) * 0.55;
        r = lerp(r, cHumus[0], hk); g = lerp(g, cHumus[1], hk); b = lerp(b, cHumus[2], hk);

        // Silt bloom on the highest, driest micro-relief.
        const siltK = clamp01((hgt - 0.70) * 2.4) * dr * 0.45;
        r = lerp(r, cSilt[0], siltK); g = lerp(g, cSilt[1], siltK); b = lerp(b, cSilt[2], siltK);

        // Grain modulates value only. Tinting it would add chroma noise, and
        // low chroma is the whole art direction. The aggregate `id` terms are
        // in here too: a soil crumb has its own tone, not just its own height,
        // and that per-fragment tonal mosaic is most of what makes earth read as
        // aggregated rather than as a noisy surface.
        const gk = (g1[i] - 0.5) * 0.17 + (g1b[i] - 0.5) * 0.11 + (g2[i] - 0.5) * 0.12
          + (idF[i] - 0.5) * 0.20 + (idC[i] - 0.5) * 0.26;
        r *= 1 + gk; g *= 1 + gk; b *= 1 + gk;

        // The void between crumbs is in shadow and holds the finest, dampest
        // material, so it goes dark and slightly cool rather than just darker.
        const wk = clamp01(wallF[i] * 0.42 + wallC[i] * 0.34);
        r = lerp(r, cWet[0] * 0.88, wk); g = lerp(g, cWet[1] * 0.88, wk); b = lerp(b, cWet[2] * 0.94, wk);

        // Grit. Small, mostly buried, tinted by its own per-stone hash.
        const pk = clamp01(peb.m[i]);
        if (pk > 0.002) {
          const tone = peb.t[i];
          r = lerp(r, lerp(cStone[0], cStoneP[0], tone), pk);
          g = lerp(g, lerp(cStone[1], cStoneP[1], tone), pk);
          b = lerp(b, lerp(cStone[2], cStoneP[2], tone), pk);
        }

        // Thatch sits over everything else it covers.
        const ta = th.a[i];
        if (ta > 0.003) {
          r = lerp(r, th.r[i], ta); g = lerp(g, th.g[i], ta); b = lerp(b, th.b[i], ta);
        }

        // Contact shadowing in a crack is far darker than any AO term delivers,
        // so a little of it is baked in.
        const ao = clamp01(1 - cavity[i] * 0.66);
        const aoAlb = lerp(1, ao, 0.40);
        fr[i] = r * aoAlb; fg[i] = g * aoAlb; fb[i] = b * aoAlb;

        // Dry soil is almost perfectly rough; damp soil, stone and dead stems
        // less so. Thatch is the only thing here with any sheen at all.
        let rough = 0.955 - dryK * 0.045;
        rough -= pk * 0.20;
        rough -= ta * 0.13;
        const o = i * 4;
        orm[o] = q8(ao, d);
        orm[o + 1] = q8(clamp(rough, 0.58, 0.995), d);
        orm[o + 2] = 0;
        // Cavity → wetness affinity: where rain pools and where the surface
        // goes dark and glossy first.
        orm[o + 3] = q8(clamp01(cavity[i] * 1.05 + (1 - dryK) * 0.28 - pk * 0.3), d);
      }
    }

    // Removes ~70% of the low-frequency luminance. Leaving a trace keeps the
    // surface from going clinically even; removing all of it is what makes
    // procedural ground look like felt.
    flattenLowFrequency(fr, fg, fb, S, S, Math.max(2, Math.round(S / 9)), 0.30);

    const albedo = new Uint8Array(N * 4);
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = y * S + x, o = i * 4;
        const d = dither(x, y);
        albedo[o] = q8(fr[i] / 255, d);
        albedo[o + 1] = q8(fg[i] / 255, d);
        albedo[o + 2] = q8(fb[i] / 255, d);
        albedo[o + 3] = 255;
      }
    }

    const albedoTex = this._dataTexMips(albedo, S, S, { srgb: true, name: 'groundAlbedo' });
    const pbr = this._pbrTextures(height, orm, S, S, {
      strength: 0.72, normalName: 'groundNormal', ormName: 'groundORM',
    });
    const set = {
      albedo: albedoTex, normal: pbr.normal, orm: pbr.orm,
      roughness: pbr.orm, ao: pbr.orm, height, size: S, tileMeters: TILE_M,
    };
    for (const t of [albedoTex, pbr.normal, pbr.orm]) t.userData.tileMeters = TILE_M;

    this._register('groundAlbedo', albedoTex);
    this._register('groundNormal', pbr.normal);
    this._register('groundORM', pbr.orm);
    return set;
  }

  /**
   * Moss / low undergrowth variant, for blending over soil.
   *
   * Unlike soil, moss genuinely *is* domed - a cushion is a hemisphere of
   * shoots - so a trace of the cellular cone is correct here where it was wrong
   * there. What was wrong before is that the cone was 70% of the amplitude on a
   * hard 11-cell grid, which turned a woodland floor into a tray of green
   * marbles. Cushions now get their bulk from per-cushion flat offsets and their
   * edges from the narrowed cell walls, and the *relief* is dominated by the
   * shoot layer: thousands of stamped 6-20 mm strands, because a moss cushion
   * seen from a metre away is visibly made of individual shoots and no amount
   * of fbm produces a shoot.
   */
  mossSet() {
    return this._group('moss', () => this._buildMoss(this._size('moss')));
  }

  _buildMoss(S) {
    const N = S * S;
    const seed = this.seed + 6600;
    const TILE_M = MOSS_TILE_M;
    const perM = S / TILE_M;
    const cellsFor = (mm) => clamp(Math.round((TILE_M * 1000) / mm), 2, 250);
    const finest = (mm) => Math.min(cellsFor(mm), Math.round(S / 3.2));

    // --- cushions ----------------------------------------------------------
    const cush = worley2Agg(S, S, cellsFor(90), cellsFor(90), seed + 3, { warp: 0.6 });
    const sub = worley2Agg(S, S, cellsFor(34), cellsFor(34), seed + 5, { warp: 0.5 });
    for (let i = 0; i < N; i++) {
      const a = cush.walls[i], b = sub.walls[i];
      cush.walls[i] = a * a * a;
      sub.walls[i] = b * b * b;
    }
    const cushId = boxBlurWrap(cush.id, S, S, Math.max(1, Math.round(S / 120)));
    const subId = boxBlurWrap(sub.id, S, S, Math.max(1, Math.round(S / 260)));

    // --- shoots ------------------------------------------------------------
    // One octave at LOW, two at HIGH and above. `finest(7)` is already pinned to
    // 3.2 texels a cell by the Nyquist cap, so a second octave doubled straight
    // through it - 1.6 texels a cell at 256² - and the shoot fuzz came out as
    // sparkling salt-and-pepper instead of pile.
    const fuzzF = finest(7);
    const fuzz = normalizeField(valueFbm2(S, S, {
      fx: fuzzF, octaves: octavesFor(S / fuzzF, 2, 2), gain: 0.55, seed: seed + 7,
    }));
    const sh = { a: new Float32Array(N), r: new Float32Array(N), g: new Float32Array(N), b: new Float32Array(N), h: new Float32Array(N) };
    // Shoots per tile, not per texel. This is the one that was actually shipping
    // wrong: LOW and MEDIUM bake this map at 256 and were getting a quarter of
    // HIGH's pile, which is a thinner moss, not a lower-resolution one. See
    // SPRITE_REF.
    stampFibres(S, spriteCount(26), makeRNG(seed + 701), {
      lenMin: 0.006 * perM, lenMax: 0.021 * perM,
      widMin: 0.0009 * perM, widMax: 0.0022 * perM,
      bend: 0.45, taper: 0.35,
      palette: [rgb(0x4a5a2f), rgb(0x39471f), rgb(0x5d6b38), rgb(0x2c3719), rgb(0x6b7442)],
      shade: 0.26, hMin: 0.6, hMax: 1.0,
    }, sh);

    const patch = normalizeField(fbm2(S, S, { fx: cellsFor(420), octaves: 3, seed: seed + 11 }));
    const dryTip = normalizeField(fbm2(S, S, { fx: cellsFor(90), octaves: 3, gain: 0.55, seed: seed + 13 }));

    // Where the moss is deep the cushions have grown into one another and the
    // gaps between them have closed; where it is thin, every cushion is separate
    // with bare ground showing between. Cushion boundaries of a constant depth
    // everywhere is what makes cellular moss read as a regular quilt.
    for (let i = 0; i < N; i++) {
      const k = 0.45 + 1.15 * clamp01(1 - patch[i]);
      cush.walls[i] *= k;
      sub.walls[i] *= k * 0.85;
    }

    const height = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      let h = 0.5;
      h += (fuzz[i] - 0.5) * 0.30;
      h += sh.h[i] * sh.a[i] * 0.28;
      h += (cushId[i] - 0.5) * 0.18;
      h += (subId[i] - 0.5) * 0.10;
      h += (cush.cells[i] - 0.5) * 0.16;      // the honest cushion dome
      h -= cush.walls[i] * 0.20;
      h -= sub.walls[i] * 0.11;
      height[i] = h;
    }
    normalizeField(height, 0.008, 0.992);

    const hBlur = boxBlurWrapFast(height, S, S, Math.max(2, Math.round(S / 44)));
    const cavity = new Float32Array(N);
    for (let i = 0; i < N; i++) cavity[i] = clamp01((hBlur[i] - height[i]) * 2.6 + 0.18);

    // Low-saturation greens only. Anything brighter than #7a8a5a starts to read
    // as a golf course rather than as woodland floor.
    // Checked in linear, like the soil palette: this set means ~0.075 linear
    // reflectance, which is where real moss and shaded foliage sit.
    const cDeep = rgb(0x333f26);
    const cMid = rgb(0x58643c);
    const cLit = rgb(0x808c58);
    const cDryG = rgb(0x999667);
    const cSoil = rgb(0x393022);

    const fr = new Float32Array(N), fg = new Float32Array(N), fb = new Float32Array(N);
    const alpha = new Float32Array(N);
    const orm = new Uint8Array(N * 4);

    for (let y = 0; y < S; y++) {
      const row = y * S;
      for (let x = 0; x < S; x++) {
        const i = row + x;
        const d = dither(x, y);
        const hgt = height[i];

        // Vertical gradient through the cushion: dark at the base, lit at the
        // tips. This is the single detail that makes moss look like moss.
        const lit = clamp01((hgt - 0.28) * 1.55);
        let r, g, b;
        if (lit < 0.5) {
          const k = lit * 2;
          r = lerp(cDeep[0], cMid[0], k); g = lerp(cDeep[1], cMid[1], k); b = lerp(cDeep[2], cMid[2], k);
        } else {
          const k = (lit - 0.5) * 2;
          r = lerp(cMid[0], cLit[0], k); g = lerp(cMid[1], cLit[1], k); b = lerp(cMid[2], cLit[2], k);
        }

        // Individual shoots carry their own colour, over the cushion tone.
        const sa = sh.a[i];
        if (sa > 0.003) {
          r = lerp(r, sh.r[i], sa * 0.85); g = lerp(g, sh.g[i], sa * 0.85); b = lerp(b, sh.b[i], sa * 0.85);
        }

        // Sun-bleached tips on the exposed crowns.
        const dk = smoothstep(0.55, 0.9, dryTip[i]) * clamp01((hgt - 0.62) * 2.4) * 0.5;
        r = lerp(r, cDryG[0], dk); g = lerp(g, cDryG[1], dk); b = lerp(b, cDryG[2], dk);

        // Soil showing through the thin patches.
        const bare = clamp01(0.40 - patch[i]) * 1.7 * clamp01(1 - hgt * 1.2);
        r = lerp(r, cSoil[0], bare); g = lerp(g, cSoil[1], bare); b = lerp(b, cSoil[2], bare);

        const fz = 0.90 + 0.20 * fuzz[i];
        r *= fz; g *= fz; b *= fz;

        // Moss self-shadows hard - it is a deep pile, and light does not reach
        // the base of a cushion at all.
        const ao = clamp01(1 - cavity[i] * 0.82);
        const aoAlb = lerp(1, ao, 0.65);
        fr[i] = r * aoAlb; fg[i] = g * aoAlb; fb[i] = b * aoAlb;

        // Alpha is a blend mask: moss thins at the cushion edges, so terrain can
        // dissolve it into soil instead of drawing a hard boundary.
        alpha[i] = clamp01(smoothstep(0.16, 0.55, patch[i]) * 0.68 + hgt * 0.52 - bare * 0.6);

        const o = i * 4;
        orm[o] = q8(ao, d);
        orm[o + 1] = q8(clamp(0.94 - dk * 0.06 - sa * 0.05, 0.72, 0.995), d);
        orm[o + 2] = 0;
        orm[o + 3] = q8(clamp01(cavity[i] * 1.15), d);
      }
    }

    // Moss has real large-scale variation, so only two thirds of the low
    // frequency is removed - enough that no cushion group becomes a landmark.
    flattenLowFrequency(fr, fg, fb, S, S, Math.max(2, Math.round(S / 8)), 0.34);

    const albedo = new Uint8Array(N * 4);
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = y * S + x, o = i * 4;
        const d = dither(x, y);
        albedo[o] = q8(fr[i] / 255, d);
        albedo[o + 1] = q8(fg[i] / 255, d);
        albedo[o + 2] = q8(fb[i] / 255, d);
        albedo[o + 3] = q8(alpha[i], d);
      }
    }

    const albedoTex = this._dataTexMips(albedo, S, S, { srgb: true, name: 'mossAlbedo' });
    const pbr = this._pbrTextures(height, orm, S, S, {
      strength: 1.05, normalName: 'mossNormal', ormName: 'mossORM',
    });
    for (const t of [albedoTex, pbr.normal, pbr.orm]) t.userData.tileMeters = TILE_M;

    this._register('mossAlbedo', albedoTex);
    this._register('mossNormal', pbr.normal);
    this._register('mossORM', pbr.orm);

    return {
      albedo: albedoTex, normal: pbr.normal, orm: pbr.orm,
      roughness: pbr.orm, ao: pbr.orm, height, size: S, tileMeters: TILE_M,
    };
  }

  /**
   * Fallen-petal litter layer. RGB is petal colour, A is coverage, so the
   * ground shader can lerp it over soil or moss with a single mix().
   * The companion normal gives the petals just enough relief to catch a rim of
   * light at low sun - flat petals look like a decal.
   *
   * Two things were wrong here and both are scale errors, which is the usual
   * story with procedural litter. The petals were 11 cm long (a sakura petal is
   * 15 mm) and there were 150 of them per tile, so the layer read as a scatter
   * of pale ellipses rather than as drifted litter. And compositing took the
   * *maximum* coverage per texel, so a petal drawn later could not lie on top of
   * one drawn earlier: the layer had no depth. Both are fixed below, along with
   * a contact-shadow pass, because a petal lying on other petals darkens what is
   * under its edge and that shadow is most of what sells the depth.
   */
  petalLitter() {
    return this._group('petalLitter', () => this._buildPetalLitter(this._size('litter')));
  }

  _buildPetalLitter(S) {
    const N = S * S;
    const seed = this.seed + 7100;
    const rand = makeRNG(seed);
    const TILE_M = LITTER_TILE_M;
    const perM = S / TILE_M;

    const cover = new Float32Array(N);
    const height = new Float32Array(N);
    const colR = new Float32Array(N);
    const colG = new Float32Array(N);
    const colB = new Float32Array(N);

    // Petals do not fall evenly: they drift. A mid-frequency mask concentrates
    // them into banks and leaves swept-clean ground between. High-passed, so the
    // banks cannot become a landmark when the tile repeats.
    const drift = normalizeField(highpassField(
      normalizeField(fbm2(S, S, { fx: 7, fy: 4, octaves: 3, gain: 0.55, seed: seed + 21 })),
      S, S, Math.max(2, Math.round(S / 4)), 0.25));

    // Fresh, aged and browning. All pale and desaturated - a saturated pink
    // litter layer is the fastest way to make the whole scene look like candy.
    const cFresh = rgb(0xe6d2d4);
    const cPink = rgb(0xd8b9bf);
    const cAged = rgb(0xc9b9a9);
    const cBrown = rgb(0x9d8b7a);

    // Sakura petal: ~15 mm long, ~12 mm across. Everything else follows.
    const L0 = 0.017 * perM;
    const count = Math.round(N / Math.max(24, L0 * L0 * 0.6));

    for (let p = 0; p < count; p++) {
      const cx = rand() * S;
      const cy = rand() * S;
      // Reject against the drift mask so density follows the banks.
      const di = (Math.floor(cy) % S) * S + (Math.floor(cx) % S);
      const keep = rand();
      const ang = rand() * TAU;
      const lenR = rand(), widR = rand(), ageR = rand(), flipR = rand(), skewR = rand();
      // Swept-clean ground between the banks: acceptance has to reach *zero*
      // somewhere or the layer is a uniform confetti of petals, which is what
      // a floor of 0.14 was quietly guaranteeing.
      if (keep > drift[di] * 1.5 - 0.24) continue;

      const ca = Math.cos(ang), sa = Math.sin(ang);
      const len = L0 * (0.80 + 0.42 * lenR);
      const wid = len * 0.62 * (0.86 + 0.28 * widR);
      const flip = flipR < 0.5 ? 1 : -1;

      let r, g, b;
      // Two thirds fresh: this is litter under a tree in bloom, not a compost
      // heap, and the aged end is there for variety rather than as the average.
      if (ageR < 0.66) {
        const k = ageR / 0.66;
        r = lerp(cFresh[0], cPink[0], k); g = lerp(cFresh[1], cPink[1], k); b = lerp(cFresh[2], cPink[2], k);
      } else {
        const k = (ageR - 0.66) / 0.34;
        r = lerp(cAged[0], cBrown[0], k * k); g = lerp(cAged[1], cBrown[1], k * k); b = lerp(cAged[2], cBrown[2], k * k);
      }
      // A petal that has been rained on and dried goes blotchy rather than
      // evenly beige; a per-petal tone jitter is the cheapest way to say so.
      const tone = 0.90 + 0.18 * skewR;
      r *= tone; g *= tone; b *= tone;

      // Oriented bounding box rather than the square of the petal's length:
      // at these counts the difference is a third of the bake.
      const reach = wid * 0.9 + 2;
      const ex = Math.ceil(Math.abs(len * 0.6 * ca) + Math.abs(reach * sa)) + 1;
      const ey = Math.ceil(Math.abs(len * 0.6 * sa) + Math.abs(reach * ca)) + 1;
      const px = Math.round(cx), py = Math.round(cy);
      const edge = 1.1;                      // ~1 texel alpha ramp
      for (let dy = -ey; dy <= ey; dy++) {
        const yy = ((py + dy) % S + S) % S;
        const row = yy * S;
        for (let dx = -ex; dx <= ex; dx++) {
          // Rotate into petal space: +lx runs base→tip.
          const lx = dx * ca + dy * sa;
          const ly = (-dx * sa + dy * ca) * flip;
          const t = (lx + len * 0.5) / len;
          if (t < -0.05 || t > 1.05) continue;
          const dist = petalDist(t, ly, len, wid, 0.15, wid * 0.3);
          const a = clamp01(dist / edge + 0.5);
          if (a <= 0.004) continue;
          const xx = ((px + dx) % S + S) % S;
          const k = row + xx;
          // Painter's order: a petal that fell later lies on top of one that
          // fell earlier. Taking the max, as this used to, gives a layer with
          // no stacking at all.
          const wt = petalWidth(t, wid);
          const across = wt > 1e-4 ? clamp(ly / wt, -1, 1) : 0;
          // A petal lying on the ground curls: the edges lift, the middle sits
          // down. Cross-section is a shallow U, not a flat plate.
          const curl = across * across * 0.55 + 0.22;
          const shade = 0.86 + 0.26 * curl;
          const vein = Math.exp(-(across * across) * 26) * 0.10;
          // Straight-alpha `over`. `lerp(dst, src, a)` onto a zero-initialised
          // buffer is a premultiply by accident: it left every texel of the
          // one-texel alpha ramp holding `a * petalColour`, i.e. near black,
          // and the consumer's own `mix(ground, litter.rgb, litter.a)` then
          // drew that as a dark outline around every fallen petal. Measured
          // before the fix: alpha-0.15 texels had luminance 27 where the petal
          // body reads 185. Normalising by the accumulated coverage is the
          // exact unpremultiplied composite and is also what the alpha-weighted
          // mip chain assumes it is being handed.
          const na = a + cover[k] * (1 - a);
          const wS = na > 1e-6 ? a / na : 0;
          colR[k] = lerp(colR[k], r * (shade + vein), wS);
          colG[k] = lerp(colG[k], g * (shade + vein), wS);
          colB[k] = lerp(colB[k], b * (shade + vein), wS);
          // Height keeps the plain alpha ramp on purpose: it feathers from
          // ground level into the petal across the outline, which is what stops
          // the normal map cutting a cliff at every silhouette.
          height[k] = lerp(height[k], 0.30 + curl * 0.70, a);
          cover[k] = na;
        }
      }
    }

    // A few reddish pedicel and calyx fragments. They are what a real drift of
    // cherry litter has in it, and they break the uniform pinkness better than
    // any amount of hue jitter on the petals themselves.
    {
      const bits = { a: new Float32Array(N), r: new Float32Array(N), g: new Float32Array(N), b: new Float32Array(N), h: new Float32Array(N) };
      stampFibres(S, spriteCount(3400), makeRNG(seed + 77), {
        lenMin: 0.006 * perM, lenMax: 0.020 * perM,
        widMin: 0.0008 * perM, widMax: 0.0020 * perM,
        bend: 0.35, taper: 0.4,
        palette: [rgb(0x7d5148), rgb(0x8f6b56), rgb(0x5d4038), rgb(0x6b5a44)],
        shade: 0.2, hMin: 0.6, hMax: 1.0,
      }, bits);
      for (let i = 0; i < N; i++) {
        const a = bits.a[i];
        if (a <= 0.003) continue;
        // Same unpremultiplied `over` as the petals: a pedicel lying on swept
        // ground has cover 0 underneath it, so weighting by `a` alone would
        // leave its own edge texels black.
        const na = a + cover[i] * (1 - a);
        const wS = na > 1e-6 ? a / na : 0;
        colR[i] = lerp(colR[i], bits.r[i], wS);
        colG[i] = lerp(colG[i], bits.g[i], wS);
        colB[i] = lerp(colB[i], bits.b[i], wS);
        height[i] = lerp(height[i], 0.30 + bits.h[i] * 0.55, a);
        cover[i] = na;
      }
    }

    // Soften the height so the normal map does not produce a hard cliff at
    // every petal outline; real petals feather into the ground.
    const hSoft = boxBlurWrap(height, S, S, 1);
    // Contact shadow: a texel that sits below its own neighbourhood is under
    // the edge of something. Without this the layer looks printed on.
    const hLow = boxBlurWrapFast(hSoft, S, S, Math.max(2, Math.round(S / 40)));

    const data = new Uint8Array(N * 4);
    for (let y = 0; y < S; y++) {
      const row = y * S;
      for (let x = 0; x < S; x++) {
        const i = row + x;
        const o = i * 4;
        const d = dither(x, y);
        const a = cover[i];
        const shadow = clamp01(1 - clamp01((hLow[i] - hSoft[i]) * 2.2) * 0.34);
        // RGB is defined everywhere (bled outward from the nearest petal tone)
        // so mip filtering can never drag black into a petal edge.
        const rr = (a > 0.002 ? colR[i] : cAged[0] * 0.8) * shadow;
        const gg = (a > 0.002 ? colG[i] : cAged[1] * 0.8) * shadow;
        const bb = (a > 0.002 ? colB[i] : cAged[2] * 0.8) * shadow;
        data[o] = q8(rr / 255, d);
        data[o + 1] = q8(gg / 255, d);
        data[o + 2] = q8(bb / 255, d);
        data[o + 3] = q8(a, d);
      }
    }

    const albedoTex = this._dataTexMips(data, S, S, { srgb: true, name: 'petalLitter', alphaWeighted: true });
    const chain = normalMipChain(hSoft, S, S, { strength: 0.85 });
    const normalTex = new THREE.DataTexture(chain.levels[0].data, S, S, THREE.RGBAFormat, THREE.UnsignedByteType);
    this._apply(normalTex, { srgb: false, repeat: true, mipmaps: true, anisotropy: this.maxAnisotropy });
    normalTex.mipmaps = chain.levels;
    normalTex.generateMipmaps = false;
    normalTex.name = 'petalLitterNormal';
    normalTex.needsUpdate = true;
    for (const t of [albedoTex, normalTex]) t.userData.tileMeters = TILE_M;
    this._register('petalLitter', albedoTex);
    this._register('petalLitterNormal', normalTex);
    return { albedo: albedoTex, normal: normalTex, size: S, tileMeters: TILE_M };
  }

  /**
   * Macro variation - the other half of the anti-repetition strategy.
   *
   * The detail maps have had their low frequencies deliberately stripped out
   * (see `highpassField` and `flattenLowFrequency`), which is what stops the eye
   * finding a landmark inside a 2 m tile. This map puts the low frequencies back
   * at a scale where repetition is beyond the far plane. Tile it at
   * `userData.tileMeters` (96 m by default) and modulate the detail maps with
   * it; the ratio to the detail tiling is about 1:48, and every octave in here
   * is chosen so that *no* single frequency dominates - a macro map with one
   * strong 3-cycle blob just moves the landmark problem out to 96 m instead of
   * solving it.
   *
   *   R = broad value drift (0.5 = neutral, multiply around it)
   *   G = moss weight      B = wetness / low ground     A = petal-litter weight
   */
  groundMacro() {
    return this.get('groundMacro', () => {
      const S = this._size('macro');
      const seed = this.seed + 7700;

      // Five octaves at gain 0.62 rather than four at 0.52: a slow rolloff
      // spreads the energy across 3-48 cycles instead of piling it into the
      // first two, so the field has structure at every zoom the player can
      // reach rather than one continent-sized blotch.
      const value = normalizeField(fbm2(S, S, { fx: 3, octaves: 5, gain: 0.62, lacunarity: 2.07, seed: seed + 1 }));
      // Vegetation patch boundaries are lobed, not blobby: a patch of moss ends
      // where the ground stops being damp, and that boundary follows drainage.
      // Warping the moss field by a second field produces exactly that.
      const mossW = normalizeField(fbm2(S, S, { fx: 4, octaves: 4, gain: 0.58, seed: seed + 2 }));
      const mossClump = worley2Scaled(S, S, 9, 9, seed + 3, 0, 0.92, 12, 0.6);
      // Wetness follows the *valleys* of a ridged field, which is where water
      // would actually collect, rather than an arbitrary noise threshold.
      const low = normalizeField(fbm2(S, S, { fx: 5, octaves: 4, gain: 0.5, kind: 2, seed: seed + 4 }));
      const lowFine = normalizeField(fbm2(S, S, { fx: 17, octaves: 3, gain: 0.5, kind: 2, seed: seed + 6 }));
      // Litter drifts downwind, so its field is stretched along one axis, and it
      // banks up in the hollows the wetness field already found.
      const litter = normalizeField(fbm2(S, S, { fx: 7, fy: 3, octaves: 4, gain: 0.58, seed: seed + 5 }));

      const data = new Uint8Array(S * S * 4);
      for (let y = 0; y < S; y++) {
        for (let x = 0; x < S; x++) {
          const i = y * S + x;
          const o = i * 4;
          const d = dither(x, y);
          // Compressed toward the middle: this channel multiplies the ground
          // albedo, and a macro map that swings the full 0..1 turns a field into
          // a patchwork quilt. ±22% is as far as real ground drifts.
          data[o] = q8(clamp01(0.5 + (value[i] - 0.5) * 0.88), d);
          const damp = clamp01(smoothstep(0.42, 0.06, low[i]) * 0.7 + smoothstep(0.4, 0.08, lowFine[i]) * 0.45);
          data[o + 1] = q8(clamp01(smoothstep(0.40, 0.84, mossW[i] * 0.55 + mossClump[i] * 0.42 + damp * 0.35)), d);
          data[o + 2] = q8(damp, d);
          data[o + 3] = q8(clamp01(smoothstep(0.46, 0.90, litter[i] * 0.85 + damp * 0.3)), d);
        }
      }
      const t = this._dataTexMips(data, S, S, { srgb: false, name: 'groundMacro' });
      t.userData.tileMeters = 96;
      return t;
    });
  }

  // -------------------------------------------------------------------------
  // LEAF 3b - the terrain surface suite (world/terrain.js's three ground maps)
  // -------------------------------------------------------------------------

  /**
   * Detail / normal+AO / macro maps for world/terrain.js's splat shader.
   *
   * Built as one group because all three share their source fields, and because
   * terrain.js always asks for all three: computing the aggregate structure
   * three times would triple the most expensive part of the bake for nothing.
   *
   * CHANNEL CONTRACT - fixed by terrain.js's fragment shader, not by taste:
   *
   *   detail    R  bare-soil layer height. Also drives ±20% of the ground's
   *                albedo value and ±6% of its roughness.
   *             G  grass-thatch layer height.
   *             B  moss layer height. Also part of the albedo value term and
   *                the snow mask.
   *             A  fallen-petal layer height, sparse.
   *                All four are heights for `terrainHeightBlend(w, h, 0.22)`,
   *                which forms `w + h*0.5` and admits any layer within 0.22 of
   *                the leader. A channel's *spread* therefore decides how
   *                decisively it wins, which is why every one is retargeted to
   *                an explicit standard deviation below.
   *                Sampled twice: at 2 m (near) and at 32 m (mid, quarter-
   *                turned), so its high frequencies are also its mid-scale
   *                structure and there must be no scale at which it is boring.
   *
   *   normalAO  RGB tangent-space normal, +Y green up the V axis, where U runs
   *                along world +X and V along world +Z.
   *             A  cavity AO, applied as mix(1, a, 0.75).
   *                Sampled at 2 m only, and faded out entirely past
   *                55-210 m depending on tier.
   *
   *   macro     R  dryness   G  lushness   B  moss affinity   A  mottle.
   *                Sampled at 1024 m (region) and 128 m (patch). A and B also
   *                drive the 3.4 m domain warp that breaks the near tile's
   *                repeat, so their gradient is not free - see the note in
   *                `_buildTerrainMacro`.
   *
   * WHAT THIS REPLACES, because it is the instructive part: terrain.js's local
   * fallback bake built its micro-relief height from `fbm*0.55 + invWorley(24
   * cells)*0.30 + invWorley(12 cells)*0.22`. Inverted F1 is a radial cone
   * centred on every feature point, so 52% of the relief was identical round
   * domes on a 8 cm and a 17 cm grid - and the cavity-AO channel derived from
   * the same field then put a bright disc on top of each one. That is the
   * regular carpet of soft pale blobs visible through the grass in every capture
   * of this build, and it survives any change to the tile size because the
   * problem is the feature *shape*, not its scale.
   *
   * The inverted-F1 cone appears in exactly one place below - a 0.16 trace in
   * the moss channel, modulated by where the mat is thick, because a moss
   * cushion genuinely is a dome and a soil crumb is not. It is nowhere in the
   * micro-relief height at all. Granular structure there comes from
   * `worley2Agg`: a mosaic of flat-topped fragments (`id`) separated by narrowed
   * voids (`walls`). Grain carries the largest single share of the relief (~55%
   * of the RMS gradient), strands are stamped as strands and stones as
   * part-buried stones, and the whole micro-relief is *solved* to an RMS slope
   * of TERRAIN_SLOPE_RMS_DEG rather than being handed an amplitude.
   *
   * VERIFIED, on the finished bytes rather than by construction. Numbers below
   * are from the HIGH bake (detail/normalAO 256², macro 128², seed 20240401);
   * re-derive them by setting `textures.verbose = true` before first use, or by
   * re-running the same tests against `tex.image.data`.
   *
   *   Seamlessness. For each channel and each axis, the RMS step across the wrap
   *   pair divided by the RMS step between interior neighbours - 1.0 is a
   *   perfectly periodic field, and a real seam is several times the interior:
   *     terrainDetail    0.92 .. 1.12   terrainNormalAO  0.95 .. 1.08
   *     terrainMacro     0.57 .. 1.63 (128 wrap pairs; the outlier is G, worth
   *                      5/255 on a channel sampled at 8 m a texel)
   *   The mip chain wraps too (levels 1/3/5 measured at 0.86 .. 1.20), which
   *   matters because the 32 m mid tap lives several levels up.
   *
   *   Domain-warp budget. The B and A gradient solve converges in 2 passes to
   *   0.124 against terrain.js's calibrated 0.126, i.e. a warp Jacobian inside
   *   its published det 0.62..1.40 row rather than its "blurs" or "folds" rows.
   *
   *   Distributions. detail R 0.480/0.179  G 0.454/0.169  B 0.471/0.205,
   *   A mean 0.229 max 0.687 with 25.2% of texels over 0.5 - the sparse-mask
   *   distribution terrain.js's petal-litter weight of 1.5 is calibrated
   *   against, and it is resolution-independent (see `perSqM`).
   *
   *   Micro-relief. gradRMS 0.212, encode strength 0.458, i.e. exactly the
   *   TERRAIN_SLOPE_RMS_DEG the solve asks for.
   *
   *   Colour space. All three maps are NoColorSpace. They are data - splat
   *   heights, a tangent-space normal, cavity AO, region masks - and every one
   *   of them would be silently gamma-decoded on upload if it were tagged sRGB.
   *
   *   Cost. 290 ms for the whole suite on a laptop CPU (grain 34 · aggregate 65
   *   · thatch 29 · pebbles 1 · moss 58 · petals 24 · height+AO 11 · normal 13 ·
   *   channels 15 · upload 21 · macro 27). It is the second most expensive bake
   *   in the file and it is baked once, lazily, behind the loading screen.
   */
  terrainSurface() {
    return this._group('terrainSurface', () => this._buildTerrainSurface(this._size('terrain')));
  }

  _buildTerrainSurface(S) {
    const N = S * S;
    const seed = this.seed + 8300;
    const TILE_M = TERRAIN_TILE_M;
    /** Lattice cells across the tile for a feature of the given size in mm. */
    const cellsFor = (mm) => clamp(Math.round((TILE_M * 1000) / mm), 2, 250);
    /** As `cellsFor`, but never finer than ~3.2 texels a cell: past that the
     *  base mip aliases against its own lattice and the ground sparkles. */
    const finest = (mm) => Math.min(cellsFor(mm), Math.max(2, Math.round(S / 3.2)));
    /** Texels per metre - every sprite size below is authored in metres. */
    const perM = S / TILE_M;
    /**
     * Sprite counts are a DENSITY (per square metre of ground), not a fraction
     * of the texel count.
     *
     * This was a real bug and it only showed at ULTRA, which is exactly why it
     * survived: the counts were `N / 70`, `N / 16` and `N / 7`, i.e.
     * proportional to S², while every sprite's *area* is also proportional to S²
     * because its size is authored in metres. Coverage therefore grew with the
     * square of the resolution. Measured on the petal channel, which is a sparse
     * mask and so is not retargeted afterwards to hide it: mean 0.229 and 25.2%
     * coverage at 256², mean 0.496 and 74.8% at ULTRA's 512². ULTRA laid down
     * three times the fallen petals of HIGH, saturated the channel, and broke
     * the distribution terrain.js's litter weight of 1.5 is calibrated against.
     * The thatch layers had the same defect, hidden by `retargetField` putting
     * the mean and spread back afterwards - but a mat four times as dense is a
     * different surface, not a rescaled one.
     *
     * The densities below are the ones the 256² bake actually ran at, so HIGH is
     * byte-for-byte unchanged and only ULTRA moves - onto the same surface at
     * twice the texel density, which is what a quality tier is supposed to buy.
     */
    const tileM2 = TILE_M * TILE_M;
    const perSqM = (n) => Math.max(1, Math.round(n * tileM2));

    // Phase timing, so the cost of this bake is a measurement rather than a
    // guess when someone next has to defend it against the loading budget.
    // Zero overhead when `verbose` is off: one closure and one branch per phase.
    const clock = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
    const phases = this.verbose ? [] : null;
    let tPhase = phases ? clock() : 0;
    const mark = (label) => {
      if (!phases) return;
      const now = clock();
      phases.push(`${label} ${(now - tPhase).toFixed(1)}`);
      tPhase = now;
    };

    // --- grain: the dominant relief ----------------------------------------
    // Two different *kinds* of noise at neighbouring, non-harmonic frequencies.
    // Value noise alone leaves a faint square weave at three texels a cell (its
    // lattice is axis-aligned and it has no gradient to hide it); gradient noise
    // alone is identically zero at every lattice point and leaves that zero set
    // as a grid. Superimposed, each conceals the other's signature.
    //
    // `finest(11)` asks for 11 mm and gets what the resolution can carry: the
    // Nyquist cap binds here, so the realised finest grain is 25 mm at 256 and
    // 12.5 mm at ULTRA's 512. That is the honest number and it is why ULTRA
    // looks meaningfully different up close rather than merely sharper.
    const fFine = finest(11);
    const fMid = Math.max(3, Math.round(fFine * 0.46));
    const fCoarse = cellsFor(58);
    const grainA = normalizeField(valueFbm2(S, S, { fx: fFine, octaves: 1, seed: seed + 21 }));
    const grainB = normalizeField(fbm2(S, S, { fx: fMid, octaves: 2, gain: 0.5, spc: 6, seed: seed + 23 }));
    const grainC = normalizeField(valueFbm2(S, S, { fx: fCoarse, octaves: 2, gain: 0.52, seed: seed + 27 }));
    mark('grain');

    // --- aggregate structure -----------------------------------------------
    // Soil crumb is a mosaic of flat-ish fragments at slightly different heights
    // separated by thin voids - never a field of domes. `id` is the per-fragment
    // flat value, `walls` the F2-F1 crease; `cells` (the cone) is not used here
    // at all. Both layers are domain-warped, because one feature point per cell
    // of a square grid is readable as a square grid however hard it is jittered.
    const aggC = worley2Agg(S, S, cellsFor(95), cellsFor(95), seed + 3, { warp: 0.66 });
    const aggF = worley2Agg(S, S, cellsFor(38), cellsFor(38), seed + 9, { warp: 0.55 });
    const wallC = aggC.walls, wallF = aggF.walls;
    // Narrow both creases hard. F2-F1 comes back as a broad linear ramp, and the
    // void between two crumbs is a slot, not a valley - an unnarrowed wall mask
    // is what makes cellular soil read as a paved mosaic.
    for (let i = 0; i < N; i++) {
      const a = wallC[i], b = wallF[i];
      wallC[i] = a * a * a;
      wallF[i] = b * b * b * b;
    }
    // Soften the plateau steps by a texel or two. The step sits on the crease
    // between two fragments, which is where a discontinuity belongs, but a
    // one-texel cliff makes a razor normal that sparkles under a low sun.
    const idC = boxBlurWrap(aggC.id, S, S, Math.max(1, Math.round(S / 200)));
    const idF = boxBlurWrap(aggF.id, S, S, Math.max(1, Math.round(S / 320)));
    mark('aggregate');

    // Aggregation is patchy in reality - soil is crumbed where roots and worms
    // worked it and smooth where they did not - and structure that is uniformly
    // strong everywhere is the other way procedural ground gives itself away.
    // Deliberately mid-frequency (18-25 cm): any coarser and it becomes the
    // landmark the eye locks onto to find the 2 m repeat.
    const mat = normalizeField(fbm2(S, S, {
      fx: cellsFor(180), fy: cellsFor(260), octaves: 3, gain: 0.55, seed: seed + 31,
    }));
    for (let i = 0; i < N; i++) {
      const k = 1 - 0.5 * mat[i];
      wallC[i] *= k; wallF[i] *= k;
    }

    // --- strands and stones -------------------------------------------------
    // No noise field produces a strand: fbm gives blobs, Worley gives cells and
    // anisotropic fbm gives an axis-aligned corduroy. Dead stems under a metre
    // of pampas are the single most characteristic thing about this ground, so
    // they are rasterised as sprites, each inside its own oriented box.
    //
    // No colour buffers at all: this suite stores layer *heights* and
    // world/terrain.js supplies the palette - uColEarth/uColEarthDry,
    // uColThatch/uColThatchDry, uColMoss, uColPetal; there has been no
    // "uColGrass" since the ground stopped drawing the sward - so every byte of
    // strand colour computed here was written and never read. `stampFibres`
    // still draws the palette index and the tone jitter, so `a` and `h` are
    // byte-identical to the version that kept them.
    const thatchPal = [rgb(0x8d8058), rgb(0xa39263), rgb(0x6f6441)];
    const th = { a: new Float32Array(N), h: new Float32Array(N) };
    stampFibres(S, perSqM(234), makeRNG(seed + 401), {
      // 35-140 mm of broken dead stem, 3.5-9 mm across.
      lenMin: 0.035 * perM, lenMax: 0.140 * perM,
      widMin: 0.0035 * perM, widMax: 0.0090 * perM,
      bend: 0.34, taper: 0.55, palette: thatchPal, shade: 0.22,
      hMin: 0.55, hMax: 1.0, mask: mat, maskBias: 0.10, maskGain: 1.10,
    }, th);
    // A second, much finer and much denser layer: the matted fragment litter
    // that fills the gaps between whole stems. This is what carries the fibrous
    // read at the 32 m mid tap, where it is magnified sixteen times.
    const thF = { a: new Float32Array(N), h: new Float32Array(N) };
    stampFibres(S, perSqM(1024), makeRNG(seed + 409), {
      lenMin: 0.012 * perM, lenMax: 0.048 * perM,
      widMin: 0.0020 * perM, widMax: 0.0050 * perM,
      bend: 0.45, taper: 0.35, palette: thatchPal, shade: 0.20,
      hMin: 0.35, hMax: 0.85, mask: mat, maskBias: 0.32, maskGain: 0.70,
    }, thF);
    mark('thatch');

    // Part-buried grit. `bury` is what stops scattered stones reading as a floor
    // of marbles: a stone in soil shows a shallow cap, not a hemisphere - which
    // is the same mistake, in sprite form, that the inverted-Worley domes were.
    //
    // The sizes are a deliberate compromise with the texel grid, and worth being
    // explicit about because it is the sort of thing that gets "fixed" back to
    // wrong. Real meadow grit is 3-10 mm, and 7.8 mm is one texel here: a
    // physically honest pebble is *half a texel* and contributes nothing but a
    // little roughness. So the fine layer stays physically sized (it does exactly
    // that useful nothing, and comes into its own at ULTRA's 3.9 mm texel) and
    // the layer that has to actually read as stones lying in the ground is
    // 16-44 mm gravel, at five to a square metre. That is a real thing to find in
    // a meadow, and it is 2-6 texels across, which is the smallest object that
    // survives a mip and an anisotropic tap.
    const peb = { h: new Float32Array(N), m: new Float32Array(N), t: new Float32Array(N) };
    stampPebbles(S, cellsFor(60), 0.24, makeRNG(seed + 555), {
      rMin: 0.0035 * perM, rMax: 0.0080 * perM, bury: 0.60,
    }, peb);
    stampPebbles(S, cellsFor(240), 0.30, makeRNG(seed + 557), {
      rMin: 0.0080 * perM, rMax: 0.0220 * perM, bury: 0.68,
    }, peb);
    mark('pebbles');

    // --- moss micro-structure ----------------------------------------------
    // WHERE moss grows is the macro map's business; this channel only says what
    // moss *is* at centimetre scale. Keeping it fine is not a shortcut, it is
    // the division of labour that stops the near tile carrying a 40 cm patch
    // that then repeats every 2 m across the whole field.
    //
    // A moss cushion genuinely is domed, unlike a soil crumb, so a trace of the
    // cellular cone is correct here where it would be wrong above - a trace, at
    // a tenth of the amplitude the fallback bake gave it.
    //
    // 85 mm cushions, not 55: at 55 every cell got a crisp complete outline at
    // four texels a cell and the channel came out as a tray of uniform cobbles,
    // which is the same failure as the domes wearing a different shape. Bigger
    // cells, a softened mosaic and a fine grain that dominates *within* each
    // cushion is what turns cobbles back into moss.
    const mossFrag = worley2Agg(S, S, cellsFor(85), cellsFor(85), seed + 63, { warp: 0.72 });
    const mossWall = mossFrag.walls;
    for (let i = 0; i < N; i++) { const w = mossWall[i]; mossWall[i] = w * w * w; }
    // Soften the per-cushion plateaus. A cushion has no edge - it thins out.
    const mossSoft = boxBlurWrap(mossFrag.id, S, S, Math.max(1, Math.round(S / 90)));
    // And where the mat is cushiony at all. Moss is not uniformly deep: it is
    // thick in the damp hollows and a thin film elsewhere, and structure that is
    // equally strong across the whole tile is the tell.
    const mossPatch = normalizeField(fbm2(S, S, {
      fx: cellsFor(150), fy: cellsFor(210), octaves: 3, gain: 0.55, seed: seed + 67,
    }));
    // One octave, not two. `valueFbm2` doubles per octave, so a second octave on
    // top of 0.79*fFine lands at 2.0 texels a cell - under Nyquist for the base
    // mip. It came out as salt-and-pepper that would sparkle through the albedo
    // micro term and wash out entirely one mip later, which is the worst of both.
    const mossGrain = normalizeField(valueFbm2(S, S, {
      fx: Math.max(3, Math.round(fFine * 0.62)), octaves: 1, seed: seed + 65,
    }));
    mark('moss');

    // --- fallen petals ------------------------------------------------------
    // Litter blows into drifts and catches in hollows; a uniform scatter is
    // instantly readable as procedural. 20 cm drifts, fine enough not to become
    // the tile's landmark.
    const drift = normalizeField(fbm2(S, S, {
      fx: cellsFor(210), fy: cellsFor(300), octaves: 3, gain: 0.58, seed: seed + 71,
    }));
    const flake = { h: new Float32Array(N), a: new Float32Array(N) };
    // Real fallen sakura petals are 12-18 mm; at 7.8 mm a texel that is under
    // three texels and no silhouette survives, so these are stamped at 18-36 mm
    // - within a factor of two of life and legible at ULTRA's 3.9 mm texel.
    // 2340 candidates per square metre gives ~25% coverage after drift
    // rejection, which is the distribution terrain.js's litter weight of 1.5 was
    // calibrated against; the measured coverage is printed at the end of this
    // function under `verbose` and must stay resolution-independent (see
    // `perSqM`).
    stampFlakes(S, perSqM(2340), makeRNG(seed + 611), {
      lenMin: 0.018 * perM, lenMax: 0.036 * perM,
      notch: 0.15, hMin: 0.62, hMax: 1.0,
      mask: drift, maskBias: 0.10, maskGain: 0.92,
    }, flake);
    mark('petals');

    // --- micro-relief height ------------------------------------------------
    // Amplitudes are "how steep", not "how big": a normal map shows the
    // *gradient*, and the RMS gradient of a band-limited layer goes as its
    // amplitude divided by its wavelength. The first balance here made that
    // mistake in the other direction from the map it replaces - the 11 mm grain
    // had 0.34 against the 95 mm clods' 0.15, which is a gradient ratio of five
    // to one, and the finished surface was uniform sandpaper with no aggregate
    // structure visible at all. Fine grain must still lead (real earth is
    // dominated by it at centimetre scale) but the coarse layers need enough
    // amplitude to survive the division by their own wavelength.
    //
    // Measured at 256², in RMS gradient units, this stack now lands roughly:
    // grain 55%, crumb voids and clod joints 22%, fragment offsets 13%, thatch
    // and grit 10%. That is a surface with visible structure at every scale from
    // one texel to a tenth of the tile, which is the whole point.
    const height = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      let h = 0.5;
      h += (grainA[i] - 0.5) * 0.22;      // finest grain, ~11 mm
      h += (grainB[i] - 0.5) * 0.20;      // ~24 mm
      h += (grainC[i] - 0.5) * 0.24;      // ~58 mm
      h -= wallF[i] * 0.30;               // crumb voids, ~38 mm cells
      h -= wallC[i] * 0.34;               // clod joints, ~95 mm cells
      h += (idF[i] - 0.5) * 0.20;         // per-fragment flat offsets
      h += (idC[i] - 0.5) * 0.32;
      h += th.h[i] * th.a[i] * 0.34;      // stems lie on top of all of it
      h += thF.h[i] * thF.a[i] * 0.16;
      h += peb.h[i] * 0.22;
      height[i] = h;
    }
    normalizeField(height, 0.004, 0.996);

    // --- cavity AO ----------------------------------------------------------
    // The AO channel is where the fallback bake did most of its visible damage:
    // derived from a field of domes, it put a bright disc on the top of every
    // one, which is what made the blobs *pale*. Occlusion under a meadow belongs
    // in the creases between crumbs and nowhere else, so it is driven mostly by
    // the narrowed wall masks and only partly by the local dip, and its mean
    // sits close to 1 with thin dark lines rather than half the tile lit.
    const hBlur = boxBlurWrapFast(height, S, S, Math.max(2, Math.round(S / 26)));
    const ao = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const dip = clamp01((hBlur[i] - height[i]) * 2.6);
      const occ = clamp01(dip * 0.60 + wallF[i] * 0.58 + wallC[i] * 0.46);
      ao[i] = 1 - occ * 0.34;
    }
    mark('height+AO');

    // --- normal amplitude, solved rather than dialled ------------------------
    // `normalFromHeight` scales the Sobel gradient by `strength * S / 128`, so
    // the RMS surface slope it produces is atan(k * gradientRMS). Inverting that
    // makes the relief resolution-independent, tier-independent and immune to
    // any later change to the height stack - and it is the whole fix for
    // "strong round bumps make it look like bubble wrap", because the amplitude
    // can no longer drift up as fields are added.
    const gRMS = gradientRMS(height, S, S);
    const kWant = Math.tan(TERRAIN_SLOPE_RMS_DEG * PI / 180) / Math.max(gRMS, 1e-6);
    const nrmStrength = kWant * 128 / S;
    const normalData = this.normalFromHeight(height, S, S, { strength: nrmStrength, wrap: true });
    // normalFromHeight puts the source height in alpha; terrain.js wants cavity
    // AO there instead.
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = y * S + x;
        normalData[i * 4 + 3] = q8(ao[i], dither(x, y));
      }
    }
    mark('normal');

    // --- detail channels ----------------------------------------------------
    // Every one is high-passed. A tile reads as a tile because the eye finds a
    // landmark and then finds it again one tile away, and landmarks are
    // low-frequency by definition - so a map that repeats every 2 m has no
    // business carrying anything above about half a metre. Broad drift is the
    // macro map's job and only its job.
    const chR = new Float32Array(N);
    const chG = new Float32Array(N);
    const chB = new Float32Array(N);
    const chA = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      // R - bare soil. Grain plus the fragment mosaic plus grit; the thatch
      // subtracts, because where stems lie you see stems, not soil structure.
      //
      // The crumb terms are modulated by `mat`, so the tile has worked, crumbed
      // patches and smoother ones rather than one uniform grade of aggregate
      // everywhere. Uniform structure is the second-order version of the same
      // mistake as a uniform lattice: nothing in soil is evenly anything.
      const crumbK = 0.45 + 1.15 * mat[i];
      chR[i] = 0.5
        + (grainA[i] - 0.5) * 0.26
        + (grainB[i] - 0.5) * 0.22
        + (grainC[i] - 0.5) * 0.20
        + (idC[i] - 0.5) * 0.46 * crumbK
        + (idF[i] - 0.5) * 0.26 * crumbK
        - wallC[i] * 0.34
        - wallF[i] * 0.24
        + peb.m[i] * 0.34
        - th.a[i] * 0.12;

      // G - DEAD thatch: the two stamped fibre layers, sitting in their mat.
      // Not "grass". This channel is straw and broken stem lying on the soil;
      // world/grass.js draws everything that is alive and green.
      chG[i] = 0.34
        + mat[i] * 0.15
        + th.h[i] * th.a[i] * 0.50
        + thF.h[i] * thF.a[i] * 0.34
        + (grainA[i] - 0.5) * 0.16;

      // B - moss: softened cushions where the mat is thick, a trace of dome,
      // and a fine crumb that dominates within each cushion.
      // The fine terms are deliberately the *minority* here, unlike in R. This
      // channel is a splat height with a standard deviation of 0.21, so the
      // blend sees +/-0.105 against a 0.22 admission band: energy at one texel
      // would fragment a moss patch into per-texel speckle at close range, where
      // the mid-scale cushion structure fragments it into cushions instead.
      const cushK = 0.30 + 1.05 * mossPatch[i];
      chB[i] = 0.5
        + (mossSoft[i] - 0.5) * 0.44 * cushK
        - mossWall[i] * 0.18 * cushK
        + (mossFrag.cells[i] - 0.5) * 0.16 * mossPatch[i]
        + (mossGrain[i] - 0.5) * 0.24
        + (grainA[i] - 0.5) * 0.10;

      // A - fallen petals. Not high-passed: it is a sparse mask, and recentring
      // a sparse mask on 0.5 lifts its empty background into the blend.
      //
      // The gain and bias are set to reproduce the *distribution* terrain.js
      // calibrated its litter weight of 1.5 against - "peaks only reach ~0.64
      // and exceed 0.7 essentially never", measured coverage near 30%. This is
      // a splat height, not a real one, and matching it is what keeps the pink
      // under the canopy a scatter of petals in the hollows rather than the
      // decal ring the number was chosen to avoid.
      chA[i] = clamp01(flake.a[i] * (0.40 + 0.34 * flake.h[i]));
    }

    const hpR = Math.max(2, Math.round(S / 7));
    highpassField(chR, S, S, hpR);
    highpassField(chG, S, S, hpR);
    highpassField(chB, S, S, hpR);

    // Explicit distributions. `terrainHeightBlend` forms `w + h*0.5` and admits
    // any layer within 0.22 of the leader, so a channel's standard deviation is
    // exactly how decisively it competes - and these were chosen to sit close to
    // what the fallback bake produced (measured means 0.45 / 0.37 / 0.49) so
    // terrain.js's splat weights, which were tuned against those, still land
    // where their author put them. Only the spread is widened, and only enough
    // that soil, thatch and moss each genuinely break through somewhere.
    retargetField(chR, 0.48, 0.18);
    retargetField(chG, 0.46, 0.19);
    retargetField(chB, 0.47, 0.21);
    mark('channels');

    const detailData = new Uint8Array(N * 4);
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = y * S + x, o = i * 4;
        const d = dither(x, y);
        detailData[o] = q8(chR[i], d);
        detailData[o + 1] = q8(chG[i], d);
        detailData[o + 2] = q8(chB[i], d);
        detailData[o + 3] = q8(chA[i], d);
      }
    }

    // --- upload -------------------------------------------------------------
    // Anisotropy 4 at and below HIGH is exactly what terrain.js asked its own
    // bake for, so this override costs the sampler nothing; ULTRA, which already
    // targets 30 fps, gets 8.
    const aniso = this.tier === 'ultra' ? Math.min(8, this.maxAnisotropy) : Math.min(4, this.maxAnisotropy);
    const detail = this._dataTexMips(detailData, S, S, {
      srgb: false, name: 'terrainDetail', anisotropy: aniso,
    });
    // A tangent-space normal encodes as an affine function of its components, so
    // box-filtering the encoded bytes IS averaging the vectors - which is the
    // correct mip filter here, and the shortened mean vector it leaves behind is
    // the right answer too: a footprint whose normals disagreed should shade
    // flatter. (`normalMipChain` would additionally hand the lost variance to a
    // roughness chain, but terrain.js takes its roughness from `uLayerRough`,
    // not from this map, so there is nothing to hand it to.)
    const normalAO = this._dataTexMips(normalData, S, S, {
      srgb: false, name: 'terrainNormalAO', anisotropy: aniso,
    });
    mark('detail+normal upload');
    const macro = this._buildTerrainMacro(this._size('terrainMacro'));
    mark('macro');

    detail.userData.tileMeters = TILE_M;
    normalAO.userData.tileMeters = TILE_M;

    // --- verification -------------------------------------------------------
    // Seamlessness is asserted, not assumed: every kernel above is periodic by
    // construction, but "by construction" is exactly the claim that turns out to
    // be false when one sprite stamper forgets a modulo. This measures the
    // finished bytes.
    this._seamCheckRGBA(detailData, S, S, 'terrainDetail');
    this._seamCheckRGBA(normalData, S, S, 'terrainNormalAO');

    if (this.verbose) {
      let cov = 0, aMax = 0, aMean = 0;
      for (let i = 0; i < N; i++) {
        const v = chA[i];
        if (v >= 0.5) cov++;
        if (v > aMax) aMax = v;
        aMean += v;
      }
      const sR = fieldStats(chR), sG = fieldStats(chG), sB = fieldStats(chB);
      console.info(
        `[TextureFactory] terrainSurface ${S}² - RMS slope ${TERRAIN_SLOPE_RMS_DEG}° `
        + `(gradRMS ${gRMS.toFixed(4)}, strength ${nrmStrength.toFixed(3)}); `
        + `R ${sR.mean.toFixed(3)}/${sR.sd.toFixed(3)} `
        + `G ${sG.mean.toFixed(3)}/${sG.sd.toFixed(3)} `
        + `B ${sB.mean.toFixed(3)}/${sB.sd.toFixed(3)} `
        + `A mean ${(aMean / N).toFixed(3)} max ${aMax.toFixed(3)} cover>0.5 ${(100 * cov / N).toFixed(1)}%`);
      console.info(`[TextureFactory] terrainSurface phases (ms): ${phases.join(' · ')}`);
    }

    this._register('terrain.detail', detail);
    this._register('terrain.normalAO', normalAO);
    this._register('terrain.macro', macro);

    return { detail, normalAO, macro, size: S, tileMeters: TILE_M };
  }

  /**
   * Region and patch variation: R dryness, G lushness, B moss affinity,
   * A mottle. Linear data, no colour.
   *
   * terrain.js samples this one texture at two scales - 1024 m for the region
   * and 128 m quarter-turned for the patch - and sums the two, which widens the
   * distribution instead of averaging it flat. That only works if the map has
   * real structure across its whole spectrum, so the octave ladders run as deep
   * as the resolution allows rather than piling their energy into the first two
   * - except for the two channels the warp differentiates, which are capped in
   * cells instead. See point 2.
   *
   * TWO THINGS HERE ARE LOAD-BEARING FOR ANOTHER MODULE. Both are stated with
   * the numbers actually measured off the bake this replaces, because the first
   * draft of this comment quoted remembered ones and got them backwards:
   *
   *  1. The distributions. A channel's standard deviation IS how decisively the
   *     layer it feeds competes, because `terrainHeightBlend` only admits a
   *     layer within `sharpness` (0.22) of the leader. The four targets are
   *     R 0.46/0.20  G 0.51/0.20  B 0.46/0.19  A 0.50/0.145 and the finished
   *     bake hits them to three decimals (verified: the verbose line below
   *     prints the achieved moments).
   *
   *     WHAT THEY PRODUCE, measured this pass by running terrain.js's own
   *     `terrainHeightBlend` over 120k samples of the real baked bytes with
   *     slope, cavity and altitude drawn from the heightfield's real ranges:
   *
   *       bare earth 58.7%   dead thatch 35.8%   moss 5.6%
   *       visible (share > 0.15) at 79% / 57% / 12% of samples
   *
   *     Those are the CURRENT layer semantics. Do not retune against the older
   *     figures this comment used to quote ("soil 25.7 / grass 56.2 / moss
   *     18.1"): they described a terrain material that carried a living-green
   *     "grass" layer at over half the ground, which is precisely what made the
   *     ground read as a mottled blue-green bog and why that layer was deleted.
   *     The ground is now substrate - earth leading, straw supporting, moss a
   *     hollow-and-shade minority - and world/grass.js owns the sward. Raising
   *     G here to "get the green back" would reopen exactly that bug.
   *
   *     R's mean is held at 0.46 rather than 0.50 for a different reason now: it
   *     drives `gDry`, the damp-umber-to-pale-crust axis, which is a factor of
   *     3.7 in linear luminance and by far the strongest broad tonal drift on
   *     the ground. Centring it high bleaches the whole field toward dry crust;
   *     0.46 leaves the median at gDry 0.34, i.e. mostly damp earth with dry
   *     ridges, which is what a meadow floor is.
   *  2. The GRADIENT of A and B, which is a separate constraint from their
   *     spread and the one that is easy to break by accident. terrain.js
   *     domain-warps its near detail lookup by `(tMes.ba - 0.5) * 3.4 m` and
   *     published the Jacobian table that picked 3.4 (see TERRAIN_WARP_GRAD_RMS).
   *     Gradient is sd times frequency, so putting the same contrast at a higher
   *     frequency breaks that bound without touching a single number in
   *     terrain.js. The first version of this function did exactly that: sd
   *     barely moved (A 0.176 -> 0.145, B 0.160 -> 0.188) but the octave ladders
   *     were derived from the texture resolution, so the per-texel gradient rose
   *     2.6x at 128² and 3.4x at ULTRA's 256², putting the warp determinant at
   *     0.09..2.05 and -0.81..2.71 respectively - past terrain.js's own "visibly
   *     blurs" row and into its "FOLDS" row, with ULTRA worse than HIGH.
   *     So the two warp-driving channels get a ladder capped in CELLS rather
   *     than in texels, and the finished gradient is measured and solved for.
   */
  _buildTerrainMacro(S) {
    const N = S * S;
    const seed = this.seed + 8700;

    /**
     * Deepest octave ladder from `base` whose finest octave still sits at 3.2 or
     * more texels a cell.
     *
     * Not decoration: a hand-picked octave count is a Nyquist bug waiting for a
     * resolution change, and this map has two of them (128 and 256). Five
     * octaves from 6 cells reaches 96, which at 128² is 1.3 texels a cell - pure
     * aliasing, contributing sparkle to the base mip and literally nothing one
     * level down, while costing a full-resolution pass to compute. Deriving the
     * count from the resolution is the only way "as much spectrum as possible"
     * and "no octave past Nyquist" can both be true on both sizes.
     */
    const cap = Math.max(2, Math.round(S / 3.2));
    const octFor = (base) => {
      let o = 1, f = base;
      while (o < 6 && f * 2 <= cap) { f *= 2; o++; }
      return o;
    };
    /**
     * As `octFor`, but for the two channels terrain.js domain-warps by: capped
     * in CELLS as well as in texels, so it is the same ladder at 128² and at
     * ULTRA's 256².
     *
     * A texel-derived cap is exactly right for a channel that is only ever
     * *looked at* and exactly wrong for one that is *differentiated*: doubling
     * the resolution doubles the finest frequency, which doubles the field's
     * gradient at the same standard deviation, which doubles the warp Jacobian.
     * That is how ULTRA ended up folding the near detail lookup while HIGH only
     * blurred it. 22 cells is where the fallback bake's warp channels sat
     * (pfbm base 6 x 3 octaves and base 2 x 5) and where terrain.js's published
     * det 0.62..1.40 was measured.
     */
    const WARP_CELL_CAP = 22;
    const octWarp = (base) => {
      const c = Math.min(WARP_CELL_CAP, cap);
      let o = 1, f = base;
      while (o < 6 && f * 2 <= c) { f *= 2; o++; }
      return o;
    };

    const dry = normalizeField(fbm2(S, S, { fx: 3, octaves: octFor(3), gain: 0.62, lacunarity: 2.11, seed: seed + 1 }));
    const lush = normalizeField(fbm2(S, S, { fx: 4, octaves: octFor(4), gain: 0.60, lacunarity: 2.03, seed: seed + 2 }));
    // Moss follows drainage, and drainage follows the *valleys* of a ridged
    // field - which is a real reason for a boundary to be lobed rather than
    // blobby, and the reason a thresholded fbm never looks like vegetation.
    const drain = normalizeField(fbm2(S, S, { fx: 4, octaves: octWarp(4), gain: 0.5, kind: 2, seed: seed + 3 }));
    const drainF = normalizeField(fbm2(S, S, { fx: 9, octaves: octWarp(9), gain: 0.5, kind: 2, seed: seed + 4 }));
    const mossW = normalizeField(fbm2(S, S, { fx: 6, octaves: octWarp(6), gain: 0.58, seed: seed + 5 }));
    // Mottle is stretched along one axis: the things it stands for - old burn,
    // drift, grazing - all have a direction. Base 3, not 6: the fallback bake
    // ran this channel from base 2, and tripling its fundamental was most of
    // why the warp gradient blew up (its gradient-to-spread ratio went 0.077 to
    // 0.241 while its spread actually *fell*).
    const mottle = normalizeField(fbm2(S, S, { fx: 3, fy: 2, octaves: octWarp(3), gain: 0.50, seed: seed + 6 }));

    const cDry = new Float32Array(N);
    const cLush = new Float32Array(N);
    const cMoss = new Float32Array(N);
    const cMot = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      cDry[i] = dry[i];
      // Lushness is mildly anti-correlated with dryness (r ~ -0.18) rather than
      // independent. Fully independent fields let a patch be both parched and
      // lush, which is where "procedural patchwork" comes from; fully
      // anti-correlated collapses two degrees of freedom into one.
      cLush[i] = lush[i] * 0.88 + (1 - dry[i]) * 0.17;
      // Wide smoothstep windows on purpose. A narrow one is a contrast
      // expander - 0.46..0.08 multiplies the field's gradient by 1.5/0.38 ≈ 3.9
      // inside the transition band - and this channel's gradient is spent
      // budget, not free (see the header). 0.60 and 0.52 wide give the same
      // lobed drainage boundary at a third less slope.
      const damp = clamp01(smoothstep(0.62, 0.02, drain[i]) * 0.72 + smoothstep(0.56, 0.04, drainF[i]) * 0.42);
      cMoss[i] = mossW[i] * 0.55 + damp * 0.46;
      cMot[i] = mottle[i];
    }
    // 0.46, not 0.50. R does two jobs in terrain.js and both want it low.
    //
    // It enters `wEarth` at +0.85 and `wThatch` at -0.14, so it is the dial that
    // strips dead straw off and exposes bare soil - and it is ALSO the dominant
    // term of `gDry`, the damp-umber -> pale-grey-crust axis, which spans a
    // factor of 3.7 in linear luminance. Centring it at 0.50 pulls the median
    // gDry from 0.34 to 0.42 and lightens the whole field toward dry crust,
    // which is the one direction that makes soil stop reading as damp earth
    // under a sward and start reading as a dusty track. 0.46 keeps the median
    // firmly on the damp side (measured gDry p05/p50/p95 = 0.00 / 0.34 / 0.75)
    // while still letting ridges and dry macro patches crust over properly.
    retargetField(cDry, 0.46, 0.20, 0.02, 0.98);
    retargetField(cLush, 0.51, 0.20, 0.02, 0.98);

    // --- the domain-warp budget, measured and solved -------------------------
    // B and A are what terrain.js displaces its near detail lookup by. Their
    // spread is set here for the splat, and their *gradient* is then checked
    // against the Jacobian bound terrain.js published and brought back inside it
    // if it is over. Both operations are needed and they pull in opposite
    // directions, so this is a loop rather than a formula: blur (which removes
    // gradient), retarget (which puts the contrast back), measure again.
    //
    // The blur radius is in world metres, not texels, so ULTRA converges to the
    // same field as HIGH rather than to a sharper one. Bounded at four passes;
    // if it ever ran out it would warn rather than ship a folding warp.
    let cMossW = cMoss, cMotW = cMot, warpPasses = 0;
    const warpRMS = () => Math.hypot(centralGradRMS(cMossW, S, S), centralGradRMS(cMotW, S, S))
      * (S / TERRAIN_MESO_M) * TERRAIN_WARP_M;
    const retargetWarp = () => {
      retargetField(cMossW, 0.46, 0.19, 0.01, 0.98);
      // 0.145, not 0.20: this channel drives the warp *and* the ±8% tint drift,
      // the puddle mask and the near normal scale, none of which want a swing.
      retargetField(cMotW, 0.50, 0.145, 0.04, 0.96);
    };
    retargetWarp();
    while (warpRMS() > TERRAIN_WARP_GRAD_RMS && warpPasses < 4) {
      warpPasses++;
      const r = warpPasses * Math.max(1, Math.round(S / 64));
      cMossW = boxBlurWrapFast(cMossW, S, S, r);
      cMotW = boxBlurWrapFast(cMotW, S, S, r);
      retargetWarp();
    }
    const warpFinal = warpRMS();
    if (warpFinal > TERRAIN_WARP_GRAD_RMS * 1.15) {
      console.warn(`[TextureFactory] terrainMacro warp gradient ${warpFinal.toFixed(3)} exceeds the `
        + `${TERRAIN_WARP_GRAD_RMS} world/terrain.js calibrated DETAIL_WARP=${TERRAIN_WARP_M} m against; `
        + 'the near detail lookup will over-warp and lose mip accuracy.');
    }

    const data = new Uint8Array(N * 4);
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = y * S + x, o = i * 4;
        const d = dither(x, y);
        data[o] = q8(cDry[i], d);
        data[o + 1] = q8(cLush[i], d);
        data[o + 2] = q8(cMossW[i], d);
        data[o + 3] = q8(cMotW[i], d);
      }
    }
    // Anisotropy 1: this map is sampled from raw world coordinates at 1/1024 and
    // 1/128, so its screen-space derivative is minute and an anisotropic tap
    // would fetch sixteen texels to return the one it already had.
    const t = this._dataTexMips(data, S, S, { srgb: false, name: 'terrainMacro', anisotropy: 1 });
    t.userData.tileMeters = 1024;
    this._seamCheckRGBA(data, S, S, 'terrainMacro');
    if (this.verbose) {
      const sD = fieldStats(cDry), sL = fieldStats(cLush);
      const sM = fieldStats(cMossW), sT = fieldStats(cMotW);
      console.info(`[TextureFactory] terrainMacro ${S}² - R ${sD.mean.toFixed(3)}/${sD.sd.toFixed(3)} `
        + `G ${sL.mean.toFixed(3)}/${sL.sd.toFixed(3)} B ${sM.mean.toFixed(3)}/${sM.sd.toFixed(3)} `
        + `A ${sT.mean.toFixed(3)}/${sT.sd.toFixed(3)}; warp gradient ${warpFinal.toFixed(3)} `
        + `(budget ${TERRAIN_WARP_GRAD_RMS}, ${warpPasses} smoothing pass${warpPasses === 1 ? '' : 'es'})`);
    }
    return t;
  }

  /**
   * Assert that an RGBA map really wraps.
   *
   * Every kernel in this file is periodic by construction - but "by
   * construction" is precisely the claim that quietly stops being true when one
   * sprite stamper drops a modulo or a blur radius exceeds the half-period, and
   * a seam on ground that tiles every 2 m is visible from anywhere in the world.
   *
   * The test: a periodic field's wrap-around column pair is just one more
   * adjacent column pair, so its mean absolute step must be an unremarkable draw
   * from the distribution of all w of them. Comparing it against ONE interior
   * pair - which is what this did first - is a statistic with 6% noise on each
   * side, and it duly cried seam on a map whose wrap column ranked 202nd of 256
   * and sat below the 95th percentile: the baseline column simply happened to be
   * a smooth one. Comparing against the whole distribution has no such failure
   * mode, and a genuine seam is not a marginal outlier but a step several times
   * anything the interior produces, so the test demands the wrap pair beat both
   * the mean and the worst interior pair before it says anything.
   *
   * All four channels are accumulated in one traversal per axis: ~1 ms at 256²,
   * which is worth paying permanently for a class of bug that is invisible in a
   * unit test and unmissable in the shipped frame.
   */
  _seamCheckRGBA(data, w, h, name) {
    const colStep = new Float64Array(w * 4);      // per column pair, per channel
    for (let y = 0; y < h; y++) {
      const row = y * w * 4;
      for (let x = 0; x < w; x++) {
        const a = row + x * 4;
        const b = row + ((x + 1) % w) * 4;
        const o = x * 4;
        colStep[o] += Math.abs(data[b] - data[a]);
        colStep[o + 1] += Math.abs(data[b + 1] - data[a + 1]);
        colStep[o + 2] += Math.abs(data[b + 2] - data[a + 2]);
        colStep[o + 3] += Math.abs(data[b + 3] - data[a + 3]);
      }
    }
    const rowStep = new Float64Array(h * 4);
    for (let y = 0; y < h; y++) {
      const ra = y * w * 4, rb = ((y + 1) % h) * w * 4;
      const o = y * 4;
      for (let x = 0; x < w; x++) {
        const a = ra + x * 4, b = rb + x * 4;
        rowStep[o] += Math.abs(data[b] - data[a]);
        rowStep[o + 1] += Math.abs(data[b + 1] - data[a + 1]);
        rowStep[o + 2] += Math.abs(data[b + 2] - data[a + 2]);
        rowStep[o + 3] += Math.abs(data[b + 3] - data[a + 3]);
      }
    }

    // How many standard deviations above the interior pairs the wrap pair sits.
    //
    // A plain ratio to the mean is not a usable criterion, and the numbers say
    // so: measured over 240 channel/axis samples of maps that provably tile, the
    // wrap pair reached 1.67x the mean at 128² purely by chance, because a 128²
    // map only has 128 pairs to draw from and each is averaged over only 128
    // rows. The z-score normalises that away. Same 240 samples: median -0.18,
    // p95 1.65, worst 4.71. Against three deliberately non-tiling maps (a
    // horizontal ramp, and a three-column edge as a stamper that dropped a
    // modulo would leave): 5.7 to 9.7. The gap is where the threshold goes.
    let worst = -Infinity, worstLabel = '';
    const judge = (steps, n, wrapIdx, axis) => {
      for (let c = 0; c < 4; c++) {
        let sum = 0, maxOther = 0;
        for (let i = 0; i < n; i++) {
          if (i === wrapIdx) continue;
          const v = steps[i * 4 + c];
          sum += v;
          if (v > maxOther) maxOther = v;
        }
        const mean = sum / (n - 1);
        // A flat channel has no interior step to be measured against; nothing to
        // say, and dividing by it would manufacture a warning.
        if (mean < 0.05) continue;
        let v2 = 0;
        for (let i = 0; i < n; i++) {
          if (i === wrapIdx) continue;
          const d = steps[i * 4 + c] - mean;
          v2 += d * d;
        }
        const sd = Math.sqrt(v2 / (n - 1)) || 1e-6;
        const seam = steps[wrapIdx * 4 + c];
        const z = (seam - mean) / sd;
        if (z > worst) { worst = z; worstLabel = `${axis} ch${c}`; }
        // Both conditions: z alone has a tail, and "worst pair" alone happens to
        // one channel in four by chance. A real seam is comfortably both.
        if (z > 6 && seam > maxOther) {
          console.warn(`[TextureFactory] "${name}" does not tile across ${axis} ch${c}: `
            + `wrap step ${seam.toFixed(1)} is ${z.toFixed(1)} sd above the interior pairs `
            + `(mean ${mean.toFixed(1)}, worst ${maxOther.toFixed(1)}).`);
        }
      }
    };
    judge(colStep, w, w - 1, 'U');
    judge(rowStep, h, h - 1, 'V');

    if (this.verbose) {
      console.info(`[TextureFactory] "${name}" wraps; worst wrap-pair z = `
        + `${worst.toFixed(2)} sd (${worstLabel})`);
    }
    return worst;
  }

  // -------------------------------------------------------------------------
  // LEAF 4 - foliage cards
  // -------------------------------------------------------------------------

  /**
   * Five-petal sakura blossom card (Somei-Yoshino).
   * RGBA albedo with a ramped cutout alpha, plus a matching normal built from
   * the petal-curl height field so a blossom cluster still shades when the sun
   * rakes across it. Use `alphaTest: 0.5`.
   */
  blossomCard() {
    return this._group('blossom', () => this._buildBlossom(this._size('blossom')));
  }

  _buildBlossom(S) {
    const N = S * S;
    const rand = makeRNG(this.seed + 8100);

    const PETALS = 5;
    const baseRot = rand() * TAU;
    // Per-petal jitter. A perfectly regular pentagon is the tell.
    const pCos = new Float32Array(PETALS);
    const pSin = new Float32Array(PETALS);
    const pLen = new Float32Array(PETALS);
    const pWid = new Float32Array(PETALS);
    const pNotch = new Float32Array(PETALS);
    const pLift = new Float32Array(PETALS);
    const pPhase = new Float32Array(PETALS);
    for (let i = 0; i < PETALS; i++) {
      const a = baseRot + (i / PETALS) * TAU + (rand() - 0.5) * 0.20;
      // Hoisted out of the pixel loop: this is five sin/cos per texel otherwise,
      // which on a 512² card is 2.6 M transcendentals for no reason at all.
      pCos[i] = Math.cos(a);
      pSin[i] = Math.sin(a);
      pLen[i] = 0.90 * (0.92 + 0.16 * rand());
      pWid[i] = 0.44 * (0.90 + 0.20 * rand());
      pNotch[i] = 0.11 + 0.07 * rand();
      pLift[i] = 0.75 + 0.5 * rand();
      pPhase[i] = rand() * TAU;
    }

    const cover = new Float32Array(N);
    const height = new Float32Array(N);
    const colR = new Float32Array(N);
    const colG = new Float32Array(N);
    const colB = new Float32Array(N);

    // Base is pinker than the tip: a real Somei-Yoshino petal is near-white
    // with the colour concentrated at the claw and along the veins.
    const cTip = rgb(0xfbf5f5);
    const cMidP = rgb(0xf2dee1);
    const cBase = rgb(0xdcaeb8);
    const cVein = rgb(0xd7a4b0);

    const px2 = 2 / S;                     // texel size in the -1..1 card space
    const edge = px2 * 1.6;                // ~1.6 texel alpha ramp: survives mips
    // Vein width is a property of the flower, not of the texture: authoring it
    // as "1.4 texels" meant the veins were hairlines at 512 and coarse ribs at
    // 256, so the LOW tier drew a visibly different species. Physical width,
    // with a one-texel floor so it can still be antialiased.
    const veinW = Math.max(px2 * 1.1, 0.010);
    // The height field is composited with a much softer ramp than the alpha.
    // Both share the same silhouette, so a height ramp as narrow as the alpha
    // ramp puts a full 90° of normal turn inside the 1.6 texels that the alpha
    // test cuts through, and the card gets a hard bright rim all the way round.
    const hEdge = edge * 5;

    const mottle = fbm2(S, S, { fx: 9, octaves: 3, seed: this.seed + 8181 });
    normalizeField(mottle);

    for (let y = 0; y < S; y++) {
      const cy = ((y + 0.5) / S) * 2 - 1;
      for (let x = 0; x < S; x++) {
        const cx = ((x + 0.5) / S) * 2 - 1;
        const i = y * S + x;

        // Composite the petals front-to-back with a plain 'over'. A real
        // corolla is imbricate - each petal overlaps the next and the last
        // tucks under the first - so simple ordered compositing is both the
        // correct model and the one that antialiases overlaps for free.
        // Starting from a pale tint rather than black is what keeps the mip
        // chain from dragging darkness into the cutout edge.
        let outA = 0;
        let oR = cMidP[0] * 0.94, oG = cMidP[1] * 0.94, oB = cMidP[2] * 0.94;
        let oH = 0.30;

        for (let p = 0; p < PETALS; p++) {
          const ca = pCos[p], sa = pSin[p];
          const lx = cx * ca + cy * sa;      // along the petal, 0 at flower centre
          const ly = -cx * sa + cy * ca;     // across
          const L = pLen[p], W = pWid[p];
          if (lx < -0.08 || lx > L * 1.12) continue;
          if (ly > W * 1.25 || ly < -W * 1.25) continue;
          const t = lx / L;
          // Margin wobble. A petal outline that is an exact analytic curve is
          // the last thing keeping this from reading as a photograph of a
          // flower; real margins are faintly crimped and slightly asymmetric.
          // Two harmonics along the blade, per petal, at about a percent of the
          // length - enough to break the curve, far too little to change shape.
          const wob = (Math.sin(t * 9.3 + pPhase[p]) * 0.6 + Math.sin(t * 21.7 + pPhase[p] * 2.3) * 0.4)
            * 0.014 * L * (ly < 0 ? -1 : 1);
          const dist = petalDist(t, ly, L, W, pNotch[p], W * 0.30) + wob;
          const alpha = clamp01(dist / edge + 0.5);
          if (alpha <= 0.002) continue;

          // Colour along the blade.
          const g = smoothstep(0.02, 0.72, t);
          let r0, g0, b0;
          if (g < 0.5) {
            const k = g * 2;
            r0 = lerp(cBase[0], cMidP[0], k); g0 = lerp(cBase[1], cMidP[1], k); b0 = lerp(cBase[2], cMidP[2], k);
          } else {
            const k = (g - 0.5) * 2;
            r0 = lerp(cMidP[0], cTip[0], k); g0 = lerp(cMidP[1], cTip[1], k); b0 = lerp(cMidP[2], cTip[2], k);
          }

          // Veins: five, fanning from the claw and following the widening
          // blade rather than running parallel, which is what they do in life.
          const wt = petalWidth(t, W);
          let vein = 0;
          if (wt > 1e-4) {
            // Five equal-width, equal-amplitude Gaussians at rel = 0, ±0.4, ±0.8.
            // Their maximum is always the one nearest the sample, so index it
            // rather than evaluating five exponentials per texel - which was
            // 1.3 M transcendentals on a 512² card, and the largest single
            // cost in the blossom bake.
            const rel = ly / wt;
            let q = Math.round(rel * 2.5);
            if (q > 2) q = 2; else if (q < -2) q = -2;
            const sN = (rel - q * 0.40) * wt / veinW;
            vein = Math.exp(-sN * sN)
              * smoothstep(0.06, 0.35, t) * (1 - smoothstep(0.86, 1.0, t));
          }
          const vk = vein * 0.30;
          r0 = lerp(r0, cVein[0], vk); g0 = lerp(g0, cVein[1], vk); b0 = lerp(b0, cVein[2], vk);

          // Petals overlap; the one on top gets a touch of light, the one under
          // it a touch of shadow. Without this the flower reads as a flat star.
          const overlap = clamp01(1 - Math.abs(ly) / (wt + 1e-4));
          const shade = 0.90 + 0.14 * overlap * pLift[p];
          const mott = 0.97 + 0.06 * mottle[i];
          r0 *= shade * mott; g0 *= shade * mott; b0 *= shade * mott;

          // Height: the petal is a shallow saddle - dished across, lifted at
          // the tip. Veins ride slightly proud of the blade.
          const across = wt > 1e-4 ? ly / wt : 0;
          const hh = 0.40
            + (1 - across * across) * 0.22 * pLift[p]
            + smoothstep(0.3, 1.0, t) * 0.10
            + vein * 0.06;

          oR += (r0 - oR) * alpha;
          oG += (g0 - oG) * alpha;
          oB += (b0 - oB) * alpha;
          oH += (hh - oH) * clamp01(dist / hEdge + 0.5);
          outA = alpha + outA * (1 - alpha);
        }

        cover[i] = outA;
        height[i] = oH;
        colR[i] = oR; colG[i] = oG; colB[i] = oB;
      }
    }

    // --- stamens ----------------------------------------------------------
    // ~34 filaments with anthers. Drawn after the petals so they sit on top,
    // and only evaluated inside the central disc.
    {
      const NF = 34;
      const cFil = rgb(0xf7f0e4);
      const cAnth = rgb(0xc9a86a);
      const cAnthD = rgb(0x9d7f45);
      // Stride 5: cos, sin, length, bend, anther radius. Same reason as the
      // petals - trig belongs outside the pixel loop.
      const FS = 5;
      const fils = new Float32Array(NF * FS);
      for (let f = 0; f < NF; f++) {
        const a = rand() * TAU;
        fils[f * FS] = Math.cos(a);
        fils[f * FS + 1] = Math.sin(a);
        fils[f * FS + 2] = 0.20 + 0.30 * rand();
        fils[f * FS + 3] = (rand() - 0.5) * 0.55;
        fils[f * FS + 4] = 0.016 + 0.012 * rand();
      }
      // One bounding box per filament, not every filament against every texel
      // of the central disc. The old loop was 34 evaluations across ~94k texels
      // - 3.2 million - where a filament actually covers about 1,800 of them,
      // and it was the single most expensive block in the whole file. Drawing
      // them in sequence rather than by maximum alpha is also the more correct
      // model: a stamen in front occludes the one behind it.
      const c0 = S * 0.5;
      const toTex = S * 0.5;                 // card units -> texels
      for (let f = 0; f < NF; f++) {
        const ca = fils[f * FS], sa = fils[f * FS + 1];
        const len = fils[f * FS + 2], bend = fils[f * FS + 3], ar = fils[f * FS + 4];
        // The filament arcs, so its lateral reach is the arc offset at full
        // length plus the anther radius plus a texel of ramp.
        const reach = Math.abs(bend * len * len) + ar + 0.012;
        const ex = Math.ceil((Math.abs((len + ar) * ca) + Math.abs(reach * sa)) * toTex) + 2;
        const ey = Math.ceil((Math.abs((len + ar) * sa) + Math.abs(reach * ca)) * toTex) + 2;
        const px = Math.round(c0), py = Math.round(c0);
        for (let dy = -ey; dy <= ey; dy++) {
          const y = py + dy;
          if (y < 0 || y >= S) continue;
          const cy = ((y + 0.5) / S) * 2 - 1;
          for (let dx = -ex; dx <= ex; dx++) {
            const x = px + dx;
            if (x < 0 || x >= S) continue;
            const cx = ((x + 0.5) / S) * 2 - 1;
            const lx = cx * ca + cy * sa;
            const ly = -cx * sa + cy * ca;
            if (lx < -0.02 || lx > len + ar * 2) continue;
            // Filaments arc outward rather than running dead straight.
            const arc = bend * lx * lx;
            const off = ly - arc;
            const tt = clamp01(lx / Math.max(len, 1e-4));
            const halfW = (0.0085 * (1 - 0.35 * tt));
            const dFil = halfW - Math.abs(off);
            const aFil = lx < len ? clamp01(dFil / (px2 * 1.1) + 0.5) : 0;
            // Anther head at the tip.
            const dxA = lx - len, dyA = off;
            const dA = Math.hypot(dxA, dyA * 1.35);
            const aAnth = clamp01((ar - dA) / (px2 * 1.1) + 0.5);

            let a = 0, bR = 0, bG = 0, bB = 0, bH = 0;
            if (aAnth >= aFil) {
              a = aAnth;
              const shade = 0.82 + 0.36 * clamp01(1 - Math.hypot(dxA, dyA) / (ar + 1e-4));
              bR = lerp(cAnthD[0], cAnth[0], shade); bG = lerp(cAnthD[1], cAnth[1], shade); bB = lerp(cAnthD[2], cAnth[2], shade);
              bH = 0.86;
            } else {
              a = aFil;
              const shade = 0.88 + 0.20 * clamp01(1 - Math.abs(off) / (halfW + 1e-4));
              bR = cFil[0] * shade; bG = cFil[1] * shade; bB = cFil[2] * shade;
              bH = 0.72 + tt * 0.08;
            }
            if (a <= 0.004) continue;
            const i = y * S + x;
            colR[i] = lerp(colR[i], bR, a);
            colG[i] = lerp(colG[i], bG, a);
            colB[i] = lerp(colB[i], bB, a);
            height[i] = lerp(height[i], bH, a);
            cover[i] = a + cover[i] * (1 - a);
          }
        }
      }
    }

    // Flood RGB outward so mip chains never pull background into an edge.
    this._bleedFloat(colR, colG, colB, cover, S, S, 4, [
      cMidP[0] * 0.9, cMidP[1] * 0.9, cMidP[2] * 0.9,
    ]);

    const data = new Uint8Array(N * 4);
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = y * S + x;
        const o = i * 4;
        const d = dither(x, y);
        data[o] = q8(colR[i] / 255, d);
        data[o + 1] = q8(colG[i] / 255, d);
        data[o + 2] = q8(colB[i] / 255, d);
        data[o + 3] = q8(cover[i], d);
      }
    }

    const albedoTex = this._dataTexMips(data, S, S, {
      srgb: true, name: 'blossomAlbedo', repeat: false,
      // Alpha-weighted so no background tint creeps into the silhouette, and
      // coverage-preserving so a blossom cluster at 30 m still has the same
      // amount of flower in it as one at 3 m. Plain alpha averaging shrinks an
      // alpha-tested cutout by a few percent per level, which is why distant
      // procedural foliage so often thins out into nothing.
      alphaWeighted: true, preserveCoverage: true, alphaTest: 0.5,
    });
    const chain = normalMipChain(height, S, S, { strength: 0.55, wrap: false });
    const normalTex = new THREE.DataTexture(chain.levels[0].data, S, S, THREE.RGBAFormat, THREE.UnsignedByteType);
    this._apply(normalTex, { srgb: false, repeat: false, mipmaps: true, anisotropy: this.maxAnisotropy });
    normalTex.mipmaps = chain.levels;
    normalTex.generateMipmaps = false;
    normalTex.name = 'blossomNormal';
    normalTex.needsUpdate = true;
    this._register('blossomAlbedo', albedoTex);
    this._register('blossomNormal', normalTex);
    return { albedo: albedoTex, normal: normalTex, size: S };
  }

  /**
   * Single detached petal, oriented base-at-bottom, tip-at-top.
   * Height carries a real longitudinal curl so a falling petal flashes as it
   * tumbles instead of reading as a flat quad.
   */
  petalCard() {
    return this._group('petal', () => this._buildPetal(this._size('petal')));
  }

  _buildPetal(S) {
    const N = S * S;
    const rand = makeRNG(this.seed + 8600);

    const cover = new Float32Array(N);
    const height = new Float32Array(N);
    const colR = new Float32Array(N);
    const colG = new Float32Array(N);
    const colB = new Float32Array(N);

    const cTip = rgb(0xfaf2f3);
    const cMidP = rgb(0xeed6da);
    const cBase = rgb(0xd8a7b3);
    const cVein = rgb(0xcf9dab);
    const cBruise = rgb(0xc9b39c);

    // Card space is [-1,1]². The petal is sized to fill it: a sakura petal is
    // about as wide as it is long (roughly 12 mm × 15 mm), so a square card
    // wastes almost nothing, and filling it means the consumer's quad is not
    // mostly empty texels being alpha-tested for nothing.
    const MARGIN = 0.03;               // fraction of the card left clear
    const L = 2 * (1 - 2 * MARGIN);    // petal length in card units
    const W = 1.02;                    // max half-width ≈ 0.8 * W = 0.82
    const notch = 0.14;
    const px2 = 2 / S;
    const edge = px2 * 1.6;
    const curlDir = rand() < 0.5 ? 1 : -1;
    const bruise = normalizeField(fbm2(S, S, { fx: 5, octaves: 3, seed: this.seed + 8686 }));
    // Vein width as a fraction of the card, not as a texel count: authored in
    // texels, the LOW tier's 128² card would draw ribs where the 256² draws
    // hairlines. One texel is the floor so it can still antialias.
    const veinW = Math.max(px2 * 1.1, 0.020);
    // The height field feathers out five times slower than the alpha does, so
    // the normal is not turning through 90° inside the 1.6 texels the alpha
    // test cuts through - that is what puts a hard lit rim around a cutout.
    const hEdge = edge * 5;
    const wobPhase = rand() * TAU;

    for (let y = 0; y < S; y++) {
      // Base at v=0, tip at v=1; the petal fills the card vertically.
      const t = (((y + 0.5) / S) - MARGIN) / (1 - 2 * MARGIN);
      for (let x = 0; x < S; x++) {
        const i = y * S + x;
        const ly = ((x + 0.5) / S) * 2 - 1;
        // Faint crimping of the margin; an exact analytic outline is the last
        // thing that reads as procedural once the colour is right.
        const wob = (Math.sin(t * 8.7 + wobPhase) * 0.6 + Math.sin(t * 19.3 + wobPhase * 2.1) * 0.4)
          * 0.013 * L * (ly < 0 ? -1 : 1);
        const dist = petalDist(t, ly, L, W, notch, W * 0.30) + wob;
        const alpha = clamp01(dist / edge + 0.5);
        const wt = petalWidth(t, W);
        const across = wt > 1e-4 ? clamp(ly / wt, -1, 1) : 0;

        let r, g, b;
        const gg = smoothstep(0.0, 0.68, t);
        if (gg < 0.5) {
          const k = gg * 2;
          r = lerp(cBase[0], cMidP[0], k); g = lerp(cBase[1], cMidP[1], k); b = lerp(cBase[2], cMidP[2], k);
        } else {
          const k = (gg - 0.5) * 2;
          r = lerp(cMidP[0], cTip[0], k); g = lerp(cMidP[1], cTip[1], k); b = lerp(cMidP[2], cTip[2], k);
        }

        // Veins, fanning with the blade.
        let vein = 0;
        if (wt > 1e-4) {
          // Nearest of five equally-spaced identical Gaussians; see the blossom.
          const rel = ly / wt;
          let q = Math.round(rel * 2.5);
          if (q > 2) q = 2; else if (q < -2) q = -2;
          const sN = (rel - q * 0.40) * wt / veinW;
          vein = Math.exp(-sN * sN)
            * smoothstep(0.05, 0.32, t) * (1 - smoothstep(0.88, 1.0, t));
        }
        r = lerp(r, cVein[0], vein * 0.26);
        g = lerp(g, cVein[1], vein * 0.26);
        b = lerp(b, cVein[2], vein * 0.26);

        // A faint bruise near the claw: petals detach by tearing, and a
        // spotless petal looks synthetic.
        const bk = smoothstep(0.72, 0.97, bruise[i]) * (1 - smoothstep(0.12, 0.5, t)) * 0.35;
        r = lerp(r, cBruise[0], bk); g = lerp(g, cBruise[1], bk); b = lerp(b, cBruise[2], bk);

        // Curl: one edge lifts more than the other, and the whole blade
        // twists slightly along its length.
        const twist = (t - 0.5) * 0.55 * curlDir;
        const c = clamp(across + twist, -1.2, 1.2);
        const h = 0.42 + c * c * 0.34 * curlDir * 0.5 + c * 0.16 * curlDir
          + smoothstep(0.55, 1.0, t) * 0.08 + vein * 0.05;

        // Bake the curl's own shading - the lifted edge catches more sky.
        const shade = 0.88 + 0.24 * clamp01(0.5 + c * 0.5 * curlDir);
        r *= shade; g *= shade; b *= shade;

        cover[i] = alpha;
        // Continuous, not a step at the silhouette: the old `alpha > 0.02 ? h :
        // 0.42` put a one-texel cliff exactly on the visible edge and the normal
        // map turned it into a bright wire around every falling petal.
        height[i] = lerp(0.42, h, clamp01(dist / hEdge + 0.5));
        colR[i] = r; colG[i] = g; colB[i] = b;
      }
    }

    this._bleedFloat(colR, colG, colB, cover, S, S, 4, [cMidP[0], cMidP[1], cMidP[2]]);

    const data = new Uint8Array(N * 4);
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = y * S + x;
        const o = i * 4;
        const d = dither(x, y);
        data[o] = q8(colR[i] / 255, d);
        data[o + 1] = q8(colG[i] / 255, d);
        data[o + 2] = q8(colB[i] / 255, d);
        data[o + 3] = q8(cover[i], d);
      }
    }

    const albedoTex = this._dataTexMips(data, S, S, {
      srgb: true, name: 'petalAlbedo', repeat: false,
      alphaWeighted: true, preserveCoverage: true, alphaTest: 0.5,
    });
    const chain = normalMipChain(height, S, S, { strength: 0.7, wrap: false });
    const normalTex = new THREE.DataTexture(chain.levels[0].data, S, S, THREE.RGBAFormat, THREE.UnsignedByteType);
    this._apply(normalTex, { srgb: false, repeat: false, mipmaps: true, anisotropy: this.maxAnisotropy });
    normalTex.mipmaps = chain.levels;
    normalTex.generateMipmaps = false;
    normalTex.name = 'petalNormal';
    normalTex.needsUpdate = true;
    this._register('petalAlbedo', albedoTex);
    this._register('petalNormal', normalTex);
    return { albedo: albedoTex, normal: normalTex, size: S };
  }

  /**
   * Grass blade atlas - four variants in four columns (`userData.atlas`).
   * RGB is the blade's own colour gradient and detail, A is the cutout.
   * Even if the grass system builds real blade geometry, this is the right
   * texture for its LOD billboards and for any single-quad distant fill.
   */
  grassBlades() {
    return this._group('grass', () => this._buildGrass(this._size('grass')));
  }

  _buildGrass(S) {
    const W = S, H = S;
    const N = W * H;
    const COLS = 4;
    const colW = W / COLS;
    const rand = makeRNG(this.seed + 9100);

    const cover = new Float32Array(N);
    const height = new Float32Array(N);
    const colR = new Float32Array(N);
    const colG = new Float32Array(N);
    const colB = new Float32Array(N);

    // Low-saturation, slightly grey-green: pampas and dry meadow, not lawn.
    const cRoot = rgb(0x3c4529);
    const cMid = rgb(0x5c6539);
    const cUpper = rgb(0x7b7c4e);
    const cDry = rgb(0x9a8f61);
    const cStraw = rgb(0xa89a6c);
    const cBrown = rgb(0x8a7550);

    // `u` spans -1..1 across a column, so one texel is 2/colW wide in u units.
    const edge = (2 / colW) * 1.5;

    for (let c = 0; c < COLS; c++) {
      const x0 = Math.round(c * colW);
      const x1 = Math.round((c + 1) * colW);
      const widthMul = 0.62 + 0.5 * rand();
      const bend = (rand() - 0.5) * 0.42;
      const dryK = rand();          // how strawy this variant is
      const tipCut = 0.90 + 0.09 * rand();
      const creaseOff = (rand() - 0.5) * 0.20;
      const striSeed = rand() * 100;
      const striN = 5 + (rand() * 4 | 0);      // 5-8 ribs across the blade
      // Where this blade's tip died back. Almost every blade in a real sward has
      // a browned, frayed or broken tip; four pristine blades is a tell.
      const dieBack = 0.55 + 0.42 * rand();
      const nickAt = 0.35 + 0.5 * rand();
      const nickDepth = rand() < 0.55 ? 0.18 + 0.3 * rand() : 0;
      const nickSide = rand() < 0.5 ? -1 : 1;

      for (let y = 0; y < H; y++) {
        // v = 0 at the root, 1 at the tip.
        const v = (y + 0.5) / H;
        // Blade half-width. A grass blade is not a triangle: it keeps most of
        // its width for the first half and then draws out into a long fine
        // point. `pow(., 0.55)` on the remaining length gives that profile;
        // the 0.72 the first draft used made a plain wedge.
        const taper = Math.pow(clamp01((tipCut - v) / tipCut), 0.55);
        // Sized to fill its column: the blade is the only thing in there, and
        // leaving half the column empty just wastes texels and fill rate.
        const halfW = 0.74 * widthMul * taper * (0.70 + 0.30 * Math.pow(1 - v, 0.35));
        // A bite out of ONE edge, where an insect or the weather took it.
        // Symmetric damage would just look like the blade narrowing.
        const nickF = nickDepth > 0
          ? nickDepth * Math.exp(-Math.pow((v - nickAt) / 0.045, 2))
          : 0;
        // Blades curve along their whole length, not just at the tip: `v*v`
        // keeps the lower two thirds dead straight, which is the giveaway.
        const cxOff = bend * v * (0.32 + 0.68 * v);

        for (let x = x0; x < x1; x++) {
          const u = ((x - x0) + 0.5) / colW * 2 - 1;   // -1..1 across the column
          const lx = u - cxOff;
          const i = y * W + x;
          const hw = (lx < 0 ? -1 : 1) === nickSide ? halfW * (1 - nickF) : halfW;
          const d = hw - Math.abs(lx);
          const alpha = v > tipCut + 0.02 ? 0 : clamp01(d / edge + 0.5);

          const across = halfW > 1e-5 ? clamp(lx / halfW, -1, 1) : 0;

          // Colour: dark at the root, olive through the middle, drying toward
          // the tip. Dryness rises with v and with the variant's own dryness.
          const dryness = clamp01(v * v * (0.35 + dryK * 0.9));
          let r, g, b;
          const t = clamp01(v * 1.15);
          if (t < 0.5) {
            const k = t * 2;
            r = lerp(cRoot[0], cMid[0], k); g = lerp(cRoot[1], cMid[1], k); b = lerp(cRoot[2], cMid[2], k);
          } else {
            const k = (t - 0.5) * 2;
            r = lerp(cMid[0], cUpper[0], k); g = lerp(cMid[1], cUpper[1], k); b = lerp(cMid[2], cUpper[2], k);
          }
          const dr = dryness * 0.9;
          r = lerp(r, cDry[0], dr); g = lerp(g, cDry[1], dr); b = lerp(b, cDry[2], dr);
          const strawK = smoothstep(dieBack, 1.0, v) * (0.45 + dryK * 0.5);
          r = lerp(r, cStraw[0], strawK); g = lerp(g, cStraw[1], strawK); b = lerp(b, cStraw[2], strawK);
          // The last few percent of a dead-back tip goes properly brown.
          const brownK = smoothstep(dieBack * 0.5 + 0.5, 1.0, v) * 0.55;
          r = lerp(r, cBrown[0], brownK); g = lerp(g, cBrown[1], brownK); b = lerp(b, cBrown[2], brownK);

          // Central crease: a blade folds along its midrib, so the middle is
          // a highlight and the two halves fall away. This is the whole reason
          // grass sparkles in raking light.
          const cr = across - creaseOff;
          const crease = Math.exp(-(cr * cr) * 22);
          const fold = 1 - Math.abs(cr) * 0.55;
          // A fine pale margin. Grass leaves are thinner and more translucent at
          // the edge, so they catch light there - it is a small thing that
          // separates a blade from a painted green sliver.
          const margin = Math.pow(clamp01(Math.abs(across)), 7) * 0.16;
          const shade = (0.80 + 0.30 * fold) * (0.94 + 0.16 * crease) + margin;

          // Longitudinal striations: continuous ribs running the length of the
          // blade. The first draft multiplied them by sin(v * 40), which is a
          // ladder of horizontal bands - a pattern grass does not have and the
          // eye finds immediately. The lengthwise variation is now a slow
          // envelope instead, so the ribs stay ribs.
          const stri = 0.965 + 0.045 * Math.cos(across * PI * striN + striSeed)
            * (0.55 + 0.45 * Math.sin(v * 2.7 + striSeed));

          r *= shade * stri; g *= shade * stri; b *= shade * stri;

          cover[i] = alpha;
          colR[i] = r; colG[i] = g; colB[i] = b;
          // Continuous across the silhouette - a step here draws a hard wire
          // down both edges of every blade once the normal map is applied.
          const hh = 0.35 + crease * 0.35 + (1 - Math.abs(cr)) * 0.22;
          height[i] = lerp(0.35, hh, clamp01(d / (edge * 4) + 0.5));
        }
      }
    }

    this._bleedFloat(colR, colG, colB, cover, W, H, 5, [cMid[0], cMid[1], cMid[2]]);

    const data = new Uint8Array(N * 4);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const o = i * 4;
        const d = dither(x, y);
        data[o] = q8(colR[i] / 255, d);
        data[o + 1] = q8(colG[i] / 255, d);
        data[o + 2] = q8(colB[i] / 255, d);
        data[o + 3] = q8(cover[i], d);
      }
    }

    const albedoTex = this._dataTexMips(data, W, H, {
      srgb: true, name: 'grassBlade', repeat: false,
      alphaWeighted: true, preserveCoverage: true, alphaTest: 0.5,
    });
    albedoTex.userData.atlas = { cols: COLS, rows: 1 };
    const chain = normalMipChain(height, W, H, { strength: 0.5, wrap: false });
    const normalTex = new THREE.DataTexture(chain.levels[0].data, W, H, THREE.RGBAFormat, THREE.UnsignedByteType);
    this._apply(normalTex, { srgb: false, repeat: false, mipmaps: true, anisotropy: this.maxAnisotropy });
    normalTex.mipmaps = chain.levels;
    normalTex.generateMipmaps = false;
    normalTex.name = 'grassBladeNormal';
    normalTex.needsUpdate = true;
    normalTex.userData.atlas = { cols: COLS, rows: 1 };
    this._register('grassBlade', albedoTex);
    this._register('grassBladeNormal', normalTex);
    return { albedo: albedoTex, normal: normalTex, cols: COLS, rows: 1, size: S };
  }

  // -------------------------------------------------------------------------
  // LEAF 4 - sky
  // -------------------------------------------------------------------------

  /**
   * Equirectangular star map with a physically-motivated population.
   *
   *  - Magnitudes follow log N ∝ 0.6 m (the real cumulative star count law),
   *    sampled by inverse transform, so the sky is dominated by faint stars
   *    with a handful of bright ones - not a uniform sprinkle.
   *  - Colours come from a B−V distribution converted to effective temperature
   *    with Ballesteros' formula and then through math.js's `kelvinToRGB`.
   *    The result is mostly white-blue with a scattering of amber giants.
   *  - Positions are laid out with `fibonacciSphere` plus jitter, so density is
   *    uniform per solid angle. Splat radii are widened by 1/sin(theta) toward
   *    the poles to cancel equirectangular stretch - without that, stars turn
   *    into vertical smears at the zenith, which is the classic tell.
   *  - Perceptual (0.45) magnitude compression keeps 8 faint magnitudes inside
   *    8-bit sRGB while still letting the brightest stars core out to white.
   */
  starMap() {
    return this.get('starMap', () => {
      const W = this._size('stars');
      const H = W >> 1;
      const N = W * H;
      // `let`, not `const`: at ULTRA this is 4096×2048, so the accumulator is
      // 100 MB and `mipChainRGBA` wants another 134 MB for its own float copy
      // of level 0. Dropping the reference before the chain is built keeps the
      // two peaks from overlapping, which on a machine with no dedicated VRAM
      // and a shared pool is the difference between a pause and a swap storm.
      let acc = new Float32Array(N * 3);
      const rand = makeRNG(this.seed + 31337);

      // --- Milky Way ------------------------------------------------------
      // Generated small and upsampled: it is a smooth glow, so full-res noise
      // would be pure waste on a 2048×1024 canvas.
      {
        const MW = 256, MH = 128;
        const clump = normalizeField(fbm2(MW, MH, { fx: 6, fy: 3, octaves: 5, gain: 0.55, seed: this.seed + 4242 }));
        const dust = normalizeField(fbm2(MW, MH, { fx: 9, fy: 4, octaves: 4, gain: 0.6, kind: 2, seed: this.seed + 4343 }));
        // Galactic plane tilted off the equator so it does not look like a
        // texture-space band.
        const tilt = 1.05, ct = Math.cos(tilt), st = Math.sin(tilt);
        const SIGMA = 0.16;
        // The band is a function of the direction's dot with a fixed pole, and
        // for a tilted pole in the y-z plane that collapses to
        //   dp = cos(theta)*ct + sin(theta)*sin(phi)*st
        // so only sin(phi) varies along a row. Hoisting it out of the inner
        // loop removes two transcendentals per texel across 2 M texels.
        const sinPhi = new Float32Array(W);
        for (let x = 0; x < W; x++) sinPhi[x] = Math.sin(((x + 0.5) / W) * TAU);
        // Beyond this |dp| the Gaussian is below the 1/255 visibility floor;
        // rejecting on a compare rather than evaluating exp() skips ~70% of the
        // canvas for free.
        const DP_CUT = SIGMA * Math.sqrt(2 * Math.log(400));
        const invTwoSig2 = 1 / (2 * SIGMA * SIGMA);
        // BILINEAR, not nearest. The glow field is 256x128 and the canvas is up
        // to 4096x2048, so a nearest tap replicates each source texel into a 16x
        // block - and the Milky Way is the one thing in the frame that is pure
        // low-frequency gradient, which is exactly where a replicated block is
        // visible. It quilted the whole band into 1.4-degree squares of sky, and
        // the squares survived into the mips because they are real signal.
        // Wrapped in x (the map is a full 360 degrees) and clamped in y.
        const scaleX = MW / W;
        for (let y = 0; y < H; y++) {
          const v = (y + 0.5) / H;
          const theta = PI * (1 - v);
          const A = Math.cos(theta) * ct;
          const B = Math.sin(theta) * st;
          const fy = v * MH - 0.5;
          const y0i = Math.floor(fy);
          const ty = fy - y0i;
          const r0 = clamp(y0i, 0, MH - 1) * MW;
          const r1 = clamp(y0i + 1, 0, MH - 1) * MW;
          const row3 = y * W * 3;
          for (let x = 0; x < W; x++) {
            const dp = A + B * sinPhi[x];
            if (dp > DP_CUT || dp < -DP_CUT) continue;
            const band = Math.exp(-dp * dp * invTwoSig2);
            const fx = x * scaleX - 0.5;
            const x0i = Math.floor(fx);
            const tx = fx - x0i;
            const c0 = ((x0i % MW) + MW) % MW;
            const c1 = (c0 + 1) % MW;
            const a00 = r0 + c0, a10 = r0 + c1, a01 = r1 + c0, a11 = r1 + c1;
            const cTop = clump[a00] + (clump[a10] - clump[a00]) * tx;
            const cBot = clump[a01] + (clump[a11] - clump[a01]) * tx;
            const cl = cTop + (cBot - cTop) * ty;
            const dTop = dust[a00] + (dust[a10] - dust[a00]) * tx;
            const dBot = dust[a01] + (dust[a11] - dust[a01]) * tx;
            const du = dTop + (dBot - dTop) * ty;
            const glow = band * (0.35 + 0.9 * cl) * (1 - 0.72 * du);
            const i3 = row3 + x * 3;
            // Faintly cool; the integrated light of the galaxy is bluish-white.
            acc[i3] += glow * 0.052;
            acc[i3 + 1] += glow * 0.056;
            acc[i3 + 2] += glow * 0.070;
          }
        }
      }

      // --- stars ------------------------------------------------------------
      const COUNT = W >= 2048 ? 9000 : 5200;
      const M_MIN = -1.4, M_MAX = 6.6;
      // Inverse transform for pdf ∝ 10^(0.6 m) on [M_MIN, M_MAX].
      const kA = Math.pow(10, 0.6 * M_MIN), kB = Math.pow(10, 0.6 * M_MAX);
      const dir = { x: 0, y: 0, z: 0 };

      for (let s = 0; s < COUNT; s++) {
        fibonacciSphere(s, COUNT, dir);
        // Jitter off the spiral: a Fibonacci lattice is *too* even for stars
        // and reads as a woven pattern at the zenith.
        const jx = (rand() - 0.5) * 0.030, jy = (rand() - 0.5) * 0.030, jz = (rand() - 0.5) * 0.030;
        let dxv = dir.x + jx, dyv = dir.y + jy, dzv = dir.z + jz;
        const dl = 1 / Math.hypot(dxv, dyv, dzv);
        dxv *= dl; dyv *= dl; dzv *= dl;

        const mag = Math.log10(kA + rand() * (kB - kA)) / 0.6;

        // B−V: three populations. Brighter stars skew hotter, which is a real
        // selection effect (luminous O/B stars are visible much further away).
        const pick = rand();
        let bv;
        const hotBias = clamp01((3.0 - mag) / 6.0) * 0.30;
        if (pick < 0.30 + hotBias) bv = clamp(rand.gaussian(0.02, 0.20), -0.34, 0.42);
        else if (pick < 0.72) bv = clamp(rand.gaussian(0.62, 0.22), 0.2, 1.05);
        else bv = clamp(rand.gaussian(1.25, 0.32), 0.75, 1.95);

        // Ballesteros 2012.
        const T = 4600 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62));
        const c = kelvinToRGB(T);
        const cmax = Math.max(c.r, c.g, c.b, 1e-5);
        const cr = c.r / cmax, cg = c.g / cmax, cb = c.b / cmax;

        // Perceptual compression: a linear flux ramp would put eight
        // magnitudes across a range of 1600:1, and the faint end would quantise
        // to zero in 8 bits. The 1.3 gain deliberately lets the brightest dozen
        // stars clip to a white core, which is exactly what they do to the eye.
        const amp = Math.pow(10, -0.4 * (mag - M_MIN) * 0.45) * 1.3;

        // Equirect placement.
        const theta = Math.acos(clamp(dyv, -1, 1));
        const v = 1 - theta / PI;
        const phi = Math.atan2(dzv, -dxv);
        const u = ((phi / TAU) % 1 + 1) % 1;
        const cx = u * W, cy = v * H;
        const sinT = Math.max(0.08, Math.sin(theta));

        // PSF: core sigma grows slowly with brightness; a faint star is one
        // texel, a first-magnitude star is a small disc with a halo.
        const sig = 0.62 + 1.15 * Math.pow(amp, 0.55);
        const sigX = Math.min(sig / sinT, sig * 6);
        const rx = Math.ceil(sigX * 3) + 1;
        const ry = Math.ceil(sig * 3) + 1;
        const ix = Math.round(cx), iy = Math.round(cy);

        for (let dy2 = -ry; dy2 <= ry; dy2++) {
          const yy = iy + dy2;
          if (yy < 0 || yy >= H) continue;
          const fy = (yy + 0.5 - cy) / sig;
          for (let dx2 = -rx; dx2 <= rx; dx2++) {
            const xx = ((ix + dx2) % W + W) % W;
            const fx = (xx + 0.5 - cx);
            // Wrap the horizontal distance the short way round.
            let fxw = fx;
            if (fxw > W * 0.5) fxw -= W;
            if (fxw < -W * 0.5) fxw += W;
            const fxn = fxw / sigX;
            const r2 = fxn * fxn + fy * fy;
            if (r2 > 9) continue;
            // Core plus a wide, weak halo - a pure Gaussian looks like a dot,
            // and the halo is what bloom picks up.
            const e = Math.exp(-r2 * 0.5) + 0.11 * Math.exp(-r2 * 0.09);
            const w = amp * e;
            const i3 = (yy * W + xx) * 3;
            acc[i3] += w * cr;
            acc[i3 + 1] += w * cg;
            acc[i3 + 2] += w * cb;
          }
        }

        // Diffraction spikes on the handful of genuinely bright stars.
        if (mag < 0.6) {
          const spikeLen = Math.round((3 + 9 * (1 - clamp01((mag + 1.4) / 2))) * (W / 2048) + 3);
          for (let k = 1; k <= spikeLen; k++) {
            const t2 = 1 - k / (spikeLen + 1);
            const falloff = amp * 0.16 * t2 * t2;
            const kx = Math.round(k / sinT);
            SPIKE_X[0] = ix + kx; SPIKE_Y[0] = iy;
            SPIKE_X[1] = ix - kx; SPIKE_Y[1] = iy;
            SPIKE_X[2] = ix; SPIKE_Y[2] = iy + k;
            SPIKE_X[3] = ix; SPIKE_Y[3] = iy - k;
            for (let q = 0; q < 4; q++) {
              const yy = SPIKE_Y[q];
              if (yy < 0 || yy >= H) continue;
              const xw = ((SPIKE_X[q] % W) + W) % W;
              const i3 = (yy * W + xw) * 3;
              acc[i3] += falloff * cr;
              acc[i3 + 1] += falloff * cg;
              acc[i3 + 2] += falloff * cb;
            }
          }
        }
      }

      // --- encode -----------------------------------------------------------
      // Stored sRGB-encoded so three's colour management hands the shader back
      // linear flux. Bright cores clip to white, which is what they do to the
      // eye anyway; the coloured wings survive.
      const data = new Uint8Array(N * 4);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const i = y * W + x;
          const i3 = i * 3;
          const o = i * 4;
          data[o + 3] = 255;
          const r = acc[i3], g = acc[i3 + 1], b = acc[i3 + 2];
          // Most of a night sky is genuinely black; skipping those texels is
          // worth more than any micro-optimisation of the encode itself.
          if (r <= 0 && g <= 0 && b <= 0) continue;
          const d = dither(x, y);
          data[o] = q8(linToSrgb(r), d);
          data[o + 1] = q8(linToSrgb(g), d);
          data[o + 2] = q8(linToSrgb(b), d);
        }
      }

      acc = null;                          // see the allocation note above

      const t = new THREE.DataTexture(data, W, H, THREE.RGBAFormat, THREE.UnsignedByteType);
      t.colorSpace = THREE.SRGBColorSpace;
      t.wrapS = THREE.RepeatWrapping;      // longitude wraps
      t.wrapT = THREE.ClampToEdgeWrapping; // latitude does not
      t.minFilter = THREE.LinearMipmapLinearFilter;
      t.magFilter = THREE.LinearFilter;
      // Hand-built, and this is the map where it matters most. A star map is a
      // near-black field with isolated bright points, and glGenerateMipmap on an
      // SRGB8_ALPHA8 texture is implementation-defined about whether it decodes
      // first. Filtering the *encoded* bytes of a point source overstates the
      // average by a large factor - the sky washes to grey the moment the dome
      // is minified, which is exactly when a night sky should be getting darker.
      t.mipmaps = mipChainRGBA(data, W, H, { srgb: true });
      t.generateMipmaps = false;
      t.anisotropy = this.maxAnisotropy;
      t.name = 'starMap';
      t.needsUpdate = true;
      return t;
    });
  }

  /** Exact linear→sRGB, for anything that is not in a per-texel loop. */
  _linToSrgb(v) {
    const c = v <= 0 ? 0 : v >= 1 ? 1 : v;
    return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  }

  /**
   * Rain streak atlas - eight variants in eight columns (`userData.atlas`).
   *
   * A pure intensity mask: all four channels carry the same value, so it works
   * as `map` (additive), as `alphaMap` (three samples `.g`) or as a raw
   * lookup, without the consumer having to know which. Deliberately untinted - 
   * rain takes the colour of the sky behind it, and baking a blue tint here
   * would fight the atmosphere system at sunset. Tiles vertically
   * (wrapT = Repeat) so a shader can scroll it without a seam.
   *
   * Real streaks are not uniform lines: a falling drop oscillates, so its
   * streak has bright and dim bands along its length, and the refracted core
   * is much brighter than the surrounding blur. Both are modelled here.
   */
  rainStreaks() {
    return this.get('rainStreak', () => {
      const W = this._size('rain');
      const H = W;
      const COLS = 8;
      const colW = W / COLS;
      const data = new Uint8Array(W * H * 4);
      const rand = makeRNG(this.seed + 5150);

      for (let c = 0; c < COLS; c++) {
        const x0 = Math.round(c * colW);
        const x1 = Math.round((c + 1) * colW);
        const sigma = colW * (0.055 + 0.055 * rand());
        const wander = (rand() - 0.5) * colW * 0.16;
        const oscF = 5 + rand() * 9;
        const oscP = rand() * TAU;
        // A falling drop oscillates, so its streak is brighter and dimmer
        // along its length - but only by a fifth or so. At 0.38 the streak beads
        // up into a dotted line, which is a shutter artefact, not rain.
        const oscA = 0.08 + 0.15 * rand();
        const bright = 0.72 + 0.28 * rand();
        const headV = 0.06 + 0.08 * rand();   // where the drop head sits

        // Half a column, less a texel: the widest the streak's halo may reach
        // before it starts writing into the neighbouring variant. The halo has
        // a sigma of about 4 texels and the column is only 32 wide at HIGH, so
        // without this window a bright streak leaves a visible ghost down the
        // edge of the atlas cell next door - and the ghost survives into the
        // mips, where the two cells are averaged together anyway.
        const halfCol = (x1 - x0) * 0.5 - 1;

        for (let y = 0; y < H; y++) {
          const v = (y + 0.5) / H;
          // Asymmetric taper. A motion-blurred drop is not symmetric: the
          // leading edge is where the drop actually is and arrives abruptly,
          // while the tail is the exposure trail and dies away slowly. The
          // exponent on the two halves is what encodes that, and `sin^0.42`
          // both ends - which is what this used to be - reads as a lozenge.
          const ends = v < headV
            ? Math.pow(clamp01(v / Math.max(headV, 1e-3)), 0.85)
            : Math.pow(clamp01((1 - v) / (1 - headV)), 0.38);
          // Drop oscillation along the streak.
          const osc = 1 - oscA * (0.5 - 0.5 * Math.cos(v * oscF * TAU + oscP));
          // The head (leading drop) is denser than the tail.
          const head = 1 + 1.25 * Math.exp(-Math.pow((v - headV) / 0.05, 2));
          const cxp = (x0 + x1) * 0.5 + wander * Math.sin(v * PI * 1.7);
          const amp = ends * osc * head * bright;

          for (let x = x0; x < x1; x++) {
            const dx = (x + 0.5) - cxp;
            const n = dx / sigma;
            // Bright refractive core plus a soft outer blur.
            const core = Math.exp(-n * n * 0.5);
            const halo = Math.exp(-n * n * 0.055) * 0.20;
            // Smooth window to zero at the cell boundary.
            const win = smoothstep(halfCol, halfCol * 0.55, Math.abs(dx));
            const a = clamp01(amp * (core * 0.86 + halo) * win);
            const o = (y * W + x) * 4;
            const b = q8(a, dither(x, y));
            data[o] = b; data[o + 1] = b; data[o + 2] = b; data[o + 3] = b;
          }
        }
      }

      const t = new THREE.DataTexture(data, W, H, THREE.RGBAFormat, THREE.UnsignedByteType);
      // A mask, not a colour: no sRGB decode, or the falloff curve gets bent.
      t.colorSpace = THREE.NoColorSpace;
      t.wrapS = THREE.ClampToEdgeWrapping;
      t.wrapT = THREE.RepeatWrapping;
      t.minFilter = THREE.LinearMipmapLinearFilter;
      t.magFilter = THREE.LinearFilter;
      t.generateMipmaps = true;
      t.anisotropy = this.maxAnisotropy;
      t.name = 'rainStreak';
      t.userData.atlas = { cols: COLS, rows: 1 };
      t.needsUpdate = true;
      return t;
    });
  }

  /**
   * Splash ripple: an expanding-ring normal + alpha for raindrop impacts.
   * RGB = tangent-space normal of the ring crest, A = ring alpha.
   * Animate by scaling the quad and fading A; the normal stays correct because
   * the crest profile is scale-invariant.
   */
  rainRipple() {
    return this.get('rainRipple', () => {
      const S = this._size('ripple');
      const N = S * S;
      const height = new Float32Array(N);
      const alpha = new Float32Array(N);
      const rand = makeRNG(this.seed + 5250);
      // Two concentric crests - a splash is a primary ring with a weaker
      // secondary chasing it - plus a slight ellipticity so it is not a stencil.
      const ecc = 1 + (rand() - 0.5) * 0.08;
      for (let y = 0; y < S; y++) {
        const cy = ((y + 0.5) / S) * 2 - 1;
        for (let x = 0; x < S; x++) {
          const cx = (((x + 0.5) / S) * 2 - 1) * ecc;
          const r = Math.hypot(cx, cy);
          const i = y * S + x;
          const w1 = Math.exp(-Math.pow((r - 0.80) / 0.085, 2));
          const w2 = Math.exp(-Math.pow((r - 0.52) / 0.11, 2)) * 0.45;
          const trough = -Math.exp(-Math.pow((r - 0.66) / 0.07, 2)) * 0.30;
          height[i] = 0.5 + (w1 + w2 + trough) * 0.5;
          // Alpha fades to nothing at the quad edge so no hard square shows.
          alpha[i] = clamp01((w1 + w2 * 0.8) * 1.15) * clamp01((1 - r) * 4);
        }
      }
      const nrm = this.normalFromHeight(height, S, S, { strength: 0.55, wrap: false });
      for (let i = 0; i < N; i++) nrm[i * 4 + 3] = Math.min(255, (alpha[i] * 255) | 0);
      const t = new THREE.DataTexture(nrm, S, S, THREE.RGBAFormat, THREE.UnsignedByteType);
      this._apply(t, { srgb: false, repeat: false, mipmaps: true });
      t.name = 'rainRipple';
      t.needsUpdate = true;
      return t;
    });
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------

  /** Dilate float RGB outward across transparent texels; see `_alphaBleed`. */
  _bleedFloat(r, g, b, a, w, h, passes, fallback) {
    const filled = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) filled[i] = a[i] > 0.004 ? 1 : 0;
    const next = new Uint8Array(w * h);
    for (let p = 0; p < passes; p++) {
      next.set(filled);
      let changed = 0;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const i = y * w + x;
          if (filled[i]) continue;
          let sr = 0, sg = 0, sb = 0, n = 0;
          for (let dy = -1; dy <= 1; dy++) {
            const yy = y + dy;
            if (yy < 0 || yy >= h) continue;
            for (let dx = -1; dx <= 1; dx++) {
              const xx = x + dx;
              if (xx < 0 || xx >= w || (!dx && !dy)) continue;
              const j = yy * w + xx;
              if (!filled[j]) continue;
              sr += r[j]; sg += g[j]; sb += b[j]; n++;
            }
          }
          if (n) {
            r[i] = sr / n; g[i] = sg / n; b[i] = sb / n;
            next[i] = 1;
            changed++;
          }
        }
      }
      filled.set(next);
      if (!changed) break;
    }
    for (let i = 0; i < w * h; i++) {
      if (!filled[i]) { r[i] = fallback[0]; g[i] = fallback[1]; b[i] = fallback[2]; }
    }
  }

  /**
   * As `_dataTex`, but with the mip chain built here rather than by the driver.
   * `opts.levels` may carry a chain already produced by `mipChainRGBA`/
   * `ormMipChain`; otherwise one is generated with the given options.
   */
  _dataTexMips(data, w, h, opts) {
    const t = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.UnsignedByteType);
    this._apply(t, {
      srgb: !!opts.srgb,
      repeat: opts.repeat !== false,
      mipmaps: true,
      anisotropy: opts.anisotropy ?? this.maxAnisotropy,
    });
    t.mipmaps = opts.levels || mipChainRGBA(data, w, h, opts);
    t.generateMipmaps = false;
    if (opts.name) t.name = opts.name;
    t.needsUpdate = true;
    return t;
  }

  /**
   * Normal map plus the ORM whose roughness absorbs the normal detail every mip
   * level throws away. Returned together because the roughness chain is only
   * correct if it is built from *this* normal chain's variance.
   *
   * @param {Float32Array} height 0..1 height field
   * @param {Uint8Array} orm level-0 ORM bytes
   * @returns {{normal:THREE.Texture, orm:THREE.Texture}}
   */
  _pbrTextures(height, orm, w, h, opts = {}) {
    const chain = normalMipChain(height, w, h, {
      strength: opts.strength ?? 1,
      wrap: opts.repeat !== false,
    });
    const normal = new THREE.DataTexture(chain.levels[0].data, w, h, THREE.RGBAFormat, THREE.UnsignedByteType);
    this._apply(normal, {
      srgb: false,
      repeat: opts.repeat !== false,
      mipmaps: true,
      anisotropy: opts.anisotropy ?? this.maxAnisotropy,
    });
    normal.mipmaps = chain.levels;
    normal.generateMipmaps = false;
    normal.name = opts.normalName || 'normal';
    normal.needsUpdate = true;

    const ormTex = this._dataTexMips(orm, w, h, {
      srgb: false,
      repeat: opts.repeat !== false,
      name: opts.ormName || 'orm',
      levels: ormMipChain(orm, w, h, chain.sigma2),
    });
    return { normal, orm: ormTex };
  }

  /** Build + configure + register a 2D DataTexture in one step. */
  _dataTex(data, w, h, opts) {
    const t = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.UnsignedByteType);
    this._apply(t, {
      srgb: !!opts.srgb,
      repeat: opts.repeat !== false,
      mipmaps: opts.mipmaps !== false,
      anisotropy: opts.anisotropy ?? this.maxAnisotropy,
    });
    if (opts.name) t.name = opts.name;
    t.needsUpdate = true;
    return t;
  }

  _make3D(data, n, opts = {}) {
    const t = new THREE.Data3DTexture(data, n, n, n);
    t.format = THREE.RGBAFormat;
    t.type = THREE.UnsignedByteType;
    // Trilinear and repeating in all three axes: the volume is built to tile,
    // and a cloud raymarch samples it far outside 0..1.
    t.minFilter = THREE.LinearFilter;
    t.magFilter = THREE.LinearFilter;
    t.wrapS = t.wrapT = t.wrapR = THREE.RepeatWrapping;
    t.generateMipmaps = false;
    t.colorSpace = THREE.NoColorSpace;
    t.unpackAlignment = 1;
    t.name = opts.name || 'noise3D';
    t.needsUpdate = true;
    return t;
  }

  _apply(t, o = {}) {
    if (o.srgb === true) t.colorSpace = THREE.SRGBColorSpace;
    else if (o.srgb === false) t.colorSpace = THREE.NoColorSpace;
    if (o.repeat === false) t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
    else if (o.repeat === true) t.wrapS = t.wrapT = THREE.RepeatWrapping;
    if (o.wrapS) t.wrapS = o.wrapS;
    if (o.wrapT) t.wrapT = o.wrapT;
    const mips = o.mipmaps !== false;
    // A hand-built chain always wins. `_apply` is called again on rebake with
    // the recipe's stored options, and turning `generateMipmaps` back on for a
    // texture that already carries `mipmaps` makes three ignore the chain and
    // ask the driver to filter sRGB bytes in the encoded domain - the exact
    // failure SECTION 2b exists to avoid.
    t.generateMipmaps = mips && !(t.mipmaps && t.mipmaps.length > 1);
    t.minFilter = o.minFilter || (mips ? THREE.LinearMipmapLinearFilter : THREE.LinearFilter);
    t.magFilter = o.magFilter || THREE.LinearFilter;
    if (o.anisotropy !== undefined) t.anisotropy = Math.min(this.maxAnisotropy, o.anisotropy);
    else if (mips) t.anisotropy = this.maxAnisotropy;
    if (o.name) t.name = o.name;
    t.needsUpdate = true;
    return t;
  }

  _register(name, tex, t0) {
    if (!tex) return tex;
    if (!tex.name) tex.name = name;
    this.cache.set(name, tex);
    this.stats.textures++;
    const img = tex.image;
    if (img && img.width) {
      const d = img.depth || 1;
      // Every tiled surface in this file ships a *hand-built* chain, which means
      // `generateMipmaps` is false on exactly the textures that do have mips.
      // Testing that flag alone under-reported the whole suite by a third.
      const mipped = tex.generateMipmaps || (tex.mipmaps && tex.mipmaps.length > 1);
      this.stats.bytes += img.width * img.height * d * 4 * (mipped ? 1.34 : 1);
    }
    if (t0 !== undefined) {
      const ms = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
      this.stats.ms += ms;
      if (this.verbose) console.info(`[TextureFactory] ${name} ${ms.toFixed(1)}ms`);
    }
    return tex;
  }

  _group(key, builder) {
    let g = this._groups.get(key);
    if (!g) {
      const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
      try {
        g = builder();
      } catch (err) {
        console.error(`[TextureFactory] group "${key}" failed:`, err);
        const fb = makeFallback(key);
        g = { albedo: fb, normal: fb, orm: fb, roughness: fb, ao: fb };
      }
      const ms = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
      this.stats.ms += ms;
      if (this.verbose) console.info(`[TextureFactory] group ${key} ${ms.toFixed(1)}ms`);
      this._groups.set(key, g);
    }
    return g;
  }

  /** Name → generator, so `get('barkAlbedo')` works with no generator argument. */
  _makeBuiltins() {
    const m = new Map();
    const add = (names, fn) => {
      for (const n of names) m.set(normKey(n), fn);
    };
    add(['bluenoise', 'blue', 'bn'], () => this.blueNoise());
    add(['cloudshape', 'perlinworley', 'cloudnoise', 'cloudshape3d'], () => this.perlinWorley3D());
    add(['clouddetail', 'clouderosion', 'clouddetail3d'], () => this.cloudDetail3D());
    add(['curlnoise', 'curl', 'curl2d'], () => this.curlNoise2D());
    add(['noise2d', 'utilitynoise', 'noise'], () => this.get('noise2Ddefault', () => this.noise2D()));

    add(['bark', 'barkalbedo', 'barkdiffuse', 'barkcolor', 'barkcolour'], () => this.barkSet().albedo);
    add(['barknormal'], () => this.barkSet().normal);
    add(['barkorm', 'barkroughness', 'barkao', 'barkrough'], () => this.barkSet().orm);

    add(['ground', 'groundalbedo', 'soil', 'soilalbedo', 'earth'], () => this.groundSet().albedo);
    add(['groundnormal', 'soilnormal'], () => this.groundSet().normal);
    add(['groundorm', 'groundroughness', 'groundao', 'soilorm'], () => this.groundSet().orm);
    add(['groundmacro', 'macrovariation'], () => this.groundMacro());

    // world/terrain.js's three maps. These names are in FACTORY_OWNED, so they
    // win over the generator terrain.js passes as its fallback. 'macro' aliases
    // here rather than to groundMacro: the two have different channel meanings
    // (groundMacro is R value / G moss / B damp / A litter, this is R dry /
    // G lush / B moss / A mottle) and the terrain shader is the only consumer
    // that ever asked for a map called "macro".
    add(['terraindetail', 'terrainsplat', 'grounddetail'], () => this.terrainSurface().detail);
    add(['terrainnormalao', 'terrainnormal', 'groundmicronormal'], () => this.terrainSurface().normalAO);
    add(['terrainmacro', 'macro'], () => this.terrainSurface().macro);

    add(['moss', 'mossalbedo', 'undergrowth'], () => this.mossSet().albedo);
    add(['mossnormal'], () => this.mossSet().normal);
    add(['mossorm', 'mossroughness', 'mossao'], () => this.mossSet().orm);

    add(['petallitter', 'litter', 'fallenpetals'], () => this.petalLitter().albedo);
    add(['petallitternormal', 'litternormal'], () => this.petalLitter().normal);

    add(['blossom', 'blossomalbedo', 'flower', 'sakuraflower'], () => this.blossomCard().albedo);
    add(['blossomnormal', 'flowernormal'], () => this.blossomCard().normal);
    add(['petal', 'petalalbedo', 'petalcard'], () => this.petalCard().albedo);
    add(['petalnormal'], () => this.petalCard().normal);

    add(['grass', 'grassblade', 'grassalbedo', 'blade'], () => this.grassBlades().albedo);
    add(['grassbladenormal', 'grassnormal'], () => this.grassBlades().normal);

    add(['stars', 'starmap', 'starfield', 'night'], () => this.starMap());
    add(['rain', 'rainstreak', 'rainstreaks'], () => this.rainStreaks());
    add(['rainripple', 'splash', 'ripple'], () => this.rainRipple());
    return m;
  }

  // -------------------------------------------------------------------------
  // Quality / lifecycle
  // -------------------------------------------------------------------------

  /**
   * Accepts either a tier string or the `state.quality` object.
   *
   * Anything already baked is rebuilt *in place* - the THREE.Texture objects
   * keep their identity, so every material holding a reference picks up the new
   * resolution without being told. Dropping to LOW genuinely halves the 2D maps
   * and roughly halves the cloud volume's linear size (an 8× memory cut), which
   * is exactly the kind of saving that matters on an iGPU with no dedicated VRAM.
   */
  onQualityChange(quality) {
    const tier = typeof quality === 'string' ? quality : quality?.tier;
    if (!tier) return;
    // `_lastBakedTier`, not `this.tier`: when this factory holds a live
    // reference to the same `state.quality` the manager writes, that field has
    // already changed by the time the event is delivered, so comparing against
    // it reports "nothing moved" and the rebake never happens. Null means
    // nothing has been baked yet, so there is nothing to rebuild either.
    const before = this._lastBakedTier;
    this.tier = tier;
    if (!before || before === tier) return;
    const kinds = ['bark', 'ground', 'moss', 'litter', 'macro', 'blossom', 'petal',
      'grass', 'stars', 'rain', 'noise2d', 'cloudShape', 'cloudDetail',
      'terrain', 'terrainMacro'];
    // Rebaking costs hundreds of milliseconds, so only do it when the size
    // policy actually moved for something already baked. Several tier changes
    // (medium↔low for most maps) change nothing at all. Both sides are asked
    // for explicitly so neither reads the live source mid-comparison.
    const changed = kinds.some((k) => this._size(k, tier) !== this._size(k, before));
    if (!changed) return;
    if (this.verbose) console.info(`[TextureFactory] rebake ${before} -> ${tier}`);
    this.rebake();
  }

  setQuality(tier) {
    this.onQualityChange(tier);
  }

  /**
   * Rebuild every baked texture at the current tier's sizes, reusing the
   * existing Texture objects. `dispose()` on each first so the driver releases
   * the old storage before the new upload - three allocates immutable texture
   * storage, so an in-place resize without a dispose would be ignored.
   */
  rebake() {
    const oldGroups = this._groups;
    const oldCache = this.cache;
    this._groups = new Map();
    this.cache = new Map();
    this.stats.textures = 0;
    this.stats.bytes = 0;

    // A group exposes the same Texture under several keys (`orm`, `roughness`
    // and `ao` are one object) and again through the cache, so without this the
    // ground ORM was disposed and re-adopted four times per rebake - four
    // redundant GPU deletes and re-uploads of a megabyte.
    const adopted = new Set();

    const adopt = (target, source) => {
      if (!target || !source || target === source) return;
      if (adopted.has(target)) return;
      adopted.add(target);
      target.dispose();
      target.image = source.image;
      target.format = source.format;
      target.type = source.type;
      target.colorSpace = source.colorSpace;
      target.wrapS = source.wrapS;
      target.wrapT = source.wrapT;
      if (source.wrapR !== undefined) target.wrapR = source.wrapR;
      target.minFilter = source.minFilter;
      target.magFilter = source.magFilter;
      // Hand-built chains live on `mipmaps`; forgetting this leaves the adopted
      // texture pointing at the *old* tier's levels while its base level is the
      // new size, which GL rejects outright.
      target.mipmaps = source.mipmaps;
      target.generateMipmaps = source.generateMipmaps;
      target.anisotropy = source.anisotropy;
      target.userData = source.userData;
      target.needsUpdate = true;
    };

    // Rebuild groups first; they populate the cache with their members.
    for (const key of oldGroups.keys()) {
      let rebuilt = null;
      switch (key) {
        case 'bark': rebuilt = this._group('bark', () => this._buildBark(this._size('bark'))); break;
        case 'ground': rebuilt = this._group('ground', () => this._buildGround(this._size('ground'))); break;
        case 'moss': rebuilt = this._group('moss', () => this._buildMoss(this._size('moss'))); break;
        case 'petalLitter': rebuilt = this._group('petalLitter', () => this._buildPetalLitter(this._size('litter'))); break;
        case 'blossom': rebuilt = this._group('blossom', () => this._buildBlossom(this._size('blossom'))); break;
        case 'petal': rebuilt = this._group('petal', () => this._buildPetal(this._size('petal'))); break;
        case 'grass': rebuilt = this._group('grass', () => this._buildGrass(this._size('grass'))); break;
        // Rebuilt as a group so the three maps keep their Texture identity - 
        // world/terrain.js holds direct references to all three in `_tex` and
        // never asks for them again after init().
        case 'terrainSurface': rebuilt = this._group('terrainSurface', () => this._buildTerrainSurface(this._size('terrain'))); break;
        default: break;
      }
      const old = oldGroups.get(key);
      if (rebuilt && old) {
        for (const k of Object.keys(old)) {
          if (old[k] && old[k].isTexture) adopt(old[k], rebuilt[k]);
        }
        // The non-texture members describe the bake, and the bake just changed.
        // Keeping the old object without refreshing them left `size` and the
        // CPU-side `height` field describing the *previous* tier: after a HIGH
        // -> ULTRA change `groundSet()` reported size 512 with a 512² height
        // array while its textures were 1024², so anything sampling the height
        // field by `y * size + x` read the wrong surface (and would read out of
        // bounds if it trusted the texture's own dimensions instead).
        for (const k of Object.keys(rebuilt)) {
          if (rebuilt[k] && rebuilt[k].isTexture) continue;
          old[k] = rebuilt[k];
        }
        // Keep the original objects as the canonical ones.
        this._groups.set(key, old);
      }
    }

    // Then anything baked through get() with a stored recipe.
    for (const [name, old] of oldCache) {
      if (this.cache.has(name)) {
        adopt(old, this.cache.get(name));
        this.cache.set(name, old);
        continue;
      }
      const recipe = this._recipes.get(name);
      if (recipe) {
        try {
          const fresh = recipe.gen();
          if (recipe.opts) this._apply(fresh, recipe.opts);
          adopt(old, fresh);
        } catch (err) {
          console.error(`[TextureFactory] rebake of "${name}" failed:`, err);
        }
        this.cache.set(name, old);
        continue;
      }
      const b = this._builtins.get(normKey(name));
      if (b) {
        try {
          const fresh = b();
          adopt(old, fresh);
        } catch (err) {
          console.error(`[TextureFactory] rebake of built-in "${name}" failed:`, err);
        }
      }
      this.cache.set(name, old);
    }
  }

  /** Human-readable summary; handy from devtools while budgeting the load. */
  report() {
    return {
      tier: this.tier,
      textures: this.cache.size,
      approxMB: +(this.stats.bytes / (1024 * 1024)).toFixed(2),
      bakeMs: +this.stats.ms.toFixed(1),
      names: Array.from(this.cache.keys()),
    };
  }

  dispose() {
    if (this._offQuality) { this._offQuality(); this._offQuality = null; }
    for (const t of this.cache.values()) t?.dispose?.();
    for (const g of this._groups.values()) {
      for (const k of Object.keys(g)) g[k]?.dispose?.();
    }
    this.cache.clear();
    this._groups.clear();
    this._recipes.clear();
    this.stats.textures = 0;
    this.stats.bytes = 0;
  }
}
