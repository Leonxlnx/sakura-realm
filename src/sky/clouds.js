/**
 * sky/clouds.js - VolumetricClouds
 *
 * A raymarched cloud layer built for an integrated GPU. The whole design is
 * organised around one number: how many density samples per screen pixel per
 * second we can afford. Everything else follows from that.
 *
 *   trace   -> a small RGBA16F buffer, one ray per texel, checkerboarded so only
 *              1/4 of the half-res pixels are traced on any given frame
 *   resolve -> reprojects the previous half-res frame with the previous camera
 *              matrices, variance-clips it against this frame's trace, writes the
 *              new half-res history
 *   shadow  -> a small top-down map: for each ground texel, march toward the sun
 *              through the slab. Terrain/grass read this so clouds visibly drag
 *              their shadows across the field.
 *   composite -> a screen-aligned triangle at the far plane inside the main scene,
 *              premultiplied-alpha blended. Depth testing gives pixel-exact
 *              occlusion by terrain and the tree for free, at full resolution,
 *              even though the clouds themselves are half res.
 *
 * The density field is Schneider-style (Nubis / "The Real-Time Volumetric
 * Cloudscapes of Horizon Zero Dawn"): a tiling perlin-worley base eroded by a
 * high-frequency worley detail volume, shaped vertically by per-type height
 * gradients and horizontally by a weather map. Lighting is Hillaire-style
 * energy-conserving integration with a multiple-scattering octave loop.
 *
 * Both 3D volumes are baked on the GPU at init into a tiled 2D target, read back
 * once, and uploaded as Data3DTextures - no assets, no per-frame cost, and it
 * avoids the multi-second stall a JS-side worley bake would cost.
 */

import * as THREE from 'three';
import { clamp, clamp01, damp, lerp, makeRNG, smoothstep, TAU } from '../core/math.js';
import { QUALITY } from '../core/state.js';

// ---------------------------------------------------------------------------
// Bake resolutions. Shape is the expensive one: 128^3 RGBA8 = 8 MB, which the
// 780M can hold comfortably and which gives ~4 texels per worley cell at the
// finest channel - enough to stay free of cell-boundary aliasing.
// ---------------------------------------------------------------------------
const SHAPE_SIZE = 128;
const SHAPE_TILES_X = 16;
const SHAPE_TILES_Y = 8;
const DETAIL_SIZE = 64;
const DETAIL_TILES_X = 8;
const DETAIL_TILES_Y = 8;
const WEATHER_SIZE = 256;
const BLUENOISE_SIZE = 64;

/** Segments in the coverage inverse-CDF table. 64 puts each segment at 1.6% of
 *  sky, which is finer than the dial is ever driven. */
const COV_LUT_STEPS = 64;

/** Artistically compressed planet radius: clouds curve down to the horizon at
 *  ~21 km instead of ~120 km, which is what makes a cloud layer read as a layer
 *  rather than an infinite ceiling. */
const PLANET_RADIUS = 260000;

/** Checkerboard divisor: 1 traced pixel per DIV x DIV block of the history
 *  buffer, so a full refresh takes DIV*DIV frames. */
const CHECKER_DIV = 2;
/** Bayer-2 visiting order - consecutive frames land on diagonally opposite
 *  texels, which converges much faster than a raster scan. */
const CHECKER_ORDER = [0, 0, 1, 1, 1, 0, 0, 1];

// ===========================================================================
// GLSL
// ===========================================================================

const FS_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

/** Composite vertex: force the fragment to the far plane so the depth test does
 *  the occlusion work against whatever the scene already rendered. */
const COMPOSITE_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 1.0, 1.0);
}
`;

// ---------------------------------------------------------------------------
// One-time volume bake
// ---------------------------------------------------------------------------

const BAKE_FRAG = /* glsl */ `
uniform float uSize;
uniform float uTilesX;
uniform float uMode;   // 0 = shape volume, 1 = detail volume

vec3 hash33(vec3 p) {
  p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
           dot(p, vec3(269.5, 183.3, 246.1)),
           dot(p, vec3(113.5, 271.9, 124.6)));
  return fract(sin(p) * 43758.5453123) * 2.0 - 1.0;
}

// Tileable gradient noise. The mod() on the lattice index is what makes the
// volume wrap seamlessly, which is non-negotiable: a seam in a tiling cloud
// volume shows up as a straight line across the sky.
float gdot(vec3 i, vec3 o, vec3 f, float freq) {
  return dot(hash33(mod(i + o, vec3(freq))), f - o);
}

float gradientNoise(vec3 p, float freq) {
  vec3 pf = p * freq;
  vec3 i = floor(pf);
  vec3 f = fract(pf);
  vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  float n000 = gdot(i, vec3(0.0, 0.0, 0.0), f, freq);
  float n100 = gdot(i, vec3(1.0, 0.0, 0.0), f, freq);
  float n010 = gdot(i, vec3(0.0, 1.0, 0.0), f, freq);
  float n110 = gdot(i, vec3(1.0, 1.0, 0.0), f, freq);
  float n001 = gdot(i, vec3(0.0, 0.0, 1.0), f, freq);
  float n101 = gdot(i, vec3(1.0, 0.0, 1.0), f, freq);
  float n011 = gdot(i, vec3(0.0, 1.0, 1.0), f, freq);
  float n111 = gdot(i, vec3(1.0, 1.0, 1.0), f, freq);
  float x00 = mix(n000, n100, u.x);
  float x10 = mix(n010, n110, u.x);
  float x01 = mix(n001, n101, u.x);
  float x11 = mix(n011, n111, u.x);
  return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z) * 0.5 + 0.5;
}

float perlinFBM(vec3 p, float freq) {
  float sum = 0.0, amp = 0.5, norm = 0.0;
  for (int i = 0; i < 4; i++) {
    sum += amp * gradientNoise(p, freq);
    norm += amp;
    freq *= 2.0;
    amp *= 0.5;
  }
  return sum / norm;
}

// Tileable worley. Returns distance to the nearest feature point, 0..~0.9.
float worley(vec3 p, float freq) {
  vec3 pf = p * freq;
  vec3 id = floor(pf);
  vec3 f = fract(pf);
  float minD = 8.0;
  for (int z = -1; z <= 1; z++) {
    for (int y = -1; y <= 1; y++) {
      for (int x = -1; x <= 1; x++) {
        vec3 o = vec3(float(x), float(y), float(z));
        vec3 h = hash33(mod(id + o, vec3(freq))) * 0.5 + 0.5;
        vec3 d = o + h - f;
        minD = min(minD, dot(d, d));
      }
    }
  }
  return sqrt(minD);
}

/** Inverted worley: 1 at cell centres, so it reads as billowy blobs. */
float billow(vec3 p, float freq) {
  return clamp(1.0 - worley(p, freq), 0.0, 1.0);
}

void main() {
  vec2 fc = floor(gl_FragCoord.xy);
  float tx = floor(fc.x / uSize);
  float ty = floor(fc.y / uSize);
  float slice = ty * uTilesX + tx;
  vec3 uvw = (vec3(mod(fc.x, uSize), mod(fc.y, uSize), slice) + 0.5) / uSize;

  if (uMode < 0.5) {
    // Shape volume. R = perlin-worley, GBA = three worley octaves the sampling
    // shader recombines into an FBM. Keeping the octaves separate lets the
    // runtime weight them by ALTITUDE - broad shoulders low in a cloud, tight
    // cauliflower granulation at the top - which is where the billow shape
    // actually comes from.
    float pf = perlinFBM(uvw, 4.0);
    float w4 = billow(uvw, 4.0);
    float w8 = billow(uvw, 8.0);
    float w16 = billow(uvw, 16.0);
    float wfbm = w4 * 0.625 + w8 * 0.25 + w16 * 0.125;
    // Perlin-Worley (Schneider): worley carves rounded lobes out of perlin's
    // connected filaments. Note this remap compresses the distribution into a
    // narrow band around 0.65 - the CPU percentile stretch that runs after the
    // readback is what pulls it back to a full 0..1 range. Skip that step and
    // every cloud in the sky comes out as one uniform grey value, which is
    // exactly the "flat fog" failure this volume used to produce.
    float pw = clamp((pf - (wfbm - 1.0)) / (2.0 - wfbm), 0.0, 1.0);
    // A little of the finest worley octave folded into the base channel. The
    // detail volume fades out with distance for cost reasons, so without this
    // the far half of the sky degenerates into smooth blobs; this granulation
    // is part of the SHAPE and therefore survives at any distance for free.
    pw = clamp(pw - (1.0 - w16) * 0.12, 0.0, 1.0);
    gl_FragColor = vec4(pw, w4, w8, w16);
  } else {
    // Detail volume: pure worley, used only to erode the shape's edges.
    float d4 = billow(uvw, 4.0);
    float d8 = billow(uvw, 8.0);
    float d16 = billow(uvw, 16.0);
    gl_FragColor = vec4(d4, d8, d16, 1.0);
  }
}
`;

// ---------------------------------------------------------------------------
// Shared density field + helpers, injected into the trace and shadow shaders so
// both agree exactly on what a cloud is. A mismatch here shows up as shadows
// that do not line up with the clouds casting them.
// ---------------------------------------------------------------------------

const FIELD_UNIFORMS_GLSL = /* glsl */ `
uniform sampler3D uShapeTex;
uniform sampler3D uDetailTex;
uniform sampler2D uWeatherTex;

uniform vec3  uPlanetCenter;
uniform float uInnerRadius;
uniform float uThickness;

/** Coverage threshold window, solved on the CPU from state.clouds.coverage. It
 *  used to be derived per sample - a pow() plus two mixes of pure uniforms,
 *  evaluated once for the primary sample and again for every one of the light
 *  march's steps and its long reach, i.e. ~9 redundant transcendentals per lit
 *  pixel-sample. The compiler cannot hoist it because it lives behind two
 *  function calls that also take varying arguments. Solving it once per frame on
 *  the CPU is exactly equivalent and is what pays for the extra shaping below. */
uniform float uCovLo;
uniform float uCovHi;
uniform float uDensity;
uniform float uErosion;
uniform float uTypeBias;
uniform float uAnvil;

uniform vec3  uShapeOffset;
uniform vec3  uDetailOffset;
uniform vec2  uWeatherOffset;
uniform float uShapeScale;
uniform float uDetailScale;
uniform float uWeatherScale;
uniform vec2  uWindDir;
uniform float uShear;
`;

const FIELD_GLSL = /* glsl */ `
float remap(float v, float a, float b, float c, float d) {
  return c + (v - a) * (d - c) / max(b - a, 1e-5);
}

/** 0 at the cloud base, 1 at the top; outside [0,1] means outside the slab. */
float heightFraction(vec3 p) {
  return (length(p - uPlanetCenter) - uInnerRadius) / uThickness;
}

vec4 sampleWeather(vec3 p) {
  vec2 uv = (p.xz + uWeatherOffset) * uWeatherScale;
  return texture(uWeatherTex, uv);
}

/**
 * Vertical profile of a cloud, from a flat base at h = 0 to its crown.
 *
 * This is ONE profile whose smoothstep EDGES are morphed by cloud type, not
 * three separately-evaluated profiles crossfaded together. That matters twice
 * over. Artistically, crossfading a stratus sheet (which ends at h = 0.30) with
 * a cumulus (which ends at h = 0.98) gives an intermediate type a DOUBLE
 * shoulder - a sheet with a ghost of a tower fading through it - instead of a
 * single cloud that is simply taller. Morphing the edges gives a continuous
 * family of real profiles. And it costs three smoothsteps instead of eight, on
 * a function that runs once for the primary sample plus once per light-march
 * step, which is where the budget for the extra shaping below comes from.
 *
 * The base is a near-discontinuity, and that is the point. Below the lifting
 * condensation level there is no cloud at all, so a real cumulus deck has a FLAT
 * bottom sitting at one altitude across the whole sky. Fading a cloud in over
 * the bottom tenth of the slab is the single biggest "CG cloud" tell there is.
 * The shoulder must stay a little wider than one fine march step or the shelf
 * becomes a sub-step discontinuity and stair-steps along the base. One step is
 * range/(CLOUD_STEPS*1.55), so it scales with the step count: ~13 m looking up
 * at 45 degrees on HIGH but ~32 m on LOW, where a fixed 18 m shoulder would be
 * half a step and would band visibly. BASE_SHOULDER carries that ratio in from
 * _defines(), so every tier gets the hardest base it can actually resolve.
 */
float heightGradient(float h, float type, float anvil) {
  float a = clamp(type * 2.0, 0.0, 1.0);        // stratus -> stratocumulus
  float b = clamp(type * 2.0 - 1.0, 0.0, 1.0);  // stratocumulus -> cumulus
  // Where the top rolloff starts and ends. stratus 0.10/0.30, stratocumulus
  // 0.34/0.66, cumulus 0.50/0.98, cumulonimbus the whole slab.
  float topA = mix(mix(0.10, 0.34, a), 0.50, b);
  float topB = mix(mix(0.30, 0.66, a), 0.98, b);
  topA = mix(topA, 0.90, anvil);
  topB = mix(topB, 1.00, anvil);

  float g = smoothstep(0.0, 0.025 * BASE_SHOULDER, h) * (1.0 - smoothstep(topA, topB, h));
  // The waist. A convective cloud is narrowest just above its base and swells
  // to its widest around 40% of the way up; only the convective types get it,
  // a stratus sheet has no waist at all.
  g *= mix(1.0, mix(0.86, 1.0, smoothstep(0.02, 0.42, h)), max(b, anvil));
  // ...and a thunderhead flares back out into its anvil near the top.
  g *= 1.0 + anvil * 0.45 * smoothstep(0.50, 0.86, h);
  return g;
}

/**
 * Liquid water content through the depth of a cloud. It climbs from the base to
 * roughly two thirds of the way up, then falls off as dry air entrained through
 * the top evaporates the crown.
 *
 * Note the base is DENSE, not thin. A cumulus base looks dark because almost no
 * sunlight survives the trip down through the body above it - not because there
 * is less water there. Modelling the darkness as thinness was why the old bases
 * read as grey haze instead of a hard shadowed shelf.
 */
float densityProfile(float h) {
  return mix(0.84, 1.0, smoothstep(0.0, 0.36, h)) * (1.0 - 0.32 * smoothstep(0.60, 1.0, h));
}

/**
 * Maps the weather field through the artistic coverage dial. 0 is genuinely
 * clear and 1 genuinely socked in, and the response in between is close to
 * linear - uCovLo/uCovHi are not guesses, they are read straight off the
 * INVERSE CDF of the field this function samples, so state.clouds.coverage
 * really does mean "fraction of sky with cloud". See _solveCoverage().
 */
float coverageFrom(vec4 w, float h) {
  float field = w.r * 0.66 + w.g * 0.34;
  float cov = smoothstep(uCovLo, uCovHi, field);
  // Anvils spread outward with altitude - the top of a thunderhead is wider
  // than its base, which is most of what sells a storm silhouette.
  cov = mix(cov, min(cov * 1.7, 1.0), uAnvil * w.a * smoothstep(0.55, 0.95, h));
  return cov;
}

/** Shear + weather-driven domain warp, shared by shape and detail so the two
 *  octaves stay locked to the same air mass while still parallaxing. */
vec3 warpPosition(vec3 p, float h) {
  return p + vec3(uWindDir.x, 0.0, uWindDir.y) * (h * uShear);
}

/**
 * Normalised cloud shape 0..1 from the low-frequency volume only. Erosion can
 * only ever subtract from this, which makes it a conservative upper bound the
 * raymarcher can use to skip empty space without stepping over a cloud.
 *
 * The weather sample is passed in rather than fetched, so the light march's near
 * steps can reuse the primary sample's - a ~1 km approximation on a field whose
 * features are several kilometres wide, and it removes six 2D fetches per lit
 * sample. Only the long reach, which travels far enough for that to stop being
 * true, fetches its own.
 */
float shapeNorm(vec3 p, float h, vec4 weather, out float coverage, out float grain) {
  coverage = 0.0;
  grain = 1.0;
  if (h <= 0.0 || h >= 1.0) return 0.0;
  coverage = coverageFrom(weather, h);
  if (coverage <= 0.002) return 0.0;

  float type = clamp(weather.b + uTypeBias, 0.0, 1.0);

  // The weather channels also warp the lookup, which hides the fact that the
  // shape volume repeats every few kilometres. Raised alongside the tighter
  // weather tile: more, smaller clouds means the shape volume's period sits
  // closer to the size of an individual cloud, so it needs more decorrelation.
  vec3 sp = warpPosition(p, h) * uShapeScale + uShapeOffset
          + vec3(weather.b, weather.a, weather.g) * 0.55;

  vec4 s = texture(uShapeTex, sp);
  // Handed back to densityFull() so it can erode the silhouette with the finest
  // worley octave WITHOUT a second fetch. It is returned rather than used here
  // because only the primary march decides a silhouette; making the light march
  // and the shadow map pay for it too would be seven eighths of the cost for
  // none of the benefit.
  grain = s.a;

  // CAULIFLOWER. The three worley octaves in GBA are already in this one fetch,
  // so weighting them by altitude is free: broad rounded shoulders down near the
  // base where a thermal is still one coherent bubble, tight granulation up top
  // where it has broken into a hundred competing turrets. A single fixed FBM
  // weighting - what this used to do - gives every part of the cloud the same
  // lump size, and that reads as noise rather than as convection.
  float hi = smoothstep(0.26, 0.90, h);
  float fbm = mix(s.g * 0.70 + s.b * 0.23 + s.a * 0.07,
                  s.g * 0.26 + s.b * 0.42 + s.a * 0.32, hi);

  // Worley carves lobes out of the perlin-worley base. The window is narrower
  // than the textbook remap(x, fbm-1, 1, ...): that version divides by ~1.5 and
  // shifts up by ~0.5, which crushes the whole field into the top third of its
  // range. This one keeps the slope near 1.25 so the shape actually spans 0..1
  // and clouds get an inside and an outside.
  //
  // The carving FLOOR then climbs with altitude, and that is the actual
  // cauliflower. Down at the base the floor is low, so the worley valleys barely
  // bite and the thermal stays one broad connected mass. Up in the crown the
  // floor is close to 1, so the same valleys cut nearly to zero and the mass
  // separates into individually rounded turrets with real gaps between them. A
  // constant floor gives the same lump depth everywhere, which reads as a dome
  // with noise on it rather than as boiling convection.
  //
  // The min() is a hard guarantee, not a taste call. remap(s.r, floor, 1, 0, 1)
  // has slope 1/(1-floor), so a floor that is free to reach 0.86 turns the
  // lookup into a near-binary threshold on a texel-scale signal - which aliases
  // into crawling speckle the temporal filter then smears. Capping the floor
  // caps the slope at ~3.1, which is as much contrast as a 128^3 volume sampled
  // at ~19 m steps can carry without breaking up.
  float carve = mix(0.66, 0.96, hi);
  float carveFloor = min(fbm * carve - 0.24 + 0.10 * hi, 0.68);
  float base = clamp(remap(s.r, carveFloor, 1.0, 0.0, 1.0), 0.0, 1.0);
  // Rounded billows, not fog: an S-curve pushes the midtones apart so a cloud
  // has a solid core and a distinct margin instead of one grey level everywhere.
  base = base * base * (3.0 - 2.0 * base);
  // A second, gentler S high in the cloud. Two stacked smoothsteps push the
  // midtones further apart than one steeper curve does without clipping either
  // end into a flat plateau - and a plateau is exactly what turns a crown back
  // into a smooth dome. Weighted by altitude so the base keeps its full range.
  base = mix(base, base * base * (3.0 - 2.0 * base), 0.55 * hi);
  // Stratus is a sheet, not a collection of blobs. As the type slides toward
  // stratus, flatten the contrast so a genuine overcast reads as solid overcast
  // and not as a ceiling full of holes.
  base = mix(base * 0.46 + 0.54, base, smoothstep(0.08, 0.44, type));
  base *= heightGradient(h, type, uAnvil * weather.a);

  return clamp((base - (1.0 - coverage)) / max(coverage, 1e-3), 0.0, 1.0);
}

/** Shape-only density - the cheap path, used for skipping, the light march and
 *  the shadow map. */
float densityLow(vec3 p, float h, vec4 weather) {
  float coverage, grain;
  float n = shapeNorm(p, h, weather, coverage, grain);
  return n * coverage * uDensity * densityProfile(h);
}

float densityLow(vec3 p, float h) {
  return densityLow(p, h, sampleWeather(p));
}

/**
 * Full density: the shape bitten back at its margins, in two stages.
 *
 * Only the PRIMARY march calls this, and that is deliberate - a silhouette is
 * decided by the ray that draws it, so making the light march and the shadow map
 * reproduce this work would be seven eighths of the cost for none of the look.
 *
 * Both stages share one rim term. Erosion bites at the MARGINS: the core of a
 * cumulus is solid liquid water and stays solid, and it is only the outer skin,
 * where the shape value is already low, that dry air entrains into and tears
 * apart. Applying a detail volume uniformly is what makes CG volumetrics look
 * like dirty cotton wool instead of cloud. The flat base is spared as well - 
 * shear and entrainment attack the top and the sides of a thermal, while the
 * condensation level itself stays a clean shelf.
 */
float densityFull(vec3 p, float h, vec4 weather, float erode) {
  float coverage, grain;
  float n = shapeNorm(p, h, weather, coverage, grain);
  if (n <= 0.0) return 0.0;

  float rim = 1.0 - smoothstep(0.07, 0.66, n);
  rim *= mix(0.32, 1.0, smoothstep(0.0, 0.17, h));

  // Stage 1 - the finest worley octave of the SHAPE volume, which came back from
  // the fetch shapeNorm already made, so this costs no bandwidth at all and
  // therefore survives at any distance. That matters: the detail volume fades
  // out with range for cost reasons, and most of the sky is beyond that fade,
  // which used to leave every distant cloud with a perfectly smooth silhouette
  // tracing the weather map's contour. A large, clean, hard-edged boundary with
  // no structure on it is the most obviously synthetic thing a cloud renderer
  // can draw, and it is exactly what the noon capture shows. Keeping the margins
  // ragged for free is also what lets the detail fade be pulled IN, which is a
  // net saving rather than a net cost.
  n = clamp(remap(n, (1.0 - grain) * rim * (0.14 + 0.46 * uErosion), 1.0, 0.0, 1.0), 0.0, 1.0);

  // Stage 2 - the real detail volume, for the near field only, where its
  // features are still larger than a pixel and larger than a march step.
  if (erode > 0.002) {
    vec3 dp = warpPosition(p, h) * uDetailScale + uDetailOffset;
    vec3 det = texture(uDetailTex, dp).rgb;
    float dfbm = det.r * 0.625 + det.g * 0.25 + det.b * 0.125;
    // Wispy torn shreds low down where shear drags cloud out sideways, rounded
    // granulation up top where the turrets are still boiling upward.
    float modifier = mix(1.0 - dfbm, dfbm, clamp(h * 3.0 - 0.30, 0.0, 1.0));

    // 1.45 rather than 1.7: stage 1 has already taken a bite here, so the near
    // field would otherwise be eroded appreciably harder than before and the
    // clouds would thin out. The two together subtract about what this one used
    // to on its own - but only this half fades with range.
    n = clamp(remap(n, modifier * uErosion * erode * rim * 1.45, 1.0, 0.0, 1.0), 0.0, 1.0);
  }

  // Every remap above has a >= 0 and can only reduce n, so densityLow() remains
  // a conservative upper bound and the raymarcher's coarse stride can never step
  // over a cloud.
  return n * coverage * uDensity * densityProfile(h);
}

/**
 * Ray vs sphere. Returns (near, far); x > y means no intersection. Used for the
 * two shells that bound the cloud slab.
 */
vec2 raySphere(vec3 ro, vec3 rd, vec3 ce, float r) {
  vec3 oc = ro - ce;
  float b = dot(oc, rd);
  float c = dot(oc, oc) - r * r;
  float disc = b * b - c;
  if (disc < 0.0) return vec2(1.0, -1.0);
  disc = sqrt(disc);
  return vec2(-b - disc, -b + disc);
}

/**
 * Entry/exit distance of the ray through the curved cloud slab. Handles the
 * camera below, inside and above the layer - the player can fly.
 * Returns false when the ray never meets the slab within maxDist.
 */
bool slabInterval(vec3 ro, vec3 rd, float outerRadius, float maxDist, out float t0, out float t1) {
  float camR = length(ro - uPlanetCenter);
  vec2 ti = raySphere(ro, rd, uPlanetCenter, uInnerRadius);
  vec2 to = raySphere(ro, rd, uPlanetCenter, outerRadius);
  t0 = 0.0; t1 = 0.0;

  if (camR < uInnerRadius) {
    // Below the layer: enter through the inner shell, leave through the outer.
    // Rays pointing down exit the inner shell on the far side of the planet,
    // which lands beyond maxDist and is rejected - exactly right.
    t0 = ti.y;
    t1 = to.y;
  } else if (camR < outerRadius) {
    t0 = 0.0;
    t1 = (ti.x > 0.0) ? ti.x : to.y;   // may drop out of the bottom
  } else {
    if (to.x > to.y || to.y < 0.0) return false;   // looking away from the layer
    t0 = max(to.x, 0.0);
    t1 = (ti.x > t0) ? ti.x : to.y;
  }
  // maxDist is an absolute reach, not a range: near-horizon rays enter the slab
  // 20 km out and would otherwise ask for a 100 km march.
  t1 = min(t1, maxDist);
  return t1 > t0;
}
`;

// ---------------------------------------------------------------------------
// Trace pass
// ---------------------------------------------------------------------------

const TRACE_FRAG = /* glsl */ `
${FIELD_UNIFORMS_GLSL}

uniform sampler2D uBlueNoise;
uniform mat4  uInvViewProj;
uniform vec3  uCamPos;
uniform vec2  uHistSize;
uniform vec2  uTraceSize;
uniform vec2  uBayer;
uniform float uDiv;
uniform float uFrameIndex;
uniform float uSampleIndex;
uniform float uSubJitter;

uniform vec3  uLightDir;
uniform vec3  uLightColor;
uniform vec3  uMSColor;
uniform vec3  uAmbientTop;
uniform vec3  uAmbientBottom;
// Aerial perspective is a two-colour vertical gradient plus a forward-scatter
// lobe, not one flat colour. See the note where it is applied.
uniform vec3  uHazeColor;        // horizon
uniform vec3  uHazeZenithColor;  // straight up
uniform vec3  uHazeSunColor;     // extra in-scatter looking toward the sun
uniform vec3  uLightningColor;
uniform float uLightning;

uniform float uSigmaE;
/** 1 / uDensity, so the in-scatter curve below can be expressed as a fraction of
 *  the densest medium the dial can produce without a per-sample divide. */
uniform float uInvDensity;
uniform float uAlbedo;
uniform float uPowder;
uniform float uPowderDepth;
uniform float uVertProb;
uniform float uSilverIntensity;
uniform float uSilverSpread;
uniform float uHaze;
uniform float uMaxDist;
uniform float uDetailFadeStart;
uniform float uDetailFadeEnd;
uniform float uOuterRadius;
uniform float uLightStep;
uniform float uLightSpan;

varying vec2 vUv;

${FIELD_GLSL}

/**
 * Henyey-Greenstein, deliberately without the 1/4pi normalisation: the sun term
 * here is an artistic irradiance (state.sun.intensity), not radiometric watts,
 * and carrying the normalisation would just force a compensating gain constant
 * somewhere less honest. g > 0 forward scattering, g < 0 back scattering.
 */
float hg(float cosT, float g) {
  float g2 = g * g;
  float denom = max(1.0 + g2 - 2.0 * g * cosT, 1e-4);
  return (1.0 - g2) / (denom * sqrt(denom));
}

/**
 * Three-lobe approximation to the Mie phase function of a cloud droplet.
 *
 * Mie scattering off 10-micron water is not one lobe and cannot be faked with
 * one. There is a needle-sharp forward spike (the silver lining, and the reason
 * a thin cloud edge in front of the sun goes incandescent), a BROAD forward
 * pedestal that carries most of the energy out to 60-90 degrees, and a small
 * backscatter shoulder that lights the face of a cloud when the sun is behind
 * you. The old dual lobe had the spike and the shoulder but no pedestal, so the
 * only way it could produce a silver lining at all was a g = 0.85 lobe wide
 * enough to smear a soft halo over twenty-plus degrees of sky, while still
 * leaving the 60-100 degree range - where most of a daylit sky actually sits - 
 * around 30% too dark. Both halves of that are visible in the current build:
 * mushy highlights near the sun and dull grey everywhere else.
 *
 * The weights sum to 1 exactly as the old mix() did, and every HG lobe
 * integrates to the same total over the sphere, so this is an energy-conserving
 * REDISTRIBUTION, not a gain change. Measured against the old function: the peak
 * at zero degrees roughly doubles while 15-25 degrees drops to ~0.65 - a
 * genuinely tight silver lining instead of a halo - 60-100 degrees gains ~40%
 * (lit water rather than grey), backscatter lands within 10%, and the
 * solid-angle-weighted mean outside the forward cone moves by 9%, i.e. there is
 * no exposure shift to chase with a compensating gain.
 *
 * It is evaluated once per PIXEL, since the view/light angle is constant along a
 * ray, so the third lobe costs nothing per sample.
 *
 * The ecc argument shrinks per multiple-scattering octave: light that has
 * bounced three times has forgotten which way it came in, so its phase function
 * flattens toward isotropic. That progression is most of what makes the octave
 * sum look like real multiple scattering rather than three copies of one lobe.
 */
float miePhase(float cosT, float ecc) {
  return hg(cosT,  0.91 * ecc) * 0.52    // forward spike - the silver lining
       + hg(cosT,  0.38 * ecc) * 0.24    // forward pedestal - side-lit luminosity
       + hg(cosT, -0.32 * ecc) * 0.24;   // backscatter shoulder - the sunlit face
}

const vec3 CONE[8] = vec3[8](
  vec3( 0.00,  0.00,  0.00),
  vec3( 0.42, -0.28,  0.31),
  vec3(-0.35,  0.44, -0.21),
  vec3( 0.18,  0.36, -0.48),
  vec3(-0.46, -0.19,  0.37),
  vec3( 0.29, -0.41, -0.34),
  vec3(-0.24,  0.15,  0.51),
  vec3( 0.51,  0.22,  0.11)
);

/**
 * March toward the light, accumulating optical depth.
 *
 * The step ladder is GEOMETRIC (x1.55 per step) rather than linear, and the CPU
 * solves uLightStep so the ladder always spans uLightSpan whatever the step
 * count is. That matters more than it sounds: the shading of a cloud is decided
 * almost entirely by the first ~100 m of medium above each sample, so a linear
 * ladder whose first step is already 100 m simply misses the gradient that
 * separates a sunlit crown from the shoulder just below it. Geometric spacing
 * puts the first sample ~45 m out at HIGH and ~18 m at ULTRA while the tail
 * still reaches a kilometre and a half - so raising cloudLightSteps buys near
 * field precision instead of just more of the same.
 *
 * The cone offsets blur the result, a cheap stand-in for the fact that light
 * arrives from a solid angle rather than a point.
 */
float lightMarch(vec3 p, vec4 weather, float jitter) {
  float optical = 0.0;
  float s = uLightStep;
  float t = uLightStep * jitter * 0.5;
  for (int i = 0; i < CLOUD_LIGHT_STEPS; i++) {
    vec3 lp = p + uLightDir * (t + s * 0.5) + CONE[i] * (t * 0.30);
    optical += densityLow(lp, heightFraction(lp), weather) * s;
    t += s;
    s *= 1.55;
  }
  // One long reach to catch distant occluders without paying for the steps in
  // between - this is what puts one cloud's shadow on the cloud behind it, and
  // it is where raking sunset light gets its contrast from.
  //
  // It fetches its OWN weather. It used to reuse the primary sample's, on the
  // grounds that the reach was a small fraction of a weather tile. That is no
  // longer true: the tile shrank from 24 km to 12.5 km to get individual clouds,
  // and at a low sun this reach is ~7 km, i.e. over half a tile. Reusing the
  // near weather there does not blur the answer, it asks about the wrong cloud
  // entirely - a sample under clear sky inherits the occlusion of a cloud
  // kilometres away, which is a hard-edged dark stamp on exactly the low-sun
  // shot the reach exists to serve. One 2D fetch per lit sample, paid for many
  // times over by the pow() and the five smoothsteps removed from the field.
  //
  // Its weight is capped at one slab thickness, and that cap is load-bearing.
  // This is a SINGLE point sample standing in for the entire reach, and
  // uLightSpan grows to 3.8x the slab as the sun drops - so uncapped, one texel
  // whose reach happened to land inside a core claimed ~1.5 km of medium while
  // its neighbour claimed none. That is a hard-edged black stamp with no
  // gradient around it, on exactly the low-sun shot the long reach exists to
  // serve. At a high sun the cap is inactive (0.55 * span is already below it),
  // so nothing about the noon look changes.
  vec3 fp = p + uLightDir * (uLightSpan * 2.6);
  float reachWeight = min(uLightSpan * 0.55, uThickness * 0.85);
  optical += densityLow(fp, heightFraction(fp), sampleWeather(fp)) * reachWeight;
  return optical;
}

void main() {
  // This trace texel owns exactly one texel of the history buffer, chosen by
  // this frame's Bayer offset.
  vec2 traceTexel = floor(vUv * uTraceSize);
  vec2 histTexel = traceTexel * uDiv + uBayer;

  // Spatio-temporal blue noise: a blue-noise value in space, golden-ratio
  // advanced in time, so the residual is both invisible per frame and averages
  // to zero across the temporal filter.
  float bn = texture(uBlueNoise, gl_FragCoord.xy / ${BLUENOISE_SIZE.toFixed(1)}).r;
  float jitter = fract(bn + uFrameIndex * 0.6180339887);

  // Interleaved-gradient noise, used only to decorrelate the sub-pixel offset
  // from the march offset. Driving both from the same blue-noise value - which
  // is what this did - locks the two together: a texel that starts its march
  // late also samples up and to the right, every frame, so the two errors add
  // instead of cancelling and the residual never averages away.
  float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));

  // Sub-pixel jitter on an R2 low-discrepancy sequence, offset per pixel (per
  // axis) so neighbours stay decorrelated. The temporal filter integrates these
  // into genuine supersampling, so a static camera resolves cloud edges far
  // finer than the half-res buffer it is stored in - for no extra cost.
  //
  // Driven by uSampleIndex (traces of THIS texel), not uFrameIndex (frames). A
  // history texel is only traced every DIV*DIV = 4 frames, so advancing R2 by
  // the frame number advances it by 4 per usable sample - and 4 * 0.7548776662
  // is 3.0195, i.e. the x offset creeps 0.0195 per trace and needs 51 traces
  // (~3.4 s) to cross the texel. The supersampling in x was therefore not
  // happening at all: it was a slow bias, not a sequence. Counting traces makes
  // the stride exactly one R2 step again, which is what the sequence is for.
  vec2 r2 = fract(vec2(0.7548776662, 0.5698402909) * uSampleIndex + vec2(ign, bn));
  vec2 histUv = (histTexel + 0.5 + (r2 - 0.5) * uSubJitter) / uHistSize;

  vec4 ndc = vec4(histUv * 2.0 - 1.0, 1.0, 1.0);
  vec4 world = uInvViewProj * ndc;
  vec3 rd = normalize(world.xyz / world.w - uCamPos);
  vec3 ro = uCamPos;

  float t0, t1;
  if (!slabInterval(ro, rd, uOuterRadius, uMaxDist, t0, t1)) {
    gl_FragColor = vec4(0.0);
    return;
  }

  float range = t1 - t0;

  // Steps grow linearly with distance: far detail is below a pixel anyway. The
  // base step is solved so the ramp covers the whole range in CLOUD_STEPS steps.
  const float GROWTH = 1.1;
  float baseStep = range / (float(CLOUD_STEPS) * (1.0 + GROWTH * 0.5));

  float cosT = dot(rd, uLightDir);

  // The phase function depends only on the view/light angle, which is CONSTANT
  // along a ray - yet it was being evaluated once per octave per march step:
  // three phase evaluations, so nine Henyey-Greenstein lobes, each a sqrt and a
  // divide, per lit sample - all recomputing a value that cannot change. A ray
  // crossing a decent cumulus lights 15-25 samples, so this was several hundred
  // redundant transcendentals per cloud pixel. Hoisting it is the single largest
  // saving in this shader, and it is also what lets miePhase() afford a third
  // lobe: the extra cost lands once per pixel, not once per sample.
  float phase[3];
  phase[0] = miePhase(cosT, 1.0);
  phase[1] = miePhase(cosT, 0.72);
#if MS_OCTAVES > 2
  phase[2] = miePhase(cosT, 0.5184);
#else
  phase[2] = 0.0;
#endif

  // Silver lining: a razor-thin forward lobe that only survives in thin cloud.
  float silver = uSilverIntensity * pow(clamp(cosT, 0.0, 1.0), 24.0);
  // pow(cosT, 24) is below 0.001 outside roughly 25 degrees of the sun, so on
  // the overwhelming majority of the sky the silver term contributes nothing
  // while still costing an exp on every lit sample. The test is uniform along a
  // ray and near-uniform across a warp (cosT varies smoothly in screen space),
  // so skipping it is close to free and pays back one transcendental per sample
  // everywhere the sun is not.
  bool doSilver = silver > 0.001;

  vec3 scatter = vec3(0.0);
  float transmittance = 1.0;
  float depthNum = 0.0;
  float depthDen = 0.0;

  float t = t0 + baseStep * jitter;
  // High-water mark: everything below this has already been sampled in FINE
  // mode, so the step-back taken when re-entering fine mode can never regress
  // onto it.
  //
  // Without it the two modes oscillate wherever densityLow() is positive but
  // densityFull() has eroded the medium away - a region that stage-1 erosion
  // and the 12.5 km weather tile between them have made much more common. Four
  // empty fine steps drop the march back to coarse; the coarse branch tests the
  // SAME shape volume, still finds it non-zero, and steps back one fine step
  // onto the sample it just took, re-running sampleWeather + the shape fetch +
  // densityFull for a result it already has. Net progress across those two
  // iterations is zero, and a ray crossing a broken cloudscape hits that
  // boundary several times, burning roughly one step in five out of a
  // compile-time-fixed budget. With the mark, re-entry resumes exactly where
  // fine mode left off: no repeated work, and no coarse 2x stride is ever taken
  // through medium the shape volume says might be occupied.
  float tCovered = t0;
  bool refining = false;
  int emptyRun = 0;

  for (int i = 0; i < CLOUD_STEPS; i++) {
    if (t > t1 || transmittance < 0.012) break;

    float grow = 1.0 + GROWTH * clamp((t - t0) / max(range, 1e-3), 0.0, 1.0);
    float stepLen = baseStep * grow;
    vec3 p = ro + rd * t;
    float h = heightFraction(p);

    if (!refining) {
      // Coarse stride: double steps, shape only, no lighting. One 3D fetch.
      if (densityLow(p, h) > 0.0) {
        // Step back exactly one fine step - never past the high-water mark, so
        // t can neither regress into already-integrated medium nor re-sample a
        // stretch fine mode has already rejected - and re-enter in fine mode so
        // we do not clip the cloud edge.
        t = max(tCovered, t - stepLen);
        refining = true;
        emptyRun = 0;
        continue;
      }
      t += stepLen * 2.0;
      continue;
    }

    float erode = 1.0 - smoothstep(uDetailFadeStart, uDetailFadeEnd, t);
    vec4 weather = sampleWeather(p);
    float density = densityFull(p, h, weather, erode);

    if (density <= 0.0) {
      emptyRun++;
      t += stepLen;
      if (emptyRun > 3) {          // back to cheap striding
        refining = false;
        tCovered = t;              // ...from here, never from behind it
      }
      continue;
    }
    emptyRun = 0;

    float sigmaE = max(density * uSigmaE, 1e-7);

    float optical = lightMarch(p, weather, jitter);

    // Multiple scattering as an octave sum (Wrenninge / "energy conserving
    // analytic multiple scattering"): each octave sees a thinner medium, a
    // wider phase function and carries less energy. Two or three octaves is
    // enough to stop deep cloud from going pitch black.
    //
    // The first octave is kept separate from the rest because it is the only
    // one that is still DIRECT sunlight. Everything after it has bounced, and
    // bounced light inside a cloud has been filtered by the droplets and by the
    // sky the cloud is sitting in - so it gets its own, cooler colour. That
    // split is two vec3 mads and it is most of what makes a cumulus read as
    // having an inside rather than being a lit shell.
    float eDirect = phase[0] * exp(-optical * uSigmaE);
    float eMulti = 0.0;
    float att = 0.52, weight = 0.5;
    for (int n = 1; n < MS_OCTAVES; n++) {
      eMulti += weight * phase[n] * exp(-optical * uSigmaE * att);
      att *= 0.52;      // each octave sees a thinner medium
      weight *= 0.5;    // and carries half the energy
    }

    // Beer-Powder: the dark-edge term that turns a cloud's silhouette from a
    // cut-out into something with a rind. Measured against a FIXED depth of
    // medium rather than against the raw density value, so it means the same
    // thing whatever scale the density dial happens to be on. Only visible on
    // the lit side of a cloud, hence the view/light weighting. Bounded at 1 - 
    // this darkens thin edges, it must never brighten cores past the energy we
    // actually integrated.
    float powder = 1.0 - exp(-sigmaE * uPowderDepth);
    float powderW = uPowder * (0.5 - 0.5 * cosT);
    float powderM = mix(1.0, powder, powderW);
    eDirect *= powderM;
    // Multiply-scattered light arrives from every direction, so it fills the
    // powder notch back in rather than being darkened by it.
    eMulti *= mix(1.0, powderM, 0.45);

    // In-scatter probability (Schneider): brighter toward the top of a cloud
    // and inside dense cores. This is what gives clouds internal form instead
    // of reading as a uniform grey shell.
    //
    // Measured as a FRACTION of the densest medium the dial can produce, not
    // against an absolute constant. It used to be clamp(density * 3.2), which
    // saturates at density 0.31 - but density peaks near uDensity (0.62 by
    // default, and the weather system pushes it higher in a storm), so the
    // entire top two thirds of the field sat pinned at the curve's maximum and
    // the term contributed no gradient at all exactly where the cloud has the
    // most form to show. Normalising makes the curve span the shape field's real
    // range at any setting of the density dial, and it is one multiply by a
    // uniform rather than a per-sample divide.
    float prob = clamp(remap(h, 0.3, 0.85, 0.55, 2.0), 0.5, 2.0);
    float depthProb = 0.05 + pow(clamp(density * uInvDensity * 1.15, 0.0, 1.0), prob);
    // The vertical term encodes "light arrives from above, so the underside of
    // a cloud sits in its own shadow". That is true at noon and FALSE at dawn,
    // when the sun is raking in horizontally and the bases are the brightest
    // thing in the sky. The light march already models that correctly on its
    // own - a base sample's ray to a low sun exits sideways through very little
    // medium - so leaving this on at low sun double-counts the occlusion and
    // stamps the shot we most want out of the scene. uVertProb fades it out as
    // the sun drops.
    float vertProb = pow(clamp(remap(h, 0.03, 0.20, 0.06, 1.0), 0.0, 1.0), 0.75);
    vertProb = mix(1.0, vertProb, uVertProb);
    // 0.72 rather than 0.62: with depthProb now spanning its full range instead
    // of clipping, this term finally has a gradient to deliver, and letting more
    // of it through is the cheapest contrast between a cloud's sunlit crown and
    // its shadowed shoulder that exists. It is also what makes the base read as
    // a hard dark shelf rather than a soft grey fade.
    float inScat = mix(1.0, depthProb * vertProb, 0.72);
    eDirect *= inScat;
    eMulti *= inScat;

    if (doSilver) eDirect += silver * exp(-optical * uSigmaE * uSilverSpread);

    // Bound the forward lobe. Looking straight through a thin edge at the sun,
    // miePhase peaks near 80, and an unbounded spike in a temporally accumulated
    // buffer is a firefly that takes a second to decay.
    //
    // A soft knee, NOT min(). min() plateaus, and a plateau in a quantity that
    // varies smoothly across the sky draws a hard-edged bright patch with a
    // visible boundary wherever the clamp engages - exactly the sort of
    // suspiciously clean white blob that has no business being in a cloud
    // render. x/(1+x/k) asymptotes to the same ceiling with a continuous
    // derivative, for one extra divide.
    eDirect = eDirect / (1.0 + eDirect * 0.125);
    eMulti = eMulti / (1.0 + eMulti * 0.3333);

    // Ambient in-scatter, and this is where the flat dark base actually comes
    // from. Sky light has to fight its way DOWN through everything above the
    // sample, ground bounce has to fight its way UP through everything below,
    // and near the base of a 700 m cumulus that is five optical depths of
    // medium. Without these two terms the interior of a cloud is lit exactly as
    // brightly as its shell and the whole thing reads as a paper cut-out.
    // 1/(1+x) rather than exp(-x): same shape where it matters, one divide.
    float skyVis = 1.0 / (1.0 + sigmaE * (1.0 - h) * uThickness * 0.55);
    float gndVis = 1.0 / (1.0 + sigmaE * h * uThickness * 0.85);
    vec3 ambient = uAmbientTop * skyVis + uAmbientBottom * gndVis;
    ambient += uLightningColor * (uLightning * (0.25 + 0.75 * (1.0 - h)));

    // Source term is radiance * the SCATTERING coefficient, not density: cloud
    // droplets have an albedo near 1, so sigmaS ~= sigmaE. Getting this wrong
    // decouples brightness from optical depth and the clouds blow out.
    float sigmaS = sigmaE * uAlbedo;
    vec3 S = (uLightColor * eDirect + uMSColor * eMulti + ambient) * sigmaS;

    // Energy-conserving analytic integration over the step (Hillaire). This is
    // the difference between smooth clouds and clouds that band at low step
    // counts, and it costs one exp we already need.
    float stepTrans = exp(-sigmaE * stepLen);
    vec3 integrated = (S - S * stepTrans) / sigmaE;
    scatter += transmittance * integrated;

    float absorbed = transmittance * (1.0 - stepTrans);
    depthNum += absorbed * t;
    depthDen += absorbed;

    transmittance *= stepTrans;
    t += stepLen;
  }

  float alpha = 1.0 - transmittance;
  if (alpha <= 0.0006) {
    gl_FragColor = vec4(0.0);
    return;
  }

  // Aerial perspective. Clouds 15 km out must sit in the same haze as the
  // terrain below them or the sky reads as a matte painting.
  //
  // The haze colour has to be the sky radiance ALONG THIS RAY, and getting that
  // wrong is what produced the hard-edged pale slab in the noon capture. This
  // used to mix toward one flat colour - the horizon's, which at midday is
  // nearly white. A cloud 8 km out, 20 degrees up, therefore had two thirds of
  // its radiance replaced by near-white while the sky it was drawn against was
  // still deep blue. The result is a large region of uniform pale grey with a
  // clean boundary and no internal structure: it reads as a clipping artifact
  // because, effectively, it was one. Blending horizon->zenith by ray elevation
  // makes the veil converge to the sky actually behind the cloud, so distant
  // cloud dissolves instead of being stamped over the sky. Two mixes and a
  // sqrt, once per pixel, not per sample.
  float meanDepth = depthDen > 0.0 ? depthNum / depthDen : t0;
  float fog = 1.0 - exp(-meanDepth * uHaze);
  vec3 haze = mix(uHazeColor, uHazeZenithColor, sqrt(clamp(rd.y, 0.0, 1.0)));
  // Haze is forward-scattering too: looking toward a low sun through 10 km of
  // air, the veil in front of a cloud is far brighter than the veil beside it.
  // This is what makes a backlit cloud sit in a glow instead of on top of one.
  float fwd = clamp(cosT, 0.0, 1.0);
  float fwd2 = fwd * fwd;
  haze += uHazeSunColor * (fwd2 * fwd2 * fwd);
  scatter = mix(scatter, haze * alpha, fog);

  // Dissolve into the haze before the hard march limit so there is no ring.
  float edge = 1.0 - smoothstep(uMaxDist * 0.72, uMaxDist * 0.99, meanDepth);
  scatter *= edge;
  alpha *= edge;

  gl_FragColor = vec4(scatter, alpha);
}
`;

// ---------------------------------------------------------------------------
// Temporal resolve
// ---------------------------------------------------------------------------

const RESOLVE_FRAG = /* glsl */ `
uniform sampler2D uTrace;
uniform sampler2D uHistory;
uniform vec2  uHistSize;
uniform vec2  uTraceSize;
uniform vec2  uBayer;
uniform float uDiv;
uniform float uHistValid;
uniform mat4  uInvViewProj;
uniform mat4  uPrevViewProj;
uniform vec3  uCamPos;
uniform vec3  uPlanetCenter;
uniform float uMidRadius;
uniform float uSharpen;
uniform float uTracedFeedback;
uniform float uMotion;

varying vec2 vUv;

vec2 raySphereFar(vec3 ro, vec3 rd, vec3 ce, float r) {
  vec3 oc = ro - ce;
  float b = dot(oc, rd);
  float c = dot(oc, oc) - r * r;
  float disc = b * b - c;
  if (disc < 0.0) return vec2(1.0, -1.0);
  disc = sqrt(disc);
  return vec2(-b - disc, -b + disc);
}

void main() {
  vec2 px = floor(vUv * uHistSize);
  vec2 blk = floor(px / uDiv);
  vec2 local = px - blk * uDiv;
  vec2 tuv = (blk + 0.5) / uTraceSize;

  vec4 raw = texture2D(uTrace, tuv);
  bool traced = all(lessThan(abs(local - uBayer), vec2(0.5)));

  // The 4-tap cross of the coarse trace. It is needed for the variance clip
  // below, and it is fetched for EVERY texel rather than only the reprojected
  // ones: the checkerboard makes traced diverge inside every 2x2 quad, so the
  // warp was already paying for this path whichever branch it claimed to take.
  vec2 ts = 1.0 / uTraceSize;
  vec4 s1 = texture2D(uTrace, tuv + vec2( ts.x, 0.0));
  vec4 s2 = texture2D(uTrace, tuv + vec2(-ts.x, 0.0));
  vec4 s3 = texture2D(uTrace, tuv + vec2(0.0,  ts.y));
  vec4 s4 = texture2D(uTrace, tuv + vec2(0.0, -ts.y));

  // First two moments of the raw neighbourhood. Both are taken from the
  // UNSHARPENED samples: the clip below has to describe the signal the history
  // was built from, and feeding it a sharpened centre tap inflates sigma
  // exactly where the sharpen just did its work, which quietly re-opens the
  // clip and lets stale history back in at every cloud edge.
  vec4 m1 = (raw + s1 + s2 + s3 + s4) * 0.2;
  vec4 m2 = (raw * raw + s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4) * 0.2;

  // Unsharp mask on the FRESH sample, using taps we already have - free. A
  // half-res trace read through a bilinear upscale loses most of the contrast
  // at a cloud's silhouette; putting a little of it back here is what makes the
  // margins read as crisp-then-wispy instead of uniformly soft.
  //
  // Applied to the fresh sample only, never to the shader's own output, so it
  // cannot feed back on itself and ring. Clamped to the local extrema so it can
  // create no new maximum: an unbounded sharpen on a temporally accumulated
  // buffer is a firefly generator.
  vec4 nb = (m1 * 5.0 - raw) * 0.25;
  vec4 lo4 = min(min(s1, s2), min(s3, s4));
  vec4 hi4 = max(max(s1, s2), max(s3, s4));
  vec4 cur = clamp(raw + (raw - nb) * uSharpen, min(lo4, raw), max(hi4, raw));
  cur.a = clamp(cur.a, 0.0, 1.0);
  cur.rgb = max(cur.rgb, vec3(0.0));

  if (uHistValid < 0.5) {
    gl_FragColor = cur;
    return;
  }

  // Reproject through the point where this ray meets the middle of the slab.
  // At cloud distances the parallax from a single frame of camera translation
  // is negligible, but doing it properly keeps low-flying shots from smearing.
  vec2 uv = (px + 0.5) / uHistSize;
  vec4 ndc = vec4(uv * 2.0 - 1.0, 1.0, 1.0);
  vec4 world = uInvViewProj * ndc;
  vec3 rd = normalize(world.xyz / world.w - uCamPos);
  vec2 hit = raySphereFar(uCamPos, rd, uPlanetCenter, uMidRadius);
  // The miss sentinel is (1, -1), i.e. x > y - and its x is POSITIVE, so the
  // old "hit.x > 0.0" test accepted it and anchored the reprojection at t = 1 m
  // (clamped up to 100 m). A miss only happens with the camera ABOVE the deck
  // looking away from it, but there it is every ray in the upper half of the
  // frame: the anchor sits a hundred metres from the eye instead of at cloud
  // distance, so a metre of translation moves prevUv by a degree and the whole
  // sky above the layer smears. Reject the sentinel before reading x.
  float t = (hit.x > hit.y) ? 20000.0
          : (hit.x > 0.0 ? hit.x : (hit.y > 0.0 ? hit.y : 20000.0));
  t = clamp(t, 100.0, 60000.0);
  vec3 wp = uCamPos + rd * t;

  vec4 prevClip = uPrevViewProj * vec4(wp, 1.0);
  if (prevClip.w <= 0.0) { gl_FragColor = cur; return; }
  vec2 prevUv = prevClip.xy / prevClip.w * 0.5 + 0.5;

  // Off-screen history used to be a BINARY reject: one texel blended 90% of an
  // accumulated, supersampled history, the texel beside it was raw quarter-res
  // trace. Across a frame edge during any pan that draws a hard straight seam
  // between a smooth region and a noticeably coarser one, parallel to whichever
  // border the reprojection ran off. Ramping the feedback out over the last two
  // texels of the history buffer costs two subtractions and removes the seam.
  vec2 edgeTexels = min(prevUv, 1.0 - prevUv) * uHistSize;
  float border = clamp(min(edgeTexels.x, edgeTexels.y) * 0.5, 0.0, 1.0);
  if (border <= 0.0) {
    gl_FragColor = cur;
    return;
  }

  // Variance clipping against this frame's coarse neighbourhood. Softer than a
  // min/max box - a box this coarse would throw away the extra resolution the
  // history is carrying, which is the entire point of the checkerboard.
  vec4 sigma = sqrt(max(m2 - m1 * m1, vec4(0.0)));
  // Floor on sigma, and it is not cosmetic. Inside a large smooth cloud the
  // five coarse taps agree to within float noise, so sigma collapses to ~0 and
  // clamp(hist, m1-0, m1+0) pins the output to the quarter-res mean EXACTLY.
  // Every history texel in that neighbourhood then resolves to one value, the
  // accumulated sub-pixel detail is discarded, and the region flattens into a
  // plateau whose boundary - where sigma finally becomes non-zero - is a hard
  // visible edge. A clip window that never closes below a fraction of the local
  // level keeps the filter a filter instead of a quantiser.
  sigma = max(sigma, vec4(0.004) + abs(m1) * 0.06);

  vec4 hist = texture2D(uHistory, prevUv);

  // Screen-space motion drives everything: the faster the camera turns, the
  // tighter the clip and the more we lean on the fresh sample. uMotion is the
  // CPU's angular-velocity estimate, folded in so a fast turn tightens the
  // filter everywhere at once instead of only where the reprojection happens to
  // have moved far in screen space - which is what used to leave a smear
  // through the centre of the frame during a flick.
  float vel = length((uv - prevUv) * uHistSize);
  float motion = clamp(max(vel / 8.0, uMotion), 0.0, 1.0);
  float k = mix(2.6, 0.75, motion);
  hist = clamp(hist, m1 - sigma * k, m1 + sigma * k);

  // A freshly traced texel used to be written out raw, on the grounds that it
  // is ground truth. It is - but only for the sub-pixel position this frame's
  // R2 jitter happened to pick, and next time this texel is traced the jitter
  // will be somewhere else inside it. Replacing it outright therefore threw the
  // supersampling away and left one texel in four resampling itself every
  // fourth frame, which is precisely the fine grain that was crawling over the
  // brighter cloud masses. Keeping a little history is what turns the jitter
  // into resolution instead of noise; the variance clip above is what keeps
  // that from becoming ghosting.
  float feedback = traced ? mix(uTracedFeedback, 0.0, motion)
                          : mix(0.90, 0.35, motion);
  gl_FragColor = mix(cur, hist, feedback * border);
}
`;

// ---------------------------------------------------------------------------
// Composite into the main scene
// ---------------------------------------------------------------------------

const COMPOSITE_FRAG = /* glsl */ `
uniform sampler2D uClouds;
uniform float uOpacity;
uniform float uDither;
varying vec2 vUv;

#include <common>

void main() {
  // Premultiplied: rgb is in-scattered radiance already weighted by alpha, so a
  // plain bilinear upscale is correct and needs no alpha renormalisation.
  vec4 c = texture2D(uClouds, vUv) * uOpacity;
  gl_FragColor = vec4(max(c.rgb, vec3(0.0)), clamp(c.a, 0.0, 1.0));
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  // A cloud is a huge, almost perfectly smooth gradient - exactly the signal
  // 8-bit output banding is most visible on. Interleaved-gradient noise at
  // sub-LSB amplitude costs about five ALU and removes it. Scaled by alpha so
  // it can never disturb pixels the clouds do not cover.
  //
  // The amplitude is proportional to the local level, and that is not a detail.
  // Under the composer this shader writes into a LINEAR half-float target and
  // post/pipeline.js applies the tone curve much later, so the two include
  // directives above compile to nothing. A flat 1/255 of linear signal is
  // therefore about half an output LSB in the highlights - correct - but on a
  // moonlit cloud sitting at a linear 0.01 it is a twenty percent perturbation,
  // i.e. visible static exactly where the sky is quietest. Scaling by luminance
  // tracks what one output LSB is actually worth after the curve.
  float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
  float lum = dot(gl_FragColor.rgb, vec3(0.2126, 0.7152, 0.0722));
  gl_FragColor.rgb += (ign - 0.5) * uDither * (0.015 + lum) * gl_FragColor.a;
}
`;

// ---------------------------------------------------------------------------
// Cloud shadow map
// ---------------------------------------------------------------------------

const SHADOW_FRAG = /* glsl */ `
${FIELD_UNIFORMS_GLSL}

uniform vec3  uShadowCenter;     // world xz of the map centre, y = reference ground height
uniform float uShadowExtent;
uniform vec3  uLightDir;
uniform float uSigmaE;
uniform float uOuterRadius;
uniform float uShadowStrength;

varying vec2 vUv;

${FIELD_GLSL}

/** Optical depth from a world point along dir through the slab. */
float slabOptical(vec3 p, vec3 dir, float maxDist, int steps) {
  float t0, t1;
  if (!slabInterval(p, dir, uOuterRadius, maxDist, t0, t1)) return 0.0;
  float range = t1 - t0;
  float stepLen = range / float(steps);
  float optical = 0.0;
  float t = t0 + stepLen * 0.5;
  for (int i = 0; i < CLOUD_SHADOW_STEPS; i++) {
    if (i >= steps) break;
    vec3 sp = p + dir * t;
    optical += densityLow(sp, heightFraction(sp));
    t += stepLen;
  }
  return optical * stepLen;
}

void main() {
  vec2 world = uShadowCenter.xz + (vUv - 0.5) * uShadowExtent;
  vec3 p = vec3(world.x, uShadowCenter.y, world.y);

  // Sun. As it drops toward the horizon the shadow ray runs almost parallel to
  // the layer and the cast shadow becomes meaningless (and enormous), so fade
  // it out - by then direct light is nearly gone anyway.
  float elev = uLightDir.y;
  float sunFade = smoothstep(0.02, 0.16, elev) * uShadowStrength;
  float sunT = 1.0;
  if (sunFade > 0.001) {
    float optical = slabOptical(p, uLightDir, 26000.0, CLOUD_SHADOW_STEPS);
    sunT = exp(-optical * uSigmaE);
    sunT = mix(1.0, sunT, sunFade);
  }

  // Straight up: how much sky light reaches this point. Cheaper (fewer steps)
  // and useful for damping ambient under an overcast.
  float zenithOptical = slabOptical(p, vec3(0.0, 1.0, 0.0), 12000.0, 4);
  float zenithT = exp(-zenithOptical * uSigmaE);
  zenithT = mix(1.0, zenithT, uShadowStrength);

  gl_FragColor = vec4(clamp(sunT, 0.0, 1.0), clamp(zenithT, 0.0, 1.0), 0.0, 1.0);
}
`;

// ===========================================================================
// CPU-side procedural data
// ===========================================================================

/** 16 fixed unit gradients - avoids 4 trig calls per noise evaluation. */
const GRAD16 = (() => {
  const g = new Float32Array(32);
  for (let i = 0; i < 16; i++) {
    const a = (i / 16) * TAU;
    g[i * 2] = Math.cos(a);
    g[i * 2 + 1] = Math.sin(a);
  }
  return g;
})();

function ihash(x, y, seed) {
  let h = (x | 0) * 374761393 + (y | 0) * 668265263 + (seed | 0) * 2147483647;
  h = (h ^ (h >> 13)) * 1274126177;
  return (h ^ (h >> 16)) >>> 0;
}

/**
 * Tileable 2D gradient noise. math.js's simplex is not periodic and the weather
 * map must wrap, otherwise flying far enough shows a seam in the cloud field.
 */
function tileNoise2D(x, y, period, seed) {
  const xi = Math.floor(x), yi = Math.floor(y);
  const xf = x - xi, yf = y - yi;
  const u = xf * xf * xf * (xf * (xf * 6 - 15) + 10);
  const v = yf * yf * yf * (yf * (yf * 6 - 15) + 10);
  const x0 = ((xi % period) + period) % period;
  const y0 = ((yi % period) + period) % period;
  const x1 = (x0 + 1) % period;
  const y1 = (y0 + 1) % period;
  const g = (gx, gy, dx, dy) => {
    const idx = (ihash(gx, gy, seed) & 15) * 2;
    return GRAD16[idx] * dx + GRAD16[idx + 1] * dy;
  };
  const n00 = g(x0, y0, xf, yf);
  const n10 = g(x1, y0, xf - 1, yf);
  const n01 = g(x0, y1, xf, yf - 1);
  const n11 = g(x1, y1, xf - 1, yf - 1);
  const a = n00 + u * (n10 - n00);
  const b = n01 + u * (n11 - n01);
  return (a + v * (b - a)) * 0.5 + 0.5;
}

function tileFBM2D(x, y, baseFreq, octaves, seed) {
  let sum = 0, amp = 0.5, norm = 0, freq = baseFreq;
  for (let i = 0; i < octaves; i++) {
    sum += amp * tileNoise2D(x * freq, y * freq, freq, seed + i * 37);
    norm += amp;
    freq *= 2;
    amp *= 0.5;
  }
  return sum / norm;
}

/**
 * Percentile stretch, in place, on an interleaved RGBA byte buffer.
 *
 * This is the fix for the single worst problem the cloud field had. Every
 * sensible way of combining perlin with worley - the Schneider remap, a screen
 * blend, anything - ends up compressing the result into a narrow band well away
 * from 0 and 1. The runtime then subtracts a coverage threshold from it, and
 * because the field only varies by ~0.1 either side of its mean, the only thing
 * left with any dynamic range is the vertical height gradient. Result: every
 * cloud is the same density, cloud boundaries follow the smooth weather map
 * contour instead of the cloud noise, and the sky fills with flat grey smears
 * bounded by hard, suspiciously clean edges.
 *
 * The data is already 8-bit after readback, so a 256-bin histogram is exact and
 * a LUT makes the rewrite one table lookup per byte. Percentiles rather than
 * min/max because a single outlier texel would otherwise decide the scale.
 * Degenerate channels (the detail volume's constant alpha) are left alone.
 */
function stretchChannels(rgba, loPct, hiPct) {
  const texels = rgba.length >> 2;
  const hist = new Uint32Array(256);
  const lut = new Uint8Array(256);
  const loCount = texels * loPct;
  const hiCount = texels * hiPct;

  for (let c = 0; c < 4; c++) {
    hist.fill(0);
    for (let i = c; i < rgba.length; i += 4) hist[rgba[i]]++;

    let acc = 0;
    let lo = 0;
    for (let v = 0; v < 256; v++) {
      acc += hist[v];
      if (acc >= loCount) { lo = v; break; }
    }
    acc = 0;
    let hi = 255;
    for (let v = 255; v >= 0; v--) {
      acc += hist[v];
      if (acc >= hiCount) { hi = v; break; }
    }
    if (hi - lo < 8) continue;   // constant or near-constant channel

    const s = 255 / (hi - lo);
    for (let v = 0; v < 256; v++) {
      lut[v] = Math.max(0, Math.min(255, Math.round((v - lo) * s)));
    }
    for (let i = c; i < rgba.length; i += 4) rgba[i] = lut[rgba[i]];
  }
}

/**
 * Void-and-cluster blue noise (Ulichney 1993). A white-noise dither at these
 * sample counts is instantly readable as static; blue noise pushes the error
 * into frequencies the temporal filter and the eye both discard.
 */
function generateBlueNoise(size, seed) {
  const N = size * size;
  const rng = makeRNG(seed);
  const binary = new Uint8Array(N);
  const energy = new Float32Array(N);

  const R = 4, sigma = 1.9, inv2s2 = 1 / (2 * sigma * sigma);
  const kox = [], koy = [], kw = [];
  for (let dy = -R; dy <= R; dy++) {
    for (let dx = -R; dx <= R; dx++) {
      const d2 = dx * dx + dy * dy;
      if (d2 > R * R) continue;
      kox.push(dx); koy.push(dy); kw.push(Math.exp(-d2 * inv2s2));
    }
  }
  const K = kw.length;

  const splat = (idx, s) => {
    const x = idx % size, y = (idx / size) | 0;
    for (let k = 0; k < K; k++) {
      const xx = (x + kox[k] + size) % size;
      const yy = (y + koy[k] + size) % size;
      energy[yy * size + xx] += s * kw[k];
    }
  };
  const pick = (wantBinary, wantMax) => {
    let bi = -1, bv = wantMax ? -Infinity : Infinity;
    for (let i = 0; i < N; i++) {
      if (binary[i] !== wantBinary) continue;
      const e = energy[i];
      if (wantMax ? e > bv : e < bv) { bv = e; bi = i; }
    }
    return bi;
  };

  const ones = Math.max(1, Math.floor(N * 0.1));
  let placed = 0;
  while (placed < ones) {
    const i = Math.floor(rng() * N);
    if (!binary[i]) { binary[i] = 1; splat(i, 1); placed++; }
  }
  // Relax the initial pattern until it is maximally uniform.
  for (let it = 0; it < 512; it++) {
    const tight = pick(1, true);
    binary[tight] = 0; splat(tight, -1);
    const vd = pick(0, false);
    if (vd === tight) { binary[tight] = 1; splat(tight, 1); break; }
    binary[vd] = 1; splat(vd, 1);
  }

  const proto = binary.slice();
  const protoEnergy = energy.slice();
  const rank = new Int32Array(N);

  // Phase 1: peel the prototype apart, ranking downward from `ones - 1`.
  for (let r = ones - 1; r >= 0; r--) {
    const tight = pick(1, true);
    binary[tight] = 0; splat(tight, -1);
    rank[tight] = r;
  }
  // Phase 2: fill the prototype's voids up to half density.
  binary.set(proto);
  energy.set(protoEnergy);
  const half = (N + 1) >> 1;
  for (let r = ones; r < half; r++) {
    const vd = pick(0, false);
    binary[vd] = 1; splat(vd, 1);
    rank[vd] = r;
  }
  // Phase 3: the minority is now the zeros - rebuild the energy from them.
  energy.fill(0);
  for (let i = 0; i < N; i++) if (!binary[i]) splat(i, 1);
  for (let r = half; r < N; r++) {
    let bi = -1, bv = -Infinity;
    for (let i = 0; i < N; i++) {
      if (binary[i]) continue;
      if (energy[i] > bv) { bv = energy[i]; bi = i; }
    }
    if (bi < 0) break;
    binary[bi] = 1; splat(bi, -1);
    rank[bi] = r;
  }

  const data = new Uint8Array(N);
  for (let i = 0; i < N; i++) data[i] = Math.min(255, Math.floor(((rank[i] + 0.5) / N) * 256));
  return data;
}

// ===========================================================================

// Scratch - hoisted to module scope so update() never allocates.
const _v3a = new THREE.Vector3();
const _quatA = new THREE.Quaternion();
const _colA = new THREE.Color();
const _colB = new THREE.Color();
const _colClear = new THREE.Color();
// Sanitised copies of the three sky colours, refreshed once per frame.
const _skyZen = new THREE.Color();
const _skyHor = new THREE.Color();
const _skyGnd = new THREE.Color();
const _size = new THREE.Vector2();
const _weatherScratch = { x: 0, y: 0, z: 0, w: 0 };

/**
 * Every number this module consumes per frame is written by a sibling - weather
 * owns state.clouds, wind owns state.wind - and a single NaN out of either is
 * PERMANENT here rather than transient. damp() feeds its own output back in, and
 * the advection offsets are accumulators wrapped with `%`, which propagates NaN
 * forever. Once one lands, every uniform in the field block is NaN, every ray
 * misses, and the cloud layer silently disappears for the rest of the session
 * with nothing in the console. Two comparisons per read is a cheap insurance
 * premium against a bug that presents as "clouds stopped working an hour ago".
 */
const fin = (v, fallback) => (typeof v === 'number' && v - v === 0 ? v : fallback);

/**
 * The same guard for a Color we have already copied out of a sibling's state.
 *
 * Every scalar this module reads goes through fin(), but the COLOURS did not,
 * and they are the same trap: sky.zenithColor / horizonColor / groundColor and
 * sun.color all land in trace uniforms, a NaN uniform makes the trace write NaN,
 * and RESOLVE_FRAG's mix(cur, hist, feedback) then folds that NaN into the
 * history buffer, where it is self-sustaining. One bad frame out of
 * sky/atmosphere.js and the cloud layer is gone for the session with nothing in
 * the console - which is precisely the failure mode fin() exists to prevent.
 * Mutates our own scratch copy, never the sibling's object.
 */
const finColor = (c, r, g, b) => {
  if (!(c.r - c.r === 0)) c.r = r;
  if (!(c.g - c.g === 0)) c.g = g;
  if (!(c.b - c.b === 0)) c.b = b;
  return c;
};

export class VolumetricClouds {
  constructor(ctx) {
    this.ctx = ctx;
    this.renderer = ctx.renderer;
    this.state = ctx.state;

    /** Set false if init fails - everything downstream degrades to "no clouds"
     *  rather than taking the whole scene down with it. */
    this.ready = false;
    this.enabled = true;

    // ---- artistic scales, deliberately named rather than buried in the shader
    this.shapeTileSize = 5200;      // world units per repeat of the shape volume
    this.detailTileSize = 420;
    // World units per repeat of the weather map. This is the dial that decides
    // how BIG an individual cloud is, and it was the main reason the sky read as
    // two or three continent-sized smears: at 24 km, the coverage field's
    // lowest octave has ~8 km features, and the near field that actually fills
    // the screen (everything inside ~5 km) is smaller than one of them. So the
    // camera was almost always inside a single cloud mass, which is why the
    // silhouette was one long clean contour instead of a cloudscape. At 12.5 km
    // the coverage features land around 4 km with 1.4 km ragged edges on top - 
    // fair-weather cumulus scale - and the sky gets a population of clouds with
    // gaps of blue between them. Costs nothing: same texture, different scale.
    this.weatherTileSize = 12500;
    this.shearAmount = 340;         // horizontal offset from base to top
    this.shapeEvolveRate = 0.0024;  // shape-volume tiles per second
    this.detailRiseSpeed = 6.0;     // m/s convective updraught in the detail
    // density -> per-metre extinction. Raised from 0.052 alongside the shape
    // rework: the normalised shape field has a lower mean than the old
    // compressed one (that is the whole point - it now spans its range instead
    // of hovering near 0.77), so the same visual opacity needs more extinction
    // per unit density. Net optical depth through a mid-cloud column is
    // unchanged; the CONTRAST between core and margin is what went up.
    this.extinctionScale = 0.070;
    this.sunScatterScale = 0.62;
    this.moonScatterScale = 3.2;
    this.maxMarchDistance = 34000;
    this.shadowExtent = 6000;
    this.shadowStrength = 0.85;

    // ---- smoothed cloud parameters (weather morphing lives here)
    this._p = {
      coverage: fin(ctx.state.clouds.coverage, 0.42),
      density: fin(ctx.state.clouds.density, 0.62),
      altitude: fin(ctx.state.clouds.altitude, 900),
      thickness: fin(ctx.state.clouds.thickness, 700),
      absorption: fin(ctx.state.clouds.absorption, 0.85),
      erosion: fin(ctx.state.clouds.erosion, 0.35),
      storminess: fin(ctx.state.clouds.storminess, 0),
      speed: fin(ctx.state.clouds.speed, 1),
    };

    // Coverage threshold window, solved once per frame in _updateParams() and
    // read by both the shader and getCoverageAt(). Seeded here so a sibling that
    // queries coverage during its own init() gets a sane answer rather than NaN;
    // the inverse-CDF table it really wants does not exist until the weather map
    // has been built, so this first solve takes the analytic fallback.
    this._covLUT = null;
    this._covLo = 1.0;
    this._covHi = 1.18;
    this._solveCoverage(this._p.coverage);

    // ---- advection accumulators (world units; wrapped to the tile period so
    //      float precision never degrades no matter how long the app runs)
    this._shapeOffset = new THREE.Vector3();
    this._detailOffset = new THREE.Vector3();
    this._weatherOffset = new THREE.Vector2();

    this._frame = 0;
    this._lastResizeFrame = -1000;
    this._historyValid = 0;
    this._historyIndex = 0;
    this._shadowTick = 0;
    this._shadowInterval = 2;
    this._prevViewProj = new THREE.Matrix4();
    this._viewProj = new THREE.Matrix4();
    this._invViewProj = new THREE.Matrix4();
    this._prevCamPos = new THREE.Vector3();
    this._prevCamQuat = new THREE.Quaternion();
    this._planetCenter = new THREE.Vector3();
    this._lightDir = new THREE.Vector3(0, 1, 0);
    this._lightColor = new THREE.Color();
    this._msColor = new THREE.Color();
    /** Sum of 1.55^i over the light-march step count - the shader's geometric
     *  ladder is normalised by this so its total reach is step-count agnostic. */
    this._lightGeomSum = 1;
    this._ambientTop = new THREE.Color();
    this._ambientBottom = new THREE.Color();
    this._hazeColor = new THREE.Color();
    this._hazeZenith = new THREE.Color();
    this._hazeSun = new THREE.Color();
    this._lightningColor = new THREE.Color(0.72, 0.80, 1.0);
    this._shadowCenter = new THREE.Vector3();

    /**
     * Published to sibling systems - terrain, grass and scatter bind these
     * objects straight into their own materials (the {value:} wrappers are
     * stable for the lifetime of this system, so binding once is enough).
     *
     *   uCloudShadowMap    R = fraction of DIRECT sun reaching this ground point
     *                      G = fraction of SKY light reaching it (zenith march)
     *   uCloudShadowMatrix world position -> map UV in .xy (ClampToEdge)
     *   uCloudShadowParams (centreX, centreZ, 1/extent, strength)
     *
     * Sample as:
     *   vec2 uv = (worldPos.xz - p.xy) * p.z + 0.5;
     *   float sunShadow = texture2D(map, uv).r;   // 1 = lit, 0 = under cloud
     * Fade toward 1 outside 0..1 if you care about the map's edge.
     */
    this.shadowUniforms = {
      uCloudShadowMap: { value: null },
      uCloudShadowMatrix: { value: new THREE.Matrix4() },
      uCloudShadowParams: { value: new THREE.Vector4(0, 0, 1 / this.shadowExtent, this.shadowStrength) },
      uCloudShadowStrength: { value: this.shadowStrength },
    };
    this.shadowMatrix = this.shadowUniforms.uCloudShadowMatrix.value;
    this.shadowParams = this.shadowUniforms.uCloudShadowParams.value;

    /** 0..1 fraction of direct sunlight reaching the camera's ground position.
     *  CPU-side approximation - no GPU readback, safe to poll every frame. */
    this.sunTransmittance = 1.0;

    this._targets = { trace: null, history: [null, null], shadow: null };
    this._sizes = { hist: new THREE.Vector2(), trace: new THREE.Vector2() };
    /** Drawing-buffer size the current targets were sized from. Watched rather
     *  than the quality dial - see the note in update(). */
    this._allocW = 0;
    this._allocH = 0;
    this._ownedTextures = [];
  }

  // -------------------------------------------------------------------------
  // Init
  // -------------------------------------------------------------------------

  async init() {
    try {
      this._fsScene = new THREE.Scene();
      this._fsCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
      this._fsGeom = new THREE.PlaneGeometry(2, 2);
      // Kept on the instance: _fsMesh.material is reassigned every pass, so this
      // bootstrap material is otherwise unreachable by dispose() and leaks a GL
      // program for the lifetime of the page on every HMR reload.
      this._fsBootMaterial = new THREE.MeshBasicMaterial();
      this._fsMesh = new THREE.Mesh(this._fsGeom, this._fsBootMaterial);
      this._fsMesh.frustumCulled = false;
      this._fsScene.add(this._fsMesh);

      const tex = this.ctx.textures;
      this.shapeTex = this._fromFactory(tex, 'clouds.shape3D', () =>
        this._bakeVolume(SHAPE_SIZE, SHAPE_TILES_X, SHAPE_TILES_Y, 0));
      this.detailTex = this._fromFactory(tex, 'clouds.detail3D', () =>
        this._bakeVolume(DETAIL_SIZE, DETAIL_TILES_X, DETAIL_TILES_Y, 1));
      this.weatherTex = this._fromFactory(tex, 'clouds.weather2D', () => this._buildWeatherMap());
      // If the factory handed back a memoised texture from a previous instance
      // our generator never ran, so recover the CPU copy from the texture.
      if (!this._weatherData && this.weatherTex.image && this.weatherTex.image.data) {
        this._weatherData = this.weatherTex.image.data;
      }
      this.blueNoiseTex = this._fromFactory(tex, 'clouds.blueNoise', () => this._buildBlueNoise());

      // Both the shader's threshold and getCoverageAt() read _covLo/_covHi, so
      // this has to be solved from the real field before the materials capture
      // their first values.
      this._buildCoverageLUT();
      this._solveCoverage(this._p.coverage);

      this._buildMaterials();
      this._allocTargets();
      this._buildCompositeMesh();

      this.ready = true;
    } catch (err) {
      console.error('[VolumetricClouds] init failed, clouds disabled:', err);
      this.ready = false;
      this.enabled = false;
      if (this.mesh) this.mesh.visible = false;
    }
  }

  /** Route texture creation through the shared factory so it is memoised and
   *  disposed once, but fall back to owning it if the factory misbehaves. */
  _fromFactory(factory, name, generator) {
    if (factory && typeof factory.get === 'function') {
      try {
        const t = factory.get(name, generator);
        if (t && t.isTexture) return t;
      } catch (err) {
        console.warn(`[VolumetricClouds] TextureFactory.get("${name}") failed, owning it locally:`, err);
      }
    }
    const t = generator();
    this._ownedTextures.push(t);
    return t;
  }

  link(systems) {
    this._atmosphere = systems.atmosphere || null;
  }

  // -------------------------------------------------------------------------
  // Baking
  // -------------------------------------------------------------------------

  /**
   * Renders the volume as a grid of Z slices into a byte target, reads it back
   * once and repacks it into a Data3DTexture. Rendering to a 3D target directly
   * would save the readback but is a far less travelled code path; a one-off
   * 8 MB glReadPixels during the loading screen is the cheaper risk.
   */
  _bakeVolume(size, tilesX, tilesY, mode) {
    const renderer = this.renderer;
    const w = size * tilesX;
    const h = size * tilesY;
    if (w > renderer.capabilities.maxTextureSize) {
      throw new Error(`cloud volume bake needs a ${w}px texture, GPU max is ${renderer.capabilities.maxTextureSize}`);
    }

    const rt = new THREE.WebGLRenderTarget(w, h, {
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: false,
      stencilBuffer: false,
    });
    rt.texture.colorSpace = THREE.NoColorSpace;
    rt.texture.generateMipmaps = false;

    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uSize: { value: size },
        uTilesX: { value: tilesX },
        uMode: { value: mode },
      },
      vertexShader: FS_VERT,
      fragmentShader: BAKE_FRAG,
      depthTest: false,
      depthWrite: false,
    });

    const prevRT = renderer.getRenderTarget();
    this._fsMesh.material = mat;
    renderer.setRenderTarget(rt);
    renderer.render(this._fsScene, this._fsCamera);

    const flat = new Uint8Array(w * h * 4);
    renderer.readRenderTargetPixels(rt, 0, 0, w, h, flat);
    renderer.setRenderTarget(prevRT);

    mat.dispose();
    rt.dispose();

    // Give every channel back its full 0..1 range before it becomes a texture.
    // ~16 M byte operations for the 128^3 shape volume, once, during the
    // loading screen. See stretchChannels() for why this is not optional.
    stretchChannels(flat, 0.002, 0.002);

    // Repack tiles -> slices. readRenderTargetPixels is bottom-up and the bake
    // shader derives its coordinates from gl_FragCoord, which is too, so the
    // round trip is consistent.
    const rowBytes = size * 4;
    const volume = new Uint8Array(size * size * size * 4);
    for (let z = 0; z < size; z++) {
      const tx = z % tilesX;
      const ty = (z / tilesX) | 0;
      for (let y = 0; y < size; y++) {
        const src = ((ty * size + y) * w + tx * size) * 4;
        const dst = ((z * size + y) * size) * 4;
        volume.set(flat.subarray(src, src + rowBytes), dst);
      }
    }

    const tex = new THREE.Data3DTexture(volume, size, size, size);
    tex.format = THREE.RGBAFormat;
    tex.type = THREE.UnsignedByteType;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.wrapS = tex.wrapT = tex.wrapR = THREE.RepeatWrapping;
    tex.generateMipmaps = false;
    tex.unpackAlignment = 4;
    tex.colorSpace = THREE.NoColorSpace;
    tex.needsUpdate = true;
    return tex;
  }

  /**
   * The weather map decides where clouds are, what type they are and where the
   * towering ones live. Built on the CPU so the same array can answer
   * getCoverageAt() without a GPU readback.
   */
  _buildWeatherMap() {
    const S = WEATHER_SIZE;
    const N = S * S;
    // Four independent fields. Low is the weather "system" scale (systems a few
    // km across), high gives those systems ragged edges, type selects the cloud
    // family and anvil marks where towers are allowed to form.
    const specs = [
      { freq: 3, oct: 4, seed: 101 },   // R coverage, low frequency
      { freq: 9, oct: 3, seed: 211 },   // G coverage, high frequency
      { freq: 2, oct: 2, seed: 331 },   // B cloud type
      { freq: 1, oct: 2, seed: 439 },   // A anvil / tower field
    ];
    const data = new Uint8Array(N * 4);
    const scratch = new Float32Array(N);
    const inv = 1 / S;

    for (let c = 0; c < 4; c++) {
      const spec = specs[c];
      let min = Infinity, max = -Infinity;
      for (let y = 0; y < S; y++) {
        for (let x = 0; x < S; x++) {
          const v = tileFBM2D(x * inv, y * inv, spec.freq, spec.oct, spec.seed);
          scratch[y * S + x] = v;
          if (v < min) min = v;
          if (v > max) max = v;
        }
      }
      // Normalise each channel to the full 0..1 range. The coverage curve in
      // the shader is fitted to this distribution - leave it un-normalised and
      // the coverage dial only works over part of its travel.
      const scale = 1 / Math.max(max - min, 1e-6);
      for (let i = 0; i < N; i++) {
        data[i * 4 + c] = Math.round(clamp01((scratch[i] - min) * scale) * 255);
      }
    }
    this._weatherData = data;
    const tex = new THREE.DataTexture(data, S, S, THREE.RGBAFormat, THREE.UnsignedByteType);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;
    tex.colorSpace = THREE.NoColorSpace;
    tex.needsUpdate = true;
    return tex;
  }

  /**
   * Inverse CDF of the weather map's coverage field, so the artistic dial can be
   * turned into a threshold by ASKING the data rather than guessing.
   *
   * The old curve was a hand-fitted pow() plus two lerps, documented as making
   * state.clouds.coverage mean "fraction of sky with cloud". Measured against
   * the map this file actually bakes, it did not: dial 0.20 produced a mean
   * coverage of 0.107, dial 0.42 (the PARTLY_CLOUDY preset, and state.js's
   * default) produced 0.646, and everything from 0.55 upward produced 0.86-1.00.
   * So the default sky was two thirds covered - no blue between the clouds, on
   * exactly the shot the smaller weather tile exists to fix - and the whole top
   * half of the dial was dead, leaving OVERCAST 0.90, RAIN 0.97 and STORM 1.00
   * indistinguishable in silhouette.
   *
   * A threshold at the (1 - dial) quantile of the field puts, by construction,
   * `dial` of the map above it. That is the definition the comment always
   * claimed, it needs no fitting, and it survives anyone editing the weather
   * map's frequencies or seeds - which a hardcoded curve silently would not.
   *
   * Built once, from the same bytes the GPU samples. 65 floats.
   */
  _buildCoverageLUT() {
    this._covLUT = null;
    const data = this._weatherData;
    if (!data || data.length < WEATHER_SIZE * WEATHER_SIZE * 4) return;

    const N = WEATHER_SIZE * WEATHER_SIZE;
    const BINS = 1024;
    const hist = new Uint32Array(BINS);
    let min = 1, max = 0;
    for (let i = 0; i < N; i++) {
      // Must match coverageFrom()'s `w.r * 0.66 + w.g * 0.34` exactly.
      const f = (data[i * 4] * 0.66 + data[i * 4 + 1] * 0.34) / 255;
      if (f < min) min = f;
      if (f > max) max = f;
      let b = Math.round(f * (BINS - 1));
      if (!(b >= 0)) b = 0; else if (b > BINS - 1) b = BINS - 1;
      hist[b]++;
    }

    const K = COV_LUT_STEPS;
    const lut = new Float32Array(K + 1);
    let acc = 0, bin = 0;
    for (let k = 0; k <= K; k++) {
      const want = (k / K) * N;
      while (bin < BINS - 1 && acc + hist[bin] < want) { acc += hist[bin]; bin++; }
      lut[k] = bin / (BINS - 1);
    }
    // Push both ends clear of the field's real range by more than half the
    // widest threshold window, so dial 0 leaves genuinely nothing above the
    // threshold and dial 1 leaves genuinely nothing below it. Without this the
    // extremes land half a window inside the data and "clear" keeps a wisp.
    lut[0] = min - 0.40;
    lut[K] = max + 0.40;
    this._covLUT = lut;
  }

  /**
   * Solve the coverage threshold window for a dial value. Allocation-free, one
   * table lookup and a lerp - it runs every frame.
   *
   * The WIDTH of the window is the second half of the artistic control: the
   * coverage field crosses it over however much horizontal distance a cloud's
   * margin occupies, so it sets how abruptly a silhouette gives out. Too narrow
   * and clouds end at a contour of the weather map - the machine-cut edge the
   * whole erosion rework is fighting. Too wide and coverage becomes a smooth
   * gradient with nothing at 0 or 1, which is fog. It opens with the dial
   * because an overcast ceiling should give out softly and a lone fair-weather
   * cumulus should not.
   */
  _solveCoverage(cov) {
    const d = clamp01(fin(cov, 0.42));
    const w = lerp(0.24, 0.40, d);
    const lut = this._covLUT;
    if (!lut) {
      // Pre-bake fallback: the old analytic curve. Only ever seen by a sibling
      // that asks for coverage before our init() has built the weather map.
      const cv = Math.pow(d, 0.78);
      this._covLo = lerp(1.0, -0.75, cv);
      this._covHi = this._covLo + lerp(0.18, 1.05, cv);
      return;
    }
    const f = (1 - d) * COV_LUT_STEPS;
    let i = Math.floor(f);
    if (!(i >= 0)) i = 0; else if (i > COV_LUT_STEPS - 1) i = COV_LUT_STEPS - 1;
    const m = lerp(lut[i], lut[i + 1], f - i);
    this._covLo = m - w * 0.5;
    this._covHi = m + w * 0.5;
  }

  _buildBlueNoise() {
    const data = generateBlueNoise(BLUENOISE_SIZE, 0x5A47);
    const tex = new THREE.DataTexture(data, BLUENOISE_SIZE, BLUENOISE_SIZE, THREE.RedFormat, THREE.UnsignedByteType);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.minFilter = THREE.NearestFilter;
    tex.magFilter = THREE.NearestFilter;
    tex.generateMipmaps = false;
    tex.unpackAlignment = 1;
    tex.colorSpace = THREE.NoColorSpace;
    tex.needsUpdate = true;
    return tex;
  }

  // -------------------------------------------------------------------------
  // Materials
  // -------------------------------------------------------------------------

  _buildMaterials() {
    // One shared uniform block: the trace and shadow passes must sample an
    // identical field or the shadows will not match the clouds.
    const f = {
      uShapeTex: { value: this.shapeTex },
      uDetailTex: { value: this.detailTex },
      uWeatherTex: { value: this.weatherTex },
      uPlanetCenter: { value: this._planetCenter },
      uInnerRadius: { value: PLANET_RADIUS + this._p.altitude },
      uThickness: { value: this._p.thickness },
      uCovLo: { value: this._covLo },
      uCovHi: { value: this._covHi },
      uDensity: { value: this._p.density },
      uErosion: { value: this._p.erosion },
      uTypeBias: { value: 0 },
      uAnvil: { value: 0 },
      uShapeOffset: { value: new THREE.Vector3() },
      uDetailOffset: { value: new THREE.Vector3() },
      uWeatherOffset: { value: new THREE.Vector2() },
      uShapeScale: { value: 1 / this.shapeTileSize },
      uDetailScale: { value: 1 / this.detailTileSize },
      uWeatherScale: { value: 1 / this.weatherTileSize },
      uWindDir: { value: new THREE.Vector2(1, 0) },
      uShear: { value: this.shearAmount },
    };
    this._field = f;

    const outerRadius = { value: PLANET_RADIUS + this._p.altitude + this._p.thickness };
    const sigmaE = { value: this.extinctionScale * this._p.absorption };
    const lightDirU = { value: this._lightDir };

    this.traceMaterial = new THREE.ShaderMaterial({
      defines: this._defines(),
      uniforms: Object.assign({}, f, {
        uBlueNoise: { value: this.blueNoiseTex },
        uInvViewProj: { value: this._invViewProj },
        uCamPos: { value: new THREE.Vector3() },
        uHistSize: { value: new THREE.Vector2(1, 1) },
        uTraceSize: { value: new THREE.Vector2(1, 1) },
        uBayer: { value: new THREE.Vector2(0, 0) },
        uDiv: { value: CHECKER_DIV },
        uFrameIndex: { value: 0 },
        // Number of times each history texel has been traced, i.e. frame /
        // (DIV*DIV). The R2 sub-pixel sequence steps on this, not on the frame.
        uSampleIndex: { value: 0 },
        // One history texel of jitter. Wider than this and the variance clip
        // starts rejecting the history during camera motion.
        uSubJitter: { value: 1.0 },
        uLightDir: lightDirU,
        uLightColor: { value: new THREE.Vector3() },
        uMSColor: { value: new THREE.Vector3() },
        uAmbientTop: { value: new THREE.Vector3() },
        uAmbientBottom: { value: new THREE.Vector3() },
        uHazeColor: { value: new THREE.Vector3() },
        uHazeZenithColor: { value: new THREE.Vector3() },
        uHazeSunColor: { value: new THREE.Vector3() },
        uLightningColor: { value: new THREE.Vector3() },
        uLightning: { value: 0 },
        uSigmaE: sigmaE,
        uInvDensity: { value: 1 / Math.max(this._p.density, 0.02) },
        uAlbedo: { value: 0.96 },
        uPowder: { value: 1.0 },
        // Depth of medium the powder notch is measured over, in metres. Roughly
        // one optical depth of a healthy cumulus shoulder.
        uPowderDepth: { value: 130.0 },
        // 1 = the "cloud bases are self-shadowed" approximation is fully on
        // (high sun), 0 = off (low sun, where the light march is authoritative).
        uVertProb: { value: 1.0 },
        // Solved per frame from uThickness and the sun elevation; see
        // _updateLightMarch(). The shader's ladder is geometric, so these two
        // numbers are all it needs.
        uLightStep: { value: 45.0 },
        uLightSpan: { value: 1050.0 },
        uSilverIntensity: { value: 1.6 },
        // Optical-depth falloff of the silver lobe: it must die inside the
        // first fraction of an optical depth or every cloud gets a halo.
        uSilverSpread: { value: 6.0 },
        uHaze: { value: 0.00004 },
        uMaxDist: { value: this.maxMarchDistance },
        uDetailFadeStart: { value: 2500 },
        uDetailFadeEnd: { value: 11000 },
        uOuterRadius: outerRadius,
      }),
      vertexShader: FS_VERT,
      fragmentShader: TRACE_FRAG,
      depthTest: false,
      depthWrite: false,
      blending: THREE.NoBlending,
    });

    this.resolveMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uTrace: { value: null },
        uHistory: { value: null },
        uHistSize: { value: new THREE.Vector2(1, 1) },
        uTraceSize: { value: new THREE.Vector2(1, 1) },
        uBayer: { value: new THREE.Vector2(0, 0) },
        uDiv: { value: CHECKER_DIV },
        uHistValid: { value: 0 },
        uInvViewProj: { value: this._invViewProj },
        uPrevViewProj: { value: this._prevViewProj },
        uCamPos: { value: new THREE.Vector3() },
        uPlanetCenter: { value: this._planetCenter },
        uMidRadius: { value: PLANET_RADIUS + this._p.altitude + this._p.thickness * 0.5 },
        uSharpen: { value: 0.34 },
        uTracedFeedback: { value: 0.55 },
        uMotion: { value: 0 },
      },
      vertexShader: FS_VERT,
      fragmentShader: RESOLVE_FRAG,
      depthTest: false,
      depthWrite: false,
      blending: THREE.NoBlending,
    });

    this.shadowMaterial = new THREE.ShaderMaterial({
      defines: this._defines(),
      uniforms: Object.assign({}, f, {
        uShadowCenter: { value: this._shadowCenter },
        uShadowExtent: { value: this.shadowExtent },
        uLightDir: lightDirU,
        uSigmaE: sigmaE,
        uOuterRadius: outerRadius,
        uShadowStrength: { value: this.shadowStrength },
      }),
      vertexShader: FS_VERT,
      fragmentShader: SHADOW_FRAG,
      depthTest: false,
      depthWrite: false,
      blending: THREE.NoBlending,
    });

    this.compositeMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uClouds: { value: null },
        uOpacity: { value: 1 },
        // ~2.4 output LSB per unit luminance: the ACES curve's midtone slope, so
        // the dither lands at roughly half an LSB once the grade has run.
        uDither: { value: 2.4 / 255 },
      },
      vertexShader: COMPOSITE_VERT,
      fragmentShader: COMPOSITE_FRAG,
      // Not flagged transparent on purpose: this keeps the clouds in the opaque
      // list, so they draw after the sky dome and terrain but before petals,
      // rain and mist - the correct slot for a distant volumetric layer.
      transparent: false,
      depthTest: true,
      depthWrite: false,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor,                    // colour is premultiplied
      blendDst: THREE.OneMinusSrcAlphaFactor,
      blendEquationAlpha: THREE.AddEquation,
      blendSrcAlpha: THREE.OneFactor,
      blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
      toneMapped: true,
    });
    this._outerRadiusU = outerRadius;
    this._sigmaEU = sigmaE;
    this._applyTierTunables();
  }

  _defines() {
    const q = this.state.quality;
    // Compile-time loop bounds. These are defines rather than uniforms so the
    // driver can bound (and partly unroll) every loop - the contract's "no
    // unbounded loops in fragment shaders" rule, enforced by construction.
    const steps = clamp(Math.round(q.cloudSteps || 48), 8, 128);
    // CONE[] in the trace shader has exactly 8 entries.
    const lightSteps = clamp(Math.round(q.cloudLightSteps || 6), 2, 8);
    // A third scattering octave is what stops thick cloud interiors going flat
    // black, and it reuses the light march's optical depth - two exps and a
    // phase eval. Only the very cheapest tier goes without.
    const msOctaves = steps >= 32 ? 3 : 2;
    // Normaliser for the light march's geometric step ladder. Solved here (not
    // in the shader) so the march spans the same physical distance at every
    // tier and extra light steps buy near-field precision rather than reach.
    let sum = 0;
    let g = 1;
    for (let i = 0; i < lightSteps; i++) { sum += g; g *= 1.55; }
    this._lightGeomSum = sum;
    // Width of the flat cloud base, as a multiple of the ~18 m authored on HIGH.
    // A fine march step is range/(steps*1.55), so halving the step count doubles
    // the step and a fixed shoulder falls below it - at which point the base is a
    // discontinuity between samples and stair-steps across the sky. Scaled off
    // the step count, LOW gets a base it can actually resolve and ULTRA gets a
    // harder shelf than HIGH can afford. Compile-time, so it folds into the
    // smoothstep constants and costs nothing.
    const shoulder = steps >= 64 ? '0.72' : steps >= 40 ? '1.0' : steps >= 28 ? '1.45' : '2.3';
    return {
      CLOUD_STEPS: String(steps),
      CLOUD_LIGHT_STEPS: String(lightSteps),
      MS_OCTAVES: String(msOctaves),
      BASE_SHOULDER: shoulder,
      CLOUD_SHADOW_STEPS: String(q.tier === QUALITY.LOW ? 6 : 8),
    };
  }

  /**
   * Tier-dependent tunables that do not need a shader recompile. The detail
   * fade is the important one: erosion is a 3D texture fetch per lit sample, so
   * pulling the fade in is the single cheapest way to buy back milliseconds.
   */
  _applyTierTunables() {
    const tier = this.state.quality.tier;
    const u = this.traceMaterial.uniforms;
    // The fades are pulled IN from where they used to be (HIGH was 2500/11000).
    // densityFull()'s first erosion stage now bites the silhouette using a
    // channel that is already in the shape fetch, at every distance and for no
    // texture bandwidth at all, so the detail volume no longer has to be kept
    // alive out to 11 km purely to stop far clouds going smooth. That is a 3D
    // fetch removed from a large fraction of lit samples, and it is part of what
    // pays for the extra shaping: past ~7 km the detail volume's features are
    // well under a pixel and its only remaining job is one stage 1 does better.
    let start, end, maxDist, sharpen;
    switch (tier) {
      case QUALITY.LOW:    start = 300;  end = 1600;  maxDist = 24000; sharpen = 0.30; break;
      case QUALITY.MEDIUM: start = 1000; end = 4200;  maxDist = 30000; sharpen = 0.34; break;
      case QUALITY.ULTRA:  start = 3500; end = 13000; maxDist = 38000; sharpen = 0.26; break;
      default:             start = 2000; end = 7500;  maxDist = 34000; sharpen = 0.34; break;
    }
    u.uDetailFadeStart.value = start;
    u.uDetailFadeEnd.value = end;
    u.uMaxDist.value = maxDist;
    this.maxMarchDistance = maxDist;
    // The coarser the trace, the more contrast the bilinear upscale eats, so the
    // lower tiers get MORE unsharp - ULTRA traces at full resolution and needs
    // almost none. It costs nothing either way: the taps are already fetched.
    this.resolveMaterial.uniforms.uSharpen.value = sharpen;
  }

  _buildCompositeMesh() {
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.compositeMaterial);
    this.mesh.frustumCulled = false;
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;
    this.mesh.matrixAutoUpdate = false;
    // 1001, not 1000. sky/atmosphere.js puts its dome at 1000 and is also
    // opaque, and three breaks a renderOrder tie on material.id - i.e. on which
    // system happened to build its ShaderMaterial first. That currently works
    // (atmosphere inits ahead of us) but it is an accident: anything that
    // rebuilds the sky material later gives it a higher id, and the dome would
    // then paint over the clouds instead of under them. One integer removes the
    // tie. Still below every transparent draw, so petals, rain and mist are
    // unaffected.
    this.mesh.renderOrder = 1001;
    this.mesh.name = 'VolumetricClouds';
    this.ctx.scene.add(this.mesh);
  }

  // -------------------------------------------------------------------------
  // Render targets
  // -------------------------------------------------------------------------

  _floatType() {
    const ext = this.renderer.extensions;
    const ok = ext.has('EXT_color_buffer_half_float') || ext.has('EXT_color_buffer_float');
    return ok ? THREE.HalfFloatType : THREE.UnsignedByteType;
  }

  _allocTargets() {
    const renderer = this.renderer;
    renderer.getDrawingBufferSize(_size);
    // Record what we sized against even on the early-out path, so update()'s
    // change detector does not retrigger every frame.
    this._allocW = _size.x;
    this._allocH = _size.y;
    const q = this.state.quality;
    const res = clamp(q.cloudResolution || 0.5, 0.2, 1.0);

    // NOTE: state.quality.resolutionScale is deliberately NOT applied here.
    // core/engine.js and post/pipeline.js both fold it into
    // renderer.setPixelRatio, so getDrawingBufferSize() already carries it.
    // Multiplying by it again squared the adaptive scale: at resolutionScale
    // 0.7 the clouds fell to 0.49 of the intended buffer - 35% of screen linear
    // resolution instead of 50% - so the one thing that visibly blocked up was
    // the sky, precisely when the frame was already struggling.
    const div = CHECKER_DIV;
    const histW = Math.max(div * 2, Math.round((_size.x * res) / div) * div);
    const histH = Math.max(div * 2, Math.round((_size.y * res) / div) * div);
    const traceW = histW / div;
    const traceH = histH / div;

    if (this._sizes.hist.x === histW && this._sizes.hist.y === histH && this._targets.trace) return;

    const type = this._floatType();
    const opts = {
      format: THREE.RGBAFormat,
      type,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
      stencilBuffer: false,
      generateMipmaps: false,
    };

    this._disposeTargets();

    this._targets.trace = new THREE.WebGLRenderTarget(traceW, traceH, Object.assign({}, opts, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
    }));
    this._targets.history[0] = new THREE.WebGLRenderTarget(histW, histH, opts);
    this._targets.history[1] = new THREE.WebGLRenderTarget(histW, histH, opts);
    for (const rt of [this._targets.trace, this._targets.history[0], this._targets.history[1]]) {
      rt.texture.colorSpace = THREE.NoColorSpace;
      rt.texture.wrapS = rt.texture.wrapT = THREE.ClampToEdgeWrapping;
    }

    this._sizes.hist.set(histW, histH);
    this._sizes.trace.set(traceW, traceH);

    const t = this.traceMaterial.uniforms;
    t.uHistSize.value.set(histW, histH);
    t.uTraceSize.value.set(traceW, traceH);
    const r = this.resolveMaterial.uniforms;
    r.uHistSize.value.set(histW, histH);
    r.uTraceSize.value.set(traceW, traceH);

    this._allocShadowTarget();
    this._historyValid = 0;
  }

  _allocShadowTarget() {
    const tier = this.state.quality.tier;
    const size = tier === QUALITY.LOW ? 128 : tier === QUALITY.MEDIUM ? 256 : 512;
    if (this._targets.shadow && this._targets.shadow.width === size) return;
    this._targets.shadow?.dispose();
    this._targets.shadow = new THREE.WebGLRenderTarget(size, size, {
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
      stencilBuffer: false,
      generateMipmaps: false,
    });
    this._targets.shadow.texture.colorSpace = THREE.NoColorSpace;
    this._targets.shadow.texture.wrapS = THREE.ClampToEdgeWrapping;
    this._targets.shadow.texture.wrapT = THREE.ClampToEdgeWrapping;
    this.shadowUniforms.uCloudShadowMap.value = this._targets.shadow.texture;
    this._shadowInterval = tier === QUALITY.LOW ? 4 : 2;
    // A freshly allocated target holds undefined memory, and terrain/grass may
    // sample it before our first shadow pass runs. Publish a valid identity
    // transform and a fully-lit map so nobody ever reads garbage.
    this._updateShadowTransform();
    this._clearShadowMap();
  }

  /** Publishes world -> shadow-map UV for whatever is currently in the map. */
  _updateShadowTransform() {
    const inv = 1 / this.shadowExtent;
    const c = this._shadowCenter;
    this.shadowParams.set(c.x, c.z, inv, this.shadowStrength);
    this.shadowMatrix.identity();
    // Column-major: uv.x = e0*x + e8*z + e12, uv.y = e1*x + e9*z + e13.
    const e = this.shadowMatrix.elements;
    e[0] = inv; e[4] = 0; e[8] = 0; e[12] = 0.5 - c.x * inv;
    e[1] = 0;   e[5] = 0; e[9] = inv; e[13] = 0.5 - c.z * inv;
    this.shadowUniforms.uCloudShadowStrength.value = this.shadowStrength;
  }

  _disposeTargets() {
    this._targets.trace?.dispose();
    this._targets.history[0]?.dispose();
    this._targets.history[1]?.dispose();
    this._targets.trace = null;
    this._targets.history[0] = null;
    this._targets.history[1] = null;
  }

  resize() {
    if (!this.ready) return;
    this._allocTargets();
  }

  onQualityChange(quality) {
    if (!this.ready) return;
    const d = this._defines();
    Object.assign(this.traceMaterial.defines, d);
    Object.assign(this.shadowMaterial.defines, d);
    this.traceMaterial.needsUpdate = true;
    this.shadowMaterial.needsUpdate = true;
    this._applyTierTunables();
    this._allocTargets();
    this._allocShadowTarget();
    this._historyValid = 0;
    const off = !quality.cloudSteps;
    this.enabled = !off;
    if (this.mesh) this.mesh.visible = !off;
  }

  // -------------------------------------------------------------------------
  // Frame
  // -------------------------------------------------------------------------

  update(dt, state) {
    if (!this.ready) return;

    const q = state.quality;
    if (!q.cloudSteps) {
      if (this.enabled) {
        this.enabled = false;
        this.mesh.visible = false;
        this._clearShadowMap();
      }
      return;
    }
    if (!this.enabled) {
      this.enabled = true;
      this.mesh.visible = true;
      this._historyValid = 0;
    }

    // Adaptive resolution never reaches resize(): the quality manager moves
    // state.quality.resolutionScale, post/pipeline.js turns that into a
    // renderer.setPixelRatio call, and the drawing buffer changes underneath us
    // with no resize event anywhere. So watch the drawing buffer itself - that
    // catches a pixel-ratio change AND a window resize, in two integer compares,
    // and it needs no assumption about who applies the scale. Rate-limited to
    // once per 60 frames so a sliding scale cannot reallocate every frame; a
    // real window resize goes through resize() and is not delayed.
    this.renderer.getDrawingBufferSize(_size);
    if ((_size.x !== this._allocW || _size.y !== this._allocH)
        && this._frame - this._lastResizeFrame > 60) {
      this._lastResizeFrame = this._frame;
      this._allocTargets();
    }

    this._frame++;
    // Clamped rather than trusted: dt reaches the smoothing filters and the
    // advection accumulators, both of which feed their own output back in, so a
    // single bad frame time would be permanent rather than momentary.
    const step = clamp(fin(dt, 0), 0, 0.25);
    this._updateParams(step, state);
    this._updateAdvection(step, state);
    this._updateLighting(state);
    this._updateCamera(step, state);

    const renderer = this.renderer;
    const prevRT = renderer.getRenderTarget();
    const prevAutoClear = renderer.autoClear;
    renderer.autoClear = true;

    this._renderTrace(renderer);
    this._renderResolve(renderer);
    this._renderShadow(renderer, state);

    renderer.setRenderTarget(prevRT);
    renderer.autoClear = prevAutoClear;

    this.compositeMaterial.uniforms.uClouds.value = this._targets.history[this._historyIndex].texture;

    // Remember this frame's camera for next frame's reprojection.
    this._prevViewProj.copy(this._viewProj);
    this._prevCamPos.copy(this.ctx.camera.position);
    this.ctx.camera.getWorldQuaternion(this._prevCamQuat);
    this._historyValid = 1;
  }

  /** Critically damped morph of every cloud parameter, so a weather state
   *  machine that snaps its targets still produces a smooth sky. */
  _updateParams(dt, state) {
    const c = state.clouds;
    const p = this._p;
    // Transitions are slow on purpose: real cloud fields take minutes.
    const k = lerp(0.35, 1.1, clamp01(fin(state.weather.transition, 1)));
    // Every target goes through fin() before it reaches damp(). damp() is
    // a = lerp(a, b, ...), so a NaN target does not produce one bad frame - it
    // latches into the accumulator and the layer never comes back.
    p.coverage = damp(p.coverage, clamp01(fin(c.coverage, 0.42)), 0.45 * k, dt);
    p.density = damp(p.density, Math.max(0, fin(c.density, 0.62)), 0.5 * k, dt);
    // Keep the deck above the player: a zero altitude puts the inner shell below
    // the camera, which makes slabInterval take its "inside the layer" branch at
    // ground level and fills the screen with fog.
    p.altitude = damp(p.altitude, clamp(fin(c.altitude, 900), 120, 20000), 0.3, dt);
    p.thickness = damp(p.thickness, clamp(fin(c.thickness, 700), 50, 8000), 0.3, dt);
    p.absorption = damp(p.absorption, Math.max(0.05, fin(c.absorption, 0.85)), 0.6, dt);
    p.erosion = damp(p.erosion, clamp01(fin(c.erosion, 0.35)), 0.6, dt);
    p.storminess = damp(p.storminess, clamp01(fin(c.storminess, 0)), 0.35, dt);
    p.speed = damp(p.speed, clamp(fin(c.speed, 1), -4, 4), 0.6, dt);

    const f = this._field;
    // Coverage threshold window. This was a pow() and two mixes evaluated inside
    // coverageFrom() - i.e. once per density sample, and the light march takes
    // seven of those per lit sample plus its long reach. It is a function of one
    // uniform, so solving it here is exactly equivalent and removes roughly nine
    // transcendentals per lit pixel-sample. getCoverageAt() reads the same two
    // numbers, so the CPU mirror can no longer drift from the shader.
    this._solveCoverage(p.coverage);
    f.uCovLo.value = this._covLo;
    f.uCovHi.value = this._covHi;
    f.uDensity.value = p.density;
    this.traceMaterial.uniforms.uInvDensity.value = 1 / Math.max(p.density, 0.02);
    f.uErosion.value = p.erosion;
    f.uThickness.value = p.thickness;
    f.uInnerRadius.value = PLANET_RADIUS + p.altitude;
    this._outerRadiusU.value = PLANET_RADIUS + p.altitude + p.thickness;
    this.resolveMaterial.uniforms.uMidRadius.value = PLANET_RADIUS + p.altitude + p.thickness * 0.5;
    this._sigmaEU.value = this.extinctionScale * p.absorption;

    // Storms push the field toward towering cumulonimbus with flared tops;
    // heavy overcast without storm flattens it into stratus instead.
    const overcastFlatten = clamp01((p.coverage - 0.72) * 2.4) * (1 - p.storminess);
    f.uTypeBias.value = clamp(0.18 + p.storminess * 0.55 - overcastFlatten * 0.65, -0.6, 0.8);
    f.uAnvil.value = clamp01(p.storminess * 1.15 - 0.08);
    f.uShear.value = this.shearAmount * lerp(1, 1.9, p.storminess);
    // Storm cloud reads darker partly because it is thicker and partly because
    // larger droplets absorb more. The thickness is handled by the field; this
    // is the absorption half, and it is what keeps a thunderhead from looking
    // like a white cumulus that happens to be big.
    this.traceMaterial.uniforms.uAlbedo.value = lerp(0.97, 0.84, p.storminess);
  }

  /** Advect the field. Shape, detail and weather move at different speeds and
   *  slightly different headings, which produces the parallax that makes a
   *  cloud field read as volume rather than a scrolling texture. */
  _updateAdvection(dt, state) {
    const w = state.wind;
    const speed = this._p.speed
      * lerp(6, 26, clamp01(fin(w.strength, 3.2) / 14))
      * clamp(fin(w.gust, 1), 0.6, 2.2);
    // The offsets below are accumulators wrapped with `%`, and NaN % 1 is NaN - 
    // one bad heading would freeze the whole field at NaN for the session.
    let dx = fin(w.direction.x, 1), dz = fin(w.direction.y, 0);
    const dLen = Math.sqrt(dx * dx + dz * dz);
    if (dLen < 1e-4) { dx = 1; dz = 0; } else { dx /= dLen; dz /= dLen; }

    const shapeTile = this.shapeTileSize;
    const detailTile = this.detailTileSize;
    const weatherTile = this.weatherTileSize;

    // Offsets are kept in texture space and wrapped to [0,1) so long sessions
    // never lose precision in the sampler.
    const so = this._shapeOffset;
    so.x = (so.x + (dx * speed * dt) / shapeTile) % 1;
    so.z = (so.z + (dz * speed * dt) / shapeTile) % 1;
    // Slow drift through the third axis of the volume: this is what makes a
    // cloud field evolve rather than just slide past. One tile per ~7 minutes,
    // which is roughly the lifetime of a fair-weather cumulus.
    so.y = (so.y - dt * this.shapeEvolveRate * this._p.speed) % 1;

    const doff = this._detailOffset;
    // Detail runs ~1.6x faster and drifts across the shape heading, so the two
    // octaves separate instead of moving as one rigid texture. It also rises at
    // a convective updraught speed, which reads as boiling at the cloud tops.
    const ddx = dx * 0.86 - dz * 0.5;
    const ddz = dz * 0.86 + dx * 0.5;
    doff.x = (doff.x + (ddx * speed * 1.6 * dt) / detailTile) % 1;
    doff.z = (doff.z + (ddz * speed * 1.6 * dt) / detailTile) % 1;
    doff.y = (doff.y - (this.detailRiseSpeed * this._p.speed * dt) / detailTile) % 1;

    const wo = this._weatherOffset;
    // Weather systems crawl - a quarter of the cloud speed.
    wo.x = (wo.x + dx * speed * 0.25 * dt) % weatherTile;
    wo.y = (wo.y + dz * speed * 0.25 * dt) % weatherTile;

    const f = this._field;
    f.uShapeOffset.value.copy(so);
    f.uDetailOffset.value.copy(doff);
    f.uWeatherOffset.value.copy(wo);
    f.uWindDir.value.set(dx, dz);
  }

  _updateLighting(state) {
    const sun = state.sun;
    const moon = state.moon;
    const sunW = clamp01(fin(sun.visibility, 0)) * Math.max(0, fin(sun.intensity, 0));
    const moonW = clamp01(fin(moon.visibility, 0)) * Math.max(0, fin(moon.intensity, 0));
    const useMoon = moonW > sunW;

    const src = useMoon ? moon : sun;
    this._lightDir.copy(src.direction);
    if (!(this._lightDir.lengthSq() > 1e-6)) this._lightDir.set(0, 1, 0);
    else this._lightDir.normalize();

    // Handover. The two sources swap the instant moonW crosses sunW, but their
    // artistic scales differ by 5x (0.62 vs 3.2) - so a plain switch multiplies
    // the light on every cloud by five between two consecutive frames, at dusk,
    // and the temporal filter then drags that pop across the sky for the best
    // part of a second. `moonness` ramps from 0 at the crossover to 1 once one
    // source clearly owns the sky, and everything that differs between the two
    // rides it, so the energy is continuous through the swap. Directions still
    // flip, but at that moment both sources deliver the same (small) energy, so
    // what changes is which face is faintly lit, not how bright the sky is.
    const winW = useMoon ? moonW : sunW;
    const loseW = useMoon ? sunW : moonW;
    const moonness = useMoon ? clamp01((winW - loseW) / Math.max(winW, 1e-6)) : 0;

    const scale = lerp(this.sunScatterScale, this.moonScatterScale, moonness);
    finColor(this._lightColor.copy(src.color), 1, 1, 1).multiplyScalar(winW * scale);

    const sky = state.sky;
    // Sanitised once, used everywhere below. See finColor(): these three land in
    // trace uniforms and a NaN in any of them is permanent, not transient.
    const zen = finColor(_skyZen.copy(sky.zenithColor), 0.18, 0.36, 0.72);
    const hor = finColor(_skyHor.copy(sky.horizonColor), 0.72, 0.80, 0.92);
    const gnd = finColor(_skyGnd.copy(sky.groundColor), 0.22, 0.24, 0.20);

    // Colour of the multiply-scattered octaves. Light that has bounced three or
    // four times inside a cloud has spent long enough among the droplets - and
    // among the sky the cloud is embedded in - to lose the sun's warmth, which
    // is why a real cumulus has a warm rim and a cool grey-blue core. Blend
    // toward the zenith hue at MATCHED luminance so this is purely a hue shift
    // and cannot smuggle in extra energy.
    const lLum = Math.max(1e-5, 0.2126 * this._lightColor.r + 0.7152 * this._lightColor.g + 0.0722 * this._lightColor.b);
    _colB.copy(zen);
    const zLum = Math.max(1e-5, 0.2126 * _colB.r + 0.7152 * _colB.g + 0.0722 * _colB.b);
    _colB.multiplyScalar(lLum / zLum);
    this._msColor.copy(this._lightColor).lerp(_colB, lerp(0.36, 0.20, moonness));

    // Ambient: sky above, ground bounce below. Overcast kills both. These two
    // are no longer a convex blend - the shader attenuates each by how much
    // medium sits between the sample and its source - so the coefficients are
    // the light arriving at a fully EXPOSED top / base, not an average.
    const amb = Math.max(0, fin(sky.ambientIntensity, 1));
    const overcast = 1 - 0.55 * clamp01(this._p.coverage * this._p.density * 1.4);
    this._ambientTop.copy(zen).multiplyScalar(amb * 0.66 * overcast);
    this._ambientBottom.copy(gnd).multiplyScalar(amb * 0.22 * overcast);
    // Bases pick up the horizon at sunset - this is what makes low sun rake
    // colour across the undersides instead of leaving them flat grey. It rides
    // the ground-bounce term, which is the one the shader lets survive down at
    // the base of a cloud, so it lands exactly where it is wanted.
    _colA.copy(hor).multiplyScalar(amb * 0.46 * clamp01(1.2 - Math.abs(this._lightDir.y) * 2.0));
    this._ambientBottom.add(_colA);

    // Aerial perspective, as the sky radiance along the ray rather than one flat
    // colour - see the note where it is applied in TRACE_FRAG. The zenith end is
    // deliberately a little brighter than the raw zenith sky: the veil in front
    // of a cloud is lit air, and the atmosphere shader's zenith value is the
    // radiance of the WHOLE column, not of the first few kilometres of it.
    const dayness = clamp01(fin(sun.visibility, 0));
    this._hazeColor.copy(hor).multiplyScalar(lerp(0.55, 0.95, dayness));
    this._hazeZenith.copy(zen).lerp(hor, 0.30)
      .multiplyScalar(lerp(0.60, 1.05, dayness));
    // Forward-scatter lobe. Only the sun makes a haze glow worth having; the
    // moon rides `moonness` down to nothing for the same reason the silver
    // lining does. Strongest when the sun is low, which is when the light is
    // travelling through the most air between the cloud and the camera.
    const lowSunHaze = clamp01(1.0 - Math.abs(this._lightDir.y) * 1.4);
    this._hazeSun.copy(hor)
      .multiplyScalar(dayness * (1 - moonness) * lerp(0.10, 0.85, lowSunHaze));

    const t = this.traceMaterial.uniforms;
    t.uLightColor.value.set(this._lightColor.r, this._lightColor.g, this._lightColor.b);
    t.uMSColor.value.set(this._msColor.r, this._msColor.g, this._msColor.b);
    t.uAmbientTop.value.set(this._ambientTop.r, this._ambientTop.g, this._ambientTop.b);
    t.uAmbientBottom.value.set(this._ambientBottom.r, this._ambientBottom.g, this._ambientBottom.b);
    t.uHazeColor.value.set(this._hazeColor.r, this._hazeColor.g, this._hazeColor.b);
    t.uHazeZenithColor.value.set(this._hazeZenith.r, this._hazeZenith.g, this._hazeZenith.b);
    t.uHazeSunColor.value.set(this._hazeSun.r, this._hazeSun.g, this._hazeSun.b);

    const flash = clamp01(fin(state.weather.lightning, 0));
    _colA.copy(this._lightningColor).multiplyScalar(flash * flash * 5.0);
    t.uLightningColor.value.set(_colA.r, _colA.g, _colA.b);
    t.uLightning.value = flash;

    // Haze strength tracks the weather's own fog density so clouds and terrain
    // agree about how thick the air is.
    t.uHaze.value = lerp(0.000028, 0.000105, clamp01(fin(state.weather.fogDensity, 0) / 0.02));
    // The silver rim is at its most dramatic with a low sun behind the cloud.
    // Moonlight is far too weak to produce a real silver lining, and leaving it
    // on is what makes CG night skies look like underexposed day skies. Both of
    // these ride `moonness` rather than the boolean for the same reason the
    // scatter scale does: a step change here is a visible flicker at dusk.
    const lowSun = clamp01(1.0 - Math.abs(this._lightDir.y));
    t.uSilverIntensity.value = lerp(lerp(0.9, 2.6, lowSun), lerp(0.20, 0.55, lowSun), moonness);
    t.uPowder.value = lerp(1.0, 0.55, moonness);

    // The light march has to reach far enough to see the medium that is
    // actually shadowing each sample. Straight overhead that is a bit more than
    // the slab thickness; with the sun on the horizon the ray runs the long way
    // through the layer and needs several times that, and this is the whole
    // reason a low sun can rake UNDER a cloud deck and light the bases instead
    // of leaving them a flat silhouette. Solving the ladder here keeps the
    // shader down to two uniforms and no per-pixel setup.
    const span = this._p.thickness * lerp(3.8, 1.5, clamp01(Math.abs(this._lightDir.y) * 2.2));
    t.uLightSpan.value = span;
    t.uLightStep.value = span / Math.max(this._lightGeomSum, 1e-3);
    // The powder notch is a fixed depth of medium, scaled with the layer so it
    // stays a constant fraction of a cloud rather than a constant number of
    // metres in a slab that the weather system resizes.
    t.uPowderDepth.value = this._p.thickness * 0.19;
    // Hand the cloud bases over to the light march as the sun drops. See the
    // vertProb comment in TRACE_FRAG - this is what lets a low sun light the
    // undersides instead of the shader insisting they must be in shadow.
    t.uVertProb.value = smoothstep(0.05, 0.34, Math.abs(this._lightDir.y));

    this.shadowMaterial.uniforms.uShadowStrength.value =
      this.shadowStrength * clamp01(fin(sun.visibility, 0) * 1.5);
  }

  _updateCamera(dt, state) {
    const cam = this.ctx.camera;
    cam.updateMatrixWorld();
    this._viewProj.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    this._invViewProj.copy(this._viewProj).invert();

    // Planet centre sits under the camera: the cloud layer is then always level
    // where you stand and curves away toward the horizon, which is what a real
    // cloud deck does. Moving it with the camera is invisible because the noise
    // is sampled in absolute world space.
    this._planetCenter.set(cam.position.x, -PLANET_RADIUS, cam.position.z);

    this.traceMaterial.uniforms.uCamPos.value.copy(cam.position);
    this.resolveMaterial.uniforms.uCamPos.value.copy(cam.position);

    // History rejection, in two stages.
    //
    // A hard binary kill is reserved for genuine discontinuities - a teleport,
    // or a turn so violent that nothing on screen was on screen last frame.
    // Killing the history dumps the resolve back to a blocky 1/4-res image for
    // one frame, so it must not fire during ordinary play.
    //
    // Everything short of that is handled continuously: an angular-RATE signal
    // (normalised by dt, so it means the same thing at 30 and 144 fps) that
    // tightens the variance clip and drops the feedback across the whole frame.
    // Screen-space velocity alone cannot do this job - near the centre of a
    // fast pan the reprojection error is small in pixels but the underlying
    // sample is stale, and that is exactly where the smear used to sit.
    const moved = _v3a.copy(cam.position).sub(this._prevCamPos).length();
    cam.getWorldQuaternion(_quatA);
    const turned = _quatA.angleTo(this._prevCamQuat);
    if (moved > 400 || turned > 1.2) this._historyValid = 0;

    const turnRate = turned / Math.max(dt, 1e-3);          // rad/s
    const moveRate = moved / Math.max(dt, 1e-3);           // m/s
    this.resolveMaterial.uniforms.uMotion.value = clamp(
      Math.max(turnRate / 2.2, moveRate / 260), 0, 1
    );

    // Track the CPU-side sun occlusion for whoever wants a cheap "is the sun
    // behind a cloud" answer without a GPU readback.
    this.sunTransmittance = this._estimateSunTransmittance(cam.position, state);
  }

  _renderTrace(renderer) {
    const idx = this._frame % (CHECKER_DIV * CHECKER_DIV);
    const bx = CHECKER_ORDER[idx * 2];
    const by = CHECKER_ORDER[idx * 2 + 1];
    const t = this.traceMaterial.uniforms;
    t.uBayer.value.set(bx, by);
    t.uFrameIndex.value = this._frame % 64;
    // One R2 step per TRACE of a given history texel, not per frame - see the
    // note at the r2 line in TRACE_FRAG.
    t.uSampleIndex.value = Math.floor(this._frame / (CHECKER_DIV * CHECKER_DIV)) % 64;
    this.resolveMaterial.uniforms.uBayer.value.set(bx, by);

    this._fsMesh.material = this.traceMaterial;
    renderer.setRenderTarget(this._targets.trace);
    renderer.render(this._fsScene, this._fsCamera);
  }

  _renderResolve(renderer) {
    const next = 1 - this._historyIndex;
    const r = this.resolveMaterial.uniforms;
    r.uTrace.value = this._targets.trace.texture;
    r.uHistory.value = this._targets.history[this._historyIndex].texture;
    r.uHistValid.value = this._historyValid;

    this._fsMesh.material = this.resolveMaterial;
    renderer.setRenderTarget(this._targets.history[next]);
    renderer.render(this._fsScene, this._fsCamera);
    this._historyIndex = next;
  }

  _renderShadow(renderer, state) {
    if (--this._shadowTick > 0) return;
    this._shadowTick = this._shadowInterval;

    const rt = this._targets.shadow;
    // Snap the map to whole texels so the shadow pattern does not crawl as the
    // player walks - the single most visible artifact of a camera-following map.
    const texel = this.shadowExtent / rt.width;
    const cam = this.ctx.camera;
    this._shadowCenter.set(
      Math.round(cam.position.x / texel) * texel,
      state.player.groundHeight || 0,
      Math.round(cam.position.z / texel) * texel
    );
    this.shadowMaterial.uniforms.uShadowExtent.value = this.shadowExtent;

    this._fsMesh.material = this.shadowMaterial;
    renderer.setRenderTarget(rt);
    renderer.render(this._fsScene, this._fsCamera);

    this._updateShadowTransform();
  }

  /** Writes a fully-lit shadow map so consumers never see stale darkening when
   *  clouds are switched off. */
  _clearShadowMap() {
    const rt = this._targets.shadow;
    if (!rt) return;
    const renderer = this.renderer;
    const prevRT = renderer.getRenderTarget();
    renderer.getClearColor(_colClear);
    const prevAlpha = renderer.getClearAlpha();
    renderer.setRenderTarget(rt);
    renderer.setClearColor(0xffffff, 1);
    renderer.clear(true, false, false);
    renderer.setClearColor(_colClear, prevAlpha);
    renderer.setRenderTarget(prevRT);
  }

  // -------------------------------------------------------------------------
  // CPU-side queries for sibling systems
  // -------------------------------------------------------------------------

  /** Bilinear sample of the baked weather map, in the same advected frame the
   *  GPU uses. Returns the four raw channels through `out`. */
  _sampleWeather(x, z, out) {
    const data = this._weatherData;
    if (!data) { out.x = out.y = out.z = out.w = 0; return out; }
    const S = WEATHER_SIZE;
    const u = ((x + this._weatherOffset.x) / this.weatherTileSize) * S;
    const v = ((z + this._weatherOffset.y) / this.weatherTileSize) * S;
    const x0 = Math.floor(u), y0 = Math.floor(v);
    const fx = u - x0, fy = v - y0;
    const ix0 = ((x0 % S) + S) % S, iy0 = ((y0 % S) + S) % S;
    const ix1 = (ix0 + 1) % S, iy1 = (iy0 + 1) % S;
    const i00 = (iy0 * S + ix0) * 4, i10 = (iy0 * S + ix1) * 4;
    const i01 = (iy1 * S + ix0) * 4, i11 = (iy1 * S + ix1) * 4;
    const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy);
    const w01 = (1 - fx) * fy, w11 = fx * fy;
    const inv = 1 / 255;
    out.x = (data[i00] * w00 + data[i10] * w10 + data[i01] * w01 + data[i11] * w11) * inv;
    out.y = (data[i00 + 1] * w00 + data[i10 + 1] * w10 + data[i01 + 1] * w01 + data[i11 + 1] * w11) * inv;
    out.z = (data[i00 + 2] * w00 + data[i10 + 2] * w10 + data[i01 + 2] * w01 + data[i11 + 2] * w11) * inv;
    out.w = (data[i00 + 3] * w00 + data[i10 + 3] * w10 + data[i01 + 3] * w01 + data[i11 + 3] * w11) * inv;
    return out;
  }

  /**
   * Cloud coverage 0..1 at a world position, matching the shader's coverage
   * curve at mid-slab height. Cheap enough to call per frame per system.
   */
  getCoverageAt(x, z) {
    if (!this._weatherData) return 0;
    const w = this._sampleWeather(x, z, _weatherScratch);
    const field = w.x * 0.66 + w.y * 0.34;
    // Mirror of coverageFrom() in FIELD_GLSL. It reads the SAME two numbers the
    // shader does rather than re-deriving them, so the two cannot drift apart.
    const lo = this._covLo;
    const hi = this._covHi;
    const t = clamp01((field - lo) / Math.max(hi - lo, 1e-4));
    return t * t * (3 - 2 * t);
  }

  /** Fraction of direct sun reaching (x, z) - the CPU twin of the shadow map,
   *  for systems that need a scalar rather than a texture. */
  _estimateSunTransmittance(pos, state) {
    if (!this._weatherData) return 1;
    const dir = state.sun.direction;
    // Guarded like everything else read from a sibling: this value is published
    // on `this.sunTransmittance` for anyone to poll, and Math.max(NaN, 0.02) is
    // NaN, which would walk straight through the weather lookup and hand every
    // caller a NaN.
    const elev = Math.max(fin(dir.y, 1), 0.02);
    // Where the ray from this point crosses the middle of the cloud slab.
    const reach = (this._p.altitude + this._p.thickness * 0.5) / elev;
    const cx = fin(pos.x, 0) + fin(dir.x, 0) * reach;
    const cz = fin(pos.z, 0) + fin(dir.z, 0) * reach;
    const cov = this.getCoverageAt(cx, cz);
    const optical = cov * this._p.density * this._p.thickness * this.extinctionScale * this._p.absorption * 0.45;
    const t = Math.exp(-optical);
    return lerp(1, t, clamp01(fin(state.sun.visibility, 0) * 1.5));
  }

  get shadowTexture() { return this._targets.shadow ? this._targets.shadow.texture : null; }
  get cloudShadowMap() { return this.shadowTexture; }
  get cloudShadowTexture() { return this.shadowTexture; }
  get cloudShadowMatrix() { return this.shadowMatrix; }
  getShadowTexture() { return this.shadowTexture; }
  getShadowMatrix() { return this.shadowMatrix; }
  getCloudShadowUniforms() { return this.shadowUniforms; }

  // -------------------------------------------------------------------------

  dispose() {
    if (this.mesh) {
      this.ctx.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
    }
    this.compositeMaterial?.dispose();
    this.traceMaterial?.dispose();
    this.resolveMaterial?.dispose();
    this.shadowMaterial?.dispose();
    this._fsBootMaterial?.dispose();
    this._fsGeom?.dispose();
    this._disposeTargets();
    this._targets.shadow?.dispose();
    this._targets.shadow = null;
    this.shadowUniforms.uCloudShadowMap.value = null;
    // Textures handed to the TextureFactory are the factory's to dispose; only
    // the ones we had to own ourselves are freed here.
    for (const t of this._ownedTextures) t.dispose();
    this._ownedTextures.length = 0;
    this._weatherData = null;
    this._covLUT = null;
    this.ready = false;
  }
}
