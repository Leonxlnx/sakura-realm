/**
 * grass.js - dense instanced meadow.
 *
 * Rewritten from scratch. The previous implementation submitted ~90k instances and
 * 320k triangles per frame that produced zero visible pixels: forcing every bend
 * term to zero and the blade width to 300 screen pixels still rendered nothing, so
 * the strips were degenerate before rasterisation. Rather than keep bisecting a
 * 1900-line shader, this is a clean, deliberately simple implementation whose
 * resting pose is provable by inspection.
 *
 * DESIGN
 *
 *   One blade geometry, five segments, eleven vertices, nine triangles. Every
 *   blade in the world is that geometry, instanced.
 *
 *   The world is a grid of chunks that follows the camera. Each chunk owns one
 *   instance buffer filled deterministically from its integer cell coordinates, so
 *   a chunk that scrolls out and back is bit-identical - no popping, no reshuffle.
 *   Instances are generated in Halton order, which makes ANY PREFIX of the buffer
 *   an evenly distributed subset. Distance LOD is therefore just a smaller
 *   `instanceCount`: no second geometry, no second buffer, no transition seam.
 *
 *   Bending uses the constant-curvature arc below. Its A -> 0 limit is exactly
 *   (0, H*s, 0), so with no wind and no lean every blade is exactly vertical.
 *   That property is what the old system lost, and it is worth the closed form.
 *
 * COST at HIGH, 1080p, on the reference Radeon 780M: ~55k blades drawn across
 * ~45 chunks after frustum culling, 9 triangles each.
 */

import {
  BufferGeometry,
  BufferAttribute,
  InstancedBufferGeometry,
  InstancedBufferAttribute,
  Mesh,
  Group,
  ShaderMaterial,
  Vector2,
  Vector3,
  Color,
  Sphere,
  DoubleSide,
  DynamicDrawUsage,
} from 'three';

import { createNoise, makeRNG, halton, clamp01, lerp } from '../core/math.js';

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

/**
 * Base chunk edge length in metres, at ground level.
 *
 * Reach is ((GRID-1)/2) * CHUNK_BASE. Raising the chunk SIZE rather than the
 * grid count buys view distance for free: draw calls stay fixed at the chunk
 * count, and only the per-chunk instance budget has to grow to hold density.
 * Adding rings instead would have cost ~40 more draw calls, and on this GPU
 * draw calls are the scarcer resource.
 */
const CHUNK_BASE = 15;

/**
 * Altitude clipmap. From the air you cannot resolve individual blades, but you
 * CAN see the edge of the field - so above each of these heights the chunk grid
 * is scaled up by the matching factor, trading blade density for reach at a
 * constant chunk count and therefore a constant draw-call cost.
 *
 * Reach becomes 72 m walking, 144 m / 288 m / 576 m climbing. Entries are
 * ordered low to high and carry hysteresis (`exit`) so hovering exactly on a
 * boundary cannot thrash the whole grid.
 */
const ALTITUDE_BANDS = [
  { enter: 0, exit: 0, scale: 1 },
  { enter: 18, exit: 14, scale: 2 },
  { enter: 55, exit: 44, scale: 4 },
  { enter: 150, exit: 120, scale: 8 },
];
/**
 * Grid is GRID x GRID chunks, centred on the camera; must be odd. Reach is
 * ((GRID-1)/2) * CHUNK metres, and that - not the fade distance - is the real
 * limit on how far the field extends. Every extra ring costs draw calls, so the
 * field ends at 72 m and the fade is deliberately long so it dissolves into the
 * terrain colour instead of ending on a visible edge.
 */
const GRID = 13;
/**
 * Instance buffer capacity per chunk. Near chunks draw all of it.
 *
 * 24000 over a 12 m chunk is ~167 blades/m2 immediately around the camera,
 * which is the density at which the ground stops being visible between blades.
 * The previous 7600 (~53/m2) left the soil showing through everywhere and was
 * the main reason the field read as sparse from standing height.
 *
 * This is affordable only because the distance falloff below was tightened at
 * the same time: total instance count actually DROPS, because the old curve was
 * spending most of the budget on chunks 40-70 m out that occupy a handful of
 * pixels each.
 */
const MAX_PER_CHUNK = 26000;

/**
 * Blade segments. 4 -> 9 verts, 7 tris.
 *
 * Dropped from 5 once the density went up: at 450k blades the vertex and
 * triangle load is the dominant cost, and 7 tris instead of 9 is a flat 22 %
 * saving across the whole field. The arc these approximate is smooth and the
 * blade is only a few pixels wide, so the lost segment is not visible - whereas
 * the frame time it buys is what keeps the field this dense at all.
 */
const SEGMENTS = 4;

/**
 * Sward modes - how the field is cut.
 *
 * `densityMul` is deliberately INVERSE to height. A blade hides ground roughly
 * in proportion to its own projected area, so halving the height halves the
 * cover each blade gives; keeping the count constant would open the soil right
 * up, which is exactly what a naive "short grass" setting looks like. Short
 * modes therefore get many more, slightly wider blades - which is also what a
 * real mown lawn is: a much denser stand of much smaller leaves.
 *
 * `hMin`/`hMax` are metres before per-clump scaling.
 */
export const SWARD_MODES = {
  // `meadowNear` pulls terrain.js's earth -> meadow ground blend toward the
  // camera. Short blades physically cannot hide the soil no matter how many of
  // them there are, so the substrate itself has to read as turf; with a tall
  // sward the blades do the hiding and honest earth underneath is correct.
  // `droopMul` above 1 lays short blades over so they overlap and close the mat,
  // which is also what mown grass actually does.
  // `chunkScale` is the real density dial, not `densityMul`. Per-chunk instance
  // count is capped at MAX_PER_CHUNK, and the near chunks already sit AT that
  // cap - so multiplying density there does exactly nothing, which is why the
  // first lawn attempt drew the same blade count as tall grass, only shorter,
  // and read as sparse spikes on bare ground. Shrinking the chunk puts the same
  // capped budget over less area, which is a real increase in blades per m2:
  // 26 000 over a 15 m chunk is 115/m2, over 8.2 m it is 385/m2.
  // The cost is reach - ((GRID-1)/2) * chunk - and that is the right trade,
  // because a mown lawn is read close up and terrain's meadow blend carries the
  // distance.
  lawn:   { label: 'Lawn',   hMin: 0.060, hMax: 0.185, chunkScale: 0.55, densityMul: 1.0, widthMul: 1.85, droopMul: 1.35, meadowNear: 0.04 },
  meadow: { label: 'Meadow', hMin: 0.190, hMax: 0.600, chunkScale: 0.75, densityMul: 1.0, widthMul: 1.25, droopMul: 0.90, meadowNear: 0.20 },
  tall:   { label: 'Tall',   hMin: 0.420, hMax: 1.480, chunkScale: 1.00, densityMul: 1.0, widthMul: 1.00, droopMul: 1.00, meadowNear: 1.00 },
};
const DEFAULT_MODE = 'tall';
/**
 * Blade HALF-width at the base, metres. Real meadow grass is 4-10 mm across, so
 * the half-width belongs around 0.003. The first pass used 0.008-0.019, which
 * rendered 2-4 cm ribbons that read as leeks rather than grass.
 */
const W_MIN = 0.0026;
const W_MAX = 0.0062;

/** Maximum bend angle at the tip under a full gale, radians. */
const THETA_MAX = 1.25;
/**
 * Resting droop, radians, before the per-blade height weighting. Tall meadow
 * grass arcs 40-70 degrees at the tip under its own weight; a field of straight
 * upright blades is the single loudest "procedural grass" tell.
 */
const REST_LEAN = 0.62;

/** Radius inside which grass is thinned as if trampled, around the tree trunk. */
const TRAMPLE_RADIUS = 3.4;

/**
 * `falloff` is the half-density distance in metres: instance count per chunk
 * scales by 1/(1+(d/falloff)^2). Tightening it concentrates the budget in the
 * near field, where a blade is many pixels tall, and starves the far field,
 * where a whole chunk covers a few dozen pixels and the terrain colour carries
 * the look anyway.
 */
const QUALITY_PRESETS = {
  low: { density: 0.26, radius: 52, falloff: 7 },
  medium: { density: 0.5, radius: 72, falloff: 9 },
  high: { density: 1.0, radius: 90, falloff: 13 },
  ultra: { density: 1.25, radius: 90, falloff: 16 },
};

// Scratch - module scope so update() never allocates.
const _v3 = new Vector3();

/**
 * Resolution of the per-chunk coarse field lattice. 17x17 over a 15 m chunk is a
 * sample every 94 cm, comfortably finer than the shortest wavelength in the
 * height and clump fields, and 289 samples instead of 26 000.
 */
const GRID_SAMPLES = 17;

/**
 * Allocation-free per-instance PRNG.
 *
 * The fill needs a stream seeded from the instance INDEX so that slicing the
 * work across frames cannot change a byte. Calling makeRNG() per instance does
 * that correctly but allocates a closure and five helper methods each time - 
 * 156 000 allocations per chunk, which turned a 22 ms fill into an 874 ms
 * garbage-collection storm. This is the same mulberry32 stream over one
 * module-scope word instead.
 */
let _rngState = 0;
function _seedAt(s) {
  _rngState = s >>> 0;
}
function _rand() {
  _rngState = (_rngState + 0x6d2b79f5) | 0;
  let t = Math.imul(_rngState ^ (_rngState >>> 15), 1 | _rngState);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
/** Instances filled per slice. ~2600 keeps a slice near 2 ms of CPU. */
const FILL_SLICE = 2600;
const _fieldH = new Float32Array(GRID_SAMPLES * GRID_SAMPLES);
const _fieldC = new Float32Array(GRID_SAMPLES * GRID_SAMPLES);
const _fieldP = new Float32Array(GRID_SAMPLES * GRID_SAMPLES);

// ---------------------------------------------------------------------------
// Blade geometry
// ---------------------------------------------------------------------------

/**
 * A blade as a triangle strip narrowing to a point.
 *
 * aParam.x = s, arc-length fraction 0..1 along the blade
 * aParam.y = side, -1 or +1 (0 at the tip vertex)
 */
function createBladeGeometry() {
  const rows = SEGMENTS + 1;
  const vertexCount = rows * 2 - 1; // tip is a single vertex
  const pos = new Float32Array(vertexCount * 3);
  const par = new Float32Array(vertexCount * 2);

  let v = 0;
  for (let i = 0; i < rows; i++) {
    const s = i / SEGMENTS;
    if (i === SEGMENTS) {
      pos[v * 3] = 0; pos[v * 3 + 1] = s; pos[v * 3 + 2] = 0;
      par[v * 2] = s; par[v * 2 + 1] = 0;
      v++;
    } else {
      for (const side of [-1, 1]) {
        pos[v * 3] = side; pos[v * 3 + 1] = s; pos[v * 3 + 2] = 0;
        par[v * 2] = s; par[v * 2 + 1] = side;
        v++;
      }
    }
  }

  const indices = [];
  for (let i = 0; i < SEGMENTS - 1; i++) {
    const a = i * 2, b = i * 2 + 1, c = (i + 1) * 2, d = (i + 1) * 2 + 1;
    indices.push(a, c, b, b, c, d);
  }
  // Final pair -> tip triangle.
  const a = (SEGMENTS - 1) * 2, b = (SEGMENTS - 1) * 2 + 1, tip = vertexCount - 1;
  indices.push(a, tip, b);

  const geo = new BufferGeometry();
  geo.setAttribute('position', new BufferAttribute(pos, 3));
  geo.setAttribute('aParam', new BufferAttribute(par, 2));
  geo.setIndex(indices);
  return geo;
}

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

// ShaderMaterial (not Raw): three injects precision, position, modelViewMatrix,
// projectionMatrix and cameraPosition. Redeclaring any of them is a compile error.
const VERT = /* glsl */ `
attribute vec2 aParam;      // s along blade, side (-1|0|1)

attribute vec4 iPos;        // xyz world base, w height (m)
attribute vec4 iShape;      // x halfWidth, y yaw, z stiffness, w restLean
attribute vec4 iVar;        // x hueMix, y valueMix, z dryness, w phase

uniform float uTime;
uniform vec2  uWindDir;
uniform float uWindStrength;   // already includes the gust envelope
uniform float uTurbulence;
uniform vec2  uFade;           // x fade-out start, y fade-out end (metres)
uniform vec3  uTree;           // xz trunk centre, z unused -> w in uTreeR
uniform float uTreeR;
uniform float uPixelScale;     // radians of view angle per pixel

varying vec3 vNormal;
varying vec3 vWorld;
varying vec3 vTangent;
varying vec4 vShade;           // s, ao, dryness, valueMix

/**
 * Constant-curvature arc.
 *
 * A blade of length H whose tip has turned through angle A, evaluated at arc
 * fraction s, in the plane spanned by +Y and the horizontal unit vector d:
 *
 *     y(s) = H * sin(A*s) / A
 *     x(s) = H * (1 - cos(A*s)) / A
 *
 * As A -> 0 these tend to H*s and 0. The small-angle branch below uses the
 * leading Taylor terms so the transition is smooth and the A == 0 case is EXACTLY
 * vertical - which is the entire resting pose of the field.
 */
void arc(float A, float s, float H, out float along, out float lateral, out float ang) {
  ang = A * s;
  if (abs(A) < 1e-3) {
    along = H * s * (1.0 - ang * ang / 6.0);
    lateral = H * ang * s * 0.5;
  } else {
    along = H * sin(ang) / A;
    lateral = H * (1.0 - cos(ang)) / A;
  }
}

void main() {
  float s = aParam.x;
  float side = aParam.y;

  vec3 base = iPos.xyz;
  float H = iPos.w;

  float dist = distance(base, cameraPosition);

  // Distance dissolve. Height alone carries it: a blade whose height reaches zero
  // is exactly degenerate, whereas fading width alone would leave a hairline
  // lying on the ground.
  float fade = 1.0 - smoothstep(uFade.x, uFade.y, dist);
  H *= fade;

  // Thin the sward where the trunk stands, as if walked on.
  float toTree = distance(base.xz, uTree.xy);
  H *= mix(0.35, 1.0, smoothstep(uTreeR * 0.35, uTreeR, toTree));

  // --- bend ---------------------------------------------------------------
  float yaw = iShape.y;
  vec2 facing = vec2(cos(yaw), sin(yaw));
  float stiffness = iShape.z;

  // Quadratic drag against a saturating stiffness: a gale flattens the field
  // toward THETA_MAX rather than driving blades through the ground.
  float q = uWindStrength / stiffness;
  float bend = (q * q) / (1.0 + q * q);

  // Per-blade flutter, tip-weighted, only meaningful once there is some wind.
  float flutter = sin(uTime * 5.3 + iVar.w * 6.2831) * uTurbulence * 0.16
                * (0.3 + 0.7 * bend);

  vec2 B = facing * iShape.w                       // resting lean
         + uWindDir * (bend * ${THETA_MAX.toFixed(3)})   // wind
         + vec2(-uWindDir.y, uWindDir.x) * flutter;      // cross-wind flutter

  float A = length(B);
  vec2 d = A > 1e-5 ? B / A : vec2(0.0);

  float along, lateral, ang;
  arc(A, s, H, along, lateral, ang);

  vec3 centre = base + vec3(d.x * lateral, along, d.y * lateral);

  // Tangent along the blade at s. At A == 0 this is exactly +Y.
  vec3 tangent = normalize(vec3(d.x * sin(ang), cos(ang), d.y * sin(ang)));

  // Width axis is perpendicular to the bend plane, so the blade keeps its area
  // as it curves instead of foreshortening into a line.
  vec3 widthAxis = normalize(vec3(-d.y, 0.0, d.x) + vec3(1e-4, 0.0, 0.0));

  // Taper: full width at the base, pinched to nothing at the tip.
  float taper = pow(1.0 - s, 0.62);
  float halfW = iShape.x * taper;

  // Never let a blade fall below roughly one pixel wide, or the field turns into
  // a shimmering dashed mess at distance. Capped so near blades keep true width.
  float minW = dist * uPixelScale * 0.75;
  halfW = clamp(max(halfW, minW), 0.0, iShape.x * 3.5);

  vec3 world = centre + widthAxis * (side * halfW);

  // Camber: the blade is a shallow trough, not a flat ribbon. Bowing the cross
  // section gives it a real surface normal and kills the paper look.
  float camber = (1.0 - side * side) * halfW * 0.55;
  world += tangent * 0.0 + normalize(cross(widthAxis, tangent)) * camber;

  vec3 faceN = normalize(cross(widthAxis, tangent));
  // Rotate the normal outward across the width so the blade shades as a curved
  // surface rather than a plane.
  vNormal = normalize(faceN + widthAxis * (side * 0.55));
  vTangent = tangent;
  vWorld = world;
  vShade = vec4(s, mix(0.35, 1.0, s), iVar.z, iVar.y);

  gl_Position = projectionMatrix * modelViewMatrix * vec4(world, 1.0);
}
`;

const FRAG = /* glsl */ `
uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform float uSunIntensity;
uniform vec3 uSkyColor;
uniform vec3 uGroundColor;
uniform float uAmbient;
uniform vec3 uColBase;
uniform vec3 uColTip;
uniform vec3 uColDry;
uniform float uWetness;
uniform vec3 uFogColor;
uniform float uFogDensity;

varying vec3 vNormal;
varying vec3 vWorld;
varying vec3 vTangent;
varying vec4 vShade;

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(cameraPosition - vWorld);
  // Two-sided: a blade seen from behind must not go black.
  if (dot(N, V) < 0.0) N = -N;

  vec3 L = normalize(uSunDir);

  float s = vShade.x;
  vec3 albedo = mix(uColBase, uColTip, smoothstep(0.35, 1.0, s));
  albedo = mix(albedo, uColDry, vShade.z * smoothstep(0.45, 1.0, s));
  albedo *= mix(0.82, 1.18, vShade.w);
  albedo *= mix(1.0, 0.68, uWetness);

  // Ambient occlusion down the blade: the base of the sward is in shadow, and
  // this gradient is most of what gives the field depth.
  float ao = vShade.y;

  float ndl = max(dot(N, L), 0.0);
  vec3 diffuse = uSunColor * uSunIntensity * ndl;

  // Transmission. A grass blade is thin enough to glow when the sun is behind
  // it; this single term does more for realism than anything else here.
  float back = max(dot(-N, L), 0.0);
  float wrap = max(dot(N, L) * 0.5 + 0.5, 0.0);
  float trans = pow(max(dot(V, -L), 0.0), 3.0) * 0.65 + back * 0.35;
  vec3 transmission = uSunColor * uSunIntensity * trans * 1.35 * vec3(0.85, 1.0, 0.55);

  vec3 ambient = mix(uGroundColor, uSkyColor, N.y * 0.5 + 0.5) * uAmbient;

  vec3 color = albedo * (diffuse * 0.85 + ambient * ao + wrap * 0.12) + albedo * transmission;

  // Specular sheen along the blade - sharp when wet, soft when dry.
  vec3 Hv = normalize(L + V);
  float sheen = pow(max(dot(N, Hv), 0.0), mix(28.0, 90.0, uWetness));
  color += uSunColor * sheen * mix(0.05, 0.30, uWetness) * uSunIntensity * ao;

  // Exponential-squared distance fade into the sky.
  float dist = length(cameraPosition - vWorld);
  float f = 1.0 - exp(-pow(dist * uFogDensity, 2.0));
  color = mix(color, uFogColor, clamp(f, 0.0, 1.0));

  gl_FragColor = vec4(color, 1.0);
}
`;

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

export class GrassField {
  constructor(ctx) {
    this.ctx = ctx;
    this.scene = ctx.scene;
    this.camera = ctx.camera;
    this.state = ctx.state;

    this.group = new Group();
    this.group.name = 'grass';
    this.group.frustumCulled = false;
    this.scene.add(this.group);

    this.noise = createNoise(9182);
    this.blade = createBladeGeometry();

    /** chunkKey -> chunk record */
    this.chunks = new Map();
    /** Chunk records not currently in the grid, ready to be refilled. */
    this.pool = [];

    this.stats = { chunks: 0, visible: 0, instances: 0 };

    this._preset = QUALITY_PRESETS.high;
    this._lastCellX = Infinity;
    this._lastCellZ = Infinity;
    /** Current chunk edge length; grows with altitude, see ALTITUDE_BANDS. */
    this._chunk = CHUNK_BASE;
    this._band = 0;
    this.modeName = DEFAULT_MODE;
    this._mode = SWARD_MODES[DEFAULT_MODE];
    /** Chunk keys waiting to be filled; drained under a time budget each frame. */
    this._pending = [];
    /** Chunk currently being filled across frames, if any. */
    this._building = null;
    this._firstBuild = true;

    this.material = new ShaderMaterial({
      name: 'GrassField',
      vertexShader: VERT,
      fragmentShader: FRAG,
      side: DoubleSide,
      // Everything is opaque: alpha blending across 50k overlapping blades would
      // be both wrong (no sort order) and ruinously expensive on an iGPU.
      transparent: false,
      depthWrite: true,
      uniforms: {
        uTime: { value: 0 },
        uWindDir: { value: new Vector2(1, 0) },
        uWindStrength: { value: 0 },
        uTurbulence: { value: 0.4 },
        uFade: { value: new Vector2(70, 100) },
        uTree: { value: new Vector2(0, 0) },
        uTreeR: { value: TRAMPLE_RADIUS },
        uPixelScale: { value: 0.001 },

        uSunDir: { value: new Vector3(0, 1, 0) },
        uSunColor: { value: new Color(1, 1, 1) },
        uSunIntensity: { value: 1 },
        uSkyColor: { value: new Color(0.5, 0.7, 1) },
        uGroundColor: { value: new Color(0.2, 0.18, 0.14) },
        uAmbient: { value: 1 },

        // Low-saturation meadow greens. Anything more chromatic reads as arcade
        // turf and breaks the muted art direction immediately.
        uColBase: { value: new Color('#3a5630') },
        uColTip: { value: new Color('#82a552') },
        uColDry: { value: new Color('#b3ad6a') },
        uWetness: { value: 0 },

        uFogColor: { value: new Color(0.72, 0.78, 0.86) },
        uFogDensity: { value: 0.0022 },
      },
    });
  }

  link(systems) {
    this.terrain = systems.terrain;
    this.wind = systems.wind;
    this.tree = systems.tree;
    // The mode may have been chosen before terrain existed to be told about it.
    this.terrain?.setMeadowNear?.(this._mode.meadowNear ?? 1);
  }

  async init() {
    this.onQualityChange(this.state.quality);
  }

  /**
   * Switch how the field is cut: 'lawn' | 'meadow' | 'tall'.
   *
   * Every chunk's instance buffer encodes the mode's heights and widths, so the
   * whole grid is retired and refilled. That is a few milliseconds of CPU, which
   * is why it is a discrete user action rather than something animated.
   */
  setMode(name) {
    const mode = SWARD_MODES[name];
    if (!mode || name === this.modeName) return;
    this.modeName = name;
    this._mode = mode;
    this._chunk = CHUNK_BASE * ALTITUDE_BANDS[this._band].scale * mode.chunkScale;
    this._applyFade();
    for (const record of this.chunks.values()) {
      this.group.remove(record.mesh);
      this.pool.push(record);
    }
    this.chunks.clear();
    this._lastCellX = Infinity;
    this._lastCellZ = Infinity;
    // The grid is empty again, so refill it on the fast budget rather than
    // dribbling it in over two seconds of visibly bare ground.
    this._firstBuild = true;
    this.terrain?.setMeadowNear?.(mode.meadowNear ?? 1);
    this.ctx.bus?.emit('grass:mode', name);
  }

  /** Cycle lawn -> meadow -> tall -> lawn. Bound to G. */
  cycleMode() {
    const names = Object.keys(SWARD_MODES);
    this.setMode(names[(names.indexOf(this.modeName) + 1) % names.length]);
  }

  onQualityChange(quality) {
    this._preset = QUALITY_PRESETS[quality.tier] || QUALITY_PRESETS.high;
    this._applyFade();
  }

  /** Fade window follows whatever the grid currently reaches. */
  _applyFade() {
    const scale = ALTITUDE_BANDS[this._band ?? 0].scale;
    const reach = ((GRID - 1) / 2) * CHUNK_BASE * scale * (this._mode?.chunkScale ?? 1);
    const radius = Math.min(this._preset.radius * scale, reach);
    // Long dissolve: the last 45% of the reach is spent fading, so the field
    // thins into the ground colour rather than stopping at a line.
    this.material.uniforms.uFade.value.set(radius * 0.55, radius);
    // Force a full rebuild so density changes take effect immediately.
    this._lastCellX = Infinity;
    this._lastCellZ = Infinity;
  }

  resize(width, height) {
    // Radians of vertical view angle per pixel - used for the sub-pixel width floor.
    const fov = (this.camera.fov * Math.PI) / 180;
    this.material.uniforms.uPixelScale.value = fov / Math.max(1, height);
  }

  // -------------------------------------------------------------------------

  /**
   * Deterministic per-chunk instance fill, RESUMABLE. Same cell in -> same bytes
   * out regardless of how the work is sliced, because the RNG is re-seeded and
   * fast-forwarded rather than carried across calls.
   *
   * Measured on the reference machine: a whole 26 000-instance chunk costs
   * 21.9 ms of CPU plus ~18 ms to upload its 1.19 MB of instance data. One chunk
   * is far too much to do inside a frame, and it cannot be split by chunk
   * because a chunk is atomic - which is why walking across a cell boundary
   * still dropped a frame even after the per-blade field sampling was made
   * cheap. Slicing it internally is the only fix that keeps both the density and
   * the draw-call count.
   *
   * @returns {boolean} true when the chunk is complete
   */
  _fillChunkSlice(record, cellX, cellZ, count) {
    const M = this._mode;
    const originX = cellX * this._chunk;
    const originZ = cellZ * this._chunk;

    const iPos = record.iPos;
    const iShape = record.iShape;
    const iVar = record.iVar;
    const terrain = this.terrain;
    const noise = this.noise;

    // Per-instance RNG seeded from the instance index, so a slice boundary
    // cannot change a single byte. A carried-over stream would make the result
    // depend on how the work happened to be divided.
    const seed = (cellX * 73856093) ^ (cellZ * 19349663);

    // ---- coarse field pre-pass ----------------------------------------------
    // Terrain height and both clump fields are LOW FREQUENCY - the shortest
    // wavelength in play is the clump field's ~22 m - so sampling them per blade
    // was paying 26 000 terrain queries and 52 000 fbm evaluations to reproduce
    // a surface that a 9x9 lattice captures to within a millimetre. That cost
    // ~50 ms per chunk, and one chunk fill is atomic, so no amount of spreading
    // work across frames could hide it: walking across a cell boundary dropped a
    // whole frame. Bilinear interpolation off this grid is ~300x cheaper and
    // visually identical.
    const N = GRID_SAMPLES;
    const fh = record.fieldH;
    const fc = record.fieldC;
    const fp = record.fieldP;
    // Only on the first slice; the lattice is per-chunk and does not change.
    if (record.progress === 0) {
      const step = this._chunk / (N - 1);
      for (let gz = 0; gz < N; gz++) {
        for (let gx = 0; gx < N; gx++) {
          const sx = originX + gx * step;
          const sz = originZ + gz * step;
          const k = gz * N + gx;
          fh[k] = terrain ? terrain.getHeight(sx, sz) : 0;
          fc[k] = noise.fbm2D(sx * 0.045, sz * 0.045, 3) * 0.5 + 0.5;
          fp[k] = noise.fbm2D(sx * 0.011 + 40, sz * 0.011 - 17, 2) * 0.5 + 0.5;
        }
      }
      record.minY = Infinity;
      record.maxY = -Infinity;
    }
    /** Bilinear sample of one coarse field at chunk-local fractions u, v. */
    const sample = (field, u, v) => {
      const fx = u * (N - 1);
      const fz = v * (N - 1);
      const x0 = Math.min(N - 2, fx | 0);
      const z0 = Math.min(N - 2, fz | 0);
      const tx = fx - x0;
      const tz = fz - z0;
      const a = field[z0 * N + x0];
      const b = field[z0 * N + x0 + 1];
      const c = field[(z0 + 1) * N + x0];
      const d = field[(z0 + 1) * N + x0 + 1];
      return (a + (b - a) * tx) * (1 - tz) + (c + (d - c) * tx) * tz;
    };

    const end = Math.min(MAX_PER_CHUNK, record.progress + count);
    let minY = record.minY;
    let maxY = record.maxY;

    for (let i = record.progress; i < end; i++) {
      _seedAt(seed ^ (i * 2654435761));
      // Halton keeps any prefix of the buffer evenly spread, which is what makes
      // instanceCount alone a valid LOD.
      const u = halton(i + 1, 2);
      const v = halton(i + 1, 3);
      const x = originX + u * this._chunk;
      const z = originZ + v * this._chunk;

      const y = sample(fh, u, v);

      // Clumping: a low-frequency field thins the sward into patches and stands
      // rather than a uniform carpet.
      const clump = sample(fc, u, v);
      const patch = sample(fp, u, v);
      const density = clamp01(clump * 0.65 + patch * 0.55);

      // Rejected blades collapse to zero height rather than being skipped, so the
      // buffer layout stays fixed and the prefix property survives.
      //
      // 1.15 killed roughly half the buffer, so the effective density was half
      // the nominal one and the thin patches went bald to bare soil. At 1.9 the
      // sward stays closed and the clump field expresses itself through HEIGHT
      // variation instead of through gaps, which is what a real meadow does.
      const alive = _rand() < density * 1.9 ? 1 : 0;

      const heightScale = lerp(0.72, 1.28, patch) * lerp(0.85, 1.15, _rand());
      const h = alive ? lerp(M.hMin, M.hMax, clump) * heightScale : 0;

      iPos[i * 4] = x;
      iPos[i * 4 + 1] = y;
      iPos[i * 4 + 2] = z;
      iPos[i * 4 + 3] = h;

      // Gravity droop scales with how tall the blade is: a long blade cannot hold
      // itself up, and this correlation is most of what separates a meadow from a
      // bristle brush. Squared so the tallest blades arc markedly harder.
      const tall = clamp01((h - M.hMin) / Math.max(1e-4, M.hMax - M.hMin));
      const droop = REST_LEAN * M.droopMul * (0.22 + 1.45 * tall * tall) * (0.55 + 0.9 * _rand());

      iShape[i * 4] = lerp(W_MIN, W_MAX, _rand()) * lerp(0.8, 1.25, tall) * M.widthMul;
      iShape[i * 4 + 1] = _rand() * Math.PI * 2;
      iShape[i * 4 + 2] = lerp(0.75, 1.6, _rand());
      iShape[i * 4 + 3] = droop;

      iVar[i * 4] = _rand();
      iVar[i * 4 + 1] = _rand();
      iVar[i * 4 + 2] = clamp01(patch * 0.8 + _rand() * 0.4);
      iVar[i * 4 + 3] = _rand();

      if (h > 0) {
        if (y < minY) minY = y;
        if (y + h > maxY) maxY = y + h;
      }
    }

    record.minY = minY;
    record.maxY = maxY;
    record.progress = end;
    if (end < MAX_PER_CHUNK) return false;

    if (minY === Infinity) { minY = 0; maxY = 1; }

    // One upload per chunk, at completion - uploading each slice would multiply
    // the 1.19 MB transfer by the number of slices.
    record.iPosAttr.needsUpdate = true;
    record.iShapeAttr.needsUpdate = true;
    record.iVarAttr.needsUpdate = true;

    // A correct bounding sphere is essential: the previous implementation left a
    // unit sphere at the origin, so three culled chunks by the wrong volume.
    const cx = originX + this._chunk * 0.5;
    const cz = originZ + this._chunk * 0.5;
    const cy = (minY + maxY) * 0.5;
    const halfDiag = Math.sqrt(2) * this._chunk * 0.5;
    const radius = Math.sqrt(halfDiag * halfDiag + ((maxY - minY) * 0.5) ** 2) + 1.0;
    record.geometry.boundingSphere = new Sphere(new Vector3(cx, cy, cz), radius);
    record.mesh.position.set(0, 0, 0);
    record.centre.set(cx, cy, cz);
    record.cellX = cellX;
    record.cellZ = cellZ;
    return true;
  }

  _acquireChunk() {
    const existing = this.pool.pop();
    if (existing) return existing;

    const geometry = new InstancedBufferGeometry();
    geometry.index = this.blade.index;
    geometry.setAttribute('position', this.blade.getAttribute('position'));
    geometry.setAttribute('aParam', this.blade.getAttribute('aParam'));

    const iPos = new Float32Array(MAX_PER_CHUNK * 4);
    const iShape = new Float32Array(MAX_PER_CHUNK * 4);
    const iVar = new Float32Array(MAX_PER_CHUNK * 4);

    const iPosAttr = new InstancedBufferAttribute(iPos, 4);
    const iShapeAttr = new InstancedBufferAttribute(iShape, 4);
    const iVarAttr = new InstancedBufferAttribute(iVar, 4);
    iPosAttr.setUsage(DynamicDrawUsage);
    iShapeAttr.setUsage(DynamicDrawUsage);
    iVarAttr.setUsage(DynamicDrawUsage);

    geometry.setAttribute('iPos', iPosAttr);
    geometry.setAttribute('iShape', iShapeAttr);
    geometry.setAttribute('iVar', iVarAttr);
    geometry.instanceCount = MAX_PER_CHUNK;

    const mesh = new Mesh(geometry, this.material);
    mesh.frustumCulled = true;
    mesh.matrixAutoUpdate = false;
    mesh.updateMatrix();
    mesh.castShadow = false;
    mesh.receiveShadow = false;

    return {
      geometry, mesh, iPos, iShape, iVar,
      iPosAttr, iShapeAttr, iVarAttr,
      centre: new Vector3(), cellX: 0, cellZ: 0,
      progress: 0, minY: Infinity, maxY: -Infinity,
      fieldH: new Float32Array(GRID_SAMPLES * GRID_SAMPLES),
      fieldC: new Float32Array(GRID_SAMPLES * GRID_SAMPLES),
      fieldP: new Float32Array(GRID_SAMPLES * GRID_SAMPLES),
    };
  }

  _rebuildGrid(camCellX, camCellZ) {
    const half = (GRID - 1) >> 1;
    const keep = new Set();

    for (let dz = -half; dz <= half; dz++) {
      for (let dx = -half; dx <= half; dx++) {
        const cx = camCellX + dx;
        const cz = camCellZ + dz;
        // Circular footprint: the corners of a square grid are past the fade
        // distance anyway and would only cost draw calls.
        if (dx * dx + dz * dz > half * half + half) continue;
        keep.add(`${cx},${cz}`);
      }
    }

    // Retire chunks that fell out of range.
    for (const [key, record] of this.chunks) {
      if (keep.has(key)) continue;
      this.group.remove(record.mesh);
      this.chunks.delete(key);
      this.pool.push(record);
    }

    // QUEUE the new ones rather than filling them here.
    //
    // _fillChunk walks MAX_PER_CHUNK instances, each doing a terrain.getHeight
    // and three fbm samples. At 26 000 blades that is several milliseconds per
    // chunk, and crossing a cell boundary while walking retires and re-fills a
    // whole row at once - tens of chunks, hundreds of milliseconds, inside one
    // frame. That is the "fps is fine, then suddenly 0, then fine again" stall,
    // and it also drags the adaptive-resolution controller down and back up,
    // which is the flashing and blurring that comes with it.
    this._pending.length = 0;
    for (const key of keep) {
      if (this.chunks.has(key)) continue;
      this._pending.push(key);
    }
    // Nearest first: the chunk you are standing in must appear this frame, the
    // one 80 m away can wait a few.
    const cx0 = (camCellX + 0.5) * this._chunk;
    const cz0 = (camCellZ + 0.5) * this._chunk;
    this._pending.sort((a, b) => this._keyDistSq(a, cx0, cz0) - this._keyDistSq(b, cx0, cz0));

    this.stats.chunks = this.chunks.size + this._pending.length;
  }

  /** Squared distance from a chunk key's centre to a world point. */
  _keyDistSq(key, x, z) {
    const comma = key.indexOf(',');
    const dx = (+key.slice(0, comma) + 0.5) * this._chunk - x;
    const dz = (+key.slice(comma + 1) + 0.5) * this._chunk - z;
    return dx * dx + dz * dz;
  }

  /**
   * Fill queued chunks under a wall-clock budget so streaming can never stall a
   * frame. The budget is generous on the first build (nothing is on screen yet)
   * and tight afterwards.
   */
  _drainPending(budgetMs) {
    if (!this._pending.length && !this._building) return;
    const start = performance.now();

    while (performance.now() - start < budgetMs) {
      if (!this._building) {
        if (!this._pending.length) break;
        const key = this._pending.shift();
        const comma = key.indexOf(',');
        const record = this._acquireChunk();
        record.progress = 0;
        this._building = { key, record, cx: +key.slice(0, comma), cz: +key.slice(comma + 1) };
      }
      const b = this._building;
      const done = this._fillChunkSlice(b.record, b.cx, b.cz, FILL_SLICE);
      if (done) {
        this.group.add(b.record.mesh);
        this.chunks.set(b.key, b.record);
        this._building = null;
      }
    }
    this.stats.chunks = this.chunks.size + this._pending.length;
  }

  update(dt, state) {
    const cam = this.camera.position;
    const u = this.material.uniforms;

    // --- altitude clipmap ---------------------------------------------------
    // Height above the ground under the camera, not absolute Y: standing on a
    // rise must not switch bands.
    const groundY = this.terrain ? this.terrain.getHeight(cam.x, cam.z) : 0;
    const altitude = cam.y - groundY;

    let band = this._band;
    // Climb while the next band's enter threshold is cleared.
    while (band + 1 < ALTITUDE_BANDS.length && altitude >= ALTITUDE_BANDS[band + 1].enter) band++;
    // Descend while below the current band's exit threshold (hysteresis).
    while (band > 0 && altitude < ALTITUDE_BANDS[band].exit) band--;

    if (band !== this._band) {
      this._band = band;
      this._chunk = CHUNK_BASE * ALTITUDE_BANDS[band].scale * this._mode.chunkScale;
      // Chunk contents are keyed to cell coordinates at a given size, so a size
      // change invalidates every one of them. Retire the lot; the grid rebuild
      // below refills from the pool.
      for (const record of this.chunks.values()) {
        this.group.remove(record.mesh);
        this.pool.push(record);
      }
      this.chunks.clear();
      this._lastCellX = Infinity;
      this._lastCellZ = Infinity;
      this._firstBuild = true;
      // Reach scales with the grid, so the dissolve distance has to follow it or
      // blades would vanish long before the field's edge.
      this._applyFade();
    }

    // --- streaming ----------------------------------------------------------
    const cellX = Math.floor(cam.x / this._chunk);
    const cellZ = Math.floor(cam.z / this._chunk);
    if (cellX !== this._lastCellX || cellZ !== this._lastCellZ) {
      this._lastCellX = cellX;
      this._lastCellZ = cellZ;
      this._rebuildGrid(cellX, cellZ);
    }

    // 12 ms on the very first build - the loading screen is still up and an
    // empty field would be worse than a slow one. 1.6 ms afterwards, which is
    // roughly one chunk per frame: the field fills in over a few frames as you
    // walk, and no single frame ever pays for a whole row.
    this._drainPending(this._firstBuild ? 12 : 1.6);
    if (this._firstBuild && !this._pending.length) this._firstBuild = false;

    // --- per-chunk LOD ------------------------------------------------------
    // Screen-space density is roughly constant if instance count falls with the
    // square of distance. The 1/(1+(d/k)^2) form keeps the nearest chunks full
    // without a discontinuity at d = 0.
    if (this.ctx.input?.wasPressed('KeyG')) this.cycleMode();

    const density =
      this._preset.density * (state.quality.grassDensity ?? 1) * this._mode.densityMul;
    const fadeEnd = u.uFade.value.y;
    let total = 0;
    let visible = 0;

    for (const record of this.chunks.values()) {
      _v3.copy(record.centre);
      const d = Math.max(0, _v3.distanceTo(cam) - this._chunk * 0.71);
      if (d > fadeEnd) {
        record.mesh.visible = false;
        continue;
      }
      record.mesh.visible = true;
      // The falloff is a near-field metric, so it has to scale with the grid or
      // an aerial band would starve every chunk to its floor count.
      const k = this._preset.falloff * ALTITUDE_BANDS[this._band].scale;
      const falloff = 1 / (1 + (d / k) * (d / k));
      const count = Math.max(
        32,
        Math.min(MAX_PER_CHUNK, Math.round(MAX_PER_CHUNK * falloff * density))
      );
      record.geometry.instanceCount = count;
      total += count;
      visible++;
    }
    this.stats.visible = visible;
    this.stats.instances = total;

    // --- wind ---------------------------------------------------------------
    const wind = state.wind;
    u.uTime.value = state.time.elapsed;
    u.uWindDir.value.copy(wind.direction);
    // getGustAt gives the travelling fronts the tree reads too, so field and
    // canopy move as one system.
    const gust = this.wind?.getGustAt
      ? this.wind.getGustAt(cam.x, cam.z, state.time.elapsed)
      : wind.gust;
    u.uWindStrength.value = wind.strength * (gust || 1) * 0.42;
    u.uTurbulence.value = wind.turbulence;

    // --- lighting -----------------------------------------------------------
    const sun = state.sun;
    const moon = state.moon;
    const dayMix = sun.visibility;
    u.uSunDir.value.copy(dayMix > 0.02 ? sun.direction : moon.direction);
    u.uSunColor.value.copy(dayMix > 0.02 ? sun.color : moon.color);
    u.uSunIntensity.value = dayMix > 0.02 ? sun.intensity : moon.intensity;
    u.uSkyColor.value.copy(state.sky.horizonColor);
    u.uGroundColor.value.copy(state.sky.groundColor);
    u.uAmbient.value = state.sky.ambientIntensity * 0.55;
    u.uWetness.value = state.weather.wetness;
    u.uFogColor.value.copy(state.weather.fogColor);
    u.uFogDensity.value = state.weather.fogDensity;

    // --- tree trample -------------------------------------------------------
    if (this.tree?.position) {
      u.uTree.value.set(this.tree.position.x, this.tree.position.z);
      u.uTreeR.value = TRAMPLE_RADIUS;
    }
  }

  dispose() {
    for (const record of this.chunks.values()) {
      this.group.remove(record.mesh);
      record.geometry.dispose();
    }
    for (const record of this.pool) record.geometry.dispose();
    this.chunks.clear();
    this.pool.length = 0;
    this.blade.dispose();
    this.material.dispose();
    this.scene.remove(this.group);
  }
}
