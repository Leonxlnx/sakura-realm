/**
 * birds.js - high birds: wheeling flocks and lone soarers, 60-250 m up.
 *
 * Everything here is about MOTION and SILHOUETTE. At the distances these are
 * seen a bird is three to eight pixels across; nobody will ever resolve a
 * feather. What the eye actually reads, in order:
 *
 *   1. the shape of the path - flocks must wheel, reform and drift, never
 *      run a fixed circle,
 *   2. the wingbeat - a fast powered downstroke and a slow recovery, with the
 *      hand trailing the arm. A sine wave reads as a machine. And crucially it
 *      is INTERMITTENT: a burst of three or four strokes, then a coast. At the
 *      range these are seen the beat itself is a couple of pixels of blur, so
 *      what the eye really tracks is that blur switching on and off,
 *   3. the bank - a bird that changes direction without rolling into it looks
 *      instantly, unmistakably wrong,
 *   4. that they are DARK against a bright sky, and that they carry the same
 *      aerial perspective as everything else at that range.
 *
 * Design notes worth reading before editing:
 *
 * - Positions are stored CAMERA-RELATIVE in x/z and absolute in y, and the mesh
 *   is translated to (camX, 0, camZ) each frame. The whole flock volume
 *   therefore travels with the player - the same trick weather/precipitation.js
 *   uses - so there is never a world-sized simulation and floats stay small.
 *   A player who outruns a flock pushes it past the volume edge, where it fades
 *   out, is rigidly recycled to the far side and fades back in. Rigid, because
 *   wrapping birds individually would shred the flock.
 *
 * - VERTICALLY the same trick, one-way and only for the eye. The altitude band
 *   is pinned above whichever is higher of the ground and the eye, which is what
 *   guarantees a bird is always seen against sky and never below the horizon.
 *   Fly mode climbs at well over 100 m/s, far faster than a damped chase can
 *   follow, so the band follows a rising eye INSTANTLY and every bird is
 *   translated by the same delta - again rigid, again unseeable. A falling eye is
 *   followed slowly and the birds are left where they are, so a descent does not
 *   drag them down. A rise driven by the GROUND instead of the eye is deliberately
 *   NOT carried: it moves at most 15 m/s, the per-bird altitude spring handles it,
 *   and carrying it would ratchet the flock up a hill at a time.
 *
 * - Flocking is real boids (separation / alignment / cohesion) inside a flock
 *   only, plus an ORBIT around a slowly drifting roost point: a radial spring
 *   toward a target radius and a tangential term whose sign and strength come
 *   from a slow noise. That is a limit cycle, not a circle - the flock wheels,
 *   stalls, mills and reverses on its own. The roost drifts on 2D simplex noise
 *   and never repeats.
 *
 * - Everything is flat typed arrays and there is not a single allocation in
 *   update(). The per-frame cost is dominated by the O(n²) neighbour loop,
 *   which is bounded at 26 birds per flock - about 2.7 k pair tests a frame.
 *
 * - The material owns its fog. It deliberately does NOT set `fog: true`: three
 *   r180's refreshFogUniforms() reads fogColor/fogDensity/fogNear/fogFar off any
 *   material with that flag and throws inside renderer.render() if they are
 *   missing, which silently drops the rest of the frame's render list. Instead
 *   the vertex shader integrates the same height-stratified optical depth
 *   weather/atmosfx.js uses, from parameters copied off that system when it is
 *   present, so birds and terrain agree at range without any coupling.
 */

import {
  BufferAttribute,
  Color,
  DoubleSide,
  DynamicDrawUsage,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  Mesh,
  ShaderMaterial,
  Sphere,
  Vector2,
  Vector3,
  Vector4,
} from 'three';

import { EVENTS } from '../core/state.js';
import {
  TAU,
  clamp,
  clamp01,
  createNoise,
  damp,
  lerp,
  makeRNG,
  smoothstep,
  worley2D,
} from '../core/math.js';

// ---------------------------------------------------------------------------
// Population
//
// Slots are allocated once at the ULTRA size (108 birds, ~9 KB of typed arrays)
// so a tier change never touches a GPU buffer. Tiers only cap how many groups
// and members may become active; everything else fades through the same path
// the weather uses, so quality switches are as invisible as a passing shower.
// ---------------------------------------------------------------------------

const MAX_FLOCKS = 4;
const PER_FLOCK = 26;
const MAX_LONERS = 4;
const MAX_GROUPS = MAX_LONERS + MAX_FLOCKS;
const MAX_BIRDS = MAX_LONERS + MAX_FLOCKS * PER_FLOCK; // 108

const TIERS = {
  low: { flocks: 1, perFlock: 12, loners: 1 },
  medium: { flocks: 2, perFlock: 18, loners: 2 },
  high: { flocks: 3, perFlock: 22, loners: 3 },
  ultra: { flocks: 4, perFlock: PER_FLOCK, loners: MAX_LONERS },
};

// ---------------------------------------------------------------------------
// Flight-model constants. Units are metres, seconds, radians.
// ---------------------------------------------------------------------------

/** Half-extent of the camera-centred volume the flocks live in. */
const WRAP_R = 430;
/** Total roost excursion, per axis, from the point the group was seeded at. */
const ROOST_SPREAD_FLOCK = 225;
const ROOST_SPREAD_LONE = 320;
/**
 * The roost is split into a fixed WORLD anchor plus a slow noise wander, and the
 * two fractions sum to 1 so the excursion envelope above is unchanged.
 *
 * The anchor is the important half. Deriving the roost purely from a function of
 * time, in camera-relative space - which is what this did first - welds it to the
 * player: the whole orbit centre translates with the camera, the radial spring
 * drags the flock along behind it, and a walker at 5 m/s tows the same flock
 * across the map forever. Measured before the fix: after 317 m of walking a flock
 * that started 169 m ahead had been reeled in to 33 m and stayed there.
 */
const ROOST_ANCHOR = 0.55;
const ROOST_WANDER = 0.45;
/**
 * Radius at which a recycled group's roost is dropped back into the volume.
 * Far enough out that the fade-in is not read as a spawn, near enough that the
 * wander has room to move before the group leaves again.
 */
const RESEED_R = WRAP_R * 0.58;
/**
 * Edge fade band. Wider than the boundary needs, because alpha follows this ramp
 * through a damp: the extra runway is what a player flying into the boundary
 * spends bleeding the group off before it is allowed to recycle.
 */
const EDGE_FADE_IN = WRAP_R * 0.74;
const EDGE_FADE_OUT = WRAP_R * 0.97;
/** Vertical extent of the band above `bandBase`. */
const BAND_HEIGHT = 190;
/** Clearance kept above the ground and above the camera. Guarantees sky behind. */
const BAND_OVER_GROUND = 60;
const BAND_OVER_CAMERA = 38;

/**
 * Preferred personal space, per bird. It has to VARY: with one shared radius the
 * separation and cohesion terms reach equilibrium at exactly that distance and
 * the flock crystallises into a lattice with every nearest-neighbour gap
 * identical to a centimetre. That reads as a formation, not a flock.
 */
const SEP_MIN = 2.4;
const SEP_MAX = 5.2;
const NEIGH_RADIUS2 = 15 * 15;
const SEP_W = 15.0;
const ALIGN_W = 1.15;
const COH_W = 0.30;
const ROOST_RADIAL = 2.7;
const ROOST_TANGENT = 2.3;
const ALT_SPRING = 0.09;
const ALT_DAMP = 0.55;
const WIND_DRAG = 0.13;
const MAX_ACCEL = 9.5;
const MAX_ACCEL2 = MAX_ACCEL * MAX_ACCEL;
/** Coordinated-turn gravity. Real 9.81; nudged up so banks read without cartooning. */
const BANK_G = 8.6;
const MAX_ROLL = 1.22;
/** Wind aloft runs stronger and steadier than the wind in the grass. */
const WIND_ALOFT = 1.55;
/** Thermal cell spacing is 1 / this, i.e. ~560 m. */
const THERMAL_FREQ = 0.0018;
/**
 * Lateral wander. A bird holding a perfectly clean line is the other half of the
 * lattice problem - real ones are constantly making small corrections. The rate
 * is quantised into buckets so the per-frame rotation is eight trig calls for the
 * whole population rather than two per bird.
 */
const WANDER_ACCEL = 1.7;
/**
 * Bout (flap-glide) flight. Small and medium birds do not hold a steady beat:
 * they take three or four powered strokes, then hold the wings out and coast
 * for about half as long again. At 150 m the wingbeat itself is two pixels of
 * blur - what actually reads is that blur switching ON and OFF, plus the shallow
 * bob in the path it causes. It is the single strongest "that is alive" cue at
 * this range, and it costs about ten flops a bird.
 *
 * The big soaring loners do not do it at all (their `boutAmt` is zero), and it
 * is suppressed on anything currently riding lift - a bird in a thermal holds
 * its wings out and does not bound.
 */
const BOUT_HZ_MIN = 0.55;
const BOUT_HZ_MAX = 0.85;
/** Fraction of the cycle under power, plus the two transition ramps. */
const BOUT_POWER_END = 0.62;
const BOUT_RAMP_IN = 0.10;
const BOUT_RAMP_OUT = 0.14;
/** Mean of the duty curve above. Subtracted so the bob carries no net lift. */
const BOUT_MEAN = 0.64;
/**
 * Peak vertical acceleration of the bob. Real bounding flight is ballistic - a
 * full -g on the closed-wing phase - but at these periods that only buys ~0.3 m
 * of amplitude anyway, and half a g keeps it clear of the steering budget.
 */
const BOUT_LIFT = 6.2;
/** Wingbeat amplitude that survives the glide phase. */
const BOUT_FLOOR = 0.12;
const WANDER_BUCKETS = 6;
const WANDER_RATE = new Float32Array(WANDER_BUCKETS);
for (let k = 0; k < WANDER_BUCKETS; k++) WANDER_RATE[k] = 0.35 + (k / (WANDER_BUCKETS - 1)) * 1.15;
const _wanC = new Float32Array(WANDER_BUCKETS);
const _wanS = new Float32Array(WANDER_BUCKETS);

// ---------------------------------------------------------------------------
// Module-scope scratch. Nothing in update() may allocate.
// ---------------------------------------------------------------------------

const _cam = new Vector3();
const _col = new Color();
const _col2 = new Color();
const _size = new Vector2();
const _wind = { x: 0, y: 0, z: 0 };

// ---------------------------------------------------------------------------
// Geometry - one bird, 17 vertices, 9 triangles.
//
// Local frame: +x spanwise (in half-span units), +y up, -z forward. The wing is
// split at a wrist so the vertex shader can articulate it at two hinges; the
// spanwise decomposition (how much of a vertex's offset belongs to the arm and
// how much to the hand) is baked per vertex so the shader never has to branch.
// ---------------------------------------------------------------------------

const WRIST = 0.46;

function buildBirdGeometry() {
  const pos = [];
  const art = []; // x side, y arm span, z hand span (or tail-fan factor), w wing mask
  const idx = [];

  const wingVert = (side, ax, y, z) => {
    pos.push(side * ax, y, z);
    art.push(side, Math.min(ax, WRIST), Math.max(ax - WRIST, 0), 1);
    return pos.length / 3 - 1;
  };
  const bodyVert = (x, y, z, side = 0, tailFan = 0) => {
    pos.push(x, y, z);
    art.push(side, 0, tailFan, 0);
    return pos.length / 3 - 1;
  };
  const tri = (a, b, c) => idx.push(a, b, c);

  // Wings. Inner panel is a quad (root -> wrist), the hand is a single swept
  // triangle. Three triangles a wing is all the silhouette this ever needs.
  for (let s = 0; s < 2; s++) {
    const side = s === 0 ? 1 : -1;
    const r0 = wingVert(side, 0.10, 0, -0.10); // root, leading edge
    const r1 = wingVert(side, 0.10, 0, 0.22); // root, trailing edge
    const w0 = wingVert(side, WRIST, 0, -0.13); // wrist, leading
    const w1 = wingVert(side, WRIST, 0, 0.15); // wrist, trailing
    const tp = wingVert(side, 1.0, 0, 0.14); // tip, swept back
    tri(r0, w0, w1);
    tri(r0, w1, r1);
    tri(w0, tp, w1);
  }

  // Body. A shallow keel rather than a flat plate: seen exactly edge-on a flat
  // body vanishes, and the two-centimetre V costs nothing and never does.
  const b0 = bodyVert(0, 0.0, -0.62); // bill
  const b1 = bodyVert(-0.075, -0.022, -0.05);
  const b2 = bodyVert(0.075, -0.022, -0.05);
  const b3 = bodyVert(0, 0.0, 0.36);
  tri(b0, b1, b2);
  tri(b1, b3, b2);

  // Tail. The two tips carry a fan factor; the shader spreads them with the
  // bank angle, because a turning bird fans and twists its tail.
  const t0 = bodyVert(0, 0.0, 0.30);
  const t1 = bodyVert(-0.11, 0.0, 0.62, -1, 1);
  const t2 = bodyVert(0.11, 0.0, 0.62, 1, 1);
  tri(t0, t1, t2);

  const geo = new InstancedBufferGeometry();
  geo.setAttribute('position', new BufferAttribute(new Float32Array(pos), 3));
  geo.setAttribute('aArt', new BufferAttribute(new Float32Array(art), 4));
  geo.setIndex(new BufferAttribute(new Uint16Array(idx), 1));
  return geo;
}

// ---------------------------------------------------------------------------
// GLSL
// ---------------------------------------------------------------------------

const BIRD_VERT = /* glsl */ `
  #define BIRD_TAU 6.283185307

  uniform float uPixelScale;
  uniform vec2  uMinPx;      // x minimum span in pixels, y minimum chord in pixels
  uniform vec2  uWingAmp;    // x downstroke amplitude, y upstroke amplitude (rad)
  uniform vec2  uWrist;      // x wrist lag gain, y chordwise sweep gain
  uniform float uDuty;       // fraction of the cycle spent on the downstroke

  uniform vec3  uKeyDir;
  uniform vec3  uKeyColor;
  uniform vec3  uSkyAmbient;
  uniform vec3  uSkyBounce;
  uniform vec3  uPlumage;
  uniform vec3  uBelly;
  uniform vec3  uFlashColor;
  uniform float uFlash;
  uniform float uGlobalAlpha;

  uniform vec3  uBirdZenith;
  uniform vec3  uBirdHorizonSun;
  uniform vec3  uBirdHorizonAway;
  uniform vec3  uBirdGround;
  uniform vec3  uBirdMistColor;
  uniform vec4  uBirdMist;   // x density  y falloff  z base height
  uniform vec4  uBirdHaze;   // x density  y falloff  z base height
  uniform float uBirdMaxOpacity;

  attribute vec4 aArt;       // x side, y arm span, z hand span / tail fan, w wing mask
  attribute vec3 aPos;       // instance origin, mesh space
  attribute vec3 aFwd;       // unit heading
  attribute vec4 aMotion;    // x roll, y wing phase (cycles), z wingspan (m), w alpha
  attribute vec2 aBeat;      // x flap amplitude 0..1, y soaring dihedral (rad)

  varying vec3 vColor;
  varying float vAlpha;

  /**
   * Asymmetric wingbeat. The powered downstroke takes uDuty of the cycle and the
   * recovery takes the rest; the piecewise-linear phase remap is C1 continuous
   * because cos() has exactly zero slope at u = 0.5, which is where the seam
   * lands. Returns +1 at the top of the upstroke, -1 at the bottom of the
   * downstroke.
   */
  float beatCurve(float t) {
    float u = t < uDuty
      ? (t / max(uDuty, 1e-3)) * 0.5
      : 0.5 + (t - uDuty) / max(1.0 - uDuty, 1e-3) * 0.5;
    return cos(u * BIRD_TAU);
  }

  /**
   * Optical depth through a medium of density d(h) = density * exp(-(h-base)*k).
   * Substituting dh = dirY * ds collapses the integral to a difference of the
   * endpoint densities. Same derivation and same bounded form weather/atmosfx.js
   * uses, so a bird and the treeline behind it never disagree about the haze.
   */
  float birdOpticalDepth(float density, float falloff, float baseY, float camY, float dirY, float dist) {
    float hStart = max(camY - baseY, 0.0);
    float eStart = exp(-falloff * hStart);
    float x = falloff * dirY * dist;
    if (abs(x) < 1e-3) return clamp(density * eStart * dist * (1.0 - 0.5 * x), 0.0, 24.0);
    float hEnd = max(camY + dirY * dist - baseY, 0.0);
    return clamp(density * (eStart - exp(-falloff * hEnd)) * (dist / x), 0.0, 24.0);
  }

  /** Sky radiance along a direction, from the same three probes atmosfx publishes. */
  vec3 birdSky(vec3 dir, float cosSun) {
    float az = clamp(cosSun * 0.5 + 0.5, 0.0, 1.0);
    vec3 horizon = mix(uBirdHorizonAway, uBirdHorizonSun, az * az);
    vec3 c = mix(horizon, uBirdZenith, pow(clamp(dir.y, 0.0, 1.0), 0.62));
    return mix(c, uBirdGround, smoothstep(0.0, 0.4, -dir.y) * 0.8);
  }

  void main() {
    float span = aMotion.z;
    float halfSpan = span * 0.5;

    // Everything below is evaluated at the INSTANCE origin, not at the vertex,
    // so distance-dependent terms are constant across a bird and cannot shear
    // when the wing swings.
    vec3 instWorld = (modelMatrix * vec4(aPos, 1.0)).xyz;
    vec3 toCam = cameraPosition - instWorld;
    float dist = length(toCam);
    vec3 V = toCam / max(dist, 1e-4);   // bird -> camera
    vec3 dir = -V;                      // camera -> bird

    // ---- wing kinematics -------------------------------------------------
    float amp = aBeat.x;
    float bArm = beatCurve(fract(aMotion.y));
    // The hand trails the arm by an eighth of a cycle. That lag alone is most of
    // what separates a bird from a flapping paper triangle.
    float bHand = beatCurve(fract(aMotion.y - 0.115));

    float a1 = (bArm > 0.0 ? bArm * uWingAmp.y : bArm * uWingAmp.x) * amp + aBeat.y;
    // Pure lag differential as the wrist angle: on the downstroke the arm drops
    // faster than the hand, so the tip flicks up and back; on the recovery it
    // reverses. The soaring droop is proportional to the dihedral, which gives a
    // gliding bird the arm-up / hand-down gull kink for free.
    float a2 = (bHand - bArm) * uWrist.x * amp - aBeat.y * 1.25;
    // A real wing gets visibly shorter going up - it is partly folded.
    float fold = 1.0 - 0.30 * amp * max(bArm, 0.0);
    float sweep = uWrist.y * amp * bArm;

    float ci = cos(a1), si = sin(a1);
    float co = cos(a1 + a2), so = sin(a1 + a2);
    float armS = aArt.y;
    float handS = aArt.z * fold;
    float radial = ci * armS + co * handS;
    float rise = si * armS + so * handS;

    // Body and tail take the mix()'s other branch; the tail tips additionally
    // fan out with the bank angle and with how much the bird is gliding.
    float tailK = (1.0 - aArt.w) * aArt.z;
    float tailFan = 0.10 + 0.34 * abs(aMotion.x) + 0.16 * (1.0 - amp);

    vec3 loc;
    loc.x = mix(position.x + aArt.x * tailK * tailFan, aArt.x * radial, aArt.w);
    loc.y = mix(position.y, rise, aArt.w);
    loc.z = position.z + sweep * handS * aArt.w;

    // ---- sub-pixel widening ----------------------------------------------
    // A wing chord is well under a pixel past ~250 m, and a sub-pixel triangle
    // randomly hits or misses pixel centres from frame to frame: that is the
    // scintillation that makes distant particles scream "effect". Widen the two
    // thin dimensions independently to a minimum pixel footprint and scale alpha
    // by the inverse, which conserves the ink exactly.
    float pxW = dist * uPixelScale;
    float spanScale = clamp((uMinPx.x * pxW) / max(span, 1e-4), 1.0, 5.0);
    float chordScale = clamp((uMinPx.y * pxW) / max(span * 0.17, 1e-4), 1.0, 7.0);
    float energy = 1.0 / (spanScale * chordScale);

    loc.x *= spanScale;
    loc.y *= spanScale;
    loc.z *= chordScale;

    vec3 body = loc * halfSpan;

    // ---- orientation ------------------------------------------------------
    vec3 f = normalize(aFwd);
    vec3 r0 = cross(f, vec3(0.0, 1.0, 0.0));
    float rl = length(r0);
    r0 = rl > 1e-4 ? r0 / rl : vec3(1.0, 0.0, 0.0);
    vec3 u0 = cross(r0, f);
    float cr = cos(aMotion.x), sr = sin(aMotion.x);
    vec3 R = r0 * cr + u0 * sr;   // positive roll lifts the right wing (left bank)
    vec3 U = u0 * cr - r0 * sr;

    vec3 world = instWorld + R * body.x + U * body.y - f * body.z;

    // ---- shading ----------------------------------------------------------
    // Wing normal straight from the dihedral: the surface runs along
    // (side*cos a, sin a, 0), so its normal is (-side*sin a, cos a, 0).
    float na = (a1 + a2 * 0.5) * aArt.w;
    vec3 nLocal = vec3(-aArt.x * sin(na), cos(na), 0.0);
    vec3 N = R * nLocal.x + U * nLocal.y;
    N = dot(N, V) < 0.0 ? -N : N;   // thin sheet, always shade the visible face

    // Wrapped diffuse. A hard Lambert terminator sweeping across a four-pixel
    // wing once per flap strobes; the wrap keeps the transition soft, and the
    // square keeps the shadow side genuinely dark.
    float ndl = dot(N, uKeyDir);
    float diff = max(ndl * 0.5 + 0.5, 0.0);
    diff *= diff;

    // We are almost always looking UP at these, at the dark underwing. The
    // up-facing fraction of the visible normal picks between the two plumages
    // and between sky light and the much darker ground bounce.
    float upFace = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 plumage = mix(uBelly, uPlumage, upFace);
    vec3 ambient = mix(uSkyBounce, uSkyAmbient, upFace);

    // Sun through the primaries. Only when the light is behind the wing AND we
    // are looking toward it - the moment a backlit flock catches fire at dusk.
    float back = max(-ndl, 0.0);
    float toward = max(dot(dir, uKeyDir), 0.0);
    float trans = back * toward * toward * toward * aArt.w;

    vec3 col = plumage * (ambient + uKeyColor * diff * 0.5)
             + uKeyColor * plumage * trans * 2.4
             + uFlashColor * uFlash * 0.25;

    // ---- aerial perspective ------------------------------------------------
    float odMist = birdOpticalDepth(uBirdMist.x, uBirdMist.y, uBirdMist.z, cameraPosition.y, dir.y, dist);
    float odHaze = birdOpticalDepth(uBirdHaze.x, uBirdHaze.y, uBirdHaze.z, cameraPosition.y, dir.y, dist);
    float od = odMist + odHaze;
    float fogFactor = (1.0 - exp(-od)) * uBirdMaxOpacity;
    vec3 inscatter = mix(birdSky(dir, dot(dir, uKeyDir)), uBirdMistColor, odMist / max(od, 1e-4));
    col = mix(col, inscatter, fogFactor);

    // The colour mix alone leaves a faint smudge where our reconstructed sky and
    // the real sky disagree, so heavy extinction also eats the alpha. Quadratic,
    // so it only bites once the bird is genuinely lost in the murk.
    float a = aMotion.w * energy * uGlobalAlpha * (1.0 - fogFactor * fogFactor * 0.92);

    if (a < 0.0035) {
      vColor = vec3(0.0);
      vAlpha = 0.0;
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);   // outside clip space: the bird dies
      return;
    }

    vColor = col;
    vAlpha = a;
    gl_Position = projectionMatrix * viewMatrix * vec4(world, 1.0);
  }
`;

const BIRD_FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;

  void main() {
    if (vAlpha < 0.0035) discard;
    gl_FragColor = vec4(vColor, min(vAlpha, 1.0));
  }
`;

// ---------------------------------------------------------------------------

export class Birds {
  constructor(ctx) {
    this.ctx = ctx;
    this.scene = ctx.scene;
    this.camera = ctx.camera;
    this.renderer = ctx.renderer;
    this.state = ctx.state;
    this.bus = ctx.bus;

    this.tier = TIERS[ctx.state.quality?.tier] || TIERS.high;

    this.terrain = null;
    this.tree = null;
    this.atmosfx = null;

    this.mesh = null;
    this.geometry = null;
    this.material = null;

    // --- per-bird state (flat, never reallocated) ---------------------------
    const n = MAX_BIRDS;
    this.px = new Float32Array(n);
    this.py = new Float32Array(n);
    this.pz = new Float32Array(n);
    this.vx = new Float32Array(n);
    this.vy = new Float32Array(n);
    this.vz = new Float32Array(n);
    this.roll = new Float32Array(n);
    this.phase = new Float32Array(n);
    this.flapHz = new Float32Array(n);
    this.span = new Float32Array(n);
    this.cruise = new Float32Array(n);
    this.alpha = new Float32Array(n);
    this.soar = new Float32Array(n);
    this.soarBias = new Float32Array(n);
    this.thermal = new Float32Array(n);
    this.yOffset = new Float32Array(n);
    this.stx = new Float32Array(n);
    this.sty = new Float32Array(n);
    this.stz = new Float32Array(n);
    this.sepR2 = new Float32Array(n);
    this.wanX = new Float32Array(n);
    this.wanZ = new Float32Array(n);
    this.wanB = new Uint8Array(n);
    this.boutHz = new Float32Array(n);
    this.boutPh = new Float32Array(n);
    this.boutAmt = new Float32Array(n);

    // --- per-group state ----------------------------------------------------
    const g = MAX_GROUPS;
    this.gStart = new Int32Array(g);
    this.gSize = new Int32Array(g);
    this.gWeight = new Float32Array(g);
    this.gRoostX = new Float32Array(g);
    this.gRoostY = new Float32Array(g);
    this.gRoostZ = new Float32Array(g);
    // World anchor of the roost, held camera-relative and rebased every frame.
    this.gAnchX = new Float32Array(g);
    this.gAnchZ = new Float32Array(g);
    this.gSpin = new Float32Array(g);
    this.gOrbit = new Float32Array(g);
    this.gPull = new Float32Array(g);
    this.gCx = new Float32Array(g);
    this.gCy = new Float32Array(g);
    this.gCz = new Float32Array(g);
    this.gWindX = new Float32Array(g);
    this.gWindY = new Float32Array(g);
    this.gWindZ = new Float32Array(g);
    this.gSeed = new Float32Array(g);
    this.gFadeIn = new Float32Array(g);
    this.gAlphaMax = new Float32Array(g);
    this.gDormant = new Uint8Array(g);

    // --- instance buffers ---------------------------------------------------
    this.iPos = new Float32Array(n * 3);
    this.iFwd = new Float32Array(n * 3);
    this.iMotion = new Float32Array(n * 4);
    this.iBeat = new Float32Array(n * 2);

    this.noise = createNoise(0x5b17d3);
    this.rng = makeRNG(0x5b17d3);

    this._pop = 0.5;
    this._bandBase = 90;
    this._groundBase = 90;
    this._time = 0;
    this._startle = 0;
    this._baseSphereR = 1;
    this._maxSpreadR = 0;
    this._seeded = false;
    this._camValid = false;
    this._prevCamX = 0;
    this._prevCamZ = 0;
    this._liveMax = 0;
    this._frame = 0;
    this._disposed = false;

    this._onLightning = () => {
      // A strike scatters everything in the air. Additive so a burst stacks.
      this._startle = Math.min(1.4, this._startle + 1.0);
    };
    this.bus?.on?.(EVENTS.LIGHTNING_STRIKE, this._onLightning);

    this._layoutGroups();
    this._seedBirdTraits();
  }

  // -------------------------------------------------------------------------
  // Build
  // -------------------------------------------------------------------------

  /** Loners occupy the first slots so they are the last life to leave the sky. */
  _layoutGroups() {
    for (let g = 0; g < MAX_LONERS; g++) {
      this.gStart[g] = g;
      this.gSize[g] = 1;
    }
    for (let f = 0; f < MAX_FLOCKS; f++) {
      const g = MAX_LONERS + f;
      this.gStart[g] = MAX_LONERS + f * PER_FLOCK;
      this.gSize[g] = PER_FLOCK;
    }
    for (let g = 0; g < MAX_GROUPS; g++) {
      this.gSeed[g] = 3.7 + g * 11.31;
      this.gOrbit[g] = g < MAX_LONERS ? 140 : 70;
      this.gSpin[g] = g % 2 === 0 ? 1 : -1;
      this.gPull[g] = 1;
      this.gDormant[g] = 1;
    }
  }

  _seedBirdTraits() {
    const r = this.rng;
    for (let i = 0; i < MAX_BIRDS; i++) {
      const lone = i < MAX_LONERS;
      // Loners are the big soaring birds - a longer, slower, steadier wing.
      const span = lone ? r.range(1.45, 2.05) : r.range(0.72, 1.15);
      this.span[i] = span;
      // Wingbeat frequency scales roughly as 1/span across birds of a kind.
      this.flapHz[i] = (lone ? 1.55 : 3.5) / span + r.range(-0.18, 0.18);
      this.cruise[i] = lone ? r.range(10.5, 14.5) : r.range(11.0, 16.5);
      this.phase[i] = r() * 64;
      this.soarBias[i] = lone ? r.range(0.25, 0.60) : r.range(-0.15, 0.30);
      this.yOffset[i] = r.gaussian(0, lone ? 22 : 11);
      // Personal space scales with the bird, plus a wide individual spread.
      const sep = clamp(r.range(SEP_MIN, SEP_MAX) * (0.7 + 0.4 * span), SEP_MIN, SEP_MAX + 1);
      this.sepR2[i] = sep * sep;
      // Unit wander vector, rotated a fixed angle per frame rather than
      // re-evaluated with sin/cos - see WANDER_RATE.
      const wa = r() * TAU;
      this.wanX[i] = Math.cos(wa);
      this.wanZ[i] = Math.sin(wa);
      this.wanB[i] = Math.min(WANDER_BUCKETS - 1, Math.floor(r() * WANDER_BUCKETS));
      // Bout flight. The powered burst has to contain three or four whole beats
      // to read as a burst rather than a stutter, so the bout rate is DERIVED
      // from this bird's own wingbeat instead of drawn independently - otherwise
      // the fast-winged birds get eight beats a bout and the slow ones get two,
      // and neither reads. Light birds bound hardest; the loners not at all.
      this.boutHz[i] = clamp(
        (this.flapHz[i] * BOUT_POWER_END) / r.range(3.0, 4.4),
        BOUT_HZ_MIN,
        BOUT_HZ_MAX
      );
      this.boutPh[i] = r();
      this.boutAmt[i] = lone ? 0 : clamp01((1.22 - span) / 0.45) * r.range(0.55, 1.0);
      // Fixed startle direction per bird, so a strike explodes the flock outward
      // instead of shoving it as a block.
      const a = r() * TAU;
      const b = r.range(-0.5, 0.5);
      this.stx[i] = Math.cos(a);
      this.sty[i] = b;
      this.stz[i] = Math.sin(a);
      this.iMotion[i * 4 + 2] = span;
      // A slot that is never simulated still gets rasterised while it sits under
      // the live-prefix instance count. Its alpha is zero so it dies in the
      // vertex shader, but normalize() of a zero heading would produce NaNs on
      // the way there - give every slot a valid one from the start.
      this.iFwd[i * 3 + 2] = -1;
    }
  }

  async init() {
    this.geometry = buildBirdGeometry();

    const mk = (arr, size) => {
      const a = new InstancedBufferAttribute(arr, size);
      a.setUsage(DynamicDrawUsage);
      return a;
    };
    this.aPos = mk(this.iPos, 3);
    this.aFwd = mk(this.iFwd, 3);
    this.aMotion = mk(this.iMotion, 4);
    this.aBeat = mk(this.iBeat, 2);
    this.geometry.setAttribute('aPos', this.aPos);
    this.geometry.setAttribute('aFwd', this.aFwd);
    this.geometry.setAttribute('aMotion', this.aMotion);
    this.geometry.setAttribute('aBeat', this.aBeat);
    this.geometry.instanceCount = 0;

    // Real bounding sphere covering the whole instance spread. The unit sphere
    // three would otherwise compute from `position` alone culls the entire mesh
    // the moment the camera looks anywhere but straight at the origin. The
    // centre tracks the altitude band each frame and the radius is grown from
    // the birds actually published - a nominal radius derived from WRAP_R is NOT
    // enough, because WRAP_R bounds a group's CENTROID and stragglers hang tens
    // of metres outside it. Measured with the camera flying at 40 m/s, birds
    // reached 1.05x the old fixed radius, i.e. they left their own bounds.
    const halfXZ = WRAP_R * 1.06;
    const halfY = BAND_HEIGHT * 0.5 + 45;
    this._baseSphereR = Math.sqrt(halfXZ * halfXZ * 2 + halfY * halfY) + 4;
    this.geometry.boundingSphere = new Sphere(
      new Vector3(0, this._bandBase + BAND_HEIGHT * 0.5, 0),
      this._baseSphereR
    );

    this.uniforms = {
      uPixelScale: { value: 0.0008 },
      uMinPx: { value: new Vector2(2.6, 1.35) },
      // 57 degrees down, 30 up: real flapping flight is strongly asymmetric.
      uWingAmp: { value: new Vector2(1.0, 0.52) },
      uWrist: { value: new Vector2(1.35, 0.22) },
      uDuty: { value: 0.38 },

      uKeyDir: { value: new Vector3(0, 1, 0) },
      uKeyColor: { value: new Color(0, 0, 0) },
      uSkyAmbient: { value: new Color(0.08, 0.10, 0.13) },
      uSkyBounce: { value: new Color(0.02, 0.02, 0.02) },
      // Deliberately near-black and desaturated. These read as silhouettes; any
      // saturation at all makes them look like toys stuck on the sky.
      uPlumage: { value: new Color(0.052, 0.048, 0.044) },
      uBelly: { value: new Color(0.085, 0.083, 0.080) },
      uFlashColor: { value: new Color(0.78, 0.85, 1.05) },
      uFlash: { value: 0 },
      uGlobalAlpha: { value: 1 },

      uBirdZenith: { value: new Color(0.18, 0.36, 0.72) },
      uBirdHorizonSun: { value: new Color(0.72, 0.80, 0.92) },
      uBirdHorizonAway: { value: new Color(0.60, 0.72, 0.90) },
      uBirdGround: { value: new Color(0.20, 0.22, 0.19) },
      uBirdMistColor: { value: new Color(0.72, 0.78, 0.86) },
      uBirdMist: { value: new Vector4(0.00015, 0.05, 0, 0) },
      uBirdHaze: { value: new Vector4(0.00006, 0.0008, 0, 0) },
      uBirdMaxOpacity: { value: 0.965 },
    };

    this.material = new ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: BIRD_VERT,
      fragmentShader: BIRD_FRAG,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      side: DoubleSide,
      // NEVER true without owning fogColor/fogDensity/fogNear/fogFar - see the
      // header. This material integrates its own aerial perspective instead.
      fog: false,
    });
    // Keep weather/atmosfx.js's auto-adopt sweep away: this shader already does
    // the aerial term by hand and must not be patched a second time.
    this.material.userData.aerial = false;

    this.mesh = new Mesh(this.geometry, this.material);
    this.mesh.name = 'Birds';
    this.mesh.frustumCulled = true;
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;
    // renderOrder only ever sorts WITHIN a queue. three renders the whole opaque
    // list, then transmissive, then the whole transparent list - three.module.js
    // r180 line 16675 - so ordering against the sky dome and the cloud composite
    // is already settled before renderOrder is consulted at all: both of those
    // are opaque (sky/atmosphere.js leaves `transparent` unset, sky/clouds.js
    // sets `transparent: false` deliberately to stay in the opaque list) and this
    // material is transparent, so birds composite over them for free.
    //
    // What renderOrder actually decides is our slot among the other TRANSPARENTS,
    // and there a large number is simply wrong. The near-field transparents are
    // far petals (10), splashes (10), near petals and splash drops (11), rain and
    // snow (12) and the lightning bolt (13) - every one of them metres from the
    // eye, every one of them in front of a bird that is 60-250 m up. Sitting
    // above all of those punched bird silhouettes through the rain, which is a
    // case that really happens: light rain only thins the flock to about 65%.
    // Below them, and above the celestial sphere (milky way -6, stars -5, meteors
    // -4, moon -3, sun -2), which a bird correctly eclipses.
    this.mesh.renderOrder = 1;
    this.mesh.userData.noAerial = true;
    this.mesh.userData.aerial = false;
    this.mesh.visible = false;
    this.scene.add(this.mesh);
  }

  link(systems) {
    this.terrain = systems.terrain || null;
    this.tree = systems.tree || null;
    this.atmosfx = systems.atmosfx || null;
  }

  onQualityChange(quality) {
    this.tier = TIERS[quality?.tier] || TIERS.high;
  }

  // -------------------------------------------------------------------------
  // Frame
  // -------------------------------------------------------------------------

  update(dt, state) {
    if (!this.mesh || this._disposed) return;
    // A tab-switch hitch must not let a bird integrate a 200 ms step and leave
    // the volume; zero is the correct degenerate value.
    dt = dt > 0 ? Math.min(dt, 0.1) : 0;
    this._frame++;
    // Cached because _seedGroup needs the same clock the per-frame roost walk
    // uses, and it is reachable from paths that do not carry `state`.
    this._time = Number.isFinite(state.time.elapsed) ? state.time.elapsed : this._time;

    this.camera.getWorldPosition(_cam);
    const camX = _cam.x;
    const camZ = _cam.z;
    // Every position, every roost anchor and the altitude band are all held
    // RELATIVE to this point, so a single non-finite camera would subtract NaN
    // out of all 108 birds and all 8 anchors at once - and nothing downstream
    // can recover from that, because the anchors are only ever updated by
    // `-= dx`. Skipping the frame outright costs one frame of bird motion and is
    // the only thing that keeps a transient upstream fault transient.
    if (!Number.isFinite(camX) || !Number.isFinite(camZ) || !Number.isFinite(_cam.y)) return;

    // Everything horizontal is stored camera-relative, so a camera move is a
    // single subtraction per bird and float precision never degrades.
    let dx = 0;
    let dz = 0;
    if (this._camValid) {
      dx = camX - this._prevCamX;
      dz = camZ - this._prevCamZ;
    } else {
      this._camValid = true;
    }
    this._prevCamX = camX;
    this._prevCamZ = camZ;
    this.mesh.position.set(camX, 0, camZ);

    // Rebase every bird into this frame's camera-relative frame. Must happen in
    // its own pass BEFORE any steering runs: the neighbour loop reads other
    // birds' positions, and mixing shifted and unshifted ones inside one pass
    // would tear the flock apart at exactly the speed the player is flying.
    if (dx !== 0 || dz !== 0) {
      const px = this.px;
      const pz = this.pz;
      for (let i = 0; i < MAX_BIRDS; i++) {
        px[i] -= dx;
        pz[i] -= dz;
      }
    }

    // Altitude band. Anchored to whichever is higher, the ground or the camera,
    // so birds are ALWAYS above the eye - which is what guarantees they are seen
    // against sky and never against the field or below the horizon.
    let groundY = 0;
    if (this.terrain && typeof this.terrain.getHeight === 'function') {
      const h = this.terrain.getHeight(camX, camZ);
      if (Number.isFinite(h)) groundY = h;
    } else if (Number.isFinite(state.player.groundHeight)) {
      groundY = state.player.groundHeight;
    }
    // The GROUND contribution is smoothed both ways: terrain rolls under a
    // walker slowly and there is nothing to chase, and smoothing it here stops
    // the one-way rule below from ratcheting the band up to the local high
    // ground and leaving it there.
    const groundWant = groundY + BAND_OVER_GROUND;
    this._groundBase = this._seeded ? damp(this._groundBase, groundWant, 0.9, dt) : groundWant;
    const camFloor = _cam.y + BAND_OVER_CAMERA;
    const wantBase = Math.max(this._groundBase, camFloor);

    let nextBase;
    if (!this._seeded) {
      nextBase = wantBase;
    } else if (wantBase > this._bandBase) {
      // Rising: INSTANT, not smoothed. Fly mode climbs at over 100 m/s and any
      // damped chase lags by v/lambda, so the eye overtakes the band inside half
      // a second and the flock ends up below the horizon - against the field,
      // which is the one thing these must never do. Following instantly is free
      // because the birds are carried WITH the band, exactly like the horizontal
      // rebase above: a rigid translation of the whole volume produces no
      // relative motion at all and therefore cannot be seen.
      nextBase = wantBase;
    } else {
      // Falling: slow, and the birds are deliberately NOT dragged down with it.
      // They relax toward the new roost altitude through their own altitude
      // spring over a few seconds, which is what a descending player should see.
      nextBase = damp(this._bandBase, wantBase, 0.45, dt);
    }
    const dyBand = nextBase - this._bandBase;
    this._bandBase = nextBase;
    // Carry the birds rigidly ONLY while the CAMERA is the term setting the
    // floor. That is the case the rigid translation exists for: an eye climbing
    // at 100 m/s is faster than any spring, so the volume has to move with it,
    // and because nothing moves relative to anything else the translation cannot
    // be seen. Measured: with the gate, a bird's position relative to the eye
    // moves 0.04 m per frame during a 100 m/s climb - its own flight, nothing
    // else. Without the carry entirely it would be the full 1.67 m per frame.
    //
    // A rise driven by TERRAIN rolling under the player is NOT carried. The
    // ground term is followed instantly going up and over ~2 s coming down, so
    // carrying it biases the flock upward a little more with every hill crested,
    // with only the 21-second altitude spring to bring it back - measured at
    // +4.9 m of a 190 m band after 90 s of flying at 100 m/s over 34 m relief
    // (mean height above bandBase 113.9 m carried vs 109.0 m uncarried). Small,
    // but it is bias with no upside: at ground-term speeds (15 m/s at worst, and
    // 0.7 m/s while walking) the per-bird altitude spring tracks the band on its
    // own, so the carry buys nothing there.
    //
    // Either way the "never below the eye" guarantee comes from the hard floor
    // clamp in _simulate, which is unconditional and does not depend on this.
    if (dyBand > 0 && this._seeded && camFloor >= this._groundBase) {
      const py = this.py;
      for (let i = 0; i < MAX_BIRDS; i++) py[i] += dyBand;
    }
    this.geometry.boundingSphere.center.y = this._bandBase + BAND_HEIGHT * 0.5;

    if (!this._seeded) {
      this._seedPositions();
      this._seeded = true;
    }

    this._startle = this._startle > 0 ? this._startle * Math.exp(-1.9 * dt) : 0;
    if (this._startle < 1e-3) this._startle = 0;

    this._updateEnvironment(dt, state);
    this._updateGroups(dt, state, camX, camZ, dx, dz);
    this._simulate(dt, state, camX, camZ);
    // Grown from what _simulate actually published this frame, never shrunk
    // below the nominal volume, and padded for the widened mesh of the single
    // most distant bird (half-span 1.03 m, span widening capped at 5x).
    this.geometry.boundingSphere.radius = Math.max(this._baseSphereR, this._maxSpreadR + 12);
    this._upload();
  }

  /** One-off placement, deferred until the altitude band is known. */
  _seedPositions() {
    for (let g = 0; g < MAX_GROUPS; g++) this._seedGroup(g);
  }

  /**
   * The roost's wander term, as a pair in -1..1. Split into methods because the
   * seeding path and the per-frame path have to agree to the last bit: seeding
   * cancels the wander against the anchor so a re-seeded group's roost lands
   * exactly where it was asked to, and any drift between two copies of this
   * expression would show up as a jump on the frame after a recycle.
   */
  _roostWalkX(seed, t) {
    return this.noise.noise2D(seed * 1.37 + t * 0.021, seed * 0.41);
  }

  _roostWalkZ(seed, t) {
    return this.noise.noise2D(seed * 0.79, seed * 2.13 + t * 0.019);
  }

  /**
   * Place one group somewhere plausible in the volume. Called at boot, whenever
   * a group wakes from dormancy, and when a group is recycled back into the
   * volume - so a flock that returns at dawn drifts in from a sensible distance
   * instead of resuming wherever it was left.
   *
   * `dirX/dirZ`, when non-zero, is the unit direction the group must be placed
   * along: the recycle path passes the reverse of where the group left, which
   * for a moving player is always ahead of them.
   */
  _seedGroup(g, dirX = 0, dirZ = 0) {
    const r = this.rng;
    const lone = g < MAX_LONERS;
    const spread = lone ? ROOST_SPREAD_LONE : ROOST_SPREAD_FLOCK;
    const t = this._time;
    const s = this.gSeed[g];
    const wanderX = this._roostWalkX(s, t) * spread * ROOST_WANDER;
    const wanderZ = this._roostWalkZ(s, t) * spread * ROOST_WANDER;
    // Anchor first, then the roost on top of it, so the roost is a world point
    // from the very first frame and the player can genuinely leave it behind.
    let ax;
    let az;
    if (dirX !== 0 || dirZ !== 0) {
      ax = dirX * RESEED_R - wanderX;
      az = dirZ * RESEED_R - wanderZ;
    } else {
      ax = r.range(-1, 1) * spread * ROOST_ANCHOR;
      az = r.range(-1, 1) * spread * ROOST_ANCHOR;
    }
    this.gAnchX[g] = ax;
    this.gAnchZ[g] = az;
    const cx = ax + wanderX;
    const cz = az + wanderZ;
    const cy = this._bandBase + BAND_HEIGHT * r.range(0.2, 0.85);
    this.gCx[g] = cx;
    this.gCy[g] = cy;
    this.gCz[g] = cz;
    this.gRoostX[g] = cx;
    this.gRoostY[g] = cy;
    this.gRoostZ[g] = cz;
    this.gFadeIn[g] = 0;
    const heading = r() * TAU;
    const hx = Math.cos(heading);
    const hz = Math.sin(heading);
    const start = this.gStart[g];
    const size = this.gSize[g];
    for (let k = 0; k < size; k++) {
      const i = start + k;
      // A loose ellipsoid stretched along the heading - a flock in transit is
      // never a ball.
      this.px[i] = cx + r.gaussian(0, 9) + hx * r.gaussian(0, 7);
      this.py[i] = cy + r.gaussian(0, 5);
      this.pz[i] = cz + r.gaussian(0, 9) + hz * r.gaussian(0, 7);
      const s = this.cruise[i];
      this.vx[i] = hx * s + r.gaussian(0, 1.2);
      this.vy[i] = r.gaussian(0, 0.4);
      this.vz[i] = hz * s + r.gaussian(0, 1.2);
      this.alpha[i] = 0;
      this.roll[i] = 0;
      this.soar[i] = 0;
      this.thermal[i] = 0;
    }
  }

  // -------------------------------------------------------------------------

  _updateEnvironment(dt, state) {
    const u = this.uniforms;

    // Angular size of one pixel. Read from the real drawing buffer so the
    // sub-pixel widening survives adaptive resolution.
    this.renderer.getDrawingBufferSize(_size);
    const h = Math.max(1, _size.y);
    const fov = ((this.camera.fov || 45) * Math.PI) / 180;
    u.uPixelScale.value = (2 * Math.tan(fov * 0.5)) / h;

    // Key light: whichever of sun and moon is actually doing the lighting.
    const sun = state.sun;
    const moon = state.moon;
    const sunW = Math.max(0, sun.intensity) * clamp01(sun.visibility);
    const moonW = Math.max(0, moon.intensity) * clamp01(moon.visibility);
    const total = sunW + moonW;
    if (total > 1e-5) {
      const f = sunW / total;
      let kx = sun.direction.x * f + moon.direction.x * (1 - f);
      let ky = sun.direction.y * f + moon.direction.y * (1 - f);
      let kz = sun.direction.z * f + moon.direction.z * (1 - f);
      const kl = Math.sqrt(kx * kx + ky * ky + kz * kz);
      if (kl > 1e-4) {
        kx /= kl;
        ky /= kl;
        kz /= kl;
      } else {
        const src = f > 0.5 ? sun.direction : moon.direction;
        kx = src.x;
        ky = src.y;
        kz = src.z;
      }
      u.uKeyDir.value.set(kx, ky, kz);
      _col.copy(sun.color).multiplyScalar(Math.min(sun.intensity, 8) * clamp01(sun.visibility));
      _col2.copy(moon.color).multiplyScalar(Math.min(moon.intensity, 8) * clamp01(moon.visibility));
      u.uKeyColor.value.copy(_col).add(_col2).multiplyScalar(0.30);
    } else {
      u.uKeyDir.value.set(0, 1, 0);
      u.uKeyColor.value.setRGB(0, 0, 0);
    }

    // Sky light on the topside, ground bounce on the underside. Kept low: these
    // must stay clearly darker than the sky behind them at every hour, which is
    // the entire reason they read as birds and not as confetti.
    // `|| 0` throughout this method rather than Number.isFinite: NaN is falsy, so
    // it collapses to the safe default for free, and these are read every frame.
    const amb = Math.max(0.05, state.sky.ambientIntensity || 0);
    _col.copy(state.sky.zenithColor).lerp(state.sky.horizonColor, 0.45).multiplyScalar(amb * 0.30);
    u.uSkyAmbient.value.copy(_col);
    _col2.copy(state.sky.groundColor).multiplyScalar(amb * 0.16);
    u.uSkyBounce.value.copy(_col2);

    u.uFlash.value = clamp01(state.weather.lightning || 0);

    // This mesh unavoidably draws in FRONT of the cloud layer - the volumetrics
    // are opaque and this is transparent, so no renderOrder can put a bird
    // behind them. That is only correct while the whole altitude band sits clear
    // of the deck. Fade the system out across exactly that overlap rather than
    // punching hard silhouettes through the volumetrics.
    //
    // Tested on the BAND, not on the camera. The camera test this replaced
    // missed the case that actually happens: the eye levels off fifty metres
    // UNDER the cloud base, reads as nowhere near it, while the band - which is
    // pinned above the eye and reaches 190 m higher - is squarely inside it.
    // Defaulted, not trusted: a NaN deck would make every smoothstep NaN and
    // uGlobalAlpha with it, and a NaN alpha does not fail the shader's `<` kill - 
    // it rasterises whatever min(NaN, 1.0) happens to be on this driver.
    const deckBase = state.clouds.altitude || 900;
    const deckTop = deckBase + (state.clouds.thickness || 700);
    const bandTop = this._bandBase + BAND_HEIGHT;
    const intrusion =
      smoothstep(deckBase - 40, deckBase + 90, bandTop) *
      (1 - smoothstep(deckTop - 60, deckTop + 140, this._bandBase));
    u.uGlobalAlpha.value = 1 - 0.92 * intrusion * clamp01(state.clouds.coverage || 0);

    this._updateAerial(state);
  }

  /**
   * Copy the aerial-perspective parameters off weather/atmosfx.js when it is
   * present so the birds sit in exactly the same air as the treeline, and fall
   * back to scene.fog otherwise. Copies rather than aliases: sharing uniform
   * objects across systems makes an ownership question out of what is a dozen
   * float writes.
   */
  _updateAerial(state) {
    const u = this.uniforms;
    const fx = this.atmosfx;
    const fu = fx && fx.uniforms;

    if (fu && fu.uAerialHaze && fu.uAerialMist) {
      u.uBirdHaze.value.copy(fu.uAerialHaze.value);
      u.uBirdMist.value.copy(fu.uAerialMist.value);
      u.uBirdZenith.value.copy(fu.uAerialZenith.value);
      u.uBirdHorizonSun.value.copy(fu.uAerialHorizonSun.value);
      u.uBirdHorizonAway.value.copy(fu.uAerialHorizonAway.value);
      u.uBirdGround.value.copy(fu.uAerialGround.value);
      u.uBirdMistColor.value.copy(fu.uAerialMistColor.value);
      u.uBirdMaxOpacity.value = Number.isFinite(fx.maxFogOpacity) ? fx.maxFogOpacity : 0.965;
      return;
    }

    // Fallback. three's FogExp2 gives 1 - exp(-(d*x)^2); ours gives 1 - exp(-k*x).
    // The two agree at 220 m for k = d^2 * 220, the same match atmosfx uses when
    // it publishes its own FogExp2 stand-in.
    const fog = this.scene.fog;
    let d = state.weather.fogDensity;
    if (fog && fog.isFogExp2 && Number.isFinite(fog.density)) d = fog.density;
    d = clamp(Number.isFinite(d) ? d : 0.0022, 0, 0.5);
    const k = d * d * 220;
    // Height falloff of a nominal 1.2 km haze scale height: birds sit well up in
    // it, so a flat density would fog them roughly twice as hard as it should.
    u.uBirdHaze.value.set(k, 1 / 1200, 0, 0);
    u.uBirdMist.value.set(0, 0.05, 0, 0);
    u.uBirdMistColor.value.copy(state.weather.fogColor);
    u.uBirdZenith.value.copy(state.sky.zenithColor);
    u.uBirdHorizonSun.value.copy(state.sky.horizonColor);
    u.uBirdHorizonAway.value.copy(state.sky.horizonColor);
    u.uBirdGround.value.copy(state.sky.groundColor);
    u.uBirdMaxOpacity.value = 0.965;
  }

  // -------------------------------------------------------------------------

  /**
   * Population, roost drift and the per-group orbit parameters. This is where
   * the flocks decide where to be; _simulate only works out how to get there.
   */
  _updateGroups(dt, state, camX, camZ, dx, dz) {
    // The SANITISED clock, not state.time.elapsed. A single non-finite elapsed
    // would go straight into noise2D and out into gRoostX/Z, and from there into
    // px/py/pz - where it is permanent, because every recovery path in here is a
    // damp() and damp(NaN, finite) is NaN forever. update() already keeps a
    // last-good copy for exactly this reason; it has to be the one everybody uses.
    const t = this._time;
    const noise = this.noise;
    const sun = state.sun;
    const weather = state.weather;
    const wind = state.wind;

    // --- how much bird life the world supports right now -------------------
    // Diurnal, with the classic dawn and dusk peaks: birds commute at the edges
    // of the day and loaf in the middle of it.
    const elev = Number.isFinite(sun.elevation) ? sun.elevation : 0;
    const day = clamp01(sun.visibility || 0);
    const e = (elev - 0.16) / 0.20;
    const goldenHour = Math.exp(-e * e);
    const activity = clamp01(0.06 + 0.62 * day + 0.42 * goldenHour * clamp01(day * 3));

    // Birds shelter in bad weather. Rain empties the sky; a storm empties it
    // completely; strong wind aloft keeps all but the big soarers down.
    const wet = Math.max(weather.rainIntensity || 0, weather.snowIntensity || 0);
    const shelter = clamp01(wet * 1.25 + clamp01(state.clouds.storminess || 0) * 0.8);
    const windScare = smoothstep(9, 19, (wind.strength || 0) * (wind.gust || 1));
    const fogHide = smoothstep(0.004, 0.020, weather.fogDensity || 0);
    const popTarget = clamp01(
      activity * (1 - 0.94 * shelter) * (1 - 0.6 * windScare) * (1 - 0.75 * fogHide)
    );
    // Very slow: a sky that repopulates in under half a minute reads as a
    // spawner. Three seconds of time constant, plus the per-group fades.
    this._pop = damp(this._pop, popTarget, 0.35, dt);

    const pop = this._pop;
    // Loners persist far longer than flocks - a single bird crossing a leaden
    // sky is the shot worth keeping. Flocks need genuinely good conditions.
    const lonerWant = this.tier.loners * Math.pow(pop, 0.45);
    const flockWant = this.tier.flocks * Math.pow(pop, 1.25);
    const perFlock = this.tier.perFlock;

    for (let g = 0; g < MAX_GROUPS; g++) {
      const lone = g < MAX_LONERS;
      const rank = lone ? g : g - MAX_LONERS;
      const cap = lone ? this.tier.loners : this.tier.flocks;
      const want = lone ? lonerWant : flockWant;
      const target = rank < cap ? clamp01(want - rank) : 0;
      // Asymmetric on purpose. A flock arriving must drift in slowly enough that
      // nobody catches it appearing; a flock leaving is just a flock leaving, and
      // dragging that out keeps dead weight in the simulation for half a minute.
      const size = this.gSize[g];
      this.gWeight[g] = damp(this.gWeight[g], target, target > this.gWeight[g] ? 0.4 : 1.1, dt);
      // Retire once the leading member's alpha is down where a three-pixel dark
      // mark against a bright sky cannot be told from nothing. Without this the
      // exponential tail keeps a flock nominally alive - and in the instance
      // count - long after it should have gone.
      if (target <= 0 && this.gWeight[g] * size < 0.02) this.gWeight[g] = 0;
      else if (target >= 1 && 1 - this.gWeight[g] < 1e-3) this.gWeight[g] = 1;

      // Camera-relative bookkeeping. The anchor is rebased with the centroid - 
      // that subtraction is the whole of "the roost is a world point".
      this.gCx[g] -= dx;
      this.gCz[g] -= dz;
      this.gAnchX[g] -= dx;
      this.gAnchZ[g] -= dz;

      const live = this.gWeight[g] > 0.0015 || this.gAlphaMax[g] > 0.004;
      if (!live) {
        this.gFadeIn[g] = 0;
        this.gDormant[g] = 1;
        continue;
      }
      if (this.gDormant[g]) {
        this.gDormant[g] = 0;
        this._seedGroup(g);
      }

      // --- recycle ---------------------------------------------------------
      // The group has left the volume - either the player outran it or its own
      // wander carried it out. It is faded to nothing by then (the edge fade
      // runs from 0.74 to 0.97 of WRAP_R, ~99 m of travel), so re-seeding it
      // outright is unseeable and gives a genuinely different flock back rather
      // than the same one coming round again.
      //
      // Gated on the group having ACTUALLY faded, not merely on the distance:
      // the edge fade is a distance ramp but alpha follows it through a 0.45 s
      // damp, so a player closing on the boundary fast outruns the fade. At
      // 40 m/s the group was still 19% opaque when it teleported and at 120 m/s
      // it was 48% - a visible blink. Waiting costs nothing but a few hundred
      // metres of invisible simulation, which the bounding sphere now tracks.
      const cheb = Math.max(Math.abs(this.gCx[g]), Math.abs(this.gCz[g]));
      if (cheb > WRAP_R && this.gAlphaMax[g] < 0.004) {
        const len = Math.sqrt(this.gCx[g] * this.gCx[g] + this.gCz[g] * this.gCz[g]) || 1;
        this._seedGroup(g, -this.gCx[g] / len, -this.gCz[g] / len);
      }
      this.gFadeIn[g] = damp(this.gFadeIn[g], 1, 0.9, dt);

      // --- roost drift ------------------------------------------------------
      // World anchor plus two decorrelated simplex walks. Slow enough that the
      // flock always overshoots and has to come back, which is what makes it
      // wheel - but anchored, so it wheels over a fixed patch of ground.
      const s = this.gSeed[g];
      const spread = lone ? ROOST_SPREAD_LONE : ROOST_SPREAD_FLOCK;
      const wander = spread * ROOST_WANDER;
      this.gRoostX[g] = this.gAnchX[g] + this._roostWalkX(s, t) * wander;
      this.gRoostZ[g] = this.gAnchZ[g] + this._roostWalkZ(s, t) * wander;
      const na = noise.noise2D(s * 0.33 + t * 0.020, 91.5);
      this.gRoostY[g] = this._bandBase + (0.52 + 0.40 * na) * BAND_HEIGHT;

      // Tangential gain passes smoothly through zero: the flock spirals one way,
      // straightens out, mills, then wheels back the other. It cannot collapse
      // while doing so because the radial term is repulsive inside the orbit.
      this.gSpin[g] = noise.noise2D(s * 0.55 + t * 0.017, 33.0) * 1.45;
      const orbN = 0.5 + 0.5 * noise.noise2D(s * 0.91 + t * 0.008, 77.0);
      if (lone) {
        // Loners alternate, on a period of several minutes, between thermalling
        // tightly and simply crossing the sky. The crossing is expressed by
        // RELEASING the roost, not by inflating the orbit: a huge orbit radius
        // would park the bird on the volume boundary where the edge fade lives,
        // and a lone bird crossing high is the whole point of having them.
        const mode = smoothstep(-0.18, 0.18, noise.noise2D(s * 1.73 + t * 0.006, 5.0));
        this.gOrbit[g] = lerp(46 + orbN * 40, 200, mode);
        this.gPull[g] = lerp(1.0, 0.12, mode);
      } else {
        this.gOrbit[g] = 52 + orbN * 52;
        this.gPull[g] = 1;
      }

      // --- wind, sampled once per group ------------------------------------
      // Wind is coherent over far more than a flock's span, so one CPU sample at
      // the centroid is both cheaper and more correct than one per bird.
      _wind.x = 0;
      _wind.y = 0;
      _wind.z = 0;
      if (typeof wind.sample === 'function') wind.sample(camX + this.gCx[g], camZ + this.gCz[g], t, _wind);
      let gust = 1;
      if (typeof wind.getGustAt === 'function') {
        const gv = wind.getGustAt(camX + this.gCx[g], camZ + this.gCz[g], t);
        if (Number.isFinite(gv)) gust = gv;
      }
      const k = WIND_ALOFT * gust;
      this.gWindX[g] = (Number.isFinite(_wind.x) ? _wind.x : 0) * k;
      this.gWindY[g] = (Number.isFinite(_wind.y) ? _wind.y : 0) * k * 0.5;
      this.gWindZ[g] = (Number.isFinite(_wind.z) ? _wind.z : 0) * k;
    }

    // Thermal strength: convection needs sun on the ground and dies under cloud
    // and under rain. This is why birds soar at midday and flap at dawn.
    // `coverage || 0` is load-bearing, not defensive noise: thermalK multiplies
    // straight into thermal[i], which is a PERSISTENT array feeding ay, so a
    // single NaN coverage would poison the flock's velocities permanently.
    this._thermalStrength =
      2.6 *
      clamp01(Math.sin(Math.max(elev, 0)) * 1.4) *
      (1 - 0.75 * clamp01(state.clouds.coverage || 0)) *
      (1 - clamp01(wet));
    this._perFlockLive = perFlock;
  }

  // -------------------------------------------------------------------------

  _simulate(dt, state, camX, camZ) {
    const {
      px, py, pz, vx, vy, vz, roll, phase, flapHz, span, cruise, alpha,
      soar, soarBias, thermal, yOffset, stx, sty, stz, sepR2, wanX, wanZ, wanB,
      boutHz, boutPh, boutAmt,
    } = this;
    const t = this._time;   // sanitised clock - see _updateGroups
    const bandBase = this._bandBase;
    const bandTop = bandBase + BAND_HEIGHT;
    const startle = this._startle;
    const thermalK = this._thermalStrength;
    const frame = this._frame;

    // Tree exclusion volume, in camera-relative coordinates. The altitude band
    // already clears the canopy by 40 m; this is the belt-and-braces guarantee.
    let treeX = 0;
    let treeZ = 0;
    let treeR2 = 0;
    let treeTop = 0;
    if (this.tree && this.tree.canopyCenter && Number.isFinite(this.tree.canopyRadius)) {
      const c = this.tree.canopyCenter;
      if (Number.isFinite(c.x) && Number.isFinite(c.z)) {
        treeX = c.x - camX;
        treeZ = c.z - camZ;
        const r = this.tree.canopyRadius * 1.6 + 6;
        treeR2 = r * r;
        treeTop = (Number.isFinite(c.y) ? c.y : 0) + this.tree.canopyRadius + 12;
      }
    }

    // One rotation per wander bucket, reused by every bird in it.
    for (let k = 0; k < WANDER_BUCKETS; k++) {
      const ang = WANDER_RATE[k] * dt;
      _wanC[k] = Math.cos(ang);
      _wanS[k] = Math.sin(ang);
    }

    let liveMax = 0;
    let anyVisible = false;
    // Widest published bird this frame, measured from the bounding sphere's own
    // centre, so the sphere can be grown to actually contain the instances.
    const sphereY = this.geometry.boundingSphere.center.y;
    let maxSpread2 = 0;

    for (let g = 0; g < MAX_GROUPS; g++) {
      const weight = this.gWeight[g];
      const groupLive = weight > 0.0015 || this.gAlphaMax[g] > 0.004;
      const start = this.gStart[g];
      const size = this.gSize[g];
      if (!groupLive) {
        this.gAlphaMax[g] = 0;
        for (let k = 0; k < size; k++) {
          const i = start + k;
          alpha[i] = 0;
          // The instance buffer keeps whatever was last written to it. A group
          // retires with an alpha of up to 0.004 still in there, which is ABOVE
          // the vertex shader's 0.0035 kill threshold, so its birds went on
          // being rasterised - frozen in place - whenever a live group with a
          // higher slot index pushed instanceCount past them. Measured at 9355
          // slot-frames over a 5000-frame night run.
          this.iMotion[i * 4 + 3] = 0;
        }
        continue;
      }

      const lone = g < MAX_LONERS;
      // Members thin from the tail as the group's weight drops, so a fading
      // flock loses stragglers instead of dimming as a block.
      const members = lone ? 1 : Math.min(size, this._perFlockLive);
      const cheb = Math.max(Math.abs(this.gCx[g]), Math.abs(this.gCz[g]));
      const edge = 1 - smoothstep(EDGE_FADE_IN, EDGE_FADE_OUT, cheb);
      const groupVis = edge * this.gFadeIn[g];

      const roostX = this.gRoostX[g];
      const roostY = this.gRoostY[g];
      const roostZ = this.gRoostZ[g];
      // Last frame's centroid, used as the origin of the flapping wave below.
      const prevCx = this.gCx[g];
      const prevCz = this.gCz[g];
      const pull = this.gPull[g];
      const spin = this.gSpin[g] * pull;
      const orbit = this.gOrbit[g];
      const invOrbit = pull / orbit;
      const windX = this.gWindX[g];
      const windY = this.gWindY[g];
      const windZ = this.gWindZ[g];

      let sumX = 0;
      let sumY = 0;
      let sumZ = 0;
      let alphaMax = 0;
      let counted = 0;

      for (let k = 0; k < size; k++) {
        const i = start + k;

        // --- visibility ------------------------------------------------------
        const thin = clamp01(weight * members - k);
        const aTarget = thin * groupVis;
        alpha[i] = damp(alpha[i], aTarget, 2.2, dt);
        if (alpha[i] < 3e-4 && aTarget <= 0) alpha[i] = 0;
        if (alpha[i] > alphaMax) alphaMax = alpha[i];
        if (alpha[i] <= 0 && aTarget <= 0) {
          this.iMotion[i * 4 + 3] = 0;
          continue;
        }
        if (alpha[i] > 0.0035) {
          if (i + 1 > liveMax) liveMax = i + 1;
          anyVisible = true;
        }

        // --- thermals, round-robin over four frames ---------------------------
        // Worley cells are the right model for convection: isolated columns of
        // lift with sink between them, not a smooth wave. They move far slower
        // than a bird does, so a quarter-rate refresh is indistinguishable.
        if (((i + frame) & 3) === 0) {
          const wx = (camX + px[i]) * THERMAL_FREQ + t * 0.0016;
          const wz = (camZ + pz[i]) * THERMAL_FREQ;
          const cell = worley2D(wx, wz, 17);
          // Core lift, ringed by compensating sink - the reason a soaring bird
          // circles rather than tracking straight up a column.
          thermal[i] = (smoothstep(0.44, 0.02, cell) - 0.22 * smoothstep(0.42, 0.72, cell)) * thermalK;
        }

        // --- wander -------------------------------------------------------------
        // Advance this bird's wander vector by rotating it; no trig per bird.
        const wb = wanB[i];
        const wc = _wanC[wb];
        const ws = _wanS[wb];
        const nwx = wanX[i] * wc - wanZ[i] * ws;
        const nwz = wanX[i] * ws + wanZ[i] * wc;
        wanX[i] = nwx;
        wanZ[i] = nwz;

        let ax = nwx * WANDER_ACCEL;
        let ay = nwz * WANDER_ACCEL * 0.35;
        let az = nwz * WANDER_ACCEL;

        // --- bout (flap-glide) flight -----------------------------------------
        // The clock runs for every bird whether or not it is bounding, so a bird
        // coming off a thermal resumes on the correct part of the cycle instead
        // of snapping into a burst. `soar` is last frame's value, which is fine:
        // it is a damped quantity that takes the better part of a second to move.
        let bp = boutPh[i] + boutHz[i] * dt;
        bp -= Math.floor(bp);
        boutPh[i] = bp;
        let glide = 0;
        const boutK = boutAmt[i] * (1 - soar[i]);
        if (boutK > 0.01) {
          const powered =
            smoothstep(0, BOUT_RAMP_IN, bp) *
            (1 - smoothstep(BOUT_POWER_END, BOUT_POWER_END + BOUT_RAMP_OUT, bp));
          glide = (1 - powered) * boutK;
          // Climbs on the powered burst, falls on the coast. Mean-subtracted, so
          // the bob rides on top of the altitude spring without biasing it.
          ay += (powered - BOUT_MEAN) * boutK * BOUT_LIFT;
        }

        // --- boids, within the flock only -------------------------------------
        if (size > 1) {
          let nCount = 0;
          let avx = 0;
          let avy = 0;
          let avz = 0;
          let apx = 0;
          let apy = 0;
          let apz = 0;
          const xi = px[i];
          const yi = py[i];
          const zi = pz[i];
          const sep2 = sepR2[i];
          for (let m = 0; m < members; m++) {
            if (m === k) continue;
            const j = start + m;
            const ddx = xi - px[j];
            const ddy = yi - py[j];
            const ddz = zi - pz[j];
            const d2 = ddx * ddx + ddy * ddy + ddz * ddz;
            // Written as a positive range test, not `d2 > R2 || d2 < eps`, so a
            // non-finite neighbour is REJECTED rather than accepted: NaN fails
            // every comparison, so the negated form skipped it and the old form
            // let it through into the cohesion and alignment sums. One poisoned
            // bird therefore took its entire flock with it on the same frame.
            if (!(d2 >= 1e-6 && d2 <= NEIGH_RADIUS2)) continue;
            nCount++;
            avx += vx[j];
            avy += vy[j];
            avz += vz[j];
            apx += px[j];
            apy += py[j];
            apz += pz[j];
            if (d2 < sep2) {
              // Inverse-square repulsion, capped so a near-coincident pair does
              // not fire a bird across the sky.
              const kk = Math.min(1 / d2, 4) * SEP_W;
              ax += ddx * kk;
              ay += ddy * kk * 1.6;   // birds separate vertically more readily
              az += ddz * kk;
            }
          }
          if (nCount > 0) {
            const inv = 1 / nCount;
            ax += (avx * inv - vx[i]) * ALIGN_W;
            ay += (avy * inv - vy[i]) * ALIGN_W;
            az += (avz * inv - vz[i]) * ALIGN_W;
            ax += (apx * inv - xi) * COH_W;
            ay += (apy * inv - yi) * COH_W * 0.6;
            az += (apz * inv - zi) * COH_W;
          }
        }

        // --- orbit the roost ---------------------------------------------------
        // A radial spring toward `orbit` plus a tangential push. Together these
        // are a limit cycle: the flock settles onto a ring it never quite holds,
        // and because both the ring and the spin drift it wheels and reforms.
        const tox = roostX - px[i];
        const toz = roostZ - pz[i];
        const d = Math.sqrt(tox * tox + toz * toz) + 1e-4;
        const ux = tox / d;
        const uz = toz / d;
        const radial = clamp((d - orbit) * invOrbit, -1.4, 1.4);
        ax += ux * radial * ROOST_RADIAL - uz * spin * ROOST_TANGENT;
        az += uz * radial * ROOST_RADIAL + ux * spin * ROOST_TANGENT;

        // --- altitude ----------------------------------------------------------
        const targetY = roostY + yOffset[i];
        ay += (targetY - py[i]) * ALT_SPRING - vy[i] * ALT_DAMP;
        ay += thermal[i] * 0.45;

        // --- wind ---------------------------------------------------------------
        // Drag toward the air's velocity, not a bulk push: that is what lets a
        // bird hold station into a headwind and get shoved downwind on a beam.
        ax += (windX - vx[i]) * WIND_DRAG;
        ay += (windY - vy[i]) * WIND_DRAG * 0.5;
        az += (windZ - vz[i]) * WIND_DRAG;

        // --- startle -------------------------------------------------------------
        if (startle > 0) {
          ax += stx[i] * startle * 10;
          ay += sty[i] * startle * 6 - startle * 2.5;
          az += stz[i] * startle * 10;
        }

        // --- clamp and integrate ---------------------------------------------------
        const a2 = ax * ax + ay * ay + az * az;
        if (a2 > MAX_ACCEL2) {
          const sc = MAX_ACCEL / Math.sqrt(a2);
          ax *= sc;
          ay *= sc;
          az *= sc;
        }

        let nvx = vx[i] + ax * dt;
        let nvy = vy[i] + ay * dt;
        let nvz = vz[i] + az * dt;
        let speed = Math.sqrt(nvx * nvx + nvy * nvy + nvz * nvz);
        // A non-finite value reaching this point is PERMANENT, not transient.
        // Every recovery path in this loop is a damp() or an accumulation and
        // both propagate NaN forever, and the bird is not even culled on the way:
        // the vertex shader's kill is `a < 0.0035`, NaN fails every comparison,
        // so the slot goes on emitting a NaN gl_Position for the rest of the
        // session. Everything this loop reads is owned by another system and can
        // be mid-transition (clouds.coverage through thermalK, wind.sample, the
        // roost noise), so one bad frame anywhere upstream is enough to lose a
        // bird permanently. `!(speed < 1e9)` catches NaN and Infinity in a single
        // comparison, and it catches poisoned POSITIONS as well: a non-finite
        // px/py/pz reaches ax/ay/az through the orbit spring and the altitude
        // spring on the very frame it appears.
        if (!(speed < 1e9)) {
          px[i] = roostX;
          py[i] = roostY;
          pz[i] = roostZ;
          nvx = cruise[i];
          nvy = 0;
          nvz = 0;
          speed = cruise[i];
          // The steering accelerations have to go too: the coordinated-turn
          // solve below reads ax/az, so leaving them poisoned would re-infect
          // roll[i] through its damp() every single frame while the position
          // looked perfectly healthy.
          ax = 0;
          ay = 0;
          az = 0;
          // These are damped from their own previous value rather than
          // recomputed, so clearing the kinematics alone would leave them NaN.
          roll[i] = 0;
          soar[i] = 0;
          alpha[i] = 0;
          thermal[i] = 0;
          phase[i] = 0;
          boutPh[i] = 0;
        } else if (speed < 1e-4) {
          nvx = 1;
          speed = 1;
        }

        // Speed trades against climb angle: a bird bleeds airspeed going up and
        // gains it coming down. Without this a climbing flock looks weightless.
        const climb = nvy / speed;
        const want = clamp(cruise[i] * (1 - 0.42 * climb), 7.0, 24.0);
        const blend = 1 - Math.exp(-1.8 * dt);
        const scale = (speed + (want - speed) * blend) / speed;
        nvx *= scale;
        nvy *= scale;
        nvz *= scale;
        speed *= scale;

        vx[i] = nvx;
        vy[i] = nvy;
        vz[i] = nvz;
        px[i] += nvx * dt;
        py[i] += nvy * dt;
        pz[i] += nvz * dt;

        // --- floor, ceiling and the tree --------------------------------------
        if (py[i] < bandBase) {
          py[i] = bandBase;
          if (vy[i] < 0) vy[i] *= -0.15;
        } else if (py[i] > bandTop + 40) {
          py[i] = bandTop + 40;
          if (vy[i] > 0) vy[i] *= -0.15;
        }
        if (treeR2 > 0) {
          const tx = px[i] - treeX;
          const tz = pz[i] - treeZ;
          if (tx * tx + tz * tz < treeR2 && py[i] < treeTop) {
            py[i] = treeTop;
            if (vy[i] < 0) vy[i] = 0;
          }
        }

        // --- bank ---------------------------------------------------------------
        // Coordinated turn: roll until the lift vector's horizontal component
        // supplies the lateral acceleration the steering asked for.
        const invS = 1 / speed;
        const fx = nvx * invS;
        const fz = nvz * invS;
        const hl = Math.sqrt(fx * fx + fz * fz);
        if (hl > 1e-4) {
          const rx = -fz / hl;
          const rz = fx / hl;
          const lat = ax * rx + az * rz;
          const target = clamp(-Math.atan(lat / BANK_G) * 1.15, -MAX_ROLL, MAX_ROLL);
          roll[i] = damp(roll[i], target, 4.5, dt);
        }

        // --- soar or flap --------------------------------------------------------
        // Wings go still on lift and on a glide down; they beat hardest climbing
        // out of sink. Big birds give up flapping far sooner than small ones.
        const lift = thermal[i] + Math.max(0, -nvy) * 0.35;
        let soarTarget = smoothstep(0.20, 1.05, lift) + soarBias[i];
        soarTarget = clamp01(soarTarget * (0.55 + 0.42 * span[i]));
        soar[i] = damp(soar[i], soarTarget, 1.3, dt);

        let flap = 0.06 + 0.94 * (1 - soar[i]);
        flap = Math.min(1, flap * (1 + 0.4 * Math.max(0, climb)));
        // The coast phase of a bout takes the beat down to almost nothing. This
        // is the loudest single motion cue at range: a flock of dark specks whose
        // blur switches on and off, member by member, out of step.
        flap *= 1 - glide * (1 - BOUT_FLOOR);
        // A little slower when gliding, so the recovery from a glide into a beat
        // eases in rather than snapping to full rate.
        let ph = phase[i] + flapHz[i] * (0.55 + 0.65 * flap) * dt;
        if (ph > 64) ph -= 64;
        phase[i] = ph;

        // --- publish --------------------------------------------------------------
        const p3 = i * 3;
        this.iPos[p3] = px[i];
        this.iPos[p3 + 1] = py[i];
        this.iPos[p3 + 2] = pz[i];
        this.iFwd[p3] = nvx * invS;
        this.iFwd[p3 + 1] = nvy * invS;
        this.iFwd[p3 + 2] = nvz * invS;
        const m4 = i * 4;
        this.iMotion[m4] = roll[i];
        // The beat propagates through a flock: birds out front lead, the ones
        // behind follow a fraction of a cycle later. Real flocks visibly ripple
        // like that, and it is the difference between a flock and a formation.
        // Measured against the flock centroid, so the offset is bounded by the
        // flock's own size and does not drift with the flock's position.
        this.iMotion[m4 + 1] =
          ph + ((px[i] - prevCx) * this.iFwd[p3] + (pz[i] - prevCz) * this.iFwd[p3 + 2]) * 0.011;
        this.iMotion[m4 + 3] = alpha[i];
        const b2 = i * 2;
        this.iBeat[b2] = flap;
        // Soaring dihedral. Small, but it is the difference between a gliding
        // bird and a dead one: flat wings on a glide look like a paper dart. The
        // coast phase of a bout gets the same treatment for the same reason.
        this.iBeat[b2 + 1] = soar[i] * (lone ? 0.16 : 0.11) + glide * 0.075;

        const dyc = py[i] - sphereY;
        const spread2 = px[i] * px[i] + pz[i] * pz[i] + dyc * dyc;
        if (spread2 > maxSpread2) maxSpread2 = spread2;

        sumX += px[i];
        sumY += py[i];
        sumZ += pz[i];
        counted++;
      }

      if (counted > 0) {
        const inv = 1 / counted;
        this.gCx[g] = sumX * inv;
        this.gCy[g] = sumY * inv;
        this.gCz[g] = sumZ * inv;
      }
      this.gAlphaMax[g] = alphaMax;
    }

    this._liveMax = liveMax;
    this._maxSpreadR = Math.sqrt(maxSpread2);
    this.mesh.visible = anyVisible;
  }

  _upload() {
    this.geometry.instanceCount = this._liveMax;
    if (this._liveMax === 0) return;
    this.aPos.needsUpdate = true;
    this.aFwd.needsUpdate = true;
    this.aMotion.needsUpdate = true;
    this.aBeat.needsUpdate = true;
  }

  // -------------------------------------------------------------------------

  dispose() {
    this._disposed = true;
    if (this._onLightning) this.bus?.off?.(EVENTS.LIGHTNING_STRIKE, this._onLightning);
    this._onLightning = null;
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh = null;
    }
    this.geometry?.dispose();
    this.material?.dispose();
    this.geometry = null;
    this.material = null;
  }
}
