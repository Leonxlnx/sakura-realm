/**
 * celestial.js - sun, moon, stars.
 *
 * Everything here is driven by one number: `state.time.timeOfDay`. From it we
 * derive a real ephemeris (days since J2000) and run a genuine low-precision
 * astronomical model for a mid-latitude Japanese site (~35°N, Kyoto). That
 * buys us, for free, all the things that fake "rotate a vector by the clock"
 * sun code never gets right:
 *
 *   - the sun rises north-of-east in summer and south-of-east in winter,
 *   - the day lengthens as the in-game date advances,
 *   - the moon runs ~50 minutes late every day and its phase evolves,
 *   - the terminator on the moon is the real sun-moon-viewer geometry,
 *   - the star field turns about the celestial pole at the sidereal rate,
 *     with the Milky Way locked to it.
 *
 * Coordinate convention for this project: +Y up, −Z north, +X east.
 * Azimuth is measured from north, increasing eastward.
 *
 * @owner state.sun.*, state.moon.*
 */

import * as THREE from 'three';
import {
  TAU, DEG2RAD, RAD2DEG,
  clamp, clamp01, smoothstep, lerp, damp, mod,
  makeRNG, createNoise, kelvinToRGB,
} from '../core/math.js';

// ---------------------------------------------------------------------------
// Site + epoch
// ---------------------------------------------------------------------------

/** Kyoto. The JST standard meridian (135°E) is essentially the site longitude,
 *  so local clock time ≈ local mean solar time and `timeOfDay` reads naturally. */
const SITE_LAT = 35.0 * DEG2RAD;
const SITE_LON_DEG = 135.77;
const SITE_TZ = 9;

/** Days from J2000.0 (2000-01-01 12:00 TT) to 2025-04-05 00:00 UT - peak sakura.
 *  Day 0 of the sim is that date; each in-game midnight advances it by one. */
const SIM_EPOCH_DAYS = 9225.5;
/** TT − UT ≈ 69 s. Small, but it is the difference between solar noon landing
 *  at 11:57 and at 11:59, and it costs one addition. */
const DELTA_T_DAYS = 69 / 86400;

/** Angular diameters. Both are enlarged from the true 0.53° - the real disc is
 *  a pinhead on a 62° fov and reads as a bloom artefact rather than a sun. */
const SUN_DISC_ANG = 0.021;   // rad, ~1.2° (2.3× true)
const SUN_QUAD_ANG = 0.30;    // rad, aureole extent
const MOON_DISC_ANG = 0.026;  // rad, ~1.5° (2.9× true)
const MOON_QUAD_ANG = 0.075;  // rad, halo extent

/** The moon subtends ~23 px at 1080p/62°, so 256² is already an order of
 *  magnitude of oversampling; 512² only bought a slower loading screen. */
const MOON_TEX_SIZE = 256;
const MW_TEX_W = 384;
const MW_TEX_H = 192;

/** sRGB encode LUT - the band bake would otherwise call Math.pow 3× per pixel. */
const SRGB_LUT = (() => {
  const t = new Uint8Array(1025);
  for (let i = 0; i <= 1024; i++) {
    const v = i / 1024;
    t[i] = Math.round((v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055) * 255);
  }
  return t;
})();
const encodeSRGB = (v) => SRGB_LUT[v <= 0 ? 0 : v >= 1 ? 1024 : (v * 1024) | 0];

const MAX_STARS = 9000;
const STAR_MAG_MIN = -1.5;
const STAR_MAG_MAX = 6.4;
const MAX_METEORS = 4;

/** Star counts per tier. LOW keeps only the brightest - the list is flux-sorted,
 *  so this is a drawRange change with zero re-upload. */
const STAR_COUNT_BY_TIER = { low: 1600, medium: 3200, high: 6000, ultra: MAX_STARS };

/** Galactic pole / centre in J2000 equatorial coordinates. */
const NGP_RA = 192.85948 * DEG2RAD;
const NGP_DEC = 27.12825 * DEG2RAD;
const GC_RA = 266.40510 * DEG2RAD;
const GC_DEC = -28.93617 * DEG2RAD;

/** North ecliptic pole in equatorial coordinates. The moon's rotation axis is
 *  within 1.5° of this, so it is what the disc's "up" should track - not the
 *  celestial pole, which would tilt the maria by a fixed 23° error. */
const OBLIQUITY = 23.4393 * DEG2RAD;
const ECL_POLE_EQ_Y = -Math.sin(OBLIQUITY);
const ECL_POLE_EQ_Z = Math.cos(OBLIQUITY);

// ---------------------------------------------------------------------------
// Module-scope scratch. Nothing in update() allocates.
// ---------------------------------------------------------------------------

const _sunEq = { ra: 0, dec: 0, lon: 0 };
const _moonEq = { ra: 0, dec: 0, lon: 0, lat: 0 };
const _v3a = new THREE.Vector3();
const _v3b = new THREE.Vector3();
const _v3c = new THREE.Vector3();
const _right = new THREE.Vector3();
const _up = new THREE.Vector3();
const _fwd = new THREE.Vector3();
const _basis = new THREE.Matrix4();
const _v2 = new THREE.Vector2();
const _rgb = { r: 1, g: 1, b: 1 };
const _trans = { r: 1, g: 1, b: 1 };

// Optical depth at zenith, sampled at 0.65 / 0.55 / 0.45 µm.
// Rayleigh: Bodhaine et al. 1999 fit. Aerosol: Ångström with α = 1.3.
// Ozone Chappuis band peaks in the green-orange; it is why deep twilight is blue.
const TAU_RAYLEIGH = { r: 0.0480, g: 0.0965, b: 0.2160 };
const TAU_AEROSOL = { r: 0.0810, g: 0.1000, b: 0.1280 };
const TAU_OZONE = { r: 0.0090, g: 0.0250, b: 0.0040 };

// ---------------------------------------------------------------------------
// Ephemeris - low-precision series, accurate to a few arcminutes.
// Sun: Astronomical Almanac §C. Moon: Meeus, Astronomical Algorithms ch. 47
// truncated to the six largest longitude terms and four latitude terms.
// ---------------------------------------------------------------------------

function solarEquatorial(d, out) {
  const L = 280.460 + 0.9856474 * d;
  const g = (357.528 + 0.9856003 * d) * DEG2RAD;
  const lon = (L + 1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * DEG2RAD;
  const eps = (23.439 - 0.0000004 * d) * DEG2RAD;
  out.ra = Math.atan2(Math.cos(eps) * Math.sin(lon), Math.cos(lon));
  out.dec = Math.asin(Math.sin(eps) * Math.sin(lon));
  out.lon = lon;
  return out;
}

function lunarEquatorial(d, out) {
  const Lp = 218.316 + 13.176396 * d;              // mean longitude
  const M = (357.528 + 0.9856003 * d) * DEG2RAD;   // sun's mean anomaly
  const Mp = (134.963 + 13.064993 * d) * DEG2RAD;  // moon's mean anomaly
  const F = (93.272 + 13.229350 * d) * DEG2RAD;    // argument of latitude
  const D = (297.850 + 12.190749 * d) * DEG2RAD;   // mean elongation

  const lon = (Lp
    + 6.289 * Math.sin(Mp)              // equation of the centre
    + 1.274 * Math.sin(2 * D - Mp)      // evection
    + 0.658 * Math.sin(2 * D)           // variation
    + 0.214 * Math.sin(2 * Mp)
    - 0.186 * Math.sin(M)               // annual equation
    - 0.114 * Math.sin(2 * F)) * DEG2RAD;

  const lat = (5.128 * Math.sin(F)
    + 0.280 * Math.sin(Mp + F)
    - 0.278 * Math.sin(F - Mp)
    - 0.173 * Math.sin(2 * D - F)) * DEG2RAD;

  const eps = (23.439 - 0.0000004 * d) * DEG2RAD;
  const sb = Math.sin(lat), cb = Math.cos(lat);
  const sl = Math.sin(lon), cl = Math.cos(lon);
  const se = Math.sin(eps), ce = Math.cos(eps);
  out.dec = Math.asin(sb * ce + cb * se * sl);
  out.ra = Math.atan2(sl * cb * ce - sb * se, cl * cb);
  out.lon = lon;
  out.lat = lat;
  return out;
}

/** Greenwich mean sidereal time in degrees, from UT days since J2000. */
function gmstDeg(dUT) {
  return mod(280.46061837 + 360.98564736629 * dUT, 360);
}

/**
 * Bennett's refraction formula (1982), arcminutes, for apparent altitude.
 * At the horizon this lifts a body by ~34' - a bit more than its own diameter,
 * which is why the sun is geometrically already set when you watch it touch
 * the sea. Without this, sunset lands two minutes early and looks wrong.
 */
function refractionRad(altRad) {
  const h = Math.max(altRad * RAD2DEG, -1.2);
  const arcmin = 1.02 / Math.tan((h + 10.3 / (h + 5.11)) * DEG2RAD);
  return clamp(arcmin, 0, 36) * (DEG2RAD / 60);
}

/** Kasten & Young (1989) relative air mass; clamped for below-horizon bodies. */
function airMass(altRad) {
  const h = Math.max(altRad * RAD2DEG, -0.9);
  const denom = Math.sin(h * DEG2RAD) + 0.50572 * Math.pow(h + 6.07995, -1.6364);
  return clamp(1 / Math.max(denom, 1e-4), 1, 42);
}

/** Spectral transmittance through `X` air masses of a `turbidity`-thick sky. */
function extinction(X, turbidity, out) {
  const aer = clamp(turbidity - 1, 0.2, 8) * 0.34;
  out.r = Math.exp(-(TAU_RAYLEIGH.r + TAU_AEROSOL.r * aer) * X - TAU_OZONE.r * Math.min(X, 6));
  out.g = Math.exp(-(TAU_RAYLEIGH.g + TAU_AEROSOL.g * aer) * X - TAU_OZONE.g * Math.min(X, 6));
  out.b = Math.exp(-(TAU_RAYLEIGH.b + TAU_AEROSOL.b * aer) * X - TAU_OZONE.b * Math.min(X, 6));
  return out;
}

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

/** Cheap hash dither. Smooth HDR gradients over 8 bits band badly; a ±1%
 *  multiplicative jitter is below the noise floor of the grain but kills rings. */
const GLSL_DITHER = `
float ditherHash(vec2 fc) {
  return fract(sin(dot(fc, vec2(12.9898, 78.233))) * 43758.5453);
}
`;

/** Shared billboard vertex stage: squashes the quad along the local direction
 *  of the zenith so refraction can flatten a body sitting on the horizon. */
const GLSL_BILLBOARD_VERT = `
uniform vec2 uUpLocal;
uniform float uFlatten;
varying vec2 vP;
void main() {
  vec2 p = position.xy;
  vP = p;
  float a = dot(p, uUpLocal);
  p -= uUpLocal * a * (1.0 - uFlatten);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 0.0, 1.0);
}
`;

const SUN_FRAG = `
uniform vec3 uDiscColor;
uniform vec3 uGlowColor;
uniform float uDiscR;
uniform float uAureole;
uniform float uSpread;
varying vec2 vP;
${GLSL_DITHER}
void main() {
  float r = length(vP);
  float d = r / uDiscR;

  // Photosphere with Eddington limb darkening. Barely visible at noon; at
  // sunset the whole disc is dim enough that the darkened rim reads clearly.
  float mu = sqrt(max(0.0, 1.0 - min(d * d, 1.0)));
  float limb = 1.0 - 0.58 * (1.0 - pow(mu, 0.72));
  float w = max(fwidth(d), 1e-5) * 1.1;
  float disc = (1.0 - smoothstep(1.0 - w, 1.0 + w, d)) * limb;

  // Mie aureole: a steep near-forward lobe plus a wide skirt. Real haze gives
  // roughly 1/theta^1.5 out to a few degrees; haze widens the skirt, not the core.
  float t = max(d, 0.55);
  float lobe = 1.0 / (pow(t, 1.55) + 0.035);
  float skirt = exp(-t * 0.55 / uSpread);
  float glow = uAureole * (0.030 * lobe + 0.85 * skirt * skirt);
  // Window the glow to zero inside the quad's inscribed circle. Without this
  // the aureole is still ~0.02 at the corners and the billboard's square edge
  // becomes a visible seam against a dark sky.
  glow *= 1.0 - smoothstep(0.70, 1.0, r);

  vec3 col = uDiscColor * disc + uGlowColor * glow;
  col *= 1.0 + (ditherHash(gl_FragCoord.xy) - 0.5) * 0.02;
  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

const MOON_FRAG = `
uniform sampler2D uMap;
uniform vec3 uSunLocal;     // sun direction expressed in the billboard basis
uniform vec3 uTint;         // atmospheric transmittance toward the moon
uniform vec3 uEarthshine;
uniform vec3 uHaloColor;
uniform float uDiscR;
uniform float uExposure;
uniform float uLunarL;      // McEwen lunar-Lambert mixing term L(alpha)
uniform float uHalo;
uniform float uRelief;
uniform float uTexel;
varying vec2 vP;
${GLSL_DITHER}

const float HALF_PI = 1.5707963;

void main() {
  vec2 p = vP / uDiscR;
  float r2 = dot(p, p);
  float r = sqrt(r2);
  float w = max(fwidth(r), 1e-5) * 1.1;
  float mask = 1.0 - smoothstep(1.0 - w, 1.0 + w, r);

  // Sampled unconditionally: texture fetches inside non-uniform control flow
  // have undefined derivatives, and the quad is only a few thousand pixels.
  // Orthographic sphere: reconstruct the surface normal from the disc coords.
  float nz = sqrt(max(0.0, 1.0 - min(r2, 1.0)));
  vec3 n = vec3(p, nz);

  // Near-side equirect: longitude -90..90 -> u, latitude +90..-90 -> v.
  // v runs north-down: the map is drawn on a canvas (row 0 = north) and handed
  // to a DataTexture, whose first row is v = 0. Getting this backwards turns
  // the moon upside down, which is subtle enough to ship and wrong every time.
  float lat = asin(clamp(n.y, -1.0, 1.0));
  float lon = atan(n.x, max(n.z, 1e-4));
  vec2 uv = vec2(lon / HALF_PI * 0.5 + 0.5, 0.5 - lat / 3.14159265);

  vec3 albedo = texture2D(uMap, uv).rgb;
  vec3 nn = n;

  #ifdef MOON_RELIEF
    // Height lives in alpha. Slope-perturb the normal so the terminator is
    // chewed up by crater rims instead of being a clean geometric arc.
    float hx = texture2D(uMap, uv + vec2(uTexel, 0.0)).a - texture2D(uMap, uv - vec2(uTexel, 0.0)).a;
    float hy = texture2D(uMap, uv + vec2(0.0, uTexel)).a - texture2D(uMap, uv - vec2(0.0, uTexel)).a;
    float clat = max(cos(lat), 0.15);
    vec3 east  = vec3(cos(lon), 0.0, -sin(lon));
    vec3 north = vec3(-sin(lat) * sin(lon), cos(lat), -sin(lat) * cos(lon));
    // Fade relief at the limb, where the uv derivative explodes and the
    // gradient stops meaning anything.
    float k = uRelief * smoothstep(0.0, 0.35, nz);
    // +hy, not -hy: v decreases as latitude increases (see the uv note above).
    nn = normalize(n - east * (hx * k * clat) + north * (hy * k));
  #endif

  float mu0 = dot(nn, uSunLocal);
  float mu = max(nn.z, 0.02);
  float lit = smoothstep(-0.03, 0.05, mu0) * max(mu0, 0.0);

  // Lunar-Lambert (McEwen 1991): I/F = A[ 2L·mu0/(mu0+mu) + (1-L)·mu0 ].
  // The Lommel-Seeliger half carries no extra cos(view) - that is precisely why
  // a full moon reads as a flat disc lit to the limb rather than a shaded ball.
  float ls = 2.0 * lit / max(mu0 + mu, 0.05);
  float refl = uLunarL * ls + (1.0 - uLunarL) * lit;

  // Earthshine: the dark limb lit by a gibbous Earth, which sits behind us.
  float ashen = mu * (1.0 - smoothstep(0.0, 0.10, mu0));

  vec3 col = albedo * (refl * uExposure * uTint + uEarthshine * ashen) * mask;

  // Aureole around the disc - scattering in the air column, not a lens flare.
  float rq = length(vP);
  float g = uDiscR / max(rq, uDiscR * 0.98);
  float halo = uHalo * (pow(g, 3.2) * 0.55 + pow(g, 1.35) * 0.10);
  halo *= 1.0 - smoothstep(0.70, 1.0, rq);       // never touch the quad edge
  col += uHaloColor * halo * (1.0 - mask * 0.85);

  col *= 1.0 + (ditherHash(gl_FragCoord.xy) - 0.5) * 0.02;
  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

const STAR_VERT = `
uniform float uPixPerRad;
uniform float uIntensity;
uniform float uTime;
uniform float uTwinkle;
uniform float uAngularSize;
attribute float aFlux;
attribute vec3 aColor;
attribute vec2 aTwinkle;
varying vec3 vColor;
varying float vSpike;

void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vec3 dir = normalize(world.xyz - cameraPosition);
  float sinAlt = dir.y;

  float horizon = smoothstep(-0.015, 0.055, sinAlt);
  if (horizon <= 0.0) {
    // Half the catalogue is under the ground at any moment; retire those
    // vertices before they cost a fragment each.
    vColor = vec3(0.0); vSpike = 0.0;
    gl_PointSize = 1.0;
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }

  // Air mass, then Bouguer extinction. This is the single most important line
  // for a believable star field: real skies are bare within 15° of the horizon.
  float X = min(1.0 / max(sinAlt + 0.05, 0.028), 14.0);
  vec3 kmag = vec3(0.13, 0.19, 0.33);
  vec3 trans = exp(-kmag * X * 0.921034);

  // Scintillation. Amplitude grows with path length and with low-level
  // turbulence, and the colour separates as the air acts as a weak prism.
  float ph = aTwinkle.x, rate = aTwinkle.y;
  float tw = sin(uTime * rate + ph) * sin(uTime * rate * 1.61 + ph * 2.7);
  float amp = uTwinkle * clamp((X - 1.0) * 0.30, 0.0, 0.85);
  float chroma = sin(uTime * rate * 0.73 + ph * 1.3) * amp * 0.45;
  vec3 tint = trans * (1.0 + chroma * vec3(0.55, -0.08, -0.62));

  float bright = aFlux * uIntensity * (1.0 + tw * amp) * horizon;

  // Constant angular PSF, but never smaller than ~1.6 px: below that a point
  // sprite aliases into a strobe every time the camera turns. Shrink the
  // energy instead of the footprint - the correct way to band-limit a star.
  float sizePx = uPixPerRad * uAngularSize * (0.62 + 0.95 * pow(max(aFlux, 1e-4), 0.30));
  float shrink = min(1.0, sizePx / 1.62);
  gl_PointSize = clamp(max(sizePx, 1.62), 1.62, 22.0);
  bright *= shrink * shrink;

  vColor = tint * max(bright, 0.0);
  vSpike = smoothstep(1.15, 2.1, aFlux);
  gl_Position = projectionMatrix * viewMatrix * world;
}
`;

const STAR_FRAG = `
varying vec3 vColor;
varying float vSpike;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  float i = exp(-r2 * 5.2) + 0.16 * exp(-r2 * 1.5);
  #ifdef STAR_SPIKES
    if (vSpike > 0.0) {
      float sx = max(0.0, 1.0 - abs(c.x) * 7.0) * exp(-abs(c.y) * 5.0);
      float sy = max(0.0, 1.0 - abs(c.y) * 7.0) * exp(-abs(c.x) * 5.0);
      i += (sx * sx + sy * sy) * 0.30 * vSpike;
    }
  #endif
  gl_FragColor = vec4(vColor * i, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

const MW_VERT = `
varying vec2 vUv;
varying vec3 vWorld;
void main() {
  vUv = uv;
  vec4 world = modelMatrix * vec4(position, 1.0);
  vWorld = normalize(world.xyz - cameraPosition);
  gl_Position = projectionMatrix * viewMatrix * world;
}
`;

const MW_FRAG = `
uniform sampler2D uMap;
uniform float uIntensity;
varying vec2 vUv;
varying vec3 vWorld;
${GLSL_DITHER}
void main() {
  vec3 band = texture2D(uMap, vUv).rgb;
  float X = min(1.0 / max(vWorld.y + 0.05, 0.028), 14.0);
  vec3 trans = exp(-vec3(0.13, 0.19, 0.33) * X * 0.921034);
  float horizon = smoothstep(-0.01, 0.09, vWorld.y);
  vec3 col = band * trans * horizon * uIntensity;
  col *= 1.0 + (ditherHash(gl_FragCoord.xy) - 0.5) * 0.04;
  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

const METEOR_VERT = `
uniform vec4 uMA[${MAX_METEORS}];   // xyz = origin on the unit sphere, w = head arc
uniform vec4 uMB[${MAX_METEORS}];   // xyz = tangent, w = trail arc length
uniform vec4 uMC[${MAX_METEORS}];   // rgb = colour, a = brightness (0 = idle)
uniform float uWidth;
uniform float uRadius;
attribute float aMeteor;
attribute float aAlong;
attribute float aSide;
varying float vAlong;
varying float vSide;
varying vec3 vColor;

void main() {
  int idx = int(aMeteor + 0.5);
  vec4 A = uMA[idx];
  vec4 B = uMB[idx];
  vec4 C = uMC[idx];

  if (C.a <= 0.0) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);  // idle slot: park it off-clip
    vAlong = 0.0; vSide = 0.0; vColor = vec3(0.0);
    return;
  }

  // Great-circle path. The trail is the arc behind the head, clamped so it
  // grows out of the entry point rather than popping in at full length.
  float s = max(A.w - (1.0 - aAlong) * B.w, 0.0);
  float cs = cos(s), sn = sin(s);
  vec3 d = A.xyz * cs + B.xyz * sn;
  vec3 t = -A.xyz * sn + B.xyz * cs;
  vec3 side = normalize(cross(d, t));

  float w = uWidth * (0.22 + 1.7 * pow(aAlong, 6.0));
  vec3 p = normalize(d) + side * (aSide * w);

  vAlong = aAlong;
  vSide = aSide;
  vColor = C.rgb * C.a;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p * uRadius, 1.0);
}
`;

const METEOR_FRAG = `
varying float vAlong;
varying float vSide;
varying vec3 vColor;
void main() {
  float across = max(0.0, 1.0 - abs(vSide));
  float body = across * across * pow(vAlong, 2.4);
  float head = pow(vAlong, 16.0) * pow(across, 0.7) * 3.2;
  gl_FragColor = vec4(vColor * (body + head), 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

// ---------------------------------------------------------------------------

export class Celestial {
  constructor(ctx) {
    this.ctx = ctx;
    this.scene = ctx.scene;
    this.camera = ctx.camera;
    this.renderer = ctx.renderer;
    this.state = ctx.state;

    this.ready = false;
    this.tier = ctx.state.quality.tier || 'high';

    // Sky root follows the camera so the dome is effectively at infinity.
    this.root = new THREE.Group();
    this.root.name = 'Celestial';
    this.root.frustumCulled = false;
    // Equatorial frame: rotates about the celestial pole at the sidereal rate.
    this.skyFrame = new THREE.Group();
    this.skyFrame.name = 'CelestialEquatorial';
    this.skyFrame.matrixAutoUpdate = false;
    this.skyFrame.frustumCulled = false;
    this.root.add(this.skyFrame);

    this._radius = 4000;
    this._pixPerRad = 900;
    this._day = 0;
    this._lastTod = mod(ctx.state.time.timeOfDay, 24);
    this._skyDrivesStars = false;
    this._starFade = 0;
    this._cloudDrift = 0;
    this._sunCloudT = 1;
    this._moonCloudT = 1;
    this._eqMatrix = new THREE.Matrix4();
    this._scaledEq = new THREE.Matrix4();
    this._scaleV = new THREE.Vector3(1, 1, 1);
    this._eclPole = new THREE.Vector3(0, 1, 0);
    this._milkyWay = true;

    this._cloudNoise = createNoise(0x5A17);
    this._rng = makeRNG(20250405);

    this._meteors = [];
    for (let i = 0; i < MAX_METEORS; i++) {
      this._meteors.push({
        alive: false, t: 0, life: 1, arc: 0.5, trail: 0.15, mag: 1,
        ox: 0, oy: 1, oz: 0, tx: 1, ty: 0, tz: 0, r: 1, g: 1, b: 1,
      });
    }
    this._nextMeteor = 12;
  }

  // -------------------------------------------------------------------------
  // Build
  // -------------------------------------------------------------------------

  async init() {
    const quad = new THREE.PlaneGeometry(2, 2);
    this._quadGeo = quad;

    // --- sun -------------------------------------------------------------
    this.sunMat = new THREE.ShaderMaterial({
      uniforms: {
        uDiscColor: { value: new THREE.Color(1, 1, 1) },
        uGlowColor: { value: new THREE.Color(1, 1, 1) },
        uDiscR: { value: SUN_DISC_ANG / SUN_QUAD_ANG },
        uAureole: { value: 0.6 },
        uSpread: { value: 1.0 },
        uUpLocal: { value: new THREE.Vector2(0, 1) },
        uFlatten: { value: 1 },
      },
      vertexShader: GLSL_BILLBOARD_VERT,
      fragmentShader: SUN_FRAG,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
      fog: false,
    });
    this.sunMesh = new THREE.Mesh(quad, this.sunMat);
    this.sunMesh.name = 'Sun';
    this.sunMesh.renderOrder = -2;
    this.sunMesh.matrixAutoUpdate = true;
    this.root.add(this.sunMesh);

    // --- moon ------------------------------------------------------------
    this.moonTexture = this._acquireTexture('celestial:moon', () => this._buildMoonTexture(MOON_TEX_SIZE));
    this.moonMat = new THREE.ShaderMaterial({
      uniforms: {
        uMap: { value: this.moonTexture },
        uSunLocal: { value: new THREE.Vector3(0, 0, 1) },
        uTint: { value: new THREE.Color(1, 1, 1) },
        uEarthshine: { value: new THREE.Color(0, 0, 0) },
        uHaloColor: { value: new THREE.Color(0.62, 0.72, 0.95) },
        uDiscR: { value: MOON_DISC_ANG / MOON_QUAD_ANG },
        uExposure: { value: 3.0 },
        uLunarL: { value: 0.8 },
        uHalo: { value: 0.05 },
        uRelief: { value: 1.4 },
        uTexel: { value: 1 / MOON_TEX_SIZE },
        uUpLocal: { value: new THREE.Vector2(0, 1) },
        uFlatten: { value: 1 },
      },
      vertexShader: GLSL_BILLBOARD_VERT,
      fragmentShader: MOON_FRAG,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
      fog: false,
    });
    this.moonMesh = new THREE.Mesh(quad, this.moonMat);
    this.moonMesh.name = 'Moon';
    this.moonMesh.renderOrder = -3;
    this.root.add(this.moonMesh);

    // --- stars -----------------------------------------------------------
    this._buildStars();

    // --- milky way -------------------------------------------------------
    this.mwTexture = this._acquireTexture('celestial:milkyway', () => this._buildMilkyWayTexture());
    this.mwGeo = new THREE.SphereGeometry(0.995, 48, 24);
    this.mwMat = new THREE.ShaderMaterial({
      uniforms: {
        uMap: { value: this.mwTexture },
        uIntensity: { value: 0 },
      },
      vertexShader: MW_VERT,
      fragmentShader: MW_FRAG,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: true,
      side: THREE.BackSide,
      fog: false,
    });
    this.mwMesh = new THREE.Mesh(this.mwGeo, this.mwMat);
    this.mwMesh.name = 'MilkyWay';
    this.mwMesh.renderOrder = -6;
    this.mwMesh.frustumCulled = false;
    this.mwMesh.matrixAutoUpdate = false;
    this._orientMilkyWay();
    this.skyFrame.add(this.mwMesh);

    // --- meteors ---------------------------------------------------------
    this._buildMeteors();

    this.scene.add(this.root);
    this._applyTier(this.tier);
    this.ready = true;
  }

  link(systems) {
    // God rays track the sun quad; the pass turns itself off when it is hidden.
    try {
      systems?.post?.registerGodRayLight?.(this.sunMesh);
    } catch (err) {
      console.warn('[Celestial] god-ray registration failed:', err);
    }
  }

  /**
   * Route texture creation through the shared factory when it is available and
   * fall back to a local build when it is not. The memo guard means a factory
   * that calls the generator and then throws cannot make us bake twice - these
   * two textures cost tens of milliseconds each.
   */
  _acquireTexture(name, generate) {
    let made = null;
    const gen = () => {
      if (!made) made = generate();
      return made;
    };
    const f = this.ctx.textures;
    if (f && typeof f.get === 'function') {
      try {
        const t = f.get(name, gen);
        if (t && t.isTexture) {
          if (made && made !== t) made.dispose();   // factory had its own cached copy
          return t;                                 // factory owns it; we never dispose it
        }
      } catch (err) {
        console.warn(`[Celestial] TextureFactory rejected "${name}", generating locally:`, err);
      }
    }
    const local = gen();
    if (!this._ownedTextures) this._ownedTextures = [];
    this._ownedTextures.push(local);
    return local;
  }

  // -------------------------------------------------------------------------
  // Star field
  // -------------------------------------------------------------------------

  _buildStars() {
    const rng = makeRNG(0xC0FFEE);
    const N = MAX_STARS;
    const mags = new Float32Array(N);
    const dirs = new Float32Array(N * 3);
    const bv = new Float32Array(N);

    // Cumulative star counts go as ~10^(0.42 m) over the naked-eye range, so
    // sampling the inverse CDF reproduces the real "few bright, many faint"
    // texture. Uniform magnitudes give the classic flat, fake planetarium look.
    const span = STAR_MAG_MAX - STAR_MAG_MIN;
    const norm = Math.pow(10, 0.42 * span) - 1;

    const ngpx = Math.cos(NGP_DEC) * Math.cos(NGP_RA);
    const ngpy = Math.cos(NGP_DEC) * Math.sin(NGP_RA);
    const ngpz = Math.sin(NGP_DEC);

    for (let i = 0; i < N; i++) {
      mags[i] = STAR_MAG_MIN + Math.log10(1 + rng() * norm) / 0.42;

      // Uniform on the sphere, then a bounded rejection pass that concentrates
      // stars toward the galactic plane. Never fibonacciSphere here: its spiral
      // is instantly readable as procedural once you look up for two seconds.
      let x = 0, y = 0, z = 0;
      for (let attempt = 0; attempt < 6; attempt++) {
        const u = rng() * 2 - 1;
        const phi = rng() * TAU;
        const s = Math.sqrt(Math.max(0, 1 - u * u));
        x = s * Math.cos(phi); y = s * Math.sin(phi); z = u;
        // Gaussian in galactic latitude (sigma ~19°), written on sin(b) so the
        // inner loop never pays for an asin; the two agree where it matters.
        const sinB = x * ngpx + y * ngpy + z * ngpz;
        const p = 0.42 + 0.58 * Math.exp(-4.54 * sinB * sinB);
        if (rng() < p) break;
      }
      dirs[i * 3] = x; dirs[i * 3 + 1] = y; dirs[i * 3 + 2] = z;

      // B−V mixture roughly matching the naked-eye population: a fair number of
      // hot blue giants, a solar-type bulk, a red tail.
      const c = rng();
      bv[i] = c < 0.25 ? rng.range(-0.32, 0.30)
        : c < 0.70 ? rng.range(0.30, 0.82)
          : rng.range(0.82, 1.80);
    }

    // Sort brightest-first so a lower tier is a drawRange change, not a rebuild
    // - and so the stars LOW throws away are the ones nobody would miss.
    const sorted = new Array(N);
    for (let i = 0; i < N; i++) sorted[i] = i;
    sorted.sort((a, b) => mags[a] - mags[b]);

    const position = new Float32Array(N * 3);
    const color = new Float32Array(N * 3);
    const flux = new Float32Array(N);
    const twinkle = new Float32Array(N * 2);

    for (let k = 0; k < N; k++) {
      const i = sorted[k];
      position[k * 3] = dirs[i * 3];
      position[k * 3 + 1] = dirs[i * 3 + 1];
      position[k * 3 + 2] = dirs[i * 3 + 2];

      // Ballesteros (2012): B−V -> effective temperature.
      const t = 4600 * (1 / (0.92 * bv[i] + 1.70) + 1 / (0.92 * bv[i] + 0.62));
      kelvinToRGB(t, _rgb);
      const lum = Math.max(0.2126 * _rgb.r + 0.7152 * _rgb.g + 0.0722 * _rgb.b, 1e-3);
      let cr = _rgb.r / lum, cg = _rgb.g / lum, cb = _rgb.b / lum;

      // Scotopic vision is colour-blind: only the brightest stars show a tint.
      const sat = lerp(0.18, 1.0, smoothstep(5.2, 0.4, mags[i]));
      cr = lerp(1, cr, sat); cg = lerp(1, cg, sat); cb = lerp(1, cb, sat);
      color[k * 3] = cr; color[k * 3 + 1] = cg; color[k * 3 + 2] = cb;

      // Pogson flux, then a 0.55 gamma. The true 1600:1 range across the
      // naked-eye scale cannot survive tone mapping; astrophotography makes the
      // same compression, which is why photographed skies look like this.
      flux[k] = Math.pow(Math.pow(10, -0.4 * mags[i]), 0.55);

      twinkle[k * 2] = rng() * TAU;
      twinkle[k * 2 + 1] = rng.range(1.7, 5.4);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(position, 3));
    geo.setAttribute('aColor', new THREE.BufferAttribute(color, 3));
    geo.setAttribute('aFlux', new THREE.BufferAttribute(flux, 1));
    geo.setAttribute('aTwinkle', new THREE.BufferAttribute(twinkle, 2));
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1.01);
    this.starGeo = geo;

    this.starMat = new THREE.ShaderMaterial({
      uniforms: {
        uPixPerRad: { value: 900 },
        uIntensity: { value: 0 },
        uTime: { value: 0 },
        uTwinkle: { value: 0.5 },
        uAngularSize: { value: 0.00235 },
      },
      vertexShader: STAR_VERT,
      fragmentShader: STAR_FRAG,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: true,
      fog: false,
    });

    this.starPoints = new THREE.Points(geo, this.starMat);
    this.starPoints.name = 'Stars';
    this.starPoints.renderOrder = -5;
    this.starPoints.frustumCulled = false;
    this.skyFrame.add(this.starPoints);
  }

  // -------------------------------------------------------------------------
  // Meteors
  // -------------------------------------------------------------------------

  _buildMeteors() {
    const SEG = 10;
    const verts = MAX_METEORS * (SEG + 1) * 2;
    const aMeteor = new Float32Array(verts);
    const aAlong = new Float32Array(verts);
    const aSide = new Float32Array(verts);
    const position = new Float32Array(verts * 3); // unused, keeps three happy
    const indices = [];

    let v = 0;
    for (let m = 0; m < MAX_METEORS; m++) {
      const base = v;
      for (let s = 0; s <= SEG; s++) {
        const a = s / SEG;
        for (let side = 0; side < 2; side++) {
          aMeteor[v] = m;
          aAlong[v] = a;
          aSide[v] = side === 0 ? -1 : 1;
          v++;
        }
      }
      for (let s = 0; s < SEG; s++) {
        const i0 = base + s * 2;
        indices.push(i0, i0 + 1, i0 + 2, i0 + 1, i0 + 3, i0 + 2);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(position, 3));
    geo.setAttribute('aMeteor', new THREE.BufferAttribute(aMeteor, 1));
    geo.setAttribute('aAlong', new THREE.BufferAttribute(aAlong, 1));
    geo.setAttribute('aSide', new THREE.BufferAttribute(aSide, 1));
    geo.setIndex(indices);
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 2);
    this.meteorGeo = geo;

    const uMA = [], uMB = [], uMC = [];
    for (let i = 0; i < MAX_METEORS; i++) {
      uMA.push(new THREE.Vector4(0, 1, 0, 0));
      uMB.push(new THREE.Vector4(1, 0, 0, 0.1));
      uMC.push(new THREE.Vector4(1, 1, 1, 0));
    }
    this.meteorMat = new THREE.ShaderMaterial({
      uniforms: {
        uMA: { value: uMA },
        uMB: { value: uMB },
        uMC: { value: uMC },
        uWidth: { value: 0.0016 },
        uRadius: { value: this._radius },
      },
      vertexShader: METEOR_VERT,
      fragmentShader: METEOR_FRAG,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
      fog: false,
    });

    this.meteorMesh = new THREE.Mesh(geo, this.meteorMat);
    this.meteorMesh.name = 'Meteors';
    this.meteorMesh.renderOrder = -4;
    this.meteorMesh.frustumCulled = false;
    this.root.add(this.meteorMesh);
  }

  // -------------------------------------------------------------------------
  // Procedural textures
  // -------------------------------------------------------------------------

  /**
   * Near-side lunar map. RGB is albedo (authored sRGB), A is a height field the
   * shader differentiates for terminator relief. Mare are placed at their real
   * selenographic coordinates - a randomly-blobbed moon reads as fake instantly
   * because everyone has seen the real one.
   */
  _buildMoonTexture(size) {
    const S = size;
    const albedo = document.createElement('canvas');
    albedo.width = albedo.height = S;
    const a = albedo.getContext('2d');
    const height = document.createElement('canvas');
    height.width = height.height = S;
    const h = height.getContext('2d');

    const rng = makeRNG(0x1969);
    const n = createNoise(0x1969);
    const px = S / 180;                       // pixels per degree, both axes
    const X = (lon) => (0.5 + lon / 180) * S;
    const Y = (lat) => (0.5 - lat / 180) * S;

    a.fillStyle = '#8d8d8a';
    a.fillRect(0, 0, S, S);
    h.fillStyle = '#808080';
    h.fillRect(0, 0, S, S);

    // --- maria -----------------------------------------------------------
    const maria = [
      [18.4, -57.4, 30, 0.62], [32.8, -15.6, 18, 0.68], [28.0, 17.5, 11.5, 0.66],
      [8.5, 31.4, 12.5, 0.64], [17.0, 59.1, 8.0, 0.70], [-7.8, 51.3, 10.0, 0.66],
      [-15.2, 34.6, 6.0, 0.68], [-21.3, -16.6, 9.5, 0.72], [-24.4, -38.6, 6.0, 0.70],
      [56.0, 1.4, 7.0, 0.74], [13.3, 3.6, 4.5, 0.72], [-10.0, -23.0, 5.0, 0.74],
      [7.5, -30.9, 6.5, 0.72], [1.3, 87.5, 6.0, 0.72], [13.3, 86.1, 5.0, 0.74],
      [44.1, -31.5, 4.5, 0.76], [-19.4, -88.0, 7.0, 0.74], [26.0, -3.0, 4.0, 0.76],
    ];
    for (const [lat, lon, rad, dark] of maria) {
      const cx = X(lon), cy = Y(lat);
      const rx = rad * px / Math.max(Math.cos(lat * DEG2RAD), 0.28);
      const ry = rad * px;
      const seed = rng() * 100;
      a.beginPath();
      for (let i = 0; i <= 64; i++) {
        const th = (i / 64) * TAU;
        // Organic outline: a low-frequency radial perturbation, closed exactly.
        const w = 0.74 + 0.34 * n.fbm2D(Math.cos(th) * 1.4 + seed, Math.sin(th) * 1.4 + seed, 3);
        const x = cx + Math.cos(th) * rx * w;
        const y = cy + Math.sin(th) * ry * w;
        if (i === 0) a.moveTo(x, y); else a.lineTo(x, y);
      }
      a.closePath();
      const g = Math.round(141 * dark);
      a.fillStyle = `rgb(${g},${g},${Math.round(g * 1.03)})`;   // maria run faintly blue
      a.filter = `blur(${Math.max(1, rad * px * 0.06)}px)`;
      a.fill();
      a.filter = 'none';

      h.beginPath();
      h.ellipse(cx, cy, rx * 0.92, ry * 0.92, 0, 0, TAU);
      h.filter = `blur(${Math.max(2, rad * px * 0.12)}px)`;
      h.fillStyle = 'rgb(104,104,104)';                          // basins sit low
      h.fill();
      h.filter = 'none';
    }

    // --- craters ---------------------------------------------------------
    // Power-law size distribution; suppressed inside the young mare surfaces.
    // Count scales with area so the density is resolution-independent.
    const CRATERS = Math.round(600 * (S / 256) * (S / 256));
    for (let i = 0; i < CRATERS; i++) {
      const lat = Math.asin(rng() * 2 - 1) * RAD2DEG;
      const lon = rng.range(-90, 90);
      const cx = X(lon), cy = Y(lat);
      const cl = Math.max(Math.cos(lat * DEG2RAD), 0.22);

      let inMare = false;
      for (const [mlat, mlon, mrad] of maria) {
        const dl = (lon - mlon) * cl, db = lat - mlat;
        if (dl * dl + db * db < mrad * mrad * 0.8) { inMare = true; break; }
      }
      if (inMare && rng() < 0.82) continue;

      // Crater diameters follow a steep power law. The floor is one texel - 
      // anything finer is a waste of fill and turns into grey mush under the
      // mip chain. The equirect x-stretch is corrected by cos(lat).
      const u = rng();
      const r = Math.min(Math.max(0.30, 130 / S) * Math.pow(1 - u * 0.999, -1 / 1.9), 4.5) * px;
      const rx = r / cl;

      const rim = 152 + rng() * 34;
      const floor = inMare ? 78 + rng() * 14 : 96 + rng() * 22;
      const gA = a.createRadialGradient(cx, cy, r * 0.05, cx, cy, r);
      gA.addColorStop(0.0, `rgba(${floor},${floor},${floor},0.55)`);
      gA.addColorStop(0.72, `rgba(${floor},${floor},${floor},0.42)`);
      gA.addColorStop(0.88, `rgba(${rim},${rim},${rim},0.50)`);
      gA.addColorStop(1.0, 'rgba(140,140,140,0)');
      a.save();
      a.translate(cx, cy); a.scale(rx / r, 1); a.translate(-cx, -cy);
      a.fillStyle = gA;
      a.beginPath(); a.arc(cx, cy, r, 0, TAU); a.fill();
      a.restore();

      const gH = h.createRadialGradient(cx, cy, r * 0.05, cx, cy, r);
      gH.addColorStop(0.0, 'rgba(46,46,46,0.85)');
      gH.addColorStop(0.70, 'rgba(70,70,70,0.72)');
      gH.addColorStop(0.86, 'rgba(226,226,226,0.85)');
      gH.addColorStop(1.0, 'rgba(128,128,128,0)');
      h.save();
      h.translate(cx, cy); h.scale(rx / r, 1); h.translate(-cx, -cy);
      h.fillStyle = gH;
      h.beginPath(); h.arc(cx, cy, r, 0, TAU); h.fill();
      h.restore();
    }

    // --- ray systems -----------------------------------------------------
    // Tycho and friends throw bright ejecta halfway across the disc; without
    // them a full moon looks flat and dead.
    const rayCraters = [[-43.3, -11.4, 55, 0.34], [9.6, -20.1, 34, 0.26], [8.1, -38.0, 26, 0.20], [23.7, -47.4, 22, 0.30]];
    a.save();
    a.globalCompositeOperation = 'lighter';
    for (const [lat, lon, len, str] of rayCraters) {
      const cx = X(lon), cy = Y(lat);
      const rays = 46;
      for (let i = 0; i < rays; i++) {
        const th = rng() * TAU;
        const L = len * px * rng.range(0.35, 1.0);
        const wdt = px * rng.range(0.5, 2.2);
        const grad = a.createLinearGradient(cx, cy, cx + Math.cos(th) * L, cy + Math.sin(th) * L);
        const A = str * rng.range(0.35, 1.0);
        grad.addColorStop(0, `rgba(255,255,252,${A * 0.55})`);
        grad.addColorStop(0.25, `rgba(255,255,252,${A})`);
        grad.addColorStop(1, 'rgba(255,255,252,0)');
        a.strokeStyle = grad;
        a.lineWidth = wdt;
        a.beginPath();
        a.moveTo(cx, cy);
        a.lineTo(cx + Math.cos(th) * L, cy + Math.sin(th) * L);
        a.stroke();
      }
    }
    a.restore();

    // --- merge + fine grain ---------------------------------------------
    const ia = a.getImageData(0, 0, S, S);
    const ih = h.getImageData(0, 0, S, S);
    const da = ia.data, dh = ih.data;
    const inv = 1 / S;
    for (let y = 0; y < S; y++) {
      const gy = y * inv * 26, fy = y * inv * 120;
      for (let x = 0; x < S; x++) {
        const o = (y * S + x) * 4;
        // Regolith grain at two scales - breaks the smooth gradient look.
        const g = n.fbm2D(x * inv * 26, gy, 3) * 0.055
          + n.noise2D(x * inv * 120, fy) * 0.030;
        const m = 1 + g;
        da[o] = clamp(da[o] * m, 0, 255);
        da[o + 1] = clamp(da[o + 1] * m * 0.996, 0, 255);   // very slight warm bias
        da[o + 2] = clamp(da[o + 2] * m * 0.985, 0, 255);
        da[o + 3] = clamp(dh[o] + g * 210, 0, 255);         // height into alpha
      }
    }

    const tex = new THREE.DataTexture(new Uint8Array(da.buffer), S, S, THREE.RGBAFormat);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    tex.generateMipmaps = true;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    try {
      tex.anisotropy = Math.min(4, this.renderer.capabilities.getMaxAnisotropy());
    } catch (err) { /* anisotropy is a nicety, not a requirement */ }
    return tex;
  }

  /**
   * Milky Way in galactic coordinates: u = l (centre at 0.5), v = b.
   * A warm central bulge, a thinning disc, the Great Rift carved out by dust,
   * and star-cloud mottling. Stored sRGB-encoded so the 8-bit quantisation
   * lands where the signal is, not spread linearly across an unused range.
   */
  _buildMilkyWayTexture() {
    const W = MW_TEX_W, H = MW_TEX_H;
    const data = new Uint8Array(W * H * 4);
    const n = createNoise(0x9A11);

    // Column-major: every term that depends only on galactic longitude is
    // hoisted out of the row loop. That removes eight exp/pow calls per pixel
    // and is most of the difference between a 1.2 s bake and a 0.2 s one.
    for (let x = 0; x < W; x++) {
      const l = ((x + 0.5) / W - 0.5) * 360;
      const al = Math.abs(l);

      // Disc thickness: the bulge is fat, the outer arms are a thin line.
      const sigma = 4.4 + 6.5 * Math.exp(-al / 42) + 1.6 * Math.exp(-Math.pow((al - 80) / 55, 2));
      // The plane is not flat on the sky: it warps by a couple of degrees.
      const warp = 1.5 * Math.sin(l * DEG2RAD) + 0.9 * Math.sin(l * 2 * DEG2RAD + 1.1);
      // Longitudinal profile: bulge, Norma/Scutum arms, the Cygnus star cloud.
      const amp = 0.30 + 1.35 * Math.exp(-al / 38)
        + 0.34 * Math.exp(-Math.pow((al - 32) / 20, 2))
        + 0.30 * Math.exp(-Math.pow((al - 80) / 22, 2));
      // Great Rift: a dust lane hugging the plane, offset slightly south and
      // heaviest between the centre and Cygnus.
      const dustOff = warp - 0.9 + 1.4 * n.noise2D(l * 0.02, 4.7);
      const dsig = 2.0 + 1.8 * Math.exp(-al / 55);
      const dustAmp = 0.35 + 0.65 * Math.exp(-al / 70);
      // Interstellar reddening toward the centre; the outer disc runs cooler.
      const warm = Math.exp(-al / 45);
      const kr = 0.94 + 0.10 * warm, kg = 0.93 + 0.02 * warm, kb = 0.98 - 0.16 * warm;
      const lx = l * 0.030, ld = l * 0.05;

      for (let y = 0; y < H; y++) {
        const b = ((y + 0.5) / H - 0.5) * 180;   // row 0 = b −90 (DataTexture is bottom-up)

        const bb = (b - warp) / sigma;
        let v = Math.exp(-bb * bb * 0.5) * amp;

        // Star clouds.
        v *= 0.55 + 0.62 * (n.fbm2D(lx, b * 0.075, 3) * 0.5 + 0.5);

        const dz = (b - dustOff) / dsig;
        const dust = Math.exp(-dz * dz * 0.5)
          * (0.45 + 0.55 * (n.fbm2D(ld, b * 0.16 + 11, 2) * 0.5 + 0.5)) * dustAmp;
        v *= 1 - 0.78 * clamp01(dust);

        v = clamp01(v * 0.62);
        const o = (y * W + x) * 4;
        data[o] = encodeSRGB(v * kr);
        data[o + 1] = encodeSRGB(v * kg);
        data[o + 2] = encodeSRGB(v * kb);
        data[o + 3] = 255;
      }
    }

    const tex = new THREE.DataTexture(data, W, H, THREE.RGBAFormat);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    tex.generateMipmaps = true;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    return tex;
  }

  /** Orient the band sphere so its own uv grid is galactic (l along u, b along v). */
  _orientMilkyWay() {
    const gc = _v3a.set(
      Math.cos(GC_DEC) * Math.cos(GC_RA),
      Math.cos(GC_DEC) * Math.sin(GC_RA),
      Math.sin(GC_DEC)
    ).normalize();
    const ngp = _v3b.set(
      Math.cos(NGP_DEC) * Math.cos(NGP_RA),
      Math.cos(NGP_DEC) * Math.sin(NGP_RA),
      Math.sin(NGP_DEC)
    ).normalize();
    // Re-orthogonalise: the catalogue pole and centre are not exactly 90° apart.
    gc.addScaledVector(ngp, -gc.dot(ngp)).normalize();
    _v3c.copy(gc).cross(ngp);   // local +Z, keeping the basis right-handed
    // SphereGeometry puts u = 0.5 at local +X and the poles on ±Y, so mapping
    // +X to the galactic centre puts the bulge mid-texture and the seam at the
    // (dim, featureless) anticentre.
    this.mwMesh.matrix.makeBasis(gc, ngp, _v3c);
    this.mwMesh.matrixWorldNeedsUpdate = true;
  }

  // -------------------------------------------------------------------------
  // Frame
  // -------------------------------------------------------------------------

  update(dt, state) {
    if (!this.ready) return;

    // --- in-game date ----------------------------------------------------
    const tod = mod(state.time.timeOfDay, 24);
    if (tod < this._lastTod - 12) this._day++;
    else if (tod > this._lastTod + 12) this._day--;
    this._lastTod = tod;

    const dUT = SIM_EPOCH_DAYS + this._day + (tod - SITE_TZ) / 24;
    const dTT = dUT + DELTA_T_DAYS;
    const lstDeg = gmstDeg(dUT) + SITE_LON_DEG;
    const lst = lstDeg * DEG2RAD;

    solarEquatorial(dTT, _sunEq);
    lunarEquatorial(dTT, _moonEq);

    // --- equatorial -> world ---------------------------------------------
    const cL = Math.cos(lst), sL = Math.sin(lst);
    const cP = Math.cos(SITE_LAT), sP = Math.sin(SITE_LAT);
    const M = this._eqMatrix;
    M.set(
      -sL, cL, 0, 0,
      cP * cL, cP * sL, sP, 0,
      sP * cL, sP * sL, -cP, 0,
      0, 0, 0, 1
    );

    this._toWorld(_sunEq, M, _v3a);
    this._toWorld(_moonEq, M, _v3b);

    // --- geometry --------------------------------------------------------
    const sunGeoAlt = Math.asin(clamp(_v3a.y, -1, 1));
    const moonGeoAlt = Math.asin(clamp(_v3b.y, -1, 1));
    const sunRefr = refractionRad(sunGeoAlt);
    const moonRefr = refractionRad(moonGeoAlt);
    const sunAlt = sunGeoAlt + sunRefr;
    const moonAlt = moonGeoAlt + moonRefr;
    this._lift(_v3a, sunRefr);
    this._lift(_v3b, moonRefr);

    const sunAz = mod(Math.atan2(_v3a.x, -_v3a.z), TAU);
    const moonAz = mod(Math.atan2(_v3b.x, -_v3b.z), TAU);

    // Elongation from the actual 3D directions - the same quantity the moon
    // shader uses for its terminator, so disc and phase can never disagree.
    const cosElong = clamp(_v3a.dot(_v3b), -1, 1);
    const elong = Math.acos(cosElong);
    const illum = (1 - cosElong) * 0.5;
    const phaseAngleDeg = (Math.PI - elong) * RAD2DEG;
    const age = mod(_moonEq.lon - _sunEq.lon, TAU) / TAU;

    // --- atmosphere ------------------------------------------------------
    // Ground fog is aerosol too. Folding it into the effective turbidity is what
    // makes a sun setting into a mist go copper and disappear early, instead of
    // punching through a fog bank at full strength.
    const turbidity = (state.sky?.turbidity ?? 2.4)
      + clamp01(((state.weather?.fogDensity ?? 0) - 0.002) * 200) * 4;
    const sunX = airMass(sunAlt);
    const moonX = airMass(moonAlt);

    this._cloudDrift += dt * (state.clouds?.speed ?? 1) * (state.wind?.strength ?? 3) * 0.0016;
    this._sunCloudT = this._cloudTransmittance(_v3a, state, 0);
    this._moonCloudT = this._cloudTransmittance(_v3b, state, 37.5);

    // `visibility` is the broad day/night crossfade other systems read; it is
    // deliberately soft and reaches 1 only once the body is comfortably up.
    // Direct light needs a much tighter gate - the *extinction* term below is
    // what dims a low sun, and multiplying the two would leave sunset pitch black.
    const sunVis = smoothstep(-0.022, 0.115, sunAlt);
    const moonVis = smoothstep(-0.022, 0.115, moonAlt);
    const sunGate = smoothstep(-0.030, 0.010, sunAlt);
    const moonGate = smoothstep(-0.030, 0.010, moonAlt);

    // --- state.sun -------------------------------------------------------
    const sun = state.sun;
    sun.direction.copy(_v3a);
    sun.elevation = sunAlt;
    sun.azimuth = sunAz;
    sun.visibility = sunVis;

    extinction(sunX, turbidity, _trans);
    const peak = Math.max(_trans.r, _trans.g, _trans.b, 1e-5);
    // Pure transmittance at 40 air masses is a monochromatic red, which lights
    // the world like a darkroom lamp. The floors stand in for the multiply
    // scattered light that always arrives with the direct beam, and keep the
    // key light a deep orange - closer to the truth and far closer to the mood.
    sun.color.setRGB(
      _trans.r / peak,
      Math.max(_trans.g / peak, 0.17),
      Math.max(_trans.b / peak, 0.065)
    );
    const sunLuma = 0.2126 * _trans.r + 0.7152 * _trans.g + 0.0722 * _trans.b;
    // Photographic response, not radiometric: a true 1/40 sunset would render
    // as no sun at all. The hue carries the physics, the curve carries the mood.
    const cloudLight = lerp(1, 0.30, clamp01((state.clouds?.coverage ?? 0) * 0.9)) *
      lerp(1, 0.55, clamp01(state.clouds?.storminess ?? 0));
    sun.intensity = 3.35 * Math.pow(clamp01(sunLuma), 0.45) * sunGate * cloudLight;

    // --- state.moon ------------------------------------------------------
    const moon = state.moon;
    moon.direction.copy(_v3b);
    moon.elevation = moonAlt;
    moon.azimuth = moonAz;
    moon.visibility = moonVis;
    moon.phase = age;

    extinction(moonX, turbidity, _rgb);
    const mPeak = Math.max(_rgb.r, _rgb.g, _rgb.b, 1e-5);
    // Moonlight is warm sunlight in truth; it reads cool because rod vision is
    // blue-shifted (Purkinje). Push toward blue, then let extinction redden it
    // back as the moon sets - which is exactly what a low moon looks like.
    const mr = lerp(_rgb.r / mPeak, 0.62, 0.62);
    const mg = lerp(_rgb.g / mPeak, 0.74, 0.55);
    const mb = lerp(_rgb.b / mPeak, 1.00, 0.50);
    moon.color.setRGB(mr, mg, mb);

    // Lane & Irvine (1973) lunar magnitude vs phase angle: a quarter moon is
    // ~9% of a full moon, not 50%. Shadowing in the regolith does the rest.
    const relBright = Math.pow(10, -0.4 * (0.026 * phaseAngleDeg + 4e-9 * Math.pow(phaseAngleDeg, 4)));
    const moonLuma = 0.2126 * _rgb.r + 0.7152 * _rgb.g + 0.0722 * _rgb.b;
    // The 11:1 full-to-quarter ratio is real but unusable as a light level, so
    // it is gamma-compressed the same way the sun's is. A new moon still dies.
    moon.intensity = 0.28 * Math.pow(relBright, 0.6) * Math.pow(clamp01(moonLuma), 0.4) *
      moonGate * cloudLight * (1 - sunVis * 0.94);

    // --- placement -------------------------------------------------------
    this.camera.getWorldPosition(_v3c);
    this.root.position.copy(_v3c);

    // Sit just inside the far plane so distant hills and the tree line occlude
    // the bodies through the depth buffer, as they must: a sun floating in
    // front of a ridge is the single loudest "this is a skybox" tell there is.
    const R = clamp(this.camera.far * 0.85, 1500, 40000);
    if (Math.abs(R - this._radius) > this._radius * 1e-3) {
      this._radius = R;
      this.meteorMat.uniforms.uRadius.value = R;
    }
    sun.position.copy(_v3a).multiplyScalar(this._radius).add(_v3c);
    moon.position.copy(_v3b).multiplyScalar(this._radius).add(_v3c);

    // Moon's spin axis is ~ the ecliptic pole; the disc's roll tracks it so the
    // maria turn through the night by the true parallactic angle.
    this._eclPole.set(0, ECL_POLE_EQ_Y, ECL_POLE_EQ_Z).applyMatrix4(M).normalize();

    this._scaleV.setScalar(this._radius);
    this._scaledEq.copy(M).scale(this._scaleV);
    this.skyFrame.matrix.copy(this._scaledEq);
    this.skyFrame.matrixWorldNeedsUpdate = true;

    this.renderer.getDrawingBufferSize(_v2);
    const fov = (this.camera.isPerspectiveCamera ? this.camera.fov : 60) * DEG2RAD;
    this._pixPerRad = Math.max(_v2.y, 1) / (2 * Math.tan(fov * 0.5));

    this._updateSun(state, sunAlt, sunGeoAlt, sunX, sunVis, turbidity);
    this._updateMoon(state, moonAlt, moonGeoAlt, illum, phaseAngleDeg, relBright, sunVis, moonVis);
    this._updateStars(dt, state, sunAlt, moonVis, illum);
    this._updateMeteors(dt, state);
  }

  /** Equatorial (ra, dec) -> world direction through the current sky matrix. */
  _toWorld(eq, M, out) {
    const cd = Math.cos(eq.dec);
    out.set(cd * Math.cos(eq.ra), cd * Math.sin(eq.ra), Math.sin(eq.dec));
    out.applyMatrix4(M);
    return out.normalize();
  }

  /** Rotate a direction up by `d` radians in its own vertical plane. */
  _lift(v, d) {
    if (d <= 1e-6) return v;
    const hx = v.x, hz = v.z;
    const hl = Math.sqrt(hx * hx + hz * hz);
    if (hl < 1e-6) return v;
    const alt = Math.asin(clamp(v.y, -1, 1)) + d;
    const c = Math.cos(alt);
    v.set((hx / hl) * c, Math.sin(alt), (hz / hl) * c);
    return v;
  }

  /**
   * Analytic cloud occlusion along a sky direction. The volumetric layer will
   * also draw over these bodies, but that only works when a cloud happens to be
   * rasterised in front; this gives the sun a smooth, wind-advected dimming that
   * the light colour and the god rays can react to as banks drift across it.
   */
  _cloudTransmittance(dir, state, offset) {
    const c = state.clouds;
    if (!c) return 1;
    const cov = clamp01(c.coverage ?? 0);
    if (cov <= 0.002) return 1;

    // Intersect the sky ray with the cloud deck. The clamp stops the sample
    // point running to infinity for a body on the horizon; 0.045 puts the
    // grazing intersection ~20 km out, which is about where the deck really is.
    const alt = c.altitude || 900;
    const t = alt / Math.max(dir.y, 0.045);
    const wd = state.wind?.direction;
    const dx = this._cloudDrift * (wd ? wd.x : 1);
    const dz = this._cloudDrift * (wd ? wd.y : 0);
    const px = (dir.x * t) * 0.00055 - dx + offset;
    const pz = (dir.z * t) * 0.00055 - dz + offset;

    const n = this._cloudNoise.fbm2D(px, pz, 4) * 0.5 + 0.5;
    const thr = 1 - cov * 1.35;
    const cloud = smoothstep(thr, thr + 0.30, n);
    const od = cloud * (c.density ?? 0.6) * (3.6 + 4.0 * (c.storminess ?? 0));
    const T = Math.exp(-od);
    // Never let a clear sky flicker: the effect fades out with coverage.
    return lerp(1, T, clamp01(cov * 1.9));
  }

  _updateSun(state, alt, geoAlt, X, vis, turbidity) {
    const mesh = this.sunMesh;
    const visible = alt > -0.028;
    mesh.visible = visible;
    if (!visible) return;

    const R = this._radius;
    mesh.position.copy(state.sun.direction).multiplyScalar(R);
    this._faceCamera(mesh, state.sun.direction, false);
    const s = R * Math.tan(SUN_QUAD_ANG * 0.5);
    mesh.scale.set(s, s, 1);

    // Differential refraction squashes a setting sun: the lower limb is lifted
    // more than the upper one. It is a small effect and it is the thing that
    // sells a horizon sun.
    const halfAng = SUN_DISC_ANG * 0.5;
    const flat = clamp(
      (refractionRad(geoAlt + halfAng) - refractionRad(geoAlt - halfAng) + 2 * halfAng) / (2 * halfAng),
      0.55, 1
    );
    const u = this.sunMat.uniforms;
    u.uFlatten.value = flat;
    u.uUpLocal.value.set(0, 1);

    extinction(X, turbidity, _trans);
    const cloudT = this._sunCloudT;
    // 260 puts the noon disc far above any sane bloom threshold while keeping
    // the value inside a range the half-float pipeline and the god-ray mask can
    // both handle; sunset falls to an ember purely through extinction (~35).
    // Squaring the cloud term makes thin cloud eat the *disc* faster than the
    // glow, which is exactly how a hazed-over sun behaves.
    const discScale = 260 * cloudT * cloudT;
    u.uDiscColor.value.setRGB(_trans.r * discScale, _trans.g * discScale, _trans.b * discScale);

    // The aureole is the diffuse half of the same light: haze, thin cloud and
    // humidity all widen and brighten it while the disc itself dims. A low sun
    // shines through far more scattering medium, so the halo grows as it sets.
    const haze = clamp01((turbidity - 1.8) / 5) + clamp01(state.clouds?.coverage ?? 0) * 0.55
      + clamp01(state.weather?.wetness ?? 0) * 0.25;
    const diffuse = lerp(1, 3.2, 1 - cloudT);
    const glow = 1.35 * Math.pow(clamp01(cloudT), 0.35) * diffuse;
    u.uAureole.value = glow * (0.45 + 0.9 * haze) * lerp(1.45, 1.0, vis);
    u.uSpread.value = lerp(0.85, 2.6, clamp01(haze * 0.8 + (1 - cloudT) * 0.5));
    const gs = 5.0 * Math.pow(clamp01(0.30 + 0.70 * (state.sun.intensity / 3.35)), 0.8);
    u.uGlowColor.value.copy(state.sun.color).multiplyScalar(gs);
  }

  _updateMoon(state, alt, geoAlt, illum, phaseAngleDeg, relBright, sunVis, moonVis) {
    const mesh = this.moonMesh;
    const visible = alt > -0.028;
    mesh.visible = visible;
    if (!visible) return;

    const R = this._radius;
    const dir = state.moon.direction;
    mesh.position.copy(dir).multiplyScalar(R);
    this._faceCamera(mesh, dir, true);
    const s = R * Math.tan(MOON_QUAD_ANG * 0.5);
    mesh.scale.set(s, s, 1);

    const u = this.moonMat.uniforms;

    // Sun direction in the billboard basis. _faceCamera left the basis in
    // _right/_up/_fwd, so this is three dot products and no matrix inverse.
    const sd = state.sun.direction;
    u.uSunLocal.value.set(sd.dot(_right), sd.dot(_up), sd.dot(_fwd)).normalize();

    // Vertical squash, expressed in the pole-aligned local frame.
    const halfAng = MOON_DISC_ANG * 0.5;
    u.uFlatten.value = clamp(
      (refractionRad(geoAlt + halfAng) - refractionRad(geoAlt - halfAng) + 2 * halfAng) / (2 * halfAng),
      0.55, 1
    );
    _v3c.set(0, 1, 0).addScaledVector(_fwd, -_fwd.y);
    const ux = _v3c.dot(_right), uy = _v3c.dot(_up);
    const ul = Math.hypot(ux, uy);
    if (ul > 1e-4) u.uUpLocal.value.set(ux / ul, uy / ul); else u.uUpLocal.value.set(0, 1);

    const X = airMass(alt);
    extinction(X, state.sky?.turbidity ?? 2.4, _trans);
    const cloudT = this._moonCloudT;
    u.uTint.value.setRGB(_trans.r, _trans.g, _trans.b);

    // A daylight moon is roughly as bright as the sky behind it, so it must
    // read as a ghost, not a lamp. Keep enough for the pale afternoon disc.
    const dayGhost = lerp(1, 0.085, clamp01(sunVis * 1.15));
    // Hapke opposition surge: the regolith hides its own shadows within a few
    // degrees of full, and the moon jumps ~30% brighter than the geometry says.
    const opposition = 1 + 0.35 * (1 - smoothstep(0, 12, phaseAngleDeg));
    u.uExposure.value = 5.5 * opposition * dayGhost * cloudT * moonVis;

    // McEwen's lunar-Lambert mixing term: pure Lommel-Seeliger at opposition,
    // increasingly Lambertian toward the crescents.
    const a = clamp(phaseAngleDeg, 0, 140);
    u.uLunarL.value = clamp(1 - 0.019 * a + 0.000242 * a * a - 1.46e-6 * a * a * a, 0.05, 1);

    // Earthshine: the ashen glow is brightest at the thinnest crescents, when
    // the Earth that lights it is nearly full.
    const es = 0.075 * Math.pow(1 - illum, 1.4) * dayGhost * cloudT * moonVis;
    u.uEarthshine.value.setRGB(es * 0.72, es * 0.86, es * 1.15);

    const haze = clamp01(((state.sky?.turbidity ?? 2.4) - 1.8) / 5) * 0.5
      + clamp01(state.clouds?.coverage ?? 0) * 0.8
      + clamp01((state.weather?.fogDensity ?? 0) * 40);
    u.uHalo.value = 0.16 * relBright * (0.25 + haze) * dayGhost * cloudT * moonVis;
    u.uHaloColor.value.setRGB(_trans.r * 0.72, _trans.g * 0.82, _trans.b * 1.0);
  }

  /**
   * Point a billboard at the camera. `poleAligned` locks the quad's roll to the
   * moon's spin axis so its face rotates through the night by the parallactic
   * angle, the way it actually does; the sun is featureless and uses the zenith.
   */
  _faceCamera(mesh, dir, poleAligned) {
    _fwd.copy(dir).negate();
    if (poleAligned) {
      _v3c.copy(this._eclPole);
    } else {
      _v3c.set(0, 1, 0);
    }
    _right.copy(_v3c).cross(_fwd);
    if (_right.lengthSq() < 1e-8) _right.set(1, 0, 0);   // body at the reference pole
    _right.normalize();
    _up.copy(_fwd).cross(_right).normalize();
    _basis.makeBasis(_right, _up, _fwd);
    mesh.quaternion.setFromRotationMatrix(_basis);
  }

  _updateStars(dt, state, sunAlt, moonVis, illum) {
    // Prefer the atmosphere's authored curve; fall back to a physical one if
    // nothing has ever written it, so the sky is never accidentally starless.
    const ext = state.sky?.starIntensity ?? 0;
    if (!this._skyDrivesStars && ext > 0.001) this._skyDrivesStars = true;

    let target;
    if (this._skyDrivesStars) {
      target = ext;
    } else {
      // Full darkness below −14°, first stars at about −6° (civil dusk ends).
      const night = 1 - smoothstep(-0.245, -0.105, sunAlt);
      // Broken cloud still shows stars through the gaps, so the curve stays
      // near 1 until the deck closes up; an exponential killed them far too early.
      const cov = clamp01(state.clouds?.coverage ?? 0);
      target = night * (1 - smoothstep(0.15, 0.95, cov));
    }
    this._starFade = damp(this._starFade, clamp01(target), 3.2, dt);

    const wash = 1 - 0.34 * moonVis * illum;          // a bright moon kills faint stars
    const intensity = this._starFade * wash * 0.62;
    const on = intensity > 0.002;

    this.starPoints.visible = on;
    this.mwMesh.visible = on && this._milkyWay;

    if (!on) return;
    const su = this.starMat.uniforms;
    su.uIntensity.value = intensity;
    su.uPixPerRad.value = this._pixPerRad;
    su.uTime.value = state.time.elapsed;
    // Scintillation tracks low-level turbulence: still nights have steady stars.
    su.uTwinkle.value = 0.34 + 0.30 * clamp01((state.wind?.strength ?? 3) / 9)
      + 0.18 * clamp01(state.wind?.turbulence ?? 0);
    if (this._milkyWay) this.mwMat.uniforms.uIntensity.value = intensity * 0.115;
  }

  _updateMeteors(dt, state) {
    const active = this._starFade > 0.12;
    this.meteorMesh.visible = active;
    if (!active) {
      for (let i = 0; i < MAX_METEORS; i++) {
        this._meteors[i].alive = false;
        this.meteorMat.uniforms.uMC.value[i].w = 0;
      }
      return;
    }

    const rng = this._rng;
    const clear = 1 - clamp01(state.clouds?.coverage ?? 0);
    this._nextMeteor -= dt * clamp01(clear * 1.2) * this._starFade;
    if (this._nextMeteor <= 0) {
      // Mean ~40 s of dark sky. The real sporadic rate is higher, but the day
      // runs at 50× and a shooting star every fifteen seconds stops being one.
      this._nextMeteor = 15 + rng() * 50;
      this._spawnMeteor(rng);
    }

    // Band-limit the streak: below ~1.7 px wide it aliases into dashes, so pin
    // the width and pay for it in brightness instead.
    const wantWidth = 0.0011;
    const minWidth = 1.7 / Math.max(this._pixPerRad, 1);
    const width = Math.max(wantWidth, minWidth);
    this.meteorMat.uniforms.uWidth.value = width;
    const widthComp = wantWidth / width;

    const uMA = this.meteorMat.uniforms.uMA.value;
    const uMB = this.meteorMat.uniforms.uMB.value;
    const uMC = this.meteorMat.uniforms.uMC.value;

    for (let i = 0; i < MAX_METEORS; i++) {
      const m = this._meteors[i];
      if (!m.alive) { uMC[i].w = 0; continue; }
      m.t += dt;
      const p = m.t / m.life;
      if (p >= 1) { m.alive = false; uMC[i].w = 0; continue; }

      // Meteors flare as they hit denser air, then extinguish; a soft head and
      // a hard cut-out both look wrong, so ramp in fast and decay long.
      const flare = Math.sin(Math.min(p, 1) * Math.PI);
      const b = m.mag * Math.pow(flare, 0.65) * this._starFade * widthComp;

      uMA[i].set(m.ox, m.oy, m.oz, p * m.arc);
      uMB[i].set(m.tx, m.ty, m.tz, m.trail * Math.min(1, p * 3.2));
      uMC[i].set(m.r, m.g, m.b, b);
    }
  }

  _spawnMeteor(rng) {
    let slot = -1;
    for (let i = 0; i < MAX_METEORS; i++) if (!this._meteors[i].alive) { slot = i; break; }
    if (slot < 0) return;
    const m = this._meteors[slot];

    // Entry point: anywhere reasonably above the horizon.
    const alt = rng.range(0.18, 1.45);
    const az = rng() * TAU;
    const ca = Math.cos(alt);
    m.ox = ca * Math.sin(az);
    m.oy = Math.sin(alt);
    m.oz = -ca * Math.cos(az);

    // Tangent: any direction in the plane perpendicular to the entry point,
    // biased downward because that is how a radiant above you reads.
    _v3a.set(m.ox, m.oy, m.oz);
    _v3b.set(rng() * 2 - 1, rng.range(-1.1, 0.25), rng() * 2 - 1);
    _v3b.addScaledVector(_v3a, -_v3b.dot(_v3a));
    if (_v3b.lengthSq() < 1e-6) _v3b.set(1, 0, 0).addScaledVector(_v3a, -_v3a.x);
    _v3b.normalize();
    m.tx = _v3b.x; m.ty = _v3b.y; m.tz = _v3b.z;

    m.arc = rng.range(0.22, 0.95);
    m.trail = rng.range(0.05, 0.16) * (0.6 + m.arc);
    m.life = rng.range(0.42, 1.15);
    m.t = 0;

    // Most sporadics are faint; roughly one in twelve is a proper fireball.
    const fireball = rng() < 0.08;
    m.mag = fireball ? rng.range(2.4, 4.6) : rng.range(0.35, 1.25);
    if (fireball) { m.life *= 1.35; m.arc *= 1.2; m.trail *= 1.5; }

    // Ablation colours: magnesium blue-white, sodium orange, iron yellow-green.
    const c = rng();
    if (c < 0.60) { m.r = 0.86; m.g = 0.92; m.b = 1.00; }
    else if (c < 0.82) { m.r = 1.00; m.g = 0.88; m.b = 0.66; }
    else if (c < 0.94) { m.r = 0.70; m.g = 1.00; m.b = 0.76; }
    else { m.r = 1.00; m.g = 0.72; m.b = 0.52; }

    m.alive = true;
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  // No resize() hook: point sizes and streak widths are derived from the
  // renderer's live drawing-buffer size every frame, so adaptive resolution is
  // handled for free and a stale cached height can never desync them.

  onQualityChange(quality) {
    if (!quality) return;
    this._applyTier(quality.tier || 'high');
  }

  _applyTier(tier) {
    this.tier = tier;
    const count = STAR_COUNT_BY_TIER[tier] ?? STAR_COUNT_BY_TIER.high;
    if (this.starGeo) this.starGeo.setDrawRange(0, Math.min(count, MAX_STARS));

    const spikes = tier === 'high' || tier === 'ultra';
    if (this.starMat) {
      const has = !!this.starMat.defines.STAR_SPIKES;
      if (has !== spikes) {
        if (spikes) this.starMat.defines.STAR_SPIKES = '';
        else delete this.starMat.defines.STAR_SPIKES;
        this.starMat.needsUpdate = true;
      }
      // A denser field wants a slightly tighter PSF or the sky turns to soup.
      this.starMat.uniforms.uAngularSize.value = tier === 'low' ? 0.0027
        : tier === 'ultra' ? 0.00215 : 0.00235;
    }

    // The band is one full-screen additive pass with a texture fetch; that is
    // real money on an integrated GPU, so LOW does without it.
    this._milkyWay = tier !== 'low';
    if (this.mwMesh && !this._milkyWay) this.mwMesh.visible = false;

    const relief = tier !== 'low';
    if (this.moonMat) {
      const has = !!this.moonMat.defines.MOON_RELIEF;
      if (has !== relief) {
        if (relief) this.moonMat.defines.MOON_RELIEF = '';
        else delete this.moonMat.defines.MOON_RELIEF;
        this.moonMat.needsUpdate = true;
      }
    }
  }

  dispose() {
    this.ready = false;
    if (this.root && this.root.parent) this.root.parent.remove(this.root);
    this._quadGeo?.dispose();
    this.starGeo?.dispose();
    this.mwGeo?.dispose();
    this.meteorGeo?.dispose();
    this.sunMat?.dispose();
    this.moonMat?.dispose();
    this.starMat?.dispose();
    this.mwMat?.dispose();
    this.meteorMat?.dispose();
    // Only dispose textures we made ourselves - the factory owns its cache.
    if (this._ownedTextures) for (const t of this._ownedTextures) t.dispose();
    this._ownedTextures = null;
  }
}
