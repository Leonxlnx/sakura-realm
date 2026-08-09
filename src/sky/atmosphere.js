/**
 * atmosphere.js - physically based sky: Rayleigh + Mie + ozone single scattering
 * with a multiple-scattering approximation, plus the CPU-side mirror of the same
 * model that the rest of the world uses for fog and ambient tinting.
 *
 * ARCHITECTURE (this is the part that makes it affordable on a 780M):
 *
 *   transmittance LUT    256x64   RGBA16F - depends only on the medium; rebuilt
 *                                            when the medium parameters change.
 *   multi-scattering LUT  32x32   RGBA16F - Hillaire 2020 second-order + geometric
 *                                            series; also medium-only.
 *   sky-view LUT        192x108   RGBA16F - the whole sky, raymarched. Rebuilt
 *                                            every N frames (N from the quality
 *                                            tier), never per pixel.
 *   fullscreen pass - one bilinear fetch + phase functions.
 *
 * The trick that keeps the low-res sky-view LUT from smearing the solar aureole is
 * storing the Mie in-scatter *without* its phase function in the alpha channel and
 * applying the phase per pixel in the final pass. Rayleigh's phase only varies by
 * 4/3 over the whole sphere, so baking it costs nothing visually; Mie's varies by
 * ~1000x within a few degrees of the sun and absolutely cannot be baked.
 *
 * References: Bruneton & Neyret 2008 (LUT parameterisation), Hillaire 2020
 * "A Scalable and Production Ready Sky and Atmosphere Rendering Technique"
 * (sky-view + multi-scattering LUTs, analytic per-segment integration).
 *
 * Units inside the model are kilometres and "solar irradiance = 1", i.e. radiance
 * comes out around 0.006 for a daytime zenith. SKY_EXPOSURE lifts that into the
 * renderer's linear range; everything the shader outputs is already scaled by it,
 * so getSkyColor() returns values directly comparable to what is on screen.
 */

import * as THREE from 'three';
import { clamp, clamp01, smoothstep, lerp } from '../core/math.js';

// ---------------------------------------------------------------------------
// Medium constants - Earth, in km / km^-1. Hillaire's reference values.
// ---------------------------------------------------------------------------

const R_GROUND = 6360.0;
const R_TOP = 6460.0;
const H_RAYLEIGH = 8.0;
const H_MIE = 1.2;
const OZONE_CENTER = 25.0;
const OZONE_WIDTH = 15.0;

const BETA_R = [0.005802, 0.013558, 0.033100];
const BETA_M_S = 0.003996;
const BETA_M_A = 0.004440;
/** Ozone. Without it twilight goes muddy brown instead of deep blue. */
const BETA_O = [0.000650, 0.001881, 0.000085];

/**
 * Low-saturation field green - this is a pampas-grass world, not a beach. It
 * feeds the ground-bounce term of the multi-scattering LUT, the below-horizon
 * band, and state.sky.groundColor, so it should read as dry grass, not lawn.
 */
const GROUND_ALBEDO = [0.140, 0.150, 0.100];

/**
 * state.sky defaults that correspond to "physically correct Earth". Deriving the
 * scale factors from these means a HUD slider at its default changes nothing.
 */
const REF_RAYLEIGH = 1.6;
const REF_TURBIDITY = 2.4;
const REF_MIE_COEFF = 0.005;

/**
 * Radiance -> render units. Chosen so a clear noon zenith lands near 0.72 linear,
 * which ACES maps to a believable mid blue without clipping. Every other
 * brightness constant in this file is expressed in render units and divided by
 * this, so retuning exposure never desynchronises the night.
 */
const SKY_EXPOSURE = 41.0;

/**
 * The sky-irradiance proxy computed in _publishColors for a clear sky with the
 * sun 60 degrees up. Measured, not guessed - it is what makes ambientIntensity
 * read 1.0 at noon and fall off honestly from there.
 */
const NOON_REFERENCE_LUMINANCE = 1.84;

/**
 * The sun's true radiance is ~15000x the sky's; handing that to a bloom pass on an
 * iGPU produces a white screen, not a sun. We cap it: the disc still blows out to
 * white and still reddens correctly (transmittance is applied on top), but bloom
 * stays controllable.
 */
const SUN_DISC_RENDER_LUMINANCE = 105.0;
/**
 * The disc takes the *hue* of the sun-path transmittance at full strength, but
 * only a softened fraction of its dimming. Applied literally, a sunset disc drops
 * to roughly the brightness of the glow around it and stops reading as a disc; a
 * real camera exposes for it instead. 0.55 keeps it a distinctly brighter ball
 * while still going properly deep orange.
 */
const SUN_DISC_DIM_POWER = 0.55;
/**
 * Apparent solar radius in radians.
 *
 * The true value is 0.004654 (0.2666 deg), and at that size the sun is a
 * four-pixel dot at 1080p - physically right and dramatically worthless. Every
 * film and game oversizes it for the same reason a cinematographer picks a long
 * lens: the disc has to be large enough to read as a light SOURCE and to give
 * bloom something to work with. 2.4x puts it at 0.64 deg radius, which is still
 * inside the range a telephoto shot would give and well short of the cartoon
 * dinner plate that a 4x factor produces.
 */
const SUN_ANGULAR_RADIUS = 0.004654 * 2.4;

/**
 * Night sky, expressed as render-unit luminance so it survives exposure retuning.
 * A moonless sky is not black: airglow (an 87 km emission layer, brightest ~15 deg
 * above the horizon because of the van Rhijn slant factor), plus integrated
 * starlight and zodiacal light.
 */
const AIRGLOW_ZENITH = 0.0055;
const AIRGLOW_TINT = [0.52, 0.95, 0.82]; // faint, 557.7 nm dominated: green-grey
const NIGHT_FLOOR = 0.0055;
/**
 * Blue, not neutral. Below about 0.01 cd/m^2 the eye is scotopic and its peak
 * sensitivity slides from 555 nm to 507 nm - the Purkinje shift, which is why a
 * moonlit night genuinely looks blue to a viewer and grey to a photometer.
 */
const NIGHT_FLOOR_TINT = [0.45, 0.64, 1.0];
/**
 * Isotropic illumination of the aerosol layer at night. This is what keeps a soft
 * band of haze hugging the horizon after dark instead of a hard black cut.
 */
const NIGHT_AMBIENT = 0.018;
const NIGHT_AMBIENT_TINT = [0.45, 0.64, 1.0];
/**
 * How far below the horizon the stylised night terms take over. This has to
 * finish while real twilight still dominates (-4 degrees), otherwise the floor
 * fades in after the natural sky has already collapsed and the zenith visibly
 * brightens again on the way into night. Verified monotonic from +8 to -18.
 */
const NIGHT_RAMP_HI = 0.035;
const NIGHT_RAMP_LO = -0.07;
/**
 * Full-moon illuminance as a fraction of the sun's. Physically 2.5e-6, which is
 * correct and unwatchable without eye adaptation, so roughly four stops of
 * day-for-night are compressed into it. A full moon then lifts the sky several
 * times above a moonless one instead of leaving it black.
 */
const MOON_SKY_SCALE = 0.040;

// ---------------------------------------------------------------------------
// Quality tiers
// ---------------------------------------------------------------------------

const TIERS = {
  low:    { view: [96, 56],   steps: 14, ms: 16, msDirs: 16, msSteps: 12, env: 16, interval: 6, cpuRows: 2 },
  medium: { view: [128, 72],  steps: 20, ms: 24, msDirs: 24, msSteps: 14, env: 24, interval: 4, cpuRows: 3 },
  high:   { view: [192, 108], steps: 30, ms: 32, msDirs: 32, msSteps: 18, env: 32, interval: 3, cpuRows: 4 },
  ultra:  { view: [256, 144], steps: 44, ms: 32, msDirs: 48, msSteps: 24, env: 48, interval: 2, cpuRows: 6 },
};

const TR_W = 256;
const TR_H = 64;

// CPU mirror resolutions - small, because it is rebuilt on the main thread.
const CPU_TR_W = 48;
const CPU_TR_H = 12;
const CPU_MS_W = 12;
const CPU_MS_H = 5;
const CPU_AZ = 20;
const CPU_ZE = 22;
const CPU_STEPS = 10;

const INV_TAU = 1.0 / (Math.PI * 2);
const INV_PI = 1.0 / Math.PI;

// ---------------------------------------------------------------------------
// GLSL
// ---------------------------------------------------------------------------

const GLSL_ATMOS = /* glsl */ `
#define PI 3.141592653589793
#define ISO_PHASE 0.07957747154

uniform sampler2D uTransmittanceLUT;
uniform sampler2D uMultiScatterLUT;
uniform vec3 uRayleighS;
uniform vec3 uOzoneA;
uniform vec2 uMieSA;      // x = scattering, y = absorption
uniform vec2 uScaleH;     // x = rayleigh, y = mie
uniform vec3 uGroundAlbedo;
uniform vec2 uTrSize;
uniform vec2 uMsSize;

const float R_GROUND = ${R_GROUND.toFixed(1)};
const float R_TOP = ${R_TOP.toFixed(1)};

// Texel-centre <-> unit-range mapping. Without it the grazing-angle end of the
// transmittance LUT is biased by half a texel, and that is exactly where sunset
// colour lives.
float unitToUv(float x, float n) { return 0.5 / n + x * (1.0 - 1.0 / n); }
float uvToUnit(float u, float n) { return (u - 0.5 / n) / (1.0 - 1.0 / n); }

void mediumAt(float h, out vec3 sR, out float sM, out vec3 ext) {
  float hc = max(h, 0.0);
  float dR = exp(-hc / uScaleH.x);
  float dM = exp(-hc / uScaleH.y);
  float dO = max(0.0, 1.0 - abs(hc - ${OZONE_CENTER.toFixed(1)}) / ${OZONE_WIDTH.toFixed(1)});
  sR = uRayleighS * dR;
  sM = uMieSA.x * dM;
  ext = sR + vec3(sM + uMieSA.y * dM) + uOzoneA * dO;
}

// Nearest positive intersection with a sphere centred on the origin, or -1.
float raySphere(vec3 ro, vec3 rd, float rad) {
  float b = dot(ro, rd);
  float c = dot(ro, ro) - rad * rad;
  float d = b * b - c;
  if (d < 0.0) return -1.0;
  d = sqrt(d);
  float t1 = -b + d;
  if (t1 < 0.0) return -1.0;
  float t0 = -b - d;
  return t0 < 0.0 ? t1 : t0;
}

vec2 trUv(float r, float mu) {
  float H = sqrt(max(0.0, R_TOP * R_TOP - R_GROUND * R_GROUND));
  float rho = sqrt(max(0.0, r * r - R_GROUND * R_GROUND));
  float disc = r * r * (mu * mu - 1.0) + R_TOP * R_TOP;
  float d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
  float dMin = R_TOP - r;
  float dMax = rho + H;
  float xMu = dMax > dMin ? (d - dMin) / (dMax - dMin) : 0.0;
  return vec2(unitToUv(clamp(xMu, 0.0, 1.0), uTrSize.x),
              unitToUv(clamp(rho / H, 0.0, 1.0), uTrSize.y));
}

vec3 transmittanceTo(float r, float mu) {
  return texture2D(uTransmittanceLUT, trUv(clamp(r, R_GROUND, R_TOP), clamp(mu, -1.0, 1.0))).rgb;
}

vec3 multiScatterAt(float r, float mu) {
  vec2 uv = vec2(unitToUv(clamp(mu * 0.5 + 0.5, 0.0, 1.0), uMsSize.x),
                 unitToUv(clamp((r - R_GROUND) / (R_TOP - R_GROUND), 0.0, 1.0), uMsSize.y));
  return texture2D(uMultiScatterLUT, uv).rgb;
}

float phaseRayleigh(float c) { return 0.05968310366 * (1.0 + c * c); }

// Cornette-Shanks: a Henyey-Greenstein lobe carrying the correct (1 + cos^2)
// backscatter, which plain HG gets wrong.
float phaseMie(float c, float g) {
  float g2 = g * g;
  float d = max(1.0 + g2 - 2.0 * g * c, 1e-5);
  return (3.0 * (1.0 - g2) * (1.0 + c * c)) / (8.0 * PI * (2.0 + g2) * d * sqrt(d));
}

// Sky-view LUT zenith warp. Half the rows land within a few degrees of the
// horizon, where all the interesting gradient is, while the zenith still gets
// ~3 degrees per row - warping cos(theta) instead of theta would have left 15
// degree rows up there and a visible kink in the upper sky.
float zenithToV(float cosTheta) {
  float n = 1.0 - 2.0 * acos(clamp(cosTheta, -1.0, 1.0)) / PI;
  return 0.5 + 0.5 * sign(n) * sqrt(abs(n));
}
float vToZenithAngle(float v) {
  float s = v * 2.0 - 1.0;
  float n = sign(s) * s * s;
  return (1.0 - n) * (PI * 0.5);
}
`;

/** Full-coverage quad; the geometry is a PlaneGeometry(2,2) read in clip space. */
const VERT_QUAD = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

const FRAG_TRANSMITTANCE = /* glsl */ `
${GLSL_ATMOS}
varying vec2 vUv;
void main() {
  float xMu = clamp(uvToUnit(vUv.x, uTrSize.x), 0.0, 1.0);
  float xR  = clamp(uvToUnit(vUv.y, uTrSize.y), 0.0, 1.0);
  float H = sqrt(R_TOP * R_TOP - R_GROUND * R_GROUND);
  float rho = H * xR;
  float r = sqrt(rho * rho + R_GROUND * R_GROUND);
  float dMin = R_TOP - r;
  float dMax = rho + H;
  float d = dMin + xMu * (dMax - dMin);
  float mu = d <= 0.0 ? 1.0 : clamp((H * H - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
  float sx = sqrt(max(0.0, 1.0 - mu * mu));
  vec3 od = vec3(0.0);
  const float N = 40.0;
  for (int i = 0; i < 40; i++) {
    float t = (float(i) + 0.5) / N * d;
    vec2 p = vec2(sx * t, r + mu * t);
    vec3 sR; float sM; vec3 ext;
    mediumAt(length(p) - R_GROUND, sR, sM, ext);
    od += ext;
  }
  gl_FragColor = vec4(exp(-od * (d / N)), 1.0);
}
`;

const FRAG_MULTISCATTER = /* glsl */ `
${GLSL_ATMOS}
varying vec2 vUv;

vec3 fibonacciDir(int i) {
  float fi = float(i) + 0.5;
  float y = 1.0 - 2.0 * fi / float(MS_DIRS);
  float rr = sqrt(max(0.0, 1.0 - y * y));
  float th = 2.39996322972865332 * fi;
  return vec3(cos(th) * rr, y, sin(th) * rr);
}

void main() {
  float cosSun = clamp(uvToUnit(vUv.x, uMsSize.x), 0.0, 1.0) * 2.0 - 1.0;
  float alt = clamp(uvToUnit(vUv.y, uMsSize.y), 0.0, 1.0) * (R_TOP - R_GROUND);
  float r = clamp(R_GROUND + alt, R_GROUND + 0.002, R_TOP - 0.002);
  vec3 sunDir = vec3(sqrt(max(0.0, 1.0 - cosSun * cosSun)), cosSun, 0.0);
  vec3 ro = vec3(0.0, r, 0.0);

  vec3 lumSum = vec3(0.0);
  vec3 fmsSum = vec3(0.0);

  for (int dIdx = 0; dIdx < MS_DIRS; dIdx++) {
    vec3 rd = fibonacciDir(dIdx);
    float tBot = raySphere(ro, rd, R_GROUND);
    float tTop = raySphere(ro, rd, R_TOP);
    float tMax = tBot > 0.0 ? tBot : tTop;
    if (tMax <= 0.0) continue;

    vec3 L = vec3(0.0);
    vec3 fms = vec3(0.0);
    vec3 tr = vec3(1.0);
    float dt = tMax / float(MS_STEPS);

    for (int i = 0; i < MS_STEPS; i++) {
      vec3 p = ro + rd * ((float(i) + 0.3) * dt);
      float rr = length(p);
      vec3 sR; float sM; vec3 ext;
      mediumAt(rr - R_GROUND, sR, sM, ext);
      vec3 sampleT = exp(-ext * dt);
      vec3 safeE = max(ext, vec3(1e-9));
      vec3 scat = sR + vec3(sM);
      float mu = dot(p / rr, sunDir);
      float shadow = raySphere(p, sunDir, R_GROUND) >= 0.0 ? 0.0 : 1.0;
      // Analytic solution of the segment with constant coefficients - far more
      // accurate than a rectangle rule, which is why 12-24 steps is enough.
      vec3 S = transmittanceTo(rr, mu) * shadow * scat * ISO_PHASE;
      L += tr * (S - S * sampleT) / safeE;
      fms += tr * (scat - scat * sampleT) / safeE;
      tr *= sampleT;
    }

    if (tBot > 0.0) {
      vec3 n = normalize(ro + rd * tBot);
      float ndl = max(dot(n, sunDir), 0.0);
      L += tr * transmittanceTo(R_GROUND, dot(n, sunDir)) * ndl * uGroundAlbedo / PI;
    }

    lumSum += L;
    fmsSum += fms;
  }

  lumSum /= float(MS_DIRS);
  fmsSum /= float(MS_DIRS);

  // Geometric series over every further scattering order (Hillaire eq. 10).
  vec3 psi = lumSum / (1.0 - min(fmsSum, vec3(0.92)));
  gl_FragColor = vec4(psi, 1.0);
}
`;

const FRAG_SKYVIEW = /* glsl */ `
${GLSL_ATMOS}
varying vec2 vUv;

uniform vec3 uSunDir;
uniform vec3 uMoonDir;
uniform vec3 uMoonLight;      // moon colour * illuminance, sun-relative
uniform vec3 uNightAmbient;
uniform vec3 uAirglow;
uniform vec3 uNightFloor;
uniform float uMieG;
uniform float uViewR;
uniform float uExposure;
uniform vec2 uSkyViewSize;

void main() {
  // Azimuth wraps (RepeatWrapping); zenith uses the warped angle parameterisation.
  float az = (vUv.x - 0.5) * 2.0 * PI;
  float theta = vToZenithAngle(clamp(uvToUnit(vUv.y, uSkyViewSize.y), 0.0, 1.0));
  float y = cos(theta);
  float sh = sin(theta);
  vec3 rd = vec3(cos(az) * sh, y, sin(az) * sh);
  vec3 ro = vec3(0.0, uViewR, 0.0);

  float tBot = raySphere(ro, rd, R_GROUND);
  float tTop = raySphere(ro, rd, R_TOP);
  float tMax = tBot > 0.0 ? tBot : tTop;
  if (tMax <= 0.0) { gl_FragColor = vec4(0.0); return; }

  float cS = dot(rd, uSunDir);
  float pRs = phaseRayleigh(cS);
  float cM = dot(rd, uMoonDir);
  float pRm = phaseRayleigh(cM);
  // Deliberately broad moon lobe: the LUT cannot resolve a tight one without
  // texel diamonds, so the sharp core of the aureole is added analytically in the
  // final pass instead.
  float pMm = phaseMie(cM, uMieG * 0.6);
  bool moonUp = uMoonLight.g > 0.0;

  vec3 L = vec3(0.0);
  vec3 M = vec3(0.0);
  vec3 tr = vec3(1.0);

  for (int i = 0; i < SKY_STEPS; i++) {
    // Quadratic step distribution: dense near the camera where the aerosol layer
    // lives, sparse out at the top of the atmosphere where nothing happens.
    float f0 = float(i) / float(SKY_STEPS);
    float f1 = float(i + 1) / float(SKY_STEPS);
    float tA = f0 * f0 * tMax;
    float dt = f1 * f1 * tMax - tA;
    vec3 p = ro + rd * (tA + dt * 0.4);
    float rr = length(p);
    vec3 up = p / rr;

    vec3 sR; float sM; vec3 ext;
    mediumAt(rr - R_GROUND, sR, sM, ext);
    vec3 sampleT = exp(-ext * dt);
    vec3 safeE = max(ext, vec3(1e-9));
    vec3 scat = sR + vec3(sM);

    float muS = dot(up, uSunDir);
    float shS = raySphere(p, uSunDir, R_GROUND) >= 0.0 ? 0.0 : 1.0;
    vec3 Tsun = transmittanceTo(rr, muS) * shS;
    vec3 Sr = Tsun * sR * pRs + multiScatterAt(rr, muS) * scat;
    vec3 Sm = Tsun * vec3(sM);

    if (moonUp) {
      float muM = dot(up, uMoonDir);
      float shM = raySphere(p, uMoonDir, R_GROUND) >= 0.0 ? 0.0 : 1.0;
      vec3 Tm = transmittanceTo(rr, muM) * shM;
      Sr += uMoonLight * (Tm * (sR * pRm + vec3(sM) * pMm) + multiScatterAt(rr, muM) * scat);
    }

    Sr += uNightAmbient * scat;

    L += tr * (Sr - Sr * sampleT) / safeE;
    M += tr * (Sm - Sm * sampleT) / safeE;
    tr *= sampleT;
  }

  if (tBot > 0.0) {
    vec3 n = normalize(ro + rd * tBot);
    float ndl = max(dot(n, uSunDir), 0.0);
    L += tr * transmittanceTo(R_GROUND, dot(n, uSunDir)) * ndl * uGroundAlbedo / PI;
    if (moonUp) {
      float ndlM = max(dot(n, uMoonDir), 0.0);
      L += tr * uMoonLight * transmittanceTo(R_GROUND, dot(n, uMoonDir)) * ndlM * uGroundAlbedo / PI;
    }
  } else {
    // van Rhijn slant factor for an emissive shell 87 km up: 6.1x at the horizon.
    float vr = inversesqrt(max(1.0 - 0.973169 * sh * sh, 0.0035));
    L += tr * (uAirglow * vr + uNightFloor);
  }

  gl_FragColor = vec4(L * uExposure, M.g * uExposure);
}
`;

const VERT_SKY = /* glsl */ `
uniform mat4 uInvProjection;
uniform mat3 uCameraRot;
varying vec3 vRay;
void main() {
  // The view ray is an affine function of clip xy, so interpolating it
  // unnormalised and normalising per fragment is exact - and it inherits any
  // jitter baked into the projection matrix for free.
  vec4 ray = uInvProjection * vec4(position.xy, 1.0, 1.0);
  vRay = uCameraRot * ray.xyz;
  // z = w puts the sky exactly on the far plane, so LEQUAL depth testing lets the
  // terrain reject sky fragments before they are ever shaded.
  gl_Position = vec4(position.xy, 1.0, 1.0);
}
`;

const FRAG_SKY = /* glsl */ `
#define PI 3.141592653589793
uniform sampler2D uSkyViewLUT;
uniform vec2 uSkyViewSize;
uniform vec3 uSunDir;
uniform vec3 uMoonDir;
uniform vec3 uMieTint;
uniform vec3 uSunDiscColor;
uniform vec3 uMoonGlow;
uniform vec3 uLightningTint;
uniform float uMieG;
uniform float uAureoleG;
uniform float uAureoleMix;
uniform float uMieColumn;
uniform float uSunAngRad;
uniform float uDither;
varying vec3 vRay;

float phaseMie(float c, float g) {
  float g2 = g * g;
  float d = max(1.0 + g2 - 2.0 * g * c, 1e-5);
  return (3.0 * (1.0 - g2) * (1.0 + c * c)) / (8.0 * PI * (2.0 + g2) * d * sqrt(d));
}

// Hash without sine - stable at large gl_FragCoord and free of the mediump
// artefacts the classic sin(dot(..)) hash shows on some drivers.
float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

void main() {
  vec3 dir = normalize(vRay);

  // The epsilon only matters looking exactly along +-Y, where atan(0,0) is
  // undefined and the LUT is azimuth-invariant anyway - but an undefined there is
  // a NaN pixel at the zenith on some drivers.
  float u = atan(dir.z, dir.x + 1e-9) * 0.1591549431 + 0.5;
  float n = 1.0 - 2.0 * acos(clamp(dir.y, -1.0, 1.0)) / PI;
  float vUnit = clamp(0.5 + 0.5 * sign(n) * sqrt(abs(n)), 0.0, 1.0);
  float v = 0.5 / uSkyViewSize.y + vUnit * (1.0 - 1.0 / uSkyViewSize.y);
  vec4 sky = texture2D(uSkyViewLUT, vec2(u, v));

  float cosSun = dot(dir, uSunDir);
  // Two-lobe aerosol phase: the wide lobe is general haze, the narrow one is the
  // solar aureole. Both ride on the LUT's phase-less Mie integral, which already
  // thickens towards the horizon, so the glow grows down there on its own.
  float pm = mix(phaseMie(cosSun, uMieG), phaseMie(cosSun, uAureoleG), uAureoleMix);
  vec3 col = sky.rgb + sky.a * uMieTint * pm;

  // Aerosol slant path for this ray: 8.3x at the horizon, 0.9x at the zenith.
  float slant = 1.0 / (max(dir.y, 0.0) + 0.12);

  if (uMoonGlow.g > 0.0) {
    col += uMoonGlow * (uMieColumn * slant) * phaseMie(dot(dir, uMoonDir), 0.86);
  }

  // A strike lights the whole aerosol column, brightest along the horizon.
  col += uLightningTint * (slant * 0.12);

  #ifdef SUN_DISC
  // At this angular scale the chord equals the angle to 1e-6 relative, so we get
  // the angular distance without an acos.
  float ang = length(dir - uSunDir);
  float aa = max(fwidth(ang), 1e-6) * 0.5;   // ~1 pixel of edge, no softer
  float disc = 1.0 - smoothstep(uSunAngRad - aa, uSunAngRad + aa, ang);
  if (disc > 0.0) {
    float x = clamp(ang / uSunAngRad, 0.0, 1.0);
    float mu = max(sqrt(max(1.0 - x * x, 0.0)), 1e-3);
    // Wavelength dependent limb darkening (Hestroffer & Magnan): the limb is not
    // merely dimmer, it is redder.
    vec3 limb = pow(vec3(mu), vec3(0.397, 0.503, 0.652));
    col += disc * uSunDiscColor * limb;
  }
  #endif

  #ifdef DITHER
  // Triangular-PDF multiplicative dither. Multiplicative because one 8-bit step
  // is a roughly constant *relative* increment after tone mapping, so a fixed
  // relative amplitude breaks banding evenly from horizon to zenith. Static, not
  // temporal: a still sky must not crawl.
  float d = hash12(gl_FragCoord.xy) + hash12(gl_FragCoord.xy + 17.31) - 1.0;
  col *= 1.0 + d * uDither;
  #endif

  gl_FragColor = vec4(col, 1.0);

  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

// ---------------------------------------------------------------------------
// CPU mirror of the model. All scratch is module scope: nothing here allocates.
// ---------------------------------------------------------------------------

const _med = new Float64Array(7);   // sR(0..2), sM(3), ext(4..6)
const _tr3 = new Float64Array(3);
const _tr3b = new Float64Array(3);
const _ms3 = new Float64Array(3);
const _ms3b = new Float64Array(3);
const _acc = new Float64Array(4);
const _H_ATMOS = Math.sqrt(R_TOP * R_TOP - R_GROUND * R_GROUND);

function cpuMedium(P, h) {
  const hc = h > 0 ? h : 0;
  const dR = Math.exp(-hc / H_RAYLEIGH);
  const dM = Math.exp(-hc / H_MIE);
  let dO = 1 - Math.abs(hc - OZONE_CENTER) / OZONE_WIDTH;
  if (dO < 0) dO = 0;
  const ma = P.mieA * dM;
  _med[0] = P.betaR[0] * dR;
  _med[1] = P.betaR[1] * dR;
  _med[2] = P.betaR[2] * dR;
  _med[3] = P.mieS * dM;
  _med[4] = _med[0] + _med[3] + ma + BETA_O[0] * dO;
  _med[5] = _med[1] + _med[3] + ma + BETA_O[1] * dO;
  _med[6] = _med[2] + _med[3] + ma + BETA_O[2] * dO;
}

/** Nearest positive hit with a sphere at the origin, or -1. */
function raySphere3(ox, oy, oz, dx, dy, dz, rad) {
  const b = ox * dx + oy * dy + oz * dz;
  const c = ox * ox + oy * oy + oz * oz - rad * rad;
  let d = b * b - c;
  if (d < 0) return -1;
  d = Math.sqrt(d);
  const t1 = -b + d;
  if (t1 < 0) return -1;
  const t0 = -b - d;
  return t0 < 0 ? t1 : t0;
}

const phaseRayleighCPU = (c) => 0.05968310366 * (1 + c * c);

function phaseMieCPU(c, g) {
  const g2 = g * g;
  const d = Math.max(1 + g2 - 2 * g * c, 1e-5);
  return (3 * (1 - g2) * (1 + c * c)) / (8 * Math.PI * (2 + g2) * d * Math.sqrt(d));
}

// ---------------------------------------------------------------------------

const _ringDirs = [];
for (let i = 0; i < 8; i++) {
  const a = (i / 8) * Math.PI * 2;
  // A hair above the horizon: the band that fog and distant terrain read from.
  _ringDirs.push(new THREE.Vector3(Math.cos(a) * 0.9986, 0.0523, Math.sin(a) * 0.9986));
}
const _downDirs = [
  new THREE.Vector3(0, -1, 0),
  new THREE.Vector3(0.7, -0.714, 0),
  new THREE.Vector3(-0.7, -0.714, 0),
  new THREE.Vector3(0, -0.714, 0.7),
  new THREE.Vector3(0, -0.714, -0.7),
];
const _upDir = new THREE.Vector3(0, 1, 0);

export class Atmosphere {
  constructor(ctx) {
    this.ctx = ctx;
    this.state = ctx.state;
    this.scene = ctx.scene || null;
    this.renderer = ctx.renderer || null;
    this.camera = ctx.camera || null;

    /** @type {THREE.Texture|null} Cube env map for IBL - see contract section 6. */
    this.envTexture = null;

    this._ready = false;
    this._ownsSceneEnv = false;
    this._tier = TIERS[this.state?.quality?.tier] || TIERS.high;

    // Medium parameters, shared by the GPU and CPU paths.
    this._p = {
      betaR: [BETA_R[0], BETA_R[1], BETA_R[2]],
      mieS: BETA_M_S,
      mieA: BETA_M_A,
      rayleighScale: 1,
      mieScale: 1,
    };
    this._qRayleigh = -1e9;
    this._qMie = -1e9;
    this._mediumDirty = true;
    this._lastMediumFrame = -1e9;

    // Light state the shaders consume.
    this._sunDir = new THREE.Vector3(0, 1, 0);
    this._moonDir = new THREE.Vector3(0, -1, 0);
    this._moonLight = new THREE.Vector3();
    this._moonAmount = 0;
    this._viewR = R_GROUND;
    this._mieG = 0.8;
    this._aureoleG = 0.96;
    this._aureoleMix = 0.08;
    this._exposure = SKY_EXPOSURE;
    this._mieTint = new THREE.Vector3(1, 1, 1);
    this._sunT = new Float64Array([1, 1, 1]);

    // Sky-view LUT refresh bookkeeping.
    this._skyViewDirty = true;
    this._framesSinceView = 1e6;
    this._lastViewSun = new THREE.Vector3(0, -2, 0);
    this._lastViewMoon = -1;
    this._lastViewR = -1;

    // Env map / CPU table refresh bookkeeping. Both caches have to track the
    // moon as well as the sun, or a moonrise never reaches fog or IBL.
    this._envDirty = true;
    this._lastEnvSun = new THREE.Vector3(0, -2, 0);
    this._lastEnvMoonDir = new THREE.Vector3(0, -2, 0);
    this._lastEnvMoonAmt = -1;
    this._envCooldown = 0;
    this._cpuDirty = true;
    this._cpuRow = CPU_ZE;
    this._lastCpuSun = new THREE.Vector3(0, -2, 0);
    this._lastCpuMoonDir = new THREE.Vector3(0, -2, 0);
    this._lastCpuMoonAmt = -1;

    // CPU tables.
    this._cpuTr = new Float32Array(CPU_TR_W * CPU_TR_H * 3);
    this._cpuMs = new Float32Array(CPU_MS_W * CPU_MS_H * 3);
    this._cpuSky = new Float32Array(CPU_AZ * CPU_ZE * 4);
    this._cpuSunDir = new THREE.Vector3(0, 1, 0);

    this._scratchColor = new THREE.Color();
    this._colorsSeeded = false;

    // Seed the CPU model now: other systems may call getSkyColor() from their own
    // constructors, long before any GPU work has happened.
    this._syncMedium(true);
    this._sunDir.copy(this.state.sun.direction).normalize();
    this._cpuSunDir.copy(this._sunDir);
    this._buildCpuTransmittance();
    this._buildCpuMultiScatter();
    this._updateMieTint();
    this._rebuildCpuSky(0, CPU_ZE);
    this._cpuDirty = false;
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  async init() {
    if (!this.renderer) return;
    const tier = this._tier;

    this._geo = new THREE.PlaneGeometry(2, 2);
    this._passScene = new THREE.Scene();
    this._passScene.matrixWorldAutoUpdate = false;
    this._passCamera = new THREE.Camera();
    this._quad = new THREE.Mesh(this._geo, null); // material is swapped per pass
    this._quad.frustumCulled = false;
    this._quad.matrixAutoUpdate = false;
    this._passScene.add(this._quad);

    this._trRT = this._makeRT(TR_W, TR_H, THREE.ClampToEdgeWrapping);
    this._msRT = this._makeRT(tier.ms, tier.ms, THREE.ClampToEdgeWrapping);
    this._viewRT = this._makeRT(tier.view[0], tier.view[1], THREE.RepeatWrapping);

    const u = {
      uTransmittanceLUT: { value: this._trRT.texture },
      uMultiScatterLUT: { value: this._msRT.texture },
      uSkyViewLUT: { value: this._viewRT.texture },
      uRayleighS: { value: new THREE.Vector3() },
      uOzoneA: { value: new THREE.Vector3(BETA_O[0], BETA_O[1], BETA_O[2]) },
      uMieSA: { value: new THREE.Vector2() },
      uScaleH: { value: new THREE.Vector2(H_RAYLEIGH, H_MIE) },
      uGroundAlbedo: { value: new THREE.Vector3(GROUND_ALBEDO[0], GROUND_ALBEDO[1], GROUND_ALBEDO[2]) },
      uTrSize: { value: new THREE.Vector2(TR_W, TR_H) },
      uMsSize: { value: new THREE.Vector2(tier.ms, tier.ms) },
      uSkyViewSize: { value: new THREE.Vector2(tier.view[0], tier.view[1]) },
      uSunDir: { value: this._sunDir },
      uMoonDir: { value: this._moonDir },
      uMoonLight: { value: this._moonLight },
      uNightAmbient: { value: new THREE.Vector3() },
      uAirglow: { value: new THREE.Vector3() },
      uNightFloor: { value: new THREE.Vector3() },
      uMieG: { value: this._mieG },
      uAureoleG: { value: this._aureoleG },
      uAureoleMix: { value: this._aureoleMix },
      uMieColumn: { value: this._p.mieS * H_MIE },
      uViewR: { value: R_GROUND },
      uExposure: { value: SKY_EXPOSURE },
      uMieTint: { value: this._mieTint },
      uSunDiscColor: { value: new THREE.Vector3() },
      uMoonGlow: { value: new THREE.Vector3() },
      uLightningTint: { value: new THREE.Vector3() },
      uSunAngRad: { value: SUN_ANGULAR_RADIUS },
      uDither: { value: 0.0045 },
      uInvProjection: { value: new THREE.Matrix4() },
      uCameraRot: { value: new THREE.Matrix3() },
    };
    this.u = u;

    const lutOpts = { depthTest: false, depthWrite: false };
    this._trMat = new THREE.ShaderMaterial({
      name: 'Atmosphere.transmittance', uniforms: u,
      vertexShader: VERT_QUAD, fragmentShader: FRAG_TRANSMITTANCE, ...lutOpts,
    });
    this._msMat = new THREE.ShaderMaterial({
      name: 'Atmosphere.multiScatter', uniforms: u,
      vertexShader: VERT_QUAD, fragmentShader: FRAG_MULTISCATTER,
      defines: { MS_DIRS: tier.msDirs, MS_STEPS: tier.msSteps }, ...lutOpts,
    });
    this._viewMat = new THREE.ShaderMaterial({
      name: 'Atmosphere.skyView', uniforms: u,
      vertexShader: VERT_QUAD, fragmentShader: FRAG_SKYVIEW,
      defines: { SKY_STEPS: tier.steps }, ...lutOpts,
    });

    this._skyMat = new THREE.ShaderMaterial({
      name: 'Atmosphere.sky', uniforms: u,
      vertexShader: VERT_SKY, fragmentShader: FRAG_SKY,
      defines: { SUN_DISC: 1, DITHER: 1 },
      depthTest: true, depthWrite: false, side: THREE.DoubleSide, fog: false,
    });
    this._skyMat.depthFunc = THREE.LessEqualDepth;

    // The env pass needs its own camera matrices but shares every other uniform
    // object by reference, so the per-frame updates reach both materials.
    const uEnv = Object.assign({}, u);
    uEnv.uInvProjection = { value: new THREE.Matrix4() };
    uEnv.uCameraRot = { value: new THREE.Matrix3() };
    this._uEnv = uEnv;
    this._envMat = new THREE.ShaderMaterial({
      name: 'Atmosphere.env', uniforms: uEnv,
      vertexShader: VERT_SKY, fragmentShader: FRAG_SKY,
      depthTest: false, depthWrite: false, side: THREE.DoubleSide, fog: false,
    });

    this.mesh = new THREE.Mesh(this._geo, this._skyMat);
    this.mesh.name = 'SkyDome';
    this.mesh.frustumCulled = false;
    this.mesh.matrixAutoUpdate = false;
    // Drawn after every other opaque object, so the terrain has already rejected
    // most sky pixels through the depth test - the cheapest possible sky on a
    // fill-rate-bound iGPU. Still ahead of the transparent pass, so stars, clouds
    // and petals composite over it.
    this.mesh.renderOrder = 1000;
    // Camera-derived uniforms are sampled here rather than in update(), because
    // the player controller moves the camera after we run.
    this.mesh.onBeforeRender = (renderer, scene, camera) => this._syncCamera(camera);
    if (this.scene) this.scene.add(this.mesh);

    this._makeEnvTarget(tier.env);

    // Every LUT is baked once and then reused for thousands of frames, so a lost
    // context would leave us with three black textures and a black sky forever.
    // Re-baking on restore is two lines and the only way back.
    this._onContextRestored = () => {
      this._mediumDirty = true;
      this._lastMediumFrame = -1e9;
      this._skyViewDirty = true;
      this._envDirty = true;
      this._envCooldown = 0;
    };
    this.renderer.domElement.addEventListener('webglcontextrestored', this._onContextRestored);

    this._renderPass(this._trRT, this._trMat);
    this._renderPass(this._msRT, this._msMat);
    this._mediumDirty = false;
    this._ready = true;

    // Prime everything so frame zero is already correct.
    this.update(0, this.state);
    if (this._framesSinceView !== 0) this._renderSkyView();
    if (this._envDirty) this._refreshEnv();
  }

  _makeRT(w, h, wrap) {
    const rt = new THREE.WebGLRenderTarget(w, h, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      wrapS: wrap,
      wrapT: THREE.ClampToEdgeWrapping,
      depthBuffer: false,
      stencilBuffer: false,
      generateMipmaps: false,
    });
    rt.texture.colorSpace = THREE.NoColorSpace;
    rt.texture.name = 'Atmosphere.LUT';
    return rt;
  }

  _makeEnvTarget(size) {
    this._envRT = new THREE.WebGLCubeRenderTarget(size, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
      stencilBuffer: false,
      generateMipmaps: false,
    });
    this._envRT.texture.colorSpace = THREE.LinearSRGBColorSpace;
    this._envRT.texture.name = 'Atmosphere.env';

    // Borrow three's own cube face cameras so the face orientation convention is
    // guaranteed to match whatever the renderer expects, now and after upgrades.
    this._envCam = new THREE.CubeCamera(0.1, 10, this._envRT);
    this._envCam.coordinateSystem = this.renderer.coordinateSystem;
    this._envCam.updateCoordinateSystem();
    this._envCam.updateMatrixWorld(true);

    this.envTexture = this._envRT.texture;
    // Nobody else owns scene IBL. If the engine has not installed an environment
    // we install ours; we only ever replace our own texture afterwards.
    if (this.scene && this.scene.environment === null) {
      this.scene.environment = this.envTexture;
      this._ownsSceneEnv = true;
    }
  }

  onQualityChange(quality) {
    const tier = TIERS[quality?.tier] || TIERS.high;
    if (tier === this._tier) return;
    this._tier = tier;
    if (!this._ready) return;

    if (this._viewRT.width !== tier.view[0] || this._viewRT.height !== tier.view[1]) {
      this._viewRT.setSize(tier.view[0], tier.view[1]);
      this.u.uSkyViewSize.value.set(tier.view[0], tier.view[1]);
    }
    if (this._msRT.width !== tier.ms) {
      this._msRT.setSize(tier.ms, tier.ms);
      this.u.uMsSize.value.set(tier.ms, tier.ms);
    }
    if (this._viewMat.defines.SKY_STEPS !== tier.steps) {
      this._viewMat.defines.SKY_STEPS = tier.steps;
      this._viewMat.needsUpdate = true;
    }
    if (this._msMat.defines.MS_DIRS !== tier.msDirs || this._msMat.defines.MS_STEPS !== tier.msSteps) {
      this._msMat.defines.MS_DIRS = tier.msDirs;
      this._msMat.defines.MS_STEPS = tier.msSteps;
      this._msMat.needsUpdate = true;
    }
    if (this._envRT.width !== tier.env) {
      this._disposeEnv();
      this._makeEnvTarget(tier.env);
    }
    this._renderPass(this._msRT, this._msMat);
    this._skyViewDirty = true;
    this._envDirty = true;
    this._envCooldown = 0;
  }

  // -------------------------------------------------------------------------
  // Per-frame
  // -------------------------------------------------------------------------

  update(dt, state) {
    const sun = state.sun;
    const moon = state.moon;

    this._syncMedium(false);

    this._sunDir.copy(sun.direction);
    if (this._sunDir.lengthSq() < 1e-8) this._sunDir.set(0, 1, 0); else this._sunDir.normalize();
    this._moonDir.copy(moon.direction);
    if (this._moonDir.lengthSq() < 1e-8) this._moonDir.set(0, -1, 0); else this._moonDir.normalize();

    // How far into night we are. Also gates every stylised night term so nothing
    // double-counts in daylight, where all of it would be invisible anyway.
    const night = smoothstep(NIGHT_RAMP_HI, NIGHT_RAMP_LO, this._sunDir.y);

    // Illuminated fraction from the phase; a partly lit moon dims faster than its
    // lit area shrinks, because of shadowing across the limb.
    const lit = clamp01((1 - Math.cos(moon.phase * Math.PI * 2)) * 0.5);
    // Derived from geometry rather than moon.intensity: that field is a
    // directional-light value in units this module does not own.
    const moonVis = clamp01(moon.visibility) * smoothstep(-0.06, 0.05, this._moonDir.y);
    this._moonAmount = Math.pow(lit, 1.35) * moonVis * night * MOON_SKY_SCALE;
    this._moonLight.set(moon.color.r, moon.color.g, moon.color.b).multiplyScalar(this._moonAmount);

    this._mieG = clamp(state.sky.mieDirectionalG, 0.05, 0.92);
    // Narrow second lobe: the true aerosol forward peak is far sharper than a
    // single HG lobe, and it is what makes the sun feel like it is burning
    // through the air rather than pasted on.
    this._aureoleG = clamp(0.5 + this._mieG * 0.575, 0.5, 0.975);
    this._aureoleMix = 0.06 + 0.05 * clamp01((this._mieG - 0.5) / 0.42);
    this._viewR = R_GROUND + clamp((this.camera ? this.camera.position.y : 0) * 0.001, 0, 60);
    this._updateMieTint();

    if (this._ready) this._updateUniforms(dt, state, night);

    // --- CPU mirror: amortised so a refresh never costs a whole frame ------
    // 1.2e-5 is 1 - cos(0.28 deg). Any coarser and the smoothed fog colour
    // visibly pulses as each rebuild lands during a fast sunset.
    if (this._cpuRow >= CPU_ZE && (this._cpuDirty || this._lightsMoved(
      this._lastCpuSun, this._lastCpuMoonDir, this._lastCpuMoonAmt, 1.2e-5, 2e-4))) {
      this._cpuRow = 0;
      this._cpuDirty = false;
      this._lastCpuSun.copy(this._sunDir);
      this._lastCpuMoonDir.copy(this._moonDir);
      this._lastCpuMoonAmt = this._moonAmount;
      this._cpuSunDir.copy(this._sunDir);
    }
    if (this._cpuRow < CPU_ZE) {
      const rows = this._tier.cpuRows;
      this._rebuildCpuSky(this._cpuRow, this._cpuRow + rows);
      this._cpuRow += rows;
    }

    this._publishColors(dt, state);
  }

  _updateUniforms(dt, state, night) {
    const u = this.u;
    const moon = state.moon;
    const inv = night / this._exposure;

    u.uAirglow.value
      .set(AIRGLOW_TINT[0], AIRGLOW_TINT[1], AIRGLOW_TINT[2]).multiplyScalar(AIRGLOW_ZENITH * inv);
    u.uNightFloor.value
      .set(NIGHT_FLOOR_TINT[0], NIGHT_FLOOR_TINT[1], NIGHT_FLOOR_TINT[2]).multiplyScalar(NIGHT_FLOOR * inv);
    u.uNightAmbient.value
      .set(NIGHT_AMBIENT_TINT[0], NIGHT_AMBIENT_TINT[1], NIGHT_AMBIENT_TINT[2]).multiplyScalar(NIGHT_AMBIENT * inv);

    u.uMieG.value = this._mieG;
    u.uAureoleG.value = this._aureoleG;
    u.uAureoleMix.value = this._aureoleMix;
    u.uMieColumn.value = this._p.mieS * H_MIE;
    u.uViewR.value = this._viewR;
    u.uExposure.value = this._exposure;

    // Sun disc: capped radiance, hue from the real sun-path transmittance, cut
    // just below the geometric horizon. The fade spans -1.3 to -0.2 degrees,
    // which is close to where refraction actually lets go of the disc.
    const T = this._sunT;
    const tl = Math.max(T[0] * 0.2126 + T[1] * 0.7152 + T[2] * 0.0722, 1e-4);
    const discK = SUN_DISC_RENDER_LUMINANCE * smoothstep(-0.022, -0.004, this._sunDir.y)
      * Math.pow(tl, SUN_DISC_DIM_POWER) / tl;
    u.uSunDiscColor.value.set(T[0] * discK, T[1] * discK, T[2] * discK);

    // Sharp core of the moon aureole; the broad halo is baked into the LUT.
    const glowK = this._moonAmount * this._exposure * 0.35;
    u.uMoonGlow.value.set(moon.color.r * glowK, moon.color.g * glowK, moon.color.b * glowK);

    const flash = clamp01(state.weather.lightning) * 0.35;
    u.uLightningTint.value.set(flash * 0.88, flash * 0.94, flash);

    // --- sky-view LUT scheduling -----------------------------------------
    this._framesSinceView++;
    const sunMoved = 1 - this._lastViewSun.dot(this._sunDir);
    const changed =
      this._skyViewDirty ||
      sunMoved > 1e-9 ||
      Math.abs(this._moonAmount - this._lastViewMoon) > 1e-6 ||
      Math.abs(this._viewR - this._lastViewR) > 0.002;
    if (changed && this._framesSinceView >= this._tier.interval) this._renderSkyView();

    // --- env map ----------------------------------------------------------
    // 5.5e-5 is 1 - cos(0.6 deg). Below that the IBL genuinely cannot change
    // enough to see, and six face draws plus a PMREM rebuild is far too
    // expensive to spend on a difference nobody can perceive.
    this._envCooldown -= dt;
    if (this._envCooldown <= 0 && (this._envDirty || this._lightsMoved(
      this._lastEnvSun, this._lastEnvMoonDir, this._lastEnvMoonAmt, 5.5e-5, 1e-3))) {
      this._refreshEnv();
      this._envCooldown = 0.35;
    }
  }

  /**
   * Has the lighting geometry moved enough to invalidate a cache? The moon only
   * counts while it is actually contributing, so a moon wandering around below
   * the horizon never triggers a rebuild.
   */
  _lightsMoved(lastSun, lastMoonDir, lastMoonAmt, dirEps, amtEps) {
    if (1 - lastSun.dot(this._sunDir) > dirEps) return true;
    if (Math.abs(this._moonAmount - lastMoonAmt) > amtEps) return true;
    if (this._moonAmount > 0 && 1 - lastMoonDir.dot(this._moonDir) > dirEps * 4) return true;
    return false;
  }

  _syncCamera(camera) {
    if (!this._ready || !camera) return;
    this.u.uInvProjection.value.copy(camera.projectionMatrixInverse);
    this.u.uCameraRot.value.setFromMatrix4(camera.matrixWorld);
  }

  /**
   * Transmittance along the sun ray. Drives both the disc colour and the tint of
   * the Mie aureole - the aureole's spectral shape is dominated by the sun-path
   * transmittance, which is why a scalar Mie channel in the LUT plus this tint
   * reproduces an orange sunset glow without a second LUT.
   */
  _updateMieTint() {
    const t = this._sunT;
    this._cpuTransmittance(this._viewR, this._sunDir.y, t);
    const g = Math.max(t[1], 1e-4);
    this._mieTint.set(clamp(t[0] / g, 0, 8), 1, clamp(t[2] / g, 0, 8));
  }

  /** Recompute medium coefficients; rebuild the medium-only LUTs when needed. */
  _syncMedium(force) {
    const s = this.state.sky;
    const w = this.state.weather;
    const c = this.state.clouds;
    const P = this._p;

    P.rayleighScale = clamp(s.rayleigh / REF_RAYLEIGH, 0.05, 4);
    // Humid, wet and overcast air carries more aerosol: the sky turns milky
    // before the first raindrop lands. Reading weather here is what makes that
    // happen for free.
    const haze = 1
      + 0.85 * clamp01(w.fogDensity / 0.02)
      + 0.55 * clamp01(w.wetness)
      + 0.40 * clamp01(c.coverage - 0.35);
    P.mieScale = clamp((s.turbidity / REF_TURBIDITY) * (s.mieCoefficient / REF_MIE_COEFF) * haze, 0.05, 12);

    P.betaR[0] = BETA_R[0] * P.rayleighScale;
    P.betaR[1] = BETA_R[1] * P.rayleighScale;
    P.betaR[2] = BETA_R[2] * P.rayleighScale;
    P.mieS = BETA_M_S * P.mieScale;
    P.mieA = BETA_M_A * P.mieScale;

    if (this._ready) {
      this.u.uRayleighS.value.set(P.betaR[0], P.betaR[1], P.betaR[2]);
      this.u.uMieSA.value.set(P.mieS, P.mieA);
      this._skyViewDirty = true;
    }

    // The transmittance and multi-scattering LUTs are smooth in these parameters,
    // so they only chase 3% quantised steps. That stops a continuous weather
    // transition from rebuilding the expensive LUTs every single frame.
    const qR = Math.round(Math.log(P.rayleighScale) / 0.03);
    const qM = Math.round(Math.log(P.mieScale) / 0.03);
    if (force || qR !== this._qRayleigh || qM !== this._qMie) {
      this._qRayleigh = qR;
      this._qMie = qM;
      this._mediumDirty = true;
      this._cpuDirty = true;
    }
    if (this._mediumDirty && this._ready) {
      const frame = this.state.time.frame;
      if (frame - this._lastMediumFrame >= 30) {
        this._lastMediumFrame = frame;
        this._mediumDirty = false;
        this._renderPass(this._trRT, this._trMat);
        this._renderPass(this._msRT, this._msMat);
        this._buildCpuTransmittance();
        this._buildCpuMultiScatter();
        this._envDirty = true;
      }
    }
  }

  // -------------------------------------------------------------------------
  // GPU passes
  // -------------------------------------------------------------------------

  _renderPass(target, material) {
    const r = this.renderer;
    if (!r) return;
    const prevTarget = r.getRenderTarget();
    const prevFace = r.getActiveCubeFace();
    const prevMip = r.getActiveMipmapLevel();
    this._quad.material = material;
    r.setRenderTarget(target);
    r.render(this._passScene, this._passCamera);
    r.setRenderTarget(prevTarget, prevFace, prevMip);
  }

  _renderSkyView() {
    this._renderPass(this._viewRT, this._viewMat);
    this._framesSinceView = 0;
    this._skyViewDirty = false;
    this._lastViewSun.copy(this._sunDir);
    this._lastViewMoon = this._moonAmount;
    this._lastViewR = this._viewR;
  }

  _refreshEnv() {
    const r = this.renderer;
    if (!r || !this._envRT) return;
    const prevTarget = r.getRenderTarget();
    const prevFace = r.getActiveCubeFace();
    const prevMip = r.getActiveMipmapLevel();
    this._quad.material = this._envMat;
    const cams = this._envCam.children;
    for (let i = 0; i < 6; i++) {
      const cam = cams[i];
      this._uEnv.uInvProjection.value.copy(cam.projectionMatrixInverse);
      this._uEnv.uCameraRot.value.setFromMatrix4(cam.matrixWorld);
      r.setRenderTarget(this._envRT, i);
      r.render(this._passScene, this._passCamera);
    }
    r.setRenderTarget(prevTarget, prevFace, prevMip);
    // Asks three to regenerate the PMREM chain from these faces, reusing the
    // render target it already allocated for this texture.
    this._envRT.texture.needsPMREMUpdate = true;
    this._envDirty = false;
    this._lastEnvSun.copy(this._sunDir);
    this._lastEnvMoonDir.copy(this._moonDir);
    this._lastEnvMoonAmt = this._moonAmount;
  }

  // -------------------------------------------------------------------------
  // State output
  // -------------------------------------------------------------------------

  _publishColors(dt, state) {
    const sky = state.sky;
    const col = this._scratchColor;

    this.getSkyColor(_upDir, col);
    let zr = col.r, zg = col.g, zb = col.b;

    let hr = 0, hg = 0, hb = 0;
    for (let i = 0; i < 8; i++) {
      this.getSkyColor(_ringDirs[i], col);
      hr += col.r; hg += col.g; hb += col.b;
    }
    hr *= 0.125; hg *= 0.125; hb *= 0.125;

    let gr = 0, gg = 0, gb = 0;
    for (let i = 0; i < _downDirs.length; i++) {
      this.getSkyColor(_downDirs[i], col);
      gr += col.r; gg += col.g; gb += col.b;
    }
    const gk = 1 / _downDirs.length;
    gr *= gk; gg *= gk; gb *= gk;

    // Overcast greys the ambient out: what the world actually sees then is the
    // cloud deck, and a cloud deck is a diffuser, not a blue dome.
    const cov = clamp01(state.clouds.coverage);
    const storm = clamp01(state.clouds.storminess);
    const grey = clamp01((cov - 0.45) / 0.5) * 0.55;
    if (grey > 0) {
      const zl = zr * 0.2126 + zg * 0.7152 + zb * 0.0722;
      const hl = hr * 0.2126 + hg * 0.7152 + hb * 0.0722;
      zr = lerp(zr, zl, grey); zg = lerp(zg, zl, grey); zb = lerp(zb, zl, grey);
      hr = lerp(hr, hl, grey); hg = lerp(hg, hl, grey); hb = lerp(hb, hl, grey);
    }
    const dim = 1 - 0.45 * clamp01((cov - 0.4) / 0.6) - 0.25 * storm;
    zr *= dim; zg *= dim; zb *= dim;
    hr *= dim; hg *= dim; hb *= dim;
    gr *= dim; gg *= dim; gb *= dim;

    // Sky irradiance proxy: the horizon band covers far more solid angle than the
    // zenith does, so it carries the larger share.
    const lum = (zr * 0.2126 + zg * 0.7152 + zb * 0.0722) * 0.34
              + (hr * 0.2126 + hg * 0.7152 + hb * 0.0722) * 0.66;
    const ambientTarget = clamp(lum / NOON_REFERENCE_LUMINANCE, 0.012, 2.5);

    // Frame-rate independent smoothing hides the step when the CPU table is
    // refreshed. lambda 5 is a ~0.2 s tail: long enough to bury the step, short
    // enough that fog still tracks a fast sunset.
    const k = this._colorsSeeded ? 1 - Math.exp(-5 * Math.max(dt, 0)) : 1;
    this._colorsSeeded = true;

    sky.zenithColor.setRGB(
      lerp(sky.zenithColor.r, zr, k), lerp(sky.zenithColor.g, zg, k), lerp(sky.zenithColor.b, zb, k));
    sky.horizonColor.setRGB(
      lerp(sky.horizonColor.r, hr, k), lerp(sky.horizonColor.g, hg, k), lerp(sky.horizonColor.b, hb, k));
    sky.groundColor.setRGB(
      lerp(sky.groundColor.r, gr, k), lerp(sky.groundColor.g, gg, k), lerp(sky.groundColor.b, gb, k));
    sky.ambientIntensity = lerp(sky.ambientIntensity, ambientTarget, k);

    // Stars are fully in once the sun is 6 degrees down, killed by cloud, washed
    // out by a bright moon.
    const dusk = smoothstep(0.055, -0.10, this._sunDir.y);
    const moonWash = 1 - 0.45 * clamp01(this._moonAmount / MOON_SKY_SCALE);
    const cloudKill = 1 - 0.92 * clamp01(cov * Math.sqrt(cov));
    sky.starIntensity = lerp(sky.starIntensity, dusk * moonWash * cloudKill, k);
  }

  // -------------------------------------------------------------------------
  // CPU model - the same maths as the shader, sized to run on refresh only.
  // -------------------------------------------------------------------------

  _buildCpuTransmittance() {
    const P = this._p;
    const t = this._cpuTr;
    const H = _H_ATMOS;
    for (let j = 0; j < CPU_TR_H; j++) {
      const rho = H * (j / (CPU_TR_H - 1));
      const r = Math.sqrt(rho * rho + R_GROUND * R_GROUND);
      const dMin = R_TOP - r;
      const dMax = rho + H;
      for (let i = 0; i < CPU_TR_W; i++) {
        const d = dMin + (i / (CPU_TR_W - 1)) * (dMax - dMin);
        const mu = d <= 0 ? 1 : clamp((H * H - rho * rho - d * d) / (2 * r * d), -1, 1);
        const sx = Math.sqrt(Math.max(0, 1 - mu * mu));
        let o0 = 0, o1 = 0, o2 = 0;
        const N = 24;
        for (let k = 0; k < N; k++) {
          const s = ((k + 0.5) / N) * d;
          const px = sx * s;
          const py = r + mu * s;
          cpuMedium(P, Math.sqrt(px * px + py * py) - R_GROUND);
          o0 += _med[4]; o1 += _med[5]; o2 += _med[6];
        }
        const f = d / N;
        const idx = (j * CPU_TR_W + i) * 3;
        t[idx] = Math.exp(-o0 * f);
        t[idx + 1] = Math.exp(-o1 * f);
        t[idx + 2] = Math.exp(-o2 * f);
      }
    }
  }

  _cpuTransmittance(r, mu, out) {
    const rc = clamp(r, R_GROUND, R_TOP);
    const H = _H_ATMOS;
    const rho = Math.sqrt(Math.max(0, rc * rc - R_GROUND * R_GROUND));
    const disc = rc * rc * (mu * mu - 1) + R_TOP * R_TOP;
    const d = Math.max(0, -rc * mu + Math.sqrt(Math.max(disc, 0)));
    const dMin = R_TOP - rc;
    const dMax = rho + H;
    const xMu = clamp01(dMax > dMin ? (d - dMin) / (dMax - dMin) : 0) * (CPU_TR_W - 1);
    const xR = clamp01(rho / H) * (CPU_TR_H - 1);
    let i0 = Math.floor(xMu); if (i0 > CPU_TR_W - 2) i0 = CPU_TR_W - 2; if (i0 < 0) i0 = 0;
    let j0 = Math.floor(xR); if (j0 > CPU_TR_H - 2) j0 = CPU_TR_H - 2; if (j0 < 0) j0 = 0;
    const fi = clamp01(xMu - i0);
    const fj = clamp01(xR - j0);
    const t = this._cpuTr;
    const a = (j0 * CPU_TR_W + i0) * 3;
    const b = a + 3;
    const c = ((j0 + 1) * CPU_TR_W + i0) * 3;
    const e = c + 3;
    for (let k = 0; k < 3; k++) {
      const top = t[a + k] + (t[b + k] - t[a + k]) * fi;
      const bot = t[c + k] + (t[e + k] - t[c + k]) * fi;
      out[k] = top + (bot - top) * fj;
    }
    return out;
  }

  _buildCpuMultiScatter() {
    const P = this._p;
    const out = this._cpuMs;
    const DIRS = 10;
    const STEPS = 8;
    const GOLDEN = 2.39996322972865332;
    for (let j = 0; j < CPU_MS_H; j++) {
      const alt = (j / (CPU_MS_H - 1)) * (R_TOP - R_GROUND);
      const r = clamp(R_GROUND + alt, R_GROUND + 0.002, R_TOP - 0.002);
      for (let i = 0; i < CPU_MS_W; i++) {
        const cosSun = (i / (CPU_MS_W - 1)) * 2 - 1;
        const sunX = Math.sqrt(Math.max(0, 1 - cosSun * cosSun));
        let l0 = 0, l1 = 0, l2 = 0, f0 = 0, f1 = 0, f2 = 0;
        for (let n = 0; n < DIRS; n++) {
          const fn = n + 0.5;
          const dy = 1 - (2 * fn) / DIRS;
          const rr = Math.sqrt(Math.max(0, 1 - dy * dy));
          const th = GOLDEN * fn;
          const dx = Math.cos(th) * rr;
          const dz = Math.sin(th) * rr;
          const tBot = raySphere3(0, r, 0, dx, dy, dz, R_GROUND);
          const tTop = raySphere3(0, r, 0, dx, dy, dz, R_TOP);
          const tMax = tBot > 0 ? tBot : tTop;
          if (tMax <= 0) continue;
          const dt = tMax / STEPS;
          let t0 = 1, t1 = 1, t2 = 1;
          for (let k = 0; k < STEPS; k++) {
            const tt = (k + 0.3) * dt;
            const px = dx * tt, py = r + dy * tt, pz = dz * tt;
            const pr = Math.sqrt(px * px + py * py + pz * pz);
            cpuMedium(P, pr - R_GROUND);
            const e0 = Math.max(_med[4], 1e-9), e1 = Math.max(_med[5], 1e-9), e2 = Math.max(_med[6], 1e-9);
            const st0 = Math.exp(-_med[4] * dt), st1 = Math.exp(-_med[5] * dt), st2 = Math.exp(-_med[6] * dt);
            const sc0 = _med[0] + _med[3], sc1 = _med[1] + _med[3], sc2 = _med[2] + _med[3];
            const mu = (px * sunX + py * cosSun) / pr;
            const shadow = raySphere3(px, py, pz, sunX, cosSun, 0, R_GROUND) >= 0 ? 0 : 1;
            this._cpuTransmittance(pr, mu, _tr3b);
            const s0 = _tr3b[0] * shadow * sc0 * 0.07957747154;
            const s1 = _tr3b[1] * shadow * sc1 * 0.07957747154;
            const s2 = _tr3b[2] * shadow * sc2 * 0.07957747154;
            l0 += t0 * (s0 - s0 * st0) / e0;
            l1 += t1 * (s1 - s1 * st1) / e1;
            l2 += t2 * (s2 - s2 * st2) / e2;
            f0 += t0 * (sc0 - sc0 * st0) / e0;
            f1 += t1 * (sc1 - sc1 * st1) / e1;
            f2 += t2 * (sc2 - sc2 * st2) / e2;
            t0 *= st0; t1 *= st1; t2 *= st2;
          }
          if (tBot > 0) {
            const px = dx * tBot, py = r + dy * tBot, pz = dz * tBot;
            const pr = Math.sqrt(px * px + py * py + pz * pz);
            const nmu = (px * sunX + py * cosSun) / pr;
            if (nmu > 0) {
              this._cpuTransmittance(R_GROUND, nmu, _tr3b);
              const q = nmu * INV_PI;
              l0 += t0 * _tr3b[0] * q * GROUND_ALBEDO[0];
              l1 += t1 * _tr3b[1] * q * GROUND_ALBEDO[1];
              l2 += t2 * _tr3b[2] * q * GROUND_ALBEDO[2];
            }
          }
        }
        const idx = (j * CPU_MS_W + i) * 3;
        out[idx] = (l0 / DIRS) / (1 - Math.min(f0 / DIRS, 0.92));
        out[idx + 1] = (l1 / DIRS) / (1 - Math.min(f1 / DIRS, 0.92));
        out[idx + 2] = (l2 / DIRS) / (1 - Math.min(f2 / DIRS, 0.92));
      }
    }
  }

  _cpuMultiScatter(r, mu, out) {
    const x = clamp01(mu * 0.5 + 0.5) * (CPU_MS_W - 1);
    const y = clamp01((r - R_GROUND) / (R_TOP - R_GROUND)) * (CPU_MS_H - 1);
    let i0 = Math.floor(x); if (i0 > CPU_MS_W - 2) i0 = CPU_MS_W - 2; if (i0 < 0) i0 = 0;
    let j0 = Math.floor(y); if (j0 > CPU_MS_H - 2) j0 = CPU_MS_H - 2; if (j0 < 0) j0 = 0;
    const fi = clamp01(x - i0);
    const fj = clamp01(y - j0);
    const t = this._cpuMs;
    const a = (j0 * CPU_MS_W + i0) * 3;
    const b = a + 3;
    const c = ((j0 + 1) * CPU_MS_W + i0) * 3;
    const e = c + 3;
    for (let k = 0; k < 3; k++) {
      const top = t[a + k] + (t[b + k] - t[a + k]) * fi;
      const bot = t[c + k] + (t[e + k] - t[c + k]) * fi;
      out[k] = top + (bot - top) * fj;
    }
    return out;
  }

  /**
   * Raymarch one direction. Writes [r, g, b, phase-less Mie] into _acc, in the
   * same render units the shader outputs.
   */
  _cpuSkyDir(dx, dy, dz) {
    const P = this._p;
    const r0 = this._viewR;
    const sun = this._cpuSunDir;
    const mx = this._moonDir.x, my = this._moonDir.y, mz = this._moonDir.z;
    const moonAmt = this._moonAmount;
    const mc = this.state.moon.color;

    _acc[0] = _acc[1] = _acc[2] = _acc[3] = 0;
    const tBot = raySphere3(0, r0, 0, dx, dy, dz, R_GROUND);
    const tTop = raySphere3(0, r0, 0, dx, dy, dz, R_TOP);
    const tMax = tBot > 0 ? tBot : tTop;
    if (tMax <= 0) return _acc;

    const night = smoothstep(NIGHT_RAMP_HI, NIGHT_RAMP_LO, sun.y);
    const invE = night / this._exposure;
    const na0 = NIGHT_AMBIENT_TINT[0] * NIGHT_AMBIENT * invE;
    const na1 = NIGHT_AMBIENT_TINT[1] * NIGHT_AMBIENT * invE;
    const na2 = NIGHT_AMBIENT_TINT[2] * NIGHT_AMBIENT * invE;

    const pRs = phaseRayleighCPU(dx * sun.x + dy * sun.y + dz * sun.z);
    const cM = dx * mx + dy * my + dz * mz;
    const pRm = phaseRayleighCPU(cM);
    const pMm = phaseMieCPU(cM, this._mieG * 0.6);

    let l0 = 0, l1 = 0, l2 = 0, mAcc = 0;
    let t0 = 1, t1 = 1, t2 = 1;

    for (let i = 0; i < CPU_STEPS; i++) {
      const f0 = i / CPU_STEPS;
      const f1 = (i + 1) / CPU_STEPS;
      const tA = f0 * f0 * tMax;
      const dt = f1 * f1 * tMax - tA;
      const tt = tA + dt * 0.4;
      const px = dx * tt, py = r0 + dy * tt, pz = dz * tt;
      const pr = Math.sqrt(px * px + py * py + pz * pz);
      cpuMedium(P, pr - R_GROUND);
      const sr0 = _med[0], sr1 = _med[1], sr2 = _med[2], sm = _med[3];
      const e0 = Math.max(_med[4], 1e-9), e1 = Math.max(_med[5], 1e-9), e2 = Math.max(_med[6], 1e-9);
      const st0 = Math.exp(-_med[4] * dt), st1 = Math.exp(-_med[5] * dt), st2 = Math.exp(-_med[6] * dt);
      const sc0 = sr0 + sm, sc1 = sr1 + sm, sc2 = sr2 + sm;
      const ux = px / pr, uy = py / pr, uz = pz / pr;

      const muS = ux * sun.x + uy * sun.y + uz * sun.z;
      const shS = raySphere3(px, py, pz, sun.x, sun.y, sun.z, R_GROUND) >= 0 ? 0 : 1;
      this._cpuTransmittance(pr, muS, _tr3);
      this._cpuMultiScatter(pr, muS, _ms3);
      const ts1 = _tr3[1] * shS;

      let s0 = _tr3[0] * shS * sr0 * pRs + _ms3[0] * sc0 + na0 * sc0;
      let s1 = ts1 * sr1 * pRs + _ms3[1] * sc1 + na1 * sc1;
      let s2 = _tr3[2] * shS * sr2 * pRs + _ms3[2] * sc2 + na2 * sc2;
      const sm1 = ts1 * sm;

      if (moonAmt > 0) {
        const muM = ux * mx + uy * my + uz * mz;
        const shM = raySphere3(px, py, pz, mx, my, mz, R_GROUND) >= 0 ? 0 : 1;
        this._cpuTransmittance(pr, muM, _tr3b);
        this._cpuMultiScatter(pr, muM, _ms3b);
        s0 += moonAmt * mc.r * (_tr3b[0] * shM * (sr0 * pRm + sm * pMm) + _ms3b[0] * sc0);
        s1 += moonAmt * mc.g * (_tr3b[1] * shM * (sr1 * pRm + sm * pMm) + _ms3b[1] * sc1);
        s2 += moonAmt * mc.b * (_tr3b[2] * shM * (sr2 * pRm + sm * pMm) + _ms3b[2] * sc2);
      }

      l0 += t0 * (s0 - s0 * st0) / e0;
      l1 += t1 * (s1 - s1 * st1) / e1;
      l2 += t2 * (s2 - s2 * st2) / e2;
      mAcc += t1 * (sm1 - sm1 * st1) / e1;
      t0 *= st0; t1 *= st1; t2 *= st2;
    }

    if (tBot > 0) {
      const px = dx * tBot, py = r0 + dy * tBot, pz = dz * tBot;
      const pr = Math.sqrt(px * px + py * py + pz * pz);
      const nmu = (px * sun.x + py * sun.y + pz * sun.z) / pr;
      if (nmu > 0) {
        this._cpuTransmittance(R_GROUND, nmu, _tr3);
        const q = nmu * INV_PI;
        l0 += t0 * _tr3[0] * q * GROUND_ALBEDO[0];
        l1 += t1 * _tr3[1] * q * GROUND_ALBEDO[1];
        l2 += t2 * _tr3[2] * q * GROUND_ALBEDO[2];
      }
      if (moonAmt > 0) {
        const nmuM = (px * mx + py * my + pz * mz) / pr;
        if (nmuM > 0) {
          this._cpuTransmittance(R_GROUND, nmuM, _tr3b);
          const q = nmuM * INV_PI * moonAmt;
          l0 += t0 * _tr3b[0] * q * GROUND_ALBEDO[0] * mc.r;
          l1 += t1 * _tr3b[1] * q * GROUND_ALBEDO[1] * mc.g;
          l2 += t2 * _tr3b[2] * q * GROUND_ALBEDO[2] * mc.b;
        }
      }
    } else {
      const sh2 = Math.max(0, 1 - dy * dy);
      const vr = 1 / Math.sqrt(Math.max(1 - 0.973169 * sh2, 0.0035));
      const ag = AIRGLOW_ZENITH * invE;
      const nf = NIGHT_FLOOR * invE;
      l0 += t0 * (AIRGLOW_TINT[0] * ag * vr + NIGHT_FLOOR_TINT[0] * nf);
      l1 += t1 * (AIRGLOW_TINT[1] * ag * vr + NIGHT_FLOOR_TINT[1] * nf);
      l2 += t2 * (AIRGLOW_TINT[2] * ag * vr + NIGHT_FLOOR_TINT[2] * nf);
    }

    const E = this._exposure;
    _acc[0] = l0 * E; _acc[1] = l1 * E; _acc[2] = l2 * E; _acc[3] = mAcc * E;
    return _acc;
  }

  /** Refill rows [from, to) of the CPU sky table. Amortised across frames. */
  _rebuildCpuSky(from, to) {
    const tab = this._cpuSky;
    const end = Math.min(to, CPU_ZE);
    for (let j = from; j < end; j++) {
      // Same warped zenith parameterisation as the GPU LUT.
      const s = (j / (CPU_ZE - 1)) * 2 - 1;
      const n = Math.sign(s) * s * s;
      const theta = (1 - n) * (Math.PI * 0.5);
      const y = Math.cos(theta);
      const sh = Math.sin(theta);
      for (let i = 0; i < CPU_AZ; i++) {
        const az = (i / CPU_AZ - 0.5) * Math.PI * 2;
        const a = this._cpuSkyDir(Math.cos(az) * sh, y, Math.sin(az) * sh);
        const idx = (j * CPU_AZ + i) * 4;
        tab[idx] = a[0]; tab[idx + 1] = a[1]; tab[idx + 2] = a[2]; tab[idx + 3] = a[3];
      }
    }
  }

  /**
   * CPU approximation of the shader, for fog colour and ambient tinting.
   * Returns linear render-space radiance - directly comparable to what the sky
   * shader puts on screen. Allocation free when `out` is supplied.
   */
  getSkyColor(direction, out = new THREE.Color()) {
    let x = direction.x, y = direction.y, z = direction.z;
    const len = Math.sqrt(x * x + y * y + z * z);
    if (len > 1e-9) { const k = 1 / len; x *= k; y *= k; z *= k; }
    else { x = 0; y = 1; z = 0; }

    const n = 1 - 2 * Math.acos(clamp(y, -1, 1)) * INV_PI;
    const fv = clamp01(0.5 + 0.5 * Math.sign(n) * Math.sqrt(Math.abs(n))) * (CPU_ZE - 1);
    let j0 = Math.floor(fv); if (j0 > CPU_ZE - 2) j0 = CPU_ZE - 2; if (j0 < 0) j0 = 0;
    const wj = clamp01(fv - j0);

    const fu = (Math.atan2(z, x) * INV_TAU + 0.5) * CPU_AZ;
    let i0 = Math.floor(fu);
    const wi = fu - i0;
    i0 = ((i0 % CPU_AZ) + CPU_AZ) % CPU_AZ;
    const i1 = (i0 + 1) % CPU_AZ;

    const tab = this._cpuSky;
    const a = (j0 * CPU_AZ + i0) * 4;
    const b = (j0 * CPU_AZ + i1) * 4;
    const c = ((j0 + 1) * CPU_AZ + i0) * 4;
    const d = ((j0 + 1) * CPU_AZ + i1) * 4;

    const mie = lerp(lerp(tab[a + 3], tab[b + 3], wi), lerp(tab[c + 3], tab[d + 3], wi), wj);
    const sun = this._cpuSunDir;
    const cosSun = x * sun.x + y * sun.y + z * sun.z;
    const pm = lerp(phaseMieCPU(cosSun, this._mieG), phaseMieCPU(cosSun, this._aureoleG), this._aureoleMix);
    const t = this._mieTint;

    return out.setRGB(
      lerp(lerp(tab[a], tab[b], wi), lerp(tab[c], tab[d], wi), wj) + mie * t.x * pm,
      lerp(lerp(tab[a + 1], tab[b + 1], wi), lerp(tab[c + 1], tab[d + 1], wi), wj) + mie * t.y * pm,
      lerp(lerp(tab[a + 2], tab[b + 2], wi), lerp(tab[c + 2], tab[d + 2], wi), wj) + mie * t.z * pm
    );
  }

  // -------------------------------------------------------------------------

  _disposeEnv() {
    if (!this._envRT) return;
    if (this._ownsSceneEnv && this.scene && this.scene.environment === this._envRT.texture) {
      this.scene.environment = null;
    }
    // Disposing the cube texture also drops three's cached PMREM target for it.
    this._envRT.texture.dispose();
    this._envRT.dispose();
    this._envRT = null;
    this._envCam = null;
    this.envTexture = null;
  }

  dispose() {
    if (this._onContextRestored && this.renderer) {
      this.renderer.domElement.removeEventListener('webglcontextrestored', this._onContextRestored);
      this._onContextRestored = null;
    }
    if (this.mesh) {
      this.mesh.onBeforeRender = () => {};
      if (this.scene) this.scene.remove(this.mesh);
    }
    this._disposeEnv();
    this._trRT?.dispose();
    this._msRT?.dispose();
    this._viewRT?.dispose();
    this._trMat?.dispose();
    this._msMat?.dispose();
    this._viewMat?.dispose();
    this._skyMat?.dispose();
    this._envMat?.dispose();
    this._geo?.dispose();
    this._ready = false;
  }
}
