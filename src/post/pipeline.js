/**
 * post/pipeline.js - the final image.
 *
 * `main.js` calls `post.render(dt, state)` exactly once per frame and that call
 * is the ONLY draw in the frame. Everything the player ever sees passes through
 * this file, so it owns the two things that decide whether the project reads as
 * a render or as a photograph: the **tone curve** and the **order of operations**.
 *
 * ---------------------------------------------------------------------------
 * WHO OWNS COLOUR - read this before changing anything
 * ---------------------------------------------------------------------------
 * `core/engine.js` configures the renderer for the *standalone* case:
 * `toneMapping = ACESFilmicToneMapping`, `outputColorSpace = SRGBColorSpace`,
 * and it drives `renderer.toneMappingExposure` every frame from its own
 * photometric auto-exposure. Three r180 applies BOTH of those only when a
 * material renders to the **default framebuffer**:
 *
 *     WebGLPrograms.getParameters():
 *       let toneMapping = NoToneMapping;
 *       if (material.toneMapped && currentRenderTarget === null) toneMapping = renderer.toneMapping;
 *     WebGLRenderer.setProgram():
 *       colorSpace = (currentRenderTarget === null) ? outputColorSpace : LinearSRGBColorSpace;
 *
 * Under a composer the scene is rendered into a HalfFloat render target, so
 * three's tone mapping is compiled *out* of every scene material and the frame
 * arrives here as untouched scene-referred linear radiance. That is exactly what
 * we want, and it means:
 *
 *   1. **This file owns tone mapping.** It is applied once, in `FilmGradeEffect`,
 *      after bloom and light shafts have been added in linear light.
 *   2. **Engine's renderer settings are deliberately left alone.** They cost
 *      nothing while the composer drives (they are compiled out), and they keep
 *      the emergency `renderer.render(scene, camera)` fallback in `render()`
 *      producing a correct - if ungraded - image instead of a black screen.
 *   3. **`engine.autoExposure` stays ON.** The obvious move is to switch it off
 *      and read `engine.exposure`, but `Engine._updateExposure()` early-returns
 *      when it is off, which would freeze that value at 1.0 and throw the
 *      day/night adaptation away. Instead we *read* an exposure that is already
 *      published: `sky/daynight.js` calls `post.setExposure()` every frame with
 *      its art-directed curve (the authoritative one), and if that never arrives
 *      we fall back to `renderer.toneMappingExposure`, which is `engine.exposure`.
 *      Exactly one of the two is ever applied, so exposure is never doubled.
 *   4. **sRGB is encoded exactly once, by us.** `FilmGradeEffect` ends with the
 *      sRGB OETF and every pass from that point on sets `encodeOutput = false`,
 *      so three's `colorspace_fragment` never runs a second time. Doing it here
 *      instead of in the final blit is not stylistic: it puts the antialiasing
 *      pass in *perceptual* space, which is what SMAA's edge thresholds are
 *      calibrated for, and it puts grain and dither in display space, which is
 *      where film grain and 8-bit dither actually live.
 *
 * ---------------------------------------------------------------------------
 * ORDER
 * ---------------------------------------------------------------------------
 *   RenderPass            scene -> linear HDR half-float
 *   N8AOPostPass          AO, half-res, denoised; multiplies the lit colour
 *   EffectPass  #1        chromatic aberration        (CONVOLUTION, sorted first)
 *                         sun shafts / DoF / motion blur          (DEPTH)
 *                         bloom, added in linear light            (NONE)
 *                         FilmGrade: exposure -> vignette -> ACES ->
 *                         lift -> contrast -> saturation -> sRGB  (NONE)
 *   EffectPass  #2        SMAA or FXAA, then film grain + dither
 *
 * Everything except FilmGrade is optional and is *omitted from the merged
 * shader* when it is off rather than merely zeroed - see `_featureSignature`.
 * A zeroed effect still costs its texture fetches and its blend every frame,
 * which is the wrong trade on the tier that has the least fill rate to spare.
 *
 * `postprocessing` sorts effects inside a pass by `attributes` descending
 * (CONVOLUTION 2 > DEPTH 1 > NONE 0) with a stable sort, so the registration
 * order above is exactly the execution order. AO runs before anything that
 * depends on scene lighting; bloom and shafts are added in linear light before
 * the tone curve; AA and grain are last. That is the whole contract.
 *
 * ---------------------------------------------------------------------------
 * BUDGET - AMD Radeon 780M, 1080p, ~3 ms for the whole stack
 * ---------------------------------------------------------------------------
 *   AO (half-res, 12 samples, 1 denoise iteration)   ~1.2 ms   HIGH+
 *   sun shafts (0.45 scale, 32 samples)              ~0.5 ms
 *   bloom (half-res luminance + 6 mip levels)        ~0.5 ms
 *   merged effect pass                               ~0.3 ms
 *   SMAA (edges + weights + blend)                   ~0.5 ms
 * LOW switches AO, shafts and chromatic aberration off entirely, drops bloom out
 * of the merged shader (ui/hud.js clears state.quality.bloom for that tier) and
 * uses FXAA. What is left is one SRC grade plus FXAA and grain: near 0.8 ms, and
 * no depth texture is allocated at all, because nothing left in the stack
 * declares EffectAttribute.DEPTH.
 */

import {
  Color,
  HalfFloatType,
  Matrix4,
  Uniform,
  Vector2,
  Vector3,
} from 'three';

import {
  BlendFunction,
  BloomEffect,
  ChromaticAberrationEffect,
  DepthOfFieldEffect,
  EdgeDetectionMode,
  Effect,
  EffectAttribute,
  EffectComposer,
  EffectPass,
  FXAAEffect,
  GodRaysEffect,
  KernelSize,
  RenderPass,
  SMAAEffect,
  SMAAPreset,
} from 'postprocessing';

import { N8AOPostPass } from 'n8ao';

/**
 * n8ao 1.10.3 does not compose correctly with this pipeline under three r180 and
 * a HalfFloat frame buffer. Measured at 09:00, clear sky, camera 26 m from the tree:
 *
 *                     sky                 ground
 *   AO enabled        [208,210,210]       [157,172,174]   washed to grey
 *   AO disabled       [154,195,233]       [132,148,141]   correct blue / green
 *
 * It is not an occlusion-strength problem: with all scene geometry hidden, an empty
 * frame reads [16,18,24] without the pass and [50,55,69] with it, so the pass ADDS
 * light where there is no geometry at all. Toggling gammaCorrection, autosetGamma
 * and colorMultiply changes the output by less than one LSB, so it is not a colour
 * space setting - the pass output itself is wrong in this configuration.
 *
 * Rather than ship a washed-out image for the sake of a subtle effect, AO is off.
 * The depth cues it would have provided are carried instead by the per-blade
 * vertical AO gradient in world/grass.js and the cavity darkening in the bark and
 * ground materials, which cost nothing extra. Removing it also returns ~1.2 ms.
 *
 * To re-enable: set this to false and re-measure the table above.
 */
const AO_DISABLED = true;

import { QUALITY } from '../core/state.js';
import { clamp, clamp01, damp, lerp, smoothstep } from '../core/math.js';

// ---------------------------------------------------------------------------
// Per-tier cost policy. Everything expensive is a number in this table rather
// than a magic constant buried three hundred lines down.
// ---------------------------------------------------------------------------

const TIERS = {
  [QUALITY.LOW]: {
    // AO and shafts are hard-off on LOW: the brief forbids them, and on a
    // fill-rate-bound iGPU they are the two most expensive things in this file.
    allowAO: false,
    allowShafts: false,
    allowDoF: false,
    allowMotionBlur: false,
    aoSamples: 8, aoDenoiseSamples: 4, aoDenoiseIterations: 1, aoHalfRes: true,
    shaftSamples: 16, shaftScale: 0.25, shaftKernel: KernelSize.VERY_SMALL,
    bloomLevels: 4, bloomRadius: 0.78, bloomIntensity: 0.34, bloomLuminanceScale: 0.5,
    smaaPreset: SMAAPreset.LOW,
    grain: 0.006,
    aberration: 0,
    mbSamples: 6,
  },
  [QUALITY.MEDIUM]: {
    allowAO: true,
    allowShafts: true,
    allowDoF: false,
    allowMotionBlur: false,
    aoSamples: 8, aoDenoiseSamples: 4, aoDenoiseIterations: 1, aoHalfRes: true,
    shaftSamples: 24, shaftScale: 0.35, shaftKernel: KernelSize.VERY_SMALL,
    bloomLevels: 5, bloomRadius: 0.82, bloomIntensity: 0.38, bloomLuminanceScale: 0.5,
    smaaPreset: SMAAPreset.LOW,
    grain: 0.008,
    aberration: 0.00040,
    mbSamples: 6,
  },
  [QUALITY.HIGH]: {
    allowAO: true,
    allowShafts: true,
    allowDoF: true,
    allowMotionBlur: true,
    aoSamples: 12, aoDenoiseSamples: 6, aoDenoiseIterations: 1, aoHalfRes: true,
    shaftSamples: 32, shaftScale: 0.45, shaftKernel: KernelSize.SMALL,
    bloomLevels: 6, bloomRadius: 0.85, bloomIntensity: 0.40, bloomLuminanceScale: 0.5,
    smaaPreset: SMAAPreset.MEDIUM,
    grain: 0.010,
    aberration: 0.00055,
    mbSamples: 8,
  },
  [QUALITY.ULTRA]: {
    allowAO: true,
    allowShafts: true,
    allowDoF: true,
    allowMotionBlur: true,
    // Still half-res AO on ULTRA: the reference GPU is an iGPU, and 24 samples
    // at half resolution buy far more visible occlusion than 12 at full.
    aoSamples: 24, aoDenoiseSamples: 8, aoDenoiseIterations: 2, aoHalfRes: true,
    shaftSamples: 48, shaftScale: 0.5, shaftKernel: KernelSize.SMALL,
    bloomLevels: 7, bloomRadius: 0.87, bloomIntensity: 0.42, bloomLuminanceScale: 0.75,
    smaaPreset: SMAAPreset.HIGH,
    grain: 0.011,
    aberration: 0.00065,
    mbSamples: 12,
  },
};

/** Contact-AO radius in metres. Grass blades are ~0.5 m, so this reads as the
 *  darkening where a blade meets soil rather than as a large ambient term. */
const AO_RADIUS = 1.15;
/** n8ao computes `finalAo = pow(ao, intensity)` - an exponent, not a gain. */
const AO_POWER = 2.6;
/** Occluded crevices keep a little bounce light instead of going to pure black.
 *  n8ao converts this from sRGB, so author it as a screen colour. */
const AO_TINT = new Color(0.17, 0.16, 0.155);
const AO_DENOISE_RADIUS = 10;

/** Display-referred luminance at which bloom starts. Converted to the
 *  scene-referred threshold every frame by dividing through exposure. */
const BLOOM_DISPLAY_THRESHOLD = 0.86;
const BLOOM_KNEE = 0.42;

/** Corner darkening before the time-of-day trim. */
const VIGNETTE_BASE = 0.20;

/** Peak shaft contribution in display-referred units. Deliberately low: shafts
 *  that reach 1.0 stop reading as light and start reading as fog. */
const SHAFT_DISPLAY_PEAK = 0.42;

/** 180-degree shutter: half of the frame's own screen-space motion. */
const MOTION_SHUTTER = 0.5;
/** Ceiling on the smear, in UV units, so a fast mouse flick cannot melt the frame. */
const MOTION_MAX_UV = 0.045;
/** Above this frame time the reprojection is meaningless (tab switch, GC pause). */
const MOTION_RESET_DT = 0.08;

/** state.weather.fogDensity that counts as "fully hazy" for the effects that
 *  respond to aerosol. state.js ships 0.0022 as the clear-air default. */
const HAZE_REFERENCE = 0.02;

// Grade targets, blended by sun elevation. Gains are linear multipliers applied
// BEFORE the tone curve (a filter on the lens); lifts are display-referred and
// applied after it (a lifted, tinted black point).
const GAIN_DAY = new Vector3(1.0, 1.0, 1.0);
const GAIN_GOLDEN = new Vector3(1.075, 0.995, 0.925);
const GAIN_NIGHT = new Vector3(0.880, 0.955, 1.135);

const LIFT_DAY = new Vector3(0.006, 0.007, 0.010);
const LIFT_GOLDEN = new Vector3(0.013, 0.008, 0.013);
const LIFT_NIGHT = new Vector3(0.008, 0.015, 0.031);

// ---------------------------------------------------------------------------
// Module scratch. Nothing in the per-frame path allocates.
// ---------------------------------------------------------------------------
const _fwd = new Vector3();
const _focusTarget = new Vector3();
const _gain = new Vector3();
const _lift = new Vector3();
const _size = new Vector2();

/**
 * Every number this file consumes comes from a sibling system, and a single NaN
 * arriving from any of them is permanent: `damp(a, NaN)` is NaN, and the next
 * frame reads that NaN back as `a`, so the value never recovers even after the
 * source does. Sanitising at the read is the only place that is cheap.
 */
const num = (v, fallback) => (Number.isFinite(v) ? v : fallback);

/** `damp` with the poisoned-accumulator escape hatch. `target` must be finite. */
const dampTo = (current, target, lambda, dt) =>
  damp(Number.isFinite(current) ? current : target, target, lambda, dt);

// ===========================================================================
// GLSL fragments shared by the custom effects.
//
// `postprocessing` renames every function, uniform and macro an effect declares
// with a per-effect prefix when it merges them, so these can be pasted into
// several effects without colliding.
// ===========================================================================

/**
 * Rec.709 luma. `luminance()` is NOT part of three's shader chunks - the effect
 * prelude only pulls in <common> and <packing> - so anything that needs it has
 * to bring its own.
 */
const GLSL_LUMA = `
float srLuma(const in vec3 c) {
  return dot(c, vec3(0.2126, 0.7152, 0.0722));
}
`;

/** Interleaved gradient noise: smooth in screen space and *fixed* there, so a
 *  dithered sample offset never crawls the way a time-varying hash would. */
const GLSL_IGN = `
float srIgn(const in vec2 p) {
  return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}
`;

// ===========================================================================
// FilmGradeEffect - exposure, lens vignette, ACES, grade. The heart of LEAF 2.
//
// The tone curve is ACES (Stephen Hill's fit), the same matrices three r180
// ships in <tonemapping_pars_fragment>. They are reproduced here because three
// only injects that chunk into materials it is tone mapping itself, and a
// merged EffectMaterial never is. The 1/0.6 scale is three's subjective
// brightening (three.js issue #19621); keeping it means this pass and the
// standalone renderer path produce the same image. ACES rather than AgX because
// AgX's default look transform is deliberately low contrast, and the one thing
// this project's frames do not need is less contrast.
//
// The shader body below stays ASCII-only and terse on purpose: GLSL ES is
// specified over a subset of ASCII, and every one of these lines is pasted into
// a merged shader alongside five other effects.
// ===========================================================================

const GLSL_GRADE = `
uniform float uExposure;
uniform vec3 uGain;
uniform vec3 uLift;
uniform float uContrast;
uniform float uSaturation;
uniform vec3 uVignette;
${GLSL_LUMA}

/* ACES filmic, Stephen Hill's fit. See the JS note above. */
vec3 srRrtOdt(const in vec3 v) {
  vec3 a = v * (v + 0.0245786) - 0.000090537;
  vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return a / b;
}

vec3 srAces(vec3 color) {
  const mat3 IN_MAT = mat3(
    vec3(0.59719, 0.07600, 0.02840),
    vec3(0.35458, 0.90834, 0.13383),
    vec3(0.04823, 0.01566, 0.83777)
  );
  const mat3 OUT_MAT = mat3(
    vec3( 1.60475, -0.10208, -0.00327),
    vec3(-0.53108,  1.10813, -0.07276),
    vec3(-0.07367, -0.00605,  1.07602)
  );
  color *= 1.0 / 0.6;
  color = IN_MAT * color;
  color = srRrtOdt(color);
  color = OUT_MAT * color;
  return clamp(color, 0.0, 1.0);
}

/* sRGB OETF, identical to three's sRGBTransferOETF. Applied here so the
   antialiasing pass downstream operates on perceptual values. */
vec3 srEncode(const in vec3 c) {
  vec3 v = max(c, vec3(0.0));
  return mix(
    pow(v, vec3(0.41666667)) * 1.055 - vec3(0.055),
    v * 12.92,
    vec3(lessThanEqual(v, vec3(0.0031308)))
  );
}

void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {

  vec3 c = max(inputColor.rgb, vec3(0.0));

  /* 1. Lens vignette, in LINEAR light and before the sensor curve, because that
        is where a real lens loses it. y scales x into film-plane units, z
        normalises the radius so 1.0 is the corner at any aspect. */
  vec2 d = (uv - 0.5) * vec2(uVignette.y, 1.0);
  float r2 = clamp(dot(d, d) * uVignette.z, 0.0, 1.0);
  c *= 1.0 - uVignette.x * r2 * r2;

  /* 2. Exposure, then white balance, still scene-referred. Grading colour
        BEFORE the curve is what makes a warm dawn roll off warm. */
  c *= uExposure;
  c *= uGain;

  /* 3. Highlight desaturation. ACES skews very bright saturated colour; the two
        brightest things here are a blue sky and a 260-unit sun disc. Bleaching
        toward the channel maximum holds brightness but takes saturation out of
        exactly the values that would skew, as a real shoulder does. */
  float peak = max(max(c.r, c.g), c.b);
  c = mix(c, vec3(peak), smoothstep(1.0, 7.0, peak) * 0.35);

  /* 4. The tone curve: real toe and shoulder contrast, which a plain
        linear-to-sRGB transfer has none of. */
  c = srAces(c);

  /* 5. Tinted black lift. Maps 0 to uLift and holds 1.0 at 1.0, so it opens the
        shadows and puts colour in them without milking the highlights. */
  c = uLift + c * (1.0 - uLift);

  /* 6. Micro-contrast. The Hermite S fixes both endpoints and the 0.5 pivot
        exactly, so no amount of it can clip or shift mid grey. Per channel on
        purpose: it returns some of the saturation aerial perspective eats. */
  c = mix(c, c * c * (3.0 - 2.0 * c), uContrast);

  /* 7. Saturation about luma. */
  float l = srLuma(c);
  c = clamp(mix(vec3(l), c, uSaturation), 0.0, 1.0);

  /* 8. Display encode. Everything downstream is display-referred and must not
        be linearised again. */
  outputColor = vec4(srEncode(c), inputColor.a);
}
`;

class FilmGradeEffect extends Effect {
  constructor() {
    super('SakuraGrade', GLSL_GRADE, {
      blendFunction: BlendFunction.SRC,
      uniforms: new Map([
        ['uExposure', new Uniform(1)],
        ['uGain', new Uniform(new Vector3(1, 1, 1.02))],
        ['uLift', new Uniform(new Vector3(-0.012, -0.008, -0.004))],
        ['uContrast', new Uniform(0.34)],
        ['uSaturation', new Uniform(2.05)],
        // x = corner darkening, y = film-plane x scale, z = 1 / corner radius^2
        ['uVignette', new Uniform(new Vector3(VIGNETTE_BASE, 1, 2))],
      ]),
    });
  }

  /**
   * Called by EffectPass with the drawing-buffer size, including on every
   * resize. Converts UV into film-plane coordinates (x scaled by the aspect
   * ratio, normalised to the frame height) so the vignette is a circle on
   * screen rather than an ellipse, then normalises the radius so that r2 == 1
   * lands exactly on the corner at any window shape.
   */
  setSize(width, height) {
    const aspect = width > 0 && height > 0 ? width / height : 1;
    const v = this.uniforms.get('uVignette').value;
    v.y = aspect;
    // corner dot(d,d) is 0.25 * (aspect^2 + 1); this is its reciprocal.
    v.z = 4 / (aspect * aspect + 1);
  }
}

// ===========================================================================
// FilmGrainEffect - display-referred grain plus a triangular-PDF dither.
//
// Deliberately separate from the grade so it can be merged into the
// ANTIALIASING pass: grain applied before SMAA is read as edge detail, smeared
// by the blend weights and half destroyed. After AA it stays crisp, and the
// dither rides along in exactly the right place - after the sRGB encode and
// immediately before the 8-bit framebuffer quantises everything.
// ===========================================================================

const GLSL_GRAIN = `
uniform vec3 uGrain;
${GLSL_LUMA}

/* Hoskins hash13. The small multiplier is load bearing, not cosmetic: with the
   more common 443.897 variant, gl_FragCoord.x * 443.897 reaches ~8.5e5 at 1080p,
   where a 24-bit mantissa has a spacing of 2^-4, so fract() of that yields only
   sixteen distinct values. That visibly quantises both the grain and the dither
   the sky gradient depends on. At 0.1031 the product stays under 2^8 and fract()
   keeps ~16 bits. (ASCII only in here: GLSL ES is specified over an ASCII
   subset, and a stray em dash in a comment is a real compile risk.) */
float srGrainHash(const in vec3 p) {
  vec3 q = fract(p * 0.1031);
  q += dot(q, q.zyx + 31.32);
  return fract((q.x + q.y) * q.z);
}

void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {

  /* Film grain is silver-halide density: it peaks in the midtones and vanishes
     in both clipped white and true black. The 4*l*(1-l) envelope is that
     behaviour, and it is also what keeps grain from crawling visibly across a
     flat sky or a blown highlight. */
  float l = srLuma(inputColor.rgb);
  float envelope = 4.0 * l * (1.0 - l);

  /* Two decorrelated taps, so the grain carries a little chroma the way real
     film does instead of reading as a monochrome video overlay. */
  float a = srGrainHash(vec3(gl_FragCoord.xy, uGrain.y)) - 0.5;
  float b = srGrainHash(vec3(gl_FragCoord.yx, uGrain.y + 7.3)) - 0.5;
  vec3 n = vec3(a, mix(a, b, 0.6), b);

  vec3 c = inputColor.rgb + n * (uGrain.x * envelope);

  /* Triangular-PDF dither at roughly one 8-bit code. The sky is a smooth
     gradient across two thirds of the frame and bands without it. TPDF rather
     than uniform because it decorrelates the quantisation error from the
     signal, which is what removes the residual pattern rather than hiding it. */
  float t = srGrainHash(vec3(gl_FragCoord.xy, uGrain.y + 31.7))
          - srGrainHash(vec3(gl_FragCoord.xy, uGrain.y + 53.1));
  c += t * uGrain.z;

  outputColor = vec4(clamp(c, 0.0, 1.0), inputColor.a);
}
`;

class FilmGrainEffect extends Effect {
  constructor() {
    super('SakuraGrain', GLSL_GRAIN, {
      blendFunction: BlendFunction.SRC,
      uniforms: new Map([
        // x = grain amplitude, y = per-frame seed, z = dither amplitude
        ['uGrain', new Uniform(new Vector3(0.01, 0, 1 / 255))],
      ]),
    });
  }
}

// ===========================================================================
// CameraMotionBlurEffect - depth reprojection against the previous frame.
//
// Declared DEPTH rather than CONVOLUTION even though it takes neighbouring
// taps: CONVOLUTION would force it ahead of the chromatic aberration, and
// `postprocessing` refuses to merge two convolution effects into one pass.
// Like the library's own DepthOfFieldEffect it takes its off-centre taps from
// `inputBuffer` and its centre tap from the accumulated `inputColor`, so
// whatever ran before it survives at full weight in the middle of the blur.
// ===========================================================================

const GLSL_MOTION_BLUR = `
uniform mat4 uMbInvViewProj;
uniform mat4 uMbPrevViewProj;
uniform vec2 uMbParams;
${GLSL_IGN}

void mainImage(const in vec4 inputColor, const in vec2 uv, const in float depth, out vec4 outputColor) {

  /* depth == 1.0 puts the reconstructed point at infinity and w at zero. Nudging
     it just inside the far plane lets the sky reproject, and therefore smear
     correctly when the player whips the camera round, without a divide by zero. */
  float d = min(depth, 0.999995);

  vec4 clip = vec4(uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
  vec4 world = uMbInvViewProj * clip;
  world /= world.w;

  vec4 prev = uMbPrevViewProj * world;
  vec2 prevUv = (prev.xy / prev.w) * 0.5 + 0.5;

  vec2 velocity = (uv - prevUv) * uMbParams.x;
  float speed = length(velocity);
  if (speed < 1e-5) {
    outputColor = inputColor;
    return;
  }
  velocity *= min(speed, uMbParams.y) / speed;

  /* Jittering the tap positions turns the banding a fixed tap pattern produces
     into noise, which the grain downstream then hides completely. */
  float jitter = srIgn(gl_FragCoord.xy);

  vec4 sum = inputColor;
  for (int i = 1; i < MB_SAMPLES; ++i) {
    float t = (float(i) + jitter) / float(MB_SAMPLES) - 0.5;
    vec2 p = clamp(uv + velocity * t, vec2(0.0), vec2(1.0));
    sum += texture2D(inputBuffer, p);
  }

  outputColor = sum / float(MB_SAMPLES);
}
`;

class CameraMotionBlurEffect extends Effect {
  constructor(camera, samples) {
    super('SakuraMotionBlur', GLSL_MOTION_BLUR, {
      attributes: EffectAttribute.DEPTH,
      blendFunction: BlendFunction.SRC,
      // A literal upper bound on the loop, as CONTRACTS §2 requires.
      defines: new Map([['MB_SAMPLES', Math.max(2, samples | 0).toFixed(0)]]),
      uniforms: new Map([
        ['uMbInvViewProj', new Uniform(new Matrix4())],
        ['uMbPrevViewProj', new Uniform(new Matrix4())],
        // x = shutter scale, y = max UV displacement
        ['uMbParams', new Uniform(new Vector2(MOTION_SHUTTER, MOTION_MAX_UV))],
      ]),
    });

    this.camera = camera;
    this._viewProj = new Matrix4();
    this._prevViewProj = new Matrix4();
    this._primed = false;
  }

  set mainCamera(value) {
    this.camera = value;
  }

  /**
   * Runs from inside `EffectPass.render()`, which is *after* the RenderPass has
   * drawn the scene. That matters: `camera.matrixWorldInverse` is only brought
   * up to date by `renderer.render()`, so sampling it any earlier in the frame
   * would reproject against the previous frame's view matrix and blur a still
   * camera.
   */
  update(renderer, inputBuffer, deltaTime) {
    const camera = this.camera;
    this._viewProj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);

    // A tab switch, a GC pause or a teleport produces a delta the reprojection
    // cannot describe. Re-priming costs one unblurred frame and avoids a smear
    // across the whole screen.
    const discontinuity = !(deltaTime > 0) || deltaTime > MOTION_RESET_DT;
    if (!this._primed || discontinuity) {
      this._prevViewProj.copy(this._viewProj);
      this._primed = true;
    }

    this.uniforms.get('uMbInvViewProj').value.copy(this._viewProj).invert();
    this.uniforms.get('uMbPrevViewProj').value.copy(this._prevViewProj);
    this._prevViewProj.copy(this._viewProj);
  }
}

// ===========================================================================
// SunShaftsEffect - GodRaysEffect that can actually be switched off.
//
// The stock effect has no gain uniform, and `EffectPass` calls `update()` on
// every effect it owns whether or not its blend function is SKIP. That would
// leave a full-resolution occlusion copy, a light render and a radial blur
// running all night. Overriding `update()` to bail at zero intensity keeps the
// effect merged into the main pass - no extra fullscreen blend - while costing
// literally nothing whenever the sun is down, behind the player, or fully
// socked in.
// ===========================================================================

const GLSL_SHAFTS = `
#ifdef FRAMEBUFFER_PRECISION_HIGH
uniform mediump sampler2D map;
#else
uniform lowp sampler2D map;
#endif
uniform float uShaftIntensity;
uniform vec3 uShaftTint;

void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
  outputColor = vec4(texture2D(map, uv).rgb * uShaftTint * uShaftIntensity, inputColor.a);
}
`;

class SunShaftsEffect extends GodRaysEffect {
  constructor(camera, lightSource, options) {
    super(camera, lightSource, options);
    this.name = 'SakuraSunShafts'; // the inherited name would say GodRaysEffect
    this._intensity = 0;
    this.uniforms.set('uShaftIntensity', new Uniform(0));
    this.uniforms.set('uShaftTint', new Uniform(new Color(1, 1, 1)));
    this.setFragmentShader(GLSL_SHAFTS);
  }

  get intensity() {
    return this._intensity;
  }

  set intensity(value) {
    this._intensity = value > 0 ? value : 0;
    this.uniforms.get('uShaftIntensity').value = this._intensity;
  }

  /** @type {Color} hue-only tint; brightness lives in `intensity`. */
  get tint() {
    return this.uniforms.get('uShaftTint').value;
  }

  update(renderer, inputBuffer, deltaTime) {
    // Zero intensity means the ADD blend contributes nothing, so skipping the
    // three off-screen renders is free rather than merely cheap.
    if (this._intensity <= 1e-4) return;
    // GodRaysEffect reparents the light into its own scene and only puts it back
    // `if (parent !== null)`. Running it on a detached mesh therefore swallows
    // celestial's sun disc permanently - it would vanish from the sky for the
    // rest of the session. Cheaper to skip a frame of shafts than to lose the sun.
    const light = this.lightSource;
    if (!light || light.parent === null || !light.material) return;
    super.update(renderer, inputBuffer, deltaTime);
  }
}

// ===========================================================================
// GatedBloomEffect - the same trick for bloom's luminance + mipmap chain.
// ===========================================================================

class GatedBloomEffect extends BloomEffect {
  update(renderer, inputBuffer, deltaTime) {
    if (this.intensity <= 1e-4) return;
    super.update(renderer, inputBuffer, deltaTime);
  }
}

// ===========================================================================
// PostPipeline
// ===========================================================================

export class PostPipeline {
  constructor(ctx) {
    this.ctx = ctx;
    this.renderer = ctx.renderer;
    this.scene = ctx.scene;
    this.camera = ctx.camera;
    this.state = ctx.state;
    this.quality = ctx.quality || null;

    this.composer = null;
    this.renderPass = null;
    this.aoPass = null;
    /** Effect passes this module created and is responsible for disposing. */
    this._passes = [];

    this.effects = {
      grade: null, grain: null, bloom: null, shafts: null,
      dof: null, motionBlur: null, aberration: null, aa: null,
    };

    // --- god rays ----------------------------------------------------------
    // sky/celestial.js and weather/atmosfx.js both call registerGodRayLight()
    // from their link(), which main.js runs before ours. First valid
    // registration wins: atmosfx deliberately hands us celestial's own sun disc
    // when there is one, so the second call is normally the same object.
    this.godRayLight = null;

    // --- exposure ----------------------------------------------------------
    this._authoredExposure = -1;
    this._exposure = 1;
    this._exposurePrimed = false;
    /** Artistic trim applied on top of whichever exposure source is live. */
    this.exposureBias = 0.70;

    // --- resolution --------------------------------------------------------
    this._width = 0;
    this._height = 0;
    this._basePixelRatio = 1;
    this._appliedScale = -1;
    this._offResolution = null;

    // --- configuration signature (polled per frame, allocation-free) --------
    this._tier = null;
    this._featureMask = -1;
    this._aa = '';
    /** Tier whose full stack threw during construction; null while healthy. */
    this._degradedTier = null;

    // --- per-frame damped values -------------------------------------------
    this._shaftLevel = 0;
    this._focusDistance = 14;
    this._grainSeed = 0;

    this._failed = false;
    this._verifiedFrames = 0;
    this._systems = null;
    this._tree = null;
  }

  // =========================================================================
  // Lifecycle
  // =========================================================================

  async init() {
    if (!this.renderer) return;

    // HalfFloat: the scene arrives as scene-referred linear radiance and the sun
    // disc alone reaches ~260. UnsignedByte would clip it to 1.0 and leave bloom
    // and the shafts with nothing to find. FloatType would work too and costs
    // twice the bandwidth on a chip that has none to spare.
    this.composer = new EffectComposer(this.renderer, {
      frameBufferType: HalfFloatType,
      depthBuffer: true,
      stencilBuffer: false,
      multisampling: 0, // MSAA is pure fill rate; SMAA/FXAA does this job
    });

    this.renderPass = new RenderPass(this.scene, this.camera);
    this.composer.addPass(this.renderPass);

    this.renderer.getSize(_size);
    this._width = Math.max(1, Math.round(_size.x));
    this._height = Math.max(1, Math.round(_size.y));
    const scale = this._targetScale();
    this._basePixelRatio = Math.max(0.1, this.renderer.getPixelRatio() / scale);
    this._appliedScale = scale;

    // Adaptive resolution is announced rather than polled by preference; we do
    // both, because the HUD slider writes state.quality.resolutionScale directly.
    if (this.quality && typeof this.quality.onResolutionChange === 'function') {
      this._offResolution = this.quality.onResolutionChange(() => this._applyResolution(false));
    }
  }

  link(systems) {
    this._systems = systems || null;
    this._tree = (systems && systems.tree) || null;
    // Everything downstream of the render pass is built here rather than in
    // init(), because registerGodRayLight() is delivered between the two and the
    // shafts effect has to be able to join the merged pass.
    this._syncConfiguration();
  }

  /**
   * CONTRACTS §6. Called by sky/celestial.js - and defensively by
   * weather/atmosfx.js - with the mesh that stands in for the sun.
   * @param {import('three').Object3D} object3D
   */
  registerGodRayLight(object3D) {
    if (!object3D || !object3D.isObject3D || !object3D.material) return;
    // A second, different light would mean two suns in the shaft mask. Keep the
    // first: that is celestial's real disc, and it is the one the sky draws.
    if (this.godRayLight) return;
    this.godRayLight = object3D;
    // Picked up by the per-frame configuration poll; no rebuild flag needed.
  }

  /**
   * sky/daynight.js publishes its art-directed exposure curve here every frame.
   * @param {number} value linear exposure multiplier
   */
  setExposure(value) {
    if (typeof value !== 'number' || !isFinite(value) || value <= 0) return;
    this._authoredExposure = value;
  }

  /**
   * ui/hud.js toggles individual effects without going through a tier change.
   *
   * `state.quality` is `@owner core/quality.js`, so this method deliberately does
   * NOT write it - the caller owns that write (ui/hud.js sets `q[key]` itself
   * immediately before calling here). All this does is re-read the flags now
   * instead of at the top of the next frame, so the toggle lands in the frame the
   * player clicked in.
   * @param {string} key one of ao|bloom|godRays|depthOfField|motionBlur
   * @param {boolean} value ignored; the authoritative value is state.quality[key]
   */
  setEffectEnabled(key, value) {
    if (
      key !== 'ao' && key !== 'bloom' && key !== 'godRays' &&
      key !== 'depthOfField' && key !== 'motionBlur'
    ) {
      return;
    }
    this._syncConfiguration();
  }

  resize(width, height) {
    this._width = Math.max(1, Math.round(width));
    this._height = Math.max(1, Math.round(height));
    if (!this.composer) return;
    // core/engine.js resizes first (main.js order) and has already folded the
    // device pixel ratio, its own ceiling and the current render scale into
    // renderer.setPixelRatio. Recover its base so a later resolutionScale change
    // can be applied without duplicating engine's DPR policy here.
    this._basePixelRatio = Math.max(0.1, this.renderer.getPixelRatio() / this._targetScale());
    this._applyResolution(true);
  }

  onQualityChange() {
    // Nothing to do: render() polls the tier and the effect flags every frame,
    // and rebuilding render targets in the middle of main.js's twenty-system
    // event fan-out is how a tier switch turns into a two second stall.
  }

  dispose() {
    if (this._offResolution) {
      this._offResolution();
      this._offResolution = null;
    }
    this._destroyPasses();
    this._destroyAOPass();
    if (this.composer) {
      this.composer.dispose();
      this.composer = null;
    }
    this.renderPass = null;
    this.godRayLight = null;
    this._systems = null;
    this._tree = null;
  }

  // =========================================================================
  // Frame
  // =========================================================================

  /**
   * Emergency path: draw the scene straight to the canvas.
   *
   * Engine's ACES tone mapping and sRGB output are still configured on the
   * renderer, so this is a correct - if ungraded - image. Two things have to be
   * done by hand that the composer normally does for us:
   *   - `EffectComposer.setRenderer()` sets `renderer.autoClear = false`, so a
   *     bare `renderer.render()` would composite this frame on top of the last
   *     one and smear. The composer's ClearPass is what usually covers that.
   *   - a composer failure can leave an off-screen target bound.
   */
  _fallbackRender() {
    const renderer = this.renderer;
    renderer.setRenderTarget(null);
    if (renderer.autoClear === false) renderer.clear();
    renderer.render(this.scene, this.camera);
  }

  render(dt, state) {
    const renderer = this.renderer;
    if (!renderer) return;

    if (!this.composer || this._failed) {
      this._fallbackRender();
      return;
    }

    this._syncConfiguration();
    this._applyResolution(false);

    const step = clamp(dt || 0, 0, 0.1);

    this._updateExposure(step);
    this._updateGrade(step, state);
    this._updateBloom(state);
    this._updateShafts(step, state);
    this._updateDoF(step, state);
    this._updateGrain(state);

    // The first frames are the ones that can trip over a driver quirk in the
    // depth blit or a shader that will not link. Guard those, then leave the hot
    // path clean - a permanent try/catch around the only draw call in the frame
    // is not something to leave lying around.
    if (this._verifiedFrames < 3) {
      try {
        this.composer.render(step);
        this._verifiedFrames++;
      } catch (err) {
        this._failed = true;
        console.error('[PostPipeline] composer failed; falling back to a direct render.', err);
        this._fallbackRender();
      }
    } else {
      this.composer.render(step);
    }
  }

  // =========================================================================
  // Resolution
  // =========================================================================

  _targetScale() {
    return clamp(this.state.quality.resolutionScale || 1, 0.4, 1);
  }

  /**
   * Applies state.quality.resolutionScale by moving the renderer's pixel ratio
   * and resizing every render target in the stack. Reallocating targets is
   * expensive, so this is strictly edge-triggered: the per-frame call is one
   * float comparison and returns immediately in the common case.
   */
  _applyResolution(force) {
    if (!this.composer || this._width < 1) return;
    const scale = this._targetScale();
    if (!force && Math.abs(scale - this._appliedScale) < 1e-4) return;

    const ratio = this._basePixelRatio * scale;
    // three's setPixelRatio re-runs setSize internally, so the drawing buffer is
    // already the new size by the time the composer reads it back.
    if (Math.abs(this.renderer.getPixelRatio() - ratio) > 1e-4) {
      this.renderer.setPixelRatio(ratio);
    }
    this.composer.setSize(this._width, this._height, false);
    this._appliedScale = scale;
  }

  // =========================================================================
  // Configuration
  // =========================================================================

  _tierConfig() {
    return TIERS[this.state.quality.tier] || TIERS[QUALITY.HIGH];
  }

  /** Normalised antialiasing mode. Anything unrecognised means "off". */
  _resolveAA() {
    const a = this.state.quality.antialias;
    if (a === 'smaa') return 'smaa';
    if (a === 'fxaa') return 'fxaa';
    return 'none';
  }

  /**
   * A single integer that changes whenever the *shape* of the stack changes.
   * Comparing it allocates nothing, so render() can poll it every frame and pick
   * up a HUD toggle, a tier switch or a late god-ray registration without any of
   * them having to remember to notify us.
   */
  _featureSignature(cfg) {
    const q = this.state.quality;
    return (
      (q.ao && cfg.allowAO ? 1 : 0) |
      (q.godRays && cfg.allowShafts && this.godRayLight ? 2 : 0) |
      (q.depthOfField && cfg.allowDoF ? 4 : 0) |
      (q.motionBlur && cfg.allowMotionBlur ? 8 : 0) |
      // Bloom is part of the SHAPE of the stack, not just a uniform. Leaving a
      // disabled BloomEffect merged in costs a full-resolution texture fetch and
      // an ADD blend on every pixel of every frame, and ui/hud.js switches bloom
      // off for the whole LOW tier - the one tier with no fill rate to waste.
      // One recompile on a toggle the player makes at most a few times a session
      // is the cheaper side of that trade.
      (q.bloom !== false ? 16 : 0) |
      // Chromatic aberration is zero-offset on LOW, where it is exactly an
      // identity function bought with two extra full-res taps and a varying.
      (cfg.aberration > 0 ? 32 : 0)
    );
  }

  _syncConfiguration() {
    if (!this.composer) return;
    const q = this.state.quality;
    const cfg = this._tierConfig();
    // A tier the stack already failed to build on stays pinned to the minimal
    // shape; anything else would recompile-and-throw once per frame forever.
    // A tier change is the one signal worth re-trying the full stack on.
    if (this._degradedTier !== null && this._degradedTier !== q.tier) {
      this._degradedTier = null;
    }
    const degraded = this._degradedTier !== null;
    const mask = degraded ? 0 : this._featureSignature(cfg);
    const aa = degraded ? 'none' : this._resolveAA();
    if (
      this._passes.length > 0 &&
      this._tier === q.tier &&
      this._featureMask === mask &&
      this._aa === aa
    ) {
      return;
    }
    // Recorded before the build so a failure part way through cannot put us in
    // a loop that recompiles every frame.
    this._tier = q.tier;
    this._featureMask = mask;
    this._aa = aa;

    // `_rebuild` constructs library effects, and any of them can throw: an
    // EffectPass that cannot merge, an SMAA path that needs a DOM, a driver that
    // refuses another render target. Unguarded, that exception escapes render()
    // - which main.js does not catch - every single frame, and the player gets a
    // black screen plus a console flood. Retry once with the smallest stack that
    // can possibly work, then give up and take the direct-render path.
    try {
      this._rebuild(cfg, mask, aa);
    } catch (err) {
      console.error('[PostPipeline] stack rebuild failed; retrying minimal.', err);
      this._degradedTier = q.tier;
      this._featureMask = 0;
      this._aa = 'none';
      try {
        // Effects constructed before the throw are not owned by any pass yet, so
        // `_destroyPasses` cannot reach them; each one holds render targets.
        this._disposeOrphanEffects();
        this._destroyPasses();
        this._rebuild(cfg, 0, 'none');
      } catch (err2) {
        console.error('[PostPipeline] minimal stack failed too; post is off.', err2);
        this._disposeOrphanEffects();
        this._destroyPasses();
        this._failed = true;
      }
    }
  }

  /**
   * Disposes anything still sitting in the effect registry. Only used on the
   * rebuild failure path: `_destroyPasses` disposes effects through the pass that
   * owns them, and a half-built stack has effects that never reached a pass.
   * Double-disposing an effect that did reach one is harmless - three's
   * deallocate hooks are no-ops once the properties entry is gone.
   */
  _disposeOrphanEffects() {
    const fx = this.effects;
    for (const key in fx) {
      const effect = fx[key];
      if (effect && typeof effect.dispose === 'function') {
        try {
          effect.dispose();
        } catch (err) {
          console.warn('[PostPipeline] effect dispose threw:', err);
        }
      }
      fx[key] = null;
    }
  }

  _rebuild(cfg, mask, aa) {
    this._destroyPasses();

    // n8ao 1.10.3 is disabled unconditionally - see AO_DISABLED_REASON below.
    const wantAO = (mask & 1) !== 0 && !AO_DISABLED;
    const wantShafts = (mask & 2) !== 0;
    const wantDoF = (mask & 4) !== 0;
    const wantMotionBlur = (mask & 8) !== 0;
    const wantBloom = (mask & 16) !== 0;
    const wantAberration = (mask & 32) !== 0;

    // --- ambient occlusion, before anything that reads the lit colour -------
    if (wantAO) {
      this._ensureAOPass();
      this._configureAOPass(cfg);
      this.aoPass.enabled = true;
      this.composer.addPass(this.aoPass);
    } else {
      // Fully released rather than merely disabled: it owns six render targets
      // and the tiers that switch it off are the ones short of memory.
      this._destroyAOPass();
    }

    // --- the merged pass ---------------------------------------------------
    const merged = [];

    // CONVOLUTION, so postprocessing sorts it first - which is also where a lens
    // aberration physically belongs, ahead of everything the sensor does. Omitted
    // entirely when the tier offset is zero: at offset 0 the effect is an
    // identity function that still pays for two extra full-resolution taps, a
    // vertex shader and two varyings.
    if (wantAberration) {
      const aberration = new ChromaticAberrationEffect({
        offset: new Vector2(cfg.aberration, cfg.aberration * 0.62),
        radialModulation: true,
        // Nothing happens until 45% of the way to the corner: a whisper at the
        // frame edges, not a lens-flare filter over the whole image.
        modulationOffset: 0.45,
      });
      this.effects.aberration = aberration;
      merged.push(aberration);
    }

    if (wantShafts) {
      const shafts = new SunShaftsEffect(this.camera, this.godRayLight, {
        blendFunction: BlendFunction.ADD, // scattered light adds, it does not screen
        samples: cfg.shaftSamples,
        density: 0.93,
        decay: 0.92,
        weight: 0.32,
        exposure: 0.5,
        // The mask holds the raw HDR sun (~260 at noon), so the accumulator
        // saturates immediately. Clamping at 1.0 keeps the radial blur inside a
        // well-conditioned range and hands artistic control to the
        // exposure-compensated gain in _updateShafts().
        clampMax: 1.0,
        blur: true,
        kernelSize: cfg.shaftKernel,
        resolutionScale: cfg.shaftScale,
      });
      this.effects.shafts = shafts;
      merged.push(shafts);
    }

    if (wantDoF) {
      const dof = new DepthOfFieldEffect(this.camera, {
        focusDistance: this._focusDistance,
        focusRange: 8,
        bokehScale: 2.0,
        resolutionScale: 0.5,
      });
      this.effects.dof = dof;
      merged.push(dof);
    }

    if (wantMotionBlur) {
      const mb = new CameraMotionBlurEffect(this.camera, cfg.mbSamples);
      this.effects.motionBlur = mb;
      merged.push(mb);
    }

    // Bloom is added in LINEAR light, before the tone curve - the only place a
    // physically plausible bloom can go. Adding it after tone mapping is what
    // produces the milky, contrast-free glow that reads as cheap.
    if (wantBloom) {
      const bloom = new GatedBloomEffect({
        blendFunction: BlendFunction.ADD,
        mipmapBlur: true,
        luminanceThreshold: BLOOM_DISPLAY_THRESHOLD,
        luminanceSmoothing: BLOOM_KNEE,
        intensity: cfg.bloomIntensity,
        radius: cfg.bloomRadius,
        levels: cfg.bloomLevels,
      });
      // The threshold pass is full resolution by default. Half res costs a
      // quarter of the fill, matches the first mip of the blur chain exactly,
      // and averages single-pixel fireflies away before they can flicker.
      bloom.luminancePass.resolution.scale = cfg.bloomLuminanceScale;
      this.effects.bloom = bloom;
      merged.push(bloom);
    }

    const grade = new FilmGradeEffect();
    this.effects.grade = grade;
    merged.push(grade);

    const grain = new FilmGrainEffect();
    this.effects.grain = grain;

    // --- antialiasing ------------------------------------------------------
    // Grain has to land after AA (SMAA would read it as edge detail and smear
    // it), so when there is no AA pass it joins the main pass instead.
    let aaEffect = null;
    if (aa === 'smaa') {
      aaEffect = new SMAAEffect({
        preset: cfg.smaaPreset,
        // The grade hands SMAA sRGB-encoded values, which is exactly the space
        // its colour thresholds were tuned in.
        edgeDetectionMode: EdgeDetectionMode.COLOR,
      });
    } else if (aa === 'fxaa') {
      aaEffect = new FXAAEffect();
    }
    this.effects.aa = aaEffect;

    if (aaEffect === null) merged.push(grain);

    const mainPass = new EffectPass(this.camera, ...merged);
    // FilmGradeEffect ends in sRGB, so three's colorspace_fragment must not run
    // on top of it. See the colour-ownership note at the top of this file.
    mainPass.encodeOutput = false;
    mainPass.dithering = false; // FilmGrainEffect carries its own TPDF dither
    this._addPass(mainPass);

    if (aaEffect !== null) {
      // SMAA is CONVOLUTION|DEPTH and therefore sorts ahead of the grain; FXAA
      // is NONE and keeps registration order. Either way: antialias, then grain.
      const aaPass = new EffectPass(this.camera, aaEffect, grain);
      aaPass.encodeOutput = false;
      aaPass.dithering = false;
      this._addPass(aaPass);
    }

    // `addPass` already sizes each pass, but adding the first depth-consuming
    // pass makes the composer mint a fresh depth attachment at its default size.
    // One explicit resize settles every target - including that one - in a
    // single place, and it costs nothing on an unchanged size because three's
    // render targets early-out.
    if (this._width > 0) this.composer.setSize(this._width, this._height, false);

    // Re-arm the try/catch in render(). A tier switch or a HUD toggle produces a
    // stack this build has never drawn with, and that is exactly as likely to
    // trip over a driver quirk as the first three frames of the session were.
    this._verifiedFrames = 0;
  }

  _addPass(pass) {
    this._passes.push(pass);
    this.composer.addPass(pass);
  }

  /** Removes and disposes every effect pass we own. The composer, the render
   *  pass and the AO pass survive. */
  _destroyPasses() {
    if (!this.composer) return;
    for (let i = 0; i < this._passes.length; i++) {
      const pass = this._passes[i];
      this.composer.removePass(pass);
      pass.dispose(); // also disposes the effects it owns
    }
    this._passes.length = 0;
    if (this.aoPass) this.composer.removePass(this.aoPass);

    const fx = this.effects;
    fx.grade = null;
    fx.grain = null;
    fx.bloom = null;
    fx.shafts = null;
    fx.dof = null;
    fx.motionBlur = null;
    fx.aberration = null;
    fx.aa = null;
  }

  // =========================================================================
  // Ambient occlusion
  // =========================================================================

  _ensureAOPass() {
    if (this.aoPass) return;
    this.renderer.getDrawingBufferSize(_size);
    const pass = new N8AOPostPass(
      this.scene,
      this.camera,
      Math.max(1, Math.round(_size.x)),
      Math.max(1, Math.round(_size.y))
    );

    // n8ao traverses the WHOLE SCENE every frame looking for a transparent
    // material, and the moment it finds one it silently switches on a mode that
    // renders the scene twice more at full resolution. In a world with rain,
    // petals, a sky dome and a canopy that fires on frame one. Turning the
    // sniffing off is the difference between ~1.2 ms and ~6 ms.
    pass.autoDetectTransparency = false;
    pass.configuration.transparencyAware = false;

    this.aoPass = pass;
  }

  _configureAOPass(cfg) {
    const c = this.aoPass.configuration;
    // Writing through n8ao's proxy triggers a shader reconfiguration, so only
    // assign what actually differs.
    if (c.halfRes !== cfg.aoHalfRes) c.halfRes = cfg.aoHalfRes;
    if (c.aoSamples !== cfg.aoSamples) c.aoSamples = cfg.aoSamples;
    if (c.denoiseSamples !== cfg.aoDenoiseSamples) c.denoiseSamples = cfg.aoDenoiseSamples;
    if (c.denoiseIterations !== cfg.aoDenoiseIterations) {
      c.denoiseIterations = cfg.aoDenoiseIterations;
    }
    if (c.denoiseRadius !== AO_DENOISE_RADIUS) c.denoiseRadius = AO_DENOISE_RADIUS;
    if (c.aoRadius !== AO_RADIUS) c.aoRadius = AO_RADIUS;
    if (c.distanceFalloff !== 1.0) c.distanceFalloff = 1.0;
    if (c.intensity !== AO_POWER) c.intensity = AO_POWER;
    if (c.screenSpaceRadius !== false) c.screenSpaceRadius = false;
    if (c.depthAwareUpsampling !== true) c.depthAwareUpsampling = true;
    if (c.colorMultiply !== true) c.colorMultiply = true;
    // Accumulation reuses AO across frames whenever the camera is still. The
    // grass never is, so it would smear; this stays off on purpose.
    if (c.accumulate !== false) c.accumulate = false;
    if (!c.color.equals(AO_TINT)) c.color = AO_TINT.clone();
    // The AO result is linear and feeds further linear passes, so the gamma
    // correction n8ao would apply if it were the last pass must stay off.
    this.aoPass.autosetGamma = false;
    if (c.gammaCorrection !== false) c.gammaCorrection = false;
  }

  _destroyAOPass() {
    const pass = this.aoPass;
    if (!pass) return;
    if (this.composer) this.composer.removePass(pass);
    // Pass.dispose() does a shallow sweep for render targets, materials and
    // textures, which catches n8ao's targets but not the materials hanging off
    // its FullScreenTriangle helpers (and its geometry is shared, so those
    // helpers must not be disposed wholesale).
    const quads = [
      'copyQuad', 'accumulationQuad', 'depthDownsampleQuad',
      'effectShaderQuad', 'poissonBlurQuad', 'effectCompositerQuad',
    ];
    for (let i = 0; i < quads.length; i++) {
      const quad = pass[quads[i]];
      if (quad && quad.material && typeof quad.material.dispose === 'function') {
        quad.material.dispose();
      }
    }
    pass.dispose();
    this.aoPass = null;
  }

  // =========================================================================
  // Per-frame parameter updates
  // =========================================================================

  _updateExposure(dt) {
    // Prefer the day/night curve: it is the deliberate, art-directed one and it
    // is already temporally adapted. Engine's photometric exposure is the
    // fallback for a build where DayNightCycle never reached us.
    const source =
      this._authoredExposure > 0
        ? this._authoredExposure
        : num(this.renderer.toneMappingExposure, 1) || 1;

    const target = clamp(source * num(this.exposureBias, 1), 0.05, 8);
    if (!this._exposurePrimed) {
      this._exposure = target;
      this._exposurePrimed = true;
    } else {
      // ~40 ms. Not for looks - either source can step discontinuously when the
      // player scrubs time or the weather machine flips - but short enough that
      // it never reads as a second, laggy auto-exposure fighting the first.
      this._exposure = dampTo(this._exposure, target, 25, dt);
    }
  }

  _updateGrade(dt, state) {
    const grade = this.effects.grade;
    if (!grade) return;

    // `smoothstep`/`clamp` both pass NaN straight through, and everything below
    // ends up either in a damped accumulator or in a uniform, so one bad frame
    // from celestial would be permanent. Treat a non-finite sun as "just set".
    const sunY = num(state.sun.direction.y, 0);
    // Three overlapping regimes rather than a hard switch, so the grade slides
    // through dawn instead of cutting.
    const night = 1 - smoothstep(-0.09, 0.05, sunY);
    const day = smoothstep(0.09, 0.30, sunY);
    const golden = clamp01(1 - night - day);
    const wsum = night + day + golden + 1e-5;
    const wn = night / wsum;
    const wd = day / wsum;
    const wg = golden / wsum;

    _gain
      .set(0, 0, 0)
      .addScaledVector(GAIN_DAY, wd)
      .addScaledVector(GAIN_GOLDEN, wg)
      .addScaledVector(GAIN_NIGHT, wn);

    _lift
      .set(0, 0, 0)
      .addScaledVector(LIFT_DAY, wd)
      .addScaledVector(LIFT_GOLDEN, wg)
      .addScaledVector(LIFT_NIGHT, wn);

    const coverage = clamp01(num(state.clouds ? state.clouds.coverage : 0, 0));
    const overcast = smoothstep(0.40, 0.95, coverage);
    const wetness = clamp01(num(state.weather ? state.weather.wetness : 0, 0));

    // Flat overcast is exactly the case that arrives here as a near-monochrome
    // pale wash, so it gets the most curve and the most colour back. Night gets
    // less contrast: crushing it turns the blues black instead of blue.
    const contrast = clamp(0.36 + 0.11 * overcast - 0.06 * wn, 0.08, 0.52);
    const saturation = clamp(1.70 + 0.15 * overcast + 0.07 * wn - 0.05 * wetness, 0.85, 2.30);

    const u = grade.uniforms;
    u.get('uExposure').value = this._exposure;
    u.get('uGain').value.copy(_gain);
    u.get('uLift').value.copy(_lift);
    // Damped so a weather transition cannot step the whole frame's contrast in
    // a single frame.
    const c = u.get('uContrast');
    c.value = dampTo(c.value, contrast, 4, dt);
    const s = u.get('uSaturation');
    s.value = dampTo(s.value, saturation, 4, dt);
    u.get('uVignette').value.x = VIGNETTE_BASE + 0.06 * wn;
  }

  _updateBloom(state) {
    const bloom = this.effects.bloom;
    if (!bloom) return;

    // ui/hud.js owns state.quality.bloom (core/quality.js has no slot for it).
    // Undefined means on - the HUD seeds it in its constructor. The effect is
    // normally dropped from the pass entirely when this is false; this branch
    // only covers the single frame between the flag flipping and the rebuild.
    if (state.quality.bloom === false) {
      bloom.intensity = 0;
      return;
    }

    const cfg = this._tierConfig();
    const haze = clamp01(num(state.weather ? state.weather.fogDensity : 0, 0) / HAZE_REFERENCE);
    const wet = clamp01(num(state.weather ? state.weather.wetness : 0, 0));
    // Veiling glare is a real optical effect of moisture in the air and on the
    // front element: a rainy frame genuinely does bloom more.
    bloom.intensity = cfg.bloomIntensity * (1 + 0.45 * haze + 0.3 * wet);

    // The threshold has to live in the same space as the buffer it reads, and
    // that buffer is scene-referred: the same radiance is "blown out" at
    // midnight and "middle grey" at noon. Dividing a display-referred threshold
    // through exposure converts it, which is what stops bloom swallowing the
    // whole sky at noon and vanishing entirely at night.
    const inv = 1 / Math.max(this._exposure, 0.05);
    const lum = bloom.luminanceMaterial;
    lum.threshold = BLOOM_DISPLAY_THRESHOLD * inv;
    lum.smoothing = BLOOM_KNEE * inv;
  }

  _updateShafts(dt, state) {
    const shafts = this.effects.shafts;
    if (!shafts) return;

    const sun = state.sun;
    this.camera.getWorldDirection(_fwd);
    const sunY = num(sun.direction.y, 0);
    const facing = num(_fwd.dot(sun.direction), 0);

    // Screen-space shafts only exist while the source is on screen. Fading over
    // the last stretch of the frustum rather than clipping at its edge is what
    // stops the whole effect popping on as the player turns.
    const onScreen = smoothstep(0.22, 0.70, facing);
    const aboveHorizon = smoothstep(-0.02, 0.10, sunY);
    // Shafts are a low-sun phenomenon: a long path through the atmosphere and a
    // grazing angle through the canopy. Noon keeps a third of the effect.
    const lowSun = 1 - smoothstep(0.18, 0.62, sunY);

    const coverage = clamp01(num(state.clouds ? state.clouds.coverage : 0, 0));
    // Broken cloud is the classic shaft weather: a clear sky has nothing to
    // shape the beam and a fully socked-in sky has no direct beam left at all.
    const broken = smoothstep(0.06, 0.42, coverage) * (1 - smoothstep(0.60, 0.94, coverage));
    const haze = clamp01(num(state.weather ? state.weather.fogDensity : 0, 0) / HAZE_REFERENCE);
    const wet = clamp01(num(state.weather ? state.weather.wetness : 0, 0));
    const aerosol = clamp(0.42 + 0.40 * broken + 0.34 * haze + 0.18 * wet, 0, 1.35);

    const target =
      onScreen * aboveHorizon * clamp01(num(sun.visibility, 0)) *
      (0.34 + 0.66 * lowSun) * aerosol;

    this._shaftLevel = dampTo(this._shaftLevel, clamp01(target), 6, dt);

    // The composite lands in the scene-referred buffer, so a display-referred
    // target has to be divided back through exposure - the same conversion the
    // bloom threshold makes, in the opposite direction.
    shafts.intensity =
      (this._shaftLevel * SHAFT_DISPLAY_PEAK) / Math.max(this._exposure, 0.05);

    // sky/celestial.js has already applied atmospheric extinction to the sun
    // colour, which makes it exactly the right tint for light scattered out of
    // the beam. Normalised so it carries hue only, never brightness.
    const cr = num(sun.color.r, 1);
    const cg = num(sun.color.g, 1);
    const cb = num(sun.color.b, 1);
    const peak = Math.max(cr, cg, cb, 1e-4);
    shafts.tint.setRGB(cr / peak, cg / peak, cb / peak);
  }

  _updateDoF(dt, state) {
    const dof = this.effects.dof;
    if (!dof) return;

    // Focus on the tree with a deliberate near bias: pulling the plane slightly
    // in front of the trunk keeps the nearest grass readable, which is what
    // makes the shot feel like a lens rather than a blur filter.
    let distance = 14;
    const tree = this._tree;
    if (tree) {
      const target = tree.canopyCenter || tree.position;
      if (target && typeof target.x === 'number') {
        _focusTarget.set(num(target.x, 0), num(target.y, 0), num(target.z, 0));
        distance = num(this.camera.position.distanceTo(_focusTarget), 14);
        const radius = num(tree.canopyRadius, 0);
        distance = Math.max(distance - radius * 0.45, 1.5);
      }
    }
    distance = clamp(distance * 0.94, 2, 160);

    // A hard cut in focus distance reads as a rack focus; damp it.
    this._focusDistance = dampTo(this._focusDistance, distance, 3.5, dt);

    const coc = dof.cocMaterial;
    coc.focusDistance = this._focusDistance;
    // Depth of field grows with subject distance. A fixed range would make a
    // distant tree razor thin and a close one entirely sharp.
    coc.focusRange = clamp(this._focusDistance * 0.55, 3, 45);

    // Rain and fog already destroy the far field; stacking bokeh on top of that
    // only costs fill rate.
    const haze = clamp01(num(state.weather ? state.weather.fogDensity : 0, 0) / HAZE_REFERENCE);
    dof.bokehScale = lerp(2.2, 1.2, haze);
  }

  _updateGrain(state) {
    const grain = this.effects.grain;
    if (!grain) return;
    const cfg = this._tierConfig();
    const night = 1 - smoothstep(-0.09, 0.05, num(state.sun.direction.y, 0));

    const u = grain.uniforms.get('uGrain').value;
    // Real emulsion is grainier when it is pushed, and a night frame is always
    // pushed. Night is also the frame most likely to band, so both terms rise.
    u.x = cfg.grain * (1 + 0.85 * night);
    // Bounded, non-repeating per-frame phase. The golden ratio keeps successive
    // frames maximally decorrelated and the modulo keeps the value from ever
    // growing out of float precision.
    this._grainSeed = (this._grainSeed + 0.6180339887) % 1;
    u.y = this._grainSeed * 977;
    u.z = (1 / 255) * (0.9 + 0.5 * night);
  }
}
