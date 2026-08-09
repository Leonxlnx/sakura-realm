/**
 * atmosfx.js - AtmosphericFX: ground mist, height fog and aerial perspective.
 *
 * Two things live here:
 *
 *  1. AERIAL PERSPECTIVE / HEIGHT FOG.  A replacement for three's flat
 *     `FogExp2` that analytically integrates two exponentially-stratified media
 *     along the view ray:
 *       - a low, deepening, superlinear MIST layer that sits in the grass, and
 *       - a tall, saturating HAZE layer that is the real aerial perspective - 
 *         it is what dissolves the treeline into the sky.
 *     Both are Beer-Lambert: transmittance is exp(-tau) with tau the true
 *     analytic optical depth through the height profile, never a distance ramp.
 *     `state.weather.fogDensity` is an artistic dial, NOT a metre^-1
 *     coefficient - see the OPTICAL DEPTH AUDIT in the constructor for the
 *     mapping and for the tau values it produces at 3 / 30 / 300 / 3000 m.
 *
 *     The inscattered colour is taken from the sky itself (via
 *     `atmosphere.getSkyColor`) along three probes - zenith, horizon toward the
 *     sun, horizon away from the sun - so distant geometry converges on exactly
 *     the colour of the sky behind it: warmer toward the sun, bluer away from
 *     it, and with no seam along the horizon. It is applied to the LIT colour,
 *     in linear space, immediately before tone mapping.
 *
 *     THE PROBES GIVE DIRECTION; `state.weather.fogColor` GIVES LEVEL.
 *     `getSkyColor` is the CLEAR-sky model and knows nothing about the cloud
 *     deck, so in rain it hands back a bright noon horizon while the viewer is
 *     looking at a dark stratus base. Inscattering that value makes fog brighter
 *     than its own light source, and a blend of a few per cent is then enough to
 *     lift a wet trunk clean off the top of the tone curve. The probes are
 *     therefore re-exposed onto the radiance weather.js publishes before
 *     anything downstream reads them. See `_updateAtmosphereColors`.
 *
 *     The whole term is energy-conserving by construction:
 *         result = surface * T + inScatter * ( 1 - T ),   T = exp( -tau )
 *     with a CPU-side cap guaranteeing
 *         luminance( inScatter ) <= scatterCeiling * luminance( sky ).
 *     Those two together mean a fogged pixel can never be brighter than the sky
 *     it is dissolving into, in any weather, at any distance.
 *
 *  2. A LIGHT VOLUMETRIC MIST LAYER.  A small instanced stack of terrain-aware
 *     slabs that follow the camera. Density comes from a scrolling procedural
 *     noise field so the mist drifts with the wind instead of sitting still, and
 *     each slab's opacity uses the true slant path through the slab, which is
 *     what makes a handful of sheets read as a volume. Slabs fade out where they
 *     are buried in terrain and thicken where they float clear of it, so the
 *     mist visibly pools in hollows while ridges stand out of it. Each slab
 *     carries its own analytic self-shadow term, so a bank is warm and luminous
 *     along its top and cold blue-grey in its body - that gradient is what sells
 *     a dawn field. The whole stack fades out entirely below a real density, so
 *     a clear midday costs nothing. No fullscreen raymarch - the 780M cannot
 *     afford one.
 *
 * ---------------------------------------------------------------------------
 * INTEGRATING FROM ANOTHER MODULE (terrain / grass / scatter / treeline)
 * ---------------------------------------------------------------------------
 * Easiest: do nothing. Every material in the scene with `fog === true` is
 * adopted automatically and rewritten onto the aerial-perspective model. That
 * covers every material three ships, out of the box.
 *
 * For a custom ShaderMaterial, either:
 *
 *   a) keep the usual `#include <fog_pars_vertex>` / `<fog_vertex>` /
 *      `<fog_pars_fragment>` / `<fog_fragment>` quartet in your shader and set
 *      `fog: true`. They are rewritten for you, automatically. Or
 *
 *   b) use the chunks explicitly:
 *        vertex:    #include <sakura_aerial_pars_vertex>   (top)
 *                   #include <sakura_aerial_vertex>        (after project_vertex)
 *        fragment:  #include <sakura_aerial_pars_fragment> (top)
 *                   #include <sakura_aerial_fragment>      (last; writes gl_FragColor)
 *      and call `systems.atmosfx.applyToMaterial(material)` in your `link()`, or
 *      spread `systems.atmosfx.uniforms` into your `uniforms`. THE UNIFORMS ARE
 *      MANDATORY - a shader that includes the chunk without them throws on its
 *      first upload.
 *
 * The vertex chunk needs a world position. By default it derives one from
 * `mvPosition`, which is exact through instancing, batching, skinning and
 * morphing. If your vertex shader has no `mvPosition`, either
 * `#define AERIAL_WORLDPOS_PROVIDED` and assign `vec3 aerialWorldPos` before the
 * include, or `#define AERIAL_WORLDPOS_FROM_MODEL` to use
 * `modelMatrix * vec4( transformed, 1.0 )`.
 *
 * Opt out with `material.userData.aerial = false` or `object.userData.noAerial = true`.
 * The fragment chunk also exposes `vec3 aerialPerspective( vec3 color, vec3 worldPos )`
 * for use anywhere else in your shader.
 *
 * ---------------------------------------------------------------------------
 * GOD RAYS - read these every frame, after `atmosfx.update()`
 * ---------------------------------------------------------------------------
 *   atmosfx.sunScreenPosition  Vector2 in UV space clamped to [-1,2] - exactly
 *                              what `GodRaysMaterial.screenPosition` expects
 *   atmosfx.sunScreenNDC       Vector2 in [-1,1], pushed outside that range when
 *                              the sun is behind the camera
 *   atmosfx.sunScreenPixels    Vector2 in viewport pixels, y down
 *   atmosfx.sunOnScreen        false when the sun is behind the camera
 *   atmosfx.sunWorldPosition   Vector3, the sun point the projection above used
 *                              (camera-relative, so it never skews as you walk)
 *   atmosfx.sunOcclusion       0..1, clouds and canopy blocking the disc
 *   atmosfx.cloudOcclusion / atmosfx.canopyOcclusion - the two components
 *   atmosfx.godRayIntensity    0..1.6 suggested shaft strength, already damped;
 *                              peaks when the disc grazes the canopy edge
 *   atmosfx.godRayColor        Color, unit luminance, shaft tint
 *   atmosfx.godRayLight        Mesh, safe to hand straight to `GodRaysEffect`
 *   atmosfx.markOccluder(obj)  tag geometry that should block shafts. The root
 *                              is re-tagged on every scan sweep, so children
 *                              streamed in later are picked up automatically.
 *   atmosfx.unmarkOccluder(o)  stop re-tagging a root
 *   atmosfx.occluderLayer      the layer index used for that. The tree, the
 *                              terrain and the scatter root are tagged from
 *                              link(); grass deliberately is not - a field of
 *                              blades would occlude as a solid wall.
 *   atmosfx.mistiness          0..1 how much is actually airborne right now.
 *                              Shafts need something to scatter off; weight the
 *                              effect by this or they appear in clear air.
 *
 * `godRayLight` is handed to `post.registerGodRayLight()` from `link()`, so a
 * pipeline built on postprocessing's `GodRaysEffect` needs nothing else. The
 * screen-space values above exist for a hand-rolled radial-blur pass; they are
 * all recomputed every frame in `update()` and are stable objects - hold the
 * reference, do not re-read it.
 */

import {
  AdditiveBlending,
  BufferAttribute,
  CanvasTexture,
  Color,
  DataTexture,
  DoubleSide,
  FogExp2,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  LinearFilter,
  LinearMipmapLinearFilter,
  Mesh,
  MeshBasicMaterial,
  PlaneGeometry,
  RGBAFormat,
  RepeatWrapping,
  SRGBColorSpace,
  ShaderChunk,
  ShaderMaterial,
  Sphere,
  UnsignedByteType,
  Vector2,
  Vector3,
  Vector4,
} from 'three';

import { TAU, clamp, clamp01, lerp, smoothstep, damp, hash3, makeRNG } from '../core/math.js';
import { EVENTS } from '../core/state.js';

// ---------------------------------------------------------------------------
// Module-scope scratch. Nothing below may allocate inside update().
// ---------------------------------------------------------------------------

const _v3a = new Vector3();
const _v3b = new Vector3();
const _v3c = new Vector3();
const _dirZenith = new Vector3(0, 1, 0);
const _dirSunward = new Vector3();
const _dirAway = new Vector3();
const _camFwd = new Vector3();
const _cA = new Color();
const _cB = new Color();
const _cC = new Color();
const _cTmp = new Color();
/**
 * Sanitised copies of the three sibling-owned colours the palette is built from.
 * They exist so a NaN arriving from weather.js or celestial.js is stopped at the
 * door rather than after it has spread through eight uniforms - see `finite()`.
 */
const _cSun = new Color();
const _cMoon = new Color();
const _cFog = new Color();
const _sunDir = new Vector3();
const _moonDir = new Vector3();

/** Layer index other systems can enable to mark themselves as a shaft occluder. */
const OCCLUDER_LAYER = 11;
/** Mist noise tile, in texels. 256 keeps the one-off bake near 40 ms. */
const NOISE_SIZE = 256;
/** Hard ceiling on instanced mist slabs; the instance buffer is sized for this. */
const MAX_MIST_LAYERS = 8;
/**
 * World size of one noise tile, in metres. Every consumer uses this or an
 * integer divisor of it (24 m for the fine mist field, 96/3 for the second fog
 * tap), so a single wind-drift accumulator wrapped at 96 m stays seamless for
 * all of them and never loses float precision.
 */
const NOISE_TILE = 96;
/**
 * Linear-space dither amplitude at luminance 1. The shader scales it by
 * sqrt(luminance) so that one step of noise costs roughly the same fraction of
 * an 8-bit output LSB everywhere on the sRGB transfer curve - ~1.3/255 across
 * four decades. A flat linear amplitude (what this file used to do) is
 * invisible in the highlights and obvious grain in the shadows.
 */
const DITHER_BASE = 0.01;

const luminance = (c) => c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722;

/**
 * Shaft tint of last resort, pre-normalised to luminance 1 (the raw warm neutral
 * 1.00 / 0.95 / 0.86 has luminance 0.954132, hence the 1.0480... factor). Used
 * only when the sky probe collapses to black, where renormalising by luminance
 * would either explode or yield a literally black tint.
 */
const SHAFT_FALLBACK_R = 1.0 / 0.954132;
const SHAFT_FALLBACK_G = 0.95 / 0.954132;
const SHAFT_FALLBACK_B = 0.86 / 0.954132;

const QUALITY_PRESETS = {
  low: { layers: 2, rings: 18, segments: 44, radius: 72, fogNoiseTaps: 0, noiseAniso: 1 },
  medium: { layers: 3, rings: 26, segments: 60, radius: 98, fogNoiseTaps: 1, noiseAniso: 2 },
  high: { layers: 5, rings: 34, segments: 76, radius: 132, fogNoiseTaps: 2, noiseAniso: 4 },
  ultra: { layers: 7, rings: 44, segments: 96, radius: 176, fogNoiseTaps: 2, noiseAniso: 8 },
};

// ---------------------------------------------------------------------------
// GLSL
// ---------------------------------------------------------------------------

const AERIAL_PARS_VERTEX = /* glsl */ `
#ifdef USE_FOG
#ifndef SAKURA_AERIAL_VERTEX_INCLUDED
#define SAKURA_AERIAL_VERTEX_INCLUDED
	varying vec3 vAerialWorldPos;
	varying float vFogDepth;
	#if !defined( AERIAL_WORLDPOS_PROVIDED ) && !defined( AERIAL_WORLDPOS_FROM_MODEL )
		uniform mat4 uAerialInvView;
	#endif
#endif
#endif
`;

const AERIAL_VERTEX = /* glsl */ `
#ifdef USE_FOG
	#if defined( AERIAL_WORLDPOS_PROVIDED )
		vAerialWorldPos = aerialWorldPos;
		vFogDepth = distance( cameraPosition, aerialWorldPos );
	#elif defined( AERIAL_WORLDPOS_FROM_MODEL )
		vAerialWorldPos = ( modelMatrix * vec4( transformed, 1.0 ) ).xyz;
		vFogDepth = distance( cameraPosition, vAerialWorldPos );
	#else
		// mvPosition already carries instancing, batching, skinning and morphing,
		// and uAerialInvView is camera.matrixWorld, so this is exact for every
		// material three ships without duplicating any of that transform code.
		vAerialWorldPos = ( uAerialInvView * mvPosition ).xyz;
		vFogDepth = - mvPosition.z;
	#endif
#endif
`;

/**
 * Fragment half of the chunk. Built per quality tier, so the LOW tier genuinely
 * emits fewer instructions and zero texture fetches rather than just branching.
 */
function buildAerialParsFragment(fogNoiseTaps) {
  const noiseDecl =
    fogNoiseTaps > 0
      ? `
	uniform sampler2D uAerialNoise;
	uniform vec4 uAerialNoiseOffset;`
      : '';

  let noiseBody = '';
  if (fogNoiseTaps > 0) {
    // Heterogeneous density, evaluated at one point on the ray rather than
    // integrated. Front-weighted (0.4) because the exponential height profile
    // puts most of the mist near the camera for any near-level view.
    noiseBody += `
		vec3 aerialSamplePos = cameraPosition + dir * min( dist, 130.0 ) * 0.4;
		// Explicit LOD. The banks are a low-frequency signal seen at grazing
		// angles; letting the hardware pick would alias them into sparkle.
		float aerialLod = log2( 1.0 + dist * 0.05 );
		vec2 aerialUvA = aerialSamplePos.xz * uAerialParams.w + uAerialNoiseOffset.xy;
		vec2 aerialN = textureLod( uAerialNoise, aerialUvA, aerialLod ).ra;
		float aerialBank = aerialN.x * 0.62 + aerialN.y * 0.38;`;
    if (fogNoiseTaps > 1) {
      noiseBody += `
		// Second tap at 3x frequency on swapped axes: the repeat of one tile can
		// never line up with the repeat of the other, so the tiling never reads.
		vec2 aerialUvB = aerialSamplePos.zx * uAerialParams.w * 3.0 + uAerialNoiseOffset.zw;
		aerialBank = aerialBank * 0.68 + textureLod( uAerialNoise, aerialUvB, aerialLod ).g * 0.32;`;
    }
    noiseBody += `
		odMist *= mix( 0.42, 1.68, aerialSat( aerialBank ) );`;
  }

  return /* glsl */ `
#ifdef USE_FOG
#ifndef SAKURA_AERIAL_FRAGMENT_INCLUDED
#define SAKURA_AERIAL_FRAGMENT_INCLUDED

	varying vec3 vAerialWorldPos;
	varying float vFogDepth;

	uniform vec3 uAerialSunDir;
	uniform vec3 uAerialSunGlow;
	uniform vec3 uAerialMoonDir;
	uniform vec3 uAerialMoonGlow;
	uniform vec3 uAerialZenith;
	uniform vec3 uAerialHorizonSun;
	uniform vec3 uAerialHorizonAway;
	uniform vec3 uAerialGround;
	uniform vec3 uAerialMistColor;
	uniform vec3 uAerialMistSun;
	uniform vec4 uAerialMist;   // x density  y falloff  z base height  w phase g
	uniform vec4 uAerialHaze;   // x density  y falloff  z base height  w sun-glow mix
	uniform vec4 uAerialParams; // x dither base  y max opacity  z start distance  w noise scale${noiseDecl}

	float aerialSat( const in float x ) { return clamp( x, 0.0, 1.0 ); }

	/**
	 * Optical depth through a medium whose density is
	 *     d(h) = density * exp( -k * max( h - base, 0 ) )
	 * along a ray. Note the max(): below the base the profile is CAPPED FLAT, not
	 * extrapolated - an exponential continued downwards would run away to
	 * infinity in a hollow.
	 *
	 * Two segments, because of that cap:
	 *
	 *  - above the base, substituting dh = dirY * ds turns the integral into a
	 *    plain difference of the endpoint densities,
	 *        density * ( e^-k*hStart - e^-k*hEnd ) / ( k * dirY ).
	 *    That exact form matters: the naive  d0 * dist * (1-e^-x)/x  version
	 *    underflows d0 to zero while (1-e^-x)/x overflows the moment the camera
	 *    climbs a few dozen e-folding heights - 0 * inf is a NaN that latches
	 *    into every fogged material in the scene. This one is bounded everywhere.
	 *
	 *  - below the base the density is constant, so that stretch is a plain
	 *    length * density and has to be ADDED. Clamping only the endpoint HEIGHT
	 *    (what this used to do) silently deleted it, so any ray that dipped under
	 *    the base - every long grazing ray across rolling ground, which is
	 *    exactly the mid-ground - came back with too little fog and then jumped
	 *    back to full fog as soon as the ground rose again. That step is the hard
	 *    band across the horizon.
	 *
	 * Both branches are singular at dirY = 0 (a level ray) only in the ratio,
	 * which degrades to a second-order series there.
	 */
	float aerialOpticalDepth( const in float density, const in float falloff, const in float baseY, const in float camY, const in float dirY, const in float dist ) {
		float h0 = camY - baseY;
		float h1 = h0 + dirY * dist;
		float a = max( h0, 0.0 );
		float b = max( h1, 0.0 );
		// Height is linear in path length, so the fraction of the HEIGHT span
		// that lies above the base is also the fraction of the PATH. Written as
		// a select on "does this ray cross the base at all" rather than as a bare
		// ratio, because for a level ray a == b and the ratio degenerates to 0/0.
		float span = max( abs( h0 - h1 ), 1e-6 );
		float above = dist * ( min( h0, h1 ) < 0.0 ? min( abs( a - b ) / span, 1.0 ) : 1.0 );
		float below = dist - above;

		float ea = exp( - falloff * a );
		float x = falloff * ( b - a );
		float odAbove = abs( x ) < 4e-3
			? ea * above * ( 1.0 - 0.5 * x + x * x * ( 1.0 / 6.0 ) )
			: ( ea - exp( - falloff * b ) ) * above / x;

		return clamp( density * ( odAbove + below ), 0.0, 24.0 );
	}

	/**
	 * Henyey-Greenstein, renormalised so the forward peak is exactly 1 - this is
	 * a redistribution weight, never a gain. q * sqrt( q ) rather than
	 * pow( q, 1.5 ): same value, and it drops the exp2/log2 pair that pow costs
	 * on every fogged fragment in the frame.
	 */
	float aerialForwardLobe( const in float cosT, const in float g ) {
		float k = ( 1.0 - g ) * ( 1.0 - g );
		float d = 1.0 + g * g - 2.0 * g * cosT;
		float q = k / max( d, 1e-4 );
		return q * sqrt( q );
	}

	/**
	 * Sky radiance along a direction, rebuilt from three probes the CPU takes
	 * from the real sky model AND re-levelled onto the cloud deck (see
	 * _updateAtmosphereColors). Distant geometry has to converge on this or the
	 * treeline leaves a visible seam against the horizon.
	 *
	 * The zenith blend used pow( y, 0.62 ). sqrt( y ) nudged 28 % toward y tracks
	 * it to better than 0.02 over the whole 0..1 range and costs one sqrt instead
	 * of a pow.
	 */
	vec3 aerialSkyColor( const in vec3 dir, const in float cosSun ) {
		float az = aerialSat( cosSun * 0.5 + 0.5 );
		vec3 horizon = mix( uAerialHorizonAway, uAerialHorizonSun, az * az );
		float y = aerialSat( dir.y );
		float sy = sqrt( y );
		vec3 c = mix( horizon, uAerialZenith, sy + ( y - sy ) * 0.28 );
		return mix( c, uAerialGround, smoothstep( 0.0, 0.4, - dir.y ) * 0.8 );
	}

	/** Interleaved gradient noise. The fog gradient is huge; 8-bit output bands without it. */
	float aerialDither( const in vec2 p ) {
		return fract( 52.9829189 * fract( dot( p, vec2( 0.06711056, 0.00583715 ) ) ) );
	}

	vec3 aerialPerspective( const in vec3 color, const in vec3 worldPos ) {
		vec3 toFrag = worldPos - cameraPosition;
		float dist = length( toFrag );
		vec3 dir = toFrag / max( dist, 1e-4 );

		float odMist = aerialOpticalDepth( uAerialMist.x, uAerialMist.y, uAerialMist.z, cameraPosition.y, dir.y, dist );
		float odHaze = aerialOpticalDepth( uAerialHaze.x, uAerialHaze.y, uAerialHaze.z, cameraPosition.y, dir.y, dist );
		// Near geometry is most of the shaded pixels in a grass field, and at the
		// tuned coefficients this covers everything inside ~6 m of clear air (and
		// well under a metre of rain, where it should not fire). Bail out before
		// the texture fetch and both phase functions. The step this leaves at the
		// cutoff is 1 - exp(-0.002) ~= 0.2 % of one lerp toward the sky, i.e.
		// well under a single 8-bit code value.
		if ( odMist + odHaze < 0.002 ) return color;
${noiseBody}

		float od = odMist + odHaze;

		// ------------------------------------------------------------------
		// ENERGY CONSERVATION. The single most important three lines in the file.
		//
		//     result = surface * T  +  inScatter * ( 1 - T ),   T = exp( -od )
		//
		// The two weights sum to exactly 1, so the result is a convex combination
		// and can never exceed the brighter of the two ends. maxFogOpacity is a
		// FLOOR ON THE TRANSMITTANCE - a promise that some surface colour always
		// survives - and NOT a gain on the in-scatter side. Those two are only
		// the same thing while every other term behaves; the moment the
		// in-scatter carries a gain of its own, an opacity applied as a
		// multiplier stops paying for itself with a matching extinction and the
		// fog starts adding light. That is what turned a 6.5 % blend into a tree
		// rendered brighter than the sky behind it.
		//
		// The remaining half of the invariant lives on the CPU: every in-scatter
		// colour below is capped in _updateAtmosphereColors so that
		//     luminance( inScatter ) <= scatterCeiling * luminance( sky )
		// holds for every direction, which makes the whole term provably unable
		// to brighten a pixel past the sky it is dissolving into.
		// ------------------------------------------------------------------
		float transmittance = max( exp( - od ), 1.0 - uAerialParams.y );
		float inFactor = ( 1.0 - transmittance ) * smoothstep( 0.0, uAerialParams.z, dist );

		float cosSun = dot( dir, uAerialSunDir );
		float lobe = aerialForwardLobe( cosSun, uAerialMist.w );
		float sunward = aerialSat( cosSun * 0.5 + 0.5 );
		float sunward2 = sunward * sunward;

		// Moon lobe, shared by both media. x^12 as three squarings and a multiply
		// rather than pow(): same curve, no transcendental, and it is evaluated on
		// every fogged fragment day and night.
		float moonward = aerialSat( dot( dir, uAerialMoonDir ) * 0.5 + 0.5 );
		float m2 = moonward * moonward;
		float m4 = m2 * m2;
		vec3 moonCol = uAerialMoonGlow * ( m4 * m4 * m4 );

		// Mist is sunlit in the direction you look toward the sun. This is the
		// whole reason dawn mist glows warm while midday mist reads as flat grey.
		// Note there is deliberately no self-shadow term here: the smooth mist is
		// optically thin (sigma * H is ~4e-4 at the default weather) so its own
		// column shadows nothing. Only the structured slabs are dense enough to,
		// and they carry it in their vertex shader.
		//
		// uAerialSunGlow is an ADDITION on top of a full-strength in-scatter, and
		// aerialForwardLobe peaks at exactly 1, so this pair is the one place the
		// term can exceed the sky. Its budget is set on the CPU: the glow is
		// gated by how much direct beam actually reaches the air (zero under a
		// rain deck) and then capped against the headroom left by mistSun.
		vec3 mistCol = mix( uAerialMistColor, uAerialMistSun, aerialSat( sunward2 * sunward + lobe * 0.5 ) );
		mistCol += uAerialSunGlow * lobe + moonCol;

		// Dry aerosol scatters far more broadly than mist droplets do. At dawn the
		// whole sun side of the sky lifts warm; it does not glow in a cone. The
		// tight droplet lobe alone leaves a hot spot with dead air a few degrees
		// off it, which is one of the most recognisable fake-atmosphere tells.
		// sunward^4 costs two multiplies against the pow it replaced upstream.
		vec3 hazeCol = aerialSkyColor( dir, cosSun )
			+ uAerialSunGlow * ( lobe * 0.45 + sunward2 * sunward2 * 0.55 ) * uAerialHaze.w
			+ moonCol;

		// Weight the two inscatter colours by their share of the optical depth,
		// so the near-ground part of a long ray stays mist-coloured. This is a
		// COMPOSITE, not a sum: the weights are odMist/od and odHaze/od and they
		// add to 1, so two media never inscatter twice the light of one.
		vec3 inscatter = ( odMist * mistCol + odHaze * hazeCol ) / max( od, 1e-4 );
		vec3 result = color * ( 1.0 - inFactor ) + inscatter * inFactor;

		// This runs in linear space but the banding it fixes happens in the 8-bit
		// output, where the sRGB transfer curve's slope goes as L^-0.583. Scaling
		// the linear amplitude by sqrt(L) tracks that inverse closely enough that
		// the noise costs a near-constant ~1.3/255 from the deepest shadow to the
		// highlights, instead of vanishing in one and grain-blasting the other.
		float ditherLum = max( dot( result, vec3( 0.2126, 0.7152, 0.0722 ) ), 1e-4 );
		return result + ( aerialDither( gl_FragCoord.xy ) - 0.5 ) * uAerialParams.x * inFactor * sqrt( ditherLum );
	}

#endif
#endif
`;
}

const AERIAL_APPLY_FRAGMENT = /* glsl */ `
#ifdef USE_FOG
	gl_FragColor.rgb = aerialPerspective( gl_FragColor.rgb, vAerialWorldPos );
#endif
`;

const MIST_VERTEX = /* glsl */ `
#include <sakura_aerial_pars_vertex>

attribute float aTerrainY;
attribute float aLayer;

uniform vec4 uMistParams; // x baseY  y slab spacing  z height falloff  w stack height
uniform vec4 uMistLight;  // x column tau  y max(sunDir.y, eps)  z pooling gain  w sun visibility

varying float vLayerDensity;
varying float vLayerPhase;
varying float vSunLight;

void main() {
	vec4 world = modelMatrix * vec4( position, 1.0 );
	float layerY = uMistParams.x + ( aLayer + 0.5 ) * uMistParams.y;
	world.y = layerY;

	float h = max( layerY - uMistParams.x, 0.0 );
	float profile = exp( - h * uMistParams.z );
	float clearance = layerY - aTerrainY;
	// A slab buried in the ground must not draw. That is what lets a hill poke
	// out of the mist instead of the sheet cutting a hard line across it.
	float emerged = smoothstep( -0.05, 1.1, clearance );
	// ...and the converse. The slabs sit at a level world height, so a hollow
	// simply has more mist column beneath it. Weighting for that is what makes
	// the mist visibly POOL: dells fill and milk over, ridges stay clear. Without
	// it a level stack is just a level stack and the terrain reads as flat.
	float pool = 1.0 + uMistLight.z * smoothstep( 0.4, 4.5, clearance );
	// Roll the top of the stack off so the last slab never reads as a ceiling.
	float cap = 1.0 - smoothstep( 0.55, 1.0, h / max( uMistParams.w, 1e-3 ) );

	vLayerDensity = profile * emerged * cap * pool;
	vLayerPhase = aLayer;

	// Self-shadowing. Optical depth from this slab straight up along the sun,
	// through the same exponential profile the whole layer uses:
	//     tau( h ) = sigma * H * e^( -h / H ) / sunDir.y
	// so uMistLight.x carries sigma * H and this is one multiply. A mist bank is
	// warm and luminous along its top and cold blue-grey down in its body, and
	// that single gradient is most of what makes a dawn field read as a
	// photograph rather than as a grey decal laid over the grass.
	vSunLight = uMistLight.w * exp( - min( uMistLight.x * profile / uMistLight.y, 6.0 ) );

	vec3 aerialWorldPos = world.xyz;
	#include <sakura_aerial_vertex>

	gl_Position = projectionMatrix * viewMatrix * world;
}
`;

const MIST_FRAGMENT = /* glsl */ `
#include <sakura_aerial_pars_fragment>

uniform sampler2D uMistNoise;
uniform vec4 uMistScroll; // xy coarse field, zw fine field
uniform vec4 uMistShape;  // x coarse scale  y fine scale  z coverage  w warp metres
uniform vec4 uMistFade;   // x near fade  y far start  z far end  w density gain
uniform vec4 uMistParams;

varying float vLayerDensity;
varying float vLayerPhase;
varying float vSunLight;

void main() {
	if ( vLayerDensity <= 0.002 ) discard;

	vec3 wp = vAerialWorldPos;
	vec3 toEye = wp - cameraPosition;
	float dist = length( toEye );
	vec3 dir = toEye / max( dist, 1e-4 );

	// Radial fades first, before either texture fetch. The near hole and the
	// outer roll-off between them cover a real share of the annulus, and on an
	// iGPU a fetch avoided is worth more than the branch costs.
	float radialFade = smoothstep( uMistFade.x, uMistFade.x * 2.6, dist )
		* ( 1.0 - smoothstep( uMistFade.y, uMistFade.z, dist ) );
	if ( radialFade < 0.004 ) discard;

	// Golden-ratio offset per slab. Without it every slab samples the same field
	// and the stack collapses into one thick sheet with no internal structure.
	vec2 phase = vec2( vLayerPhase * 0.6180339, vLayerPhase * 0.3819660 );

	vec2 uvA = wp.xz * uMistShape.x + uMistScroll.xy + phase;
	vec4 nA = texture2D( uMistNoise, uvA );
	// Warping the second lookup by the first turns two smooth octaves into torn,
	// wispy edges instead of soap-bubble blobs.
	vec2 uvB = ( wp.xz + ( nA.rg - 0.5 ) * uMistShape.w ) * uMistShape.y + uMistScroll.zw - phase.yx;
	vec4 nB = texture2D( uMistNoise, uvB );

	float n = nA.r * 0.55 + nB.g * 0.30 + nB.b * 0.15;
	// Narrow window on purpose: fBm of fBm has a tight distribution, so a wide
	// threshold band would wash every bank edge into an even grey.
	float cover = smoothstep( uMistShape.z - 0.12, uMistShape.z + 0.16, n );
	if ( cover <= 0.001 ) discard;

	// True slant path through the slab. A grazing view cuts through far more of
	// the layer than a top-down one, and that single term is what makes a
	// handful of sheets read as a volume rather than as sheets.
	float path = uMistParams.y / max( abs( dir.y ), 0.10 );
	float alpha = ( 1.0 - exp( - cover * vLayerDensity * uMistFade.w * path ) ) * radialFade;
	if ( alpha < 0.004 ) discard;

	float cosSun = dot( dir, uAerialSunDir );
	// Two ends: shadowed mist is pure sky radiance, sunlit mist carries the sun's
	// own hue at up to ~3x that. vSunLight is the self-shadow transmittance, so
	// the top of a bank crosses to the warm end and its base never does. The
	// directional term on top of it is single-scattering: you see far more of the
	// mist's forward lobe looking into the sun than away from it.
	float towardSun = aerialSat( cosSun * 0.5 + 0.5 );
	// t^2.5 as t*t*sqrt(t), not pow( t, 2.6 ). The two curves differ by at most
	// 0.0144 (at t = 0.676) and the 0.70-wide mix below scales that to 0.0101
	// of lit, which is inside one 8-bit code after the colour mix. This
	// is the highest-overdraw shader in the module - up to seven slabs deep over
	// the same pixel - so a transcendental costs more fill here than the same one
	// does in the aerial chunk, where the file already refuses to pay for it.
	float lit = vSunLight * mix( 0.30, 1.0, towardSun * towardSun * sqrt( towardSun ) );
	vec3 col = mix( uAerialMistColor, uAerialMistSun, aerialSat( lit ) );
	// Thin edges glow more than dense cores when looking toward the sun: less
	// material to scatter through. Cores are marginally brighter in ambient.
	col += uAerialSunGlow * aerialForwardLobe( cosSun, uAerialMist.w ) * vSunLight * ( 0.95 - cover * 0.45 );
	// x^12 as three squarings - exact, and the identical form the aerial chunk
	// already uses for the same lobe. pow() here was evaluated day and night.
	float mw = aerialSat( dot( dir, uAerialMoonDir ) * 0.5 + 0.5 );
	float mw2 = mw * mw;
	float mw4 = mw2 * mw2;
	col += uAerialMoonGlow * ( mw4 * mw4 * mw4 );
	col *= mix( 0.88, 1.05, cover );

	col = aerialPerspective( col, wp );
	// Same luminance-tracked amplitude as the fog chunk. Mist sits at the very
	// bottom of the value range at dawn, which is precisely where a flat linear
	// dither turns into visible grain. Offset coordinate: aerialPerspective()
	// has ALREADY added noise from aerialDither( gl_FragCoord.xy ) a line above,
	// so reusing that exact coordinate would draw the identical sample again and
	// the two would reinforce into one double-amplitude pattern instead of
	// averaging out - worst precisely where the aerial term is strongest.
	float ditherLum = max( dot( col, vec3( 0.2126, 0.7152, 0.0722 ) ), 1e-4 );
	col += ( aerialDither( gl_FragCoord.xy + vec2( 19.0, 43.0 ) ) - 0.5 ) * uAerialParams.x * sqrt( ditherLum );

	// Dithering colour alone under-corrects: after the blend its amplitude is
	// scaled by alpha, so the long low-alpha ramps - the near hole, the outer
	// roll-off, every bank edge - are exactly where it stops working. Dithering
	// alpha instead perturbs by |col - background|, which is the right magnitude
	// there. Decorrelated coordinate so the two do not reinforce into a pattern.
	alpha += ( aerialDither( gl_FragCoord.xy + vec2( 37.0, 71.0 ) ) - 0.5 ) * ( 1.5 / 255.0 );

	gl_FragColor = vec4( col, clamp( alpha, 0.0, 1.0 ) );
}
`;

// ---------------------------------------------------------------------------
// Procedural, tileable mist noise
// ---------------------------------------------------------------------------

/**
 * Periodic value noise. `createNoise` in core/math.js is simplex and therefore
 * cannot tile; a mist texture that does not wrap shows a hard seam every tile,
 * so this lattice variant - built on the shared `hash3` - is the right tool.
 */
function periodicNoise(x, y, period, seed) {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  const u = xf * xf * (3 - 2 * xf);
  const v = yf * yf * (3 - 2 * yf);
  const x0 = ((xi % period) + period) % period;
  const y0 = ((yi % period) + period) % period;
  const x1 = (x0 + 1) % period;
  const y1 = (y0 + 1) % period;
  const a = hash3(x0, y0, seed);
  const b = hash3(x1, y0, seed);
  const c = hash3(x0, y1, seed);
  const d = hash3(x1, y1, seed);
  return lerp(lerp(a, b, u), lerp(c, d, u), v);
}

/** Tileable fBm: each octave's period doubles with its frequency, so it wraps. */
function periodicFbm(u, v, baseCells, octaves, seed, gain) {
  let sum = 0;
  let amp = 1;
  let norm = 0;
  let cells = baseCells;
  for (let o = 0; o < octaves; o++) {
    sum += amp * periodicNoise(u * cells, v * cells, cells, seed + o * 977);
    norm += amp;
    amp *= gain;
    cells *= 2;
  }
  return sum / norm;
}

function createMistNoiseTexture() {
  const size = NOISE_SIZE;
  const count = size * size;
  const float = new Float32Array(count * 4);
  const inv = 1 / size;
  for (let y = 0; y < size; y++) {
    const v = y * inv;
    for (let x = 0; x < size; x++) {
      const u = x * inv;
      const a = periodicFbm(u, v, 4, 4, 101, 0.52);
      const b = periodicFbm(u, v, 7, 4, 6101, 0.5);
      const c = periodicFbm(u, v, 11, 3, 20507, 0.55);

      const i = (y * size + x) * 4;
      // R - bank silhouette. A touch of billow gives the gaps crisp shoulders.
      float[i] = a * 0.72 + (1 - Math.abs(b * 2 - 1)) * 0.28;
      // G - uncorrelated mid field, sampled at a different world scale so the
      //     tile can never line up with itself.
      float[i + 1] = b * 0.85 + c * 0.15;
      // B - ridged wisps: the high-frequency tearing along bank edges.
      float[i + 2] = Math.pow(1 - Math.abs(c * 2 - 1), 1.7);
      // A - smooth, low-frequency field for the distance-fog bank modulation.
      float[i + 3] = a * 0.65 + b * 0.35;
    }
  }

  // Histogram stretch per channel. Summed value noise lands in a narrow band
  // around whatever mean the hash happens to have (measured: 0.32 +/- 0.05), so
  // without this the shader's coverage threshold would be tuned to an accident
  // of the hash rather than to anything meaningful, and 8-bit quantisation
  // would throw away three quarters of the available precision.
  const data = new Uint8Array(count * 4);
  for (let ch = 0; ch < 4; ch++) {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = ch; i < float.length; i += 4) {
      const value = float[i];
      if (value < lo) lo = value;
      if (value > hi) hi = value;
    }
    const scale = hi - lo > 1e-6 ? 255 / (hi - lo) : 0;
    for (let i = ch; i < float.length; i += 4) {
      data[i] = clamp01((float[i] - lo) * scale * (1 / 255)) * 255;
    }
  }
  const tex = new DataTexture(data, size, size, RGBAFormat, UnsignedByteType);
  tex.wrapS = RepeatWrapping;
  tex.wrapT = RepeatWrapping;
  tex.magFilter = LinearFilter;
  tex.minFilter = LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  tex.name = 'atmosfx.mistNoise';
  return tex;
}

// ---------------------------------------------------------------------------

export class AtmosphericFX {
  constructor(ctx) {
    this.ctx = ctx;
    this.scene = ctx.scene;
    this.camera = ctx.camera;
    this.state = ctx.state;
    this.bus = ctx.bus;

    this.occluderLayer = OCCLUDER_LAYER;

    // --- tunables. Everything derived is recomputed from state every frame. --
    /**
     * `state.weather.fogDensity` is an ARTISTIC dial that lives in roughly
     * 0.0011 (clear) .. 0.018 (declared FOG event). It is NOT a metre^-1
     * extinction coefficient: weather.js reads it back as `visibility =
     * 3 / fogDensity`, which would make a nominal partly-cloudy morning a
     * 1.4 km whiteout. Consumed literally it crushed a grass field to flat
     * blue-grey thirty metres from the eye. It is mapped here onto two real,
     * exponentially-stratified media instead - everything below is a genuine
     * per-metre extinction coefficient.
     *
     *   HAZE - tall, SATURATING aerosol column. This is aerial perspective
     *          proper: the thing that dissolves a treeline into the sky.
     *   MIST - low, SUPERLINEAR boundary-layer fog. This is what actually
     *          closes visibility during a FOG or STORM event, and it thins away
     *          within a few metres of the grass on an ordinary morning.
     *
     * Splitting it that way is not a nicety. Pushing a fog event's density into
     * a 1.1 km column would haze the zenith as hard as the horizon, which never
     * happens in nature - fog is a boundary-layer phenomenon and the air above
     * it stays clean.
     *
     * ------------------------------------------------------------------------
     * OPTICAL DEPTH AUDIT
     *
     * Level ray, eye 1.85 m above the mist base, each weather at its typical
     * hour with its own preset wind and cloud cover. sigmaEye is the total
     * extinction the eye sits in; visibility is the meteorological range
     * 3.912 / sigma; the four percentages are the actual blend toward the
     * inscattered sky, ( 1 - e^-tau ) capped by maxFogOpacity 0.965.
     *
     *  weather        sigH     sigM     H(m)  sigmaEye  vis(m)   3 m /  30 m /  300 m / 3000 m
     *  clear 12:00   2.98e-4  3.01e-5   2.6   3.13e-4   12513   0.1% /  0.9% /   8.6% /  58.7%
     *  clear 05:30   2.98e-4  1.33e-4   3.5   3.76e-4   10397   0.1% /  1.1% /  10.3% /  65.3%
     *  partly 08:30  4.20e-4  1.27e-4   3.6   4.95e-4    7900   0.1% /  1.4% /  13.3% /  74.7%
     *  petalStorm    4.99e-4  1.25e-4   4.9   5.84e-4    6704   0.2% /  1.7% /  15.5% /  79.7%
     *  overcast      6.07e-4  3.94e-4   7.3   9.12e-4    4289   0.3% /  2.6% /  23.1% /  90.2%
     *  rain 11:00    1.43e-3  2.02e-3  23.0   3.29e-3    1188   0.9% /  9.1% /  60.6% /  96.5%
     *  storm 11:00   2.31e-3  3.61e-3  43.6   5.76e-3     679   1.7% / 15.3% /  79.4% /  96.5%
     *  FOG 06:30     1.57e-3  1.54e-2  57.7   1.65e-2     237   4.7% / 37.6% /  95.8% /  96.5%
     *
     * Sanity check on each row, because a table of numbers is only worth what it
     * is checked against:
     *  - clear is under 1 % out to 30 m (it MUST be invisible in the near field
     *    or a sunlit meadow goes chalky) and still only 8.6 % at 300 m, with the
     *    whole fade saved for the kilometre scale. 12.5 km of range is a genuine
     *    clean-air day.
     *  - overcast holds 77 % of its surface colour at 300 m: flat and grey, but
     *    the treeline is still a treeline and the field is still green.
     *  - rain sits at 1.2 km of range, which is what an honest moderate rain
     *    measures, and hands back 39 % of the surface at 300 m. Combined with an
     *    in-scatter that is now the DECK's radiance rather than a clear noon
     *    sky's, that is the dark moody frame in g3-rain-noaerial.jpg with depth
     *    added, instead of the white-out in f2-rain.jpg.
     *  - storm is denser than rain everywhere, as it must be, without collapsing:
     *    21 % of surface colour survives at 300 m.
     *  - FOG is genuinely dense - 38 % gone by 30 m, effectively closed at 300 m
     * - and CANNOT go white, because the level it converges on is
     *    weather.js's own fogColor (fogLuma 0.92 through a cloud-dimmed sky),
     *    which lands around 1.5 in render units: a luminous grey, well inside
     *    the tone curve, not a clipped 3.0.
     *
     * The constants at the top of this file's history gave 0.64 %, 6.2 %, 48 %
     * and 99.8 % for a NOMINAL day. A 48 % lerp at only 300 m, toward a sky
     * radiance six to ten times the lit ground's, is what turned every early
     * capture monochrome.
     *
     * Flying at 400 m in the default weather the mist term has vanished entirely
     * and the haze has thinned to e^(-400/1100) of its ground value - the height
     * falloff is real, not decorative.
     * ------------------------------------------------------------------------
     */
    /** state.weather.fogDensity that corresponds to "one unit" of murk. */
    this.fogReference = 0.0022;

    // --- tall haze (aerial perspective) --------------------------------------
    /** Extinction of perfectly clean air, m^-1. ~32 km meteorological range. */
    this.hazeFloor = 0.00012;
    /** Extinction added at units === 1, m^-1. */
    this.hazeGain = 0.0003;
    /** Sub-linear in units: the aerosol column saturates, the ground does not. */
    this.hazeExponent = 0.75;
    /** e-folding height of the haze, in metres. Real tropospheric aerosol. */
    this.hazeHeight = 1100;
    /**
     * How much rain adds to the TALL column. Deliberately large: rain murk is a
     * deep, well-mixed medium, and putting it here instead of in the mist is
     * what keeps a level ray and one three degrees above it within a factor of
     * ~1.3 of each other rather than a factor of five. That ratio is the hard
     * band across the mid-ground.
     */
    this.hazeRainGain = 1.15;

    // --- low mist ------------------------------------------------------------
    /** Fraction of fogDensity that becomes ground mist at units === 1. */
    this.mistGain = 0.1;
    /** How much rain stirs the stratified boundary layer flat. */
    this.mistRainMix = 0.45;
    /** ...and how much it deepens what survives, with lifted spray. */
    this.mistRainDepth = 0.75;
    /** e-folding height of a thin dawn ribbon lying in the grass, metres. */
    this.mistHeightMin = 2.6;
    /** e-folding height of a full fog bank, metres. Real fog is tens deep. */
    this.mistHeightMax = 45;
    /** Live mist e-folding height; interpolated between the two every frame. */
    this._mistHeight = this.mistHeightMin;
    /** Absolute ceiling on the vertical extent of the slab stack, metres. */
    this.mistStackHeight = 26;
    /** Slab stack spans this many e-folding heights of the mist, up to the cap. */
    this.mistStackScale = 2.3;
    /** Below this base extinction the slab stack is pure cost; it fades out. */
    this.mistSlabCutIn = 6e-5;
    /**
     * Slab density relative to the analytic mist term. Well above 1 on purpose:
     * the stack only spans the bottom of the layer, so to read as banks in the
     * grass at a grazing angle it has to run denser than the smooth term it
     * sits on top of. This is the one deliberately non-physical dial here.
     *
     * Peak alpha through the whole stack, in a bank core at a fully grazing
     * view (the worst case; a typical fragment is well under it):
     *   clear midday 0.00 · partly 08:30 0.08 · clear dawn 0.13 ·
     *   partly dawn 0.48 · overcast noon 0.59 · rain / storm / fog 1.00.
     * Because the slab optical depth is (density * slant path) rather than a
     * per-sheet constant, LOW's two layers land within 8 % of HIGH's five - 
     * the stack halves in cost without halving in appearance.
     */
    this.mistStructure = 46;
    /** Extra density where a slab floats well clear of the ground - hollows. */
    this.mistPooling = 0.6;
    /**
     * Floor on the transmittance: fog never fully replaces surface colour, so
     * something of every silhouette survives at any range. Applied in the shader
     * as `max( exp(-od), 1 - maxFogOpacity )`, NOT as a multiplier on the
     * in-scatter weight - see the energy-conservation block in the chunk.
     */
    this.maxFogOpacity = 0.965;
    /**
     * Hard ceiling on the in-scattered radiance, as a multiple of the sky
     * luminance in the same frame. This is the invariant that makes the
     * white-out impossible rather than merely unlikely: no combination of the
     * sunlit-mist colour, the droplet forward lobe and the moon lobe may sum
     * past it, so a fogged pixel can never come out brighter than the sky it is
     * dissolving into. 2.2 leaves real room for the forward-scatter glow of a
     * dawn bank - mist looked into against a low sun IS brighter than the
     * average horizon - while 4.6x, which the old constants reached at dawn, is
     * not a phenomenon, it is a bug.
     *
     * It is measured against the MEAN of the two horizon probes, and the sunward
     * probe already sits above that mean, so 2.2x the mean is only about 1.6x the
     * sky the glow is actually seen against. That is a luminous backlit bank, not
     * a blown one. In the weathers that failed it never binds at all: `beam` is
     * zero under a rain deck, so the sunlit colour has already collapsed onto the
     * neutral mist colour and both lobes are dark before this is reached.
     */
    this.scatterCeiling = 2.2;
    /** 0..1 "how misty does the ground read". Published for post / god rays. */
    this.mistiness = 0;

    // --- shared uniform bag -------------------------------------------------
    // Every adopted material references THESE OBJECTS, so a frame update is a
    // handful of writes no matter how many materials took the chunk.
    this.uniforms = {
      // camera.matrixWorld *is* the inverse view matrix and three keeps it
      // current in place, so this uniform costs literally nothing per frame.
      uAerialInvView: { value: this.camera.matrixWorld },
      uAerialSunDir: { value: new Vector3(0, 1, 0) },
      uAerialSunGlow: { value: new Color(0, 0, 0) },
      uAerialMoonDir: { value: new Vector3(0, -1, 0) },
      uAerialMoonGlow: { value: new Color(0, 0, 0) },
      uAerialZenith: { value: new Color(0.18, 0.36, 0.72) },
      uAerialHorizonSun: { value: new Color(0.72, 0.8, 0.92) },
      uAerialHorizonAway: { value: new Color(0.6, 0.72, 0.9) },
      uAerialGround: { value: new Color(0.2, 0.22, 0.19) },
      uAerialMistColor: { value: new Color(0.72, 0.78, 0.86) },
      uAerialMistSun: { value: new Color(0.9, 0.85, 0.8) },
      uAerialMist: { value: new Vector4(0.00015, 1 / this._mistHeight, 0, 0.62) },
      uAerialHaze: { value: new Vector4(this.hazeFloor, 1 / this.hazeHeight, 0, 0.55) },
      uAerialParams: { value: new Vector4(DITHER_BASE, this.maxFogOpacity, 2.5, 1 / NOISE_TILE) },
      uAerialNoise: { value: null },
      uAerialNoiseOffset: { value: new Vector4() },
    };

    // three's renderer unconditionally refreshes these on any material with
    // `fog === true`. A custom ShaderMaterial without them is an instant
    // TypeError, so we hand them out to everything we adopt.
    this._fogCompat = {
      fogColor: { value: new Color(1, 1, 1) },
      fogDensity: { value: 0.0002 },
      fogNear: { value: 1 },
      fogFar: { value: 2000 },
    };

    // --- god rays -----------------------------------------------------------
    this.sunScreenPosition = new Vector2(0.5, 0.5);
    this.sunScreenNDC = new Vector2();
    this.sunScreenPixels = new Vector2();
    this.sunWorldPosition = new Vector3(0, 1000, 0);
    this.sunOnScreen = false;
    this.sunOcclusion = 0;
    this.cloudOcclusion = 0;
    this.canopyOcclusion = 0;
    this.godRayIntensity = 0;
    this.godRayColor = new Color(1, 0.95, 0.86);
    this.godRayLight = null;
    this.sunProxy = null;

    // --- internal -----------------------------------------------------------
    this._preset = QUALITY_PRESETS[ctx.state.quality.tier] || QUALITY_PRESETS.high;
    this._variant = 0;
    this._patched = new Map(); // material -> original { onBeforeCompile, customProgramCacheKey }
    this._patchBudget = 0;
    this._scanTimer = 0;
    this._warmupTimer = 5;
    this._flash = 0;
    this._lowSun = 0;
    /** 0..1 how completely a cloud deck / fog bank has replaced the clear sky. */
    this._deck = 0;
    /** 0..1 direct sunlight actually reaching the air. Gates every warm term. */
    this._beam = 1;
    /** Factor the clear-sky probes are re-exposed by to land on the real sky. */
    this._skyScale = 1;
    this._mistBase = 0;
    this._mistBaseTarget = 0;
    this._snapX = Infinity;
    this._snapZ = Infinity;
    this._resampleCooldown = 0;
    this._driftA = new Vector2();
    this._driftB = new Vector2();
    this._viewport = new Vector2(1, 1);
    this._noiseTexture = null;
    this._ownsNoiseTexture = false;
    /**
     * Occluder roots, re-tagged on every scan sweep. The tree streams blossom
     * and branch meshes in after link() runs, and a one-shot traverse would
     * leave every one of them transparent to the shaft pass - the exact
     * geometry whose silhouette makes the shafts worth having.
     *
     * An array, not a Set: this is walked from update() on every scan sweep and
     * `for..of` over a Set allocates a fresh iterator object each time. Three
     * entries, so the linear indexOf on add is free.
     */
    this._occluderRoots = [];
    this._mistMesh = null;
    this._mistMaterial = null;
    this._mistGeometry = null;
    this._skyProbeOk = true;
    this._disposed = false;
    this._effectiveMist = 0;
    this._sigmaEyeMist = 0;
    this._slabFade = 0;
    this._samplerOwner = undefined;
    this._terrainSampler = null;
    this._scanCallback = (obj) => this._considerObject(obj);
    this._occluderCallback = (obj) => obj.layers.enable(OCCLUDER_LAYER);

    this._registerChunks();

    // Scene fog is ours. Anything that somehow escapes adoption still gets a
    // consistent, if flat, falloff instead of standing out at full contrast.
    this.fog = new FogExp2(0x000000, 0.0018);
    this.fog.color.copy(this.state.weather.fogColor);
    this.scene.fog = this.fog;
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  async init() {
    // sunScreenPixels is meaningless until the viewport is known, and resize()
    // is not guaranteed to fire before the first frame. Seed it from the
    // renderer's current drawing buffer.
    const renderer = this.ctx.renderer;
    if (renderer && typeof renderer.getSize === 'function') {
      renderer.getSize(this._viewport);
      if (!(this._viewport.x > 0) || !(this._viewport.y > 0)) this._viewport.set(1, 1);
    }

    const textures = this.ctx.textures;
    // Memoised through the shared factory when it supports it, so a hot reload
    // or a second consumer never pays the bake twice.
    if (textures && typeof textures.get === 'function') {
      try {
        const t = textures.get('atmosfx.mistNoise', createMistNoiseTexture);
        if (t && t.isTexture) {
          this._noiseTexture = t;
          this._ownsNoiseTexture = false;
        }
      } catch (err) {
        console.warn('[AtmosphericFX] texture factory refused the mist noise:', err);
      }
    }
    if (!this._noiseTexture) {
      this._noiseTexture = createMistNoiseTexture();
      this._ownsNoiseTexture = true;
    }
    // The factory may hand back something configured for a different consumer.
    this._noiseTexture.wrapS = RepeatWrapping;
    this._noiseTexture.wrapT = RepeatWrapping;
    this._noiseTexture.minFilter = LinearMipmapLinearFilter;
    this._noiseTexture.magFilter = LinearFilter;
    this._noiseTexture.generateMipmaps = true;
    this._applyAnisotropy();
    this._noiseTexture.needsUpdate = true;
    this.uniforms.uAerialNoise.value = this._noiseTexture;

    this._buildMist();
    this._buildSunProxy();
  }

  link(systems) {
    this.systems = systems;
    this.atmosphere = systems.atmosphere || null;
    this.terrain = systems.terrain || null;
    this.tree = systems.tree || null;
    this.post = systems.post || null;

    // The tree is the scene's one interesting shaft occluder - shafts are a
    // partial-occlusion effect and a branching canopy is the ideal edge. The
    // terrain matters too: at dawn the sun sits low enough that a ridge in front
    // of the player must cut the shafts, or they leak straight through the hill.
    // Tagging is additive - layer 0 membership is untouched.
    for (const owner of [this.tree, this.terrain, systems.scatter]) {
      if (!owner) continue;
      const root = owner.root || owner.group || owner.object3D || owner.mesh || null;
      if (root && root.isObject3D) this.markOccluder(root);
    }

    // Prefer whatever sun body celestial already renders: a second visible sun
    // disc would double-draw the brightest object in the frame.
    const celestial = systems.celestial;
    let lightSource = null;
    if (celestial) {
      const candidates = ['sunMesh', 'sunDisc', 'sunSprite', 'sunBody', 'sunObject', 'sun'];
      for (const key of candidates) {
        const candidate = celestial[key];
        if (candidate && candidate.isObject3D && candidate.material) {
          lightSource = candidate;
          break;
        }
      }
      if (!lightSource && typeof celestial.getGodRayLight === 'function') {
        const c = celestial.getGodRayLight();
        if (c && c.isObject3D && c.material) lightSource = c;
      }
    }
    if (!lightSource && this.sunProxy) {
      // Nobody else owns a sun body, so ours becomes the visible one.
      this.sunProxy.visible = true;
      lightSource = this.sunProxy;
    }
    this.godRayLight = lightSource;
    if (this.post && typeof this.post.registerGodRayLight === 'function' && lightSource) {
      try {
        this.post.registerGodRayLight(lightSource);
      } catch (err) {
        console.warn('[AtmosphericFX] registerGodRayLight failed:', err);
      }
    }

    // A strike lights the whole air volume for a few frames. Reading
    // state.weather.lightning alone can miss a spike that rises and decays
    // between two frames, so take the event as well and let both feed the
    // same damped envelope.
    this._unsubLightning = this.bus?.on?.(EVENTS.LIGHTNING_STRIKE, () => {
      this._flash = 1;
    });

    this._resampleTerrain(true);
    this._patchBudget = 4096; // first sweep adopts everything that already exists
    this._scanScene();
  }

  resize(width, height) {
    // Math.max( NaN, 1 ) is NaN, and _viewport is the only scale factor behind
    // the published sunScreenPixels - a single bad resize would leave that
    // NaN for good. Guard, and floor at 1 px for a collapsed/minimised window.
    this._viewport.set(Math.max(finite(width, 1), 1), Math.max(finite(height, 1), 1));
  }

  onQualityChange(quality) {
    const preset = QUALITY_PRESETS[quality?.tier] || QUALITY_PRESETS.high;
    const noiseChanged = preset.fogNoiseTaps !== this._preset.fogNoiseTaps;
    const geometryChanged =
      preset.rings !== this._preset.rings ||
      preset.segments !== this._preset.segments ||
      preset.radius !== this._preset.radius;
    this._preset = preset;
    this._applyAnisotropy();

    if (noiseChanged) {
      // The chunk source itself changes, so every adopted program is rebuilt.
      this._registerChunks();
      this._variant++;
      for (const material of this._patched.keys()) material.needsUpdate = true;
      if (this._mistMaterial) this._mistMaterial.needsUpdate = true;
    }

    if (geometryChanged && this._mistMesh) {
      this._disposeMistGeometry();
      this._mistGeometry = this._createMistGeometry();
      this._mistMesh.geometry = this._mistGeometry;
      this._snapX = Infinity;
      this._snapZ = Infinity;
      this._resampleTerrain(true);
    }
    if (this._mistGeometry) this._mistGeometry.instanceCount = this._preset.layers;
  }

  dispose() {
    this._disposed = true;
    this._unsubLightning?.();
    // Layer membership is left alone on purpose: the objects belong to other
    // systems and may outlive us, and layer 11 is inert to anyone not looking
    // for it. Only the traversal roots are dropped.
    this._occluderRoots.length = 0;

    for (const [material, original] of this._patched) {
      material.onBeforeCompile = original.onBeforeCompile;
      material.customProgramCacheKey = original.customProgramCacheKey;
      if (material.userData) material.userData.aerialPatched = false;
      material.needsUpdate = true;
    }
    this._patched.clear();

    if (this._mistMesh) {
      this._mistMesh.removeFromParent();
      this._mistMesh = null;
    }
    this._disposeMistGeometry();
    this._mistMaterial?.dispose();
    this._mistMaterial = null;

    if (this.sunProxy) {
      this.sunProxy.removeFromParent();
      this.sunProxy.geometry?.dispose();
      this.sunProxy.material?.map?.dispose();
      this.sunProxy.material?.dispose();
      this.sunProxy = null;
    }
    this.godRayLight = null;

    if (this._ownsNoiseTexture) this._noiseTexture?.dispose();
    this._noiseTexture = null;
    this.uniforms.uAerialNoise.value = null;

    if (this.scene.fog === this.fog) this.scene.fog = null;

    // The ShaderChunk entries deliberately stay registered. They are four
    // strings, they leak nothing, and unregistering them would make any other
    // module that `#include`s them fail to resolve during the window between an
    // HMR teardown and the next constructor.
  }

  // -------------------------------------------------------------------------
  // Frame
  // -------------------------------------------------------------------------

  update(dt, state) {
    if (this._disposed) return;

    // dt reaches every damp() below and damp( a, b, l, NaN ) is NaN, which then
    // latches into accumulators that are never recomputed from scratch. Zero is
    // the correct degenerate value: it makes every damp a no-op for one frame.
    dt = dt > 0 ? Math.min(dt, 0.25) : 0;

    // Anyone nulling scene.fog would silently strip USE_FOG from every program
    // in the scene and black out the aerial chunk. Pointer compare, free.
    if (this.scene.fog !== this.fog) this.scene.fog = this.fog;

    this._flash = clamp01(finite(damp(this._flash, clamp01(state.weather.lightning || 0), 6.5, dt), 0));

    this._updateGroundLevel(dt);
    this._updateAtmosphereColors(dt, state);
    this._updateDensities(dt, state);
    this._updateDrift(dt, state);
    this._updateMist(dt, state);
    this._updateSun(dt, state, this.camera);

    // Late-created materials (streamed grass and scatter chunks) adopt here.
    this._warmupTimer -= dt;
    this._scanTimer -= dt;
    if (this._scanTimer <= 0) {
      this._scanTimer = this._warmupTimer > 0 ? 0.05 : 0.3;
      // Cap recompiles per sweep: patching a live material rebuilds its program,
      // and a handful of those in one frame is a visible hitch.
      this._patchBudget = 6;
      this._scanScene();
      this._refreshOccluders();
    }
  }

  /**
   * Track the height the mist pools at. Kept out of the mist-visibility branch
   * because the fog chunk uses the same base height whether or not the slabs
   * are drawing.
   */
  _updateGroundLevel(dt) {
    // A non-finite camera Y (one bad player-controller frame) would otherwise
    // walk straight into _mistBase and stay there.
    const camAbove = finite(this.camera.position.y, 0) - this._mistBase;
    this._resampleCooldown -= dt;
    if (this._resampleCooldown <= 0 && camAbove < 140) {
      this._resampleTerrain(false);
      this._resampleCooldown = 0.12;
    }
    // Damped so the pooling level glides between snap cells rather than stepping.
    const target = finite(this._mistBaseTarget, 0);
    this._mistBaseTarget = target;
    this._mistBase = finite(damp(this._mistBase, target, 3.5, dt), target);
  }

  /**
   * Rebuild the fog palette from the actual sky. Using the sky model rather than
   * a hand-picked fog colour is the only way distant geometry lands exactly on
   * the horizon colour behind it.
   *
   * ---------------------------------------------------------------------------
   * THE WHITE-OUT LIVED HERE. Read this before touching anything below.
   * ---------------------------------------------------------------------------
   * `atmosphere.getSkyColor()` is the CLEAR-sky CPU model. It has never heard of
   * the cloud deck: it happily returns a bright 11:00 horizon radiance in the
   * middle of a rainstorm. But the sky the renderer actually PUTS ON SCREEN in
   * rain is the volumetric deck composited over that dome, and the deck is
   * several times darker. Inscattering the clear-sky value while the viewer sees
   * the deck means the fog is brighter than its own light source, and then a
   * blend of only a few per cent is enough to lift a dark wet trunk clean off
   * the top of the tone curve. Measured in the shipping build: tree [221,228,236]
   * against a sky of [196,210,223] at a fog factor of 0.065.
   *
   * What made it worse rather than better is that the old code went out of its
   * way to throw the correction away. `state.weather.fogColor` is built by
   * weather.js from `state.sky.horizonColor` - which sky/atmosphere.js has
   * ALREADY multiplied by its cloud `dim` factor - and then scaled by the
   * weather's own `fogLuma` (rain 0.72, storm 0.54). It is the one value in the
   * project that knows the deck is there. The old line
   *
   *     _cTmp.copy( weather.fogColor ).multiplyScalar( horizonLum / fogLum );
   *
   * divided that value by its own luminance and re-multiplied it by the
   * clear-sky probe's, keeping the hue and discarding every last bit of the
   * dimming - in rain, a factor of dim 0.49 x fogLuma 0.72 = 0.35, thrown out
   * and replaced by 1.0.
   *
   * So the rule now is: THE PROBES GIVE DIRECTION, THE WEATHER GIVES LEVEL. The
   * three probes still supply the whole directional character - warm toward the
   * sun, blue away from it, the zenith gradient, no seam at the horizon - and
   * the absolute level is pinned to the radiance weather.js says the air has.
   * Everything downstream in this function is anchored to `horizonLum`, so
   * re-levelling the probes before it is computed fixes the mist colour, the
   * sunlit-mist colour, the glow lobes, the ground bounce and the FogExp2
   * fallback in one place.
   */
  _updateAtmosphereColors(dt, state) {
    const sun = state.sun;
    const sky = state.sky;
    const weather = state.weather;
    const u = this.uniforms;

    // Everything sibling-owned that this function reads, taken once through the
    // NaN firewall. It matters more here than anywhere else in the file: these
    // feed EIGHT uniforms that every fogged material in the scene samples, and
    // `Math.max( luminance( x ), 1e-5 )` - the guard the palette is littered
    // with - passes a NaN through untouched.
    safeColor(sun.color, _cSun, 1, 1, 1);
    safeColor(state.moon.color, _cMoon, 0.55, 0.68, 1);
    safeColor(weather.fogColor, _cFog, 0.72, 0.78, 0.86);
    safeDir(sun.direction, _sunDir, 0, 1, 0);
    safeDir(state.moon.direction, _moonDir, 0, -1, 0);

    let hx = _sunDir.x;
    let hz = _sunDir.z;
    const hl = Math.hypot(hx, hz);
    if (hl < 1e-3) {
      hx = 1;
      hz = 0;
    } else {
      hx /= hl;
      hz /= hl;
    }
    // Just above the horizon: probing exactly at y = 0 lands on whatever the sky
    // model does at its ground plane, which is not what we need to match.
    _dirSunward.set(hx * 0.9852, 0.1715, hz * 0.9852);
    _dirAway.set(-hx * 0.9852, 0.1715, -hz * 0.9852);

    this._probeSky(_dirZenith, _cA, sky.zenithColor);
    this._probeSky(_dirSunward, _cB, sky.horizonColor);
    this._probeSky(_dirAway, _cC, sky.horizonColor);

    // ---- LEVEL: re-expose the clear-sky probes onto the visible sky ---------
    const coverage = clamp01(finite(state.clouds.coverage, 0));
    const storminess = clamp01(finite(state.clouds.storminess, 0));
    const units = Math.max(finite(weather.fogDensity, 0), 0) / this.fogReference;
    // How completely something other than the clear sky has become the thing the
    // eye sees: a deck overhead, or a fog bank you are standing inside of. Both
    // ends matter - a FOG event only runs 0.60 coverage, and you still cannot see
    // the sky through it.
    const deck = Math.max(smoothstep(0.22, 0.9, coverage), smoothstep(1.6, 4.0, units));
    this._deck = deck;

    // weather.js composes fogColor as lerp( horizon, zenith, 0.25 ) through a
    // luminance-normalised chroma filter and a luminance scale, so weighting the
    // probes the same way makes this ratio a clean estimate of exactly the
    // dimming the probes are missing, and nothing else.
    const probeLum = Math.max(
      0.25 * luminance(_cA) + 0.375 * (luminance(_cB) + luminance(_cC)),
      1e-5
    );
    const airLum = luminance(_cFog);
    // Only ever darkens. The clear-sky probe is a hard upper bound on what the
    // sky can be - a deck subtracts light, it never adds any - and letting this
    // exceed 1 would reintroduce the same failure from the other direction.
    // (If the sky probe is unavailable, _probeSky falls back to state.sky, which
    // is already dimmed; the ratio then degrades to fogLuma alone, which is the
    // right conservative answer rather than a double dim.)
    const dimmed = Math.min(finite(airLum / probeLum, 1), 1);
    const skyScale = clamp(lerp(1, dimmed, deck), 0.02, 1);
    this._skyScale = skyScale;
    _cA.multiplyScalar(skyScale);
    _cB.multiplyScalar(skyScale);
    _cC.multiplyScalar(skyScale);

    // A deck is a diffuser. Under one the clear-sky model's directional
    // structure - bright sunward horizon, dark zenith - is simply not what is
    // overhead any more, and leaving it in place is the second reason the
    // mid-ground grew a hard bright band: distant ground converged on a sunward
    // horizon radiance that the sky above it no longer had. Collapse the three
    // probes toward their common horizon mean as the deck closes in.
    if (deck > 0.01) {
      const flat = deck * 0.75;
      _cTmp.copy(_cB).lerp(_cC, 0.5);
      _cB.lerp(_cTmp, flat);
      _cC.lerp(_cTmp, flat);
      // The zenith of an overcast is brighter than its horizon - the path
      // through the deck is shortest straight up - so it flattens less far.
      _cA.lerp(_cTmp, flat * 0.55);
    }

    // 1 at the horizon, 0 once the sun is comfortably up. Needed here as well as
    // below: the sunward/away split is a Mie forward-scattering effect and it
    // widens sharply as the sun's light starts running the long way through the
    // air. _updateDensities runs straight after this and reuses it.
    const lowSun = 1 - smoothstep(0.02, 0.34, _sunDir.y);
    this._lowSun = lowSun;

    // How much DIRECT BEAM actually reaches the air. Every warm term in this
    // function is a forward-scattering effect and forward scattering needs a
    // beam: under a rain deck there is none, so the sunlit-mist colour and both
    // glow lobes have to go with it. They were previously gated only on
    // `sun.visibility`, which just asks whether the sun is above the horizon and
    // is therefore 1.0 in the middle of a storm - that is the additive half of
    // the white-out, and it is why rain mist glowed.
    //
    // Deliberately NOT gated on fog density: a dawn fog bank under a mostly open
    // sky is lit by a real low sun, and that glow is the single most beautiful
    // thing this module does. The coverage curve is also deliberately late and
    // steep - half the beam survives the FOG event's 0.60 deck and four fifths
    // survives partly-cloudy, while overcast 0.90 and rain 0.97 go to nothing.
    // Broken cloud lets the sun through most of the time; a stratus sheet does
    // not, and there is no smooth physical dial between them.
    const cloudShut = smoothstep(0.25, 0.95, coverage);
    const sunVis = clamp01(finite(sun.visibility, 0));
    const beam = sunVis * (1 - cloudShut) * (1 - 0.6 * storminess);
    this._beam = beam;

    // Push the two horizon probes apart. Real aerial perspective is measurably
    // bluer away from the sun and warmer toward it than any single-scattering
    // approximation makes it, and with the extinction correctly dialled down
    // this directional split is now the ONLY thing distinguishing near haze from
    // far haze - a flat grey wash in both directions is what made the old build
    // read as monochrome even where it was not especially dense. Luminance is
    // preserved by pushHue, so this changes hue and never brightness.
    const split = (0.06 + 0.1 * lowSun * beam) * (1 - deck * 0.8);
    pushHue(_cB, 1.0, 0.945, 0.83, split);
    pushHue(_cC, 0.82, 0.9, 1.0, split);

    // The level every colour below is anchored to. After the re-exposure above
    // this is the radiance of the sky the viewer is actually looking at.
    const horizonLum = Math.max((luminance(_cB) + luminance(_cC)) * 0.5, 1e-5);

    u.uAerialZenith.value.copy(_cA);
    u.uAerialHorizonSun.value.copy(_cB);
    u.uAerialHorizonAway.value.copy(_cC);

    // Ground bounce: hue from the sky model's ground colour, magnitude from the
    // probes, so both end up in the same (unknown) exposure space. Through the
    // firewall as well - it is divided by its own luminance one line down, and
    // NaN / NaN is the one arithmetic that survives every clamp in this file.
    safeColor(sky.groundColor, _cTmp, 0.22, 0.24, 0.2);
    const groundLum = Math.max(luminance(_cTmp), 1e-5);
    u.uAerialGround.value.copy(_cTmp).multiplyScalar((horizonLum * 0.5) / groundLum);

    // ---- mist palette -----------------------------------------------------
    const fogLum = Math.max(luminance(_cFog), 1e-5);
    // Re-expose the authored fog tint to the sky's brightness before blending,
    // so a weather-driven colour change can never fight the current exposure.
    //
    // This is the line that used to launder the weather dimming away, and it is
    // safe now only because `horizonLum` has itself been pinned to the weather's
    // level a few dozen lines up: under a deck the ratio is ~1 and this is a
    // no-op, while under an open sky it correctly lifts the tint onto the sky
    // the fog is dissolving into. Re-levelling the probes and re-exposing the
    // tint to them are two halves of one operation - never restore one without
    // the other.
    _cTmp.copy(_cFog).multiplyScalar(horizonLum / fogLum);
    // Only a third of the way toward the authored tint. weather.js has already
    // applied its own fogTintWeight, and mist is the dominant term in the whole
    // near and middle field - pulling it 60% toward a neutral grey-blue threw
    // away the sky's directional character exactly where it is most visible.
    _cA.copy(_cB).lerp(_cC, 0.5).lerp(_cTmp, 0.35).multiplyScalar(0.94);

    // Rain drives mist toward neutral: airborne water desaturates fast.
    const rain = clamp01(weather.rainIntensity || 0);
    if (rain > 0) {
      const g = luminance(_cA);
      _cA.lerp(_cTmp.setRGB(g, g, g), rain * 0.35);
    }

    // Lightning lights the whole air volume, not only the surfaces in it - the
    // one moment in this file where the air is genuinely allowed to outshine the
    // sky, because for those few frames it IS the brightest thing in the world.
    // The energy ceiling below is widened by the same envelope so it does not
    // clamp the flash back out; f is squared, so this is gone in ~200 ms.
    const flash = this._flash > 0.001 ? this._flash * this._flash : 0;
    if (flash > 0) {
      _cA.addScalar(horizonLum * 1.7 * flash);
      u.uAerialZenith.value.addScalar(horizonLum * 1.2 * flash);
      u.uAerialHorizonSun.value.addScalar(horizonLum * 1.4 * flash);
      u.uAerialHorizonAway.value.addScalar(horizonLum * 1.4 * flash);
    }
    u.uAerialMistColor.value.copy(_cA);

    // Sunlit mist: the sun's own hue at up to ~3x the ambient sky luminance.
    // Anchoring to sky luminance means it glows warm without ever blowing out.
    // Gated on `beam`, not on `sunVis`: mist can only be sunlit where sunlight
    // arrives, and under a rain deck this collapses to the neutral mist colour.
    const sunHueLum = Math.max(luminance(_cSun), 1e-5);
    _cTmp.copy(_cSun).multiplyScalar((horizonLum * (0.95 + 2.05 * lowSun) * beam) / sunHueLum);
    u.uAerialMistSun.value.copy(_cA).lerp(_cTmp, 0.68);

    // Tight forward-scatter lobe. Strongest at dawn and dusk, when the ray runs
    // the length of the mist layer toward a low sun. `beam` already carries the
    // cloud term far more honestly than the old ( 1 - coverage * 0.55 ) did - 
    // that left 45 % of a full-strength dawn glow alive under a total overcast.
    const glowGain = (0.42 + 1.85 * lowSun) * beam;
    u.uAerialSunGlow.value.copy(_cSun).multiplyScalar((horizonLum * glowGain) / sunHueLum);
    u.uAerialSunDir.value.copy(_sunDir);

    // Both lobes are always evaluated. The alternative - switching the dominant
    // light at dusk - pops exactly when the audience is looking at the sky.
    const moon = state.moon;
    // Same deck gate as the sun: a moon behind stratus lights nothing either,
    // and a moon-glow lobe surviving a socked-in night is the night-time version
    // of exactly the bug this file exists to not have.
    const moonVis = clamp01(finite(moon.visibility, 0)) * (1 - sunVis) * (1 - cloudShut * 0.9);
    const moonHueLum = Math.max(luminance(_cMoon), 1e-5);
    u.uAerialMoonDir.value.copy(_moonDir);
    u.uAerialMoonGlow.value
      .copy(_cMoon)
      .multiplyScalar(
        (horizonLum * 0.85 * moonVis * clamp01(finite(moon.intensity, 0) * 4)) / moonHueLum
      );

    // ---- ENERGY CEILING ----------------------------------------------------
    // The shader's mist in-scatter is
    //     mix( mistColor, mistSun, w )  +  sunGlow * lobe  +  moonGlow * lobe'
    // with every weight in 0..1, so its radiance can reach
    //     lum( mistSun ) + lum( sunGlow ) + lum( moonGlow ).
    // Forward scattering genuinely does make mist brighter than the average sky
    // when you look into a low sun - but by a BOUNDED factor. An unbounded sum of
    // full-strength lobes on top of an already-full-strength in-scatter is the
    // classic way to make fog outshine its own light source, and at dawn the old
    // constants could reach 4.5x the horizon radiance in the sun direction.
    //
    // Capping it here, once per frame, is free: it costs three dot products on
    // the CPU instead of a clamp on every fogged fragment in the frame, and it
    // makes the invariant
    //     luminance( inScatter ) <= scatterCeiling * luminance( sky )
    // hold by construction for every direction and every material.
    const ceiling = horizonLum * this.scatterCeiling * (1 + 3 * flash);
    const mistSun = u.uAerialMistSun.value;
    // mistSun is the BASE colour of lit mist, never an addition, so it gets the
    // larger share and the two additive lobes divide whatever is left over.
    const mistSunLum = luminance(mistSun);
    if (mistSunLum > ceiling * 0.72) mistSun.multiplyScalar((ceiling * 0.72) / mistSunLum);
    const headroom = Math.max(ceiling - luminance(mistSun), 0);
    const lobeLum = luminance(u.uAerialSunGlow.value) + luminance(u.uAerialMoonGlow.value);
    if (lobeLum > headroom) {
      const k = headroom / Math.max(lobeLum, 1e-6);
      u.uAerialSunGlow.value.multiplyScalar(k);
      u.uAerialMoonGlow.value.multiplyScalar(k);
    }

    // Keep the fallback scene fog in the same visual family as the real thing.
    // three's own fog_fragment runs AFTER the output colour-space conversion, so
    // it mixes in display space and a radiance value - which mist colour is, and
    // which routinely exceeds 1 - would clip to white there. Reinhard it into
    // 0..1 first. Only unpatched materials ever see this.
    const fc = this.fog.color;
    const mc = u.uAerialMistColor.value;
    fc.setRGB(mc.r / (1 + mc.r), mc.g / (1 + mc.g), mc.b / (1 + mc.b));
  }

  /**
   * Sky probe, with a graceful fall back to state.sky if atmosphere is absent.
   *
   * The fallback is used RAW and that is deliberate. `state.sky.zenithColor` and
   * `horizonColor` are written by sky/atmosphere.js as the mean of the very same
   * `getSkyColor()` this function is trying to call, already multiplied by its
   * cloud `dim` factor - they are scene-referred RADIANCE in exactly the units
   * the probe returns, not a normalised tint. Scaling them by
   * `sky.ambientIntensity` (which this used to do) double-counts: that field is
   * `skyIrradiance / NOON_REFERENCE_LUMINANCE`, so it is ~1 at noon and 0.012 at
   * midnight, and the fallback came back up to eighty times too dark. That in
   * turn drove `probeLum` to a hair above zero, sent `airLum / probeLum` past 1,
   * and the `Math.min( ..., 1 )` clamp then pinned `skyScale` at 1 - destroying
   * precisely the "degrades to fogLuma alone" behaviour the caller documents.
   */
  _probeSky(dir, out, fallback) {
    const atmo = this.atmosphere;
    if (this._skyProbeOk && atmo && typeof atmo.getSkyColor === 'function') {
      try {
        const c = atmo.getSkyColor(dir, out);
        if (c && Number.isFinite(c.r) && Number.isFinite(c.g) && Number.isFinite(c.b)) {
          if (c !== out) out.copy(c);
          return out;
        }
      } catch (err) {
        // Warn once, then stop paying for the try/catch on every frame.
        console.warn('[AtmosphericFX] atmosphere.getSkyColor failed, using state.sky:', err);
        this._skyProbeOk = false;
      }
    }
    out.copy(fallback);
    // Last line of defence. `out` has already been overwritten by whatever
    // getSkyColor wrote before it was rejected, and the fallback itself is a
    // sibling-owned colour scaled by a sibling-owned float - so this is the only
    // point at which BOTH the probe and its fallback are known to have failed.
    // A dark neutral is the right degenerate answer: the aerial term goes quiet
    // for a frame instead of painting NaN into every fogged material at once.
    if (!Number.isFinite(out.r) || !Number.isFinite(out.g) || !Number.isFinite(out.b)) {
      out.setRGB(0.05, 0.06, 0.08);
    }
    return out;
  }

  _updateDensities(dt, state) {
    const u = this.uniforms;
    const weather = state.weather;
    const fogDensity = Math.max(weather.fogDensity || 0, 0);
    // Everything downstream is expressed in "units of nominal murk" rather than
    // in the raw dial, so the curves stay readable and the reference is one
    // number to retune if the weather system ever rescales its presets.
    const units = fogDensity / this.fogReference;

    const rain = clamp01(weather.rainIntensity || 0);
    const wetness = clamp01(weather.wetness || 0);
    // Read the three sibling-owned scalars this function pivots on exactly once,
    // through the NaN firewall, instead of letting each of the six curves below
    // decide independently what a NaN means.
    const sunY = finite(state.sun.direction.y, 0);
    const coverage = clamp01(finite(state.clouds.coverage, 0));
    const hourRaw = finite(state.time.timeOfDay, 12);

    // --- tall haze ---------------------------------------------------------
    // Sub-linear in units, so the aerosol column saturates around 2.5 km of
    // meteorological range no matter how thick the weather gets. Everything
    // beyond that belongs to the boundary layer, which is what the mist is.
    //
    // Rain is weighted heavily (1.15, was 0.35) on purpose. Falling rain and its
    // spray are a DEEP, well-mixed column - that is why a rainy hillside two
    // kilometres off goes grey while the field at your feet stays saturated - 
    // and the murk has to live in the tall medium to behave that way. Loading it
    // into the 16 m mist layer instead put a savage vertical gradient right at
    // eye level, which is the other half of why the mid-ground grew a hard band:
    // a level ray picked up five times the optical depth of one aimed three
    // degrees higher, so ground and sky met at a step instead of a fade.
    const hazeTarget = clamp(
      this.hazeFloor + this.hazeGain * Math.pow(units, this.hazeExponent) * (1 + rain * this.hazeRainGain),
      0,
      0.0035
    );

    // --- low mist ----------------------------------------------------------
    // Radiation fog forms in the cool hours and burns off as the sun climbs.
    // Driving the burn-off from the sun's ELEVATION rather than the clock keeps
    // it correct when daySpeed changes or the player scrubs time, and it is the
    // physical cause anyway: it is insolation that lifts the inversion.
    const coolness = 1 - smoothstep(-0.02, 0.42, sunY);
    // timeOfDay is documented as 0..24 but nothing enforces the wrap, and bell()
    // only folds a difference into +/-12 h - an unwrapped 25.5 or -0.5 would slip
    // the dawn peak by a whole day-length. mod() is the owner's own convention.
    const hour = ((hourRaw % 24) + 24) % 24;
    // Sharpest right at first light and again as the ground gives up its heat
    // after sunset; the broad `coolness` term carries the rest of the night.
    const dawnPeak = bell(hour, 5.3, 2.5) + bell(hour, 20.0, 2.1) * 0.5;
    const diurnal = 0.5 + 1.2 * coolness + 0.55 * dawnPeak;
    // Wind shears ground mist apart; still mornings are the foggy ones.
    const windMix = clamp01(finite(state.wind.strength * state.wind.gust, 0) / 9);
    const windFactor = lerp(1.18, 0.5, windMix * windMix);
    // Radiation fog needs a clear sky to radiate INTO. A cloud deck re-radiates
    // the ground's heat straight back down, the inversion never forms, and the
    // morning stays clear at ground level however grey it looks overhead. This
    // is why the prettiest mist arrives under the emptiest sky.
    const clearSky = 1 - coverage * 0.55;
    // A declared FOG or STORM event is advected, not radiative: it does not care
    // what time it is, a stiff breeze does not blow it away, and it arrives
    // under exactly the overcast that suppresses the radiative kind. Fade all
    // three modifiers out as the weather takes over from the diurnal cycle.
    const heavy = clamp01((units - 1) / 3);
    const radiation = lerp(diurnal * windFactor * clearSky, 1, heavy);

    // Rain is the enemy of a stratified boundary layer: the drops drag air down
    // with them and the gust front stirs the rest, so a rainy afternoon has far
    // LESS structure at ankle height than a still morning does, not more. The
    // murk is still there - it moved into the haze column above. Wetness pushes
    // gently the other way: a soaked field evaporating is where ground mist
    // comes from in the first place.
    const boundary = (1 - this.mistRainMix * rain) * (1 + 0.15 * wetness);

    // Superlinear in units on purpose. A linear map cannot serve both ends: it
    // either soups up a clear morning or leaves a declared FOG event reading as
    // light haze. Squaring it (fogDensity * units) is what puts 237 m of
    // visibility inside a fog event while leaving a nominal day at 7.9 km.
    const mistTarget = clamp(this.mistGain * fogDensity * units * radiation * boundary, 0, 0.06);

    // A thin dawn ribbon lies IN the grass; a real fog bank is tens of metres
    // deep and the eye sits inside it. The layer has to physically deepen with
    // the weather or a fog event just looks like a very bright floor.
    const depth = Math.pow(clamp01((units - 0.7) / 7.5), 1.2);
    // Cool air deepens the layer as well as thickening it - the inversion sits
    // higher before dawn than it does at noon. Rain deepens it too, and for the
    // opposite reason: not a stable inversion but a stirred one, spray lifted to
    // roof height by the same turbulence that thinned it. Deepening the layer is
    // what softens the near-horizon gradient into a fade. Clamped at both ends:
    // below 1.8 m the eye is above the whole layer and the mist stops existing,
    // and past ~60 m the analytic term is already doing everything the slabs can
    // see.
    const mistHeight = finite(
      clamp(
        lerp(this.mistHeightMin, this.mistHeightMax, depth) *
          (1 + 0.35 * coolness) *
          (1 + this.mistRainDepth * rain),
        1.8,
        60
      ),
      this.mistHeightMin
    );
    // Second guard on the accumulator, not just the target: `damp` reads its own
    // previous output, so once it is NaN a finite target never rescues it.
    this._mistHeight = finite(damp(this._mistHeight, mistHeight, 1.1, dt), mistHeight);

    const mistT = finite(mistTarget, 0);
    const mv = u.uAerialMist.value;
    mv.x = finite(damp(mv.x, mistT, 2.2, dt), mistT);
    mv.y = 1 / this._mistHeight;
    mv.z = this._mistBase;

    const hazeT = finite(hazeTarget, this.hazeFloor);
    const hv = u.uAerialHaze.value;
    hv.x = finite(damp(hv.x, hazeT, 1.4, dt), hazeT);
    hv.y = 1 / this.hazeHeight;
    hv.z = this._mistBase;
    // How much of the sun's forward lobe the tall haze carries. A low sun sends
    // its light the long way through the layer, so the whole sky-side of the
    // horizon lifts; overhead it barely registers.
    hv.w = 0.28 + 0.52 * this._lowSun;

    // Read the damped values, not the targets, so everything downstream reacts
    // to exactly the density the shader is already rendering.
    this._effectiveMist = mv.x;
    // What the mist actually contributes along a level ray at the eye. This - 
    // not the base density - is the number that decides whether the field looks
    // misty, because the eye is one to two e-folding heights up when walking.
    const eyeAbove = Math.max(finite(this.camera.position.y, 0) - this._mistBase, 0);
    this._sigmaEyeMist = mv.x * Math.exp(-eyeAbove * mv.y);
    // 0..1 artistic dial. Square-rooted against a 0.006/m reference (~650 m of
    // range) so it moves across the densities the weather system actually
    // spends its time in instead of only near a whiteout.
    this.mistiness = clamp01(Math.sqrt(this._sigmaEyeMist / 0.006));

    // Mie asymmetry: fat mist droplets scatter far more forward than dry haze.
    mv.w = lerp(0.6, 0.84, this.mistiness);

    const pv = u.uAerialParams.value;
    pv.x = DITHER_BASE;
    pv.y = this.maxFogOpacity;
    // Nothing fogs inside two and a half metres. Below that a walking player's
    // own feet would pick up inscatter, which reads as a lens smear.
    pv.z = 2.5;
    pv.w = 1 / NOISE_TILE;

    // Match the unpatched-material fallback to the real curve at 220 m, where
    // FogExp2's 1 - exp(-(d*x)^2) and our 1 - exp(-k*x) agree for d = sqrt(k/220).
    // Uses the eye-level mist term, not the base density, for the same reason.
    const fallbackSigma = Math.max(hv.x + this._sigmaEyeMist, 1e-7);
    const fallbackDensity = Math.sqrt(fallbackSigma / 220);
    this.fog.density = Number.isFinite(fallbackDensity) ? fallbackDensity : 0.0015;
  }

  _updateDrift(dt, state) {
    const wind = state.wind;
    // _driftA/_driftB are integrators: they are never recomputed, only added to.
    // One NaN increment wedges the mist noise UVs at NaN for the rest of the
    // session, which does not look like a glitch - it looks like the mist layer
    // silently disappearing.
    const speed = finite(wind.strength * wind.gust, 0);
    const dx = finite(wind.direction.x * speed * dt, 0);
    const dz = finite(wind.direction.y * speed * dt, 0);

    // The coarse bank shape barely moves while the fine detail runs across it.
    // That shear is what makes mist boil and tear instead of sliding past as one
    // sheet, and it is the single strongest cue that this is air and not a decal.
    this._driftA.x = wrapTile(this._driftA.x + dx * 0.12);
    this._driftA.y = wrapTile(this._driftA.y + dz * 0.12);
    this._driftB.x = wrapTile(this._driftB.x + (dx * 0.94 - dz * 0.34) * 0.34);
    this._driftB.y = wrapTile(this._driftB.y + (dz * 0.94 + dx * 0.34) * 0.34);

    // Negated because shifting the sample point backwards moves the pattern
    // forwards. Every scale below divides NOTHING but NOISE_TILE or a divisor of
    // it, so the wrap at NOISE_TILE stays seamless.
    const invTile = 1 / NOISE_TILE;
    const off = this.uniforms.uAerialNoiseOffset.value;
    off.set(
      -this._driftA.x * invTile,
      -this._driftA.y * invTile,
      -this._driftB.y * 3 * invTile,
      -this._driftB.x * 3 * invTile
    );

    if (this._mistMaterial) {
      const scroll = this._mistMaterial.uniforms.uMistScroll.value;
      const fine = 4 * invTile; // fine field tiles every NOISE_TILE / 4 metres
      scroll.set(off.x, off.y, -this._driftB.x * fine, -this._driftB.y * fine);
    }
  }

  // -------------------------------------------------------------------------
  // Volumetric mist slabs
  // -------------------------------------------------------------------------

  _buildMist() {
    this._mistGeometry = this._createMistGeometry();

    this._mistMaterial = new ShaderMaterial({
      name: 'AtmosphericFX.mist',
      vertexShader: MIST_VERTEX,
      fragmentShader: MIST_FRAGMENT,
      uniforms: Object.assign(
        {
          uMistNoise: { value: this._noiseTexture },
          uMistScroll: { value: new Vector4() },
          uMistShape: { value: new Vector4(1 / NOISE_TILE, 4 / NOISE_TILE, 0.86, 5) },
          uMistFade: { value: new Vector4(11, 60, 130, 0) },
          uMistParams: { value: new Vector4(0, 1, 1 / this.mistHeightMin, this.mistHeightMin * this.mistStackScale) },
          uMistLight: { value: new Vector4(0, 1, this.mistPooling, 0) },
        },
        this._fogCompat,
        this.uniforms
      ),
      defines: { AERIAL_WORLDPOS_PROVIDED: '' },
      transparent: true,
      depthWrite: false,
      depthTest: true,
      side: DoubleSide,
      fog: true,
      toneMapped: true,
    });
    // three keys its program cache on the UNRESOLVED shader source (a string ->
    // id map in WebGLShaderCache), the defines, and this hook - never on the
    // ShaderChunk entries the source `#include`s. MIST_FRAGMENT is a constant
    // string, so when onQualityChange() rewrites sakura_aerial_pars_fragment and
    // sets needsUpdate, getProgramCacheKey() returns the SAME key and
    // acquireProgram() hands back the already-compiled program: the tier switch
    // silently does nothing here. That is not only cosmetic - dropping to LOW is
    // supposed to compile the two textureLod taps out of the aerial chunk, and
    // without this the slabs keep paying for them.
    this._mistMaterial.customProgramCacheKey = () => 'sakuraMist' + this._variant;
    // Already wired by hand; keep the auto-adopt sweep away from it.
    this._mistMaterial.userData.aerial = false;

    this._mistMesh = new Mesh(this._mistGeometry, this._mistMaterial);
    this._mistMesh.name = 'AtmosphericFX.mist';
    this._mistMesh.frustumCulled = false;
    // Mist is a background element: petals and rain belong in front of it.
    this._mistMesh.renderOrder = -5;
    this._mistMesh.userData.noAerial = true;
    this._mistMesh.visible = false;
    this.scene.add(this._mistMesh);
  }

  /**
   * An annulus of concentric rings. Polar beats a square grid here: every fade
   * in this material is radial, so the tessellation follows the thing that
   * varies, and the inner hole drops the vertices the near fade would discard.
   */
  _createMistGeometry() {
    const { rings, segments, radius } = this._preset;
    // Small, not zero. At ground level the near fade already blanks everything
    // inside 11 m so the hole is free, but from the air you look straight down
    // at it - a 6 m hole reads as a puncture, a 2 m one does not.
    const inner = 2;
    const vertexCount = (rings + 1) * segments;
    const positions = new Float32Array(vertexCount * 3);
    const terrainY = new Float32Array(vertexCount);
    const radial = new Float32Array(vertexCount);
    const rand = makeRNG(0x5a4b3c);

    for (let r = 0; r <= rings; r++) {
      const t = r / rings;
      // Concentrated near the camera, where a metre of world is many pixels.
      const base = inner + (radius - inner) * Math.pow(t, 1.75);
      const stagger = (r & 1) * 0.5;
      for (let s = 0; s < segments; s++) {
        const i = r * segments + s;
        // Deterministic jitter, so no concentric-ring pattern can survive in the
        // terrain-emergence fade - the one term evaluated per vertex.
        const angle = ((s + stagger + (rand() - 0.5) * 0.35) / segments) * TAU;
        const rad = base * (1 + (rand() - 0.5) * 0.07);
        positions[i * 3] = Math.cos(angle) * rad;
        positions[i * 3 + 1] = 0;
        positions[i * 3 + 2] = Math.sin(angle) * rad;
        radial[i] = t;
      }
    }

    const IndexArray = vertexCount > 65535 ? Uint32Array : Uint16Array;
    const indices = new IndexArray(rings * segments * 6);
    let o = 0;
    for (let r = 0; r < rings; r++) {
      for (let s = 0; s < segments; s++) {
        const s1 = (s + 1) % segments;
        const a = r * segments + s;
        const b = r * segments + s1;
        const c = (r + 1) * segments + s;
        const d = (r + 1) * segments + s1;
        indices[o++] = a;
        indices[o++] = c;
        indices[o++] = b;
        indices[o++] = b;
        indices[o++] = c;
        indices[o++] = d;
      }
    }

    const layers = new Float32Array(MAX_MIST_LAYERS);
    for (let i = 0; i < MAX_MIST_LAYERS; i++) layers[i] = i;

    const geo = new InstancedBufferGeometry();
    geo.setAttribute('position', new BufferAttribute(positions, 3));
    geo.setAttribute('aTerrainY', new BufferAttribute(terrainY, 1));
    geo.setAttribute('aLayer', new InstancedBufferAttribute(layers, 1));
    geo.setIndex(new BufferAttribute(indices, 1));
    geo.instanceCount = this._preset.layers;
    geo.userData.radial = radial;
    // Manual bounds: the mesh follows the camera and is never culled, so three
    // must not walk the position buffer looking for a sphere it will not use.
    geo.boundingSphere = new Sphere(new Vector3(0, 0, 0), radius * 1.6);
    return geo;
  }

  _disposeMistGeometry() {
    this._mistGeometry?.dispose();
    this._mistGeometry = null;
  }

  /**
   * Re-sample terrain under the slab footprint. Only runs when the camera has
   * crossed a snap cell, so the cost is a few tenths of a millisecond every
   * couple of seconds of walking rather than every frame.
   */
  _resampleTerrain(force) {
    const geo = this._mistGeometry;
    if (!geo) return;
    const camera = this.camera;
    const ox = Math.round(camera.position.x / 4) * 4;
    const oz = Math.round(camera.position.z / 4) * 4;
    if (!force && ox === this._snapX && oz === this._snapZ) return;
    this._snapX = ox;
    this._snapZ = oz;
    this._mistMesh?.position.set(ox, 0, oz);

    const positions = geo.attributes.position.array;
    const attr = geo.attributes.aTerrainY;
    const heights = attr.array;
    const radial = geo.userData.radial;
    // Bound once and cached. This runs from update() whenever the camera crosses
    // a 4 m snap cell - several times a second while sprinting - and binding per
    // call would allocate a closure inside the hot path for no reason.
    const terrain = this.terrain;
    if (terrain !== this._samplerOwner) {
      this._samplerOwner = terrain;
      this._terrainSampler =
        terrain && typeof terrain.getHeight === 'function' ? terrain.getHeight.bind(terrain) : null;
    }
    const sample = this._terrainSampler;

    let minH = Infinity;
    let sum = 0;
    let count = 0;
    for (let i = 0, n = heights.length; i < n; i++) {
      let h = 0;
      if (sample) {
        h = sample(positions[i * 3] + ox, positions[i * 3 + 2] + oz);
        if (!Number.isFinite(h)) h = 0;
      }
      heights[i] = h;
      if (radial[i] < 0.45) {
        if (h < minH) minH = h;
        sum += h;
        count++;
      }
    }
    attr.needsUpdate = true;

    if (count > 0) {
      // Mist pools. The surface sits near the local low ground rather than the
      // mean, so hollows fill up and ridges stand clear of it.
      this._mistBaseTarget = minH * 0.72 + (sum / count) * 0.28 - 0.15;
    } else {
      this._mistBaseTarget = (this.state.player.groundHeight || 0) - 0.15;
    }
    if (force) this._mistBase = this._mistBaseTarget;
  }

  _updateMist(dt, state) {
    const mesh = this._mistMesh;
    if (!mesh) return;

    const layers = this._preset.layers;
    // The stack spans the visually important bottom of the mist and hands the
    // rest to the analytic term, so it has to track the layer's real depth: a
    // fixed 7 m of slabs inside a 45 m fog bank is a floor, not an atmosphere.
    const stackHeight = clamp(this._mistHeight * this.mistStackScale, 5, this.mistStackHeight);
    const camAbove = finite(this.camera.position.y, 0) - this._mistBase;

    // Below a real density the slabs are pure cost - the smooth fog term already
    // covers what little haze there is. Biggest single win on the iGPU, and at
    // the tuned coefficients this is what makes a clear midday genuinely have no
    // mist pass at all. Faded, not switched: crossing a hard threshold as the
    // weather drifts or as the player climbs is a visible flash.
    const densityFade = smoothstep(
      this.mistSlabCutIn,
      this.mistSlabCutIn * 2.4,
      this._effectiveMist
    );
    const altFade = 1 - smoothstep(stackHeight + 22, stackHeight + 58, camAbove);
    this._slabFade = densityFade * altFade;
    mesh.visible = this._slabFade > 0.003 && layers > 0;
    if (!mesh.visible) return;

    const u = this._mistMaterial.uniforms;
    const spacing = stackHeight / layers;

    const params = u.uMistParams.value;
    params.x = this._mistBase;
    params.y = spacing;
    // Slightly tighter than the analytic fog term: the slabs are the part that
    // hugs the grass, the fog term is the part that fills the middle distance.
    params.z = 1 / Math.max(this._mistHeight * 0.8, 1.2);
    params.w = stackHeight;

    const shape = u.uMistShape.value;
    // More density -> lower threshold -> banks merge into an unbroken sheet.
    // Measured against the baked field, this range walks from sparse wisps in
    // the hollows through real dawn banks to an unbroken sheet in a FOG event.
    shape.z = lerp(0.86, 0.24, this.mistiness);
    // Mean coverage the threshold implies. Only used to estimate the column
    // density for self-shadowing, so an analytic approximation is plenty.
    const meanCover = clamp01(0.25 + 0.6 * this.mistiness);

    const fade = u.uMistFade.value;
    // You never see fog on your own nose - but in a real fog event you very much
    // see it at three metres, and an 11 m clear bubble around the player would
    // be the single most obvious tell in the frame.
    fade.x = lerp(11, 3.2, this.mistiness);
    fade.y = this._preset.radius * 0.52;
    fade.z = this._preset.radius * 0.99;
    // A pure density: the `path` term in the shader supplies the length.
    fade.w = this._effectiveMist * this.mistStructure * this._slabFade;

    const light = u.uMistLight.value;
    // sigma * H for the structured (slab) density, capped: past a certain column
    // the base is simply unlit and more optical depth changes nothing visible.
    light.x = Math.min(
      this._effectiveMist * this.mistStructure * meanCover * Math.min(this._mistHeight, stackHeight),
      8
    );
    // Guard the divide. Below the horizon the sun contributes nothing anyway
    // because light.w carries its visibility, but the ratio still has to be
    // finite - and Math.max( NaN, 0.06 ) is NaN, which would put NaN straight
    // into vSunLight and blank every slab.
    light.y = Math.max(finite(state.sun.direction.y, 0), 0.06);
    light.z = this.mistPooling;
    // Through the firewall for exactly the reason light.y is: `clamp01` is a
    // pair of comparisons and every comparison against NaN is false, so
    // `clamp01( NaN )` returns NaN untouched. This one is worse than most - 
    // vSunLight is `uMistLight.w * exp( ... )`, so a NaN here multiplies the
    // whole self-shadow term and every slab fragment in the stack comes out NaN.
    light.w = clamp01(finite(state.sun.visibility, 0));

    this._mistGeometry.instanceCount = layers;
  }

  // -------------------------------------------------------------------------
  // Sun / god rays
  // -------------------------------------------------------------------------

  _buildSunProxy() {
    let map = null;
    if (typeof document !== 'undefined') {
      const size = 128;
      const canvas = document.createElement('canvas');
      canvas.width = canvas.height = size;
      const g = canvas.getContext('2d');
      const grad = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
      // A hard core with a wide, fast-falling skirt: the core carries the shafts,
      // the skirt stops the source aliasing into a polygon at half resolution.
      grad.addColorStop(0.0, 'rgba(255,255,255,1)');
      grad.addColorStop(0.22, 'rgba(255,250,238,0.92)');
      grad.addColorStop(0.5, 'rgba(255,238,214,0.30)');
      grad.addColorStop(0.78, 'rgba(255,232,208,0.06)');
      grad.addColorStop(1.0, 'rgba(255,230,205,0)');
      g.fillStyle = grad;
      g.fillRect(0, 0, size, size);
      map = new CanvasTexture(canvas);
      map.colorSpace = SRGBColorSpace;
      map.needsUpdate = true;
    }

    const material = new MeshBasicMaterial({
      map,
      color: 0xfff2e0,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: AdditiveBlending,
      toneMapped: false,
      fog: false,
    });
    material.userData.aerial = false;

    // Unit quad; _updateSun scales it to hold a constant angular size whatever
    // the camera's far plane turns out to be.
    const mesh = new Mesh(new PlaneGeometry(2, 2), material);
    mesh.name = 'AtmosphericFX.sunProxy';
    mesh.frustumCulled = false;
    mesh.renderOrder = -100;
    mesh.visible = false; // link() enables it only if celestial has no sun body
    mesh.userData.noAerial = true;
    this.sunProxy = mesh;
    this.scene.add(mesh);
  }

  _updateSun(dt, state, camera) {
    const sun = state.sun;
    // ---- NaN firewall, and it belongs here more than anywhere else ----------
    // EVERYTHING this function computes is PUBLISHED: sunWorldPosition, the
    // three screen-space vectors, the occlusion trio, godRayIntensity and
    // godRayColor, plus the sun proxy's own transform, opacity and colour. The
    // rest of the file sanitises its inputs and then damps them; here a bad
    // value goes straight out of the module and into post/pipeline.js, and a
    // NaN written into a Mesh position or a material opacity is a permanently
    // undrawn (or, on some drivers, permanently wrong) sun disc.
    //
    // These reads used to be raw, and `clamp01` does NOT absorb NaN - it is two
    // comparisons and both are false against NaN. Measured: one NaN frame in
    // `sun.color`, `sun.direction`, `sun.visibility`, `clouds.coverage`,
    // `clouds.storminess` or `camera.position` latched NaN into
    // sunWorldPosition, cloudOcclusion, sunOcclusion, the proxy's position,
    // scale and colour, and its material opacity.
    safeDir(sun.direction, _sunDir, 0, 1, 0);
    safeColor(sun.color, _cSun, 1, 1, 1);
    const sunVis = clamp01(finite(sun.visibility, 0));
    const coverage = clamp01(finite(state.clouds.coverage, 0));
    const storminess = clamp01(finite(state.clouds.storminess, 0));
    const camX = finite(camera.position.x, 0);
    const camY = finite(camera.position.y, 0);
    const camZ = finite(camera.position.z, 0);

    // Derive from the direction rather than state.sun.position: the player walks
    // far from the origin and an absolute sky-sphere point would skew.
    const distance = Math.min(2400, (finite(camera.far, 8000) || 8000) * 0.35);
    _v3a.set(camX, camY, camZ).addScaledVector(_sunDir, distance);
    // Published so a pipeline that wants to place its own shaft origin does not
    // have to reach into sunProxy (which may be celestial's mesh, or hidden).
    this.sunWorldPosition.copy(_v3a);

    if (this.sunProxy) {
      this.sunProxy.position.copy(_v3a);
      this.sunProxy.quaternion.copy(camera.quaternion);
      // Constant apparent size: ~1.4 degrees across, a soft halo around a core
      // roughly the angular size of the real disc.
      this.sunProxy.scale.setScalar(distance * 0.012);
      this.sunProxy.material.opacity = sunVis * (1 - coverage * 0.7);
      this.sunProxy.material.color.copy(_cSun);
    }

    _camFwd.set(0, 0, -1).applyQuaternion(camera.quaternion);
    const facing = _camFwd.dot(_sunDir);

    _v3b.copy(_v3a).project(camera);
    // A degenerate sun.direction puts the sun point exactly at the eye and
    // project() then divides by a w of zero: +/-Infinity, or NaN for 0/0. All
    // four screen-space values below are PUBLISHED for the shaft pass and
    // clamp() does not absorb NaN, so sanitise here, before anything downstream
    // can latch it. Far off-frame is the right degenerate answer: no shafts,
    // rather than shafts anchored nowhere.
    let ndcX = finite(_v3b.x, 4);
    let ndcY = finite(_v3b.y, 4);
    // A point behind the camera projects with a negative w, which mirrors it
    // back into the frame. Undo the mirror and throw it well outside, or a
    // consumer that trusts the value paints shafts radiating from the wrong
    // place every time the player turns around.
    if (facing <= 0) {
      ndcX = -ndcX * 4;
      ndcY = -ndcY * 4;
    }
    this.sunScreenNDC.set(ndcX, ndcY);
    // Same convention GodRaysMaterial.screenPosition wants: UV, clamped to a
    // band outside the frame so shafts still sweep in from just off screen.
    this.sunScreenPosition.set(clamp((ndcX + 1) * 0.5, -1, 2), clamp((ndcY + 1) * 0.5, -1, 2));
    this.sunScreenPixels.set(
      this.sunScreenPosition.x * this._viewport.x,
      (1 - this.sunScreenPosition.y) * this._viewport.y
    );
    this.sunOnScreen = facing > 0 && Math.abs(ndcX) <= 1 && Math.abs(ndcY) <= 1;

    // Fade rather than cut. A hard on/off as the sun leaves frame is the classic
    // screen-space-shaft tell.
    const onScreen = 1 - smoothstep(0.55, 1.45, Math.max(Math.abs(ndcX), Math.abs(ndcY)));

    const cloud = clamp01(Math.pow(coverage, 1.4) * (1 + storminess * 0.6));
    this.cloudOcclusion = cloud;

    let canopy = 0;
    const tree = this.tree;
    const cc = tree && tree.canopyCenter;
    // The centre is checked component-wise, not merely for isVector3: it is
    // sibling-owned and it is subtracted from the camera and then square-rooted,
    // so one NaN component would reach sunOcclusion via Math.max - which, like
    // clamp01, propagates NaN rather than absorbing it.
    if (cc && cc.isVector3 && Number.isFinite(cc.x) && Number.isFinite(cc.y) && Number.isFinite(cc.z)) {
      // Ray/sphere against the canopy: cheap, and it makes the shafts flicker as
      // the sun slides behind branches, which is most of the effect's charm.
      _v3c.set(cc.x - camX, cc.y - camY, cc.z - camZ);
      const tca = _v3c.dot(_sunDir);
      if (tca > 0) {
        const perp2 = Math.max(_v3c.lengthSq() - tca * tca, 0);
        const r = (finite(tree.canopyRadius, 8) || 8) * 1.15;
        canopy = 1 - clamp01(Math.sqrt(perp2) / r);
      }
    }
    this.canopyOcclusion = canopy;
    this.sunOcclusion = Math.max(cloud, canopy);

    // Shafts are a partial-occlusion phenomenon. They peak when the disc is half
    // hidden behind the canopy edge, not when it is clear and not when blocked.
    const partial = 4 * canopy * (1 - canopy);
    const lowSun = 1 - smoothstep(0.03, 0.42, _sunDir.y);
    // There are no shafts without something in the air to scatter off. Uses the
    // same normalised dial the mist stack does, so the shafts strengthen exactly
    // as the mist that justifies them becomes visible.
    const airborne = this.mistiness;
    const target = clamp(
      onScreen *
        sunVis *
        (1 - cloud * 0.8) *
        (0.5 + 0.85 * partial) *
        (1 + 0.55 * lowSun) *
        (0.55 + 0.8 * airborne),
      0,
      1.6
    );
    const targetF = finite(target, 0);
    this.godRayIntensity = finite(damp(this.godRayIntensity, targetF, 7, dt), targetF);

    // Renormalise to unit luminance, which is what the export surface promises.
    // The 1e-5 floor alone is not enough: deep at night uAerialMistSun collapses
    // toward the (dark) sky probe, and dividing a 1e-7 colour by 1e-5 yields a
    // 0.01-luminance tint, not a unit one - or exactly black if the sky model
    // hands back zeroes, which would multiply the shafts out of existence
    // instead of merely dimming them. Fall back to the authored warm neutral.
    const gc = this.godRayColor;
    gc.copy(this.uniforms.uAerialMistSun.value);
    const gcLum = luminance(gc);
    if (Number.isFinite(gcLum) && gcLum > 1e-4) gc.multiplyScalar(1 / gcLum);
    else gc.setRGB(SHAFT_FALLBACK_R, SHAFT_FALLBACK_G, SHAFT_FALLBACK_B);
  }

  /**
   * Tag an object and its subtree as a light-shaft occluder. Additive: layer 0
   * membership is untouched, so nothing about how the object normally renders
   * changes. The root is remembered and re-tagged on every scan sweep, so
   * children added later are picked up without the caller having to re-register.
   */
  markOccluder(object) {
    if (!object || !object.isObject3D) return object;
    if (this._occluderRoots.indexOf(object) === -1) this._occluderRoots.push(object);
    object.traverse(this._occluderCallback);
    return object;
  }

  /** Stop re-tagging a root. Already-tagged children keep the layer. */
  unmarkOccluder(object) {
    const roots = this._occluderRoots;
    const i = roots.indexOf(object);
    // Swap-with-last rather than splice: splice returns a new array, and this
    // is reachable from the same sweep that runs in update().
    if (i !== -1) {
      roots[i] = roots[roots.length - 1];
      roots.pop();
    }
    return object;
  }

  _refreshOccluders() {
    const roots = this._occluderRoots;
    // Backwards, so the swap-with-last removal below cannot skip an entry.
    for (let i = roots.length - 1; i >= 0; i--) {
      const root = roots[i];
      // A root that has been torn out of the graph is dead weight; drop it
      // rather than traversing an orphan subtree forever.
      if (!root.parent && root !== this.scene) {
        roots[i] = roots[roots.length - 1];
        roots.pop();
        continue;
      }
      root.traverse(this._occluderCallback);
    }
  }

  // -------------------------------------------------------------------------
  // Shader chunk plumbing
  // -------------------------------------------------------------------------

  _applyAnisotropy() {
    const caps = this.ctx.renderer?.capabilities;
    if (!this._noiseTexture || !caps || typeof caps.getMaxAnisotropy !== 'function') return;
    const aniso = Math.min(this._preset.noiseAniso, caps.getMaxAnisotropy());
    if (this._noiseTexture.anisotropy !== aniso) {
      this._noiseTexture.anisotropy = aniso;
      this._noiseTexture.needsUpdate = true;
    }
  }

  _registerChunks() {
    ShaderChunk.sakura_aerial_pars_vertex = AERIAL_PARS_VERTEX;
    ShaderChunk.sakura_aerial_vertex = AERIAL_VERTEX;
    ShaderChunk.sakura_aerial_pars_fragment = buildAerialParsFragment(this._preset.fogNoiseTaps);
    ShaderChunk.sakura_aerial_fragment = AERIAL_APPLY_FRAGMENT;
  }

  /** The chunk source, for anyone who would rather paste than `#include`. */
  getShaderChunk() {
    return {
      parsVertex: ShaderChunk.sakura_aerial_pars_vertex,
      vertex: ShaderChunk.sakura_aerial_vertex,
      parsFragment: ShaderChunk.sakura_aerial_pars_fragment,
      fragment: ShaderChunk.sakura_aerial_fragment,
      uniforms: this.uniforms,
      names: AtmosphericFX.CHUNK_NAMES,
    };
  }

  /** Copy the shared uniform entries, by reference, into a uniforms object. */
  injectUniforms(target) {
    if (!target) return target;
    const u = this.uniforms;
    for (const key in u) if (target[key] === undefined) target[key] = u[key];
    return target;
  }

  /**
   * Rewrite a material's fog onto the aerial-perspective model. Safe to call
   * more than once, safe on already-compiled materials, and a no-op on
   * materials that do not want fog.
   */
  applyToMaterial(material) {
    if (!material || !material.isMaterial) return material;
    if (material.userData && material.userData.aerial === false) return material;
    if (this._patched.has(material)) return material;

    const usesChunkDirectly =
      typeof material.fragmentShader === 'string' &&
      material.fragmentShader.indexOf('sakura_aerial') !== -1;
    const wantsFogChunks =
      typeof material.fragmentShader === 'string' &&
      material.fragmentShader.indexOf('fog_pars_fragment') !== -1;

    if (material.uniforms) {
      // A ShaderMaterial that includes the chunk needs the uniforms before its
      // first render, or three throws while uploading.
      this.injectUniforms(material.uniforms);
      // three refreshes fogColor/fogDensity on ANY material with fog === true,
      // reading them straight off the uniforms object. Missing entries are a
      // TypeError, so hand out the compatibility set to everything we touch.
      for (const key in this._fogCompat) {
        if (material.uniforms[key] === undefined) material.uniforms[key] = this._fogCompat[key];
      }
    }
    // A custom shader that wired the fog chunks but left the flag off clearly
    // wanted fog; without the flag three never defines USE_FOG and the whole
    // chunk compiles away to nothing.
    if (material.fog !== true && (usesChunkDirectly || wantsFogChunks) && material.uniforms) {
      material.fog = true;
      material.needsUpdate = true;
    }

    const original = {
      onBeforeCompile: material.onBeforeCompile,
      customProgramCacheKey: material.customProgramCacheKey,
    };
    this._patched.set(material, original);
    if (material.userData) material.userData.aerialPatched = true;

    const self = this;
    // Installed on BOTH paths, before the early return. three builds its program
    // cache key from the UNRESOLVED shader source (WebGLShaderCache maps the
    // string to an id) plus the defines plus this hook - it never sees which
    // ShaderChunk bodies the source `#include`s. A material that takes the chunk
    // directly therefore keeps its first-compiled program for the life of the
    // session, so onQualityChange()'s needsUpdate would re-resolve nothing and
    // the tier switch would be silently ignored on exactly the custom shaders
    // most likely to be expensive.
    material.customProgramCacheKey = function () {
      const base = original.customProgramCacheKey ? original.customProgramCacheKey.call(this) : '';
      return base + '|aerial' + self._variant;
    };

    if (usesChunkDirectly) return material;

    material.onBeforeCompile = function (shader, renderer) {
      original.onBeforeCompile?.call(this, shader, renderer);
      const vertex = patchVertexSource(shader.vertexShader);
      const fragment = patchFragmentSource(shader.fragmentShader);
      // All or nothing. A patched fragment reading a varying the vertex never
      // wrote is a link error, and a link error is a black screen.
      if (vertex === null || fragment === null) return;
      shader.vertexShader = vertex;
      shader.fragmentShader = fragment;
      self.injectUniforms(shader.uniforms);
    };
    material.needsUpdate = true;
    return material;
  }

  _considerObject(obj) {
    if (!obj || obj.userData?.noAerial === true || obj.userData?.aerial === false) return;
    const material = obj.material;
    if (!material) return;
    if (Array.isArray(material)) {
      for (let i = 0; i < material.length; i++) this._considerMaterial(material[i]);
    } else {
      this._considerMaterial(material);
    }
  }

  _considerMaterial(material) {
    if (!material || !material.isMaterial) return;
    if (this._patched.has(material)) return;
    if (material.userData && material.userData.aerial === false) return;
    // Raw shaders own their entire source including the version directive; there
    // is no safe automatic injection point in one.
    if (material.isRawShaderMaterial) return;

    const src = typeof material.fragmentShader === 'string' ? material.fragmentShader : '';
    // Uniform-only adoption is free and must never be rationed: a shader that
    // includes the chunk without the uniforms crashes on its very first frame.
    if (src.indexOf('sakura_aerial') !== -1) {
      this.applyToMaterial(material);
      return;
    }
    // A custom shader that wired the fog chunks but left `fog` false still
    // counts as opting in - applyToMaterial flips the flag and supplies the
    // uniforms three would otherwise crash looking for.
    const wantsFog = src.indexOf('fog_pars_fragment') !== -1 && !!material.uniforms;
    if (material.fog !== true && !wantsFog) return;
    if (this._patchBudget <= 0) return;
    this._patchBudget--;
    this.applyToMaterial(material);
  }

  _scanScene() {
    this.scene.traverse(this._scanCallback);
  }
}

AtmosphericFX.CHUNK_NAMES = Object.freeze([
  'sakura_aerial_pars_vertex',
  'sakura_aerial_vertex',
  'sakura_aerial_pars_fragment',
  'sakura_aerial_fragment',
]);

// ---------------------------------------------------------------------------
// Source rewriting
// ---------------------------------------------------------------------------

function patchVertexSource(src) {
  if (typeof src !== 'string') return null;
  if (src.indexOf('sakura_aerial') !== -1) return src;
  if (src.indexOf('#include <fog_pars_vertex>') === -1) return null;
  if (src.indexOf('#include <fog_vertex>') === -1) return null;
  return src
    .replace('#include <fog_pars_vertex>', '#include <sakura_aerial_pars_vertex>')
    .replace('#include <fog_vertex>', '#include <sakura_aerial_vertex>');
}

function patchFragmentSource(src) {
  if (typeof src !== 'string') return null;
  if (src.indexOf('sakura_aerial') !== -1) return src;
  if (src.indexOf('#include <fog_pars_fragment>') === -1) return null;

  let out = src.replace('#include <fog_pars_fragment>', '#include <sakura_aerial_pars_fragment>');

  // Applying before tone mapping keeps the whole thing in linear space, which is
  // the only space where mixing toward a physical sky radiance means anything.
  // three's own fog runs after the output transform and cannot match a sky dome.
  if (out.indexOf('#include <tonemapping_fragment>') !== -1) {
    out = out.replace(
      '#include <tonemapping_fragment>',
      '#include <sakura_aerial_fragment>\n\t#include <tonemapping_fragment>'
    );
  } else if (out.indexOf('#include <fog_fragment>') !== -1) {
    out = out.replace('#include <fog_fragment>', '#include <sakura_aerial_fragment>');
  } else {
    return null;
  }

  // Drop three's fog so it cannot apply a second time on top of ours.
  return out.replace('#include <fog_fragment>', '');
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/**
 * NaN firewall.
 *
 * Every helper this module leans on propagates NaN rather than absorbing it:
 * `clamp` is a pair of comparisons and every comparison against NaN is false, so
 * `clamp( NaN, 0, 1 )` is NaN; `smoothstep`, `lerp` and `damp` inherit that. On
 * its own that would be a one-frame blemish, but this module feeds those results
 * into DAMPED ACCUMULATORS that are read back the next frame - `_mistHeight`,
 * `_mistBase`, `_flash`, `_driftA/_driftB`, `uAerialMist.x`, `uAerialHaze.x`,
 * `godRayIntensity`. `damp( NaN, x )` is NaN forever, so a single bad frame in
 * `state.wind`, `state.sun.direction`, `state.clouds.coverage` or the camera
 * transform LATCHES. And the failure is not local: a NaN in `uAerialMist.x`
 * makes `aerialOpticalDepth` return NaN, so `mix( color, inscatter, NaN )` is
 * NaN in every fog-adopted material in the scene - terrain, grass, tree, petals
 * - permanently, until reload. Sanitising at each accumulator is the one cheap
 * place that closes all of those paths at once.
 */
function finite(x, fallback) {
  return Number.isFinite(x) ? x : fallback;
}

/**
 * NaN firewall for a sibling-owned colour.
 *
 * The scalar `finite()` above is not enough on its own, because the guards this
 * file leans on do not absorb NaN either: `Math.max( NaN, 1e-5 )` is NaN, so
 * every `Math.max( luminance( x ), 1e-5 )` in the palette passes one straight
 * through, and one NaN component then reaches `mix( color, inscatter, f )` in
 * every fog-adopted material in the scene. Colours are checked whole because a
 * partially-NaN colour is never meaningful.
 */
function safeColor(src, out, r, g, b) {
  if (src && Number.isFinite(src.r) && Number.isFinite(src.g) && Number.isFinite(src.b)) {
    return out.copy(src);
  }
  return out.setRGB(r, g, b);
}

/** Same, for a direction. Also renormalises, so a stale unnormalised vector is safe. */
function safeDir(src, out, x, y, z) {
  if (src && Number.isFinite(src.x) && Number.isFinite(src.y) && Number.isFinite(src.z)) {
    out.copy(src);
    const l = out.lengthSq();
    if (l > 1e-12) return out.multiplyScalar(1 / Math.sqrt(l));
  }
  return out.set(x, y, z);
}

/** Wrap wind drift into [0, NOISE_TILE) metres - one whole tile, so seamless. */
function wrapTile(x) {
  return x - Math.floor(x / NOISE_TILE) * NOISE_TILE;
}

/** Gaussian bump over a wrapped 24-hour clock. */
function bell(hour, center, width) {
  let d = hour - center;
  if (d > 12) d -= 24;
  else if (d < -12) d += 24;
  const t = d / width;
  return Math.exp(-t * t);
}

/** Nudge a colour toward a hue while preserving its luminance. */
function pushHue(color, r, g, b, amount) {
  const before = luminance(color);
  color.r = lerp(color.r, color.r * r, amount);
  color.g = lerp(color.g, color.g * g, amount);
  color.b = lerp(color.b, color.b * b, amount);
  const after = luminance(color);
  if (after > 1e-6) color.multiplyScalar(before / after);
  return color;
}
