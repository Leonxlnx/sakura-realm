/**
 * tree/blossoms.js - the corolla layer: 122 400 instanced blossom cards.
 *
 * ---------------------------------------------------------------------------
 * WHY IT IS BUILT THIS WAY
 * ---------------------------------------------------------------------------
 * A cherry in full bloom is a CLOUD of flowers with real thickness, and this
 * file used to say the opposite - "not a cloud of flowers, it is a SHELL of
 * them" - and was built accordingly, on a `smoothstep` window in a normalised
 * ellipsoid radius. That is where the hollow-bubble crown came from, and it is
 * worth being precise about why it is wrong, because the shell story is
 * plausible and it survived several passes.
 *
 * It is right that flowers open where light reached. It is wrong that "where
 * light reached" is a surface. Light entering a canopy is attenuated
 * EXPONENTIALLY with the depth of foliage it has crossed, so the flowering rate
 * is a smooth function with a length scale of a couple of metres - strong for
 * the outer two, half strength 1.6 m in, an eighth of it four metres in. That
 * is a cloud several metres thick over a bare interior, and it is what the
 * reference photographs show: an opaque mass with only the dark main limbs
 * visible near the trunk.
 *
 * The shell test also had a second, non-obvious failure that the depth model
 * fixes for free. The twig cloud is not uniform - it carries roughly 360 sites
 * per cubic metre halfway out and 32 at the rim - so a rule that put flowers on
 * the surface still produced a crown that was DENSEST in the middle, where a
 * view ray crosses 13 m of it, and thinnest at the rim where a ray crosses
 * four. Measured coverage of the crown's own outline was 52 %, essentially all
 * of the loss at the edges. The exponential runs the other way and cancels most
 * of that fall.
 *
 *   - Placement walks the actual twig cloud (`collectTwigSites` from
 *     branches.js) rather than scattering into a volume, so no flower is ever
 *     in mid-air and every one of them inherits the EXACT sway weights of the
 *     twig holding it. Blossom and wood are displaced by identical shader
 *     maths, so they cannot separate however hard the wind blows.
 *   - Density per site is Beer's law along TWO paths - up through the crown to
 *     the sky and sideways out through the flank - against the crown envelope
 *     branches.js grew the wood against. Both paths are needed: with the
 *     horizontal one alone, the skirt (where the crown is only 3 m across)
 *     scored as fully lit everywhere and 54 % of the whole canopy piled into
 *     the two bands between 2 m and 3 m.
 *   - Flowers stand off the wood on a SPUR, 2 to 13 cm plus the branch's own
 *     radius, because that is where a cherry flowers and because a flowered
 *     twig is a tube: widening the tube fills the space between neighbouring
 *     twigs and costs no instances at all.
 *   - Flowers come in umbels of three to five off one bud, because that is how
 *     Prunus flowers. Clustering is the single largest visual difference
 *     between this and a uniform scatter: it produces the popcorn texture the
 *     eye reads as sakura. It is also the thing most easily overdone, and it
 *     defeated three separate attempts to fix this canopy. COVERAGE IS SITES
 *     TIMES CARD AREA; clusters only control how many cards land on top of each
 *     other at one point, and once an umbel is wider than the twig it sits on,
 *     every extra flower in it is hidden behind the ones already there. The
 *     tree now offers 145 000 sites at 34 mm along the wood and fewer than one
 *     umbel lands on each. It was 2 947 sites carrying fourteen flowers apiece,
 *     and the difference between those two shapes is the whole of "the canopy
 *     looks bare".
 *   - Shading is Lambert, not Standard. Petals are matte; the entire reason
 *     they are beautiful is TRANSMISSION, which no BRDF in three provides, so
 *     the GGX lobe would be pure cost. The saved instructions buy the
 *     subsurface term instead - and on the 780M, a canopy that can fill the
 *     screen is exactly where a cheap fragment shader pays.
 *
 * One InstancedBufferGeometry, one draw call, one static upload at build time.
 * Nothing in here allocates after `build()`.
 */

import * as THREE from 'three';
import { makeRNG, clamp01, lerp, smoothstep, TAU } from '../core/math.js';
import { SAKURA_WIND_GLSL } from './branches.js';

// ===========================================================================
// Layout
// ===========================================================================

/**
 * Floats per instance in the interleaved buffer.
 *   0..2   position on the twig (tree-local metres)
 *   3..5   card "right" axis, unit
 *   6..8   card face normal, unit
 *   9..12  sway weights A (limb, branch), copied verbatim from the twig
 *   13..16 sway weights B (branchlet, twig)
 *   17..20 size, tint, budness, flutter phase
 *   21     baked crown occlusion (0 = buried in the canopy, 1 = on the shell)
 */
export const BLOSSOM_STRIDE = 22;

/**
 * Ceiling for the one-time scratch buffer. The shipping skeleton generates
 * 122 400 across 145 000 twig sites; the headroom exists so a future skeleton
 * change cannot SILENTLY truncate the canopy - hitting this cap does not throw,
 * it simply stops placing flowers, and the first symptom is a bald patch on
 * whichever side of the tree the site list happened to reach last. The buffer
 * is transient: the mesh gets an exact-size copy.
 *
 * This cap is not theoretical and it is no longer distant: it fired four times
 * during the crown-envelope rebuild, and every time the run before the warning
 * was read had a visibly lopsided canopy. There is now only 10 % of clearance,
 * so ANY change that adds flowering wood must be paired with a look at
 * CLUSTER_RATE. Do not answer a bare-looking crown by raising this - read the
 * note on CLUSTER_RATE first, and then the one on TWIG_SPACING in sakura.js.
 */
const HARD_MAX = 850000;
/** Petal spawn sites handed to tree/petals.js. */
const SPAWN_MAX = 1600;

/**
 * Per-tier canopy budget. `fraction` is how many of the generated flowers the
 * tier draws - the instance order is shuffled at build time, so ANY prefix is
 * an even thinning of the whole canopy, no rebuild and no popping region - and
 * `boost` grows the survivors so it does not simply go bald.
 *
 * FILL COST GOES AS `fraction * boost^2`, and that product is the number to
 * read, because on this GPU the canopy's cost is fill, not vertices: it is the
 * one material in the scene that can cover the screen in double-sided
 * alpha-tested fragments, each of which pays a Lambert lobe, a transmission
 * lobe and (where `receive` is on) up to two rotated-disc PCF lookups of nine
 * taps each.
 *
 * MEASURED, on the shipping seed, against the canopy as it stood before the
 * crown-envelope rebuild (74 373 generated, 66 936 drawn at HIGH at a 37.7 mm
 * mean half-size, 381 m² of quad). "Quad area" is the total area of the drawn
 * cards in square metres and it is the number the frame budget is expressed in,
 * because on this GPU the canopy's cost is fill:
 *
 *   tier    fraction  boost   active    mean half   quad m²   vs before
 *   low       0.13    1.30    15 900     43.4 mm       123      0.32
 *   medium    0.28    1.12    34 300     37.2 mm       195      0.51
 *   high      0.84    1.02   102 800     33.7 mm       482      1.27
 *   ultra     1.00    1.00   122 400     33.0 mm       551      1.45
 *
 * Read that table as the whole cost story of this pass. HIGH buys 27 % more
 * fill and 54 % more instances, and it is spent on a crown that is a third
 * wider and genuinely opaque rather than on bigger flowers. LOW and MEDIUM - 
 * the tiers the 780M will actually sit on - come DOWN by two thirds and a half,
 * which is where the extra HIGH fill is paid for, along with a wood mesh that
 * draws 1 750 branches out of 15 200 and a distance LOD that now starts at 22 m
 * instead of 30.
 *
 * `boost` used to be DERIVED as sqrt(count/active), i.e. exact area
 * compensation, capped at 1.45. Two things were wrong with that. The cap only
 * bit below fraction 0.48, so MEDIUM came out at 93 % of ULTRA's fill while
 * drawing 44 % of the triangles - the tier that this laptop GPU will actually
 * sit on was barely a degradation at all. And because exact compensation makes
 * projected area INVARIANT to `fraction`, the obvious remediation dial ("turn
 * the fraction down") moved triangle count and nothing else, which is not what
 * costs the frame. The two numbers are now independent and honest: lower
 * `fraction` and the canopy really does get cheaper.
 *
 * `receive` is the other half of the shadow trade. Turning it off at LOW pulls
 * the PCF lookup out of the most fill-bound shader in the frame; it is a
 * uniform-valued branch (three sets `receiveShadow` per object), so the whole
 * draw takes the cheap path together. The baked crown occlusion carried in the
 * instance stream is what keeps that tier from shading dead flat.
 */
const TIERS = {
  low: { fraction: 0.09, boost: 1.30, shadow: false, receive: false },
  medium: { fraction: 0.19, boost: 1.12, shadow: true, receive: true },
  // Coverage was never short of instances - it was short of PLACES. 41 461
  // flowers stacked fourteen deep on 2 947 twig points read as pom-poms on bare
  // wire; the same crown spread over 145 000 sites at less than one umbel each
  // is continuous. See CLUSTER_RATE, and SIZE_MIN for why card AREA is the
  // other half of it.
  high: { fraction: 0.66, boost: 1.06, shadow: true, receive: true },
  ultra: { fraction: 0.84, boost: 1.08, shadow: true, receive: true },
};

/** Card rim lift, in card half-widths. A cherry corolla is a shallow bowl. */
const CARD_DISH = 0.15;

/**
 * Beer's-law extinction lengths for the FLOWERING rate, in metres of canopy - 
 * one for light arriving from the sky above, one for light through the open
 * flank. `SKY_SHARE` splits the two: most of the irradiance on a crown comes
 * down through it, which is why the deep interior of a cherry is bare however
 * close the flank happens to be.
 *
 * These are set by the shape the reference photographs show: a flower mass with
 * real thickness - strong for the outer two metres, half strength 1.6 m in, an
 * eighth of it four metres in - over an interior that is bare wood and shadow.
 * Halve them and the crown is a skin over a void again (that is the failure the
 * old shell test had); double them and the flowers spread evenly through the
 * whole volume, which costs a third more instances and puts them where nothing
 * can see them.
 */
const SKY_SHARE = 0.64;
const INV_SKY_DEPTH = 1 / 5.40;
const INV_SIDE_DEPTH = 1 / 4.10;
/**
 * Extinction for the baked SHADE term. Longer than the flowering ones on
 * purpose: what reaches a buried flower is diffuse skylight forward-scattered
 * through the petals in front of it, and a cherry petal passes most of what
 * hits it. A crown that darkens as fast as it thins reads as a hole.
 */
const INV_SHADE_SKY = 1 / 4.6;
const INV_SHADE_SIDE = 1 / 3.4;
/**
 * Umbels per site at full bloom rate. This is a SCALE and nothing else - the
 * radial shape lives in the exponentials above - so it sets the instance count
 * and, through it, the canopy's cost. See HARD_MAX and TIERS.
 *
 * Below 1, which is the thing to understand about it: the tree offers 145 000
 * sites and places 122 000 flowers in umbels of three to five, so fewer than
 * one site in four gets an umbel at all and the fractional part is resolved by
 * a coin flip per site. The site walk is therefore no longer "where a flower
 * goes" - it is a fine sampling of the wood, and the acceptance turns it into a
 * stochastic scatter along every twig. That is why the pitch in sakura.js can
 * be finer than the flower without wasting anything.
 */
const CLUSTER_RATE = 1.85;

/**
 * How much skylight joins the direct beam in the transmission illuminant.
 *
 * Not a look dial so much as a units conversion: `sun.intensity` peaks at 3.35
 * while `sky.ambientIntensity` is order 1, so this is what puts the two on a
 * comparable footing. At noon the beam still outweighs the sky about three to
 * one, which is correct; at golden hour, once the beam has lost two thirds of
 * its strength to air mass, they are close, and that is exactly when the
 * canopy needs the sky's green and blue.
 */
const GLOW_SKY = 1.95;
/**
 * Extra weight the SKY gets in the transmission illuminant as the sun drops.
 *
 * This is the fix for the canopy going rust at golden hour, and it is a real
 * effect rather than a look dial: as the sun approaches the horizon its beam
 * loses most of its energy to air mass while the sky keeps scattering, so the
 * DIFFUSE fraction of the irradiance on a crown rises steeply - by late golden
 * hour it is the majority of it. Building the glow from a beam that has
 * bottomed out at (1.0, 0.17, 0.065) can only ever produce orange, however pale
 * the petal is, which is exactly what the renders showed.
 */
const GLOW_SKY_LOW = 0.85;

/**
 * Flower diameter is ~1.8x the card half-size once the alpha margin is gone.
 * At HIGH the mean half-size lands at 33.7 mm, a 6.1 cm corolla against a real
 * Somei-Yoshino's 3-4 cm, so one card stands in for about two real flowers.
 *
 * These went UP by 18 % on the linear measure - 40 % on area - and that is a
 * deliberate reversal of the previous pass, which cut them. Card area is the
 * canopy's whole cost on this GPU and it is quadratic in this number, so it is
 * the single most dangerous constant in the file. But it is also the ONLY lever
 * that buys coverage without buying instances, and instances are what the
 * shadow pass and the vertex shader are charged for: at a fixed quad area,
 * bigger cards mean fewer of them. With HARD_MAX 10 % away and the crown
 * needing to read as opaque from 20 m, taking the coverage in area rather than
 * in count was the cheaper half of the trade.
 *
 * Do not push further. At 11.5 cm - which is where this sat two passes ago - 
 * the canopy is visibly built out of individual blobs, and the coral-like
 * clumps in the old close renders were exactly that, not a shading problem.
 */
const SIZE_MIN = 0.0300;
const SIZE_MAX = 0.0505;

/**
 * Transmission lobe shape - the y and z of `uBSss`. See BLOSSOM_SSS.
 *
 * These live here, as named constants, because they used to be written TWICE:
 * once in the uniform's initialiser and again, with different values, in
 * `update()`. `update()` runs on every frame including the first, so the
 * initialiser lost every time and the shipping build ran the OLD tight lobe
 * (power 3.2, rim 0.5) that the whole golden-hour pass existed to widen - the
 * forward-scatter term carries 0.85 of the two-lobe sum, so that silently threw
 * away most of the fix. Only the wetness-dependent SCALE belongs in `update()`.
 */
const SSS_POWER = 1.8;
const SSS_RIM = 0.55;

// ===========================================================================
// Placement - pure, deterministic, testable without a GL context
// ===========================================================================

/**
 * Turns the twig cloud into instance data.
 *
 * Exported separately from the renderer so the distribution can be measured
 * headlessly; nothing in here touches three.js beyond plain arrays.
 *
 * @param {object} skel   from generateSkeleton()
 * @param {Array}  sites  from collectTwigSites()
 * @returns {{data: Float32Array, count: number, spawn: object, sphere: object}}
 */
export function generateBlossomInstances(skel, sites, opts = {}) {
  const seed = opts.seed ?? 0x5a11a9;
  const maxInstances = Math.min(opts.maxInstances ?? HARD_MAX, HARD_MAX);
  const rng = makeRNG(seed);

  /**
   * The crown envelope the WOOD was grown against, straight off the skeleton.
   *
   * This used to be re-derived here from `canopyRadius` / `canopyRadiusY`,
   * which are 90th percentiles of the twig cloud - so the flower mass and the
   * wood disagreed about where the crown was, by however much the percentile
   * happened to move that seed. One envelope, one definition, published by
   * branches.js. The fallback keeps this function usable against a skeleton
   * from an older build or a hand-built test fixture.
   */
  const env = skel.envelope || {
    r: Math.max(skel.canopyRadius, 0.5),
    y0: skel.canopyCenter.y,
    ryUp: Math.max(skel.canopyRadiusY || 3, 0.4),
    ryDown: Math.max(skel.canopyRadiusY || 3, 0.4),
  };
  const envR = Math.max(env.r, 0.5);
  const envY0 = env.y0;
  const envRyUp = Math.max(env.ryUp, 0.4);
  const envRyDown = Math.max(env.ryDown, 0.4);
  const invEnvR2 = 1 / (envR * envR);
  const invRyUp2 = 1 / (envRyUp * envRyUp);
  const invRyDown2 = 1 / (envRyDown * envRyDown);

  const cc = skel.canopyCenter;
  const ccx = cc.x, ccy = cc.y, ccz = cc.z;

  const scratch = new Float32Array(maxInstances * BLOSSOM_STRIDE);
  let n = 0;

  // Reused per flower; build-time only, but keeping them out of the loop body
  // is free and keeps the shapes monomorphic.
  let axX = 0, axY = 0, axZ = 0;
  let e1x = 0, e1y = 0, e1z = 0;
  let e2x = 0, e2y = 0, e2z = 0;

  for (let si = 0; si < sites.length; si++) {
    const s = sites[si];

    // --- how much bloom this site carries ---------------------------------
    const ey = s.y - envY0;
    const eh = s.x * s.x + s.z * s.z;
    // Normalised envelope coordinate: 1.0 is the crown's own outline, and it is
    // the SAME number branches.js grew the wood against.
    const rad = Math.sqrt(eh * invEnvR2 + ey * ey * (ey >= 0 ? invRyUp2 : invRyDown2));
    // Horizontal radius of the envelope at this height, so `inward` below comes
    // out in metres rather than in units of a normalised coordinate.
    const envAtY = envR * Math.sqrt(Math.max(0, 1 - ey * ey * (ey >= 0 ? invRyUp2 : invRyDown2)));
    /**
     * METRES OF CANOPY BETWEEN THIS SITE AND THE SKY, and then Beer's law
     * through it. This replaced a `smoothstep(0.26, 0.86, rad)` SHELL test, and
     * the difference is the whole of "the crown reads as a hollow bubble".
     *
     * A shell test says a flower is either on the surface or not. What decides
     * whether a bud opens is how much light reaches it, and light entering a
     * canopy is attenuated exponentially with the depth of foliage it crossed - 
     * a smooth function of position with a real length scale, not a window.
     * Written this way the flowering band has DEPTH: strong for the outer two
     * metres, half strength 1.6 m in, an eighth of it four metres in. The crown
     * is a cloud several metres thick over a bare interior, not a skin.
     *
     * TWO PATHS, NOT ONE, and the second one is not optional. The first version
     * of this measured depth as `(1 - rad) · envelopeRadius`, i.e. horizontally,
     * and that is wrong wherever the envelope is narrow: down in the skirt the
     * crown is only 3 m across, so every site there came out "0.3 m from the
     * surface" and flowered at full rate - including the ones a metre from the
     * bole with eight metres of crown stacked over them. Measured, that put
     * 54 % of the entire canopy into the two bands between 2 m and 3 m, a dense
     * shelf round the trunk with the flanks above it left thin. Light arrives
     * from the sky ABOVE and through the open FLANK, so both distances have to
     * be in the model, and the sky path has to carry the larger share.
     *
     * This is also what makes the crown opaque, for a reason worth writing down
     * because it is not obvious and it is the measurement that drove the pass.
     * The twig cloud is not uniform: it carries 85 sites per cubic metre at 40 %
     * of the crown radius and 13 at the rim, a 6.5:1 fall. Card area per cubic
     * metre under the old shell model therefore fell from 0.50 to 0.11 outward - 
     * the crown was DENSEST in the middle, where a view ray crosses 13 m of it,
     * and thinnest at the rim where a ray crosses four. Coverage came out at
     * 52 %, essentially all of the loss at the edges, which is exactly what the
     * renders show. These exponentials run the other way and cancel that fall,
     * so the same number of flowers covers the sky.
     */
    const hr = Math.sqrt(eh);
    const ht = hr / envR;
    // Height of the crown's own upper surface directly above this site - the
    // sky path, and the one that decides whether the interior is bare.
    const yTop = envY0 + envRyUp * Math.sqrt(Math.max(0, 1 - ht * ht));
    const dUp = yTop > s.y ? yTop - s.y : 0;
    // Horizontal distance out through the flank.
    const dSide = envAtY > hr ? envAtY - hr : 0;
    const lit = Math.max(
      0.42,
      SKY_SHARE * Math.exp(-dUp * INV_SKY_DEPTH)
        + (1 - SKY_SHARE) * Math.exp(-dSide * INV_SIDE_DEPTH)
    );
    /**
     * The ragged outer margin. Everything the reference shows of the sky is
     * here - a soft edge one crown-radius-eighth wide - and it is the one place
     * a cherry is genuinely see-through. Beyond 1.30 nothing flowers at all,
     * which is also what stops the handful of shoots that escaped the envelope
     * from reading as whiskers.
     */
    const margin = 1 - smoothstep(1.02, 1.30, rad);
    // Secondary texture, all of it modest: these decide which twigs within a
    // neighbourhood flower hardest, not where the crown is.
    const light = smoothstep(0.46, 0.92, s.ao);
    // Spur wood: 3 mm twiglets flower, 25 mm branches carry the spurs.
    const young = smoothstep(0.026, 0.005, s.r);
    // Flowering runs out toward the base of each shoot.
    const tip = smoothstep(0.06, 0.80, s.tipness);

    /**
     * Baked crown occlusion, carried into the instance stream. Until this
     * existed the canopy had NO ambient occlusion of any kind: a flower buried
     * a metre inside the crown was lit exactly as brightly as one on the rim
     * the moment the sun stopped casting into it, and at LOW - where
     * `receiveShadow` is off - that is every flower, all the time.
     *
     * Same depth measure as the bloom rate, with a LONGER extinction: what
     * reaches a shaded flower is diffuse skylight forward-scattered through the
     * petals in front of it, and petals are thin enough that a great deal gets
     * through. The rim saturates at 1.0 by construction, so the flowers that
     * carry the silhouette keep their full brightness and only the interior
     * darkens - otherwise this is a global exposure change rather than
     * occlusion.
     */
    const openness = clamp01(
      0.20 + 0.95 * (SKY_SHARE * Math.exp(-dUp * INV_SHADE_SKY)
        + (1 - SKY_SHARE) * Math.exp(-dSide * INV_SHADE_SIDE))
    );

    let w = lit * margin * (0.80 + 0.20 * young) * (0.82 + 0.18 * tip) * (0.86 + 0.14 * light);
    // Blossom is patchy at the metre scale - one bough goes over before its
    // neighbour opens. Without this the canopy reads as a uniform spray.
    const patch = hashPatch(s.x, s.y, s.z);
    w *= 0.72 + 0.52 * patch;
    w = clamp01(w);

    /**
     * Umbels at this site, with the fractional part resolved stochastically so
     * the density field is continuous rather than quantised into visible steps.
     *
     * THE MULTIPLIER IS NOW A SCALE, NOT A SHAPE. `w` above already carries the
     * whole radial gradient, so the exponent that used to sit here (1.55, to
     * "send the surplus to the merely-good sites") would compound with it and
     * hollow the crown back out; it is gone. What is left is one number that
     * sets the total flower count, and the note below is why raising it is
     * almost never what a bare-looking crown needs.
     *
     * THE MULTIPLIER WENT 2.97 -> 4.15 -> 5.6 CHASING A BARE-LOOKING CROWN, AND
     * EVERY ONE OF THOSE STEPS MADE IT WORSE. This is worth writing down because
     * it is the trap this file is built around. `clusters` is how DEEP the
     * flowers pile at one point on one twig, and the crown was not short of
     * depth - at 5.6 it was carrying fourteen flowers on each of 2 947 twig
     * points, which is a string of pom-poms, and every extra flower went into
     * making the pom-poms bigger while the 92 mm of bare wood between them
     * stayed exactly as bare. That is precisely the "clumps on bare sticks" the
     * renders show.
     *
     * The fix was upstream, in two places that cost nothing per frame:
     * `collectTwigSites` now interpolates along the wood instead of visiting
     * skeleton points (2 947 sites -> 22 286), and `branches.js` stopped killing
     * fine shoots at half their length (845 branches -> 2 561). With six times
     * the places to put a flower, the multiplier comes DOWN by a factor of four
     * and the same crown carries 45 % more flowers spread one umbel per ~50 mm
     * of twig, which is what a cherry actually does.
     *
     */
    const wf = w * CLUSTER_RATE;
    let clusters = Math.floor(wf);
    if (rng() < wf - clusters) clusters++;
    if (clusters <= 0) continue;

    // Site frame: the twig tangent, plus two perpendiculars for the offsets.
    const tx = s.tx, ty = s.ty, tz = s.tz;
    perp(tx, ty, tz);
    e1x = _px; e1y = _py; e1z = _pz;
    e2x = ty * e1z - tz * e1y;
    e2y = tz * e1x - tx * e1z;
    e2z = tx * e1y - ty * e1x;

    // Outward: flowers face away from the crown's core and a little toward the
    // sky. Measured from the ENVELOPE centre, not the twig-cloud centroid: it
    // is the same origin the depth measure above uses, so a flower's face and
    // its exposure can never disagree about which way is out.
    let ox = s.x, oy = s.y - envY0, oz = s.z;
    const ol = Math.hypot(ox, oy, oz) || 1;
    ox /= ol; oy = oy / ol + 0.30; oz /= ol;
    const ol2 = Math.hypot(ox, oy, oz) || 1;
    ox /= ol2; oy /= ol2; oz /= ol2;

    for (let c = 0; c < clusters && n < maxInstances; c++) {
      // Buds sit at nodes along the twig. The spread has to match the site
      // pitch or the umbels land on a comb: sites are 34 mm apart now, so this
      // is a hair wider than the gap and adjacent umbels interleave.
      const along = (rng() - 0.5) * 0.040;
      const bx = s.x + tx * along;
      const by = s.y + ty * along;
      const bz = s.z + tz * along;

      // The umbel's own axis leaves the twig sideways, not along it.
      const spurAz = rng() * TAU;
      const ca = Math.cos(spurAz), sa = Math.sin(spurAz);
      const spurTilt = 0.55 + 0.5 * rng();
      const st = Math.sin(spurTilt), ct = Math.cos(spurTilt);
      axX = (e1x * ca + e2x * sa) * st + tx * ct;
      axY = (e1y * ca + e2y * sa) * st + ty * ct;
      axZ = (e1z * ca + e2z * sa) * st + tz * ct;
      // ...then bends toward the light like everything else on the tree.
      axX = axX * 0.45 + ox * 0.55;
      axY = axY * 0.45 + oy * 0.55;
      axZ = axZ * 0.45 + oz * 0.55;
      const al = Math.hypot(axX, axY, axZ) || 1;
      axX /= al; axY /= al; axZ /= al;

      /**
       * THE SPUR. A cherry does not flower on its twigs, it flowers on short
       * lateral shoots that stand 3 to 9 cm clear of them, and the umbel opens
       * at the end of that. Modelling it as a point on the wood - which is what
       * this did - is the single largest reason the crown could not be made
       * opaque without absurd instance counts.
       *
       * The measurement: a flowered twig is a TUBE, and what a view ray hits is
       * the tube, not the wood. Flowers pinned to the twig gave a tube 14 cm
       * across; twigs run broadly parallel inside a spray, so those tubes
       * overlapped each other and the sprays read as fingers with sky between
       * them. Nominal optical depth through the crown was 3.2 - which would be
       * 96 % coverage if the wood were spread evenly - while measured coverage
       * was 53 %, i.e. a clumping factor of 4.3. Standing the flowers off the
       * wood widens the tube to about 26 cm, and because the widening is
       * ISOTROPIC around each twig it fills the space BETWEEN neighbours rather
       * than deepening what is already covered. It buys coverage at no instance
       * cost at all, which nothing else in this file does.
       *
       * The spur is drawn from a squared distribution so most are short: a
       * canopy of uniformly long spurs reads as fuzz, and the silhouette wants
       * a dense core with a few flowers standing proud of it.
       */
      // The site is on the twig's AXIS, so the wood's own radius has to be
      // cleared before the spur starts or the umbel opens inside the branch.
      // It cost nothing while flowering was confined to 3 mm twiglets; now that
      // spurs are allowed out onto 5 cm wood - which is where they grow on a
      // real cherry, and which is the only thing that clothes the outer half of
      // a limb - it is the difference between a flowered bough and a bough with
      // flowers buried in it.
      const spurU = rng();
      const spur = s.r + 0.020 + 0.115 * spurU * spurU;
      const ux0 = bx + axX * spur;
      const uy0 = by + axY * spur;
      const uz0 = bz + axZ * spur;

      // Shaded clusters are still in bud - a real tree is never uniformly open,
      // and the decision belongs to the UMBEL, not the flower: one bud opens
      // its three-to-five flowers within a day of each other. Clusters of buds
      // among open clusters is what a cherry mid-bloom actually looks like; a
      // random 15% of individual flowers being bud looks like noise.
      const budCluster = rng() < 0.015 + 0.12 * (1 - light) + 0.06 * (1 - tip);
      const flowers = 3 + ((rng() * 3) | 0);

      // The umbel's frame is fixed for the whole cluster, so it is built once
      // here rather than three to five times inside the flower loop. `perp`
      // draws no random numbers, so this is bit-identical to the old order and
      // saves ~50 000 hypot/normalise pairs over the tree.
      perp(axX, axY, axZ);
      const u1x = _px, u1y = _py, u1z = _pz;
      const u2x = axY * u1z - axZ * u1y;
      const u2y = axZ * u1x - axX * u1z;
      const u2z = axX * u1y - axY * u1x;

      for (let f = 0; f < flowers && n < maxInstances; f++) {
        // Pedicels fan out from the bud in a cone. sqrt keeps the fan even in
        // solid angle rather than piling every flower on the axis.
        const cone = 0.30 + 0.75 * Math.sqrt(rng());
        const az = rng() * TAU;
        const cz2 = Math.cos(az), sz2 = Math.sin(az);
        const sc = Math.sin(cone), cc2 = Math.cos(cone);
        let dxf = axX * cc2 + (u1x * cz2 + u2x * sz2) * sc;
        let dyf = axY * cc2 + (u1y * cz2 + u2y * sz2) * sc;
        let dzf = axZ * cc2 + (u1z * cz2 + u2z * sz2) * sc;

        // Pedicel: 2-5 cm on a Somei-Yoshino, and it sags under the flower.
        const ped = 0.018 + 0.034 * rng();
        const px2 = ux0 + dxf * ped;
        const py2 = uy0 + dyf * ped - ped * 0.30;
        const pz2 = uz0 + dzf * ped;

        // The flower faces along its pedicel, nodding slightly downward - the
        // droop is why a cherry canopy reads as heavy rather than as a spray.
        let nx = dxf * 0.70 + ox * 0.30 + (rng() - 0.5) * 0.55;
        let ny = dyf * 0.70 + oy * 0.30 + (rng() - 0.5) * 0.55 - 0.20;
        let nz = dzf * 0.70 + oz * 0.30 + (rng() - 0.5) * 0.55;
        const nl = Math.hypot(nx, ny, nz) || 1;
        nx /= nl; ny /= nl; nz /= nl;

        // Roll the card about its own normal so the five-petal star is not
        // aligned across the whole tree.
        perp(nx, ny, nz);
        const r1x = _px, r1y = _py, r1z = _pz;
        const r2x = ny * r1z - nz * r1y;
        const r2y = nz * r1x - nx * r1z;
        const r2z = nx * r1y - ny * r1x;
        const roll = rng() * TAU;
        const cro = Math.cos(roll), sro = Math.sin(roll);
        const rx = r1x * cro + r2x * sro;
        const ry = r1y * cro + r2y * sro;
        const rz = r1z * cro + r2z * sro;

        // A straggler or two inside an otherwise open umbel keeps it from
        // reading as a decal stamped five times.
        const bud = budCluster
          ? 0.55 + 0.45 * rng()
          : (rng() < 0.045 ? 0.40 + 0.35 * rng() : 0);
        let size = lerp(SIZE_MIN, SIZE_MAX, rng() * rng() * 0.5 + rng() * 0.5);
        // A bud is a tight ball a third the width of the open flower.
        size *= 1 - bud * 0.48;
        // Shade dwarfs everything: interior flowers are measurably smaller.
        size *= 0.86 + 0.14 * light;

        // Pigment.
        //
        // This used to be `rng()*rng()`, a distribution with a mean of 0.25 - 
        // three quarters of the canopy sat within a quarter of the way from
        // near-white to blush, and since the CARD TEXTURE is itself near-white
        // (#fbf5f5 at the petal tip), the two whites multiplied and the tree
        // lost its colour entirely under a bright sky. An out-quadratic has a
        // mean of 2/3 and a soft shoulder: most flowers now carry real blush,
        // the palest ones still exist, and nothing is forced to the endpoint.
        //
        // The bough-scale patch field feeds the hue as well as the density,
        // because on a real tree the limb that opened first is also the limb
        // whose flowers have faded palest - colour and phase are the same
        // signal, and driving both from one field is what stops the canopy
        // from looking like one flower stamped forty thousand times.
        const u = rng();
        const tint = clamp01(
          0.34 + 0.62 * u * (2 - u) + 0.22 * (patch - 0.5)
          + bud * 0.50 + (1 - tip) * 0.10
        );

        const o = n * BLOSSOM_STRIDE;
        scratch[o] = px2; scratch[o + 1] = py2; scratch[o + 2] = pz2;
        scratch[o + 3] = rx; scratch[o + 4] = ry; scratch[o + 5] = rz;
        scratch[o + 6] = nx; scratch[o + 7] = ny; scratch[o + 8] = nz;
        scratch[o + 9] = s.swA0; scratch[o + 10] = s.swA1;
        scratch[o + 11] = s.swA2; scratch[o + 12] = s.swA3;
        scratch[o + 13] = s.swB0; scratch[o + 14] = s.swB1;
        scratch[o + 15] = s.swB2; scratch[o + 16] = s.swB3;
        scratch[o + 17] = size;
        scratch[o + 18] = tint;
        scratch[o + 19] = bud;
        scratch[o + 20] = rng();
        scratch[o + 21] = openness;
        n++;
      }
    }
  }

  // The cap is a safety rail, and a safety rail that fires silently is a trap:
  // this one already shipped a 135 000-flower truncated canopy once, lopsided on
  // whichever side the site walk reached last, with nothing in any log to say
  // why. One comparison at build time closes that for good.
  if (n >= maxInstances) {
    console.warn(
      `[Blossoms] instance cap reached (${maxInstances}). The canopy is ` +
      'TRUNCATED and will look bald on one side. Lower the cluster multiplier ' +
      'in generateBlossomInstances or TWIG_SPACING in sakura.js, or raise ' +
      'HARD_MAX - but read the note on HARD_MAX first: more flowers per site is ' +
      'almost never the fix.'
    );
  }

  // --- thinning order ------------------------------------------------------
  // The instances are written out in a shuffled order so that ANY prefix is an
  // unbiased sample of the whole canopy - which is what lets a quality tier or
  // a distance LOD just lower `instanceCount` and get an even thinning instead
  // of a bald hemisphere. The size term biases the shuffle very slightly so the
  // smallest flowers go first: they contribute the least silhouette per draw.
  //
  // The sort is done on ONE packed Float64Array with no comparator, because
  // V8's comparator-free numeric TypedArray sort is roughly six times faster
  // than sorting indices through a closure - 20 ms instead of 550 ms at this
  // count, which is a fifth of the tree's whole build budget. The key occupies
  // the high bits and the index the low 17, so ordering by the packed value is
  // exactly ordering by the key.
  const krng = makeRNG(seed ^ 0x51de);
  // 2^20. This MUST exceed the instance count: the sort key and the source
  // index share one number, and the index is recovered with % IDX_BITS, so any
  // index at or above it wraps and resolves to the wrong flower. It was 2^17
  // (131 072) with a comment claiming that cleared "any plausible instance
  // count", while the canopy has carried half a million instances for a long
  // time. Every index past 131 072 therefore aliased onto a lower one, so the
  // shuffled order duplicated some flowers and silently dropped others, which
  // is what made the thinned tiers uneven around the crown.
  // Packing stays inside Number.MAX_SAFE_INTEGER: key quantised to 2^20 times
  // IDX_BITS is about 1.1e12, against a 9.0e15 ceiling.
  const IDX_BITS = 1048576;
  const packed = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const size = scratch[i * BLOSSOM_STRIDE + 17];
    const key = krng() + (SIZE_MAX - size) * 1.4;
    packed[i] = Math.round(key * 1048576) * IDX_BITS + i;
  }
  packed.sort();

  const data = new Float32Array(n * BLOSSOM_STRIDE);
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const src = (packed[i] % IDX_BITS) * BLOSSOM_STRIDE;
    const dst = i * BLOSSOM_STRIDE;
    for (let k = 0; k < BLOSSOM_STRIDE; k++) data[dst + k] = scratch[src + k];
    const x = data[dst], y = data[dst + 1], z = data[dst + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }

  // --- petal spawn sites ---------------------------------------------------
  // Taken from the FRONT of the thinned order, so every spawn point belongs to
  // a flower that exists at every quality tier, and only from the outer shell,
  // because that is where wind actually strips petals.
  const spawnPos = new Float32Array(SPAWN_MAX * 3);
  const spawnA = new Float32Array(SPAWN_MAX * 4);
  const spawnB = new Float32Array(SPAWN_MAX * 4);
  let sn = 0;
  // Clamped against `n`, not just floored at 1. With n = 0 the old `max(1, …)`
  // walked one step into a zero-length `data`, read undefined for x/y/z, and
  // published a spawn point at (NaN, NaN, NaN) - and the ellipsoid test cannot
  // reject it, because `NaN < 0.50` is false. petals.js seeds its instance
  // buffer straight from these points, and one NaN there is permanent for the
  // session. A tree with no flowers must publish no spawn points.
  const limit = n > 0 ? Math.max(1, Math.floor(n * TIERS.low.fraction)) : 0;
  // Oversampled 3x because the ellipsoid test below rejects most candidates:
  // the crown is filled through its volume now, so only about a third of any
  // sample of it sits on the outer canopy. At a stride of limit/SPAWN_MAX the
  // walk ran out of candidates at 959 of the 1 600 points petals.js sizes its
  // ring buffer for, which is not a failure but does cycle the spawn set nearly
  // twice as fast and makes the same flowers shed over and over.
  const stride = Math.max(1, Math.floor(limit / (SPAWN_MAX * 3)));
  for (let i = 0; i < limit && sn < SPAWN_MAX; i += stride) {
    const o = i * BLOSSOM_STRIDE;
    const x = data[o], y = data[o + 1], z = data[o + 2];
    const ey2 = y - envY0;
    // Same envelope measure the placement uses, so "outer canopy" means the
    // same thing to petals.js as it does above. 0.78 rather than the old 0.50
    // because the crown is filled through its volume now: half the radius is
    // deep inside the flower mass, and wind does not strip petals from there.
    const r = Math.sqrt(
      (x * x + z * z) * invEnvR2 + ey2 * ey2 * (ey2 >= 0 ? invRyUp2 : invRyDown2)
    );
    if (r < 0.78) continue;
    spawnPos[sn * 3] = x; spawnPos[sn * 3 + 1] = y; spawnPos[sn * 3 + 2] = z;
    for (let k = 0; k < 4; k++) {
      spawnA[sn * 4 + k] = data[o + 9 + k];
      spawnB[sn * 4 + k] = data[o + 13 + k];
    }
    sn++;
  }
  // Degenerate canopy (a seed that grew almost nothing): fall back to the
  // whole set rather than handing petals.js an empty array.
  if (sn === 0 && n > 0) {
    const step = Math.max(1, Math.floor(n / SPAWN_MAX));
    for (let i = 0; i < n && sn < SPAWN_MAX; i += step) {
      const o = i * BLOSSOM_STRIDE;
      spawnPos[sn * 3] = data[o];
      spawnPos[sn * 3 + 1] = data[o + 1];
      spawnPos[sn * 3 + 2] = data[o + 2];
      for (let k = 0; k < 4; k++) {
        spawnA[sn * 4 + k] = data[o + 9 + k];
        spawnB[sn * 4 + k] = data[o + 13 + k];
      }
      sn++;
    }
  }

  const sphere = {
    x: (minX + maxX) * 0.5,
    y: (minY + maxY) * 0.5,
    z: (minZ + maxZ) * 0.5,
    r: 0.5 * Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) + SIZE_MAX * 2,
  };
  if (!Number.isFinite(sphere.r)) {
    sphere.x = ccx; sphere.y = ccy; sphere.z = ccz; sphere.r = cr * 1.5;
  }

  return {
    data,
    count: n,
    spawn: { count: sn, position: spawnPos, swayA: spawnA, swayB: spawnB },
    sphere,
  };
}

// -- small helpers, build time only ------------------------------------------

let _px = 0, _py = 0, _pz = 0;

/**
 * Exactly perpendicular unit vector to (x,y,z), via the cross product with
 * whichever cardinal axis the input is LEAST aligned with - so the result is
 * both orthogonal to machine precision and never ill-conditioned. (The usual
 * "swap two components" trick is only approximately orthogonal near the
 * degenerate axis, and here the result is used directly as a card axis, where
 * a 25 degree error would visibly skew the flower.)
 */
function perp(x, y, z) {
  const ax = Math.abs(x), ay = Math.abs(y), az = Math.abs(z);
  let sx = 0, sy = 0, sz = 0;
  if (ax <= ay && ax <= az) sx = 1;
  else if (ay <= az) sy = 1;
  else sz = 1;
  _px = sy * z - sz * y;
  _py = sz * x - sx * z;
  _pz = sx * y - sy * x;
  const l = Math.hypot(_px, _py, _pz) || 1;
  _px /= l; _py /= l; _pz /= l;
}

/**
 * Smooth value field at BOUGH scale - wavelengths of 2.5 to 4 metres, so one
 * limb can be a week ahead of the one beside it. Three sines is plenty for a
 * field that is only ever a multiplier, and it costs nothing at build time.
 * (Deliberately not `noise.fbm2D`: this has to be a smooth 3D field, and three
 * sines of hand-picked wavelength say exactly what they mean.)
 */
function hashPatch(x, y, z) {
  const a = Math.sin(x * 2.05 + y * 1.02 + 1.7);
  const b = Math.sin(z * 1.74 - y * 1.31 + 4.2);
  const c = Math.sin((x + z) * 0.91 + y * 0.72 + 2.6);
  return clamp01(0.5 + 0.5 * (a * 0.45 + b * 0.35 + c * 0.30));
}

// ===========================================================================
// Renderer
// ===========================================================================

const _v3 = new THREE.Vector3();
const _quat = new THREE.Quaternion();

/** Finite-or-fallback. Same guard sakura.js applies to its wind inputs. */
const fin = (v, fallback) => (Number.isFinite(v) ? v : fallback);

export class BlossomField {
  /**
   * @param {object} ctx   the shared system context
   * @param {object} opts  { windUniforms, seed }
   */
  constructor(ctx, { windUniforms, seed = 0x5a11a9 } = {}) {
    this.ctx = ctx;
    this.seed = seed;
    this.wind = windUniforms;

    this.mesh = null;
    this.geometry = null;
    this.material = null;
    this.depthMaterial = null;
    this.buffer = null;

    this.count = 0;
    this.activeCount = 0;
    this.tierBoost = 1;
    this.spawn = null;

    this.u = null;
    this._tier = 'high';
  }

  // -------------------------------------------------------------------------

  /**
   * @param {object} skel  from generateSkeleton()
   * @param {Array}  sites from collectTwigSites()
   */
  build(skel, sites) {
    const gen = generateBlossomInstances(skel, sites, { seed: this.seed });
    this.count = gen.count;
    this.spawn = gen.spawn;

    const geo = new THREE.InstancedBufferGeometry();
    const card = buildCardGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(card.position, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(card.normal, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(card.uv, 2));
    geo.setIndex(new THREE.BufferAttribute(card.index, 1));

    // One interleaved static buffer: BLOSSOM_STRIDE floats an instance, once.
    const ib = new THREE.InstancedInterleavedBuffer(gen.data, BLOSSOM_STRIDE, 1);
    ib.setUsage(THREE.StaticDrawUsage);
    geo.setAttribute('aBPos', new THREE.InterleavedBufferAttribute(ib, 3, 0));
    geo.setAttribute('aBRight', new THREE.InterleavedBufferAttribute(ib, 3, 3));
    geo.setAttribute('aBNormal', new THREE.InterleavedBufferAttribute(ib, 3, 6));
    geo.setAttribute('aBSwayA', new THREE.InterleavedBufferAttribute(ib, 4, 9));
    geo.setAttribute('aBSwayB', new THREE.InterleavedBufferAttribute(ib, 4, 13));
    geo.setAttribute('aBParam', new THREE.InterleavedBufferAttribute(ib, 4, 17));
    geo.setAttribute('aBShade', new THREE.InterleavedBufferAttribute(ib, 1, 21));
    geo.instanceCount = gen.count;

    // Instance offsets live in the attribute stream, so three's computed bounds
    // would be the 5 cm card, not the 15 m canopy. The 2.8 m of slack on top is
    // measured rather than guessed: 1.4 m of crown-bend overshoot in a storm
    // gust plus the four summed oscillators, so the canopy can never be culled
    // while any part of it is still on screen.
    geo.boundingSphere = new THREE.Sphere(
      new THREE.Vector3(gen.sphere.x, gen.sphere.y, gen.sphere.z),
      gen.sphere.r + 2.8
    );
    geo.boundingBox = new THREE.Box3().setFromCenterAndSize(
      geo.boundingSphere.center,
      _v3.setScalar(geo.boundingSphere.radius * 2)
    );

    this.geometry = geo;
    this.buffer = ib;
    this.material = this._makeMaterial();
    this.depthMaterial = this._makeDepthMaterial();

    const mesh = new THREE.Mesh(geo, this.material);
    mesh.name = 'sakura.blossoms';
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.customDepthMaterial = this.depthMaterial;
    // Alpha-tested and depth-writing, so it stays in the opaque queue where
    // three's own front-to-back sort gives early-Z something to reject with.
    mesh.renderOrder = 0;
    mesh.matrixAutoUpdate = false;
    this.mesh = mesh;

    this.setQuality(this._tier);
    return this;
  }

  // -------------------------------------------------------------------------

  _textures() {
    const tex = this.ctx.textures;
    let albedo = null;
    try {
      albedo = tex?.blossomCard ? tex.blossomCard().albedo : tex?.get?.('blossomAlbedo');
    } catch (err) {
      console.warn('[Blossoms] blossom card unavailable:', err);
    }
    return albedo || null;
  }

  _makeMaterial() {
    const map = this._textures();
    const mat = new THREE.MeshLambertMaterial({
      map,
      color: 0xffffff,
      // Near-field cutoff. The fragment patch relaxes it with distance to
      // undo mip-averaged alpha erosion - see BLOSSOM_ALPHATEST.
      // 0.50 -> 0.42. The card's alpha is a five-pointed star, not a disc, and a
      // half cutoff eats the outer third of every petal's width - measured, the
      // crown rasterised visibly lacier than a placement model that gives each
      // card the SAME total area as a solid shape, which is the discrepancy that
      // sent this pass looking for more instances it did not need. Relaxing the
      // cutoff widens every petal by a texel or two. It costs surviving
      // fragments, not processed ones: the quad, the vertex work and the texture
      // fetch are identical either way, so it is by a wide margin the cheapest
      // coverage in this file. It cannot go much further - below about 0.35 the
      // mip-blurred margin starts showing as a halo.
      alphaTest: 0.42,
      transparent: false,
      side: THREE.DoubleSide,
      fog: true,
      dithering: true, // near-white gradients band viciously without it
    });
    mat.name = 'sakura.blossom';

    const u = {
      uBSize: { value: 1 },
      uBOpen: { value: 1 },
      // Petal ramp: pale centre -> blush -> unopened bud. The first pass sat at
      // #f7efee/#ecd2d6/#d9a3ae, which is very nearly white and washed out
      // entirely under a bright sky. Pushed toward a clear sakura pink while
      // staying well short of candy - the blush is the colour that carries the
      // canopy at distance, so it does most of the work.
      // Measured against renders: with the tint distribution now centred near
      // 0.66 and pigment absorption applied on top, the previous ramp
      // (#f2b3c2 blush, #d4788d bud) came out SALMON at noon and near-maroon
      // under a low warm sun. Both are pulled cooler and lighter - away from
      // red, slightly toward magenta - so warm light lands on them as pink
      // rather than as rust. Somei-Yoshino is barely chromatic in reality; the
      // colour has to survive being multiplied by a 2000 K sun.
      // Pulled lighter and markedly less chromatic again, because the pigment
      // was being applied THREE times over and the product is what the eye sees:
      // the card texture already ramps from #fbf5f5 at the petal tip to #dcaeb8
      // at the claw, then this ramp multiplies it, then uBThroat multiplies the
      // middle again. Measured through the whole chain, the old palette put the
      // card centre at a linear (0.57, 0.15, 0.22) - g/r = 0.26, which is not a
      // cherry blossom, it is a dark rose. Multiply that by a low sun's
      // (1.0, 0.17, 0.065) and no shading model can produce anything but rust.
      // A real Somei-Yoshino petal is barely chromatic; the colour has to come
      // from the throat gradient and the sheer number of flowers, not from
      // saturating each one.
      // PUSHED BACK TOWARD PINK. The chain above walked the palette pale three
      // separate times chasing a canopy that kept coming out rust, and it was
      // treating a symptom: the crown was dark because the transmission term
      // was worth 0.07 of the illuminant, not because the pigment was strong.
      // Now that the multiple-scattering term carries the canopy (see
      // BLOSSOM_SSS) the flowers can hold real colour and still read as light - 
      // which is the combination the brief asks for and the one a Somei-Yoshino
      // actually has. These are LIGHT colours with genuine chroma: the blush is
      // 0.98/0.72/0.81 in linear, so g/r is 0.73 rather than the 0.26 of the
      // dark rose that produced the rust, and the blue stays ABOVE the green so
      // a warm sun lands on it as pink rather than as salmon.
      // Pushed toward the reference photograph, which is a genuinely CHROMATIC
      // pink - not the near-white of a Somei-Yoshino at peak. The blush carries
      // the canopy at distance, so it does most of the work and gets the most
      // saturation; the pale is the sunlit face of a petal and stays light so
      // the crown still has internal range instead of reading as one flat wash.
      // Measured: the canopy renders [219,208,219] - near white - because a
      // bright sky ambient washes the pink out of it. The tint distribution was
      // already blush-weighted (mean 0.68), so the fix is chroma in the palette
      // itself, not more of it. Even the PALE end carries pink now, because on a
      // real tree the palest flower is still visibly not white.
      uBPale: { value: new THREE.Color('#ffd0e2') },
      uBBlush: { value: new THREE.Color('#f56ba6') },
      uBBud: { value: new THREE.Color('#d64d8b') },
      /**
       * Throat. Multiplied into the middle of the card, where the five claws
       * meet - the one part of a cherry flower that is reliably, visibly pink.
       *
       * It is here rather than in the vertex tint because a per-instance tint
       * is a FLAT colour: it can only slide the whole flower along the
       * pale→blush ramp, and sliding it far enough to read as pink from twenty
       * metres makes it read as a plastic azalea from two. A gradient inside
       * the card gives the eye a saturated core against a pale margin, which is
       * what it actually uses to judge the colour of a flower - and unlike a
       * flat tint it SURVIVES MINIFICATION, because the mip average of a card
       * with a pink centre is pinker than the card's own margin. That is the
       * whole reason the canopy can read as pink at distance while individual
       * flowers stay near-white close up.
       */
      uBThroat: { value: new THREE.Color('#f19dbe') },
      uBWet: { value: 0 },
      uBKeyDir: { value: new THREE.Vector3(0, 0, 1) },
      /**
       * TRANSMISSION ILLUMINANT - the beam PLUS skylight, not the beam alone.
       *
       * This uniform used to be the directional key colour and nothing else,
       * and that is the single largest reason the canopy went rust at golden
       * hour. What you see through a petal is not the sun; it is whatever is
       * behind the petal, and for a canopy that is overwhelmingly SKY with the
       * sun's beam threaded through it. `sun.color` is a pure transmittance
       * ratio that bottoms out at (1.0, 0.17, 0.065) - a darkroom lamp - so
       * building the glow from it alone can only ever produce orange, however
       * pale the petal is. Skylight restores the green and blue that a real
       * back-lit canopy is full of at sunset, and it costs one vector add on
       * the CPU. See update().
       */
      uBKeyColor: { value: new THREE.Color(0, 0, 0) },
      // scale, forward-lobe power, rim. Shape from the constants; only x moves.
      uBSss: { value: new THREE.Vector3(1.0, SSS_POWER, SSS_RIM) },
      // Distance window over which the alpha cutoff is relaxed. Not a fade - 
      // mip compensation. See BLOSSOM_ALPHATEST.
      uBFade: { value: new THREE.Vector2(10, 65) },
    };
    this.u = u;

    mat.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, u, this.wind);
      shader.vertexShader = patchBlossomVertex(shader.vertexShader);
      shader.fragmentShader = patchBlossomFragment(shader.fragmentShader);
    };
    // The injected code changes with nothing three hashes, so give the program
    // cache a key of its own or a sibling Lambert material could hand back a
    // program with no blossom code in it at all.
    mat.customProgramCacheKey = () => 'sakuraBlossom';
    return mat;
  }

  /**
   * Wind-displaced geometry needs a depth material that displaces IDENTICALLY,
   * or the canopy's shadow stands still while the canopy moves - the single
   * most obvious tell there is.
   */
  _makeDepthMaterial() {
    const mat = new THREE.MeshDepthMaterial({
      depthPacking: THREE.RGBADepthPacking,
      map: this._textures(),
      // 0.50 -> 0.42. The card's alpha is a five-pointed star, not a disc, and a
      // half cutoff eats the outer third of every petal's width - measured, the
      // crown rasterised visibly lacier than a placement model that gives each
      // card the SAME total area as a solid shape, which is the discrepancy that
      // sent this pass looking for more instances it did not need. Relaxing the
      // cutoff widens every petal by a texel or two. It costs surviving
      // fragments, not processed ones: the quad, the vertex work and the texture
      // fetch are identical either way, so it is by a wide margin the cheapest
      // coverage in this file. It cannot go much further - below about 0.35 the
      // mip-blurred margin starts showing as a halo.
      alphaTest: 0.42,
      side: THREE.DoubleSide,
    });
    mat.name = 'sakura.blossom.depth';
    // three overwrites map/alphaTest/side on a customDepthMaterial from the
    // visible material every frame, so these exist only to make the first
    // compile match; they are not the source of truth.
    mat.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, this.u, this.wind);
      shader.vertexShader = patchBlossomVertex(shader.vertexShader, true);
    };
    mat.customProgramCacheKey = () => 'sakuraBlossomDepth';
    return mat;
  }

  // -------------------------------------------------------------------------

  setQuality(tier) {
    this._tier = tier;
    const t = TIERS[tier] || TIERS.high;
    if (!this.geometry) return;
    // A skeleton that grew nothing must end up at zero, not at one instance
    // read out of a zero-length buffer with uBSize collapsed to 0.
    const active = this.count > 0
      ? Math.max(1, Math.min(this.count, Math.round(this.count * t.fraction)))
      : 0;
    this.activeCount = active;
    this.geometry.instanceCount = active;
    // Partial mass compensation, authored per tier rather than derived - see
    // TIERS for why exact sqrt compensation made `fraction` a dead dial.
    // NOTE: this compensates the TIER thinning only. The distance thinning in
    // sakura.js deliberately gets no size boost - there, the relaxed alpha
    // cutoff already gives each card more coverage, and doing both would turn
    // a distant canopy into a solid pink blob.
    this.tierBoost = t.boost;
    if (this.u) this.u.uBSize.value = this.tierBoost;
    if (this.mesh) {
      this.mesh.castShadow = t.shadow;
      this.mesh.receiveShadow = t.receive !== false;
    }
  }

  /**
   * Per-frame uniform sync. The key light is rebuilt exactly the way the
   * engine builds its directional pair, so the transmission term can never
   * disagree with the light that is actually reaching the tree.
   */
  update(state, camera) {
    const u = this.u;
    if (!u) return;
    const sun = state.sun, moon = state.moon;

    // Every value below lands in a uniform that multiplies the whole canopy.
    // `clamp01(NaN)` is NaN and `Math.max(0, NaN)` is NaN, so one bad frame out
    // of a sibling system would put NaN into uBKeyColor and the entire crown
    // would rasterise to garbage - 67 000 cards, no recovery until that system
    // produced a finite number again. sakura.js sanitises every one of its wind
    // inputs for exactly this reason; this path had no guard at all.
    const overcast = smoothstep(0.42, 0.96, clamp01(fin(state.clouds.coverage, 0)));
    const transmit = lerp(1, 0.3, overcast);
    const si = Math.max(0, fin(sun.intensity, 0)) * smoothstep(-0.015, 0.06, fin(sun.direction.y, 0)) * transmit;
    const mi = Math.max(0, fin(moon.intensity, 0)) * smoothstep(-0.015, 0.06, fin(moon.direction.y, 0)) * transmit;

    // One dominant key. Two transmission lobes would double the cost of the
    // only term that is ever visible at a time, and blending keeps dusk smooth.
    const total = si + mi;
    if (total > 1e-4) {
      _v3.copy(sun.direction).multiplyScalar(si).addScaledVector(moon.direction, mi);
      if (_v3.lengthSq() > 1e-10) _v3.normalize();
      else _v3.set(0, 1, 0);
    } else {
      _v3.set(0, 1, 0);
    }
    // View space, matching what three hands its own directional lights.
    //
    // Deliberately NOT camera.matrixWorldInverse: the renderer only refreshes
    // that inside post.render(), which runs after every system's update(), so
    // reading it here would be one frame behind the camera - and the identity
    // matrix on the very first frame. three computes its own light directions
    // during render with the CURRENT view matrix, so a stale key here makes the
    // transmission lobe swim behind the diffuse lighting under fast mouse look.
    // getWorldQuaternion refreshes the camera's world matrix on the way out.
    camera.getWorldQuaternion(_quat).invert();
    _v3.applyQuaternion(_quat);
    u.uBKeyDir.value.copy(_v3);

    // --- transmission illuminant ------------------------------------------
    // Beam first. This is exactly three's own directional pair, so the glow can
    // never disagree with the light actually reaching the tree.
    //
    // The COLOURS need the same finite guard as the intensities. The intensity
    // and direction terms above were sanitised and these were not, which left
    // the whole point of that guard undone: `sun.color` is written by
    // sky/celestial.js and one NaN component there reaches `uBKeyColor`
    // unfiltered, and `outgoingLight += ... * uBKeyColor * ...` turns every one
    // of the 67 000 cards into NaN - which rasterises as whatever the hardware
    // does with a NaN colour and does not recover until that system produces a
    // finite number again.
    let gr = fin(sun.color.r, 0) * si + fin(moon.color.r, 0) * mi;
    let gg = fin(sun.color.g, 0) * si + fin(moon.color.g, 0) * mi;
    let gb = fin(sun.color.b, 0) * si + fin(moon.color.b, 0) * mi;

    // Then the sky behind the canopy. Weighted toward the horizon because that
    // is the band a petal seen from ground level actually has behind it, and
    // because the zenith alone would tint the glow far too blue at noon.
    //
    // This term is what keeps the canopy LUMINOUS rather than absorbing under a
    // low sun: at golden hour the beam is a near-monochromatic orange worth
    // maybe three times the skylight, so adding the sky roughly triples the
    // canopy's blue response and doubles its green. It also gives the right
    // answer for free in two cases the beam cannot: under heavy overcast the
    // sun term collapses and the canopy keeps a soft even glow, which is what
    // a cherry does on a white day, and at night everything here goes to zero
    // together so nothing glows in the dark.
    const sky = state.sky;
    // How far the sun has dropped toward the horizon, 0 high, 1 on it.
    const lowSun = 1 - smoothstep(0.05, 0.45, fin(sun.direction.y, 1));
    const amb = Math.max(0, fin(sky.ambientIntensity, 0)) * GLOW_SKY * (1 + GLOW_SKY_LOW * lowSun);
    // And the BAND of sky shifts with it. 0.6 of the way to the horizon is
    // right at noon, when the horizon is the pale part of the sky. At sunset
    // the horizon is the orange part and the zenith is the blue one, so
    // weighting the horizon there feeds the canopy the very colour that was
    // making it read as rust; the mix walks back to 0.35 to keep the blue.
    const band = 0.6 - 0.25 * lowSun;
    gr += lerp(fin(sky.zenithColor.r, 0), fin(sky.horizonColor.r, 0), band) * amb;
    gg += lerp(fin(sky.zenithColor.g, 0), fin(sky.horizonColor.g, 0), band) * amb;
    gb += lerp(fin(sky.zenithColor.b, 0), fin(sky.horizonColor.b, 0), band) * amb;
    // Last line of defence, and it costs three compares on one vector a frame:
    // a negative illuminant would SUBTRACT from outgoingLight and punch black
    // holes through the canopy, which is a failure the tonemapper cannot hide.
    u.uBKeyColor.value.setRGB(
      gr > 0 ? gr : 0,
      gg > 0 ? gg : 0,
      gb > 0 ? gb : 0
    );

    // Wet petals go translucent and slightly darker; they also stop scattering
    // as much, because the water fills the air gaps that did the scattering.
    const wet = clamp01(fin(state.weather.wetness, 0));
    u.uBWet.value = wet;
    // ONLY the scale is per-frame. The lobe SHAPE is authored once, at
    // SSS_POWER / SSS_RIM - writing it here as well is how the widened
    // forward-scatter peak got reverted to the old tight one on every frame.
    u.uBSss.value.x = 1.0 - 0.28 * wet;
  }

  dispose() {
    this.geometry?.dispose();
    this.material?.dispose();
    this.depthMaterial?.dispose();
    this.geometry = null;
    this.material = null;
    this.depthMaterial = null;
    this.mesh = null;
    this.buffer = null;
    this.spawn = null;
    this.u = null;
  }
}

// ===========================================================================
// Card geometry
// ===========================================================================

/**
 * Four vertices, two triangles, and an analytic bowl normal.
 *
 * The card is deliberately NOT tessellated: 67 000 flowers at nine vertices
 * each is 320 k triangles for a shape whose curvature only matters to the
 * SHADING normal, and that interpolates perfectly across two triangles.
 *
 * Be clear about what the bowl does and does not do here. All four corners sit
 * at (±1, ±1), so x²+y² is 2 at every one of them and the `z` this writes is
 * the same constant on all four: the quad is exactly PLANAR, and the `dish`
 * term in the vertex shader is a rigid push-off along the face normal that
 * lifts the flower clear of its twig. What actually produces the "catches
 * light across its width" read is the NORMAL - n = normalize(-∂z/∂x, -∂z/∂y, 1)
 * of the paraboloid z = D(x²+y²) - which splays outward at each corner and
 * interpolates to face-on at the centre. Two triangles carry that perfectly
 * well; adding geometry would only buy silhouette, and the alpha cutout is
 * already the silhouette.
 */
function buildCardGeometry() {
  const position = new Float32Array(4 * 3);
  const normal = new Float32Array(4 * 3);
  const uv = new Float32Array(4 * 2);
  const index = new Uint16Array([0, 1, 2, 2, 1, 3]);
  let p = 0, q = 0;
  for (let j = 0; j < 2; j++) {
    for (let i = 0; i < 2; i++) {
      const x = i * 2 - 1;
      const y = j * 2 - 1;
      const z = CARD_DISH * (x * x + y * y);
      position[p] = x; position[p + 1] = y; position[p + 2] = z;
      // n = normalize( -dz/dx, -dz/dy, 1 ) for z = D(x^2 + y^2)
      const nx = -2 * CARD_DISH * x;
      const ny = -2 * CARD_DISH * y;
      const nl = Math.hypot(nx, ny, 1);
      normal[p] = nx / nl; normal[p + 1] = ny / nl; normal[p + 2] = 1 / nl;
      p += 3;
      uv[q] = x * 0.5 + 0.5; uv[q + 1] = y * 0.5 + 0.5;
      q += 2;
    }
  }
  return { position, normal, uv, index };
}

// ===========================================================================
// Shader surgery
// ===========================================================================

const BLOSSOM_VERT_PARS = /* glsl */ `
attribute vec3 aBPos;
attribute vec3 aBRight;
attribute vec3 aBNormal;
attribute vec4 aBSwayA;
attribute vec4 aBSwayB;
attribute vec4 aBParam;   // size, tint, budness, phase
attribute float aBShade;  // baked crown occlusion, 0 = buried, 1 = on the shell

uniform float uBSize;
uniform float uBOpen;
uniform vec3  uBPale;
uniform vec3  uBBlush;
uniform vec3  uBBud;
uniform float uBWet;

varying vec4 vBlossom;    // rgb = albedo tint, a = translucency
`;

/**
 * The whole instance transform. Folded into `beginnormal_vertex` for the lit
 * pass so that the normal and the position it feeds come out of ONE wind
 * evaluation; the depth pass has no normals, so it gets the same code with the
 * shading half removed rather than a second, drift-prone copy.
 *
 * @param {boolean} lit  emit the normal and pigment work
 */
function blossomVertBody(lit) {
  return /* glsl */ `
${lit ? `vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif` : ''}

// --- the twig this flower grew on, after the wind has moved it -------------
vec3 bBase = sakuraBend( aBPos );
${lit ? `
// sakuraBend is a POSITION field, so a finite difference along the card normal
// is exactly the local rotation it applies. The oscillator is a pure
// translation for a fixed weight set, so it cancels out of the difference and
// costs nothing here.
vec3 bN = normalize( sakuraBend( aBPos + aBNormal * 0.05 ) - bBase );
vec3 bR = aBRight - bN * dot( aBRight, bN );
bR = dot( bR, bR ) > 1e-8 ? normalize( bR ) : normalize( cross( bN, vec3( 0.0, 0.0, 1.0 ) ) );
` : `
// The shadow pass skips the frame rotation: it costs a second bend evaluation
// across every instance in every cascade, and the error it removes is a couple
// of degrees of roll on a 5 cm card. Nothing in a shadow map can resolve that.
vec3 bN = aBNormal;
vec3 bR = aBRight;
`}
vec3 bU = cross( bN, bR );

// --- corolla shape ---------------------------------------------------------
float bud = aBParam.z;
// A bud draws its petals in and pushes them forward into a cone; an open
// flower keeps the shallow bowl baked into the card.
vec2 q = position.xy * ( 1.0 - 0.44 * bud );
float dish = position.z * uBOpen * ( 1.0 - bud ) + bud * ( 0.22 + 0.42 * dot( q, q ) );
float bSize = aBParam.x * uBSize;

// --- flutter ---------------------------------------------------------------
// Fast, tiny, and weighted by the twig's own compliance, so a flower on a
// stiff inner branch barely trembles while one on a whip end never stops.
float ph = aBParam.w * 6.2831853;
float fw = uSakuraExtra.z * ( 0.30 + 0.95 * aBSwayB.z );
vec3 flut = vec3(
  sin( uSakuraTime * 5.9 + ph ),
  sin( uSakuraTime * 4.3 + ph * 1.71 ) * 0.55,
  cos( uSakuraTime * 5.1 + ph * 1.33 )
) * fw;

vec3 bLocal = bBase + sakuraOsc( aBSwayA, aBSwayB ) + flut
            + ( bR * q.x + bU * q.y + bN * dish ) * bSize;
${lit ? `
objectNormal = normalize( bR * normal.x + bU * normal.y + bN * normal.z );

// --- pigment ---------------------------------------------------------------
float tintT = aBParam.y;
vec3 bTint = mix( uBPale, uBBlush, tintT );
// 0.8 pushed too much of the canopy all the way to the bud colour. Combined
// with the raised tint distribution it made the crown read dark rust at golden
// hour - measurably DARKER than the sunlit grass behind it, which no blossom
// should ever be.
bTint = mix( bTint, uBBud, bud * 0.5 );
bTint *= 1.0 - 0.13 * uBWet;
// Pigment ABSORBS. A pink petal is not a white petal wearing a filter: it
// reflects less total light, and the deeper the pigment the less it reflects.
// Skipping this is why the blush washed out under a bright sky - the pinkest
// flowers were also the brightest, ACES clipped the red channel first, and the
// hue slid to white exactly where the colour was supposed to be strongest. Six
// percent of reflectance buys the whole canopy its colour back at noon, and
// costs nothing at dusk where the highlights are nowhere near the shoulder.
// 0.14 -> 0.07 -> 0.05. The reasoning above is sound and the term stays, but
// the palette it was written against was far more chromatic than the one above:
// with the blush this pale there is much less pigment to absorb, and at 0.07 it
// was still taking more light out of the canopy than a warm evening sun could
// put back - which is the wrong direction for the one asset in the scene that
// must never be darker than the field behind it.
bTint *= 1.0 - 0.035 * tintT;
// Baked crown occlusion. Deliberately shallow - a flower inside a cherry still
// sees a great deal of sky through the shell above it, so this is contact
// shading that gives the crown depth, not a light well. The shell sits at 1.0
// by construction (see openness in generateBlossomInstances), so the
// silhouette keeps its brightness and only the interior darkens.
bTint *= mix( 0.74, 1.0, aBShade );
// Thin pale petals transmit; a tight bud is five layers thick and does not.
// A buried flower transmits less as well: the light that would have back-lit
// it was already spent on the shell in front of it - but only partly, because
// what it spent is largely what the shell TRANSMITTED, and this is the one
// term that must not be allowed to put the crown's interior in the dark.
vBlossom = vec4( bTint, ( 1.0 - 0.78 * bud ) * ( 1.0 - 0.22 * tintT ) * ( 0.78 + 0.22 * aBShade ) );
` : ''}
`;
}

const BLOSSOM_FRAG_PARS = /* glsl */ `
varying vec4 vBlossom;
uniform vec3 uBKeyDir;     // view space, pointing from the surface to the light
uniform vec3 uBKeyColor;
uniform vec3 uBSss;        // scale, power, rim
uniform vec2 uBFade;       // mip-compensation window, metres
uniform vec3 uBThroat;
`;

/**
 * Throat gradient. `vMapUv` runs 0..1 across the card, so the distance from
 * (0.5, 0.5) is the distance from the centre of the corolla in petal-lengths.
 *
 * The window starts at 0.10 rather than 0.0 so the very centre is a flat plate
 * of colour instead of a point singularity that aliases, and ends at 0.34,
 * which is where the claws of the five petals give way to the blades - the
 * texture's own pigment ramp turns over at the same place, so the two
 * reinforce instead of fighting.
 *
 * Deliberately NOT gated on the instance tint: even the whitest Somei-Yoshino
 * has a pink throat, and it is the one part of the flower that is pink on
 * every single one of them. Gating it would have thrown away the one colour
 * cue that is universal.
 *
 * Injected AFTER the alpha test, not before it. Neither this nor the instance
 * tint touches `diffuseColor.a`, so nothing about the cutout depends on them - 
 * and the blossom card's silhouette covers only about a third of its own quad,
 * so running the pigment chain ahead of the discard spent it on roughly two
 * fragments in three. This is the one material in the scene that can fill the
 * screen; work it does on fragments it is about to throw away is the most
 * expensive kind there is.
 */
const BLOSSOM_PIGMENT = /* glsl */ `
diffuseColor.rgb *= vBlossom.rgb;
#ifdef USE_MAP
	float bThroat = 1.0 - smoothstep( 0.10, 0.34, length( vMapUv - 0.5 ) );
	// 0.80 -> 0.42. The card texture ALREADY ramps to #dcaeb8 at the claw, so
	// this was a second full-strength throat multiplied over the first one and
	// the centre of every flower came out a dark rose. The gradient still does
	// its job - survive minification, give the eye a saturated core against a
	// pale margin - at half the depth.
	// 0.45 -> 0.52 with a markedly pinker throat colour. The gradient is where
	// the canopy's colour survives minification - the mip average of a card with
	// a pink centre is pinker than its own margin - so it is the cheapest place
	// to buy "reads as clearly pink from twenty metres" without making an
	// individual flower look like a plastic azalea at two.
	diffuseColor.rgb *= mix( vec3( 1.0 ), uBThroat, bThroat * 0.52 );
#endif
`;

/**
 * Alpha-test mip compensation - a TRIM, not a rebuild.
 *
 * The usual failure this guards against is real: the blossom card's silhouette
 * covers about a third of its own texture, so a naively mipped card loses its
 * rim to a fixed 0.5 cutoff and the canopy is half gone by 40 m.
 *
 * But `textures.blossomCard()` already bakes its chain with
 * `{ alphaWeighted: true, preserveCoverage: true, alphaTest: 0.5 }` - every
 * level is rescaled so that thresholding it at 0.5 reproduces level 0's
 * coverage. Compensating a second time here is what turns a distant canopy into
 * a solid pink blob: dropping the cutoff to 0.18 pushes the silhouette out by
 * the full width of the mip-blurred alpha gradient on top of a chain that was
 * already coverage-correct. What is left to correct is only the residual
 * softening from anisotropic filtering and the bilinear tap between levels,
 * which is worth a few hundredths - hence 0.36 rather than 0.18. Still paired
 * with the distance thinning in sakura.js over roughly the same window.
 */
const BLOSSOM_ALPHATEST = /* glsl */ `
#ifdef USE_ALPHATEST
	float bCut = mix( alphaTest, 0.36, bFar );
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( bCut, bCut + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < bCut ) discard;
	#endif
#endif
`;

/**
 * Transmission. A cherry petal is ~0.1 mm of translucent tissue; the reason a
 * back-lit canopy glows is that most of the light reaching your eye went
 * THROUGH it, not off it. Frostbite's approximate translucency is the right
 * model at the right price: the lobe is centred opposite the light, pulled
 * slightly by the surface normal so it wraps around the edge of the petal.
 *
 * RECIPROCAL_PI is not cosmetic. `uBKeyColor` is built to be exactly three's
 * `directionalLight.color` (engine.js applies the same horizon gate and
 * overcast transmit before setting sunLight.intensity), and three's diffuse
 * lobe is `irradiance * BRDF_Lambert( diffuseColor )` - which carries the 1/PI.
 * Without it this term is PI times the diffuse lobe, so a back-lit petal came
 * out around 4.7x the radiance of a fully front-lit one and ACES clipped it to
 * white: the exact opposite of a pale, desaturated pink. With it the back-lit
 * peak lands about 1.5x the front-lit one, which is what translucent tissue
 * actually does.
 */
const BLOSSOM_SSS = /* glsl */ `
{
  vec3 bV = normalize( vViewPosition );

  // Two lobes, because thin tissue scatters in two distinguishable ways and
  // only one of them was here before.
  //
  // bBack is the DIFFUSE half: light that entered the far face and left the
  // near one having lost all memory of its direction. It depends only on where
  // the light is relative to the petal, not on where the camera is, so it gives
  // every back-lit flower in the crown a steady glow from any viewpoint. Its
  // absence is most of why the canopy only lit up when you stood in one spot.
  //
  // bFwd is the forward-scattered peak - the hot rim you get looking into the
  // sun through a petal. Frostbite's approximate translucency: the lobe sits
  // opposite the light, bent by the normal so it wraps around the petal's edge.
  // The half-vector is NORMALISED now; unnormalised, its length varied with the
  // angle between light and normal and quietly modulated the lobe by up to 25 %.
  // The power came down from 3.2, which was so tight the glow was invisible
  // unless the sun was almost exactly behind the flower.
  // bMulti is the third term and the one the canopy actually lives on. A crown
  // in bloom is tens of thousands of pale, highly translucent surfaces packed a
  // few centimetres apart, so the light that reaches any one of them has been
  // through several others: it is a MULTIPLE-SCATTERING medium with an albedo
  // near 0.85, and such a medium is far brighter than any single-scatter lobe
  // predicts. That is the whole reason a cherry in full bloom glows rather than
  // merely being lit, and it is what nothing here modelled.
  //
  // Its absence is measurable and it is exactly the reported defect. With only
  // the two lobes below, the transmission averaged over a crown came to about
  // 0.07 of the illuminant - a rounding error next to the diffuse term - so
  // under a low sun the canopy was left with three's Lambert lobe times a beam
  // colour that bottoms out at (1.0, 0.17, 0.065), and it went rust and ended
  // up DARKER than the sunlit grass behind it. This term is view- and
  // light-direction independent by construction, which is correct for a
  // diffusing medium and is why the crown now glows from every angle and at
  // every hour instead of only when the sun is behind it.
  float bMulti = 0.30;

  float bBack = clamp( -dot( normal, uBKeyDir ), 0.0, 1.0 );
  vec3 bH = normalize( uBKeyDir + normal * 0.35 );
  float bFwd = pow( clamp( dot( bV, -bH ), 0.0, 1.0 ), uBSss.y );
  float lt = ( bMulti + bBack * 0.95 + bFwd * 0.70 ) * uBSss.x * vBlossom.a;

  // Grazing angles see the longest path through the tissue, which is why the
  // rim of a back-lit petal is the brightest part of it.
  float bRim = pow( 1.0 - clamp( dot( normal, bV ), 0.0, 1.0 ), 3.0 );

  // Shadowing, deliberately PARTIAL.
  //
  // This term used to be the raw shadow factor, and that is backwards: a petal
  // is back-lit precisely when it is facing away from the sun, which is exactly
  // when the shadow map reports it as occluded. Gating transmission on the
  // shadow test therefore switched the glow off in every case it exists for.
  // It is also why the defect was specific to low sun - at noon the beam comes
  // down through the crown and lights its top surface unoccluded, so the canopy
  // read correctly, while at golden hour the shadow ray runs the long way
  // through the crown and essentially every flower comes back shadowed.
  //
  // A shadowed petal is not unlit, it is one or more petals deep, and what
  // reaches it is what those petals transmitted - most of it, for tissue this
  // thin. So the shadow only takes a third here, and the real depth grading
  // comes from the baked crown occlusion already carried in vBlossom.a.
  float bShadow = 1.0;
  #if defined( CSM_CASCADES ) && ( CSM_CASCADES > 0 ) && ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
    bShadow = mix( 0.74, 1.0, csmShadowFactor );
  #endif

  // Transmitted light is far LESS pigmented than reflected light: reflection
  // off a pigmented layer is many internal bounces, so its chroma compounds,
  // while transmission is a single pass through 0.1 mm of tissue. Using the
  // fully compounded albedo - throat gradient and all - is what made the glow
  // itself rust-coloured under a warm sun. It is also why a back-lit cherry
  // looks brighter than a front-lit one rather than merely differently
  // coloured. A third of the way to white keeps enough pigment that the glow is
  // still pink; half way, which is where this landed first, washed the canopy
  // to a flat grey at noon - measurably, HSV saturation 0.06.
  vec3 bTrans = mix( diffuseColor.rgb, vec3( 1.0 ), 0.32 );

  outgoingLight += bTrans * uBKeyColor * ( lt * RECIPROCAL_PI ) * ( 1.0 + uBSss.z * bRim ) * bShadow;
}
`;

function patchBlossomVertex(src, depthOnly = false) {
  if (typeof src !== 'string' || src.indexOf('aBPos') !== -1) return src;
  // The depth pass writes no varyings; emitting one the depth fragment shader
  // never declares is legal but pointless, so strip it.
  const pars = depthOnly
    ? BLOSSOM_VERT_PARS.replace(/varying vec4 vBlossom;[^\n]*\n/, '')
    : BLOSSOM_VERT_PARS;
  let out = src.replace('#include <common>', `#include <common>\n${SAKURA_WIND_GLSL}\n${pars}`);
  if (depthOnly) {
    // The depth shader never runs beginnormal_vertex, so the instance
    // transform goes in ahead of begin_vertex instead.
    out = out.replace(
      '#include <begin_vertex>',
      `${blossomVertBody(false)}\n\tvec3 transformed = bLocal;`
    );
  } else {
    out = out.replace('#include <beginnormal_vertex>', blossomVertBody(true));
    out = out.replace('#include <begin_vertex>', 'vec3 transformed = bLocal;');
  }
  return out;
}

function patchBlossomFragment(src) {
  if (typeof src !== 'string' || src.indexOf('vBlossom') !== -1) return src;
  let out = src.replace('#include <common>', `#include <common>\n${BLOSSOM_FRAG_PARS}`);
  // `bFar` has to exist before the cutoff, because the cutoff is what it is for.
  // The pigment does not, so it goes after - see BLOSSOM_PIGMENT.
  out = out.replace(
    '#include <color_fragment>',
    '#include <color_fragment>' +
      '\n\tfloat bFar = smoothstep( uBFade.x, uBFade.y, length( vViewPosition ) );'
  );
  out = out.replace('#include <alphatest_fragment>', BLOSSOM_ALPHATEST);
  // `specularmap_fragment` is the first chunk after the whole alpha chain and
  // well before `lights_lambert_fragment` copies diffuseColor into the BRDF, so
  // it is where surviving-fragment-only colour work belongs. Kept rather than
  // replaced: LambertMaterial reads `specularStrength` out of it.
  out = out.replace(
    '#include <specularmap_fragment>',
    `#include <specularmap_fragment>\n${BLOSSOM_PIGMENT}`
  );
  // Injected BEFORE opaque_fragment, which is where outgoingLight is consumed,
  // and deliberately NOT into lights_fragment_begin - the engine's cascaded
  // shadow patch owns that chunk and consuming it would silently unshadow the
  // whole canopy.
  out = out.replace('#include <opaque_fragment>', `${BLOSSOM_SSS}\n\t#include <opaque_fragment>`);
  return out;
}
