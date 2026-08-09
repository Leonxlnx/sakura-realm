/**
 * tree/branches.js - skeleton generation, swept-tube mesh, shared wind GLSL.
 *
 * ---------------------------------------------------------------------------
 * WHY IT IS BUILT THIS WAY
 * ---------------------------------------------------------------------------
 * The silhouette is the hero asset of the whole scene, so this is a
 * *biomechanical* growth sim rather than an L-system:
 *
 *   - Radii follow the pipe-model area rule at every fork, so taper is a
 *     consequence of how the tree branches rather than a curve someone tuned.
 *     Branch LENGTH then follows from radius by one allometric law
 *     (L = k·r^p), which is why a 25 cm limb comes out 6 m and a 4 mm twig
 *     22 cm without a single per-level length constant.
 *   - Gravity droop is integrated as real beam curvature, κ = M/(E·I), with
 *     I ∝ r⁴ and the distal load estimated from remaining wood + blossom mass.
 *     A thick limb barely sags at its base and whips down near the tip; the
 *     trunk does not sag at all. There is no "droop lerp".
 *   - Phototropism only switches on once a shoot is thin, which produces the
 *     signature cherry profile: long heavy limbs that sweep down and then
 *     flick up over the last 30 cm.
 *   - The crown is a LIGHT ENVELOPE, and that is now an explicit object in this
 *     file (`CROWN_*` and `envelopeQ`). A free-standing cherry in the open
 *     fills a broad oblate dome whose widest band sits low, because that is
 *     the surface at which a shoot stops being able to hold its leaves against
 *     the light and the wind. Every shoot in the tree is steered by one
 *     tropism derived from it: inside the envelope a shoot grows toward the
 *     surface, outside it there is nothing to hold and the shoot arcs back.
 *     ONE sign change does both, and it is what turns eight limbs into a dome
 *     instead of eight sprays. See CROWN_TROPISM for the mechanism it replaced
 *     and why that one could not produce a dome at any setting.
 *   - The macro structure - a short bole dividing very low into a wide fan of
 *     heavy limbs, one of them snapped off in some old storm with a crown of
 *     epicormic sprouts around the wound - is hand-authored. Fully procedural
 *     gives you a *generic* tree; a specific tree needs an author.
 *
 * ---------------------------------------------------------------------------
 * FIVE MECHANISMS THAT WERE WRONG, AND WHAT REPLACED THEM
 * ---------------------------------------------------------------------------
 * These are recorded because each one is invisible in the code but decides the
 * whole silhouette, and each one defeated an attempt to fix the tree by
 * turning the obvious dial.
 *
 * Numbers 4 and 5 are documented at TIP_RADIUS_FLOOR and CONTINUE_MIN_RADIUS
 * respectively, next to the constants that fix them. In short: `L.rMin` was
 * both the bud-viability test AND the shoot's own termination test, so
 * tightening `taperTo` on thin wood silently killed every fine shoot at half
 * its length and the crown stopped multiplying with depth (845 branches, of
 * which 357 / 251 / 147 at depths 3 / 4 / 5 - CONVERGING); and no shoot had an
 * apical continuation, so every leader in the tree ended in mid-air at a third
 * of its base diameter, which is the flat sawn-off disc on every limb in the
 * renders. Fixing both took the tree to 2 561 branches and 767 m of flowering
 * wood without touching a single growth-character constant.
 *
 * 1. THE BOLE WAS A RANDOM WALK. `r *= 1 + 0.09·noise3D(...)` ran once per
 *    segment on top of the taper, so the multiplier COMPOUNDED over the 35
 *    segments of the trunk. Measured: the bole left the ground at 0.54 m
 *    radius and arrived at the first fork at 1.33 m - a cone standing on its
 *    point, 2.65 m thick at the top. That is the "short fat pollarded stump"
 *    the whole tree read as, and raising TRUNK_RADIUS only made it worse.
 *    Burl is now a BOUNDED modulation of the pipe radius (`storedRadius`),
 *    evaluated fresh at each point and never fed back into the taper.
 *
 * 2. EVERY LIMB RADIUS WAS CLAMPED FLAT. `LEVELS[k].rMax` was an absolute
 *    constant, and `LEVELS[1].rMax = 0.28` sat far below what the pipe model
 *    handed a primary. Every one of the four primaries came out at EXACTLY
 *    0.280 m however thick the trunk was - so length, which follows from
 *    radius, was pinned too, and the crown could not scale with the bole.
 *    The caps are now `rMaxK` FRACTIONS of TRUNK_RADIUS, sized to sit above
 *    what the area rule actually delivers, so they are a safety rail against a
 *    pathological seed rather than the thing driving the tree.
 *
 * 3. THE CROWN RAN OUT OF BUDS, NOT OUT OF BUDGET. Lateral nodes were spaced
 *    at an ABSOLUTE internode in metres, so a 2 m secondary got three buds and
 *    a 0.4 m twig got none - while `maxChildren` (10/8/7/5) was never once
 *    reached and BRANCH_BUDGET (5600) was never approached. Measured: 493
 *    branches grown, ONE bud lost to crowding, 249 lost to being too thin.
 *    Raising the budget therefore did nothing at all. Internode is now derived
 *    from the branch's own length and a target NODE COUNT, with a metric
 *    floor, so every shoot carries a full complement of laterals and the
 *    budget becomes the real limit - which is what makes it spendable on crown
 *    twigs.
 *
 * The mesh side welds forks: a child's tube starts on the PARENT'S AXIS (so it
 * is embedded, never floating) and swells into a collar, while the parent grows
 * a directional shoulder toward each child. Normals come from the analytic
 * partials of the swept surface, so taper, collar, shoulder and root flare all
 * shade correctly instead of the usual purely-radial approximation.
 *
 * This module owns no scene objects. `sakura.js` orchestrates it.
 */

import * as THREE from 'three';
import { makeRNG, createNoise, clamp, clamp01, lerp, smoothstep, TAU } from '../core/math.js';

// ===========================================================================
// Species constants - one specific old Prunus, not a family of trees
// ===========================================================================

/** Area rule exponent. 2.0 is Leonardo's; measured wood lands 2.2-2.6. */
const AREA_EXP = 2.35;
/** Wood lost to the fork itself, so the tree is not a perfectly conserving pipe. */
const FORK_LOSS = 0.955;

/**
 * Allometry L = ALLO_K · r^ALLO_P (metres).
 *
 * THE EXPONENT IS THE CROWN'S DENSITY, and 0.78 was far too high at the thin
 * end. Anchors: a 21 cm primary at 7.1 m, unchanged, and a 2 mm twig at 39 cm
 * - where 0.78 gave it TEN CENTIMETRES. A one-year cherry shoot is 20 to 60 cm
 * long and 2 to 3 mm thick; the old fit made every twig in the tree a stub, and
 * that is what capped the crown's density however hard the branching ratio was
 * pushed. Measured: at 0.78 the tree carried 1 775 m of wood and the flower
 * cloud covered 56 % of its own outline, because the twig cloud was a set of
 * short sprays with sky between them. At 0.62 the same branch count carries
 * roughly twice the length, the sprays reach each other, and coverage goes to
 * where the reference photographs are.
 *
 * `ALLO_K` moved with it so the structural end of the curve did not: 18.6 at
 * 0.62 puts a 21 cm primary within 2 % of where 20.2 at 0.78 put it. Changing
 * the exponent alone would have rescaled the whole tree.
 */
const ALLO_K = 39.0;
const ALLO_P = 0.62;

/**
 * Lateral share is drawn as `lerp(lo, hi, u²)`, and the square is the whole
 * point. A cherry does not put a strong lateral at every node - it puts a
 * handful of them and fills the rest of the branch with SPURS: 3-8 cm shoots
 * that take almost no wood and carry almost all the flowers. Sampling share
 * uniformly gave every node a strong lateral, which over-subscribed the area
 * rule (eleven nodes at a mean 21 % of the parent's cross-section is 236 % of
 * it), and the pipe model answered by collapsing the leader's radius to
 * nothing a third of the way along every limb. That is what left the crown as
 * a few bare whips. Squaring pulls the mean down to lo + (hi-lo)/4, which
 * brings the total back under 1 while KEEPING the occasional heavy fork that
 * gives a limb its structure.
 */
const shareSample = (L, u) => lerp(L.share[0], L.share[1], u * u);

/**
 * Trunk. This is a RADIUS, and it is the number the whole tree is built from:
 * 0.50 measures out at 0.96 m across the bole at breast height once the taper
 * has taken its cut - inside the 0.8-1.1 m the brief asks for - widening to
 * about 2.2 m across the root buttresses where it meets the grass.
 *
 * Because the per-level radius caps are now fractions of this number and length
 * follows radius through the allometry, this single constant really does scale
 * the whole crown. It could not before: see note 2 in the header.
 */
const TRUNK_RADIUS = 0.245;
/**
 * Bole length. 6.15 m built a VASE: the limbs left the bole between 2.15 m and
 * 5.56 m and then had 8 m of their own length to spend, so the crown's mass
 * ended up between 6 m and 14 m and the whole tree was taller than it was wide.
 * A free-standing cherry in the open divides far lower than that - the bole is
 * the part of the tree that is in shade, and in the open there is none.
 *
 * At 3.95 m the eight primaries leave between 1.55 m and 3.55 m, which is what
 * puts the widest band of the crown in its lower third instead of its middle.
 */
const TRUNK_LENGTH = 6.30;
/** The trunk is grown below the ground plane so the flare can bury itself. */
const TRUNK_BURY = 0.55;
/**
 * Root flare. FLARE_AMOUNT is the axisymmetric swelling and each of the six
 * buttresses in `roots` adds its own on top, so the peak is the sum: at 0.45
 * plus a buttress of at most 0.52 the bole goes from 1.0 m across at breast
 * height to about 2.2 m across the widest pair of roots where it meets the
 * grass. Veterans really are built like that; 0.62 plus a 0.75 buttress, which
 * is where this started, put it at 2.9 m and the tree stood on a plinth.
 */
const FLARE_HEIGHT = 0.70;
const FLARE_AMOUNT = 0.09;
/** Persistent lean, in radians of heading gained per metre of trunk. */
const TRUNK_LEAN = 0.016;
const TRUNK_LEAN_AZIMUTH = 3.9;

/**
 * Bounded burl amplitude by depth. NOT accumulated into the pipe radius - it
 * modulates the STORED radius only, so the bole can be lumpy without the taper
 * drifting. See note 1 in the header for what happens when it is not bounded.
 */
const BURL = [0.105, 0.055];

// ---------------------------------------------------------------------------
// The crown envelope
// ---------------------------------------------------------------------------

/**
 * A free-standing cherry fills a broad oblate dome: wider than it is tall, with
 * its widest band LOW and a skirt that sweeps down and out below it. That shape
 * is not decoration, it is the surface at which a shoot stops being able to
 * hold leaves - outside it a shoot is unsupported, wind-stripped and shaded by
 * nothing, so it gains nothing by being there.
 *
 * These four numbers ARE that surface, as an ellipsoid with two vertical radii
 * so the underside can be much shallower than the top:
 *
 *   widest band   y = CROWN_Y0         13.7 m across
 *   crown floor   y = Y0 - RY_DOWN      2.10 m - you can walk under the skirt
 *   crown top     y = Y0 + RY_UP       11.50 m
 *   extent        9.4 m tall by 13.7 m wide, ratio 1.46
 *   widest point  (Y0 - floor) / extent = 0.31 - the lower third
 *
 * WHY THIS EXISTS AT ALL. The silhouette used to be decided by three tropisms
 * pulling against each other with no term that knew what shape the crown was
 * supposed to be: gravity pulled a primary down 0.66 rad, crowding-escape
 * pushed it up 0.79, coherent gnarl added 0.38, and the bud score paid 0.85 for
 * pointing UP against 0.006·r² for pointing out. Whatever those four were set
 * to, the answer was a narrow upright vase whose widest point was near its top,
 * because three of the four rewarded height and only one opposed it. Suppressing
 * the vertical on two of them (see the tropism blocks below) stopped them
 * fighting gravity but could not on its own produce a dome - nothing in the
 * file wanted one. This does, and it does it with one sign change: `1 - q` is
 * positive inside the envelope and negative outside, so the same expression
 * fills the crown and then caps it.
 */
const CROWN_R = 5.75;
const CROWN_Y0 = 5.10;
const CROWN_RY_UP = 7.40;
const CROWN_RY_DOWN = 3.90;
/** Radians of heading gained per metre of shoot, at unit envelope error. */
const CROWN_TROPISM = 0.72;
/**
 * Vertical share of the inside-the-envelope tropism, relative to its radial
 * share. Below 1 because a cherry in the open is a SPREADING tree: given equal
 * room out and up, it takes out. This one number is the width-to-height ratio
 * of the crown that the envelope does not already dictate, and it is what
 * decides whether the limbs reach the top of the envelope or run out of length
 * halfway up it.
 */
const CROWN_CLIMB = 0.92;
/**
 * Envelope coordinate past which a shoot self-prunes. Not 1.0: the crown
 * outline is ragged in every reference photograph and a hard stop exactly on
 * the surface reads as a trimmed hedge. 1.16 leaves a margin of about 90 cm of
 * escaped wood at the crown equator, which is what the blossom layer fades out
 * over.
 */
const CROWN_PRUNE = 1.10;
/** Scratch for the envelope gradient - build time only, but keeps shapes flat. */
const _env = { x: 0, y: 0, z: 0 };

/**
 * Normalised envelope coordinate: q < 1 inside the crown, 1 exactly on its
 * surface, > 1 outside. When `out` is given it receives the unit gradient of q,
 * which points from the crown's core toward the nearest piece of sky.
 *
 * The gradient is the gradient of q², which differs from ∇q only by the
 * positive factor 1/(2q) - and since every consumer normalises it, that factor
 * cancels and the divide is not worth doing.
 */
function envelopeQ(x, y, z, out) {
  const dy = y - CROWN_Y0;
  const ry = dy >= 0 ? CROWN_RY_UP : CROWN_RY_DOWN;
  const a = 1 / (CROWN_R * CROWN_R);
  const b = 1 / (ry * ry);
  const q = Math.sqrt((x * x + z * z) * a + dy * dy * b);
  if (out) {
    const gx = a * x, gy = b * dy, gz = a * z;
    const gl = Math.hypot(gx, gy, gz);
    if (gl > 1e-9) { out.x = gx / gl; out.y = gy / gl; out.z = gz / gl; }
    // Dead centre of the crown: no direction is more toward the light than any
    // other, so the tropism contributes nothing rather than a random kick.
    else { out.x = 0; out.y = 0; out.z = 0; }
  }
  return q;
}

/** Horizontal radius of the crown envelope at height `y`, in metres. */
function envelopeRadiusAt(y) {
  const dy = y - CROWN_Y0;
  const ry = dy >= 0 ? CROWN_RY_UP : CROWN_RY_DOWN;
  const t = 1 - (dy * dy) / (ry * ry);
  return t > 0 ? CROWN_R * Math.sqrt(t) : 0;
}

/**
 * How far out a PRIMARY limb is allowed to reach, as a fraction of the crown's
 * horizontal radius at its own height.
 *
 * The structural limbs of a cherry occupy the inner two thirds to three
 * quarters of the crown; the outer quarter is one- and two-year wood, and that
 * is what carries the flowers that make the rim. Letting a limb run all the way
 * to the envelope surface - which is what it did, because nothing stopped it
 * before the surface - is why long bare dark limbs stood clear of the flower
 * mass in every render: a secondary budding off the tip of such a limb starts
 * AT q = 1.0, is pruned within half a metre of leaving, and the last two metres
 * of the limb end up clothed in nothing.
 *
 * Measured HORIZONTALLY rather than as an envelope coordinate, and that matters:
 * a limb leaves the bole at 1.75 m, below the envelope's floor, where q is 0.88
 * on the axis and rises toward the ground. A q test would stop every limb before
 * it had climbed into the crown at all.
 */
const PRIMARY_REACH = 0.86;

/**
 * Bud scoring against the envelope. Light lives at the crown surface, so a bud
 * that carries its shoot out toward it is worth more - but only up to the
 * surface. Past it the penalty rises quadratically, which is what keeps the
 * outline a smooth convex arc instead of a spray of escapees.
 *
 * These two terms REPLACED `-vy · 0.85 + (x² + z²) · 0.006`, where the vertical
 * reward outweighed the radial one by a factor of forty at the crown's own
 * radius. That single line is most of why the tree grew upward: every bud in
 * the tree, at every depth, was paid almost a whole unit of crowding score to
 * point at the sky.
 */
const ENV_LIGHT_GAIN = 1.05;
const ENV_ESCAPE_PENALTY = 50.0;
function envelopeScore(x, y, z) {
  const q = envelopeQ(x, y, z, null);
  let s = -ENV_LIGHT_GAIN * (q < 1 ? q : 1);
  if (q > 1) { const e = q - 1; s += ENV_ESCAPE_PENALTY * e * e; }
  return s;
}

/**
 * Gravity curvature coefficient. Fixed by requiring a primary limb to turn
 * 0.3-0.7 rad over its whole length depending on how heavy it is - the
 * difference between a limb that leaves the bole at +20° and arrives at its
 * tip at -6°, which is the whole "sweeps out and gently down" read.
 *
 * It is worth knowing how weak this tropism is relative to its neighbours,
 * because two of them were quietly beating it. The gnarl noise is spatially
 * COHERENT, so over the thirty segments of a primary it does not average out;
 * and the crowding escape vector points away from the mass of wood below a
 * limb, so it pushes up. Measured on the old tuning, gravity was pulling a
 * primary down by 0.66 rad while crowding pushed it up by 0.79 and gnarl by
 * another 0.38 - which is why limbs climbed to the top of the tree instead of
 * spreading, whatever this constant was set to. Both are now suppressed in the
 * vertical on structural wood; see the tropism blocks below.
 */
const GRAVITY_K = 5.4e-5;
const MAX_CURVATURE = 0.75;
/**
 * Total curvature any one shoot may accumulate, in radians. The per-segment
 * clamp above is not enough on its own: a 3 mm twiglet is so compliant that
 * MAX_CURVATURE over its whole length would coil it into a watch spring. This
 * caps the INTEGRAL instead, which is the physically meaningful quantity - a
 * branch deflects, it does not orbit.
 */
const TURN_BUDGET = 3.4;

/**
 * Nothing but the trunk lives below this - the browse line of a grazed field.
 * It is also the crown FLOOR, and the envelope's underside is authored to sit
 * just above it (2.10 m against 1.80 m) so the skirt hangs to about head height
 * without the outermost twigs being amputated by the browse test.
 */
const BROWSE_HEIGHT = 1.80;
const BROWSE_HEIGHT_PRIMARY = 1.45;

/** Prevailing wind while the tree grew. Rakes the crown very slightly downwind. */
const PREVAILING_X = 0.944;
const PREVAILING_Z = 0.33;

/**
 * Ceiling on queued shoots. The queue is breadth-first and radius decreases
 * with depth, so it is effectively radius-ordered: were the cap to bite it
 * would trim twiglets, not limbs.
 *
 * On the shipping seed it does NOT bite - 4 734 shoots are queued against this
 * 7 000, of which 2 561 survive the radius floor. It stays because it is the
 * only thing standing between a pathological seed and a generation pass that
 * runs for a minute, and because it is cheap: one comparison per bud. Anyone
 * tuning the tree upward should know that raising this number ALONE does
 * nothing, which is exactly the trap the previous author fell into - see
 * header note 3.
 *
 * RAISED 5 200 -> 7 000 -> 34 000. The dome needs a genuine cloud of fine wood
 * rather than eight sprays, so levels 4 and 5 now branch (they did not before)
 * and level 6 exists at all; the queue went from 4 734 to roughly 24 000. When
 * this cap bites it does not warn, it just stops budding, so the crown would
 * come out quietly thin on whichever side the breadth-first walk reached last.
 * A safety rail sitting inside 10 % of the real number is not a safety rail.
 */
const BRANCH_BUDGET = 320000;

/**
 * Crowding thresholds, in the units `PointGrid.crowd` returns - a weighted
 * count of foreign skeleton points inside the probe, where a twig point is
 * worth about 0.6 and a limb point about 2.
 *
 * STEER is the working number: it fires constantly through the crown and is
 * what keeps two limbs out of the same metre of air. ABORT is now doing a
 * second job as well - the bud score it is compared against carries the
 * envelope terms (see `envelopeScore`), so a bud aimed well outside the crown
 * fails this test even in completely empty air, which is what stops the crown's
 * outline from growing whiskers. The comparison is shifted by ENV_LIGHT_GAIN at
 * the call site so this constant still means what it says about crowding.
 *
 * Do not reach for either of these to shape the crown. Sweeping ABORT from 11
 * down to 1.6 changes the twig cloud's radial distribution by 0.01 and just
 * thins the whole canopy; the shape comes from the envelope.
 */
const CROWD_STEER = 2.6;
const CROWD_ABORT = 2.8;

/**
 * Deepest level that pays for collision testing. Below this, wood is 5 mm
 * across and buried under blossom - interpenetration is invisible, and the
 * test is the single most expensive thing in generation. Raising the shoot
 * count eightfold is only affordable because of this line.
 */
const COLLIDE_MAX_DEPTH = 3;

/**
 * FOURTH MECHANISM THAT WAS WRONG: `L.rMin` was doing two incompatible jobs.
 *
 * It gates whether a BUD is worth growing (`childR > childLevel.rMin * 1.15`),
 * which is what it reads as. But it was also the shoot's own TERMINATION test
 * (`if (r < L.rMin) break`), and those two numbers cannot be the same, because
 * a shoot's radius is supposed to arrive at `r0 * L.taperTo` at its tip. When
 * `taperTo` was tightened to 0.34/0.22/0.12 to stop the twigs ending in sawn-off
 * stubs, that decay curve started crossing `rMin` LONG before the tip:
 *
 *   depth 3, r0 = 6.5 mm, rMin 3.8 mm, taperTo 0.34  -> dies at 50 % of length
 *   depth 4, r0 = 4.2 mm, rMin 2.0 mm, taperTo 0.22  -> dies at 49 %
 *   depth 5, r0 = 1.9 mm, rMin 0.6 mm, taperTo 0.12  -> dies at 54 %
 *
 * and a shoot that dies at half length never reaches most of its own lateral
 * nodes, because `nextNode` is scheduled against `maxLen`. Measured: the crown
 * held 357 tertiaries, 251 twigs and 147 twiglets - a branching ratio of 0.70
 * and then 0.59, i.e. the tree was CONVERGING with depth where a real crown
 * multiplies. That is why 41 000 flowers looked like clumps on bare sticks:
 * there were only 2 947 places to hang them.
 *
 * Termination is now the shoot's OWN taper target (it may always complete the
 * taper it was authored with), floored by a single absolute number that exists
 * only so the mesh never has to sweep a tube thinner than it can rasterise.
 * `rMin` keeps the bud-viability job it reads as.
 */
const TIP_RADIUS_FLOOR = 0.00028;

/**
 * Apical continuation. A shoot that reaches the end of its allometric length
 * does not stop existing - next season its apical bud extends it as thinner
 * wood, which is why a real limb thins continuously from bole to twig instead
 * of ending in mid-air at a third of its base diameter.
 *
 * Without this every leader in the tree terminated at whatever radius the taper
 * had reached: measured, the eight primaries ended at a median 33 % of their
 * base radius, i.e. a 15 cm flat disc hanging in the sky, and the secondaries
 * at 3.7 cm. Those are the blunt ends visible on every limb in the renders, and
 * no amount of tip taper fixes them because the taper had already done its job - 
 * the limb was simply the wrong length for its thickness.
 *
 * The continuation is the same limb, so it inherits the parent's heading and
 * sway phase rather than budding sideways at a divergence angle. It costs one
 * shoot per leader and it is the single largest source of new flowering wood at
 * the crown's outer edge, which is exactly where the silhouette is decided.
 */
const CONTINUE_MIN_RADIUS = 0.0016;
/** Wood kept across the apical joint. Slightly < 1: a bud scar is a real step. */
const CONTINUE_TAPER = 0.93;

/**
 * Per-depth growth character.
 *
 *   taperTo   fraction of base radius left at the tip from taper ALONE - 
 *             forks remove more on top of it.
 *   nodes     how many lateral buds this shoot tries to place along itself.
 *             A COUNT, not a spacing: that is what makes short shoots branch.
 *   nodeMin   metric floor on the resulting internode, so a 12 cm twiglet does
 *             not sprout six laterals a centimetre apart.
 *   rMaxK     radius ceiling as a fraction of TRUNK_RADIUS.
 *   envelope  how hard the crown envelope steers this depth. Structural wood
 *             carries the silhouette, so it is steered hardest; a twiglet is
 *             already where its parent put it.
 *
 * THE BRANCHING RATIO IS THE WHOLE OF CROWN DENSITY, and it used to CONVERGE.
 * Measured on the shipping tree: 8 primaries, 88 secondaries, 583 tertiaries,
 * 1 621 twigs, 260 twiglets - ratios 11.0, 6.6, 2.8, then 0.16. Level 5 had
 * `nodes: 0` and `maxChildren: 0`, so the tree simply STOPPED two levels above
 * the wood that actually carries flowers, and the 260 twiglets that existed
 * were apical continuations rather than branches. A crown that stops
 * multiplying is eight sprays no matter how the sprays are aimed, and that is
 * what every render of it showed: sticks with beads of flower along them and
 * sky in between.
 *
 * Levels 4 and 5 now branch and level 6 exists, `firstNode` comes in hard at
 * the structural levels so laterals start near the bole rather than a third of
 * the way out, and the ratios run 11 / 8 / 5 / 3.5 / 2.5. That multiplies
 * flowering wood roughly threefold - and flowering wood is exactly what
 * blossoms.js has to hang a solid crown on. It is the real lever: more sites
 * beats more flowers per site every single time.
 */
const LEVELS = [
  // 0 - trunk: heavy, sinuous, no droop, gnarl strongest near the ground
  {
    segLen: 0.16, taperTo: 0.62, gnarl: 0.052, gnarlFreq: 0.42, droop: 0.05,
    photo: 0.02, rMin: 0.02, rMaxK: 1.30, sweep: 0.009, maxChildren: 0,
    divergence: [0.5, 0.9], nodes: 0, nodeMin: 1, firstNode: 0.3, share: [0.2, 0.3],
    envelope: 0,
  },
  // 1 - primary limbs: long, spreading, real sag
  //
  // firstNode 0.30 -> 0.11. A third of the way along an 8 m limb is 2.6 m of
  // bare wood leaving the bole in every direction, and it is the single reason
  // the trunk and its forks stood naked in the middle of the crown. On a
  // free-standing cherry the limbs vanish into flower within a metre or two of
  // the bole; the proximal self-pruning this used to model is what happens in a
  // FOREST, where that wood is shaded by neighbours. There are none here.
  {
    segLen: 0.185, taperTo: 0.52, gnarl: 0.066, gnarlFreq: 0.62, droop: 1.05,
    photo: 0.10, rMin: 0.015, rMaxK: 0.72, sweep: 0.030,
    nodes: 44, nodeMin: 0.10, firstNode: 0.07, share: [0.11, 0.42], maxChildren: 46,
    divergence: [0.66, 1.16], envelope: 1.45,
  },
  // 2 - secondary: the scaffold of the crown, and where the weeping starts
  {
    segLen: 0.145, taperTo: 0.52, gnarl: 0.086, gnarlFreq: 0.95, droop: 0.92,
    photo: 0.28, rMin: 0.0075, rMaxK: 0.40, sweep: 0.044,
    nodes: 32, nodeMin: 0.058, firstNode: 0.08, share: [0.10, 0.44], maxChildren: 32,
    divergence: [0.56, 1.06], envelope: 1.30,
  },
  // 3 - tertiary: the flowering wood starts here
  {
    segLen: 0.100, taperTo: 0.34, gnarl: 0.112, gnarlFreq: 1.5, droop: 1.02,
    photo: 0.46, rMin: 0.0034, rMaxK: 0.185, sweep: 0.058,
    nodes: 27, nodeMin: 0.026, firstNode: 0.09, share: [0.095, 0.46], maxChildren: 27,
    divergence: [0.48, 0.98], envelope: 0.88,
  },
  // 4 - twigs. `nodes` was 6 with maxChildren 8 and it branched; the level
  // BELOW it did not, so this was the last multiplication in the tree.
  {
    segLen: 0.068, taperTo: 0.24, gnarl: 0.128, gnarlFreq: 2.4, droop: 0.86,
    photo: 0.64, rMin: 0.00105, rMaxK: 0.078, sweep: 0.074,
    nodes: 22, nodeMin: 0.016, firstNode: 0.09, share: [0.060, 0.40], maxChildren: 22,
    divergence: [0.42, 0.88], envelope: 0.70,
  },
  // 5 - twiglets: two-year spur wood. This is where a cherry's flowers live, so
  // it has to be a THICKET, not a terminal level.
  {
    segLen: 0.048, taperTo: 0.20, gnarl: 0.132, gnarlFreq: 3.6, droop: 0.70,
    photo: 0.78, rMin: 0.00042, rMaxK: 0.034, sweep: 0.080,
    nodes: 17, nodeMin: 0.012, firstNode: 0.10, share: [0.090, 0.44], maxChildren: 17,
    divergence: [0.36, 0.78], envelope: 0.55,
  },
  // 6 - spurs: last season's growth, upturned, terminal.
  //
  // Terminal wood tapers hard to a point. At 0.55 a twiglet ended at more than
  // half its base radius, which rasterises as a flat sawn-off stub - the single
  // most artificial thing about the crown at close range. Real terminal growth
  // narrows to a bud. rMin drops with it so the taper is not immediately
  // clamped back to a stub by the radius floor.
  {
    segLen: 0.036, taperTo: 0.12, gnarl: 0.136, gnarlFreq: 4.6, droop: 0.52,
    photo: 0.92, rMin: 0.00022, rMaxK: 0.018, sweep: 0.086,
    nodes: 0, nodeMin: 0.04, firstNode: 0.3, share: [0.25, 0.45], maxChildren: 0,
    divergence: [0.30, 0.66], envelope: 0.45,
  },
];

/** Resolved once, so the growth loop never repeats the multiply. */
for (const L of LEVELS) L.rMax = L.rMaxK * TRUNK_RADIUS;

/**
 * The primary limbs. This table IS the tree's character.
 * `azimuth` is a world compass angle (radians), `elevation` is from horizontal,
 * `at` is the HEIGHT above the ground at which the limb leaves the bole.
 *
 * Ten living limbs and one old break, all out of the bottom 3.58 m of a 3.95 m
 * trunk: a veteran cherry standing in the open divides LOW and hard, and the
 * crown is built almost entirely out of what those limbs do afterwards.
 *
 * THE AZIMUTHS ARE NEARLY EVEN, and that is a correction. The old table put
 * them at deliberately irregular compass angles "so the crown has no axis of
 * symmetry" and then let them drift: two limbs ended up 0.16 rad apart and
 * there was a 1.14 rad - sixty-five degree - sector with nothing in it but the
 * broken stub. Rendered from that side the crown had a bite out of it that no
 * amount of blossom density could fill, because there was no wood there to
 * flower on. Irregularity belongs in the growth (gnarl, bud choice, crowding
 * are all seeded and none of them is symmetric); the STRUCTURE has to cover
 * every compass direction or the tree is only finished from one side. These
 * are spaced 0.47 to 0.62 rad apart, which is even enough to have no hole and
 * uneven enough to have no axis.
 *
 * RE-AUTHORED FOR A DOME. The previous table was a three-tier VASE - two limbs
 * leaving at 0.21 and 0.30 rad, three at 0.58 to 0.76, and a terminal leader at
 * 1.10 rad taking everything that was left at 5.56 m up the bole. That
 * structure cannot make a dome at any tropism setting, because half its wood
 * starts above the height the dome is supposed to be widest at and the terminal
 * leader is a spire by construction. Read the elevations here as a fan: nothing
 * leaves the bole above 0.62 rad (35°), the two lowest go out at 6 and 9
 * degrees and hold the skirt, and the limb that continues the bole is
 * `crown` at 0.62 - steep enough to reach 11 m once the envelope has arced it
 * over, shallow enough that it is a limb and not a mast.
 *
 * `vigor` is the other half of the width. It multiplies the allometric length,
 * and the low limbs get the most of it (1.55 on `hero`) because in the open the
 * limb with the clearest air is the one that runs out sideways under everything
 * else. Their `droop` is high for the same reason: an 8 m limb carrying its own
 * weight plus flowers arcs over, and that arc IS the dome's outline.
 *
 * The shares rise as the trunk thins (0.15 to 0.34) so that the ninth limb is
 * not a twig off a spent leader; the terminal `crown` takes what is left. The
 * azimuths are deliberately unevenly spaced and no two are within 0.35 rad, so
 * the crown has no axis of symmetry and no two limbs shadow each other.
 */
const PRIMARIES = [
  // Attach heights were 1.75-3.58 m on a 3.95 m bole, which put the lowest limb
  // at head height and gave the tree a squat, spreading, left-to-right
  // silhouette. On the taller 6.3 m bole they start at 3.3 m, so there is a
  // clean length of trunk before the crown opens, and every elevation is
  // steepened by ~0.18 rad with droop pulled back to match: the crown now grows
  // UP and out rather than out and down.
  { key: 'hero', at: 2.30, azimuth: 3.55, elevation: 0.20, share: 0.36, vigor: 1.50, droop: 1.30, photo: 0.75 },
  { key: 'east', at: 2.50, azimuth: 0.78, elevation: 0.22, share: 0.37, vigor: 1.50, droop: 1.26, photo: 0.80 },
  { key: 'south', at: 2.70, azimuth: 5.20, elevation: 0.24, share: 0.38, vigor: 1.48, droop: 1.24, photo: 0.80 },
  { key: 'low', at: 2.90, azimuth: 1.95, elevation: 0.23, share: 0.38, vigor: 1.46, droop: 1.24, photo: 0.85 },
  { key: 'stub', at: 3.88, azimuth: 2.42, elevation: 0.66, share: 0.09, vigor: 0.22, droop: 0.64, photo: 1.00, broken: true },
  { key: 'north', at: 3.45, azimuth: 4.66, elevation: 0.30, share: 0.42, vigor: 1.58, droop: 1.12, photo: 0.90 },
  { key: 'west', at: 3.70, azimuth: 0.20, elevation: 0.33, share: 0.43, vigor: 1.58, droop: 1.08, photo: 0.85 },
  { key: 'mid', at: 4.46, azimuth: 2.98, elevation: 0.73, share: 0.24, vigor: 1.34, droop: 0.70, photo: 0.95 },
  { key: 'inner', at: 4.30, azimuth: 4.10, elevation: 0.92, share: 0.36, vigor: 1.72, droop: 0.76, photo: 1.00 },
  { key: 'upper', at: 4.50, azimuth: 1.36, elevation: 1.20, share: 0.38, vigor: 1.78, droop: 0.62, photo: 1.05 },
  // Two steep limbs, not one. The crown's top-centre - the column of air
  // directly over the bole between 7 m and 10 m - is reached by nothing else:
  // the envelope tropism pushes every shoot inside the crown radially outward,
  // so a single terminal leader is carried off the axis within a couple of
  // metres and the dome comes out with a V-shaped notch in the middle of its
  // top edge, which reads as two crowns rather than one from every side.
  { key: 'spire', at: 4.70, azimuth: 2.70, elevation: 1.34, share: 0.58, vigor: 2.02, droop: 0.26, photo: 1.10 },
  { key: 'core', at: 4.80, azimuth: 4.35, elevation: 1.42, share: 0.62, vigor: 2.06, droop: 0.22, photo: 1.12 },
  { key: 'cap-e', at: 4.58, azimuth: 0.90, elevation: 1.30, share: 0.46, vigor: 1.98, droop: 0.26, photo: 1.10 },
  { key: 'cap-w', at: 4.88, azimuth: 5.60, elevation: 1.36, share: 0.50, vigor: 2.02, droop: 0.24, photo: 1.10 },
  { key: 'fill-a', at: 2.62, azimuth: 1.48, elevation: 0.22, share: 0.40, vigor: 1.60, droop: 1.24, photo: 0.82 },
  { key: 'fill-b', at: 3.30, azimuth: 1.84, elevation: 0.40, share: 0.38, vigor: 1.54, droop: 1.06, photo: 0.88 },
  { key: 'fill-c', at: 4.05, azimuth: 1.28, elevation: 0.72, share: 0.34, vigor: 1.50, droop: 0.82, photo: 0.98 },
  { key: 'fill-d', at: 2.84, azimuth: 5.42, elevation: 0.24, share: 0.40, vigor: 1.60, droop: 1.22, photo: 0.82 },
  { key: 'fill-e', at: 3.52, azimuth: 5.96, elevation: 0.44, share: 0.38, vigor: 1.54, droop: 1.02, photo: 0.88 },
  { key: 'fill-f', at: 4.22, azimuth: 5.72, elevation: 0.76, share: 0.34, vigor: 1.50, droop: 0.80, photo: 0.98 },
  { key: 'fill-g', at: 2.96, azimuth: 6.12, elevation: 0.26, share: 0.42, vigor: 1.64, droop: 1.20, photo: 0.84 },
  { key: 'fill-h', at: 3.78, azimuth: 6.20, elevation: 0.52, share: 0.38, vigor: 1.56, droop: 0.96, photo: 0.92 },
  { key: 'fill-i', at: 3.16, azimuth: 4.14, elevation: 0.30, share: 0.40, vigor: 1.60, droop: 1.16, photo: 0.86 },
  { key: 'fill-j', at: 3.62, azimuth: 2.18, elevation: 0.42, share: 0.38, vigor: 1.56, droop: 1.04, photo: 0.90 },
  { key: 'fill-k', at: 4.34, azimuth: 0.96, elevation: 0.66, share: 0.34, vigor: 1.50, droop: 0.86, photo: 0.96 },
  { key: 'crown', at: 4.96, azimuth: 3.20, elevation: 1.48, share: 0.999, vigor: 1.84, droop: 0.24, photo: 1.10, terminal: true },
];

/** Levels 1..4 own one oscillator each; deeper wood rides its depth-4 ancestor. */
const SWAY_LEVELS = 4;
/** How much a branch deeper than level L still contributes to level L's motion. */
const SWAY_CARRY = 0.4;

/**
 * Radius classes. Mesh detail is chosen by how THICK a branch is, not by how
 * deep it sits in the hierarchy - an epicormic sprout at depth 2 and a twig at
 * depth 4 can be the same 8 mm across and deserve the same three radial
 * segments. Keying off depth spent a twelve-sided tube on both.
 */
const RADIUS_CLASS = [0.300, 0.145, 0.065, 0.028, 0.012];

function radiusClass(r) {
  for (let i = 0; i < RADIUS_CLASS.length; i++) if (r > RADIUS_CLASS[i]) return i;
  return RADIUS_CLASS.length;
}

/**
 * Mesh detail per quality tier, indexed by radius class. `stride` decimates
 * skeleton points into rings.
 *
 * `maxBranches` is the LOW/MEDIUM cost dial. Draw order is thickest-first - 
 * and a child's base radius is strictly smaller than its parent's under the
 * area rule, so a prefix of that order is always a connected sub-tree - which
 * means the branches it drops are the 2-4 mm twiglets at the very outside of
 * the crown. That is exactly the wood buried under blossom cards several times
 * its own width. The blossoms still grow on them, which is the point: out
 * there the flowers are what the eye reads, and paying for a twelve-triangle
 * tube behind each one buys nothing.
 *
 * MEASURED, on the shipping seed, after the crown-envelope rebuild took the
 * tree from 2 561 shoots to 15 200 and from 963 m of wood to 5 060 m:
 *
 *   tier     branches drawn   thinnest drawn   wood drawn   vertices  triangles
 *   low          520/15221       9.69 mm r         761 m      8 803     11 085
 *   medium       950/15221       6.74 mm r       1 136 m     15 269     18 776
 *   high       1 750/15221       4.47 mm r       1 695 m     25 772     30 165
 *   ultra      3 200/15221       2.58 mm r       2 457 m     48 566     56 651
 *
 * The `maxBranches` numbers came DOWN even though the tree has six times the
 * wood, and that is the whole reason the rebuild fits in the frame budget: HIGH
 * costs 30 165 triangles against the old table's 25 956 - 16 % more - on a
 * crown that carries five times the flowering wood. The 13 500 branches HIGH
 * does not draw are all under 9 mm ACROSS and every one of them is buried under
 * a flower cloud that is now genuinely opaque, which was not true before and is
 * what makes dropping them safe.
 *
 * Note the interaction with the terminal-bud cap. Draw order is thickest-first,
 * so a tier's undrawn set is overwhelmingly the apical continuations, and every
 * limb whose continuation is dropped needs a cap of its own or it rasterises as
 * an open tube - a flat sawn-off disc. See `apicalDrawn` in
 * buildBranchGeometry, and do not replace it with the skeleton's `hasApical`.
 *
 * The wood was never the expensive half of this asset - the 103 000
 * double-sided alpha-tested blossom cards are, and that is where the tiers do
 * their real work (see TIERS in blossoms.js).
 */
export const LOD_TIERS = {
  // maxBranches is the number of shoots that actually reach the MESH, selected
  // thickest-first. The skeleton grows BRANCH_BUDGET (34 000) of them, so at
  // 1750 the HIGH tier was meshing five percent of the tree - and because the
  // selection is thickest-first, the five percent it kept was the trunk and the
  // primary limbs while every twig was discarded. The result rendered as long
  // bare spokes with blossoms floating in clumps along them, since blossoms are
  // placed from the SKELETON's twig sites and those twigs existed everywhere the
  // mesh did not. Fine wood is cheap - a spur is 3 radial segments and 2 rings - 
  // so the honest budget is an order of magnitude higher.
  low: { radial: [8, 6, 4, 3, 3, 3], stride: [1, 2, 3, 5, 6, 7], maxDepth: 6, maxBranches: 26000 },
  medium: { radial: [10, 7, 5, 3, 3, 3], stride: [1, 2, 3, 4, 5, 6], maxDepth: 6, maxBranches: 62000 },
  high: { radial: [12, 8, 5, 4, 3, 3], stride: [1, 2, 4, 6, 8, 9], maxDepth: 6, maxBranches: 130000 },
  ultra: { radial: [16, 11, 7, 5, 4, 3], stride: [1, 2, 3, 4, 6, 7], maxDepth: 6, maxBranches: 200000 },
};

/** Bark texel scale: one texture tile covers this much surface, in metres. */
const BARK_TILE = 0.52;

// ===========================================================================
// Small helpers - build time only, so allocation here is free
// ===========================================================================

const allometricLength = (r) => ALLO_K * Math.pow(Math.max(r, 1e-5), ALLO_P);

/** Exact uniform-load cantilever deflection shape, normalised to 1 at the tip. */
const complianceShape = (u) => {
  const t = clamp01(u);
  return (t * t * (6 - 4 * t + t * t)) / 3;
};

/** Dimensionless softness: long and thin bends, short and fat does not. */
const softness = (len, r) => clamp(len / (1 + 60 * Math.pow(Math.max(r, 1e-4), 1.5)), 0, 1);

function normalize3(v) {
  const l = Math.hypot(v.x, v.y, v.z) || 1;
  v.x /= l; v.y /= l; v.z /= l;
  return v;
}

/** Any unit vector perpendicular to d, avoiding the degenerate axis. */
function perpendicular(d, out) {
  if (Math.abs(d.y) < 0.9) { out.x = -d.z; out.y = 0; out.z = d.x; }
  else { out.x = 1; out.y = 0; out.z = 0; }
  return normalize3(out);
}

/**
 * Uniform hash grid over every point grown so far. Keys are exact (no modular
 * collisions) inside the ±256 m box the tree could ever occupy.
 */
class PointGrid {
  constructor(cell) {
    this.cell = cell;
    this.inv = 1 / cell;
    this.map = new Map();
  }
  _key(ix, iy, iz) {
    return (ix + 512) * 1048576 + (iy + 512) * 1024 + (iz + 512);
  }
  insert(x, y, z, branchId, r) {
    const k = this._key(
      Math.floor(x * this.inv), Math.floor(y * this.inv), Math.floor(z * this.inv)
    );
    let cell = this.map.get(k);
    if (!cell) { cell = []; this.map.set(k, cell); }
    cell.push(x, y, z, branchId, r);
  }
  /**
   * Weighted count of foreign points near (x,y,z); `out` receives the escape
   * direction (unnormalised). Thick wood repels harder than a twig, but only
   * ~4x harder - an earlier 24x weighting made a single trunk point veto every
   * bud within half a metre of the bole and the tree came out nearly bare.
   */
  crowd(x, y, z, radius, selfId, parentId, out) {
    const r2 = radius * radius;
    const ix = Math.floor(x * this.inv), iy = Math.floor(y * this.inv), iz = Math.floor(z * this.inv);
    const span = Math.ceil(radius * this.inv);
    let total = 0;
    out.x = 0; out.y = 0; out.z = 0;
    for (let a = -span; a <= span; a++) {
      for (let b = -span; b <= span; b++) {
        for (let c = -span; c <= span; c++) {
          const cell = this.map.get(this._key(ix + a, iy + b, iz + c));
          if (!cell) continue;
          for (let i = 0; i < cell.length; i += 5) {
            const id = cell[i + 3];
            if (id === selfId || id === parentId) continue;
            const dx = x - cell[i], dy = y - cell[i + 1], dz = z - cell[i + 2];
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > r2) continue;
            const w = (1 - d2 / r2) * (0.5 + 8 * Math.min(cell[i + 4], 0.2));
            total += w;
            const inv = w / Math.sqrt(d2 + 1e-6);
            out.x += dx * inv; out.y += dy * inv; out.z += dz * inv;
          }
        }
      }
    }
    return total;
  }
  /**
   * Wood density around a point, for baked AO. Weighted so a nearby trunk
   * darkens a crotch strongly while a thicket of twigs darkens gently.
   */
  density(x, y, z, radius, selfId) {
    const r2 = radius * radius;
    const ix = Math.floor(x * this.inv), iy = Math.floor(y * this.inv), iz = Math.floor(z * this.inv);
    const span = Math.ceil(radius * this.inv);
    let total = 0;
    for (let a = -span; a <= span; a++) {
      for (let b = -span; b <= span; b++) {
        for (let c = -span; c <= span; c++) {
          const cell = this.map.get(this._key(ix + a, iy + b, iz + c));
          if (!cell) continue;
          for (let i = 0; i < cell.length; i += 5) {
            if (cell[i + 3] === selfId) continue;
            const dx = x - cell[i], dy = y - cell[i + 1], dz = z - cell[i + 2];
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > r2) continue;
            total += (1 - d2 / r2) * (0.06 + 12 * Math.min(cell[i + 4], 0.12));
          }
        }
      }
    }
    return total;
  }
}

// ===========================================================================
// Skeleton generation
// ===========================================================================

/**
 * Grows the tree. Deterministic in `seed`: same seed, same tree, every reload
 * and every quality tier.
 *
 * Costs roughly 480 ms once, at init, up from 150 ms before the crown-envelope
 * rebuild - the tree grows 15 200 shoots where it grew 2 561. It is behind the
 * loading screen and it is the whole silhouette of the hero asset, so the trade
 * is deliberate; the twig walk adds 35 ms and blossom placement 190 ms on top,
 * for about 730 ms of total build. If it has to come down, the first dial is
 * COLLIDE_MAX_DEPTH and the second is `insertEvery` - between them they are
 * most of the cost, and neither changes the crown's outline.
 *
 * @returns {object} skeleton
 */
export function generateSkeleton({ seed = 0x5ce4a1 } = {}) {
  const rng = makeRNG(seed);
  const nz = createNoise(seed ^ 0x2f1d3b);
  const grid = new PointGrid(0.5);
  /** Subsampled occluder set for the AO pass - AO is a smooth field, so a
   *  third of the occluders is indistinguishable and three times cheaper. */
  const aoGrid = new PointGrid(0.5);

  const branches = [];
  const grown = new Set();
  const queue = [];
  const escape = { x: 0, y: 0, z: 0 };
  const e1 = { x: 0, y: 0, z: 0 };
  const e2 = { x: 0, y: 0, z: 0 };
  const gvec = { x: 0, y: 0, z: 0 };

  let nextId = 1;

  queue.push({
    id: 0, parent: -1, parentIndex: -1, depth: 0,
    x: 0, y: -TRUNK_BURY, z: 0,
    dx: Math.cos(TRUNK_LEAN_AZIMUTH) * 0.05, dy: 1, dz: Math.sin(TRUNK_LEAN_AZIMUTH) * 0.05,
    r: TRUNK_RADIUS * 1.18, len: TRUNK_LENGTH + TRUNK_BURY,
    vigor: 1, droop: 1, photo: 1, broken: false, primary: 'trunk',
  });

  for (let head = 0; head < queue.length; head++) {
    const req = queue[head];
    // A shoot whose parent failed to grow has nothing to attach to. Skipping it
    // here transitively drops the whole orphaned subtree, because children are
    // only ever enqueued from inside their parent's own growth.
    if (req.parent >= 0 && !grown.has(req.parent)) continue;
    const b = growShoot(req);
    if (!b) continue;
    grown.add(b.id);
    branches.push(b);
  }

  /**
   * Grows one shoot from base to termination, scheduling its own children.
   * Tropisms are integrated per segment so curvature accumulates the way a real
   * shoot's does, rather than being applied as a post-hoc bend.
   */
  function growShoot(req) {
    const depth = req.depth;
    const L = LEVELS[Math.min(depth, LEVELS.length - 1)];
    const brng = makeRNG((seed ^ Math.imul(req.id + 1, 0x9e3779b1)) >>> 0);
    const noiseOffset = brng() * 400;

    const maxLen = req.len;
    const step = Math.max(
      0.030,
      L.segLen * clamp(maxLen / allometricLength(req.r), 0.45, 1.5)
    );
    const steps = Math.max(3, Math.ceil(maxLen / step));
    const seg = maxLen / steps;
    const taperStep = Math.pow(Math.pow(L.taperTo, 1 / Math.max(maxLen, 0.05)), seg);

    const px = [], py = [], pz = [], pr = [], ps = [];
    let x = req.x, y = req.y, z = req.z;
    const r0 = Math.min(req.r, L.rMax);
    let r = r0;
    const dir = { x: req.dx, y: req.dy, z: req.dz };
    normalize3(dir);

    /**
     * Termination radius. A shoot is entitled to complete the taper it was
     * authored with, so this sits BELOW `r0 * taperTo` - the 0.55 leaves room
     * for the wood the forks take on top of the taper without ending the shoot
     * the moment a heavy lateral is placed. `TIP_RADIUS_FLOOR` is the only
     * absolute limit, and it is a rasterisation limit, not a botanical one.
     * See TIP_RADIUS_FLOOR for what using `L.rMin` here cost the crown.
     *
     * `min` against `L.rMin`, not `max`: this must be strictly MORE permissive
     * than the old test on every shoot in the tree, or tightening the taper on
     * thin wood would start ending thick wood early. On a primary the taper
     * target is 4.7x `rMin` and `rMin` still wins; on a 6.5 mm tertiary the
     * taper target is a third of it and the taper wins.
     */
    const rStop = Math.max(TIP_RADIUS_FLOOR, Math.min(L.rMin, r0 * L.taperTo * 0.55));

    // A constant lateral sweep gives a branch one coherent arc. Noise alone
    // reads as a wiggle, not as wood that grew toward a gap in the light.
    const sweepAz = brng() * TAU;
    const sweepAmt = L.sweep * (0.4 + 1.2 * brng()) * (brng() < 0.5 ? -1 : 1);

    // Structural wood is allowed lower than fine wood: the primaries now leave
    // the bole at 1.55 m, so testing them against the fine-wood browse line at
    // 1.80 would kill the two limbs that hold the skirt on their first segment.
    const browse = depth <= 2 ? BROWSE_HEIGHT_PRIMARY : BROWSE_HEIGHT;

    // --- lateral node spacing ------------------------------------------------
    // The node COUNT is the authored quantity and the spacing follows from the
    // shoot's own length, so a 40 cm twig branches as readily as a 7 m limb.
    // Spacing it in absolute metres instead is what left the old crown as a few
    // bare whips: see header note 3.
    const nodeSpan = maxLen * (0.92 - L.firstNode);
    const internode = L.nodes > 0
      ? Math.max(L.nodeMin, nodeSpan / L.nodes) * (0.86 + 0.28 * brng())
      : Infinity;
    let nextNode = L.firstNode * maxLen;

    let childCount = 0;
    let phyllo = brng() * TAU;
    let crowdedRun = 0;
    let turned = 0;
    let s = 0;
    let terminate = false;
    /** True only if the shoot walked its whole allometric length. */
    let reachedTip = false;
    const collide = depth >= 1 && depth <= COLLIDE_MAX_DEPTH;
    // Thin wood still has to EXIST in the grid for thicker wood to steer around
    // it, but at a third of the sample rate: the probe radius is many times the
    // point spacing out there, so the field it produces is unchanged.
    // Raised from 3 to 5 on fine wood: the crown now carries three times the
    // twigs it did, and every inserted point is paid for again by every
    // collision probe that sweeps past it.
    const insertEvery = depth <= COLLIDE_MAX_DEPTH ? 1 : 5;
    const burl = depth < BURL.length ? BURL[depth] : 0;
    /**
     * Total and per-segment turn ceilings.
     *
     * The budget SCALES WITH LENGTH now, and it has to: with the allometry
     * refitted, a 2 mm twig is 39 cm rather than 10 cm, and gravity's curvature
     * on wood that thin is enormous (κ ∝ 1/r⁴, so it saturates the per-segment
     * clamp on every step). A flat 3.4 rad budget let such a shoot deflect more
     * than half a full circle before it ran out - a watch spring, not a twig.
     * A branch deflects in proportion to how long a lever it is; 1.2 rad per
     * metre, floored so even the shortest spur can nod and capped at the old
     * value for a structural limb, is that statement.
     */
    const turnTotal = clamp(1.2 * maxLen, 0.55, TURN_BUDGET);
    const turnCap = Math.min(MAX_CURVATURE, turnTotal / Math.max(maxLen, 0.12));

    const primaryFlags = depth === 0 ? PRIMARIES.map(() => false) : null;

    for (let i = 0; i <= steps; i++) {
      // Burl is a modulation of the pipe radius at THIS point, never a factor
      // fed back into `r`. The distinction is the whole of header note 1.
      const stored = burl > 0
        ? r * (1 + burl * nz.noise3D(x * 1.35, y * 1.05 + 11.3, z * 1.35))
        : r;
      px.push(x); py.push(y); pz.push(z); pr.push(stored); ps.push(s);
      if (i % insertEvery === 0) {
        grid.insert(x, y, z, req.id, r);
        if (i % 3 === 0) aoGrid.insert(x, y, z, req.id, r);
      }
      if (i === steps || terminate) break;

      // --- generic lateral buds -------------------------------------------
      if (
        depth < LEVELS.length - 1 && childCount < L.maxChildren &&
        s >= nextNode && s < maxLen * 0.92 && queue.length < BRANCH_BUDGET
      ) {
        // Apical vigour: the same bud is worth more wood the further out along
        // the parent it sits. Without it the pipe model hands the biggest
        // children to the laterals nearest the bole, whose subtrees then pile
        // up against the trunk - the crown came out solid in the middle and
        // thin at the rim, which is backwards for a cherry. Clamped because a
        // share at or above 1 would take the parent's entire cross-section.
        const share = clamp(shareSample(L, brng()) * (0.72 + 0.85 * (s / maxLen)), 0.01, 0.82);
        const childLevel = LEVELS[Math.min(depth + 1, LEVELS.length - 1)];
        const childR = Math.min(
          r * Math.pow(share, 1 / AREA_EXP) * FORK_LOSS,
          childLevel.rMax
        );
        if (childR > childLevel.rMin * 1.15) {
          phyllo += 2.39996323 + (brng() - 0.5) * 0.55;
          const div = lerp(L.divergence[0], L.divergence[1], brng());
          const chosen = chooseShootDir(x, y, z, dir, phyllo, div, childR);
          if (chosen) {
            // Only now does the parent give up the wood - an aborted bud costs
            // the branch nothing, exactly as in a real tree.
            r *= Math.pow(1 - share, 1 / AREA_EXP);
            const vig = 0.74 + 0.54 * brng();
            // A lateral may not outrun the tip it grew behind: the surviving
            // length of the parent is a real ceiling on it. Softened from the
            // old rule by carrying a fraction of the parent's TOTAL length, so
            // a bud two thirds of the way along a heavy limb still makes a
            // proper bough instead of a stick.
            const childLen = Math.min(
              allometricLength(childR) * vig,
              (maxLen - s) * 1.25 + 0.22 * maxLen
            );
            queue.push({
              id: nextId++, parent: req.id, parentIndex: branches.length,
              depth: depth + 1,
              x, y, z, dx: chosen.x, dy: chosen.y, dz: chosen.z,
              r: childR,
              len: childLen,
              vigor: vig,
              // Long shoots sag; short ones do not. Handing every child the
              // same droop of 1 made the secondaries a stiff fan, which is the
              // opposite of the slightly weeping crown a cherry has.
              droop: 0.80 + 0.55 * clamp01(childLen / 3.2),
              photo: 1, broken: false, primary: null,
            });
            childCount++;
          }
        }
        nextNode += internode * (0.78 + 0.44 * brng());
      }

      // --- hand-authored primaries off the trunk ---------------------------
      if (depth === 0) {
        for (let k = 0; k < PRIMARIES.length; k++) {
          if (primaryFlags[k]) continue;
          const P = PRIMARIES[k];
          if (s + seg < P.at + TRUNK_BURY) continue;
          primaryFlags[k] = true;
          const childR = Math.min(
            r * Math.pow(P.share, 1 / AREA_EXP) * FORK_LOSS, LEVELS[1].rMax
          );
          r *= Math.pow(Math.max(1 - P.share, 0.02), 1 / AREA_EXP);
          const ce = Math.cos(P.elevation);
          const cd = {
            x: lerp(Math.cos(P.azimuth) * ce, dir.x, 0.20),
            y: lerp(Math.sin(P.elevation), dir.y, 0.20),
            z: lerp(Math.sin(P.azimuth) * ce, dir.z, 0.20),
          };
          normalize3(cd);
          queue.push({
            id: nextId++, parent: 0, parentIndex: branches.length, depth: 1,
            x, y, z, dx: cd.x, dy: cd.y, dz: cd.z,
            r: childR,
            len: P.broken ? 1.05 : allometricLength(childR) * P.vigor,
            vigor: P.vigor, droop: P.droop, photo: P.photo,
            broken: !!P.broken, primary: P.key,
          });
          if (P.terminal) terminate = true;
        }
      }

      // --- tropism 1: gravity, κ = M/(E·I) ---------------------------------
      const distal = maxLen - s;
      const load = distal * (r * r * 8 + 0.0012); // distal wood + blossom mass
      const stiff = r * r * r * r;                // I ∝ r⁴
      let kappa = (GRAVITY_K * load) / (stiff + 1e-12);
      kappa = Math.min(kappa, turnCap) * L.droop * req.droop;
      // Only the component of gravity perpendicular to the shoot bends it - 
      // a vertical trunk carries its load in pure compression and does not sag.
      const dotUp = dir.y;
      gvec.x = -dir.x * dotUp;
      gvec.y = -1 - dir.y * dotUp;
      gvec.z = -dir.z * dotUp;
      const gl = Math.hypot(gvec.x, gvec.y, gvec.z);
      if (gl > 1e-5) {
        const g = (kappa * seg) / gl;
        dir.x += gvec.x * g; dir.y += gvec.y * g; dir.z += gvec.z * g;
        turned += kappa * seg;
      }

      // --- tropism 2: the crown envelope ------------------------------------
      // The one term in this file that knows what shape the crown is supposed
      // to be. `1 - q` is positive inside the envelope and negative outside, so
      // a single expression grows the crown out to its surface and then caps
      // it - which is what turns a fan of limbs into a dome instead of a spray.
      //
      // The two branches are NOT symmetric, and the asymmetry cost a whole
      // iteration to find. INSIDE the envelope the gradient is the WRONG
      // vector to follow: below the widest band it points down and out, so a
      // shoot obeying it digs for the skirt instead of climbing, and the first
      // build of this tropism produced a 6.6 m tree - every limb ran out
      // sideways at the height it left the bole and nothing reached 11 m.
      // A shaded shoot inside a crown does not grow toward the nearest surface,
      // it grows toward the LIGHT, and the light is out and UP. So inside, the
      // horizontal comes from the radial direction and the vertical is a climb
      // that fades to nothing at the top of the envelope.
      //
      // OUTSIDE, the gradient is exactly right and is followed in full: there
      // is nothing out there to hold a shoot up, so it comes back over the top
      // of the crown, and that arc is the silhouette the whole file is for.
      const envK = L.envelope;
      if (envK > 0) {
        const room = 1 - envelopeQ(x, y, z, _env);
        const es = CROWN_TROPISM * envK * seg;
        if (room > 0) {
          const k = (room > 1 ? 1 : room) * es;
          const hl = Math.hypot(x, z);
          // The radial push is what evacuates the crown's CORE. Applied at full
          // strength everywhere inside, it drives every shoot away from the axis,
          // and the measured result was zero twig sites within 2.2 m of the
          // trunk - a crown that reads as two lobes with a hole through the
          // middle from above, however many flowers the tree is given. A real
          // cherry's interior is thinner than its shell, not empty: shoots near
          // the axis are shaded from the side but wide open to the sky, so they
          // climb rather than flee. Fading the radial term over the inner third
          // of the crown lets them stay and carry spur wood; the vertical climb
          // below is untouched, so the core fills upward instead of outward.
          const envR = envelopeRadiusAt(y);
          const coreHold = envR > 0.05 ? clamp01(hl / (0.34 * envR)) : 1;
          if (hl > 0.05) {
            const kr = k * coreHold;
            dir.x += (x / hl) * kr;
            dir.z += (z / hl) * kr;
          }
          dir.y += k * CROWN_CLIMB * (1 - clamp01((y - CROWN_Y0) / CROWN_RY_UP));
        } else {
          // Quadratic, not linear, in the overshoot. A shoot 30 cm outside the
          // envelope is merely exposed; one two metres outside is a cantilever
          // with nothing above it and gets stripped by the first gale. A linear
          // response left the outline soft - measured, the 95th percentile of
          // the flower cloud sat 28 % beyond the envelope radius and the crown
          // had the ragged spray outline the reference does not have. Stiffening
          // it is what makes the silhouette a smooth convex arc.
          const over = -room;
          const k = -Math.min(2.2, over * (1 + 4.5 * over)) * es;
          dir.x += _env.x * k;
          dir.y += _env.y * k;
          dir.z += _env.z * k;
        }
      }

      // --- tropism 3: phototropism, thin shoots only ------------------------
      // Cut by roughly two thirds across every level. It used to run to 2.20 at
      // the twiglets and, gated only by thinness, it applied over the whole
      // length of every fine shoot in the tree - 0.87 rad of accumulated lift
      // on a single 58 cm tertiary. That is the fan of upward sprays visible on
      // the end of every limb in the renders. What survives here is the real
      // effect it stands for: the last few centimetres of a cherry shoot flick
      // up toward the light. The crown's shape is the envelope's job now.
      const thin = smoothstep(0.055, 0.006, r);
      dir.y += L.photo * req.photo * thin * thin * seg;

      // --- tropism 4: gnarl -------------------------------------------------
      perpendicular(dir, e1);
      e2.x = dir.y * e1.z - dir.z * e1.y;
      e2.y = dir.z * e1.x - dir.x * e1.z;
      e2.z = dir.x * e1.y - dir.y * e1.x;
      const f = L.gnarlFreq;
      const gA = nz.noise3D(x * f + noiseOffset, y * f, z * f);
      const gB = nz.noise3D(x * f, y * f + noiseOffset + 53.1, z * f);
      // Old wood gnarls hardest near the ground, where it has weathered longest.
      const gnarlScale = depth === 0 ? 1 - 0.55 * clamp01(y / TRUNK_LENGTH) : 1;
      const gAmt = L.gnarl * seg * gnarlScale * 3.2;
      dir.x += (e1.x * gA + e2.x * gB) * gAmt;
      // Gnarl is spatially COHERENT noise, so over the thirty segments of a
      // primary it is not a wobble that cancels - it integrates, and it was
      // contributing up to +0.38 rad of lift against gravity's -0.66. On
      // structural wood the vertical is decided by the beam equation and
      // nothing else; the gnarl stays as the lateral meander you actually see
      // in an old limb.
      dir.y += (e1.y * gA + e2.y * gB) * gAmt * (depth <= 1 ? 0.40 : 1.0);
      dir.z += (e1.z * gA + e2.z * gB) * gAmt;

      // --- tropism 5: the branch's own arc ----------------------------------
      // Structural wood arcs HORIZONTALLY. Letting the arc run in a random
      // plane perpendicular to the shoot made it the dominant vertical tropism
      // on a primary - 0.9 radians of accumulated tilt against gravity's 0.6 - 
      // so whether a limb swept low across the field or climbed to the top of
      // the tree came down to one coin flip per limb, and the crown came out
      // lopsided in a different way on every seed. Gravity owns the vertical
      // on a limb; the arc decides which way round the crown it curves.
      const sAmt = sweepAmt * seg * 3.0;
      if (depth <= 2) {
        // cross(up, dir), the horizontal binormal. Degenerate on a vertical
        // leader, which is exactly where a horizontal arc is meaningless, so
        // fall through to the free arc there.
        const hl = Math.hypot(dir.x, dir.z);
        if (hl > 0.12) {
          // Both components come from the PRE-UPDATE heading; reading back
          // dir.x on the second line would rotate by a different angle than
          // the first line assumed, and the arc would drift with segment size.
          const hx = dir.x, hz = dir.z;
          dir.x += (hz / hl) * sAmt;
          dir.z += (-hx / hl) * sAmt;
        } else {
          dir.x += e1.x * sAmt; dir.y += e1.y * sAmt; dir.z += e1.z * sAmt;
        }
      } else {
        const sa = Math.cos(sweepAz), sb = Math.sin(sweepAz);
        dir.x += (e1.x * sa + e2.x * sb) * sAmt;
        dir.y += (e1.y * sa + e2.y * sb) * sAmt;
        dir.z += (e1.z * sa + e2.z * sb) * sAmt;
      }

      // --- tropism 6: lean / lifetime of prevailing wind --------------------
      if (depth === 0) {
        dir.x += Math.cos(TRUNK_LEAN_AZIMUTH) * TRUNK_LEAN * seg;
        dir.z += Math.sin(TRUNK_LEAN_AZIMUTH) * TRUNK_LEAN * seg;
      } else {
        const wk = 0.010 * seg * thin;
        dir.x += PREVAILING_X * wk;
        dir.z += PREVAILING_Z * wk;
      }

      // --- tropism 7: crowding ---------------------------------------------
      if (collide) {
        const probe = Math.min(0.30 + 3.5 * r, 1.0);
        const c = grid.crowd(x, y, z, probe, req.id, req.parent, escape);
        if (c > CROWD_STEER) {
          const el = Math.hypot(escape.x, escape.y, escape.z) || 1;
          const push = Math.min((c - CROWD_STEER) * 0.05, 0.30) * seg * 4;
          // Same argument as the gnarl above, and it mattered more: with the
          // crown now eight times denser, the escape vector points away from
          // the mass of wood BELOW a limb, and it was pushing the primaries up
          // by +0.79 rad - more than gravity was pulling them down. A limb
          // routing around its neighbour turns sideways; it does not levitate.
          const vSuppress = depth <= 2 ? 0.30 : 1.0;
          dir.x += (escape.x / el) * push;
          dir.y += (escape.y / el) * push * vSuppress;
          dir.z += (escape.z / el) * push;
          if (++crowdedRun > 13) break;
        } else if (crowdedRun > 0) crowdedRun--;
      }

      normalize3(dir);

      x += dir.x * seg; y += dir.y * seg; z += dir.z * seg;
      s += seg;
      r *= taperStep;
      if (r < rStop) break;
      if (turned > turnTotal) break;
      if (depth >= 1 && y < browse && dir.y < 0) break;
      /**
       * Self-pruning at the crown's own outline. A shoot that ends up well
       * outside the light envelope is a cantilever with nothing above it and
       * nothing around it: it is wind-stripped, it is the first thing a hard
       * winter takes, and a cherry simply does not carry one.
       *
       * It is here for a visible reason. The blossom layer flowers on a Beer's
       * law depth measured from this same envelope, so wood outside it carries
       * almost nothing - and the tree was growing two 7 m primaries straight
       * through the outline and out the other side, which rendered as bare dark
       * whips sticking a metre and a half clear of the flower mass with a small
       * tuft on the end. Nothing in the reference photographs looks like that.
       * The tropism above bends a shoot back; this ends the ones it cannot.
       */
      // The GRACE is why this is not simply `depth >= 1`. A primary leaves the
      // bole at 1.75 m, which is below the envelope's own floor at 1.90 - it is
      // on its way up into the crown, not escaping from it - and testing it
      // from its first segment killed the lowest limb outright. The tree came
      // out with a ten-limb fan and a hole where its signature low limb should
      // have been, and nothing said why: a shoot that dies at three points is
      // simply absent from every count in the file.
      //
      // A metre and a half of grace covers the climb and nothing else: measured,
      // every primary is inside q = 0.91 by then. Exempting the limbs entirely,
      // which is what this did first, left six of them running a metre and a
      // half clear of the flower mass as bare dark whips - by far the most
      // conspicuous defect in the render, and the reason to look at the wood as
      // well as the blossom when judging the silhouette.
      // depth >= 1 is not optional either: the bole runs from y = -0.55, which
      // is q = 1.85, so testing it would prune the trunk at the ground.
      if (depth === 1) {
        if (s > 1.5 && Math.hypot(x, z) > envelopeRadiusAt(y) * PRIMARY_REACH) break;
      } else if (depth >= 2 && envelopeQ(x, y, z, null) > CROWN_PRUNE) break;
      // Only a shoot that walked every step of its own length has an apical bud
      // left to extend. One killed by crowding, by the browse line or by running
      // out of wood has genuinely finished, and giving it a continuation would
      // push a limb straight back through whatever it just escaped.
      if (i === steps - 1) reachedTip = true;
    }

    if (px.length < 3) return null;

    // --- apical continuation -------------------------------------------------
    // See CONTINUE_MIN_RADIUS. This is the same limb carrying on into thinner
    // wood, not a lateral, so it keeps the heading it ended on and takes no
    // divergence angle. It is what stops every leader in the tree from ending
    // as a flat disc of full radius hanging in the sky.
    if (
      reachedTip && !terminate && !req.broken && depth >= 1 &&
      r > CONTINUE_MIN_RADIUS && (req.contChain || 0) < 3 &&
      queue.length < BRANCH_BUDGET
    ) {
      const contDepth = Math.min(depth + 1, LEVELS.length - 1);
      const contR = Math.min(r * CONTINUE_TAPER, LEVELS[contDepth].rMax);
      if (contR > LEVELS[contDepth].rMin * 1.15) {
        const tip = px.length - 1;
        queue.push({
          id: nextId++, parent: req.id, parentIndex: branches.length,
          depth: contDepth,
          x: px[tip], y: py[tip], z: pz[tip],
          dx: dir.x, dy: dir.y, dz: dir.z,
          r: contR,
          len: allometricLength(contR) * (0.86 + 0.34 * brng()),
          vigor: req.vigor,
          // Droop follows the wood, not the generation: a 2 m continuation off a
          // primary sags like the bough it is, a 15 cm one off a twig does not.
          droop: 0.80 + 0.55 * clamp01(allometricLength(contR) / 3.2),
          photo: req.photo, broken: false, primary: req.primary,
          contChain: (req.contChain || 0) + 1,
        });
      }
    }

    // A snapped limb heals by throwing a crown of vertical epicormic sprouts
    // around the wound - the most recognisable feature of a veteran cherry.
    if (req.broken) {
      const tip = px.length - 1;
      queueEpicormics(px[tip], py[tip], pz[tip], dir, pr[tip], req.id, branches.length);
    }

    const n = px.length;
    return {
      id: req.id,
      parent: req.parent,
      indexOfParent: req.parentIndex,
      depth, n,
      x: Float32Array.from(px), y: Float32Array.from(py), z: Float32Array.from(pz),
      r: Float32Array.from(pr), s: Float32Array.from(ps),
      swA: new Float32Array(n * 4), swB: new Float32Array(n * 4),
      ao: new Float32Array(n),
      len: ps[n - 1],
      phase: brng(),
      soft: softness(ps[n - 1], req.r),
      baseRadius: req.r,
      /** Apical continuation of its parent rather than a lateral off its side.
       *  The mesh reads this to suppress the fork collar: a bud scar is a faint
       *  ring, not the 1.55x swelling a real lateral drives into its parent. */
      continuation: !!req.contChain,
      broken: !!req.broken,
      primary: req.primary,
      attachments: [],
      phases: [0, 0, 0, 0],
      rank: 0,
    };

    /**
     * Picks the least-crowded of three candidate bud directions 120° apart.
     * This is the cheap half of space colonisation, and it is what stops the
     * canopy from growing through itself.
     */
    function chooseShootDir(ox, oy, oz, parentDir, az, div, childR) {
      perpendicular(parentDir, e1);
      e2.x = parentDir.y * e1.z - parentDir.z * e1.y;
      e2.y = parentDir.z * e1.x - parentDir.x * e1.z;
      e2.z = parentDir.x * e1.y - parentDir.y * e1.x;

      let best = null, bestScore = Infinity;
      const cd = Math.cos(div), sd = Math.sin(div);
      const reach = 0.35 + 5 * childR;
      for (let k = 0; k < 3; k++) {
        const a = az + k * 2.0944;
        const ca = Math.cos(a), sa2 = Math.sin(a);
        const vx = parentDir.x * cd + (e1.x * ca + e2.x * sa2) * sd;
        const vy = parentDir.y * cd + (e1.y * ca + e2.y * sa2) * sd;
        const vz = parentDir.z * cd + (e1.z * ca + e2.z * sa2) * sd;
        const tx = ox + vx * reach, ty = oy + vy * reach, tz = oz + vz * reach;
        if (ty < browse) continue;
        // The envelope term is evaluated at EVERY depth, including the ones too
        // deep to pay for a collision query. It is four multiplies and a square
        // root, and it is what decides whether the outer half of the crown is a
        // dome or a hedgehog - leaving the deep levels to take the raw
        // phyllotactic direction, as this used to, is why the twig cloud had no
        // outline of its own and simply inherited whatever its limb was doing.
        let score = envelopeScore(tx, ty, tz);
        // Buds still prefer the light from above, but only just: at 0.85 this
        // term outweighed everything else in the function and every bud in the
        // tree aimed at the sky. See ENV_LIGHT_GAIN.
        score -= vy * 0.16;
        if (collide) {
          score += grid.crowd(tx, ty, tz, 0.35 + 4 * childR, req.id, req.parent, escape);
        }
        if (score < bestScore) { bestScore = score; best = { x: vx, y: vy, z: vz }; }
      }
      // The abort threshold is measured against the crowding term, which is the
      // only one of the three that can be large and positive on a viable bud.
      // The envelope terms span -1.05 to +9, so the comparison is shifted by
      // the light gain to keep CROWD_ABORT meaning what it says.
      if (!best || bestScore + ENV_LIGHT_GAIN > CROWD_ABORT) return null;
      return best;
    }
  }

  /** A ring of vigorous vertical sprouts around a broken limb's wound. */
  function queueEpicormics(x, y, z, limbDir, limbR, parentId, parentIndex) {
    const count = 6;
    for (let k = 0; k < count; k++) {
      const a = (k / count) * TAU + rng() * 0.7;
      const tilt = 0.35 + rng() * 0.4;
      const d = {
        x: Math.cos(a) * Math.sin(tilt) + limbDir.x * 0.15,
        y: Math.cos(tilt) + 0.55,
        z: Math.sin(a) * Math.sin(tilt) + limbDir.z * 0.15,
      };
      normalize3(d);
      const r = Math.max(limbR * (0.34 + rng() * 0.22), LEVELS[2].rMin * 2);
      queue.push({
        id: nextId++, parent: parentId, parentIndex, depth: 2,
        x: x - limbDir.x * 0.06, y: y - limbDir.y * 0.06, z: z - limbDir.z * 0.06,
        dx: d.x, dy: d.y, dz: d.z,
        r, len: allometricLength(r) * (0.9 + rng() * 0.5),
        vigor: 1.2, droop: 0.55, photo: 1.5,
        broken: false, primary: 'sprout',
      });
    }
  }

  // -- fork attachments -----------------------------------------------------
  // A child's tube begins on its parent's AXIS, so the nearest parent point is
  // exactly the point it was budded from.
  for (const b of branches) {
    if (b.indexOfParent < 0 || b.indexOfParent >= branches.length) continue;
    const p = branches[b.indexOfParent];
    // An apical continuation carries the limb on past its parent's last point,
    // so it needs no fork attachment: it is not a lateral and drives no shoulder
    // into its parent. Marked here rather than when the continuation is queued,
    // because a queued shoot can still fail to grow, and then the parent really
    // is the end of the limb.
    //
    // `hasApical` is SKELETON truth and is deliberately NOT what decides the end
    // cap. Whether a limb needs a terminal bud is a question about the drawn
    // mesh, and every tier below ULTRA drops some continuations - see
    // `apicalDrawn` in buildBranchGeometry, and do not reintroduce this flag
    // there.
    if (b.continuation) { p.hasApical = true; continue; }
    const j = nearestPointIndex(p, b.x[0], b.y[0], b.z[0]);
    const dx = b.x[1] - b.x[0], dy = b.y[1] - b.y[0], dz = b.z[1] - b.z[0];
    const l = Math.hypot(dx, dy, dz) || 1;
    p.attachments.push({ s: p.s[j], r: b.baseRadius, dx: dx / l, dy: dy / l, dz: dz / l });
  }

  // The budget is a safety rail against a pathological seed, and a rail that
  // fires silently is a trap - when it bites it simply stops budding, so the
  // crown comes out quietly thin on whichever side the breadth-first walk
  // reached last and nothing says why. It sat at 91 % of the old cap once
  // already. One comparison, at build time.
  if (queue.length >= BRANCH_BUDGET) {
    console.warn(
      `[Sakura] BRANCH_BUDGET (${BRANCH_BUDGET}) reached. The crown is thinned ` +
      'on one side and the silhouette is not the authored one. Raise the ' +
      'budget, or reduce LEVELS[].nodes / maxChildren.'
    );
  }

  computeSway(branches);
  computeSkeletonAO(branches, aoGrid);

  // -- draw order: thickest first, so any prefix is a connected sub-tree -----
  const order = branches.map((b, i) => i).sort((a, c) => branches[c].baseRadius - branches[a].baseRadius);
  for (let i = 0; i < order.length; i++) branches[order[i]].rank = i;

  const metrics = measureCanopy(branches);

  const roots = [];
  for (let i = 0; i < 6; i++) {
    roots.push({ az: (i / 6) * TAU + rng() * 0.55, amp: 0.22 + rng() * 0.30 });
  }

  return {
    seed, branches, order, roots,
    height: metrics.height,
    canopyCenter: metrics.center,
    canopyRadius: metrics.radius,
    canopyRadiusY: metrics.radiusY,
    trunkRadius: TRUNK_RADIUS,
    bury: TRUNK_BURY,
    levels: LEVELS.length,
    /**
     * The crown envelope the wood was grown against, published so blossoms.js
     * fills the SAME shape rather than re-deriving one from percentiles of the
     * twig cloud. Deriving it twice is how the flower mass and the wood ended
     * up disagreeing about where the crown was: `canopyRadius` is a 90th
     * percentile and moves with every seed and every growth change, so the
     * blossom shell's equator sat inside the crown's own outline at the sides
     * and outside it at the top.
     */
    envelope: {
      r: CROWN_R, y0: CROWN_Y0, ryUp: CROWN_RY_UP, ryDown: CROWN_RY_DOWN,
    },
  };
}


function nearestPointIndex(p, x, y, z) {
  let best = 0, bestD = Infinity;
  for (let i = 0; i < p.n; i++) {
    const dx = p.x[i] - x, dy = p.y[i] - y, dz = p.z[i] - z;
    const d = dx * dx + dy * dy + dz * dz;
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

/**
 * Per-point sway weights for the four oscillator levels.
 *
 *   w_L(s) = w_L(parent at the attach point) + shape(s/len)·soft·CARRY^(depth-L)
 *
 * The own term is exactly 0 at s = 0, so a child starts at precisely its
 * parent's value: joints are C0-continuous under ANY wind, and the mesh can
 * never tear at a fork. Deeper wood keeps contributing to shallower levels at
 * a decaying weight, which is why a twig on a swaying limb travels with the
 * limb AND flutters on its own.
 *
 * Branches are visited in creation order, which is breadth-first, so a parent
 * is always resolved before its children.
 */
function computeSway(branches) {
  const maxW = [1e-6, 1e-6, 1e-6, 1e-6];
  for (const b of branches) {
    const p = b.indexOfParent >= 0 && b.indexOfParent < branches.length ? branches[b.indexOfParent] : null;
    b.phases = p ? p.phases.slice() : [0, 0, 0, 0];
    if (b.depth >= 1 && b.depth <= SWAY_LEVELS) b.phases[b.depth - 1] = b.phase;

    const inherit = [0, 0, 0, 0];
    if (p) {
      const j = nearestPointIndex(p, b.x[0], b.y[0], b.z[0]);
      inherit[0] = p.swA[j * 4];
      inherit[1] = p.swA[j * 4 + 2];
      inherit[2] = p.swB[j * 4];
      inherit[3] = p.swB[j * 4 + 2];
    }

    const invLen = 1 / (b.len || 1);
    for (let i = 0; i < b.n; i++) {
      const shape = complianceShape(b.s[i] * invLen);
      for (let Lv = 1; Lv <= SWAY_LEVELS; Lv++) {
        let w = inherit[Lv - 1];
        if (b.depth >= Lv) w += shape * b.soft * Math.pow(SWAY_CARRY, b.depth - Lv);
        const arr = Lv <= 2 ? b.swA : b.swB;
        const off = i * 4 + ((Lv - 1) % 2) * 2;
        arr[off] = w;
        arr[off + 1] = b.phases[Lv - 1];
        if (w > maxW[Lv - 1]) maxW[Lv - 1] = w;
      }
    }
  }
  // Normalise so the shader's amplitude uniforms are honest metres of travel.
  for (const b of branches) {
    for (let i = 0; i < b.n; i++) {
      b.swA[i * 4] /= maxW[0];
      b.swA[i * 4 + 2] /= maxW[1];
      b.swB[i * 4] /= maxW[2];
      b.swB[i * 4 + 2] /= maxW[3];
    }
  }
}

/**
 * Vertex AO baked from wood density. One-time, and worth far more than it
 * costs: it darkens the crotches of forks and the canopy interior, which is
 * exactly where screen-space AO is weakest on thin geometry.
 *
 * The crown now carries thousands of twigs rather than hundreds, and this pass
 * is O(points x neighbours), so thin wood is sampled at three points along the
 * branch and interpolated. A 40 cm twiglet occupies one cell of a 0.5 m grid - 
 * the field genuinely is that smooth at its scale, and the saving is roughly
 * three quarters of the pass.
 */
function computeSkeletonAO(branches, aoGrid) {
  const sample = (b, i) => {
    const occ = aoGrid.density(b.x[i], b.y[i], b.z[i], 0.5, b.id) * 3; // ×3: subsampled
    let ao = 1 / (1 + 0.045 * occ);
    ao *= lerp(0.84, 1.0, clamp01(b.y[i] / 6)); // nothing deep in the bole sees sky
    return clamp(ao, 0.20, 1);
  };
  for (const b of branches) {
    if (b.depth <= COLLIDE_MAX_DEPTH || b.n < 6) {
      for (let i = 0; i < b.n; i++) b.ao[i] = sample(b, i);
      continue;
    }
    const mid = b.n >> 1, last = b.n - 1;
    const a0 = sample(b, 0), a1 = sample(b, mid), a2 = sample(b, last);
    for (let i = 0; i < b.n; i++) {
      b.ao[i] = i <= mid
        ? lerp(a0, a1, i / mid)
        : lerp(a1, a2, (i - mid) / (last - mid));
    }
  }
}

/**
 * Canopy centroid and radii from the outer twig cloud - for petals, fog, the
 * ground-litter ring and the god-ray occluder radius.
 *
 * The 90th percentile is deliberate: the mean would be dragged in by the dense
 * inner shoots and the maximum out by the one twig that shot away from the
 * crown, and world/scatter.js and world/terrain.js both size real features off
 * this number. It is measured about the centroid in the horizontal plane only,
 * because that is what "how far does the crown reach" means to every consumer.
 */
function measureCanopy(branches) {
  let sx = 0, sy = 0, sz = 0, count = 0, height = 0;
  for (const b of branches) {
    for (let i = 0; i < b.n; i++) if (b.y[i] > height) height = b.y[i];
    if (b.depth < 3) continue;
    for (let i = 0; i < b.n; i++) { sx += b.x[i]; sy += b.y[i]; sz += b.z[i]; count++; }
  }
  if (!count) return { center: new THREE.Vector3(0, 6, 0), radius: 6, radiusY: 4, height: height || 8 };
  const center = new THREE.Vector3(sx / count, sy / count, sz / count);
  const dr = [], dv = [];
  for (const b of branches) {
    if (b.depth < 3) continue;
    for (let i = 0; i < b.n; i++) {
      dr.push(Math.hypot(b.x[i] - center.x, b.z[i] - center.z));
      dv.push(Math.abs(b.y[i] - center.y));
    }
  }
  dr.sort((a, c) => a - c);
  dv.sort((a, c) => a - c);
  /**
   * Floored at the authored envelope, and it has to be.
   *
   * `canopyRadius` is published to world/grass.js (which thins the sward under
   * the tree), world/terrain.js (the fallen-petal litter ring), world/scatter.js,
   * weather/precipitation.js (rain shelter), weather/atmosfx.js and the HUD's
   * framing - all of which mean by it "how far does the crown reach". A 90th
   * percentile of the twig cloud stopped answering that question once the crown
   * was filled through its volume rather than skinned: the cloud's mass moved
   * inward, and the percentile fell from 6.5 m to 4.6 m on a crown that had
   * simultaneously grown WIDER. Every one of those consumers would have drawn
   * its feature two metres inside the drip line. The envelope is the crown's
   * outline by construction, so it is the floor; the percentile can still
   * exceed it on a seed that sprawls.
   */
  return {
    center,
    radius: Math.max(dr[Math.floor(dr.length * 0.90)] || 6, CROWN_R),
    radiusY: Math.max(
      dv[Math.floor(dv.length * 0.90)] || 4,
      (CROWN_RY_UP + CROWN_RY_DOWN) * 0.45
    ),
    height,
  };
}

// ===========================================================================
// Mesh generation
// ===========================================================================

const _perp = { x: 0, y: 0, z: 0 };
const _dTheta = { x: 0, y: 0, z: 0 };
const _dS = { x: 0, y: 0, z: 0 };
const _tan = { x: 0, y: 0, z: 0 };
const _prevTan = { x: 0, y: 0, z: 0 };

/**
 * Terminal bud radius profile, precomputed once per branch.
 *
 * Wood thinner than this at the tip is left alone: a 1.5 mm twig end is under a
 * pixel from anywhere you can stand, so a bud on it would be pure cost, and the
 * apex cap that comes with it would be too. Everything thicker DOES read, and
 * an uncapped swept tube is an open hole showing the inside of the branch - 
 * which, on a primary ending at 7.6 cm radius, is the flat sawn-off disc
 * visible on every limb in the renders.
 */
const BUD_MIN_RADIUS = 0.0015;

/**
 * Builds the tip descriptor for a branch, or null if the tip is too thin to be
 * worth shaping. `s0` is where the bud profile starts, `inv` normalises arc
 * length across it, `rT` is the radius the taper actually arrived at.
 *
 * @param {boolean} hasDrawnApical whether a continuation of this branch is in
 *   THIS TIER'S drawn set. It must be the tier's answer, not the skeleton's:
 *   see the note at the call site.
 */
function tipProfile(b, hasDrawnApical) {
  // A limb that carries on past this point has no tip here.
  // depth 0: the trunk always ends by forking into the terminal primary, and
  // that fork sits exactly on its last point - necking the bole into a bud
  // underneath the crown limb's collar would pinch the top of the trunk.
  if (hasDrawnApical || b.depth === 0) return null;
  const rT = b.r[b.n - 1];
  if (!(rT > BUD_MIN_RADIUS)) return null;
  const len = b.len || 1;
  // Long enough to read as a neck-then-bud, short enough that it never eats
  // into the branch proper: 9 tip-radii is about 2 cm on a twig and 70 cm on a
  // primary, which is the right scale for both.
  const budLen = Math.min(0.42 * len, 9 * rT);
  if (!(budLen > 1e-4)) return null;
  return { s0: len - budLen, inv: 1 / budLen, rT };
}

/**
 * Surface radius of the swept tube at ring `i` in outward direction `o`.
 * Includes the fork shoulder, the root flare and the terminal bud, so one
 * function feeds both the vertex position and its analytic normal - they can
 * never disagree.
 */
function surfaceRadius(b, i, ox, oy, oz, roots, tip) {
  let r = b.r[i];
  const atts = b.attachments;
  for (let k = 0; k < atts.length; k++) {
    const a = atts[k];
    const ds = b.s[i] - a.s;
    const sigma = (b.r[i] + a.r) * 1.15 + 0.015;
    if (ds < -sigma * 2.2 || ds > sigma * 2.2) continue;
    const d = ox * a.dx + oy * a.dy + oz * a.dz;
    if (d <= 0) continue;
    r += 0.70 * a.r * Math.exp(-(ds * ds) / (sigma * sigma)) * d * d * d;
  }
  if (b.depth === 0 && roots) {
    const y = b.y[i];
    if (y < FLARE_HEIGHT) {
      const k = Math.pow(1 - clamp01(y / FLARE_HEIGHT), 2.1);
      let f = 1 + FLARE_AMOUNT * k;
      const az = Math.atan2(oz, ox);
      for (let j = 0; j < roots.length; j++) {
        const c = Math.cos(az - roots[j].az);
        if (c > 0) f += k * roots[j].amp * c * c * c * c * c;
      }
      r *= f;
    }
  }
  /**
   * Terminal bud. Real terminal growth does two things a pure taper does not:
   * it NECKS behind the bud, and then the bud itself is fatter than the wood
   * carrying it. Both halves matter - a cone to nothing rasterises as a
   * flickering sliver and a plain stop rasterises as a flat disc, and the
   * silhouette that reads as a shoot rather than as a cut stick is the
   * narrow-then-round one.
   *
   * The neck is a fraction of the local radius (so a fork shoulder near the tip
   * is narrowed with everything else), the bud is an absolute swelling in tip
   * radii (so it is a bud, not a scaled copy of whatever the taper left).
   */
  if (tip) {
    const u = (b.s[i] - tip.s0) * tip.inv;
    if (u > 0) {
      const t = u > 1 ? 1 : u;
      r *= 1 - 0.30 * smoothstep(0, 0.55, t);
      r += 0.55 * tip.rT * smoothstep(0.35, 0.97, t);
    }
  }
  return r;
}

/** Centred-difference tangent of a branch polyline at point i. */
function tangentAt(b, i, out) {
  const a = Math.max(0, i - 1);
  const c = Math.min(b.n - 1, i + 1);
  out.x = b.x[c] - b.x[a];
  out.y = b.y[c] - b.y[a];
  out.z = b.z[c] - b.z[a];
  return normalize3(out);
}

/**
 * Builds every stick of wood in the tree as ONE geometry - one draw call.
 *
 * @param {object} skel  from generateSkeleton()
 * @param {string} tier  key into LOD_TIERS
 */
export function buildBranchGeometry(skel, tier = 'high') {
  const T = LOD_TIERS[tier] || LOD_TIERS.high;
  const branches = skel.branches;
  const nb = branches.length;

  // --- which branches this tier draws --------------------------------------
  const use = new Uint8Array(nb);
  let accepted = 0;
  for (let k = 0; k < skel.order.length && accepted < T.maxBranches; k++) {
    const bi = skel.order[k];
    const b = branches[bi];
    if (b.depth > T.maxDepth) continue;
    if (b.indexOfParent >= 0 && !use[b.indexOfParent]) continue;
    use[bi] = 1;
    accepted++;
  }

  // --- ring selection --------------------------------------------------------
  const ringIdx = new Array(nb);
  const ringSeg = new Uint8Array(nb);
  const ringTip = new Array(nb);
  for (let bi = 0; bi < nb; bi++) {
    if (!use[bi]) continue;
    const b = branches[bi];
    // Detail follows thickness, not hierarchy depth - see RADIUS_CLASS.
    const cls = radiusClass(b.baseRadius);
    const stride = T.stride[Math.min(cls, T.stride.length - 1)];
    const seg = T.radial[Math.min(cls, T.radial.length - 1)];
    const idx = [];
    for (let i = 0; i < b.n; i += stride) idx.push(i);
    if (idx[idx.length - 1] !== b.n - 1) idx.push(b.n - 1);
    if (idx.length < 2) { use[bi] = 0; continue; }
    ringIdx[bi] = idx;
    ringSeg[bi] = seg;
  }

  /**
   * Which branches still have an apical continuation ONCE THIS TIER HAS THINNED
   * THE TREE. `b.hasApical` is a property of the skeleton, and `maxBranches`
   * drops the thinnest wood - which is overwhelmingly the continuations, since
   * a continuation is by construction thinner than the limb it extends. Reading
   * the skeleton flag here therefore withheld the end cap from limbs whose
   * continuation was NOT drawn, leaving a swept tube open at the tip: a hole
   * looking straight down the inside of the branch, which under FrontSide
   * culling rasterises as exactly the flat sawn-off disc this whole mechanism
   * exists to remove.
   *
   * It was not a rare corner either. Measured on the shipping seed: 343 open
   * tubes at LOW - half of the 700 branches that tier draws - 277 at MEDIUM and
   * 90 at HIGH, the largest 6.6 mm in radius, i.e. 13 mm across and squarely
   * visible. ULTRA draws every shoot, so it was the one tier that looked right,
   * and it is the one tier this laptop GPU will never run.
   *
   * Computed after ring selection because that pass can still demote a branch.
   */
  const apicalDrawn = new Uint8Array(nb);
  for (let bi = 0; bi < nb; bi++) {
    if (!use[bi]) continue;
    const b = branches[bi];
    const p = b.indexOfParent;
    if (b.continuation && p >= 0 && p < nb && use[p]) apicalDrawn[p] = 1;
  }

  // --- vertex / index budget -------------------------------------------------
  let vertCount = 0, indexCount = 0;
  for (let bi = 0; bi < nb; bi++) {
    if (!use[bi]) continue;
    const idx = ringIdx[bi];
    const seg = ringSeg[bi];
    // One apex vertex and one triangle fan closes the tube. Only wood thick
    // enough for the hole to be visible pays for it - see BUD_MIN_RADIUS.
    const tip = tipProfile(branches[bi], apicalDrawn[bi] === 1);
    ringTip[bi] = tip;
    vertCount += idx.length * (seg + 1) + (tip ? 1 : 0);
    indexCount += (idx.length - 1) * seg * 6 + (tip ? seg * 3 : 0);
  }

  const position = new Float32Array(vertCount * 3);
  const normal = new Float32Array(vertCount * 3);
  const tangent = new Float32Array(vertCount * 4);
  const uv = new Float32Array(vertCount * 2);
  const swayA = new Float32Array(vertCount * 4);
  const swayB = new Float32Array(vertCount * 4);
  const bark = new Float32Array(vertCount * 2);
  const index = vertCount > 65535 ? new Uint32Array(indexCount) : new Uint16Array(indexCount);

  let v = 0, ii = 0;

  for (let bi = 0; bi < nb; bi++) {
    if (!use[bi]) continue;
    const b = branches[bi];
    const idx = ringIdx[bi];
    const rings = idx.length;
    const seg = ringSeg[bi];
    const ringVerts = seg + 1;
    const base = v;
    const tip = ringTip[bi];
    /** Sum of the last ring's surface radii, for the cap apex stand-off. */
    let capR = 0;

    // Texture layout: U repeats an integer number of tiles around the branch so
    // the seam matches exactly, and V uses the SAME metric so texels stay
    // square. This is what stops bark smearing along a taper or on a thin twig.
    const midR = b.r[Math.floor(b.n * 0.35)];
    const circumference = TAU * Math.max(midR, 0.004);
    const repeatsU = Math.max(1, Math.round(circumference / BARK_TILE));
    const vScale = circumference / repeatsU;

    // --- parallel-transport frame ------------------------------------------
    tangentAt(b, idx[0], _tan);
    perpendicular(_tan, _perp);
    let ax = _perp.x, ay = _perp.y, az = _perp.z;
    _prevTan.x = _tan.x; _prevTan.y = _tan.y; _prevTan.z = _tan.z;

    const frameX = new Float32Array(rings * 3);
    const frameY = new Float32Array(rings * 3);
    for (let ri = 0; ri < rings; ri++) {
      tangentAt(b, idx[ri], _tan);
      // Rotate the frame by the minimal rotation carrying prevTan onto tan.
      // More stable than re-deriving a frame per ring (which twists) and than
      // Frenet frames (which flip sign at every inflection point).
      const cx = _prevTan.y * _tan.z - _prevTan.z * _tan.y;
      const cy = _prevTan.z * _tan.x - _prevTan.x * _tan.z;
      const cz = _prevTan.x * _tan.y - _prevTan.y * _tan.x;
      const sinA = Math.hypot(cx, cy, cz);
      if (sinA > 1e-6) {
        const cosA = _prevTan.x * _tan.x + _prevTan.y * _tan.y + _prevTan.z * _tan.z;
        const ang = Math.atan2(sinA, cosA);
        const kx = cx / sinA, ky = cy / sinA, kz = cz / sinA;
        const c = Math.cos(ang), s = Math.sin(ang), oc = 1 - c;
        const kd = kx * ax + ky * ay + kz * az;
        const rx = ax * c + (ky * az - kz * ay) * s + kx * kd * oc;
        const ry = ay * c + (kz * ax - kx * az) * s + ky * kd * oc;
        const rz = az * c + (kx * ay - ky * ax) * s + kz * kd * oc;
        ax = rx; ay = ry; az = rz;
      }
      const d = ax * _tan.x + ay * _tan.y + az * _tan.z;
      ax -= _tan.x * d; ay -= _tan.y * d; az -= _tan.z * d;
      const al = Math.hypot(ax, ay, az) || 1;
      ax /= al; ay /= al; az /= al;
      frameX[ri * 3] = ax; frameX[ri * 3 + 1] = ay; frameX[ri * 3 + 2] = az;
      // (X, Y, tangent) right-handed => cross(N, T) runs along +V, so the
      // tangent handedness three expects is +1 everywhere.
      frameY[ri * 3] = _tan.y * az - _tan.z * ay;
      frameY[ri * 3 + 1] = _tan.z * ax - _tan.x * az;
      frameY[ri * 3 + 2] = _tan.x * ay - _tan.y * ax;
      _prevTan.x = _tan.x; _prevTan.y = _tan.y; _prevTan.z = _tan.z;
    }

    for (let ri = 0; ri < rings; ri++) {
      const i = idx[ri];
      const cxp = b.x[i], cyp = b.y[i], czp = b.z[i];
      const ux = frameX[ri * 3], uy = frameX[ri * 3 + 1], uz = frameX[ri * 3 + 2];
      const wx = frameY[ri * 3], wy = frameY[ri * 3 + 1], wz = frameY[ri * 3 + 2];

      // Collar: a child's tube starts on the parent's axis and swells, so the
      // joint is a fillet rather than two cylinders intersecting.
      //
      // An apical continuation is NOT a fork. It leaves its parent's last point
      // along its parent's own heading at 93 % of its radius, so the 1.55x
      // lateral collar would put a knuckle a third fatter than the limb in the
      // middle of a smooth taper - the one place the eye reads a limb as
      // continuous wood. A bud scar is a faint ring, and that is all it gets.
      let collar = 1;
      if (b.depth > 0) {
        const w = b.baseRadius * 1.6 + 0.015;
        const amt = b.continuation ? 0.09 : (b.depth <= 2 ? 0.55 : 0.40);
        collar = 1 + amt * Math.exp(-(b.s[i] * b.s[i]) / (w * w));
      }

      const vCoord = b.s[i] / vScale;
      const nRing = idx[clamp(ri + (ri === 0 ? 1 : -1), 0, rings - 1)];
      const dSarc = (b.s[nRing] - b.s[i]) || 1e-3;
      const lastRing = ri === rings - 1;

      for (let k = 0; k <= seg; k++) {
        const th = (k / seg) * TAU;
        const ct = Math.cos(th), st = Math.sin(th);
        const ox = ux * ct + wx * st;
        const oy = uy * ct + wy * st;
        const oz = uz * ct + wz * st;

        const r0 = surfaceRadius(b, i, ox, oy, oz, skel.roots, tip) * collar;
        const pxv = cxp + ox * r0;
        const pyv = cyp + oy * r0;
        const pzv = czp + oz * r0;

        // --- analytic surface normal: dP/dθ × dP/ds ------------------------
        const dth = 0.03;
        const c2 = Math.cos(th + dth), s2 = Math.sin(th + dth);
        const oxb = ux * c2 + wx * s2, oyb = uy * c2 + wy * s2, ozb = uz * c2 + wz * s2;
        const rb = surfaceRadius(b, i, oxb, oyb, ozb, skel.roots, tip) * collar;
        _dTheta.x = (oxb * rb - ox * r0) / dth;
        _dTheta.y = (oyb * rb - oy * r0) / dth;
        _dTheta.z = (ozb * rb - oz * r0) / dth;

        // Carries the bud's own slope into the shading normal, so the terminal
        // swelling reads as a rounded form rather than as a painted-on ring.
        const rn = surfaceRadius(b, nRing, ox, oy, oz, skel.roots, tip) * collar;
        _dS.x = (b.x[nRing] + ox * rn - pxv) / dSarc;
        _dS.y = (b.y[nRing] + oy * rn - pyv) / dSarc;
        _dS.z = (b.z[nRing] + oz * rn - pzv) / dSarc;

        let nx = _dTheta.y * _dS.z - _dTheta.z * _dS.y;
        let ny = _dTheta.z * _dS.x - _dTheta.x * _dS.z;
        let nzz = _dTheta.x * _dS.y - _dTheta.y * _dS.x;
        let nl = Math.hypot(nx, ny, nzz);
        if (nl < 1e-9) { nx = ox; ny = oy; nzz = oz; nl = 1; }
        nx /= nl; ny /= nl; nzz /= nl;
        if (nx * ox + ny * oy + nzz * oz < 0) { nx = -nx; ny = -ny; nzz = -nzz; }

        let tx = _dTheta.x, ty = _dTheta.y, tz = _dTheta.z;
        const tl = Math.hypot(tx, ty, tz) || 1;
        tx /= tl; ty /= tl; tz /= tl;

        const o3 = v * 3, o2 = v * 2, o4 = v * 4;
        position[o3] = pxv; position[o3 + 1] = pyv; position[o3 + 2] = pzv;
        normal[o3] = nx; normal[o3 + 1] = ny; normal[o3 + 2] = nzz;
        tangent[o4] = tx; tangent[o4 + 1] = ty; tangent[o4 + 2] = tz; tangent[o4 + 3] = 1;
        uv[o2] = (k / seg) * repeatsU;
        uv[o2 + 1] = vCoord;

        swayA[o4] = b.swA[i * 4]; swayA[o4 + 1] = b.swA[i * 4 + 1];
        swayA[o4 + 2] = b.swA[i * 4 + 2]; swayA[o4 + 3] = b.swA[i * 4 + 3];
        swayB[o4] = b.swB[i * 4]; swayB[o4 + 1] = b.swB[i * 4 + 1];
        swayB[o4 + 2] = b.swB[i * 4 + 2]; swayB[o4 + 3] = b.swB[i * 4 + 3];

        // Bark age: 0 = fissured old bole, 1 = smooth glossy first-year shoot.
        bark[o2] = smoothstep(0.055, 0.007, b.r[i]);
        // Upward faces see sky; undersides sit in their own shadow.
        bark[o2 + 1] = clamp01(b.ao[i] * (0.70 + 0.30 * (ny * 0.5 + 0.5)));
        // The apex sits on the axis, so its height above the last ring has to
        // come from that ring's actual radius - including the bud swelling and
        // any fork shoulder still in range. Summed over the seam-duplicated
        // vertex too; one extra sample out of seg+1 shifts the mean by under a
        // percent and is not worth a branch inside the hot loop.
        if (lastRing) capR += r0;
        v++;
      }
    }

    // --- terminal bud cap ----------------------------------------------------
    // A swept tube is open at both ends. The base end is buried in the parent's
    // wood by the collar, but the tip end is a hole looking straight down the
    // inside of the branch - with FrontSide culling that reads as a flat disc
    // punched out of the silhouette, which is exactly the "sawn-off" end.
    //
    // One apex vertex and one fan closes it. The apex stands off along the
    // tangent by slightly less than the tip radius, so the end is a blunt
    // 48-degree dome rather than a spike: a bud, not a needle, and nothing
    // narrow enough to alias into a flickering sliver.
    if (tip) {
      const last = b.n - 1;
      const rMean = capR / ((seg + 1) || 1);
      tangentAt(b, last, _tan);
      const apex = v;
      const o3 = v * 3, o2 = v * 2, o4 = v * 4;
      const h = rMean * 0.88;
      position[o3] = b.x[last] + _tan.x * h;
      position[o3 + 1] = b.y[last] + _tan.y * h;
      position[o3 + 2] = b.z[last] + _tan.z * h;
      normal[o3] = _tan.x; normal[o3 + 1] = _tan.y; normal[o3 + 2] = _tan.z;
      // Any unit vector perpendicular to the apex normal is a valid tangent
      // basis here; the frame's own X at the last ring is already one.
      tangent[o4] = frameX[(rings - 1) * 3];
      tangent[o4 + 1] = frameX[(rings - 1) * 3 + 1];
      tangent[o4 + 2] = frameX[(rings - 1) * 3 + 2];
      tangent[o4 + 3] = 1;
      uv[o2] = 0.5 * repeatsU;
      uv[o2 + 1] = (b.s[last] + h) / vScale;
      swayA[o4] = b.swA[last * 4]; swayA[o4 + 1] = b.swA[last * 4 + 1];
      swayA[o4 + 2] = b.swA[last * 4 + 2]; swayA[o4 + 3] = b.swA[last * 4 + 3];
      swayB[o4] = b.swB[last * 4]; swayB[o4 + 1] = b.swB[last * 4 + 1];
      swayB[o4 + 2] = b.swB[last * 4 + 2]; swayB[o4 + 3] = b.swB[last * 4 + 3];
      // Same age curve the rings use. Emphatically NOT a hard 1: most caps sit
      // on genuine terminal buds and come out young anyway, but a few close off
      // a limb that was stopped by crowding, and painting a 3 cm bough end
      // glossy mahogany would be worse than the hole it replaces.
      bark[o2] = smoothstep(0.055, 0.007, b.r[last]);
      bark[o2 + 1] = clamp01(b.ao[last]);
      v++;

      // Outward winding, verified against the analytic vertex normals rather
      // than derived: collapsing the far ring of the side-quad pattern onto a
      // single apex reverses the sense of the triangle, because the apex is no
      // longer displaced along +theta from the vertex it replaces.
      const ringBase = base + (rings - 1) * ringVerts;
      for (let k = 0; k < seg; k++) {
        index[ii++] = ringBase + k;
        index[ii++] = ringBase + k + 1;
        index[ii++] = apex;
      }
    }

    for (let ri = 0; ri < rings - 1; ri++) {
      const a = base + ri * ringVerts;
      const c = a + ringVerts;
      // WINDING: counter-clockwise seen from OUTSIDE the tube, which is what
      // three's default front-face test expects.
      //
      // These two triangles used to be emitted as (a+k, c+k, a+k+1) and
      // (a+k+1, c+k, c+k+1). Ring vertices run counter-clockwise around the
      // branch axis when viewed from the tip looking back down it - i.e.
      // CLOCKWISE from outside - so that order made every outward face a back
      // face. With side: FrontSide the entire outer surface of the tree was
      // culled and what rendered was the inside of the far wall: branches read
      // as flat, hollow, pale ribbons, and the near side of the trunk was
      // simply missing. Swapping the second and third index of each triangle
      // reverses the winding without touching positions or normals, which are
      // already the outward analytic partials of the swept surface.
      for (let k = 0; k < seg; k++) {
        index[ii++] = a + k;
        index[ii++] = a + k + 1;
        index[ii++] = c + k;
        index[ii++] = a + k + 1;
        index[ii++] = c + k + 1;
        index[ii++] = c + k;
      }
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(position, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(normal, 3));
  geo.setAttribute('tangent', new THREE.BufferAttribute(tangent, 4));
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  geo.setAttribute('aSwayA', new THREE.BufferAttribute(swayA, 4));
  geo.setAttribute('aSwayB', new THREE.BufferAttribute(swayB, 4));
  geo.setAttribute('aBark', new THREE.BufferAttribute(bark, 2));
  geo.setIndex(new THREE.BufferAttribute(index, 1));
  geo.computeBoundingSphere();
  geo.computeBoundingBox();
  // Sway pushes vertices outside the rest pose; grow the cull volume so the
  // tree never pops out at the edge of the screen during a storm.
  if (geo.boundingSphere) geo.boundingSphere.radius += 1.2;
  return geo;
}

// ===========================================================================
// Twig sampling for the blossom system
// ===========================================================================

/**
 * Walks the flowering wood and returns evenly spaced sites along it, each
 * carrying the EXACT sway data of the twig it sits on - so a blossom and the
 * twig holding it are displaced by identical shader maths and never separate.
 */
export function collectTwigSites(skel, { minDepth = 3, spacing = 0.075, maxRadius = 0.028 } = {}) {
  const sites = [];
  for (const b of skel.branches) {
    if (b.depth < minDepth) continue;
    const total = b.len || 0;
    if (!(total > 0) || b.n < 2) continue;
    const invLen = 1 / total;

    /**
     * Walk ARC LENGTH, interpolating between skeleton points, rather than
     * visiting the points themselves.
     *
     * The old loop could never place a site closer together than the skeleton's
     * own point spacing, and on flowering wood that is 92 mm at depth 3 - four
     * times the 22 mm `spacing` it was being asked for, so the parameter was
     * inert and the real site pitch was set by `segLen` in the LEVELS table.
     * With a blossom card about 65 mm across, sites 92 mm apart leave a bare
     * stripe of wood between every umbel, and no amount of piling more flowers
     * onto each site closes it - that only builds the flowers UPWARD into the
     * pom-poms the crown was reading as. Sites at the flower's own pitch is
     * what makes the canopy continuous, and interpolation is free: it happens
     * once, at build time, and adds no wood and no draw calls.
     */
    let i = 0;
    for (let s = 0.02; s <= total; s += spacing) {
      while (i < b.n - 2 && b.s[i + 1] < s) i++;
      const s0 = b.s[i], s1 = b.s[i + 1];
      const f = s1 > s0 ? clamp01((s - s0) / (s1 - s0)) : 0;
      const j = i * 4, j2 = (i + 1) * 4;

      const r = lerp(b.r[i], b.r[i + 1], f);
      if (r > maxRadius) continue;

      // Tangent from the segment itself. More faithful than a centred
      // difference at a vertex, and it is the frame the flower's umbel is
      // built on, so it must follow the wood exactly.
      let tx = b.x[i + 1] - b.x[i];
      let ty = b.y[i + 1] - b.y[i];
      let tz = b.z[i + 1] - b.z[i];
      const tl = Math.hypot(tx, ty, tz) || 1;
      tx /= tl; ty /= tl; tz /= tl;

      sites.push({
        x: lerp(b.x[i], b.x[i + 1], f),
        y: lerp(b.y[i], b.y[i + 1], f),
        z: lerp(b.z[i], b.z[i + 1], f),
        tx, ty, tz,
        r, depth: b.depth, rank: b.rank,
        tipness: clamp01(s * invLen),
        ao: lerp(b.ao[i], b.ao[i + 1], f),
        // Sway WEIGHTS vary along the branch and interpolate; the PHASES are
        // constant along it, so the same lerp reproduces them exactly.
        swA0: lerp(b.swA[j], b.swA[j2], f), swA1: b.swA[j + 1],
        swA2: lerp(b.swA[j + 2], b.swA[j2 + 2], f), swA3: b.swA[j + 3],
        swB0: lerp(b.swB[j], b.swB[j2], f), swB1: b.swB[j + 1],
        swB2: lerp(b.swB[j + 2], b.swB[j2 + 2], f), swB3: b.swB[j + 3],
      });
    }
  }
  return sites;
}

// ===========================================================================
// Shared wind GLSL - bark, blossoms and both depth materials use this
// ===========================================================================

/**
 * Declarations plus the two deformation functions, injected into `<common>`.
 *
 * The split is deliberate and load-bearing:
 *   `sakuraBend` is a POSITION FIELD, so it rotates normals and a blossom
 *     card's corners bend around a limb exactly like the wood does.
 *   `sakuraOsc` is a pure TRANSLATION for a given weight set, so evaluating it
 *     twice for a finite-difference normal is pointless - it cancels. That is
 *     why the normal correction costs two cheap bend evaluations and zero
 *     extra trigonometry.
 */
export const SAKURA_WIND_GLSL = /* glsl */ `
uniform float uSakuraTime;
uniform vec2  uSakuraDir;
uniform vec2  uSakuraBend;
uniform vec4  uSakuraAmp;
uniform vec4  uSakuraFreq;
uniform vec3  uSakuraExtra;   // x = 1/height, y = storm 0..1, z = flutter amplitude

// Length-preserving main bend, pinned and flat at the trunk base.
vec3 sakuraBend( vec3 p ) {
  float len = length( p );
  if ( len < 1e-4 ) return p;
  float f = p.y * uSakuraExtra.x + 1.0;
  f *= f;
  f = f * f - f;
  vec3 q = p;
  q.xz += uSakuraBend * f;
  return normalize( q ) * len;
}

// Four hierarchical oscillators: limb, branch, branchlet, twig. Weights are
// zero at every branch base, so joints cannot tear however hard the wind blows.
vec3 sakuraOsc( vec4 swA, vec4 swB ) {
  vec4 w  = vec4( swA.x, swA.z, swB.x, swB.z );
  vec4 ph = vec4( swA.y, swA.w, swB.y, swB.w ) * 6.2831853;
  vec4 a  = uSakuraAmp * w;
  vec4 s  = sin( uSakuraTime * uSakuraFreq + ph );
  // A second, incommensurate frequency turns a 1D wag into a lazy figure-eight.
  vec4 c  = cos( uSakuraTime * uSakuraFreq * 0.607 + ph * 1.31 + 1.7 );
  vec2 perp = vec2( -uSakuraDir.y, uSakuraDir.x );
  float along = dot( a, s );
  float side  = dot( a, c ) * 0.55;
  float lift  = ( a.z * c.z + a.w * s.w ) * 0.42;
  return vec3( uSakuraDir.x * along + perp.x * side,
               lift,
               uSakuraDir.y * along + perp.y * side );
}
`;

/** One shared uniform block; every tree material references these same objects. */
export function createSakuraWindUniforms() {
  return {
    uSakuraTime: { value: 0 },
    uSakuraDir: { value: new THREE.Vector2(1, 0) },
    uSakuraBend: { value: new THREE.Vector2(0, 0) },
    uSakuraAmp: { value: new THREE.Vector4(0.26, 0.15, 0.075, 0.035) },
    uSakuraFreq: { value: new THREE.Vector4(1.15, 2.1, 3.6, 6.4) },
    uSakuraExtra: { value: new THREE.Vector3(1 / 11, 0, 0.02) },
  };
}
