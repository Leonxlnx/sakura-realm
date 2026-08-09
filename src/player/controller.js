/**
 * player/controller.js - the body the camera is bolted to.
 *
 * Owns `state.player` and the transform of `ctx.camera`. Nothing else in the
 * project moves the camera; everything else reads it.
 *
 * ---------------------------------------------------------------------------
 * DESIGN NOTES
 * ---------------------------------------------------------------------------
 * The brief for this scene is "quiet, cinematic, slightly melancholy". The
 * single fastest way to destroy that is a camera that behaves like a debug
 * flycam: instant velocity, a rigid eye height glued to the heightfield, and a
 * view direction that is exactly the mouse integral. So:
 *
 *  1. VELOCITY IS INTEGRATED, NEVER ASSIGNED. Ground movement uses the Quake
 *     friction/accelerate pair, which gives a ~0.3 s ramp to top speed and a
 *     short, controllable slide on release. Air control is a fraction of ground
 *     control, so a jump commits you to your trajectory.
 *
 *  2. THE PHYSICS FOLLOW THE GROUND EXACTLY; THE CAMERA DOES NOT. The feet are
 *     snapped to `terrain.getHeight()` every frame (no interpenetration, no
 *     floating), but the *ground delta* is fed into a decaying offset that the
 *     eye lags behind by - Source-style step smoothing. Slopes then read as
 *     weight rather than as a heightfield scrolling under a rigid pole, and
 *     because the offset is driven by the delta rather than by a smoothed
 *     height, it cannot accumulate a steady-state sink on flat ground.
 *
 *  3. HEAD BOB IS DRIVEN BY DISTANCE TRAVELLED, NOT BY TIME. Phase advances
 *     with metres walked and stride length grows with speed, so footsteps stay
 *     locked to the gait at every speed and through every acceleration. A pure
 *     sine is the giveaway of cheap bob, so the vertical curve carries a second
 *     harmonic and each footfall injects an impulse into a critically damped
 *     spring - the same spring the landing impact uses. Nothing in the camera's
 *     vertical motion is a raw sine wave.
 *
 *  4. MODE CHANGES EASE, THEY DO NOT SNAP. `F` never teleports and never
 *     discards momentum: gravity and air drag cross-fade over ~0.3 s, so
 *     entering fly is a glide out of a stride and leaving it is an arc into a
 *     landing. Orientation is untouched.
 *
 * ---------------------------------------------------------------------------
 * COST
 * ---------------------------------------------------------------------------
 * One `terrain.getHeight()` per frame is mandatory (grounding). The slope probe
 * (`getNormal`, four more samples) runs on a tier-dependent stride, and the
 * organic sway octave is dropped at LOW - so LOW costs ~2 height samples per
 * frame against HIGH's ~5. Everything else is a handful of transcendentals.
 * No allocation anywhere in `update()`.
 */

import * as THREE from 'three';
import { EVENTS } from '../core/state.js';
import {
  clamp, clamp01, lerp, smoothstep, damp, dampAngle, mod,
  TAU, PI, DEG2RAD, noise,
} from '../core/math.js';

// ---------------------------------------------------------------------------
// Locomotion. Metres, seconds, radians.
// ---------------------------------------------------------------------------

/** Unhurried pace. This is a landscape to be walked through, not raced across. */
const WALK_SPEED = 3.4;
const SPRINT_SPEED = 6.7;
const CROUCH_SPEED = 1.5;

/**
 * Quake-style accelerate: per frame the speed along the wish direction gains
 * `accel · wishSpeed · dt`, capped by the remaining deficit, while friction
 * pulls the whole velocity down. Running the two against each other gives
 *
 *     dv/dt = accel·wish − friction·v   ⇒   v(t) = (accel/friction)·wish·(1 − e^(−friction·t))
 *
 * so the terminal speed is `wish · accel / friction` - **accel must exceed
 * friction or the player can never reach their own walk speed**, and the ramp
 * time is −ln(1 − friction/accel)/friction. At 7.0 against 5.6 that is 0.29 s
 * to full walk: immediate on the first frame, unhurried into the last.
 */
const GROUND_ACCEL = 7.0;
const SPRINT_ACCEL = 6.2;
const AIR_ACCEL = 1.2;
/** Fraction of ground speed the player may still steer toward while airborne. */
const AIR_CONTROL = 0.75;

/** Exponential ground friction, plus a floor so the last cm/s still stop. */
const FRICTION = 5.6;
const STOP_SPEED = 1.1;

/** Not 9.81: a physical g makes a 0.9 m hop hang for 0.86 s, which reads floaty. */
const GRAVITY = 19.0;
const JUMP_HEIGHT = 0.9;
const JUMP_VELOCITY = Math.sqrt(2 * GRAVITY * JUMP_HEIGHT);

/** Jump forgiveness: still jumpable this long after walking off an edge... */
const COYOTE_TIME = 0.12;
/** ...and a jump pressed this long before landing still fires on touchdown. */
const JUMP_BUFFER = 0.12;

/** Steepest walkable slope. Beyond this the player slides instead of climbing. */
const MAX_SLOPE = 47 * DEG2RAD;
const MAX_SLOPE_TAN = Math.tan(MAX_SLOPE);
/** Fraction of gravity that pulls you down a too-steep face. */
const SLIDE_GRAVITY = 0.55;
/** Steering authority retained while sliding. */
const SLIDE_CONTROL = 0.22;

/** Drop this far or less in one frame and you stay glued to the ground. */
const STEP_DOWN = 0.5;

/** Eye height multiplier while crouched. */
const CROUCH_RATIO = 0.56;

// ---------------------------------------------------------------------------
// Flight
// ---------------------------------------------------------------------------

const FLY_BASE_SPEED = 9.0;
/** Top of the hold-to-go-faster ramp, as a multiple of the base speed. */
const FLY_RAMP_MAX = 4.2;
/** Seconds of held input to reach the top of the ramp. */
const FLY_RAMP_TIME = 3.6;
/** Seconds to bleed the ramp away once input stops. */
const FLY_RAMP_DECAY = 1.1;
const FLY_BOOST = 2.8;
/** Velocity convergence rate with / without input. Low decay rate == long glide. */
const FLY_ACCEL_LAMBDA = 4.2;
const FLY_DRAG_LAMBDA = 1.35;
/** Wheel-adjustable multiplier on the base speed. */
const FLY_SCALE_MIN = 0.3;
const FLY_SCALE_MAX = 6.0;

/** Minimum eye clearance above the terrain in fly mode. */
const FLY_CLEARANCE = 0.55;
/**
 * Deceleration used to cushion a descent, m/s². The permitted descent speed at
 * clearance h is sqrt(2·FLY_BRAKE·h) - the standard braking-distance law - so
 * the approach begins exactly far enough out to arrive at the floor with zero
 * vertical speed, whether the player is drifting down at 2 m/s or diving at
 * 100. A fixed cushion band cannot do that: at any band width there is a dive
 * speed that blows straight through it.
 */
const FLY_BRAKE = 30.0;

/** Cross-fade rate between the walk and fly force models. */
const MODE_BLEND_LAMBDA = 7.0;

// ---------------------------------------------------------------------------
// Look
// ---------------------------------------------------------------------------

/** Radians of yaw per pixel of raw mouse movement. */
const DEFAULT_SENSITIVITY = 0.0022;
/** Just short of straight up/down: exactly ±90° makes yaw meaningless. */
const PITCH_LIMIT = 89.2 * DEG2RAD;
/**
 * Look smoothing. τ = 1/λ ≈ 26 ms - under one and a half frames at 60 Hz, and
 * well below the ~50 ms where added latency becomes perceptible. Enough to
 * absorb the frame-to-frame variance of an unsynchronised mouse, not enough to
 * feel like the camera is on a rubber band.
 */
const LOOK_LAMBDA = 38;
/** Frames of mouse input discarded after acquiring lock (Chrome emits a spike). */
const LOOK_WARMUP_FRAMES = 2;

// ---------------------------------------------------------------------------
// Camera feel
// ---------------------------------------------------------------------------

/** Metres of stride per step: a walk is ~1.5 m, a run ~2.0 m. */
const STRIDE_BASE = 0.95;
const STRIDE_PER_SPEED = 0.16;

const BOB_Y = 0.031;      // metres, peak
const BOB_X = 0.026;      // metres, peak
const BOB_ROLL = 0.62 * DEG2RAD;
const BOB_PITCH = 0.30 * DEG2RAD;

/**
 * Impulse magnitudes below are expressed as the PEAK DISPLACEMENT the event
 * produces (metres or radians), not as a spring velocity - see `springKick()`.
 * Tuning a camera in "how far does it move" is the only way these numbers stay
 * meaningful when the stiffness is changed.
 */
/** Vertical spring stiffness. ω = 17 settles a footfall in ~0.18 s. */
const SPRING_Y_OMEGA = 17.0;
const SPRING_ANG_OMEGA = 13.0;

/** Camera drop at a footfall, at walking pace. Scales with gait speed. */
const FOOTSTEP_DIP = 0.013;
const FOOTSTEP_ROLL = 0.15 * DEG2RAD;

/** The body loads before it leaves the ground. */
const JUMP_DIP = 0.020;
const JUMP_PITCH = 0.45 * DEG2RAD;

/** Landing response per m/s of impact, and the impact speed it saturates at. */
const LAND_DIP = 0.020;
const LAND_PITCH = 0.30 * DEG2RAD;
const LAND_ROLL = 0.15 * DEG2RAD;
const LAND_MAX_IMPACT = 11.0;

/**
 * Eye lag behind ground-height changes; τ = 71 ms. Steady state while climbing
 * is (ground rate)/λ, so the ~17° slopes this terrain actually produces park
 * the eye 4-6 cm below nominal while ascending - felt as weight, never seen.
 */
const STEP_SMOOTH_LAMBDA = 14.0;
const STEP_SMOOTH_CLAMP = 0.32;

/** Degrees of extra vertical FOV at full sprint, and at full flight speed. */
const FOV_SPRINT = 5.0;
const FOV_FLY = 7.0;
const FOV_LAMBDA = 5.0;

/** Radians of roll per rad/s of yaw rate, and the cap on the result. */
const LEAN_PER_YAW_RATE = 0.055;
const LEAN_FLY_PER_YAW_RATE = 0.30;
const LEAN_MAX = 6.5 * DEG2RAD;
const LEAN_LAMBDA = 5.5;
/** Roll from sideways velocity (leaning into a strafe). */
const STRAFE_LEAN = 0.9 * DEG2RAD;

/** Breathing while near-stationary. Deliberately at the edge of perception. */
const IDLE_AMPLITUDE = 0.0055;
const IDLE_RATE = 0.30;

/** Wind buffeting the head. Scales with `state.wind` so a calm day is still. */
const WIND_SWAY = 0.16 * DEG2RAD;

// ---------------------------------------------------------------------------
// Module scratch - nothing below allocates per frame.
// ---------------------------------------------------------------------------

const _euler = new THREE.Euler(0, 0, 0, 'YXZ');
/** Cached ground normal under the player, refreshed on the probe cadence. */
const _slopeNormal = new THREE.Vector3(0, 1, 0);
/** Transient normal for one-off queries - must never clobber the cached one. */
const _queryNormal = new THREE.Vector3(0, 1, 0);

/**
 * Exact critically damped spring step. Unconditionally stable at any dt, which
 * matters because the frame loop hands us up to 50 ms after a tab switch and a
 * semi-implicit integrator would explode there.
 *   x(t) = (x₀ + (v₀ + ωx₀)t)·e^(−ωt)
 */
function springStep(s, omega, dt) {
  const e = Math.exp(-omega * dt);
  const c = s.v + omega * s.x;
  const x = (s.x + c * dt) * e;
  s.v = (s.v - c * omega * dt) * e;
  s.x = x;
}

/**
 * Kicks a critically damped spring so that it peaks at exactly `peak`.
 * From rest, x(t) = v₀·t·e^(−ωt) peaks at t = 1/ω with x = v₀/(ωe), so the
 * velocity needed for a given peak is peak·ω·e. Every impulse constant in this
 * file is therefore readable as a distance or an angle rather than as a
 * dimensionless number that only makes sense next to one particular stiffness.
 */
function springKick(s, peak, omega) {
  s.v += peak * omega * Math.E;
}

/** Shortest signed difference between two angles. */
const angleDelta = (a, b) => mod(b - a + PI, TAU) - PI;

// ===========================================================================
// PlayerController
// ===========================================================================

export class PlayerController {
  constructor(ctx) {
    this.ctx = ctx;
    this.camera = ctx.camera;
    this.input = ctx.input;
    this.state = ctx.state;
    this.bus = ctx.bus;
    this.canvas = ctx.canvas;

    /** Filled in link(); until then the world is a plane at y = 0. */
    this.terrain = null;

    // -- Tunables other systems (HUD settings panel) may write ---------------
    /** Radians of rotation per pixel of mouse movement. */
    this.sensitivity = DEFAULT_SENSITIVITY;
    this.invertY = false;
    /** Whether the sprint FOV kick is applied at all. */
    this.fovKick = true;
    this.headBob = true;

    // -- Kinematics ---------------------------------------------------------
    /** Base of the player (feet in walk mode). The eye sits `_eye` above it. */
    this._pos = new THREE.Vector3(
      this.state.player.position.x,
      this.state.player.position.y - this.state.player.eyeHeight,
      this.state.player.position.z
    );
    this._vel = new THREE.Vector3();
    /** Smoothed eye height above `_pos`, follows crouch. */
    this._eye = this.state.player.eyeHeight;

    this.mode = this.state.player.mode === 'fly' ? 'fly' : 'walk';
    /** 1 while flying, 0 while walking - the cross-fade, not the mode. */
    this._flyBlend = this.mode === 'fly' ? 1 : 0;

    this._grounded = false;
    this._groundY = 0;
    this._prevGroundY = 0;
    /** Slope of the ground under the player, radians from horizontal. */
    this._slope = 0;
    this._probeFrame = -1000;

    this._coyote = 0;
    this._jumpBuffer = 0;
    this._sliding = false;

    // -- Look ---------------------------------------------------------------
    this._yaw = 0;
    this._pitch = 0;
    this._yawTarget = 0;
    this._pitchTarget = 0;
    this._yawRate = 0;
    this._lookWarmup = 0;

    // -- Camera feel --------------------------------------------------------
    this._stepPhase = 0;
    this._stepIndex = 0;
    this._footSign = 1;
    this._bobAmount = 0;
    this._airAmount = 0;
    this._stepOffset = 0;

    this._springY = { x: 0, v: 0 };
    this._springPitch = { x: 0, v: 0 };
    this._springRoll = { x: 0, v: 0 };

    this._lean = 0;
    this._sprintAmount = 0;
    this._crouchAmount = 0;

    this._fovOffset = 0;
    this._fovApplied = 0;
    this._fovSensScale = 1;

    // -- Flight -------------------------------------------------------------
    this._flyRamp = 0;
    this._flyScale = 1;
    this._flyWishX = 0;
    this._flyWishY = 0;
    this._flyWishZ = 0;

    // -- Input snapshot (reused every frame, never reallocated) --------------
    this._in = {
      active: false,
      forward: 0,
      strafe: 0,
      vertical: 0,
      moving: false,
      sprint: false,
      crouch: false,
      jump: false,
      jumpHeld: false,
    };

    // -- Pointer lock -------------------------------------------------------
    this._wasLocked = false;
    this._everLocked = false;
    this._lockDenied = false;
    this._lockCooldown = 0;
    this._lockPending = false;

    this._initialized = false;

    // Quality-driven work reduction; set properly by onQualityChange().
    this._probeStride = 1;
    this._detail = 1;

    this._onCanvasDown = this._onCanvasDown.bind(this);
    this.canvas?.addEventListener('mousedown', this._onCanvasDown);

    this._offPointerLock = this.bus?.on('input:pointerlock', (locked) => {
      // Fires before our next update(); record the moment lock was lost so the
      // re-request cooldown is measured from the right instant.
      if (!locked) this._lockCooldown = 1.35;
    });

    this.onQualityChange(this.state.quality);
  }

  // =========================================================================
  // Lifecycle
  // =========================================================================

  link(systems) {
    const t = systems?.terrain;
    if (t && typeof t.getHeight === 'function') this.terrain = t;

    // Adopt whatever the engine pointed the camera at, so the opening frame is
    // whatever the scene author chose rather than a hard-coded heading.
    this.camera.getWorldDirection(_queryNormal);
    if (_queryNormal.lengthSq() > 1e-6) {
      this._yaw = Math.atan2(-_queryNormal.x, -_queryNormal.z);
      this._pitch = clamp(Math.asin(clamp(_queryNormal.y, -1, 1)), -PITCH_LIMIT, PITCH_LIMIT);
    }
    this._yawTarget = this._yaw;
    this._pitchTarget = this._pitch;

    const p = this.state.player.position;
    this._pos.set(p.x, p.y - this.state.player.eyeHeight, p.z);
  }

  onQualityChange(quality) {
    const tier = (quality && quality.tier) || this.state.quality.tier;
    // The slope probe is four extra height samples; sampling it at 15 Hz is
    // indistinguishable from 60 Hz because the terrain's finest octave has a
    // 23 m wavelength - you cannot cross a slope change in four frames.
    this._probeStride = tier === 'low' ? 4 : tier === 'medium' ? 2 : 1;
    // 0: no noise-driven sway, no bob harmonic, no swept fly collision.
    this._detail = tier === 'low' ? 0 : tier === 'ultra' ? 2 : 1;
    this._probeFrame = -1000; // force a fresh probe at the new cadence
  }

  dispose() {
    this.canvas?.removeEventListener('mousedown', this._onCanvasDown);
    this._offPointerLock?.();
    if (typeof document !== 'undefined' && document.pointerLockElement === this.canvas) {
      document.exitPointerLock?.();
    }
  }

  // =========================================================================
  // Public API - the HUD drives the player through these, never through fields
  // =========================================================================

  get isFlying() {
    return this.mode === 'fly';
  }

  toggleMode() {
    this.setMode(this.mode === 'fly' ? 'walk' : 'fly');
  }

  /**
   * Switches locomotion model. Orientation and momentum are always preserved;
   * the force models cross-fade rather than swapping, so there is no lurch.
   */
  setMode(mode) {
    const next = mode === 'fly' ? 'fly' : 'walk';
    if (next === this.mode) return;
    const previous = this.mode;
    this.mode = next;
    this.state.player.mode = next;

    if (next === 'fly') {
      // Leaving the ground: keep the stride's momentum, drop the gait.
      this._grounded = false;
      this._coyote = 0;
      this._flyRamp = 0;
      // A small lift as the weight comes off the feet.
      springKick(this._springY, 0.012, SPRING_Y_OMEGA);
    } else {
      this._jumpBuffer = 0;
      // Falling back into the world: gravity fades in via _flyBlend, so the
      // player arcs down instead of dropping the instant the key is pressed.
      this._flyRamp = 0;
      // Flight never probes the slope; make sure the first walking frame does.
      this._probeFrame = -1000;
    }

    this.bus?.emit(EVENTS.PLAYER_MODE_CHANGED, { mode: next, previous });
  }

  /** Places the player at a world position (x, z ground-relative feet height). */
  teleport(x, y, z) {
    this._pos.set(x, y, z);
    this._vel.set(0, 0, 0);
    this._stepOffset = 0;
    this._springY.x = this._springY.v = 0;
    this._springPitch.x = this._springPitch.v = 0;
    this._springRoll.x = this._springRoll.v = 0;
    this._initialized = false;
  }

  // =========================================================================
  // Frame
  // =========================================================================

  update(dt, state) {
    // main.js already clamps, but a controller that can be stepped by hand from
    // the console must not be able to tunnel the player through the world.
    const step = clamp(dt, 0, 1 / 20);
    const profile = state.debug.showStats && this.ctx.quality?.registerCost;
    const t0 = profile ? performance.now() : 0;

    this._updatePointerLock(step);
    this._readInput(state);
    this._updateLook(step);

    // Cross-fade between the two force models. Gravity scales with (1 - blend)
    // and fly drag with blend, so at the instant of a mode change neither acts
    // and the player simply keeps moving.
    this._flyBlend = damp(this._flyBlend, this.mode === 'fly' ? 1 : 0, MODE_BLEND_LAMBDA, step);
    if (this._flyBlend < 1e-3) this._flyBlend = 0;
    else if (this._flyBlend > 1 - 1e-3) this._flyBlend = 1;

    if (!this._initialized) this._settle(state);

    this._updateStance(step, state);
    if (this.mode === 'fly') this._flyPhysics(step, state);
    else this._walkPhysics(step, state);

    this._updateCameraFeel(step, state);
    this._writeState(state);

    if (profile) this.ctx.quality.registerCost('player', performance.now() - t0);
  }

  /** First real frame: drop the player onto the ground with no visible motion. */
  _settle(state) {
    this._groundY = this._sampleGround(this._pos.x, this._pos.z);
    if (this.mode === 'walk') {
      this._pos.y = Math.max(this._pos.y, this._groundY);
      if (this._pos.y - this._groundY < STEP_DOWN) {
        this._pos.y = this._groundY;
        this._grounded = true;
      }
    }
    this._prevGroundY = this._groundY;
    this._stepOffset = 0;
    this._eye = state.player.eyeHeight;
    this._probe(state, true);
    this._initialized = true;
  }

  // =========================================================================
  // LEAF 3 - pointer lock and input
  // =========================================================================

  _onCanvasDown(event) {
    if (event.button !== 0) return;
    if (this.input.uiHasFocus) return;
    if (this.input.pointerLocked) return;
    if (this._lockCooldown > 0) {
      // Clicking during the browser's post-Escape lockout would otherwise do
      // nothing at all, which reads as a broken page. Remember the intent and
      // fire it the moment the lockout expires - the document keeps its sticky
      // activation, so the deferred request is still allowed.
      this._lockPending = true;
      return;
    }
    this._requestLock();
  }

  /**
   * Chrome rate-limits `requestPointerLock` for ~1.25 s after a user-initiated
   * Escape, and rejects the returned promise. Unhandled, that surfaces as a red
   * console error every time someone taps Escape and clicks back in - so the
   * request is made here rather than through `input.requestPointerLock()`,
   * which does not catch.
   */
  _requestLock() {
    const el = this.canvas;
    if (!el || !el.requestPointerLock) return;
    this._lockCooldown = 0.4; // don't spam while the browser thinks about it
    let result;
    try {
      result = el.requestPointerLock();
    } catch (err) {
      this._noteLockFailure();
      return;
    }
    if (result && typeof result.catch === 'function') {
      result.catch(() => this._noteLockFailure());
    }
  }

  /**
   * If lock has never once succeeded we are probably in a sandboxed iframe or a
   * browser that refuses it. Rather than leaving the world completely inert,
   * fall back to keyboard-only movement - no look control, but the player can
   * still walk, and nothing on screen looks broken.
   */
  _noteLockFailure() {
    if (!this._everLocked) this._lockDenied = true;
  }

  _updatePointerLock(dt) {
    if (this._lockCooldown > 0) {
      this._lockCooldown = Math.max(0, this._lockCooldown - dt);
      if (this._lockCooldown === 0 && this._lockPending && !this.input.pointerLocked) {
        this._lockPending = false;
        this._requestLock();
      }
    }

    const locked = this.input.pointerLocked;
    if (locked && !this._wasLocked) {
      this._everLocked = true;
      this._lockDenied = false;
      this._lockPending = false;
      // Discard the first couple of frames of movement: the pointer-lock
      // transition itself can generate a single enormous movementX/Y.
      this._lookWarmup = LOOK_WARMUP_FRAMES;
      // Re-anchor the smoothed angles so nothing eases in from a stale target.
      this._yawTarget = this._yaw;
      this._pitchTarget = this._pitch;
    } else if (!locked && this._wasLocked) {
      // Escape. Input has already cleared held keys; make sure no residual look
      // delta is applied and let momentum bleed off through normal friction.
      this._lookWarmup = 0;
      this._yawTarget = this._yaw;
      this._pitchTarget = this._pitch;
    }
    this._wasLocked = locked;
  }

  _readInput(state) {
    const input = this.input;
    const io = this._in;

    // Movement needs lock, except in the degraded no-lock fallback. Without the
    // gate, keystrokes aimed at a focused HUD control would walk the player.
    io.active = input.pointerLocked || this._lockDenied;

    if (!io.active) {
      io.forward = 0;
      io.strafe = 0;
      io.vertical = 0;
      io.moving = false;
      io.sprint = false;
      io.crouch = false;
      io.jump = false;
      io.jumpHeld = false;
      // Nothing snaps: the stance blend below stands the player back up over
      // ~0.1 s, and friction bleeds the remaining momentum off naturally.
      return;
    }

    const fwd = (input.anyDown('KeyW', 'ArrowUp') ? 1 : 0) - (input.anyDown('KeyS', 'ArrowDown') ? 1 : 0);
    const side = (input.anyDown('KeyD', 'ArrowRight') ? 1 : 0) - (input.anyDown('KeyA', 'ArrowLeft') ? 1 : 0);

    // Normalise the diagonal so W+D is not 41 % faster than W alone.
    const mag = Math.sqrt(fwd * fwd + side * side);
    if (mag > 1) {
      io.forward = fwd / mag;
      io.strafe = side / mag;
    } else {
      io.forward = fwd;
      io.strafe = side;
    }
    io.moving = mag > 0;

    const up = input.anyDown('Space', 'KeyE') ? 1 : 0;
    const down = input.anyDown('KeyC', 'KeyQ') ? 1 : 0;
    io.vertical = up - down;

    io.sprint = input.anyDown('ShiftLeft', 'ShiftRight');
    // Ctrl never reaches us: input.js drops any keydown carrying a modifier so
    // the browser keeps its shortcuts. C is the crouch key.
    io.crouch = this.mode === 'walk' && input.isDown('KeyC');
    io.jumpHeld = input.isDown('Space');
    io.jump = input.wasPressed('Space');

    if (input.wasPressed('KeyF')) this.toggleMode();
  }

  // =========================================================================
  // LEAF 3 - look
  // =========================================================================

  _updateLook(dt) {
    const input = this.input;

    if (this._lookWarmup > 0) {
      this._lookWarmup--;
    } else if (input.pointerLocked && (input.mouseDX !== 0 || input.mouseDY !== 0)) {
      // Compensate for the sprint FOV kick: without this, widening the lens
      // silently speeds the camera up, which reads as the controls changing
      // under the player's hands mid-sprint.
      const sens = this.sensitivity * this._fovSensScale;
      this._yawTarget -= input.mouseDX * sens;
      this._pitchTarget += (this.invertY ? input.mouseDY : -input.mouseDY) * sens;
      this._pitchTarget = clamp(this._pitchTarget, -PITCH_LIMIT, PITCH_LIMIT);
      // Keep yaw bounded; a long session of spinning would otherwise eat float
      // precision and make the smoothing visibly coarse.
      this._yawTarget = mod(this._yawTarget + PI, TAU) - PI;
    }

    const prevYaw = this._yaw;
    this._yaw = dampAngle(this._yaw, this._yawTarget, LOOK_LAMBDA, dt);
    this._yaw = mod(this._yaw + PI, TAU) - PI;
    this._pitch = damp(this._pitch, this._pitchTarget, LOOK_LAMBDA, dt);

    // Yaw rate feeds the lean. Smoothed hard: the per-frame rate is far too
    // noisy to drive a roll angle directly.
    const rate = dt > 1e-5 ? angleDelta(prevYaw, this._yaw) / dt : 0;
    this._yawRate = damp(this._yawRate, rate, 9, dt);
  }

  // =========================================================================
  // LEAF 1 - stance and walking
  // =========================================================================

  _updateStance(dt, state) {
    const io = this._in;

    // On foot, sprint requires forward intent - sprinting backwards is a debug
    // camera. In flight the modifier is a plain boost in whatever direction.
    const wantSprint = io.sprint && !io.crouch && (this.mode === 'fly' || io.forward > 0.1);
    this._sprintAmount = damp(this._sprintAmount, wantSprint ? 1 : 0, 8, dt);
    this._crouchAmount = damp(this._crouchAmount, io.crouch ? 1 : 0, 9, dt);

    const stand = state.player.eyeHeight;
    // `_crouchAmount` carries the crouch's time constant, so the damp here is
    // fast enough (τ = 25 ms) to add no perceptible softness on top of it. Its
    // real job is `state.player.eyeHeight` itself: if the HUD writes that field
    // the eye must glide rather than teleport.
    const target = lerp(stand, stand * CROUCH_RATIO, this._crouchAmount * (1 - this._flyBlend));
    this._eye = damp(this._eye, target, 40, dt);
  }

  _walkPhysics(dt, state) {
    const io = this._in;
    const vel = this._vel;
    const pos = this._pos;

    this._probe(state, false);

    const gravityScale = 1 - this._flyBlend;
    const sliding = this._grounded && this._slope > MAX_SLOPE;
    this._sliding = sliding;

    // -- wish direction, in the horizontal plane of the aim ------------------
    const sinY = Math.sin(this._yaw);
    const cosY = Math.cos(this._yaw);
    // forward = (-sin yaw, 0, -cos yaw); right = (cos yaw, 0, -sin yaw)
    let wishX = -sinY * io.forward + cosY * io.strafe;
    let wishZ = -cosY * io.forward - sinY * io.strafe;
    const wishLen = Math.sqrt(wishX * wishX + wishZ * wishZ);
    if (wishLen > 1e-5) {
      wishX /= wishLen;
      wishZ /= wishLen;
    }

    let wishSpeed = lerp(WALK_SPEED, SPRINT_SPEED, this._sprintAmount);
    wishSpeed = lerp(wishSpeed, CROUCH_SPEED, this._crouchAmount);
    wishSpeed *= wishLen;

    // Climbing costs speed. Falls out of the slope probe, so it is consistent
    // with the slope that will later block the move outright.
    if (this._grounded && wishLen > 1e-5) {
      const uphill = this._uphillDot(wishX, wishZ);
      if (uphill > 0) {
        wishSpeed *= lerp(1, 0.55, smoothstep(0, MAX_SLOPE, this._slope) * uphill);
      }
    }

    // -- friction ------------------------------------------------------------
    if (this._grounded && !sliding) {
      const speed = Math.sqrt(vel.x * vel.x + vel.z * vel.z);
      if (speed > 1e-4) {
        const control = Math.max(speed, STOP_SPEED);
        const drop = control * FRICTION * dt;
        const scale = Math.max(0, speed - drop) / speed;
        vel.x *= scale;
        vel.z *= scale;
      }
    } else if (sliding) {
      // Only a light drag on a slide, otherwise a steep face feels like velcro.
      const scale = Math.exp(-1.4 * dt);
      vel.x *= scale;
      vel.z *= scale;
    }

    // -- acceleration --------------------------------------------------------
    const accel = this._grounded
      ? lerp(GROUND_ACCEL, SPRINT_ACCEL, this._sprintAmount)
      : AIR_ACCEL;

    // Reduced authority is expressed as a lower *ceiling*, not a lower gain:
    // dropping the gain instead would only make the controls feel mushy while
    // still letting the player reach full speed. In the air you may steer, but
    // you may not out-accelerate your run; on a face too steep to climb you can
    // barely fight the slide at all.
    let cap;
    if (!this._grounded) cap = wishSpeed * AIR_CONTROL;
    else if (sliding) cap = wishSpeed * SLIDE_CONTROL;
    else cap = wishSpeed;

    if (wishLen > 1e-5 && cap > 1e-4) {
      const current = vel.x * wishX + vel.z * wishZ;
      const add = cap - current;
      if (add > 0) {
        const gain = Math.min(accel * cap * dt, add);
        vel.x += wishX * gain;
        vel.z += wishZ * gain;
      }
    }

    // Sliding down a face the player cannot climb.
    if (sliding) {
      const gx = -_slopeNormal.x;
      const gz = -_slopeNormal.z;
      const gl = Math.sqrt(gx * gx + gz * gz);
      if (gl > 1e-5) {
        // Downhill is the negated gradient; magnitude follows sin of the slope.
        const push = GRAVITY * SLIDE_GRAVITY * Math.sin(this._slope) * dt;
        vel.x -= (gx / gl) * push;
        vel.z -= (gz / gl) * push;
      }
    }

    // -- jump ----------------------------------------------------------------
    this._coyote = this._grounded && !sliding ? COYOTE_TIME : Math.max(0, this._coyote - dt);
    this._jumpBuffer = io.jump ? JUMP_BUFFER : Math.max(0, this._jumpBuffer - dt);
    if (this._jumpBuffer > 0 && this._coyote > 0 && gravityScale > 0.5) {
      vel.y = JUMP_VELOCITY * lerp(1, 0.8, this._crouchAmount);
      this._grounded = false;
      this._coyote = 0;
      this._jumpBuffer = 0;
      // The body loads before it rises: a small downward camera impulse sells
      // the crouch-and-push far better than a bigger jump arc would.
      springKick(this._springY, -JUMP_DIP, SPRING_Y_OMEGA);
      springKick(this._springPitch, JUMP_PITCH, SPRING_ANG_OMEGA);
    }

    // -- gravity, with a shorter hop if the key is released early ------------
    if (this._grounded) {
      // A little downward bias keeps the feet welded to convex ground; without
      // it the player skips off every crest instead of following it.
      vel.y = -2.0;
    } else {
      const rising = vel.y > 0;
      const g = GRAVITY * gravityScale * (rising && !io.jumpHeld ? 1.9 : 1);
      vel.y -= g * dt;
      // Terminal velocity: nothing good happens past this and it bounds the
      // per-frame displacement the ground test has to cope with.
      if (vel.y < -55) vel.y = -55;
    }

    // -- horizontal integration, with slope blocking -------------------------
    // Sampled after the jump block, which has already cleared `_grounded`: a
    // jump must never stick to the ground or trigger the landing spring.
    const wasGrounded = this._grounded;
    let nx = pos.x + vel.x * dt;
    let nz = pos.z + vel.z * dt;
    let ground = this._sampleGround(nx, nz);

    const stepDist = Math.sqrt(vel.x * vel.x + vel.z * vel.z) * dt;

    if (wasGrounded && stepDist > 1e-4) {
      // Rise over run along the direction of travel - a *gradient*, not a
      // height. Comparing raw heights with a fixed tolerance would let the
      // player creep up a cliff at walking pace and change what is climbable
      // with the framerate; a ratio is exact at any speed and any dt, because
      // the heightfield is analytic and its finite difference converges on the
      // true directional derivative as the step shrinks.
      if ((ground - this._groundY) / stepDist > MAX_SLOPE_TAN) {
        // Too steep. Project the uphill component out of the velocity so the
        // player slides along the contour rather than sticking to the face - 
        // catching on invisible geometry is the classic tell of a naive
        // heightfield controller.
        // A one-off query: it must not overwrite the cached slope normal, which
        // the sliding and uphill-cost terms above are still reading.
        if (this.terrain) this.terrain.getNormal(nx, nz, _queryNormal);
        else _queryNormal.set(0, 1, 0);
        let ux = -_queryNormal.x;
        let uz = -_queryNormal.z;
        const ul = Math.sqrt(ux * ux + uz * uz);
        if (ul > 1e-5) {
          ux /= ul;
          uz /= ul;
          const into = vel.x * ux + vel.z * uz;
          if (into > 0) {
            vel.x -= ux * into;
            vel.z -= uz * into;
          }
        }
        nx = pos.x + vel.x * dt;
        nz = pos.z + vel.z * dt;
        ground = this._sampleGround(nx, nz);
        // The contour curves, so one projection is not always enough. If the
        // deflected move still climbs, refuse it outright - that guarantees the
        // player can never walk up a wall, whatever the terrain does.
        const dist2 = Math.sqrt(vel.x * vel.x + vel.z * vel.z) * dt;
        if (dist2 > 1e-4 && (ground - this._groundY) / dist2 > MAX_SLOPE_TAN) {
          nx = pos.x;
          nz = pos.z;
          ground = this._groundY;
          vel.x *= 0.2;
          vel.z *= 0.2;
        }
      }
    }
    pos.x = nx;
    pos.z = nz;

    // -- vertical integration and grounding ----------------------------------
    pos.y += vel.y * dt;
    const gap = pos.y - ground;

    if (gap <= 0) {
      const impact = -vel.y;
      pos.y = ground;
      vel.y = 0;
      this._grounded = true;
      if (!wasGrounded && impact > 1.2) this._land(impact);
    } else if (wasGrounded && vel.y <= 0 && gap < STEP_DOWN) {
      // Walking off a lip or down a slope: stay glued rather than launching.
      pos.y = ground;
      vel.y = 0;
      this._grounded = true;
    } else {
      this._grounded = false;
    }

    // -- eye lag over ground changes -----------------------------------------
    if (this._grounded && wasGrounded) {
      this._stepOffset -= ground - this._prevGroundY;
      this._stepOffset = clamp(this._stepOffset, -STEP_SMOOTH_CLAMP, STEP_SMOOTH_CLAMP);
    }
    this._groundY = ground;
    this._prevGroundY = ground;
  }

  _land(impact) {
    const i = Math.min(impact, LAND_MAX_IMPACT);
    springKick(this._springY, -i * LAND_DIP, SPRING_Y_OMEGA);
    // The head pitches down as the knees absorb, and rolls onto the leading
    // foot - landing perfectly square is the tell of a camera with no body.
    springKick(this._springPitch, -i * LAND_PITCH, SPRING_ANG_OMEGA);
    springKick(this._springRoll, this._footSign * i * LAND_ROLL, SPRING_ANG_OMEGA);
    // The eye-lag offset is meaningless across a landing; the spring owns the
    // vertical from here.
    this._stepOffset = clamp(this._stepOffset, -0.05, 0.05);
  }

  // =========================================================================
  // LEAF 2 - flight
  // =========================================================================

  _flyPhysics(dt, state) {
    const io = this._in;
    const vel = this._vel;
    const pos = this._pos;

    // Wheel trims the cruising speed. Logarithmic, so a notch is the same
    // proportional change whether you are crawling or crossing the valley.
    const wheel = this.input.wheelDelta;
    if (io.active && wheel !== 0) {
      // `deltaY` is not a portable unit: Chrome reports ~100 per notch (pixel
      // mode), Firefox ~3 (line mode) and trackpads a stream of small values.
      // Clamping the magnitude into a sane number of "notches" makes the
      // control behave the same everywhere instead of being inert on Firefox.
      const notches = Math.sign(wheel) * clamp(Math.abs(wheel) / 100, 0.6, 3);
      this._flyScale = clamp(
        this._flyScale * Math.exp(-notches * 0.16),
        FLY_SCALE_MIN,
        FLY_SCALE_MAX
      );
    }

    // -- wish direction: full 6DOF from the aim basis ------------------------
    const sinY = Math.sin(this._yaw);
    const cosY = Math.cos(this._yaw);
    const sinP = Math.sin(this._pitch);
    const cosP = Math.cos(this._pitch);

    let wx = -sinY * cosP * io.forward + cosY * io.strafe;
    let wy = sinP * io.forward + io.vertical;
    let wz = -cosY * cosP * io.forward - sinY * io.strafe;

    const wl = Math.sqrt(wx * wx + wy * wy + wz * wz);
    const hasInput = wl > 1e-4;
    if (wl > 1) {
      wx /= wl;
      wy /= wl;
      wz /= wl;
    }

    // -- speed ramp ----------------------------------------------------------
    if (hasInput) {
      // Committing to a heading builds speed; reversing dumps it. Otherwise the
      // ramp turns the whole world into a slalom at 40 m/s.
      const align = wx * this._flyWishX + wy * this._flyWishY + wz * this._flyWishZ;
      if (align < 0.35 && this._flyRamp > 0.05) this._flyRamp *= 0.45;
      this._flyRamp = Math.min(1, this._flyRamp + dt / FLY_RAMP_TIME);
      this._flyWishX = wx;
      this._flyWishY = wy;
      this._flyWishZ = wz;
    } else {
      this._flyRamp = Math.max(0, this._flyRamp - dt / FLY_RAMP_DECAY);
    }

    // `_sprintAmount` is smoothed in _updateStance so the boost eases in.
    const boost = lerp(1, FLY_BOOST, this._sprintAmount);
    // Quadratic ramp: the first second stays controllable, the last second
    // really moves. Linear feels like nothing is happening until it is too fast.
    const ramp = lerp(1, FLY_RAMP_MAX, this._flyRamp * this._flyRamp);
    const speed = FLY_BASE_SPEED * this._flyScale * ramp * boost;

    // -- inertia -------------------------------------------------------------
    // Scaled by the mode blend so the first fraction of a second after pressing
    // F preserves the walk's momentum instead of snapping onto the fly model.
    const lambda = (hasInput ? FLY_ACCEL_LAMBDA : FLY_DRAG_LAMBDA) * this._flyBlend;
    vel.x = damp(vel.x, wx * speed, lambda, dt);
    vel.y = damp(vel.y, wy * speed, lambda, dt);
    vel.z = damp(vel.z, wz * speed, lambda, dt);

    // Any gravity still bleeding through from a just-abandoned walk.
    if (this._flyBlend < 1) vel.y -= GRAVITY * (1 - this._flyBlend) * dt;

    // Horizontal first, so the ground test below sees the column the player is
    // actually going to end the frame in.
    pos.x += vel.x * dt;
    pos.z += vel.z * dt;

    // -- ground -------------------------------------------------------------
    let ground = this._sampleGround(pos.x, pos.z);
    // At speed a single sample can straddle a ridge. One extra midpoint tap on
    // MEDIUM and up is enough to stop the camera clipping through a crest.
    if (this._detail > 0) {
      const travel = Math.sqrt(vel.x * vel.x + vel.z * vel.z) * dt;
      if (travel > 1.5) {
        const mid = this._sampleGround(pos.x - vel.x * dt * 0.5, pos.z - vel.z * dt * 0.5);
        if (mid > ground) ground = mid;
      }
    }
    this._groundY = ground;
    this._prevGroundY = ground;

    // Cushioned descent. Capping the *speed* by the room left to stop in means
    // the deceleration is spread over exactly the distance it needs and the
    // arrival is always a settle - never a bounce off an invisible plane, and
    // never a dive straight through one.
    const floorY = ground + FLY_CLEARANCE - this._eye;
    const above = pos.y - floorY;
    if (above > 0) {
      const vMax = Math.sqrt(2 * FLY_BRAKE * above);
      if (vel.y < -vMax) vel.y = -vMax;
    } else if (vel.y < 0) {
      vel.y = 0;
    }

    pos.y += vel.y * dt;

    // The braking law leaves at most a frame of overshoot, and terrain rising
    // under a fast horizontal pass can still outrun it, so the clearance is
    // enforced exactly. Below the floor the correction is centimetres.
    if (pos.y < floorY) {
      pos.y = floorY;
      if (vel.y < 0) vel.y = 0;
    }

    this._grounded = false;
    this._sliding = false;
    this._coyote = 0;
    // Deliberately no slope probe here: nothing in flight reads it, and
    // setMode('walk') forces a fresh one before the first walking frame.
  }

  // =========================================================================
  // Terrain queries
  // =========================================================================

  _sampleGround(x, z) {
    return this.terrain ? this.terrain.getHeight(x, z) : 0;
  }

  /** Refreshes the cached slope/normal on the quality-dependent cadence. */
  _probe(state, force) {
    const frame = state.time.frame;
    if (!force && frame - this._probeFrame < this._probeStride) return;
    this._probeFrame = frame;
    if (this.terrain) {
      this.terrain.getNormal(this._pos.x, this._pos.z, _slopeNormal);
    } else {
      _slopeNormal.set(0, 1, 0);
    }
    this._slope = Math.acos(clamp(_slopeNormal.y, -1, 1));
  }

  /** How much of a horizontal direction points uphill, 0..1. */
  _uphillDot(x, z) {
    const gx = -_slopeNormal.x;
    const gz = -_slopeNormal.z;
    const gl = Math.sqrt(gx * gx + gz * gz);
    if (gl < 1e-5) return 0;
    return clamp01((x * gx + z * gz) / gl);
  }

  // =========================================================================
  // LEAF 3 - head bob, springs, lean, FOV
  // =========================================================================

  _updateCameraFeel(dt, state) {
    const vel = this._vel;
    const io = this._in;
    const walkWeight = 1 - this._flyBlend;

    const horizSpeed = Math.sqrt(vel.x * vel.x + vel.z * vel.z);
    // Flight cares about the full velocity - a vertical dive is still speed.
    const speed3D = vel.length();

    // Airborne fade: the gait should die when the feet leave the ground, but
    // not instantly, or every small hop chops the bob off mid-stride.
    this._airAmount = damp(this._airAmount, this._grounded ? 1 : 0, this._grounded ? 9 : 6, dt);

    // -- gait ----------------------------------------------------------------
    // Stride grows with speed, so step *frequency* lands near 2.3 Hz walking
    // and 3.3 Hz running - the real numbers for a human, and the reason this
    // does not read as a metronome.
    const stride = clamp(STRIDE_BASE + STRIDE_PER_SPEED * horizSpeed, 0.85, 2.3) *
      lerp(1, 0.72, this._crouchAmount);

    let bobTarget = 0;
    if (this.headBob) {
      bobTarget = smoothstep(0.35, 2.2, horizSpeed) * this._airAmount * walkWeight;
      bobTarget *= lerp(1, 1.45, this._sprintAmount) * lerp(1, 0.62, this._crouchAmount);
      // Fade the gait out when the input is released even though momentum
      // remains - the player is coasting to a halt, not still striding.
      if (!io.moving) bobTarget *= 0.55;
    }
    this._bobAmount = damp(this._bobAmount, bobTarget, 6, dt);

    if (this._bobAmount > 1e-4 && stride > 1e-3) {
      // Phase is metres walked, not seconds elapsed: acceleration cannot slip
      // the footfalls out of sync with the motion.
      this._stepPhase += (horizSpeed * dt) / stride;
      const idx = Math.floor(this._stepPhase);
      if (idx > this._stepIndex) {
        this._stepIndex = idx;
        this._footSign = -this._footSign;
        // A footfall is an impulse, not a curve. This is what stops the bob
        // reading as a sine wave however carefully the sine is shaped.
        const weight = this._bobAmount * clamp(horizSpeed / WALK_SPEED, 0.4, 2.0);
        springKick(this._springY, -FOOTSTEP_DIP * weight, SPRING_Y_OMEGA);
        springKick(this._springRoll, this._footSign * FOOTSTEP_ROLL * weight, SPRING_ANG_OMEGA);
      }
      if (this._stepPhase > 8192) {
        // Keep the phase where float precision is still fine over a long
        // session. Shifting by an even number of steps leaves both the gait
        // curves and the left/right alternation exactly continuous.
        this._stepPhase -= 8192;
        this._stepIndex -= 8192;
      }
    }

    // -- springs -------------------------------------------------------------
    springStep(this._springY, SPRING_Y_OMEGA, dt);
    springStep(this._springPitch, SPRING_ANG_OMEGA, dt);
    springStep(this._springRoll, SPRING_ANG_OMEGA, dt);

    // -- eye lag decay -------------------------------------------------------
    this._stepOffset = damp(this._stepOffset, 0, STEP_SMOOTH_LAMBDA, dt);

    // -- bob curves ----------------------------------------------------------
    // θ advances by π per step, so sin(θ) is one cycle per stride pair (sway)
    // and cos(2θ) is one cycle per step (rise and fall).
    const theta = this._stepPhase * PI;
    const amp = this._bobAmount;
    let bobY = 0;
    let bobX = 0;
    let bobRoll = 0;
    let bobPitch = 0;
    if (amp > 1e-4) {
      const c2 = Math.cos(2 * theta);
      // Second harmonic: the real centre-of-mass curve is not a sinusoid - it
      // falls faster than it rises. 0.18 is enough to break the pattern, and
      // the 1/1.18 keeps the peak excursion at exactly BOB_Y either way.
      const h = this._detail > 0 ? 0.18 : 0;
      const harmonic = h * Math.cos(4 * theta);
      bobY = (-(c2 + harmonic) / (1 + h)) * BOB_Y * amp;
      const s1 = Math.sin(theta);
      bobX = s1 * BOB_X * amp;
      bobRoll = s1 * BOB_ROLL * amp;
      bobPitch = Math.sin(2 * theta) * BOB_PITCH * amp;
    }

    // -- idle life -----------------------------------------------------------
    // A perfectly static camera is the loudest "this is a program" tell there
    // is. Breathing plus a wind-driven drift keeps it alive for ~free. It has
    // to survive into fly mode too - a motionless hover is exactly where a
    // frozen camera gives the game away - so stillness is measured from actual
    // motion rather than simply switched off with the mode.
    const stillness = 1 - clamp01(
      this._bobAmount * 2.2 * walkWeight + (this._flyBlend * speed3D) / 6
    );
    let idleY = 0;
    let idlePitch = 0;
    let idleYaw = 0;
    if (stillness > 1e-3) {
      const t = state.time.elapsed;
      // Two incommensurate rates so the breath never visibly loops.
      const breath = Math.sin(t * TAU * IDLE_RATE) * 0.75 + Math.sin(t * TAU * IDLE_RATE * 0.41) * 0.25;
      idleY = breath * IDLE_AMPLITUDE * stillness;
      idlePitch = breath * 0.06 * DEG2RAD * stillness;

      if (this._detail > 0) {
        const w = state.wind;
        // Buffeting scales with the wind the world is actually running, so a
        // still day is a still camera and a storm shoves you around a little.
        // 14 m/s is the storm end of the WindField's range.
        const gust = clamp01((w.strength * w.gust) / 14);
        if (gust > 1e-3) {
          // Noise rather than sines: the head is being pushed, not oscillating.
          idleYaw = noise.noise2D(t * 0.19, 11.3) * WIND_SWAY * gust * stillness;
          idlePitch += noise.noise2D(t * 0.23, -7.1) * WIND_SWAY * 0.6 * gust * stillness;
        }
      }
    }

    // -- lean ----------------------------------------------------------------
    const sinY = Math.sin(this._yaw);
    const cosY = Math.cos(this._yaw);

    // Turning rolls the camera slightly into the turn; flight banks properly,
    // and the two cross-fade with the mode rather than switching.
    const leanWalk = LEAN_PER_YAW_RATE * lerp(0.28, 1, smoothstep(0.4, WALK_SPEED, horizSpeed));
    const leanFly = LEAN_FLY_PER_YAW_RATE * clamp01(horizSpeed / 18);
    const leanScale = lerp(leanWalk, leanFly, this._flyBlend);
    let leanTarget = clamp(-this._yawRate * leanScale, -LEAN_MAX, LEAN_MAX);

    // Strafing leans into the sideways velocity too (on foot only).
    if (walkWeight > 0.01) {
      const lateral = vel.x * cosY - vel.z * sinY; // velocity along +right
      leanTarget -= clamp(lateral / WALK_SPEED, -1.5, 1.5) *
        STRAFE_LEAN * walkWeight * this._airAmount;
    }
    this._lean = damp(this._lean, leanTarget, LEAN_LAMBDA, dt);

    // -- compose the camera transform ---------------------------------------
    const eyeY = this._pos.y + this._eye + this._stepOffset + this._springY.x + bobY + idleY;
    // Lateral bob rides the camera's right vector, (cos yaw, 0, -sin yaw).
    this.camera.position.set(
      this._pos.x + bobX * cosY,
      eyeY,
      this._pos.z - bobX * sinY
    );

    const pitch = clamp(
      this._pitch + this._springPitch.x + bobPitch + idlePitch,
      -PITCH_LIMIT,
      PITCH_LIMIT
    );
    const roll = this._lean + this._springRoll.x + bobRoll;
    _euler.set(pitch, this._yaw + idleYaw, roll, 'YXZ');
    this.camera.quaternion.setFromEuler(_euler);

    this._updateFov(dt, horizSpeed, speed3D);
  }

  /**
   * FOV kick. The engine owns the base FOV (it re-derives it from a fixed focal
   * length on every resize) and explicitly preserves whatever offset we have
   * dialled in, so we only ever track our own offset and add it back.
   */
  _updateFov(dt, horizSpeed, speed3D) {
    const cam = this.camera;
    const base = cam.fov - this._fovApplied;

    let target = 0;
    if (this.fovKick) {
      if (this._flyBlend > 0.01) {
        // Speed-proportional, not throttle-proportional: the kick tracks how
        // fast you are actually going, so it eases in with the ramp.
        target += FOV_FLY * smoothstep(8, 55, speed3D) * this._flyBlend;
      }
      if (this._flyBlend < 0.99) {
        // Only kicks in once the sprint is actually delivering speed.
        const s = this._sprintAmount * smoothstep(WALK_SPEED * 0.9, SPRINT_SPEED * 0.95, horizSpeed);
        target += FOV_SPRINT * s * (1 - this._flyBlend);
      }
    }

    this._fovOffset = damp(this._fovOffset, target, FOV_LAMBDA, dt);
    if (Math.abs(this._fovOffset) < 1e-3) this._fovOffset = 0;

    const next = base + this._fovOffset;
    if (Math.abs(next - cam.fov) > 1e-4) {
      cam.fov = next;
      cam.updateProjectionMatrix();
      this._fovApplied = this._fovOffset;
    }

    // Keep the mouse moving the same number of world-degrees per pixel as the
    // lens widens. tan ratio, not fov ratio - the projection is not linear.
    this._fovSensScale =
      Math.tan(cam.fov * 0.5 * DEG2RAD) / Math.tan(Math.max(base, 1) * 0.5 * DEG2RAD);
  }

  // =========================================================================
  // State publication
  // =========================================================================

  _writeState(state) {
    const p = state.player;
    // The eye position without bob or springs: downstream systems (terrain
    // streaming, wind, fog) want the player's location, not the camera shake.
    p.position.set(this._pos.x, this._pos.y + this._eye, this._pos.z);
    p.velocity.copy(this._vel);

    // Documented as the *horizontal* look direction - used for wind-relative
    // effects, which have no use for pitch.
    const sinY = Math.sin(this._yaw);
    const cosY = Math.cos(this._yaw);
    p.forward.set(-sinY, 0, -cosY);

    p.mode = this.mode;
    p.grounded = this._grounded;
    p.sprinting = this._sprintAmount > 0.5;
  }
}
