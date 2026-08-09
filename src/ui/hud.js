/**
 * hud.js — the readout, the settings panel and the cinematic tools.
 *
 * Design brief: the scene is the point. A HUD that announces itself destroys a
 * quiet landscape faster than any rendering bug, so this one is built to be
 * *noticed second*. Three surfaces, all of them optional:
 *
 *   1. READOUT (top-left)  A clock, the phase of the day, the weather, the wind.
 *      No boxes, no borders, no icons that need a legend. One hairline rule.
 *   2. PERF (top-right)    Two numbers at low opacity. Expands into a real
 *      per-system cost breakdown only when asked (state.debug.showStats).
 *   3. PANEL (Tab)         Everything you could want to change, arranged so you
 *      can find it, and out of the way until you do.
 *
 * Plus the things that make a viewer stay: photo mode, a slow automatic orbit of
 * the tree, PNG capture straight off the drawing buffer, and a twenty-second
 * time-lapse of a whole day.
 *
 * ---------------------------------------------------------------------------
 * PERFORMANCE — why this file looks paranoid
 * ---------------------------------------------------------------------------
 * update() runs inside the same 16.6 ms budget as the renderer, and a naive HUD
 * is genuinely capable of costing more than the grass:
 *
 *  - Writing `textContent` invalidates layout for the subtree. Doing it for six
 *    fields every frame is 360 forced layouts a second for text that changes a
 *    few times a minute. EVERY display write here is gated on the value actually
 *    having changed, and the comparisons themselves run on a timer (8 Hz for the
 *    readout, 3 Hz for perf, 4 Hz for the open panel) rather than per frame.
 *  - `backdrop-filter` on an always-visible element is a full-screen blur of
 *    whatever the GPU just rendered, every frame, forever. Only the settings
 *    panel gets glass, and only while it is open. LOW drops it entirely.
 *  - Anything that moves continuously (the wind vane, the gust meter, the
 *    time-lapse hairline, the cost bars) moves by `transform` only, so it stays
 *    on the compositor and never touches layout.
 *  - No allocation in update(): vector/quaternion scratch is module scope, and
 *    the only strings built are ones being written to the DOM anyway.
 *
 * ---------------------------------------------------------------------------
 * STATE OWNERSHIP
 * ---------------------------------------------------------------------------
 * This is a settings panel, not a simulation system. Where an owner exposes an
 * API it is used (quality.set / quality.setAdaptive / quality.setResolutionScale
 * / dayNight.setTimeOfDay / dayNight.applyPreset / weather.setWeather). Two
 * deliberate exceptions, both sanctioned:
 *
 *  - `state.quality.*` effect flags. core/quality.js's `_apply()` writes only
 *    the keys in its APPLIED_KEYS list on a tier switch and its
 *    `_detectExternalEdits()` watches only `adaptive` and `resolutionScale`, so
 *    the effect booleans are unclaimed between tier switches. We write the flag
 *    and then emit EVENTS.QUALITY_CHANGED, which main.js fans out to every
 *    system's onQualityChange() — the same delivery path a tier switch uses, so
 *    consumers need no second code path.
 *  - `state.wind.strength`, and only once the user has switched the wind off
 *    "Auto". weather/wind.js recomputes strength every frame from the weather
 *    profile with a slow damp, so a one-shot write would decay away; the manual
 *    pin is re-asserted each frame (a sub-1% steady-state error against that
 *    damp, invisible) and stops the instant Auto comes back on. If WindField
 *    ever grows a real setter we prefer it — see `_applyWind()`.
 *
 * `state.debug.hideUI` and `state.debug.showStats` are owned here and honoured
 * here: set either from anywhere and the HUD reacts on the next frame.
 *
 * ---------------------------------------------------------------------------
 * HONESTY
 * ---------------------------------------------------------------------------
 * A switch that does nothing is worse than no switch. Two gates outside this
 * file can silently swallow an effect toggle, and both are surfaced rather than
 * hidden: post/pipeline.js refuses several effects below a given tier (its
 * TIERS `allow*` flags, mirrored by EFFECT_FLOOR below), and it force-disables
 * ambient occlusion outright behind a module-private constant. The AO case
 * cannot be imported, so it is *observed* — see `_probeAO()`. Either way the
 * row disables itself and says why.
 *
 * Markup/class contract for this file and for loading.js lives at the top of
 * ./styles.css.
 */

import { Vector3, Quaternion, Matrix4 } from 'three';
import { clamp01, damp, mod, RAD2DEG } from '../core/math.js';
import { EVENTS } from '../core/state.js';
import { WEATHER_LABELS, WEATHER_ORDER } from '../weather/weather.js';
import { TIER_ORDER, TIER_PRESETS } from '../core/quality.js';

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

/** Seconds after the first frame before the HUD fades in, so the title card has
 *  finished leaving before anything else arrives. Pairs with .hud.is-booting. */
const REVEAL_DELAY = 0.7;

/** Readout refresh period. The clock ticks once per in-game minute; 8 Hz is an
 *  order of magnitude more often than anything on this surface can change. */
const READOUT_PERIOD = 1 / 8;
const READOUT_PERIOD_LOW = 1 / 4;
/** Frame counters are noisy — a slower refresh reads calmer and costs less. */
const PERF_PERIOD = 1 / 3;
/** How often an open panel re-reads the world to keep its controls honest. */
const PANEL_SYNC_PERIOD = 1 / 4;

/** Real seconds for the time-lapse to cover 24 in-game hours. */
const TIMELAPSE_SECONDS = 20;

/** Controls hint: seconds of sustained movement that dismisses it, the hard
 *  timeout for a viewer who never touches the keyboard, and the speed² above
 *  which the player counts as "moving" (0.6 m/s — slower than a walk). */
const HINT_MOVE_SECONDS = 0.8;
const HINT_TIMEOUT = 16;
const HINT_SPEED_SQ = 0.35;

/** Orbit: radians/second — one revolution every ~140 s. Slow enough that a
 *  still frame grabbed at any moment is composed rather than smeared. */
const ORBIT_RATE = 0.045;
/** Exponential rate of the eased hand-over between player camera and orbit. */
const ORBIT_BLEND_LAMBDA = 1.6;

/** Toast dwell, ms. Must outlast the CSS fade. */
const TOAST_MS = 2600;
/** Panel close animation, ms. Must match the .panel transition in styles.css. */
const PANEL_CLOSE_MS = 260;

/** Wind vane is re-rotated only when the heading moves this far, in degrees. */
const ARROW_EPSILON_DEG = 2;
/** Top of the gust meter, m/s. The storm profile peaks around 10 plus gusts. */
const WIND_METER_MAX = 14;

/** How near a named moment the clock must be for that chip to read as current. */
const PRESET_MATCH_HOURS = 0.35;

const HOURS = 24;

/** Tier ordinal, so "is this tier at least X" is a comparison and not a search. */
const TIER_RANK = Object.freeze({ low: 0, medium: 1, high: 2, ultra: 3 });

/**
 * Lowest tier at which each effect switch actually reaches something.
 *
 * These mirror the `allow*` flags in post/pipeline.js's TIERS table — that file
 * is owned by someone else and its table is not exported, so this is a copy and
 * is marked as one. It is used for LABELLING ONLY: the flag is still written and
 * announced exactly as before, so if the pipeline's policy changes the worst
 * this can do is describe a row slightly wrongly, never break one.
 *
 * `bloom` and `shadows` are honoured at every tier (bloom by the merged pass,
 * shadows by core/engine.js), so they floor at 0.
 */
const EFFECT_FLOOR = Object.freeze({
  ao: TIER_RANK.medium,
  godRays: TIER_RANK.medium,
  depthOfField: TIER_RANK.high,
  motionBlur: TIER_RANK.high,
  bloom: 0,
  shadows: 0,
});

/** Frames to let the post stack configure itself before believing what we see. */
const PROBE_FRAME = 4;

// ---------------------------------------------------------------------------
// Module scratch — update() must not allocate
// ---------------------------------------------------------------------------

const _playerPos = new Vector3();
const _playerQuat = new Quaternion();
const _orbitEye = new Vector3();
const _orbitTarget = new Vector3();
const _orbitQuat = new Quaternion();
const _lookMat = new Matrix4();
const _up = new Vector3(0, 1, 0);

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** 13.5 -> "13:30". */
function hourToClock(h) {
  const total = Math.floor(mod(h, HOURS) * 60 + 1e-6);
  const hh = Math.floor(total / 60) % 24;
  const mm = total % 60;
  return `${hh < 10 ? '0' : ''}${hh}:${mm < 10 ? '0' : ''}${mm}`;
}

/** Local date stamp for capture filenames. Never called per frame. */
function dateStamp() {
  const d = new Date();
  const p = (n) => (n < 10 ? '0' : '') + n;
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(
    d.getMinutes()
  )}${p(d.getSeconds())}`;
}

/** Shortest signed distance between two clock hours, in hours. */
function hourDistance(a, b) {
  return Math.abs(mod(a - b + 12, HOURS) - 12);
}

const SVG_NS = 'http://www.w3.org/2000/svg';

/** The wind vane. Drawn rather than typed so it can be rotated by transform. */
function windVane() {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '-12 -12 24 24');
  svg.setAttribute('width', '12');
  svg.setAttribute('height', '12');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  svg.setAttribute('class', 'wind__vane');
  const path = document.createElementNS(SVG_NS, 'path');
  // A slim dart, tip up. The tail notch is what lets a 12px glyph read as a
  // direction instead of as a triangle.
  path.setAttribute('d', 'M0,-10 L6.2,8 L0,4.4 L-6.2,8 Z');
  path.setAttribute('fill', 'currentColor');
  svg.appendChild(path);
  return svg;
}

// ---------------------------------------------------------------------------
// HUD
// ---------------------------------------------------------------------------

export class HUD {
  constructor(ctx) {
    this.ctx = ctx;
    this.state = ctx.state;
    this.bus = ctx.bus;
    this.input = ctx.input;
    this.camera = ctx.camera;
    this.canvas = ctx.canvas;
    this.quality = ctx.quality;
    this.root = ctx.uiRoot || document.body;

    // Siblings, resolved in link(). Everything guards for null so a stubbed or
    // failed sibling degrades one row of the panel instead of throwing on boot.
    this.dayNight = null;
    this.weather = null;
    this.wind = null;
    this.player = null;
    this.tree = null;
    this.post = null;
    this.terrain = null;

    // -- display caches ------------------------------------------------------
    this._cache = {
      clock: '',
      phase: '',
      weather: '',
      /** Wind speed in tenths of a m/s — a number, so no string is built to
       *  discover that nothing changed. */
      wind: -1,
      meter: -1,
      /** UNWRAPPED heading last written to the vane — see _updateReadout. */
      vaneDeg: 0,
      fps: -1,
      band: '',
      tier: '',
      scale: -1,
      draws: -1,
      tris: -1,
      cpu: -1,
      lapse: -1,
    };

    // -- timers --------------------------------------------------------------
    this._readoutAcc = 0;
    this._readoutPeriod = READOUT_PERIOD;
    this._perfAcc = 0;
    this._syncAcc = 0;
    this._age = 0;
    this._revealed = false;

    // -- modes ---------------------------------------------------------------
    this._panelOpen = false;
    this._panelCloseTimer = 0;
    this._photo = false;
    this._orbit = false;
    this._orbitBlend = 0;
    this._orbitAngle = 0;
    this._orbitTime = 0;
    this._orbitCenter = new Vector3(0, 6, 0);
    this._orbitRadius = 16;
    /** World Y of the trunk base. The framing is measured from here, not from
     *  y = 0 — world/terrain.js raises its clearing, so the two differ. */
    this._orbitBase = 0;
    this._hiddenApplied = null;
    /** Last seen value of state.debug.hideUI — tracked separately from
     *  `_hiddenApplied` because the two edges are not the same edge. */
    this._hideUIApplied = false;
    this._photoNoteTimer = 0;

    this._hintVisible = true;
    this._hintPinned = false;
    this._hintMoveTime = 0;
    this._hintAge = 0;

    this._lapse = { active: false, hours: 0, prevSpeed: 0, prevPaused: false, setSpeed: 0 };

    // -- wind override -------------------------------------------------------
    /** null = follow the weather. A number = the speed the user pinned. */
    this._windPin = null;
    /** True when WindField accepted a real setter, so no per-frame pin is due. */
    this._windSetter = false;

    // -- capture -------------------------------------------------------------
    this._captureArmed = false;
    this._captureTimer = 0;
    this._captureLink = document.createElement('a');
    this._captureLink.style.display = 'none';
    this._captureUrl = '';

    // -- ui focus gate -------------------------------------------------------
    this._pointerInPanel = false;
    this._focusInPanel = false;
    this._gateApplied = false;

    /** Panel control synchronisers, registered by the builders below. */
    this._sync = [];
    /** Control the pointer is currently dragging — never fight the user. */
    this._liveControl = null;
    this._bloomPinned = false;
    this._toastTimer = 0;
    this._sensBase = 0.0022;

    // -- effect availability -------------------------------------------------
    /** One record per Effects row; see _buildEffectRow / _refreshEffects. */
    this._effectRows = [];
    /**
     * Tri-state verdict on ambient occlusion. null = not observable yet,
     * true = post/pipeline.js really built the pass, false = it refused.
     * post/pipeline.js gates AO behind a module-private `AO_DISABLED` constant
     * that we cannot import, so the only honest answer comes from watching.
     */
    this._aoLive = null;

    this._reduced = !!(
      window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches
    );

    this._onKeyDown = this._onKeyDown.bind(this);
    this._onPointerUp = this._onPointerUp.bind(this);

    this._buildDOM();

    window.addEventListener('keydown', this._onKeyDown, true);
    window.addEventListener('pointerup', this._onPointerUp);

    this._offQuality = this.bus?.on(EVENTS.QUALITY_CHANGED, () => this._onQualityEvent());
    this._offWeather = this.bus?.on(EVENTS.WEATHER_CHANGED, () => this._syncPanel(true));

    // Bloom has no entry in the tier preset table (core/quality.js owns that and
    // we may not edit it). Seed the flag here so post/pipeline.js has something
    // to read from frame one, and give LOW a genuinely cheaper default.
    const q = this.state.quality;
    if (q.bloom === undefined) q.bloom = q.tier !== 'low';

    this.onQualityChange(q);
  }

  // =========================================================================
  // Construction
  // =========================================================================

  _buildDOM() {
    const root = el('div', 'hud is-booting');
    this.el = root;

    // ---- readout ----------------------------------------------------------
    const readout = el('section', 'hud__readout');
    readout.setAttribute('aria-label', 'World readout');

    const clockLine = el('div', 'readout__clock');
    this._elClock = el('span', 'readout__time', '--:--');
    this._elPhase = el('span', 'readout__phase', '');
    clockLine.append(this._elClock, this._elPhase);

    const rule = el('div', 'readout__rule');
    rule.setAttribute('aria-hidden', 'true');

    const facts = el('div', 'readout__facts');

    const skyRow = el('div', 'fact');
    this._elWeather = el('span', 'fact__val', '');
    skyRow.append(el('span', 'fact__key', 'Sky'), this._elWeather);

    const windRow = el('div', 'fact');
    const windVal = el('span', 'fact__val fact__val--wind');
    this._elVane = windVane();
    this._elWind = el('span', 'wind__value', '');
    windVal.append(this._elVane, this._elWind);
    windRow.append(el('span', 'fact__key', 'Wind'), windVal);

    const meter = el('div', 'wind__meter');
    meter.setAttribute('aria-hidden', 'true');
    this._elWindBar = el('i', 'wind__bar');
    meter.append(this._elWindBar);

    facts.append(skyRow, windRow, meter);
    readout.append(clockLine, rule, facts);

    // ---- perf -------------------------------------------------------------
    const perf = el('section', 'hud__perf');
    perf.setAttribute('aria-label', 'Performance');

    const perfLine = el('div', 'perf__line');
    this._elFps = el('span', 'perf__fps', '--');
    this._elTier = el('span', 'perf__tier', '');
    this._elScale = el('span', 'perf__scale', '');
    perfLine.append(
      this._elFps,
      el('span', 'perf__unit', 'fps'),
      el('span', 'perf__dot', '·'),
      this._elTier,
      this._elScale
    );

    this._elPerfDetail = el('div', 'perf__detail');
    this._elPerfDetail.hidden = true;
    // '--', not '0': these are only written once a real reading arrives, and
    // "0 draw calls" is a claim about the frame rather than an absence of one.
    // It also keeps the -1 "no reading" sentinel in _updatePerf from having to
    // fight the initial cache value, which is -1 for the same reason.
    this._elCpu = el('span', 'perf__num', '--');
    this._elDraws = el('span', 'perf__num', '--');
    this._elTris = el('span', 'perf__num', '--');
    const statRow = (label, node) => {
      const r = el('div', 'perf__row');
      r.append(el('span', 'perf__label', label), node);
      return r;
    };
    this._elCosts = el('div', 'perf__costs');
    this._elPerfDetail.append(
      statRow('cpu ms', this._elCpu),
      statRow('draws', this._elDraws),
      statRow('tris', this._elTris),
      this._elCosts
    );
    /** Fixed row pool — the cost table must not churn nodes at 3 Hz. */
    this._costRows = [];

    perf.append(perfLine, this._elPerfDetail);

    // ---- time-lapse hairline ---------------------------------------------
    const lapse = el('div', 'hud__lapse');
    lapse.setAttribute('aria-hidden', 'true');
    this._elLapseFill = el('i', 'lapse__fill');
    lapse.append(this._elLapseFill);
    this._elLapse = lapse;

    // ---- first-run hint ---------------------------------------------------
    const hint = el('div', 'hud__hint');
    hint.setAttribute('role', 'note');
    hint.innerHTML =
      '<span class="hint__row"><kbd>Click</kbd> to look around</span>' +
      '<span class="hint__row"><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd>' +
      ' <span class="hint__word">walk</span> &middot; <kbd>Shift</kbd>' +
      ' <span class="hint__word">run</span> &middot; <kbd>F</kbd>' +
      ' <span class="hint__word">fly</span></span>' +
      '<span class="hint__row hint__row--dim"><kbd>Tab</kbd> settings &middot;' +
      ' <kbd>P</kbd> photo &middot; <kbd>L</kbd> time-lapse &middot;' +
      ' <kbd>H</kbd> hide</span>';
    this._elHint = hint;

    // ---- photo-mode chrome ------------------------------------------------
    const photo = el('div', 'hud__photo');
    photo.setAttribute('aria-hidden', 'true');
    photo.innerHTML =
      '<i class="photo__tick photo__tick--tl"></i><i class="photo__tick photo__tick--tr"></i>' +
      '<i class="photo__tick photo__tick--bl"></i><i class="photo__tick photo__tick--br"></i>';
    this._elPhotoNote = el('div', 'photo__note');
    this._elPhotoNote.innerHTML =
      '<kbd>F2</kbd> capture &middot; <kbd>O</kbd> orbit &middot; <kbd>P</kbd> exit';
    photo.append(this._elPhotoNote);
    this._elPhoto = photo;

    // ---- toast ------------------------------------------------------------
    // Deliberately outside the hide-everything rule: an explicit action taken in
    // photo mode still deserves an acknowledgement, and the canvas capture does
    // not include the DOM anyway.
    const toast = el('div', 'hud__toast');
    toast.setAttribute('role', 'status');
    toast.setAttribute('aria-live', 'polite');
    this._elToast = toast;

    root.append(readout, perf, lapse, hint, photo, toast, this._buildPanel());
    this.root.appendChild(root);
    this.root.appendChild(this._captureLink);
  }

  // -------------------------------------------------------------------------
  // Panel scaffolding
  // -------------------------------------------------------------------------

  _buildPanel() {
    const panel = el('aside', 'panel');
    panel.hidden = true;
    panel.id = 'sakura-settings';
    panel.tabIndex = -1;
    panel.setAttribute('aria-label', 'Settings');
    this._elPanel = panel;

    // The head echoes the title card: the same serif, the same tracked kana
    // sitting quietly under the name. Two words, no chrome.
    const head = el('header', 'panel__head');
    const heading = el('div', 'panel__heading');
    const kicker = el('span', 'panel__kicker', '設定');
    kicker.setAttribute('lang', 'ja');
    kicker.setAttribute('aria-hidden', 'true');
    heading.append(el('h2', 'panel__title', 'Settings'), kicker);

    const close = el('button', 'panel__close', 'Esc');
    close.type = 'button';
    close.setAttribute('aria-label', 'Close settings');
    close.addEventListener('click', () => this.setPanelOpen(false));
    head.append(heading, close);

    const body = el('div', 'panel__body');
    this._elPanelBody = body;

    const foot = el('footer', 'panel__foot');
    foot.append(
      el('span', 'panel__note', 'Only the quality tier is remembered'),
      // "Tab closes" was a lie once focus is inside the sheet: _trapTab keeps Tab
      // cycling the panel's own controls, and Esc is the only way out. A settings
      // panel whose whole premise is that a control never lies about itself does
      // not get to lie in its own footer.
      el('span', 'panel__note panel__note--dim', 'Esc closes · Tab cycles')
    );

    panel.append(head, body, foot);

    // The focus gate: WASD must not leak into the player controller while the
    // pointer is over the panel or a control holds keyboard focus.
    panel.addEventListener('pointerenter', () => {
      this._pointerInPanel = true;
      this._refreshGate();
    });
    panel.addEventListener('pointerleave', () => {
      this._pointerInPanel = false;
      this._refreshGate();
    });
    // pointerenter only fires when the pointer CROSSES the boundary. Opening the
    // panel underneath a stationary cursor — which is what happens every time
    // someone reaches for Tab without moving the mouse — never crosses anything,
    // so the gate would stay open and the first WASD press would walk the player
    // away behind the sheet. The first move over the panel repairs it. Guarded,
    // so the steady state is one boolean test per pointermove and nothing else.
    panel.addEventListener('pointermove', () => {
      if (this._pointerInPanel) return;
      this._pointerInPanel = true;
      this._refreshGate();
    });
    panel.addEventListener('focusin', () => {
      this._focusInPanel = true;
      this._refreshGate();
    });
    panel.addEventListener('focusout', () => {
      // focusout fires before the next element receives focus; defer the read.
      requestAnimationFrame(() => {
        this._focusInPanel = !!(this._elPanel && this._elPanel.contains(document.activeElement));
        this._refreshGate();
      });
    });
    // A slider drag that wanders outside the panel must not be re-synced from
    // under the user's thumb; a global pointerup releases the claim.
    panel.addEventListener('pointerdown', (e) => {
      const t = e.target;
      if (t && t.tagName === 'INPUT') this._liveControl = t;
    });

    return panel;
  }

  _group(title) {
    const g = el('section', 'group');
    g.append(el('h3', 'group__title', title));
    this._elPanelBody.appendChild(g);
    return g;
  }

  /**
   * Label left, control right.
   *
   * The hint span is created for `''` as well as for real text — a row whose
   * caption can change later (the effect switches say why they are unavailable)
   * needs the node to exist from the start so no layout ever has to be inserted.
   * Pass `null`/`undefined` for a row whose caption is permanently absent.
   * `.row__hint:empty` is display:none, so an empty span costs nothing.
   */
  _row(group, label, hint) {
    const r = el('div', 'row');
    const l = el('div', 'row__label');
    l.append(el('span', 'row__name', label));
    let hintEl = null;
    if (hint !== null && hint !== undefined) {
      hintEl = el('span', 'row__hint', hint);
      l.append(hintEl);
    }
    const c = el('div', 'row__control');
    r.append(l, c);
    group.appendChild(r);
    return { row: r, label: l, control: c, hintEl };
  }

  /** Label on its own line, control full width beneath. */
  _block(group, label, hint) {
    const { row, label: l, control, hintEl } = this._row(group, label, hint);
    row.classList.add('row--block');
    return { row, label: l, control, hintEl };
  }

  /**
   * Segmented radio group.
   *
   * Roving tabindex: exactly one chip per group is in the tab ring, and the
   * arrow keys move between them. Without it, Tab through this panel is 30-odd
   * stops — seven of them just to cross the weather strip — and the focus ring
   * jumps around a control that reads visually as one object.
   *
   * Deliberate deviation from the ARIA radiogroup pattern: arrowing moves focus
   * but does NOT select. Selection here is not free — a tier chip rebuilds the
   * post stack and every shadow cascade — and arrowing across four of them to
   * reach the one you want would rebuild three times on the way. Space/Enter
   * commits, which is the "manual activation" variant of the same pattern.
   *
   * @param {() => string} read live value; '' selects nothing
   * @param {(v:string) => void} write
   */
  _options(host, items, read, write, extraClass) {
    const wrap = el('div', `opts${extraClass ? ' ' + extraClass : ''}`);
    wrap.setAttribute('role', 'radiogroup');
    const buttons = [];
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      const b = el('button', 'opt', item.label);
      b.type = 'button';
      b.setAttribute('role', 'radio');
      b.setAttribute('aria-checked', 'false');
      b.tabIndex = i === 0 ? 0 : -1;
      b.dataset.value = item.value;
      if (item.title) b.title = item.title;
      b.addEventListener('click', () => write(item.value));
      buttons.push(b);
      wrap.appendChild(b);
    }

    wrap.addEventListener('keydown', (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      let next = -1;
      const cur = buttons.indexOf(document.activeElement);
      switch (e.key) {
        case 'ArrowRight':
        case 'ArrowDown':
          next = mod((cur < 0 ? -1 : cur) + 1, buttons.length);
          break;
        case 'ArrowLeft':
        case 'ArrowUp':
          next = mod((cur < 0 ? 1 : cur) - 1, buttons.length);
          break;
        case 'Home':
          next = 0;
          break;
        case 'End':
          next = buttons.length - 1;
          break;
        default:
          return;
      }
      e.preventDefault();
      // The window-level handler runs in the capture phase and would otherwise
      // never see this, but stopping here also keeps the arrows off the page.
      e.stopPropagation();
      // Move the tab stop with the focus, immediately. Leaving it on the
      // selected chip would put focus on a tabIndex -1 element, and the panel's
      // Tab trap only recognises elements that are in the tab ring — the very
      // next Tab would walk straight out of the sheet into the browser chrome.
      for (let i = 0; i < buttons.length; i++) {
        const want = i === next ? 0 : -1;
        if (buttons[i].tabIndex !== want) buttons[i].tabIndex = want;
      }
      buttons[next].focus({ preventScroll: true });
    });

    host.appendChild(wrap);
    this._sync.push(() => {
      const v = read();
      let checked = -1;
      for (let i = 0; i < buttons.length; i++) {
        const b = buttons[i];
        const on = b.dataset.value === v;
        if (on) checked = i;
        if ((b.getAttribute('aria-checked') === 'true') !== on) {
          b.setAttribute('aria-checked', on ? 'true' : 'false');
        }
      }
      // The tab stop follows the selection — unless the keyboard is already
      // parked on a different chip, in which case moving it out from under the
      // user would drop them out of the tab ring mid-decision. With nothing
      // selected at all (the Moments strip whenever the clock is between two
      // named hours) the first chip keeps the group reachable.
      const focused = buttons.indexOf(document.activeElement);
      const stop = focused >= 0 ? focused : checked < 0 ? 0 : checked;
      for (let i = 0; i < buttons.length; i++) {
        const want = i === stop ? 0 : -1;
        if (buttons[i].tabIndex !== want) buttons[i].tabIndex = want;
      }
    });
    return wrap;
  }

  /** Momentary actions, laid out like the option chips. */
  _actions(host, items) {
    const wrap = el('div', 'opts opts--actions');
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      const b = el('button', 'opt', item.label);
      b.type = 'button';
      if (item.title) b.title = item.title;
      b.addEventListener('click', item.onClick);
      if (item.ref) this[item.ref] = b;
      wrap.appendChild(b);
    }
    host.appendChild(wrap);
    return wrap;
  }

  /** @returns {{button: HTMLButtonElement, row: HTMLElement, hintEl: HTMLElement|null}} */
  _switch(group, label, hint, read, write) {
    const { row, control, hintEl } = this._row(group, label, hint);
    const b = el('button', 'switch');
    b.type = 'button';
    b.setAttribute('role', 'switch');
    b.setAttribute('aria-checked', 'false');
    b.setAttribute('aria-label', label);
    b.innerHTML = '<i class="switch__track"><i class="switch__knob"></i></i>';
    b.addEventListener('click', () => {
      if (b.disabled) return;
      write(!read());
      this._syncPanel(true);
    });
    control.appendChild(b);
    this._sync.push(() => {
      const on = !!read();
      if ((b.getAttribute('aria-checked') === 'true') !== on) {
        b.setAttribute('aria-checked', on ? 'true' : 'false');
      }
    });
    return { button: b, row, hintEl };
  }

  /**
   * Range slider with a live value readout.
   * @param {{min:number,max:number,step:number}} opts
   * @param {(v:number)=>string} format
   */
  _slider(group, label, hint, opts, read, write, format) {
    const { row, label: labelEl, control } = this._block(group, label, hint);
    const input = document.createElement('input');
    input.type = 'range';
    input.className = 'slider';
    input.min = String(opts.min);
    input.max = String(opts.max);
    input.step = String(opts.step);
    input.setAttribute('aria-label', label);
    const out = el('output', 'slider__value', '');
    // <output> carries an implicit role="status" — i.e. it IS a polite live
    // region, with no attribute here saying so. Two of the five readouts in this
    // panel track a number the world moves on its own (Render scale under the
    // adaptive controller, Wind speed while the weather owns it), and the 4 Hz
    // panel sync rewrites them: measured at ~29 rewrites per 400 frames with the
    // panel merely open and untouched, which a screen reader would read aloud,
    // four times a second, for as long as the panel is on screen. The visible
    // number is a duplicate of the control's own value; `aria-valuetext` below is
    // the channel that is supposed to carry it, and it announces on change only.
    out.setAttribute('aria-hidden', 'true');
    labelEl.appendChild(out);

    let lastText = '';
    let lastFill = -1;
    const paint = (v) => {
      const text = format(v);
      if (text !== lastText) {
        lastText = text;
        out.textContent = text;
        // A range input announces its raw number by default, which for the
        // clock is "13.52" and for day speed is a bare float. The formatted
        // string is the one the sighted user reads; give it to everyone.
        input.setAttribute('aria-valuetext', text);
      }
      // Filled portion of the track, handed to CSS as a custom property so the
      // browser paints it in the track's own gradient rather than as a layer.
      const f = clamp01((v - opts.min) / (opts.max - opts.min || 1));
      if (Math.abs(f - lastFill) > 0.002) {
        lastFill = f;
        input.style.setProperty('--fill', (f * 100).toFixed(1) + '%');
      }
    };

    input.addEventListener('input', () => {
      const v = parseFloat(input.value);
      if (!Number.isFinite(v)) return;
      write(v);
      // Paint what actually landed, not what was asked for. Owners clamp and
      // snap: quality.setResolutionScale rides a fixed ladder and refuses to go
      // below the tier's floor, and dayNight wraps 24:00 round to 00:00. Without
      // this the number under the thumb would show a value the world rejected
      // until the 4 Hz sync quietly corrected it a quarter of a second later.
      const landed = read();
      if (!Number.isFinite(landed)) {
        paint(v);
        return;
      }
      paint(landed);
      // A refusal larger than one step means the owner really did move the
      // value somewhere else — take the thumb with it rather than leaving it
      // parked under the cursor at a setting that is not in effect. Every
      // slider here is monotonic over its whole range (the clock stops at 23:59
      // precisely so it cannot wrap), so this can only ever be a clamp.
      if (Math.abs(landed - v) > (opts.step || 0.01)) input.value = String(landed);
    });

    control.appendChild(input);
    this._sync.push(() => {
      if (input === document.activeElement || input === this._liveControl) return;
      const v = read();
      if (!Number.isFinite(v)) return;
      const shown = parseFloat(input.value);
      if (Math.abs(shown - v) > (opts.step || 0.01) * 0.49) input.value = String(v);
      paint(v);
    });
    row.classList.add('row--slider');
    return input;
  }

  // -------------------------------------------------------------------------
  // Effect switches — the only controls that can be overruled from outside
  // -------------------------------------------------------------------------

  /**
   * One row of the Effects group, plus the bookkeeping that lets it tell the
   * truth about itself.
   *
   * `read()` reports the flag AND the availability, so an overruled effect
   * shows OFF rather than showing ON while doing nothing — which is the exact
   * failure this whole mechanism exists to prevent.
   */
  _buildEffectRow(group, label, key, hint) {
    const q = this.state.quality;
    const rec = {
      key,
      hint,
      /** Latest reason the row is unavailable; '' when it works. */
      reason: '',
      shownHint: hint,
      off: false,
      button: null,
      row: null,
      hintEl: null,
      floor: EFFECT_FLOOR[key] || 0,
    };
    const parts = this._switch(
      group,
      label,
      hint,
      () => !!q[key] && rec.reason === '',
      (v) => this._setEffect(key, v)
    );
    rec.button = parts.button;
    rec.row = parts.row;
    rec.hintEl = parts.hintEl;
    this._effectRows.push(rec);
    return rec;
  }

  /**
   * Decide, once, whether ambient occlusion is real in this build.
   *
   * post/pipeline.js keeps `AO_DISABLED` module-private, so there is nothing to
   * import and nothing to ask. What there is: the pipeline allocates `aoPass`
   * if and only if it actually intends to run AO. So when every gate we *can*
   * read says the pass should exist and it does not, AO has been overruled.
   *
   * Only conclusive while the flag is on and the tier permits it, and only once
   * the stack has had a few frames to configure itself — before that, a null
   * pass means "not built yet", not "refused". Inconclusive leaves the verdict
   * at null and the row alone, so a future build that re-enables AO is described
   * correctly with no change here.
   */
  _probeAO() {
    if (this._aoLive !== null) return;
    const post = this.post;
    if (!post || this.state.time.frame < PROBE_FRAME) return;
    const q = this.state.quality;
    if (!q.ao) return;
    if ((TIER_RANK[q.tier] ?? TIER_RANK.high) < EFFECT_FLOOR.ao) return;
    this._aoLive = !!post.aoPass;
  }

  /**
   * Re-label and enable/disable every effect row. Cheap enough for the 4 Hz
   * panel sync: six rows, and every DOM write is gated on a change.
   */
  _refreshEffects() {
    if (this._effectRows.length === 0) return;
    this._probeAO();
    const rank = TIER_RANK[this.state.quality.tier] ?? TIER_RANK.high;

    for (let i = 0; i < this._effectRows.length; i++) {
      const r = this._effectRows[i];
      let reason = '';
      if (rank < r.floor) {
        const need = TIER_PRESETS[TIER_ORDER[r.floor]];
        reason = `Needs ${need ? need.label : 'a higher tier'} or above`;
      } else if (r.key === 'ao' && this._aoLive === false) {
        reason = 'Off in this build — it lifts the black point';
      }

      if (reason !== r.reason) {
        r.reason = reason;
        const off = reason !== '';
        r.off = off;
        r.button.disabled = off;
        r.button.setAttribute('aria-disabled', off ? 'true' : 'false');
        r.row.classList.toggle('is-unavailable', off);
      }
      const text = reason || r.hint;
      if (text !== r.shownHint) {
        r.shownHint = text;
        if (r.hintEl) r.hintEl.textContent = text;
      }
    }
  }

  // -------------------------------------------------------------------------
  // Panel content — built in link(), once every sibling exists
  // -------------------------------------------------------------------------

  _populatePanel() {
    const state = this.state;
    const q = state.quality;

    // ---- Quality ----------------------------------------------------------
    const gQuality = this._group('Quality');
    {
      const { control } = this._block(gQuality, 'Tier', 'Low is genuinely cheaper');
      this._options(
        control,
        TIER_ORDER.map((t) => ({
          value: t,
          label: TIER_PRESETS[t].label,
          title: `${TIER_PRESETS[t].label} — targets ${TIER_PRESETS[t].targetFps} fps`,
        })),
        () => q.tier,
        (v) => {
          if (this.quality) this.quality.set(v);
          else q.tier = v;
          this._syncPanel(true);
        }
      );
    }

    this._switch(
      gQuality,
      'Adaptive resolution',
      'Trades pixels for framerate',
      () => q.adaptive,
      (v) => {
        if (this.quality) this.quality.setAdaptive(v);
        else q.adaptive = v;
      }
    );

    this._sliderScale = this._slider(
      gQuality,
      'Render scale',
      null,
      { min: 0.55, max: 1, step: 0.05 },
      () => q.resolutionScale,
      (v) => {
        if (this.quality) this.quality.setResolutionScale(v);
        else q.resolutionScale = v;
      },
      (v) => `${Math.round(v * 100)}%`
    );

    // ---- Effects ----------------------------------------------------------
    const gFx = this._group('Effects');
    this._buildEffectRow(gFx, 'Ambient occlusion', 'ao', 'Contact shade in the grass');
    this._buildEffectRow(gFx, 'Bloom', 'bloom', 'Light bleeding off highlights');
    this._buildEffectRow(gFx, 'God rays', 'godRays', 'Shafts through the canopy');
    this._buildEffectRow(gFx, 'Depth of field', 'depthOfField', 'Expensive at 1080p');
    this._buildEffectRow(gFx, 'Motion blur', 'motionBlur', 'Smears fast turns');
    this._buildEffectRow(gFx, 'Shadows', 'shadows', 'Cascaded, from the sun and moon');

    // ---- Time -------------------------------------------------------------
    const gTime = this._group('Time of day');
    this._slider(
      gTime,
      'Clock',
      null,
      // Stops one minute short of midnight on purpose: 24.00 wraps to 0.00 in
      // DayNightCycle, and a thumb that teleports from the far right to the far
      // left mid-drag reads as a bug however correct it is. 00:00 is at min.
      { min: 0, max: 23.99, step: 0.01 },
      () => (this.dayNight ? this.dayNight.getTimeOfDay() : state.time.timeOfDay),
      (v) => {
        this._cancelLapse(false);
        if (this.dayNight) this.dayNight.setTimeOfDay(v, { adapt: 'snap' });
        else state.time.timeOfDay = v;
      },
      hourToClock
    );

    this._slider(
      gTime,
      'Day speed',
      'In-game hours per real second',
      { min: 0, max: 3, step: 0.01 },
      () => state.time.daySpeed,
      (v) => {
        this._cancelLapse(false);
        if (this.dayNight) this.dayNight.setDaySpeed(v);
        else state.time.daySpeed = v;
      },
      (v) => (v < 0.005 ? 'frozen' : `${v.toFixed(2)} h/s`)
    );

    this._switch(
      gTime,
      'Hold the sun still',
      null,
      () => state.time.paused,
      (v) => {
        this._cancelLapse(false);
        if (this.dayNight) this.dayNight.setPaused(v);
        else state.time.paused = v;
      }
    );

    if (this.dayNight && typeof this.dayNight.getPresets === 'function') {
      const { control } = this._block(gTime, 'Moments', 'Eased transition, not a cut');
      // getPresets() is cached by DayNightCycle and only rebuilt if the site or
      // season moves, so holding the array is safe and reading it is free.
      const presets = this.dayNight.getPresets();
      this._options(
        control,
        presets.map((p) => ({
          value: p.key,
          label: p.label,
          title: `${p.label} — ${hourToClock(p.hour)}`,
        })),
        () => {
          // Light a chip only when the clock is genuinely standing on it;
          // "nearest of eight" would leave one lit at every hour of the day.
          const now = this.dayNight.getTimeOfDay();
          let bestKey = '';
          let bestDist = Infinity;
          for (let i = 0; i < presets.length; i++) {
            const d = hourDistance(presets[i].hour, now);
            if (d < bestDist) {
              bestDist = d;
              bestKey = presets[i].key;
            }
          }
          return bestDist <= PRESET_MATCH_HOURS ? bestKey : '';
        },
        (key) => {
          this._cancelLapse(false);
          this.dayNight.applyPreset(key, {
            transition: this._reduced ? 0 : 1.6,
            direction: 'shortest',
            adapt: 'fast',
          });
          this._syncPanel(true);
        },
        'opts--wrap'
      );
    }

    {
      const { control } = this._block(gTime, 'Time-lapse', `A whole day in ${TIMELAPSE_SECONDS}s`);
      this._actions(control, [
        {
          label: 'Run a whole day',
          ref: '_btnLapse',
          title: 'L',
          onClick: () => this.toggleTimelapse(),
        },
      ]);
    }

    // ---- Weather ----------------------------------------------------------
    const gWeather = this._group('Weather');
    {
      const { control } = this._block(gWeather, 'Conditions', 'Blends over eight seconds');
      this._options(
        control,
        WEATHER_ORDER.map((w) => ({ value: w, label: WEATHER_LABELS[w] })),
        () => state.weather.target,
        (v) => {
          if (this.weather) this.weather.setWeather(v, 8);
          else state.weather.target = v;
          this._syncPanel(true);
        },
        'opts--wrap'
      );
    }

    if (this.weather) {
      this._switch(
        gWeather,
        'Let the weather drift',
        'Autonomous, over minutes',
        () => !!this.weather.auto,
        (v) => this.weather.setAuto(v)
      );
      const { control } = this._block(gWeather, 'Poke', null);
      this._actions(control, [
        { label: 'Lightning', onClick: () => this.weather.triggerLightning(1) },
      ]);
    }

    // ---- Wind -------------------------------------------------------------
    const gWind = this._group('Wind');
    this._switch(
      gWind,
      'Follow the weather',
      'Off pins the speed by hand',
      () => this._windPin === null,
      (v) => {
        if (v) {
          this._windPin = null;
          this._applyWind(null);
        } else {
          this._windPin = state.wind.strength;
          this._applyWind(this._windPin);
        }
      }
    );
    this._sliderWind = this._slider(
      gWind,
      'Wind speed',
      null,
      { min: 0, max: WIND_METER_MAX, step: 0.1 },
      () => (this._windPin === null ? state.wind.strength : this._windPin),
      (v) => {
        this._windPin = v;
        this._applyWind(v);
      },
      (v) => `${v.toFixed(1)} m/s`
    );

    // ---- Camera -----------------------------------------------------------
    if (this.player) {
      const gCam = this._group('Camera');
      const base = this._sensBase;
      this._slider(
        gCam,
        'Look sensitivity',
        null,
        { min: 0.35, max: 2.5, step: 0.05 },
        () => (base > 0 ? this.player.sensitivity / base : 1),
        (v) => {
          this.player.sensitivity = base * v;
        },
        (v) => `${v.toFixed(2)}×`
      );
      this._switch(
        gCam,
        'Head bob',
        null,
        () => !!this.player.headBob,
        (v) => {
          this.player.headBob = v;
        }
      );
      this._switch(
        gCam,
        'Invert look',
        null,
        () => !!this.player.invertY,
        (v) => {
          this.player.invertY = v;
        }
      );
    }

    // ---- Cinematic --------------------------------------------------------
    const gShot = this._group('Cinematic');
    this._switch(
      gShot,
      'Photo mode',
      'Hides every overlay',
      () => this._photo,
      (v) => this.setPhotoMode(v)
    );
    this._switch(
      gShot,
      'Automatic orbit',
      'A slow circle of the tree',
      () => this._orbit,
      (v) => this.setOrbit(v)
    );
    {
      const { control } = this._block(gShot, 'Capture', 'PNG, at render resolution');
      this._actions(control, [
        { label: 'Take a photo', title: 'F2', onClick: () => this.capture() },
      ]);
    }

    // ---- Display ----------------------------------------------------------
    const gDisplay = this._group('Display');
    this._switch(
      gDisplay,
      'Performance detail',
      'Per-system cost breakdown',
      () => state.debug.showStats,
      (v) => {
        state.debug.showStats = !!v;
        this._applyStats();
      }
    );
    this._switch(
      gDisplay,
      'Show the readout',
      null,
      () => !state.debug.hideUI,
      (v) => {
        state.debug.hideUI = !v;
      }
    );
    {
      const { control } = this._block(gDisplay, 'Controls', null);
      this._actions(control, [{ label: 'Show the key list', onClick: () => this.showHint(true) }]);
    }

    // ---- Keys -------------------------------------------------------------
    const gKeys = this._group('Keys');
    const keys = el('dl', 'keys');
    const KEYMAP = [
      ['W A S D', 'Walk — hold Shift to run'],
      ['Space C', 'Jump and crouch (fly: up and down)'],
      ['F', 'Toggle walking and flying'],
      ['Tab', 'Settings'],
      ['H', 'Hide or show the readout'],
      ['P', 'Photo mode'],
      ['O', 'Automatic orbit'],
      ['L', 'Time-lapse a whole day'],
      ['F2', 'Capture a photo'],
      ['Esc', 'Close the panel, leave photo mode'],
    ];
    for (let i = 0; i < KEYMAP.length; i++) {
      const dt = el('dt', 'keys__key');
      dt.innerHTML = KEYMAP[i][0]
        .split(' ')
        .map((k) => `<kbd>${k}</kbd>`)
        .join(' ');
      keys.append(dt, el('dd', 'keys__desc', KEYMAP[i][1]));
    }
    gKeys.appendChild(keys);
  }

  // =========================================================================
  // Lifecycle
  // =========================================================================

  link(systems) {
    this.dayNight = systems?.dayNight ?? null;
    this.weather = systems?.weather ?? null;
    this.wind = systems?.wind ?? null;
    this.player = systems?.player ?? null;
    this.tree = systems?.tree ?? null;
    this.post = systems?.post ?? null;
    this.terrain = systems?.terrain ?? null;

    // Whatever the controller shipped with is 1.00x on the sensitivity slider.
    if (this.player && this.player.sensitivity > 0) this._sensBase = this.player.sensitivity;

    this._refreshOrbitRig();

    this._populatePanel();
    this._applyStats();
    this._syncPanel(true);
  }

  resize(width, height) {
    // Short or narrow viewports cannot carry the perf breakdown and the full
    // hint without crowding the frame; the stylesheet keys off this one class.
    this.el?.classList.toggle('is-tight', height < 560 || width < 700);
  }

  onQualityChange(quality) {
    const tier = (quality && quality.tier) || this.state.quality.tier;
    const low = tier === 'low';
    // LOW: half the readout refresh rate, and no backdrop blur behind the panel.
    // A full-screen blur of the framebuffer is the single most expensive thing a
    // HUD can ask an integrated GPU for, and on LOW that budget is already spent.
    this._readoutPeriod = low ? READOUT_PERIOD_LOW : READOUT_PERIOD;
    this.el?.classList.toggle('is-cheap', low);
  }

  dispose() {
    this._cancelLapse(true);
    this.setOrbit(false);
    if (this._gateApplied && this.input) this.input.uiHasFocus = false;

    window.removeEventListener('keydown', this._onKeyDown, true);
    window.removeEventListener('pointerup', this._onPointerUp);
    this._offQuality?.();
    this._offWeather?.();

    if (this._toastTimer) clearTimeout(this._toastTimer);
    if (this._panelCloseTimer) clearTimeout(this._panelCloseTimer);
    if (this._captureTimer) clearTimeout(this._captureTimer);
    this._captureArmed = false;
    if (this._captureUrl) {
      URL.revokeObjectURL(this._captureUrl);
      this._captureUrl = '';
    }
    this._captureLink.remove();
    this.el?.remove();
    this._sync.length = 0;
    this._costRows.length = 0;
    this._effectRows.length = 0;
  }

  // =========================================================================
  // Frame
  // =========================================================================

  update(dt, state) {
    const root = this.el;
    if (!root) return;

    this._age += dt;
    if (!this._revealed && this._age >= REVEAL_DELAY) {
      this._revealed = true;
      root.classList.remove('is-booting');
    }

    // Visibility is a two-line boolean, but the class write must not happen
    // every frame — toggling a class dirties style for the whole subtree.
    const hideUI = !!state.debug.hideUI;
    const hidden = hideUI || this._photo;
    if (hidden !== this._hiddenApplied) {
      this._hiddenApplied = hidden;
      root.classList.toggle('is-hidden', hidden);
      // Coming back into view must not wait up to 125 ms for stale text.
      if (!hidden) {
        this._readoutAcc = this._readoutPeriod;
        this._perfAcc = PERF_PERIOD;
      }
    }

    // "Hide the UI" means all of it, panel included — and this has to watch
    // hideUI's OWN edge, not `hidden`'s. `hidden` is a disjunction: with photo
    // mode already on it is true before H is pressed and true after, so hanging
    // the close off it would silently do nothing in exactly the case where
    // someone is trying to clear the screen for a shot.
    //
    // Photo mode alone deliberately does NOT close the panel: it is the surface
    // the photo-mode switch lives on, and closing it would leave the only way
    // back out on the keyboard. The capture reads the WebGL drawing buffer,
    // which never contains any DOM, so an open panel cannot reach the shot.
    if (hideUI !== this._hideUIApplied) {
      this._hideUIApplied = hideUI;
      if (hideUI && this._panelOpen) this.setPanelOpen(false);
    }

    this._updateTimelapse(dt, state);
    this._updateOrbit(dt);
    this._updateHint(dt, state);
    this._updatePhotoNote(dt);

    if (!hidden) {
      this._readoutAcc += dt;
      if (this._readoutAcc >= this._readoutPeriod) {
        this._readoutAcc = 0;
        this._updateReadout(state);
      }
      this._perfAcc += dt;
      if (this._perfAcc >= PERF_PERIOD) {
        this._perfAcc = 0;
        this._updatePerf(state);
      }
    }

    if (this._panelOpen) {
      this._syncAcc += dt;
      if (this._syncAcc >= PANEL_SYNC_PERIOD) {
        this._syncAcc = 0;
        this._syncPanel(false);
      }
    }

    // Re-assert a manual wind speed. wind.js damps state.wind.strength toward
    // the weather profile every frame, so a pin has to be renewed; it is one
    // float write, and only while the user holds manual control.
    if (this._windPin !== null && !this._windSetter) state.wind.strength = this._windPin;
  }

  // -------------------------------------------------------------------------
  // Readout
  // -------------------------------------------------------------------------

  _updateReadout(state) {
    const c = this._cache;
    const dn = this.dayNight;

    // --- clock + phase ---
    const clock = dn ? dn.getClockString() : hourToClock(state.time.timeOfDay);
    if (clock !== c.clock) {
      c.clock = clock;
      this._elClock.textContent = clock;
    }
    const phase = dn ? dn.getPhaseName() : '';
    if (phase !== c.phase) {
      c.phase = phase;
      this._elPhase.textContent = phase;
    }

    // --- weather ---
    // Mid-transition the honest label is the destination: the sky is already
    // visibly moving toward it, and naming "Rain" before the first drop is
    // exactly what a forecast strip does.
    const w = state.weather;
    const label = WEATHER_LABELS[w.target] || WEATHER_LABELS[w.type] || '';
    const arriving = w.target !== w.type && w.transition < 0.985;
    const text = arriving ? `${label}…` : label;
    if (text !== c.weather) {
      c.weather = text;
      this._elWeather.textContent = text;
    }

    // --- wind ---
    // The number is the mean flow (it moves over seconds, so the text is almost
    // never rewritten); the meter carries the gust, which moves every frame and
    // therefore has to be a transform.
    const wind = state.wind;
    // A NaN that reached state.wind would otherwise be laundered straight into
    // the DOM ("NaN m/s", `scaleX(NaN)`, `rotate(NaNdeg)`) and — worse for the
    // vane — into our own cache, where it would latch permanently because every
    // subsequent comparison against NaN is false.
    const mean = Number.isFinite(wind.strength) ? wind.strength : 0;
    // Compare the ROUNDED number, not the formatted string: the mean flow moves
    // continuously, so `toFixed` would mint a fresh string eight times a second
    // for a value that changes its printed form once every few seconds. The
    // string is built only when it is about to be written.
    const meanR = Math.round(mean * 10);
    if (meanR !== c.wind) {
      c.wind = meanR;
      this._elWind.textContent = `${(meanR / 10).toFixed(1)} m/s`;
    }
    const gust = Number.isFinite(wind.gust) ? wind.gust : 1;
    const meter = clamp01((mean * gust) / WIND_METER_MAX);
    if (Math.abs(meter - c.meter) > 0.008) {
      c.meter = meter;
      this._elWindBar.style.transform = `scaleX(${meter.toFixed(3)})`;
    }

    // Vane points where the air is going, in view space, so it stays meaningful
    // as the player turns. right = cross(forward, up) = (-fz, 0, fx).
    const f = state.player.forward;
    let fx = f.x;
    let fz = f.z;
    const flen = Math.hypot(fx, fz);
    if (flen > 1e-4) {
      fx /= flen;
      fz /= flen;
    } else {
      fx = 0;
      fz = -1;
    }
    const along = wind.direction.x * fx + wind.direction.y * fz;
    const lateral = wind.direction.x * -fz + wind.direction.y * fx;
    const deg = Math.atan2(lateral, along) * RAD2DEG;
    if (!Number.isFinite(deg)) return;
    // The angle written to the element is CONTINUOUS, not the wrapped value in
    // (-180,180]. .wind__vane transitions `transform` linearly, and CSS
    // interpolates the number, not the direction — so writing 179deg and then
    // -179deg makes the dart travel 358 degrees the wrong way round. That wrap
    // is not an edge case here: the bearing is view-relative, so it is crossed
    // every single time the player turns their head past the wind's back
    // bearing. Accumulating the shortest signed step keeps the transition short.
    const delta = mod(deg - c.vaneDeg + 180, 360) - 180;
    if (!(Math.abs(delta) < ARROW_EPSILON_DEG)) {
      c.vaneDeg += delta;
      this._elVane.style.transform = `rotate(${c.vaneDeg.toFixed(1)}deg)`;
    }
  }

  _updatePerf(state) {
    const c = this._cache;
    const perf = state.perf;

    // state.debug.showStats is public — anything may flip it, not just our own
    // switch. This early-returns when it is already in sync.
    this._applyStats();

    // Every value below is guarded before it reaches a cache. A NaN that lands in
    // one is permanent damage of exactly the kind the wind readout is already
    // protected from: `NaN !== NaN` is true, so the change-gate stops gating and
    // this block rewrites textContent three times a second for the rest of the
    // session — and `NaN >= x` is false, so the fps band latches to red. Not
    // theoretical: perf.fps and perf.avgFrameMs are both divisions, and
    // resolutionScale is written by an adaptive controller that also divides.
    // -1 is the sentinel: it can never collide with a real reading, and it
    // compares normally, so one bad frame costs one write and no more.
    const fps = Number.isFinite(perf.fps) ? Math.round(perf.fps) : -1;
    if (fps !== c.fps) {
      c.fps = fps;
      this._elFps.textContent = fps > 0 && fps < 1000 ? String(fps) : '--';
    }

    // Three bands, so a glance tells you whether the frame is healthy without
    // having to read the number at all. Scored against the tier's OWN target,
    // which is why this cannot live inside the "fps changed" branch above: the
    // target moves when the tier does, so switching HIGH -> ULTRA at a steady
    // 45 fps has to repaint green even though the number never moved.
    const target = perf.targetMs > 0 ? 1000 / perf.targetMs : 60;
    // 'none' matches no rule in styles.css, so an unreadable counter falls back to
    // the neutral --paper-60. Colouring "we have no number" red would be a lie
    // about the frame, and the colour is the part people read without looking.
    const band =
      fps < 0 ? 'none' : fps >= target * 0.92 ? 'good' : fps >= target * 0.7 ? 'fair' : 'poor';
    if (band !== c.band) {
      c.band = band;
      this._elFps.dataset.band = band;
    }

    const tierKey = state.quality.tier;
    if (tierKey !== c.tier) {
      c.tier = tierKey;
      this._elTier.textContent = TIER_PRESETS[tierKey] ? TIER_PRESETS[tierKey].label : tierKey;
    }

    // Render scale is only worth naming when it is not 100%.
    const rawScale = state.quality.resolutionScale;
    const scale = Number.isFinite(rawScale) ? Math.round(rawScale * 100) : -1;
    if (scale !== c.scale) {
      c.scale = scale;
      this._elScale.textContent = scale > 0 && scale < 100 ? ` · ${scale}%` : '';
    }

    if (!state.debug.showStats) return;

    const rawCpu = Number.isFinite(perf.cpuMs) ? perf.cpuMs : perf.frameMs;
    const cpuR = Number.isFinite(rawCpu) ? Math.round(rawCpu * 10) : -1;
    if (cpuR !== c.cpu) {
      c.cpu = cpuR;
      this._elCpu.textContent = cpuR >= 0 ? (cpuR / 10).toFixed(1) : '--';
    }
    const draws = Number.isFinite(perf.drawCalls) ? perf.drawCalls : -1;
    if (draws !== c.draws) {
      c.draws = draws;
      this._elDraws.textContent = draws >= 0 ? String(draws) : '--';
    }
    const tris = Number.isFinite(perf.triangles) ? perf.triangles : 0;
    if (tris !== c.tris) {
      c.tris = tris;
      this._elTris.textContent = tris > 9999 ? `${Math.round(tris / 1000)}k` : String(tris);
    }
    this._updateCosts();
  }

  /**
   * Per-system cost table. Rows are pooled and reused, and each cell is written
   * only when its value changes — a stable breakdown costs nothing to keep up.
   */
  _updateCosts() {
    const q = this.quality;
    if (!q || typeof q.getCostBreakdown !== 'function') return;
    const list = q.getCostBreakdown();
    const shown = Math.min(list.length, 8);

    while (this._costRows.length < shown) {
      const row = el('div', 'cost');
      const name = el('span', 'cost__name', '');
      const track = el('span', 'cost__track');
      const bar = el('i', 'cost__bar');
      track.appendChild(bar);
      const val = el('span', 'cost__ms', '');
      row.append(name, track, val);
      this._elCosts.appendChild(row);
      this._costRows.push({ row, name, bar, val, lastName: '', lastMs: -1, lastShare: -1 });
    }

    for (let i = 0; i < this._costRows.length; i++) {
      const r = this._costRows[i];
      const visible = i < shown;
      if (r.row.hidden === visible) r.row.hidden = !visible;
      if (!visible) continue;

      const item = list[i];
      if (item.label !== r.lastName) {
        r.lastName = item.label;
        r.name.textContent = item.label;
      }
      // Same sentinel discipline as the counters above: a system that publishes a
      // non-finite cost must not be able to jam this row's change-gate open, and
      // clamp01(NaN) is NaN, which would freeze the bar at whatever it last was
      // rather than showing the honest zero.
      const ms = Number.isFinite(item.ms) ? Math.round(item.ms * 10) : -1;
      if (ms !== r.lastMs) {
        r.lastMs = ms;
        r.val.textContent = ms >= 0 ? (ms / 10).toFixed(1) : '--';
      }
      const share = Number.isFinite(item.share) ? clamp01(item.share) : 0;
      if (Math.abs(share - r.lastShare) > 0.01) {
        r.lastShare = share;
        r.bar.style.transform = `scaleX(${share.toFixed(3)})`;
      }
    }
  }

  _applyStats() {
    const on = !!this.state.debug.showStats;
    // Already in sync — `hidden` is the negation of `on` by definition, so this
    // is the no-op test, not a mismatch test. Called at 3 Hz; it must not write.
    if (this._elPerfDetail.hidden === !on) return;
    this._elPerfDetail.hidden = !on;
    // Repaint the numbers on the next tick instead of waiting for a change.
    this._cache.cpu = -1;
    this._cache.draws = -1;
    this._cache.tris = -1;
  }

  // -------------------------------------------------------------------------
  // First-run hint
  // -------------------------------------------------------------------------

  _updateHint(dt, state) {
    if (!this._hintVisible) return;
    this._hintAge += dt;

    const moving = state.player.velocity.lengthSq() > HINT_SPEED_SQ;
    if (moving) {
      this._hintMoveTime += dt;
    } else {
      // Decay rather than reset: one stutter should not restart the timer, but
      // standing still again does mean they have not started playing yet.
      this._hintMoveTime = Math.max(0, this._hintMoveTime - dt * 0.5);
    }

    // A hint the user asked for waits for movement only — no timeout, because
    // they explicitly opened it.
    const moveLimit = this._hintPinned ? HINT_MOVE_SECONDS * 2 : HINT_MOVE_SECONDS;
    if (this._hintMoveTime >= moveLimit) {
      this.showHint(false);
      return;
    }
    if (!this._hintPinned && this._hintAge >= HINT_TIMEOUT) this.showHint(false);
  }

  /** @param {boolean} show */
  showHint(show) {
    const v = !!show;
    if (!v && !this._hintVisible) return;
    this._hintVisible = v;
    this._hintPinned = v;
    this._hintMoveTime = 0;
    this._hintAge = 0;
    this._elHint.classList.toggle('is-gone', !v);
  }

  _updatePhotoNote(dt) {
    if (!this._photo || this._photoNoteTimer <= 0) return;
    this._photoNoteTimer -= dt;
    if (this._photoNoteTimer <= 0) this._elPhotoNote.classList.add('is-gone');
  }

  // =========================================================================
  // Settings panel
  // =========================================================================

  setPanelOpen(open) {
    const v = !!open;
    if (v === this._panelOpen) return;
    this._panelOpen = v;
    this.el.classList.toggle('is-panel', v);

    if (this._panelCloseTimer) {
      clearTimeout(this._panelCloseTimer);
      this._panelCloseTimer = 0;
    }

    if (v) {
      this._elPanel.hidden = false;
      // The panel is unusable under pointer lock — the cursor is captured.
      this.input?.exitPointerLock?.();
      this.showHint(false);
      this._syncPanel(true);
      this._syncAcc = 0;
      // Give the sheet one frame at its off-screen offset so the transition
      // actually runs instead of being collapsed by style batching.
      requestAnimationFrame(() => {
        if (!this._panelOpen) return;
        this._elPanel.classList.add('is-open');
        // Move focus inside. Without this a keyboard user could never reach the
        // panel at all: Tab from outside is intercepted below and would just
        // close it again. With it, Tab cycles the panel and Esc leaves.
        const focusables = this._focusables();
        (focusables[0] || this._elPanel).focus({ preventScroll: true });
      });
    } else {
      this._elPanel.classList.remove('is-open');
      // Hand the keyboard back to the world rather than leaving focus on a
      // control that is about to be display:none.
      if (this._elPanel.contains(document.activeElement)) document.activeElement.blur();
      this._pointerInPanel = false;
      this._focusInPanel = false;
      this._liveControl = null;
      // Keep it displayed until the slide-out finishes, then take it out of the
      // layout and the accessibility tree entirely.
      const delay = this._reduced ? 0 : PANEL_CLOSE_MS;
      this._panelCloseTimer = setTimeout(() => {
        this._panelCloseTimer = 0;
        if (!this._panelOpen) this._elPanel.hidden = true;
      }, delay);
    }
    this._refreshGate();
  }

  /** @param {boolean} force sync even while closed, and drop the drag guard */
  _syncPanel(force) {
    if (!this._panelOpen && !force) return;
    if (force) this._liveControl = null;
    // Availability first: the effect switches' read() closures consult it, so
    // running it after the sync would leave every row one pass out of date.
    this._refreshEffects();
    for (let i = 0; i < this._sync.length; i++) this._sync[i]();

    // Render scale is meaningless while the adaptive controller owns it.
    if (this._sliderScale) {
      const auto = !!this.state.quality.adaptive;
      if (this._sliderScale.disabled !== auto) this._sliderScale.disabled = auto;
    }
    if (this._sliderWind) {
      const auto = this._windPin === null;
      if (this._sliderWind.disabled !== auto) this._sliderWind.disabled = auto;
    }
    if (this._btnLapse) {
      const on = this._lapse.active;
      if ((this._btnLapse.dataset.on === '1') !== on) {
        this._btnLapse.dataset.on = on ? '1' : '0';
        this._btnLapse.textContent = on ? 'Stop the day' : 'Run a whole day';
      }
    }
  }

  _onQualityEvent() {
    // A tier change rewrites every effect flag from the preset table. Bloom is
    // not in that table (core/quality.js owns it and we may not edit it), so it
    // gets a tier-appropriate default here unless the user has pinned it.
    const q = this.state.quality;
    if (!this._bloomPinned) q.bloom = q.tier !== 'low';
    this._syncPanel(true);
  }

  /**
   * Effect toggles. state.quality is shared with every system, so the flag is
   * written and then announced — main.js fans EVENTS.QUALITY_CHANGED out to
   * every system's onQualityChange(), which is exactly how a tier switch is
   * delivered, so consumers need no second code path.
   */
  _setEffect(key, value) {
    const q = this.state.quality;
    const v = !!value;
    if (q[key] === v) return;
    q[key] = v;
    if (key === 'bloom') this._bloomPinned = true;

    // Nudge the post stack directly too: it may hold compiled passes it can
    // enable without a full reconfigure.
    if (this.post && typeof this.post.setEffectEnabled === 'function') {
      this.post.setEffectEnabled(key, v);
    }
    this.bus?.emit(EVENTS.QUALITY_CHANGED, q);
  }

  /**
   * Manual wind speed. A real setter is preferred if WindField ever grows one —
   * a system-owned setter can damp toward the value instead of being overwritten
   * by its own update() a frame later.
   * @param {number|null} value null restores weather-driven wind
   */
  _applyWind(value) {
    const wind = this.wind;
    this._windSetter = false;
    if (wind && typeof wind.setStrength === 'function') {
      wind.setStrength(value === null ? null : value);
      this._windSetter = true;
      return;
    }
    if (value !== null) this.state.wind.strength = value;
  }

  _refreshGate() {
    if (!this.input) return;
    // Orbit drives the camera itself; letting WASD through would have the player
    // walking away underneath a locked-off shot.
    const gate = (this._panelOpen && (this._pointerInPanel || this._focusInPanel)) || this._orbit;
    if (gate === this._gateApplied) return;
    this._gateApplied = gate;
    this.input.uiHasFocus = gate;
  }

  // =========================================================================
  // Cinematic tools
  // =========================================================================

  setPhotoMode(on) {
    const v = !!on;
    if (v === this._photo) return;
    this._photo = v;
    this.el.classList.toggle('is-photo', v);
    this._elPhoto.classList.toggle('is-on', v);
    if (v) {
      this.showHint(false);
      this._elPhotoNote.classList.remove('is-gone');
      this._photoNoteTimer = 4;
    } else {
      this.setOrbit(false);
      this._photoNoteTimer = 0;
    }
    this._syncPanel(true);
  }

  setOrbit(on) {
    const v = !!on;
    if (v === this._orbit) return;
    this._orbit = v;
    if (v) {
      // The tree finishes building long after link() in the worst case; re-read
      // its anchors so the first shot is framed on the real canopy.
      this._refreshOrbitRig();
      // Enter from wherever the camera already is, so the swing-in reads as a
      // dolly rather than a cut: seed the orbit angle from the current bearing.
      const dx = this.camera.position.x - this._orbitCenter.x;
      const dz = this.camera.position.z - this._orbitCenter.z;
      if (dx * dx + dz * dz > 1e-4) this._orbitAngle = Math.atan2(dz, dx);
      if (!this._photo) this.setPhotoMode(true);
      this.input?.exitPointerLock?.();
    }
    this._refreshGate();
    this._syncPanel(true);
  }

  toggleTimelapse() {
    if (this._lapse.active) this._cancelLapse(true);
    else this.startTimelapse();
  }

  startTimelapse() {
    const state = this.state;
    const L = this._lapse;
    if (L.active) return;
    L.active = true;
    L.hours = 0;
    L.prevSpeed = state.time.daySpeed;
    L.prevPaused = state.time.paused;
    L.setSpeed = HOURS / TIMELAPSE_SECONDS;

    if (this.dayNight) {
      this.dayNight.cancelScrub?.();
      this.dayNight.setPaused(false);
      this.dayNight.setDaySpeed(L.setSpeed);
    } else {
      state.time.paused = false;
      state.time.daySpeed = L.setSpeed;
    }
    this._elLapse.classList.add('is-on');
    this._cache.lapse = -1;
    this._elLapseFill.style.transform = 'scaleX(0)';
    this._toast(`A whole day in ${TIMELAPSE_SECONDS} seconds`);
    this._syncPanel(true);
  }

  /** @param {boolean} restore put the clock speed back where it was */
  _cancelLapse(restore) {
    const L = this._lapse;
    if (!L.active) return;
    L.active = false;
    this._elLapse.classList.remove('is-on');
    if (restore) {
      if (this.dayNight) {
        this.dayNight.setDaySpeed(L.prevSpeed);
        this.dayNight.setPaused(L.prevPaused);
      } else {
        this.state.time.daySpeed = L.prevSpeed;
        this.state.time.paused = L.prevPaused;
      }
    }
    this._syncPanel(true);
  }

  _updateTimelapse(dt, state) {
    const L = this._lapse;
    if (!L.active) return;

    // Someone else took the clock (a preset jump, the speed slider, a pause):
    // stand down without stomping on whatever they set.
    if (Math.abs(state.time.daySpeed - L.setSpeed) > 1e-4 || state.time.paused) {
      this._cancelLapse(false);
      return;
    }

    // Measured from hours actually travelled, not from wall-clock seconds, so
    // the run covers exactly one cycle however the speed ramp behaves. Falls
    // back to the commanded speed if the sibling does not publish the measured
    // one — a missing `factors` must not throw inside the frame loop.
    const measured = this.dayNight?.factors?.hoursPerSecond;
    const hps = Number.isFinite(measured) ? Math.abs(measured) : L.setSpeed;
    L.hours += hps * dt;

    const f = clamp01(L.hours / HOURS);
    if (Math.abs(f - this._cache.lapse) > 0.004) {
      this._cache.lapse = f;
      this._elLapseFill.style.transform = `scaleX(${f.toFixed(3)})`;
    }
    if (L.hours >= HOURS) {
      this._cancelLapse(true);
      this._toast('Sunrise to sunrise');
    }
  }

  /**
   * Re-read the tree's framing anchors. Called from link() and again on every
   * entry into orbit, so a tree that finishes building (or is re-placed onto
   * different ground) after link() re-frames itself rather than staying on the
   * stub values SakuraTree carries in its constructor.
   */
  _refreshOrbitRig() {
    const c = this.tree?.canopyCenter;
    if (c && Number.isFinite(c.x) && Number.isFinite(c.y) && Number.isFinite(c.z)) {
      this._orbitCenter.copy(c);
    }
    // world/terrain.js lifts its clearing (PAD_RISE) so the trunk base is NOT
    // at y = 0, and tree.canopyCenter is absolute world space. Framing off the
    // absolute Y would scale every height by the ground level under the tree.
    const baseY = this.tree?.position?.y;
    this._orbitBase = Number.isFinite(baseY) ? baseY : 0;

    const radius = this.tree?.canopyRadius;
    this._orbitRadius = Math.max((Number.isFinite(radius) ? radius : 8) * 2.6, 14);
  }

  /**
   * Slow orbit of the tree.
   *
   * Runs from HUD.update(), which main.js schedules AFTER the player controller
   * and BEFORE the render — so the camera transform found here is exactly what
   * the player wanted this frame, and whatever is left is what gets rendered.
   * Blending between the two means neither entering nor leaving is a cut, and
   * the controller keeps running untouched underneath the whole time.
   */
  _updateOrbit(dt) {
    const target = this._orbit ? 1 : 0;
    if (target === 0 && this._orbitBlend <= 0) return;

    this._orbitBlend = damp(this._orbitBlend, target, ORBIT_BLEND_LAMBDA, dt);
    if (target === 1 && this._orbitBlend > 0.9995) this._orbitBlend = 1;
    else if (target === 0 && this._orbitBlend < 0.0008) this._orbitBlend = 0;

    const cam = this.camera;
    _playerPos.copy(cam.position);
    _playerQuat.copy(cam.quaternion);

    this._orbitTime += dt;
    this._orbitAngle = mod(this._orbitAngle + ORBIT_RATE * dt, Math.PI * 2);
    const t = this._orbitTime;
    const centre = this._orbitCenter;
    // Every vertical below is an offset from the trunk base, never an absolute
    // world Y: `rise` is the canopy's height above its own ground, so the shot
    // frames the same way whatever height the terrain puts the tree at.
    const base = this._orbitBase;
    const rise = Math.max(0, centre.y - base);

    // Radius and height breathe on slow, mutually incommensurate periods so the
    // path never visibly repeats — a perfect circle is the tell that gives away
    // every automatic turntable ever shipped.
    const r = this._orbitRadius * (1 + 0.06 * Math.sin(t * 0.083));
    const h = base + rise * 0.7 + 2.6 + 1.7 * Math.sin(t * 0.061 + 1.1);

    _orbitEye.set(
      centre.x + Math.cos(this._orbitAngle) * r,
      h,
      centre.z + Math.sin(this._orbitAngle) * r
    );

    // Never clip through the ground, and never sit so low that the near plane
    // eats the grass.
    if (this.terrain && typeof this.terrain.getHeight === 'function') {
      const ground = this.terrain.getHeight(_orbitEye.x, _orbitEye.z);
      if (Number.isFinite(ground) && _orbitEye.y < ground + 1.6) _orbitEye.y = ground + 1.6;
    }

    // Aim a little below the canopy centre: putting the mass of blossom in the
    // upper third is what makes the shot read as a portrait of the tree.
    _orbitTarget.set(centre.x, base + rise * 0.72 + 0.5 * Math.sin(t * 0.047), centre.z);

    _lookMat.lookAt(_orbitEye, _orbitTarget, _up);
    _orbitQuat.setFromRotationMatrix(_lookMat);

    // Quaternions are a double cover of SO(3): without this the slerp can take
    // the long way round and spin the camera through a full rotation on entry.
    if (_playerQuat.dot(_orbitQuat) < 0) {
      _orbitQuat.set(-_orbitQuat.x, -_orbitQuat.y, -_orbitQuat.z, -_orbitQuat.w);
    }

    // Smoothstep the blend so the hand-over has no velocity discontinuity at
    // either end; a raw exponential arrives with a visible residual drift.
    const b = this._orbitBlend;
    const e = b * b * (3 - 2 * b);
    cam.position.lerpVectors(_playerPos, _orbitEye, e);
    cam.quaternion.slerpQuaternions(_playerQuat, _orbitQuat, e);
    cam.updateMatrixWorld(true);
  }

  // -------------------------------------------------------------------------
  // Capture
  // -------------------------------------------------------------------------

  /**
   * Grab the drawing buffer as a PNG.
   *
   * The renderer is created with `preserveDrawingBuffer: false`, so the pixels
   * are only readable between the draw and the compositor's swap. main.js
   * queues its next frame at the TOP of the current one, which means any
   * callback registered now lands after it in the same rAF batch — i.e. after
   * post.render() has run and before the buffer is presented. That is the only
   * window in which this works, and it is a reliable one.
   */
  capture() {
    if (this._captureArmed) return;
    this._captureArmed = true;
    requestAnimationFrame(() => {
      if (!this._captureArmed) return;
      this._captureArmed = false;
      clearTimeout(this._captureTimer);
      this._captureTimer = 0;
      this._grab();
    });
    // rAF never fires in a background tab. Without this, one capture requested
    // just before a tab switch would latch the flag and kill the feature for the
    // rest of the session.
    clearTimeout(this._captureTimer);
    this._captureTimer = setTimeout(() => {
      this._captureTimer = 0;
      this._captureArmed = false;
    }, 1200);
  }

  _grab() {
    const canvas = this.canvas;
    if (!canvas || typeof canvas.toBlob !== 'function') {
      this._toast('This browser cannot export the canvas');
      return;
    }
    const w = canvas.width;
    const h = canvas.height;
    const clock = (this.dayNight ? this.dayNight.getClockString() : '').replace(':', '');
    const name = `sakura-realm-${dateStamp()}${clock ? '-' + clock : ''}.png`;

    canvas.toBlob((blob) => {
      if (!blob) {
        this._toast('Capture failed');
        return;
      }
      // One link element, one live object URL. Revoking the previous one on the
      // way in stops a session of a hundred captures from pinning a hundred
      // full-resolution PNGs in memory.
      if (this._captureUrl) URL.revokeObjectURL(this._captureUrl);
      this._captureUrl = URL.createObjectURL(blob);
      this._captureLink.href = this._captureUrl;
      this._captureLink.download = name;
      this._captureLink.click();
      this._toast(`Saved · ${w}×${h}`);
    }, 'image/png');
  }

  // -------------------------------------------------------------------------
  // Toast
  // -------------------------------------------------------------------------

  _toast(message) {
    const t = this._elToast;
    t.textContent = message;
    t.classList.add('is-on');
    if (this._toastTimer) clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => {
      this._toastTimer = 0;
      t.classList.remove('is-on');
    }, TOAST_MS);
  }

  // =========================================================================
  // Keyboard
  // =========================================================================

  _onPointerUp() {
    if (this._liveControl) this._liveControl = null;
  }

  /**
   * Enabled, reachable controls inside the panel, in DOM order. Built on demand
   * (only on a Tab keypress) because disabled state changes as the panel syncs.
   * @returns {HTMLElement[]}
   */
  _focusables() {
    const nodes = this._elPanel.querySelectorAll('button, input, select, [tabindex="0"]');
    const out = [];
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      if (!n.disabled && !n.hidden && n.tabIndex !== -1) out.push(n);
    }
    return out;
  }

  /** Cyclic focus containment for the open panel. */
  _trapTab(e) {
    const list = this._focusables();
    if (list.length === 0) return;
    const first = list[0];
    const last = list[list.length - 1];
    const active = document.activeElement;
    if (e.shiftKey && (active === first || active === this._elPanel)) {
      e.preventDefault();
      last.focus({ preventScroll: true });
    } else if (!e.shiftKey && active === last) {
      e.preventDefault();
      first.focus({ preventScroll: true });
    }
  }

  /**
   * The HUD binds its own listener rather than polling core/input.js, and that
   * is deliberate: `input.wasPressed()` returns false while `uiHasFocus` is set,
   * which is exactly when Esc and Tab have to keep working. None of these codes
   * is read by any other system, so nothing is being stolen from the game.
   *
   * Capture phase, so Esc reaches us even from inside a focused slider.
   */
  _onKeyDown(e) {
    if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;

    switch (e.code) {
      case 'Tab':
        // Inside the panel, Tab belongs to the focus ring — but it must not walk
        // out of the sheet into the browser chrome, or the next Tab would close
        // the panel from under a keyboard user.
        if (this._panelOpen && this._elPanel.contains(document.activeElement)) {
          this._trapTab(e);
          return;
        }
        e.preventDefault();
        this.setPanelOpen(!this._panelOpen);
        return;
      case 'Escape':
        // The panel wins whenever the keyboard is actually inside it. Photo mode
        // deliberately no longer closes the panel (the photo switch lives on it),
        // so "photo on AND panel open" is now a reachable state — and under the
        // plain orbit -> photo -> panel order, a user typing in the sheet who
        // pressed Esc got photo mode dismissed behind them while the sheet they
        // were looking at stayed put. Esc, from inside a sheet, closes the sheet.
        if (
          this._panelOpen &&
          (this._focusInPanel || (e.target && this._elPanel.contains(e.target)))
        ) {
          this.setPanelOpen(false);
        } else if (this._orbit) this.setOrbit(false);
        else if (this._photo) this.setPhotoMode(false);
        else if (this._panelOpen) this.setPanelOpen(false);
        else return;
        e.preventDefault();
        return;
      case 'F2':
        e.preventDefault();
        this.capture();
        return;
      default:
        break;
    }

    // Letter keys must not fire while a control has focus: Space, the arrows and
    // every letter belong to whatever the user is typing into or dragging.
    const t = e.target;
    if (
      t &&
      (t.tagName === 'INPUT' ||
        t.tagName === 'TEXTAREA' ||
        t.tagName === 'SELECT' ||
        t.isContentEditable)
    ) {
      return;
    }
    // Nor anywhere inside the open panel. A keyboard user stepping through the
    // switches is on <button>s, not inputs, so the test above lets them through
    // — and P would drop them into photo mode, H would blank the readout, and O
    // would take the camera away, all from a keystroke that was meant for the
    // control under their finger.
    if (this._panelOpen && t && this._elPanel.contains(t)) return;

    switch (e.code) {
      case 'KeyH':
        this.state.debug.hideUI = !this.state.debug.hideUI;
        this._syncPanel(true);
        break;
      case 'KeyP':
        this.setPhotoMode(!this._photo);
        break;
      case 'KeyO':
        this.setOrbit(!this._orbit);
        break;
      case 'KeyL':
        this.toggleTimelapse();
        break;
      default:
        return;
    }
    e.preventDefault();
  }
}
