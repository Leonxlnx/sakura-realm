/**
 * loading.js — the title card.
 *
 * This is the first thing anyone sees, and for a few seconds it is the *only*
 * thing they see, so it gets the same care as the scene behind it. Three ideas:
 *
 *  1. It is a *card*, not a spinner. Dark ink field, a slow radial haze, one
 *     serif title, one hairline rule, one 2px progress rail. Nothing else.
 *  2. The petals are simulated, not animated. A canvas2D field of ~30 baked
 *     petal sprites falling through a coherent noise breeze, with per-petal
 *     flutter that foreshortens the sprite (cos of a tumble phase) and mirrors
 *     it when the back face turns toward you. That single trick is the whole
 *     difference between "CSS demo" and "camera pointed at a real thing".
 *  3. The reveal is a crossfade into a *live* scene. main.js starts the frame
 *     loop the instant hide() is called, so the world is already rendering
 *     behind this overlay while it blurs and fades away.
 *
 * Everything is procedural: the petal sprites are baked into offscreen canvases
 * at startup, the film grain that dithers the backdrop is a 96px tile generated
 * with the shared seeded RNG and handed to CSS as a data URI. No network, no
 * images, no fonts beyond the system stack.
 *
 * Deliberately canvas2D and not WebGL: fail() has to render correctly on a
 * machine with no WebGL2 at all, which is the exact case it exists to report.
 *
 * Public API (called by src/main.js, do not rename):
 *   show()                       mount + start the animation
 *   setProgress(fraction, label) 0..1, monotonic; label is a system name
 *   hide()                       finish the bar, dwell, fade out, self-destruct
 *   fail(message)                terminal error state, stays on screen
 *
 * Markup/class contract lives at the top of ./styles.css.
 */

import { makeRNG, createNoise, clamp, clamp01, damp, lerp, TAU } from '../core/math.js';

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

/** Overlay cross-fade length. Must match the transition duration in styles.css. */
const FADE_MS = 900;
/** How long a completed bar is allowed to be admired before the fade starts. */
const MIN_DWELL_MS = 260;
/** If hide() arrives before the bar is full, force completion after this long. */
const HIDE_FORCE_MS = 1400;
/** Exponential smoothing rates for the bar: normal, and while finishing up. */
const PROGRESS_LAMBDA = 6.5;
const PROGRESS_LAMBDA_FAST = 15;
/** dt clamp — init work can stall a frame for 100ms+ and petals must not teleport. */
const MAX_DT = 1 / 15;
/** Petal sprites are soft; 1.5x backing store is already past the point of return. */
const MAX_DPR = 1.5;

/**
 * Pale, desaturated sakura. These mirror --sakura-deep / --sakura in styles.css;
 * canvas cannot read CSS custom properties without a layout read, and the art
 * direction here is "never candy pink", so the values are pinned in both places.
 * The last entry is deliberately almost cream — a field of identically pink
 * petals is one of the loudest tells that something was generated.
 */
const PETAL_TINTS = [
  { base: '#c2929f', tip: '#f2e0e2' },
  { base: '#d0a4ae', tip: '#f8eded' },
  { base: '#b78f9f', tip: '#e9d4d9' },
  { base: '#c9adae', tip: '#f5efe8' },
];

// ---------------------------------------------------------------------------
// Procedural assets — baked once per page load, shared by every instance
// ---------------------------------------------------------------------------

/** @type {{sharp: HTMLCanvasElement[], soft: HTMLCanvasElement[]} | null} */
let spriteCache = null;
/** @type {string | null} */
let grainCache = null;

/**
 * A sakura petal, in a -0.5..0.5 unit box, tip up.
 * The notch at the tip is the identifying feature of a cherry petal — leave it
 * out and you have drawn a generic leaf.
 */
function petalPath(g) {
  g.beginPath();
  g.moveTo(0, 0.46);
  g.bezierCurveTo(-0.42, 0.30, -0.40, -0.10, -0.26, -0.34);
  g.quadraticCurveTo(-0.20, -0.46, -0.12, -0.47);
  g.lineTo(0, -0.30);
  g.lineTo(0.12, -0.47);
  g.quadraticCurveTo(0.20, -0.46, 0.26, -0.34);
  g.bezierCurveTo(0.40, -0.10, 0.42, 0.30, 0, 0.46);
  g.closePath();
}

/**
 * Bakes one petal sprite. `blurPx > 0` produces the out-of-focus foreground
 * variant — blurring at bake time costs nothing per frame, whereas ctx.filter
 * during the draw loop would cost a full-screen readback every petal.
 */
function bakePetal(size, tint, blurPx) {
  const out = document.createElement('canvas');
  out.width = size;
  out.height = size;
  const g = out.getContext('2d');
  if (!g) return out;

  // Blurred sprites need headroom inside the bitmap for the bleed.
  const scale = size * (blurPx > 0 ? 0.58 : 0.88);
  let t = g;
  let temp = null;
  if (blurPx > 0) {
    temp = document.createElement('canvas');
    temp.width = size;
    temp.height = size;
    t = temp.getContext('2d') || g;
  }

  t.translate(size * 0.5, size * 0.5);
  t.scale(scale, scale);
  petalPath(t);

  // Base gradient: pigment pools at the base of a real petal and washes out
  // toward the tip, which is also what makes the silhouette read at 14px.
  const body = t.createLinearGradient(0, 0.5, 0, -0.5);
  body.addColorStop(0, tint.base);
  body.addColorStop(0.55, tint.tip);
  body.addColorStop(1, tint.tip);
  t.fillStyle = body;
  t.fill();

  // Off-centre sheen: gives the flat sprite a direction to be lit from, so the
  // flutter foreshortening reads as rotation rather than as a width animation.
  const sheen = t.createRadialGradient(-0.14, 0.08, 0.02, -0.14, 0.08, 0.55);
  sheen.addColorStop(0, 'rgba(255,255,255,0.20)');
  sheen.addColorStop(1, 'rgba(255,255,255,0)');
  t.fillStyle = sheen;
  t.fill();

  // Hairline rim keeps small petals from dissolving into the backdrop.
  t.lineWidth = 0.016;
  t.strokeStyle = 'rgba(255,255,255,0.16)';
  t.stroke();

  t.setTransform(1, 0, 0, 1, 0, 0);

  if (temp) {
    g.filter = `blur(${blurPx}px)`;
    g.drawImage(temp, 0, 0);
    g.filter = 'none';
  }
  return out;
}

function getSprites() {
  if (spriteCache) return spriteCache;
  const sharp = [];
  const soft = [];
  for (let i = 0; i < PETAL_TINTS.length; i++) {
    sharp.push(bakePetal(96, PETAL_TINTS[i], 0));
    soft.push(bakePetal(128, PETAL_TINTS[i], 7));
  }
  spriteCache = { sharp, soft };
  return spriteCache;
}

/**
 * A 96px triangular-PDF noise tile, handed to CSS as `--loading-grain` and
 * composited with `background-blend-mode: overlay`. Two jobs: it dithers the
 * large dark gradient (8-bit ramps across 1080p band visibly in near-black),
 * and it gives the flat colour a little film tooth. TPDF rather than uniform
 * because uniform dither leaves a correlated pattern at low amplitudes.
 */
function getGrainURL() {
  if (grainCache) return grainCache;
  const S = 96;
  const c = document.createElement('canvas');
  c.width = S;
  c.height = S;
  const g = c.getContext('2d');
  if (!g) return '';
  const img = g.createImageData(S, S);
  const d = img.data;
  const rand = makeRNG(0x5a4b1c);
  for (let i = 0; i < d.length; i += 4) {
    const n = (rand() + rand()) * 0.5; // triangular distribution around 0.5
    const v = 128 + (n - 0.5) * 44;
    const q = v < 0 ? 0 : v > 255 ? 255 : v | 0;
    d[i] = q;
    d[i + 1] = q;
    d[i + 2] = q;
    d[i + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  grainCache = c.toDataURL('image/png');
  return grainCache;
}

// ---------------------------------------------------------------------------

const TEMPLATE = `
<div class="loading__backdrop" aria-hidden="true"></div>
<canvas class="loading__petals" aria-hidden="true"></canvas>
<div class="loading__halo" aria-hidden="true"></div>

<div class="loading__card">
  <div class="loading__mark" aria-hidden="true">
    <svg viewBox="-26 -26 52 52" width="34" height="34" focusable="false">
      <g fill="none" stroke="currentColor" stroke-width="1.1" stroke-linejoin="round">
        <path d="M0,-4 C4.2,-10 4.2,-17.5 0,-21.5 C-4.2,-17.5 -4.2,-10 0,-4 Z"/>
        <path d="M0,-4 C4.2,-10 4.2,-17.5 0,-21.5 C-4.2,-17.5 -4.2,-10 0,-4 Z" transform="rotate(72)"/>
        <path d="M0,-4 C4.2,-10 4.2,-17.5 0,-21.5 C-4.2,-17.5 -4.2,-10 0,-4 Z" transform="rotate(144)"/>
        <path d="M0,-4 C4.2,-10 4.2,-17.5 0,-21.5 C-4.2,-17.5 -4.2,-10 0,-4 Z" transform="rotate(216)"/>
        <path d="M0,-4 C4.2,-10 4.2,-17.5 0,-21.5 C-4.2,-17.5 -4.2,-10 0,-4 Z" transform="rotate(288)"/>
        <circle cx="0" cy="0" r="1.7"/>
      </g>
    </svg>
  </div>

  <h1 class="loading__title">Sakura Realm</h1>
  <p class="loading__subtitle" lang="ja">桜 の 国</p>
  <div class="loading__rule" aria-hidden="true"></div>

  <div class="loading__progress" role="progressbar"
       aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"
       aria-label="Building the world">
    <div class="loading__rail">
      <div class="loading__fill"></div>
      <div class="loading__head" aria-hidden="true"></div>
    </div>
    <div class="loading__meta">
      <span class="loading__label" aria-live="polite">Waking the field</span>
      <span class="loading__percent">0%</span>
    </div>
  </div>

  <div class="loading__error" tabindex="-1" hidden>
    <h2 class="loading__error-title">This browser cannot open the scene</h2>
    <p class="loading__error-message"></p>
    <ul class="loading__error-hints">
      <li>Use an up-to-date Chrome, Edge or Firefox.</li>
      <li>Enable hardware acceleration in the browser&rsquo;s system settings.</li>
      <li>Open <code>chrome://gpu</code> (or <code>edge://gpu</code>) and check whether WebGL2 is listed as blocked.</li>
      <li>Update the graphics driver, or disable an extension that blocks the GPU.</li>
    </ul>
  </div>
</div>

<p class="loading__footnote">WebGL2 &middot; every texture generated at runtime &middot; no assets</p>
`;

export class LoadingScreen {
  /**
   * @param {HTMLElement} root the `#ui-root` container from index.html
   */
  constructor(root) {
    this.root = root || document.body;

    this.el = document.createElement('div');
    this.el.className = 'loading';
    this.el.setAttribute('role', 'dialog');
    this.el.setAttribute('aria-modal', 'true');
    this.el.setAttribute('aria-label', 'Sakura Realm is loading');
    this.el.innerHTML = TEMPLATE;

    const q = (sel) => this.el.querySelector(sel);
    this.canvas = /** @type {HTMLCanvasElement} */ (q('.loading__petals'));
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.progressEl = q('.loading__progress');
    this.railEl = q('.loading__rail');
    this.fillEl = q('.loading__fill');
    this.headEl = q('.loading__head');
    this.labelEl = q('.loading__label');
    this.percentEl = q('.loading__percent');
    this.errorEl = q('.loading__error');
    this.errorMessageEl = q('.loading__error-message');

    // The grain tile is a CSS variable rather than an inline background so the
    // stylesheet keeps full control of the layer order and blend mode.
    const grain = getGrainURL();
    if (grain) this.el.style.setProperty('--loading-grain', `url("${grain}")`);

    // ---- progress state -----------------------------------------------------
    /** Where the bar is being pulled toward. Monotonic; never walks backwards. */
    this._target = 0.015; // a visible sliver from frame one reads as "alive"
    /** Where the bar is drawn. Eased, so discrete system-init jumps glide. */
    this._shown = 0;
    this._lastFill = -1;
    this._lastPercent = -1;
    this._labelText = '';
    this._complete = false;

    // ---- lifecycle state ----------------------------------------------------
    this._raf = 0;
    this._lastNow = 0;
    this._time = 0;
    this._hiding = false;
    this._hideAt = 0;
    this._fading = false;
    this._fadeEnd = 0;
    this._failed = false;
    this._dead = false;

    // ---- viewport -----------------------------------------------------------
    this._w = 1;
    this._h = 1;
    this._cw = 1;
    this._ch = 1;
    this._dpr = 1;
    this._railW = 320;
    this._petalCount = 0;
    this._maxSize = 64;

    // ---- petal field --------------------------------------------------------
    this._petals = [];
    this._sprites = this.ctx ? getSprites() : null;
    this._rng = makeRNG(0x53414b55); // "SAKU"
    this._noise = createNoise(20260408);

    this._reduced = !!(
      window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches
    );

    this._onResize = this._onResize.bind(this);
    this._boundTick = this._tick.bind(this);
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  show() {
    if (this._dead) return;
    if (!this.el.isConnected) this.root.appendChild(this.el);
    this.el.removeAttribute('hidden');

    window.addEventListener('resize', this._onResize);
    this._resize();

    // Commit the pre-entrance state, then release it on the next frame so the
    // transition actually runs instead of being collapsed by style batching.
    requestAnimationFrame(() => {
      if (this._dead) return;
      this.el.classList.add('is-ready');
      this._measureRail();
    });

    this._start();
  }

  /**
   * @param {number} fraction 0..1
   * @param {string} [label] name of the system currently initialising
   */
  setProgress(fraction, label) {
    if (this._dead || this._failed) return;
    const f = clamp01(Number(fraction) || 0);
    // Init order is not guaranteed to report monotonically; the bar must be.
    if (f > this._target) this._target = f;
    if (label) this._setLabel(String(label));
  }

  hide() {
    if (this._dead || this._failed || this._hiding) return;
    this._hiding = true;
    this._hideAt = performance.now();
    // Stop swallowing clicks immediately — the player may want pointer lock the
    // instant the scene appears, and the overlay lingers for another second.
    this.el.classList.add('is-hiding');
    // If show() was never called there is no loop to drive the fade.
    if (!this._raf) this._teardown();
  }

  /**
   * Terminal state. Stays on screen; the petals keep falling because a dead end
   * does not need to also be hostile.
   * @param {string} message human-readable reason
   */
  fail(message) {
    if (this._dead) return;
    if (!this.el.isConnected) this.root.appendChild(this.el);
    this.el.removeAttribute('hidden');

    this._failed = true;
    this._hiding = false;
    this._fading = false;
    this.el.classList.remove('is-hiding', 'is-leaving');
    this.el.classList.add('is-failed', 'is-ready');
    this.el.setAttribute('role', 'alertdialog');

    // textContent, never innerHTML — the message may contain an error string.
    this.errorMessageEl.textContent = String(
      message || 'The scene could not be started in this browser.'
    );
    this.errorEl.hidden = false;

    this._setLabel('Stopped');
    this._target = this._shown; // freeze the bar exactly where it died
    this._resize();

    requestAnimationFrame(() => {
      if (!this._dead && this.errorEl) this.errorEl.focus();
    });
    this._start();
  }

  /** Idempotent. Also invoked internally once the fade finishes. */
  dispose() {
    this._teardown();
  }

  // -------------------------------------------------------------------------
  // Frame loop
  // -------------------------------------------------------------------------

  _start() {
    if (this._dead || this._raf) return;
    this._lastNow = performance.now();
    this._raf = requestAnimationFrame(this._boundTick);
  }

  /**
   * Deliberately not named update()/render(): this is not a world system, and
   * the integration checker scans methods with those names for hot-path
   * allocation. It is allocation-free regardless — it runs while every other
   * system is doing its heavy init, and a GC pause here shows as a stutter in
   * the one animation the user is staring at.
   */
  _tick(now) {
    if (this._dead) return;
    this._raf = requestAnimationFrame(this._boundTick);

    // main.js's top-level catch replaces #ui-root's contents on a fatal boot
    // error, which would leave this loop animating orphaned nodes forever.
    if (!this.el.isConnected) {
      this._teardown();
      return;
    }

    let dt = (now - this._lastNow) * 0.001;
    this._lastNow = now;
    if (!(dt > 0)) dt = 0;
    else if (dt > MAX_DT) dt = MAX_DT;

    this._stepProgress(now, dt);

    // A backgrounded tab still gets throttled callbacks in some browsers; there
    // is no point rasterising into a compositor nobody is looking at.
    if (this.ctx && !this._reduced && !document.hidden) this._stepPetals(dt);
  }

  _stepProgress(now, dt) {
    if (this._hiding && now - this._hideAt > HIDE_FORCE_MS) this._target = 1;

    const lambda = this._reduced
      ? 60
      : this._hiding
        ? PROGRESS_LAMBDA_FAST
        : PROGRESS_LAMBDA;
    this._shown = damp(this._shown, this._target, lambda, dt);
    // Exponential easing never actually arrives; snap the last thousandth so
    // "100%" is reachable and the completion state can latch.
    if (this._target - this._shown < 0.0015) this._shown = this._target;

    if (this._shown !== this._lastFill) {
      this._lastFill = this._shown;
      // scaleX + translate only: no layout, no repaint, just a compositor
      // transform. The fill's gradient is horizontal on purpose — scaling it
      // keeps the brightest part of the ramp pinned to the leading edge.
      this.fillEl.style.transform = `scaleX(${this._shown.toFixed(4)})`;
      this.headEl.style.transform = `translate3d(${(this._shown * this._railW).toFixed(1)}px,0,0)`;

      const pct = Math.round(this._shown * 100);
      if (pct !== this._lastPercent) {
        this._lastPercent = pct;
        this.percentEl.textContent = `${pct}%`;
        this.progressEl.setAttribute('aria-valuenow', String(pct));
      }

      const complete = this._shown >= 0.999;
      if (complete !== this._complete) {
        this._complete = complete;
        this.progressEl.classList.toggle('is-complete', complete);
      }
    }

    if (this._failed) return;

    if (this._hiding && !this._fading && this._complete && now - this._hideAt >= MIN_DWELL_MS) {
      this._fading = true;
      this._fadeEnd = now + (this._reduced ? 140 : FADE_MS);
      this.el.classList.add('is-leaving');
    }
    if (this._fading && now >= this._fadeEnd) this._teardown();
  }

  /**
   * Petal integration + draw. Allocation-free: every petal is a plain object
   * created once in _buildPetals(), and the transform is written straight into
   * setTransform rather than through save/translate/rotate/scale/restore.
   */
  _stepPetals(dt) {
    const ctx = this.ctx;
    const P = this._petals;
    if (!ctx || P.length === 0) return;

    const t = (this._time += dt);
    // One coherent breeze for the whole field, so petals move like air is
    // moving rather than like each one has its own private weather.
    const breeze = 16 + this._noise.noise2D(t * 0.045, 3.7) * 26;

    const dpr = this._dpr;
    const W = this._w;
    const H = this._h;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this._cw, this._ch);

    for (let i = 0; i < P.length; i++) {
      const p = P[i];

      // Lateral motion = large-scale field (neighbours agree) + a private sway.
      const drift =
        this._noise.noise2D(p.x * 0.0016, t * 0.13 + p.seed) * p.driftAmp +
        Math.sin(t * p.swayF + p.swayP) * p.swayA;

      p.x += (breeze * p.depth + drift) * dt;
      p.y += p.fall * dt;
      p.phase += p.flutterF * dt;

      const fl = Math.cos(p.phase); // signed: <0 means the back face is toward us
      const afl = fl < 0 ? -fl : fl;
      // A petal spins fastest when it presents its full face to the airflow.
      p.rot += p.rotSpeed * dt * (0.45 + 0.55 * afl);

      if (p.y - p.size > H) {
        this._respawn(p, -p.size);
      } else if (p.x - p.size > W) {
        p.x = -p.size;
      } else if (p.x + p.size < 0) {
        p.x = W + p.size;
      }

      // Back faces are in their own shadow; edge-on petals catch a highlight as
      // they flick through the light. Both are what makes real petals sparkle.
      const e = 1 - afl;
      let a = p.alpha * (fl < 0 ? 0.74 : 1) * (1 + 0.55 * e * e * e);
      if (a > 0.96) a = 0.96;

      // Keep the matrix invertible when the petal is perfectly edge-on.
      const sxf = afl < 0.07 ? (fl < 0 ? -0.07 : 0.07) : fl;
      const c = Math.cos(p.rot);
      const s = Math.sin(p.rot);
      const sx = p.size * sxf * dpr;
      const sy = p.size * dpr;

      ctx.globalAlpha = a;
      ctx.setTransform(c * sx, s * sx, -s * sy, c * sy, p.x * dpr, p.y * dpr);
      ctx.drawImage(p.sprite, -0.5, -0.5, 1, 1);
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
  }

  // -------------------------------------------------------------------------
  // Petal field construction
  // -------------------------------------------------------------------------

  _respawn(p, y) {
    const r = this._rng;
    const m = this._maxSize;
    p.x = -m + r() * (this._w + m * 2);
    p.y = y;
    p.phase = r() * TAU;
    p.rot = r() * TAU;
    p.swayP = r() * TAU;
  }

  /**
   * Density scales with viewport area so a 4K window is not sparse and a small
   * one is not a blizzard. Petals are created in depth order and stay that way
   * (respawning preserves depth), so the draw loop is implicitly back-to-front
   * with no per-frame sort.
   */
  _buildPetals() {
    if (!this.ctx || !this._sprites) return;
    const r = this._rng;
    const P = this._petals;
    P.length = 0;

    const count = Math.round(clamp((this._w * this._h) / 34000, 20, 46));
    const bokeh = Math.round(clamp(count * 0.12, 2, 5));
    this._petalCount = count;

    // Petal size tracks viewport width, otherwise the field looks like a
    // different scene at 1280 than it does at 3840.
    const ui = clamp(this._w / 1280, 0.72, 1.2);
    let maxSize = 0;

    const sharp = this._sprites.sharp;
    for (let i = 0; i < count; i++) {
      const depth = (i + 0.5) / count; // 0 = far haze, 1 = near
      const size = lerp(13, 42, Math.pow(depth, 1.35)) * ui;
      if (size > maxSize) maxSize = size;
      P.push({
        x: -1,
        y: 0,
        depth,
        size,
        sprite: sharp[(r() * sharp.length) | 0],
        alpha: lerp(0.22, 0.72, depth),
        fall: lerp(28, 95, depth) * ui,
        rot: 0,
        rotSpeed: r.range(-0.5, 0.5),
        phase: 0,
        flutterF: r.range(0.5, 1.5) * (0.6 + depth * 0.6),
        swayF: r.range(0.35, 0.9),
        swayP: 0,
        swayA: lerp(6, 22, depth),
        driftAmp: lerp(8, 26, depth),
        seed: r() * 100,
      });
    }

    // A handful of large, very transparent, pre-blurred petals in front of the
    // focal plane. Fake depth of field for the price of five drawImage calls —
    // it is the single cheapest thing that stops a 2D layer looking like a 2D layer.
    const soft = this._sprites.soft;
    for (let i = 0; i < bokeh; i++) {
      const depth = 1.15 + (i / Math.max(1, bokeh)) * 0.35;
      const size = lerp(84, 190, r()) * ui;
      if (size > maxSize) maxSize = size;
      P.push({
        x: -1,
        y: 0,
        depth,
        size,
        sprite: soft[(r() * soft.length) | 0],
        alpha: r.range(0.085, 0.16),
        fall: lerp(110, 190, r()) * ui,
        rot: 0,
        rotSpeed: r.range(-0.25, 0.25),
        phase: 0,
        flutterF: r.range(0.25, 0.5),
        swayF: r.range(0.2, 0.5),
        swayP: 0,
        swayA: 26,
        driftAmp: 30,
        seed: r() * 100,
      });
    }

    this._maxSize = maxSize;

    // Seed the field mid-fall so the screen is never briefly empty at t=0.
    for (let i = 0; i < P.length; i++) {
      this._respawn(P[i], r() * (this._h + this._maxSize * 2) - this._maxSize);
    }

    // Reduced motion gets one static, composed frame instead of nothing.
    if (this._reduced) this._drawStatic();
  }

  /** Single frame for prefers-reduced-motion; the tick never touches the canvas then. */
  _drawStatic() {
    const wasReduced = this._reduced;
    this._reduced = false;
    this._stepPetals(0);
    this._reduced = wasReduced;
  }

  // -------------------------------------------------------------------------
  // Layout
  // -------------------------------------------------------------------------

  _onResize() {
    this._resize();
  }

  _resize() {
    if (this._dead) return;
    const w = Math.max(1, window.innerWidth);
    const h = Math.max(1, window.innerHeight);
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);

    const sizeChanged = Math.abs(w - this._w) > this._w * 0.25 || this._petals.length === 0;
    this._w = w;
    this._h = h;
    this._dpr = dpr;

    if (this.canvas) {
      const cw = Math.max(1, Math.round(w * dpr));
      const ch = Math.max(1, Math.round(h * dpr));
      if (cw !== this._cw || ch !== this._ch) {
        this._cw = cw;
        this._ch = ch;
        // Assigning width/height also clears the canvas and resets its state,
        // which is exactly what we want here.
        this.canvas.width = cw;
        this.canvas.height = ch;
        if (this._reduced && this._petals.length) this._drawStatic();
      }
    }

    // Rebuild only on a real change of scale — a rebuild teleports every petal,
    // and window drags fire resize dozens of times a second.
    const wantCount = Math.round(clamp((w * h) / 34000, 20, 46));
    if (sizeChanged || wantCount !== this._petalCount) this._buildPetals();

    this._measureRail();
  }

  /** The one layout read in this file, and it only happens on show/resize. */
  _measureRail() {
    if (!this.railEl) return;
    const width = this.railEl.getBoundingClientRect().width;
    if (width > 0) {
      this._railW = width;
      this.headEl.style.transform = `translate3d(${(this._shown * width).toFixed(1)}px,0,0)`;
    }
  }

  _setLabel(text) {
    if (text === this._labelText) return;
    this._labelText = text;
    this.labelEl.textContent = text;
    // Restart the swap keyframe. The reflow is on a single small inline
    // element and happens a dozen times over the whole load.
    this.labelEl.classList.remove('is-swap');
    void this.labelEl.offsetWidth;
    this.labelEl.classList.add('is-swap');
  }

  // -------------------------------------------------------------------------

  _teardown() {
    if (this._dead) return;
    this._dead = true;

    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = 0;
    window.removeEventListener('resize', this._onResize);

    // Release the full-screen backing store immediately; on an iGPU that is
    // real memory bandwidth the scene wants back.
    if (this.canvas) {
      this.canvas.width = 1;
      this.canvas.height = 1;
    }
    if (this.el && this.el.parentNode) this.el.parentNode.removeChild(this.el);

    this._petals.length = 0;
    this.ctx = null;
    this._sprites = null;
  }
}
