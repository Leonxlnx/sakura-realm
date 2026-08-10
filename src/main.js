/**
 * main.js - bootstrap and frame loop.
 *
 * This file owns SYSTEM ORDER and nothing else. Every subsystem is a class that
 * follows the lifecycle contract documented in /CONTRACTS.md:
 *
 *   constructor(ctx)          synchronous, cheap - build objects, no heavy work
 *   async init()              optional - heavy/async setup, may report progress
 *   link(systems)             optional - grab sibling references after all exist
 *   update(dt, state)         optional - per-frame work
 *   resize(width, height)     optional - viewport changed
 *   onQualityChange(quality)  optional - quality tier changed at runtime
 *   dispose()                 optional - release GPU resources
 *
 * Update order below is deliberate and load-bearing. Read the comments before
 * moving anything: several systems consume state written earlier in the same frame.
 */

import { inject as injectAnalytics } from '@vercel/analytics';

import { createWorldState, EventBus, EVENTS } from './core/state.js';
import { Input } from './core/input.js';
import { Engine } from './core/engine.js';
import { QualityManager } from './core/quality.js';
import { TextureFactory } from './core/textures.js';

import { DayNightCycle } from './sky/daynight.js';
import { Celestial } from './sky/celestial.js';
import { Atmosphere } from './sky/atmosphere.js';
import { VolumetricClouds } from './sky/clouds.js';

import { WindField } from './weather/wind.js';
import { WeatherSystem } from './weather/weather.js';
import { Precipitation } from './weather/precipitation.js';
import { AtmosphericFX } from './weather/atmosfx.js';

import { Terrain } from './world/terrain.js';
import { GrassField } from './world/grass.js';
import { Scatter } from './world/scatter.js';
import { Birds } from './world/birds.js';

import { SakuraTree } from './tree/sakura.js';
import { PetalSystem } from './tree/petals.js';

import { PlayerController } from './player/controller.js';
import { PostPipeline } from './post/pipeline.js';
import { HUD } from './ui/hud.js';
import { LoadingScreen } from './ui/loading.js';

// Vercel Web Analytics. Called once, at module scope, so a failure inside boot()
// cannot take the page view with it. The endpoint it posts to only exists on a
// Vercel deployment, so running this repo locally or self-hosting it elsewhere
// sends nothing anywhere: in dev the call only logs to the console, and in a
// self-hosted production build the request has no route to hit.
injectAnalytics({ mode: import.meta.env.DEV ? 'development' : 'production' });

async function boot() {
  const canvas = document.getElementById('viewport');
  const uiRoot = document.getElementById('ui-root');

  const state = createWorldState();
  const bus = new EventBus();
  const loading = new LoadingScreen(uiRoot);
  loading.show();

  // ---------------------------------------------------------------------------
  // Core
  // ---------------------------------------------------------------------------
  const engine = new Engine({ canvas, state, bus });
  if (!engine.supported) {
    loading.fail(
      'This scene needs WebGL2. Try Chrome or Edge with hardware acceleration enabled.'
    );
    return;
  }

  const input = new Input(canvas, bus);
  const quality = new QualityManager({ renderer: engine.renderer, state, bus });
  const textures = new TextureFactory({ renderer: engine.renderer });

  /** Shared context handed to every system. */
  const ctx = {
    canvas,
    uiRoot,
    renderer: engine.renderer,
    scene: engine.scene,
    camera: engine.camera,
    state,
    bus,
    input,
    quality,
    textures,
    /** Populated below; systems reach siblings through this in link(). */
    systems: {},
  };

  // ---------------------------------------------------------------------------
  // Construct - order here is irrelevant, only update order matters.
  // ---------------------------------------------------------------------------
  const systems = ctx.systems;
  systems.dayNight = new DayNightCycle(ctx);
  systems.celestial = new Celestial(ctx);
  systems.atmosphere = new Atmosphere(ctx);
  systems.clouds = new VolumetricClouds(ctx);
  systems.wind = new WindField(ctx);
  systems.weather = new WeatherSystem(ctx);
  systems.precipitation = new Precipitation(ctx);
  systems.atmosfx = new AtmosphericFX(ctx);
  systems.terrain = new Terrain(ctx);
  systems.grass = new GrassField(ctx);
  systems.scatter = new Scatter(ctx);
  systems.birds = new Birds(ctx);
  systems.tree = new SakuraTree(ctx);
  systems.petals = new PetalSystem(ctx);
  systems.player = new PlayerController(ctx);
  systems.post = new PostPipeline(ctx);
  systems.hud = new HUD(ctx);

  const list = Object.values(systems);

  // ---------------------------------------------------------------------------
  // Async init with progress reporting
  // ---------------------------------------------------------------------------
  /**
   * Yield to the browser so the loading screen repaints between systems.
   *
   * Deliberately races rAF against a timer: rAF does not fire at all in a hidden
   * tab, so awaiting it alone would stall boot forever for anyone who opens the
   * page in a background tab and switches to it later.
   */
  const yieldToBrowser = () =>
    new Promise((resolve) => {
      let settled = false;
      const done = () => {
        if (settled) return;
        settled = true;
        resolve();
      };
      requestAnimationFrame(done);
      setTimeout(done, 32);
    });

  const initable = list.filter((s) => typeof s.init === 'function');
  let done = 0;
  for (const system of initable) {
    const label = system.constructor.name.replace(/([a-z])([A-Z])/g, '$1 $2');
    loading.setProgress(done / initable.length, label);
    await yieldToBrowser();
    try {
      await system.init();
    } catch (err) {
      console.error(`[boot] ${system.constructor.name}.init() failed:`, err);
    }
    done++;
  }
  loading.setProgress(1, 'Ready');

  // Every system now exists and is initialised - safe to grab siblings.
  for (const system of list) system.link?.(systems);

  // ---------------------------------------------------------------------------
  // Resize
  // ---------------------------------------------------------------------------
  // A hidden tab, a display:none container, or a document that has not been laid
  // out yet all report a viewport of 0x0. Sizing render targets from that collapses
  // the post-processing chain to 1x1 and every frame comes out black - permanently,
  // because nothing re-fires once the viewport becomes real. So: fall back to a sane
  // size, and re-measure on every signal that layout may have changed.
  const FALLBACK_W = 1280;
  const FALLBACK_H = 720;
  let lastW = 0;
  let lastH = 0;

  const onResize = () => {
    let w = window.innerWidth || document.documentElement.clientWidth || 0;
    let h = window.innerHeight || document.documentElement.clientHeight || 0;
    if (w < 2 || h < 2) {
      w = FALLBACK_W;
      h = FALLBACK_H;
    }
    if (w === lastW && h === lastH) return;
    lastW = w;
    lastH = h;

    engine.resize(w, h);
    for (const system of list) system.resize?.(w, h);
    bus.emit(EVENTS.RESIZE, { width: w, height: h });
  };

  window.addEventListener('resize', onResize);
  document.addEventListener('visibilitychange', onResize);
  // Catches the case where the element gains size without a window resize event.
  if (typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(onResize).observe(document.documentElement);
  }
  onResize();

  bus.on(EVENTS.QUALITY_CHANGED, (q) => {
    for (const system of list) system.onQualityChange?.(q);
  });

  // ---------------------------------------------------------------------------
  // Frame loop
  // ---------------------------------------------------------------------------
  let last = performance.now();
  let running = true;

  /**
   * Advance and render exactly one frame.
   *
   * Split out from the rAF driver so a frame can be stepped manually - from
   * devtools, from an automated capture harness, or in a hidden tab where rAF
   * never fires. `window.SAKURA.tick(dt)` is the entry point.
   */
  function tick(dt) {
    state.time.delta = dt;
    state.time.elapsed += dt;
    state.time.frame++;

    const frameStart = performance.now();
    quality.beginFrame(frameStart);

    // -- Environment: time -> sun -> sky colours. Everything downstream reads these.
    systems.dayNight.update(dt, state);
    systems.celestial.update(dt, state);
    systems.atmosphere.update(dt, state);

    // -- Weather drives cloud/fog/wetness targets; wind consumes weather strength.
    systems.weather.update(dt, state);
    systems.wind.update(dt, state);

    // -- Player moves first so world streaming sees this frame's camera position.
    systems.player.update(dt, state);

    // -- Ground before anything that samples ground height.
    systems.terrain.update(dt, state);
    systems.grass.update(dt, state);
    systems.scatter.update(dt, state);
    systems.birds.update(dt, state);

    // -- Tree, then petals (petals spawn from the tree's current blossom positions).
    systems.tree.update(dt, state);
    systems.petals.update(dt, state);

    // -- Sky volumetrics and precipitation last among world systems.
    systems.clouds.update(dt, state);
    systems.precipitation.update(dt, state);
    systems.atmosfx.update(dt, state);

    systems.hud.update(dt, state);

    // -- Render through the post stack.
    systems.post.render(dt, state);

    quality.endFrame(performance.now());
    input.endFrame();
  }

  /** rAF driver. Clamps dt so a tab switch or GC pause cannot teleport the player. */
  function frame(now) {
    if (!running) return;
    requestAnimationFrame(frame);
    const rawDt = (now - last) / 1000;
    last = now;
    tick(Math.min(rawDt, 1 / 20));
  }

  loading.hide();
  requestAnimationFrame((t) => {
    last = t;
    frame(t);
  });

  // ---------------------------------------------------------------------------
  // Teardown (mostly for HMR during development)
  // ---------------------------------------------------------------------------
  const dispose = () => {
    running = false;
    window.removeEventListener('resize', onResize);
    for (const system of list) system.dispose?.();
    input.dispose();
    engine.dispose();
    bus.clear();
  };

  if (import.meta.hot) import.meta.hot.dispose(dispose);

  // Handy for poking at the scene from devtools.
  window.SAKURA = { ctx, systems, state, engine, dispose, tick };

  // Dev-only capture/benchmark harness. Tree-shaken out of production builds.
  if (import.meta.env.DEV) {
    import('./dev/capture.js')
      .then((m) => m.installCapture(window.SAKURA))
      .catch((err) => console.warn('[capture] unavailable:', err));
  }
}

boot().catch((err) => {
  console.error('[boot] fatal:', err);
  const root = document.getElementById('ui-root');
  if (root) {
    root.innerHTML = `<div class="fatal-error"><h1>Failed to start</h1><pre>${
      err?.stack || err
    }</pre></div>`;
  }
});
