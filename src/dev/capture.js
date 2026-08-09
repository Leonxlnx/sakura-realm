/**
 * capture.js - dev-only rendering harness.
 *
 * The scene can only be judged by looking at it, and a hidden/headless tab never
 * composites, so ordinary screenshots are unavailable. This drives the frame loop
 * manually and POSTs the canvas to the dev server's /__shot endpoint (see
 * vite.config.js), which writes it to captures/<name>.jpg.
 *
 * Loaded only when import.meta.env.DEV is true - never part of a production build.
 *
 *   await SAKURA_CAPTURE.shoot('dawn', { tod: 5.4, pos: [0, 2, 24], look: [0, 8, 0] })
 *   await SAKURA_CAPTURE.sequence()        // the standard review set
 */

/** Shots that together exercise the full lighting, weather and scale range. */
export const REVIEW_SET = [
  { name: 'a-dawn',      tod: 5.4,  weather: 'clear',        pos: [0, 2.0, 26],  look: [0, 8, 0] },
  { name: 'b-morning',   tod: 8.5,  weather: 'partlyCloudy', pos: [0, 2.0, 26],  look: [0, 8, 0] },
  { name: 'c-noon',      tod: 12.5, weather: 'partlyCloudy', pos: [0, 2.0, 22],  look: [0, 9, 0] },
  { name: 'd-golden',    tod: 18.4, weather: 'clear',        pos: [14, 2.0, 18], look: [0, 8, 0] },
  { name: 'e-night',     tod: 23.5, weather: 'clear',        pos: [0, 2.0, 24],  look: [0, 9, 0] },
  { name: 'f-storm',     tod: 15.0, weather: 'storm',        pos: [0, 2.0, 28],  look: [0, 9, 0] },
  { name: 'g-fog',       tod: 6.6,  weather: 'fog',          pos: [0, 1.8, 20],  look: [0, 5, 0] },
  { name: 'h-grass',     tod: 17.6, weather: 'clear',        pos: [8, 0.9, 12],  look: [-6, 1.1, -2] },
  { name: 'i-canopy',    tod: 16.0, weather: 'clear',        pos: [3, 1.7, 7],   look: [0, 11, 0] },
  { name: 'j-aerial',    tod: 10.0, weather: 'partlyCloudy', pos: [0, 46, 62],   look: [0, 6, 0] },
];

export function installCapture(sakura) {
  const { ctx, state, systems, tick } = sakura;

  /**
   * Render one configured frame and write it to captures/<name>.jpg.
   *
   * The player controller owns the camera and rewrites it every tick, so it is
   * temporarily neutered - otherwise any camera override is undone before the draw.
   */
  async function shoot(name, opts = {}) {
    const player = systems.player;
    const savedUpdate = player.update;
    const savedSpeed = state.time.daySpeed;
    if (opts.freezePlayer !== false) player.update = () => {};

    try {
      if (opts.tod !== undefined) {
        state.time.timeOfDay = opts.tod;
        state.time.daySpeed = 0;
      }
      if (opts.weather) systems.weather.setWeather?.(opts.weather, opts.blend ?? 0.01);
      if (opts.wind !== undefined) {
        state.wind.strength = opts.wind;
        state.wind.gust = 1;
        state.wind.turbulence = opts.turbulence ?? state.wind.turbulence;
      }

      const place = () => {
        if (opts.pos) ctx.camera.position.set(opts.pos[0], opts.pos[1], opts.pos[2]);
        if (opts.look) ctx.camera.lookAt(opts.look[0], opts.look[1], opts.look[2]);
      };

      // Warm-up ticks let chunk streaming, temporal reprojection, auto-exposure and
      // weather blending settle. Without this every shot shows a half-converged frame.
      const warm = opts.warm ?? 90;
      for (let i = 0; i < warm; i++) {
        place();
        tick(1 / 60);
      }
      place();
      ctx.camera.updateMatrixWorld(true);
      tick(1 / 60);

      const canvas = ctx.renderer.domElement;
      const url = canvas.toDataURL('image/jpeg', opts.quality ?? 0.85);
      const res = await fetch(`/__shot?name=${encodeURIComponent(name)}`, {
        method: 'POST',
        body: url,
      });
      const json = await res.json();
      return { name, ok: json.ok, bytes: json.bytes, fps: Math.round(state.perf.fps) };
    } finally {
      player.update = savedUpdate;
      state.time.daySpeed = savedSpeed;
    }
  }

  /** Run a list of shots (defaults to REVIEW_SET) sequentially. */
  async function sequence(list = REVIEW_SET, opts = {}) {
    const out = [];
    for (const shot of list) {
      out.push(await shoot(shot.name, { ...shot, ...opts }));
    }
    return out;
  }

  /** Measure steady-state frame cost at a given resolution. */
  function benchmark({ width = 1920, height = 1080, frames = 60, warm = 30 } = {}) {
    const gl = ctx.renderer.getContext();
    ctx.renderer.setSize(width, height, false);
    ctx.camera.aspect = width / height;
    ctx.camera.updateProjectionMatrix();
    for (let i = 0; i < warm; i++) tick(1 / 60);
    gl.finish();
    const t0 = performance.now();
    for (let i = 0; i < frames; i++) tick(1 / 60);
    gl.finish();
    const ms = (performance.now() - t0) / frames;
    const info = ctx.renderer.info;
    return {
      msPerFrame: +ms.toFixed(2),
      fps: Math.round(1000 / ms),
      drawCalls: info.render.calls,
      triangles: info.render.triangles,
      programs: info.programs?.length ?? -1,
      textures: info.memory.textures,
      geometries: info.memory.geometries,
    };
  }

  window.SAKURA_CAPTURE = { shoot, sequence, benchmark, REVIEW_SET };
  console.info('[capture] SAKURA_CAPTURE.shoot / .sequence / .benchmark ready');
}
