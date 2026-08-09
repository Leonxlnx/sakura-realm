# Sakura Realm - Build Contracts

Read this fully before writing a line. Every module is built in parallel by a
different author; these contracts are the only thing keeping them integrable.

---

## 1. What we are building

A browser-based, AAA-looking real-time sakura landscape:

- Physically-based **sky** with atmospheric scattering, sun, moon, stars.
- **Volumetric clouds** that respond to a full **weather system** (clear → overcast → rain → storm).
- A **day/night cycle** with correct light colour, intensity and shadow direction.
- An **endless field of tall grass**, hundreds of thousands of blades, moving in real wind.
- One **ultra-detailed procedural sakura tree** at the origin, with bark, branches,
  blossom clusters, and **petals that detach and fall** with believable aerodynamics.
- A **player** who can walk (FPS, grounded, head bob) or fly (6DOF).
- Everything runs in a browser at a locked framerate.

Art direction: *quiet, cinematic, slightly melancholy*. Think Ghost of Tsushima's
pampas fields and Sekiro's sakura - not a bright cartoon. Prefer soft light,
strong atmospheric perspective, low-saturation greens, and pink that reads as
**pale and desaturated**, never candy/hot-pink.

---

## 2. Hard constraints - read twice

**Target GPU is an AMD Radeon 780M - an integrated laptop GPU.** This is the single
most important engineering constraint in the project. It has no dedicated VRAM and
roughly the fill rate of a 2016 midrange discrete card.

Non-negotiable consequences:

- **Fill rate is the enemy.** Every fullscreen pass costs real milliseconds.
  Volumetrics render at half resolution or less, then upscale.
- **Draw calls must stay low.** Instance aggressively. Target < 250 draw calls total.
- **No unbounded loops in fragment shaders.** Raymarch step counts come from
  `state.quality.*` and must have a compile-time constant upper bound.
- **Everything must degrade.** Implement `onQualityChange(quality)` so the LOW tier
  actually runs. LOW must hold 60fps at 1080p on this iGPU; ULTRA may target 30.
- **Budget: 16.6 ms total frame time at HIGH, 1080p.** Rough allocation:
  clouds 3.5 ms · grass 3.5 ms · shadows 2 ms · terrain+tree 2 ms · post 3 ms · rest 2 ms.

**No external asset downloads.** There is no network fetch of textures or models.
Every texture is **generated procedurally at runtime** (canvas 2D or a GPU pass)
via `core/textures.js`, and every mesh is generated in code. This is a hard rule - 
the project must run from `file://`-adjacent local dev with zero assets on disk.

---

## 3. Environment

| Thing | Value |
|---|---|
| three.js | **0.180.0** (`import * as THREE from 'three'`) |
| postprocessing | **6.39.4** (pmndrs - merged effect passes, much faster than three's EffectComposer on iGPU) |
| n8ao | **1.10.3** (`N8AOPostPass`) |
| bundler / dev server | Vite 6 (`npm run dev` → http://localhost:5173) |
| module system | native ESM, bare imports resolved by Vite |

Addons import from the deep path, e.g.
`import { CSM } from 'three/addons/csm/CSM.js'` is **not** available in 0.180 - 
verify any addon exists in `node_modules/three/examples/jsm/` before importing it.
When in doubt, implement it yourself rather than importing something that may not exist.

Three r180 notes that bite people:
- `renderer.outputColorSpace = THREE.SRGBColorSpace` (not `outputEncoding`).
- `THREE.ColorManagement.enabled` is `true` by default; author colours in sRGB via
  `new THREE.Color('#rrggbb')` and they are converted to linear automatically.
- `renderer.useLegacyLights` is gone. Lights are physically based; `intensity` on a
  `DirectionalLight` is in lux-ish units.
- `WebGLRenderer.outputColorSpace` interacts with postprocessing: when the
  `postprocessing` library owns the final pass, set the renderer to
  `THREE.LinearSRGBColorSpace` **only if** the pipeline applies its own output
  conversion. Follow whatever `post/pipeline.js` establishes.

---

## 4. System lifecycle contract

Every module exports **one named class** (name fixed per the ownership table below).

```js
export class MySystem {
  constructor(ctx) {}            // cheap, synchronous. No GPU allocation of size.
  async init() {}                // optional: heavy setup (texture bake, geometry gen)
  link(systems) {}               // optional: grab sibling systems, called after all init()
  update(dt, state) {}           // optional: per-frame. dt is seconds, already clamped.
  resize(width, height) {}       // optional
  onQualityChange(quality) {}    // optional: quality === state.quality
  dispose() {}                   // optional: free geometries, materials, render targets
}
```

`ctx` contains:

```js
{ canvas, uiRoot, renderer, scene, camera, state, bus, input, quality, textures, systems }
```

- `state` - the world state object from `core/state.js`. **Read the file.** It is the contract.
- `bus` - `EventBus` from `core/state.js`. Use `EVENTS.*` constants, never raw strings.
- `textures` - `TextureFactory`, the only place procedural textures are created/cached.
- `systems` - sibling registry, only safe to use from `link()` onward.

### State ownership

`core/state.js` marks every field with `@owner`. **Write only the fields you own.**
Reading anything is always fine. If you need to influence a field you do not own,
call the owner's method - do not write it behind their back.

---

## 5. File ownership map

Each file has exactly one author. **Do not create, edit, or delete any file outside
your own list.** If you need something from another module, code against the
contract in this document and assume it will exist.

| File | Exported class | Owns |
|---|---|---|
| `core/engine.js` | `Engine` | renderer, scene, camera, lighting rig, shadows |
| `core/quality.js` | `QualityManager` | quality tiers, adaptive resolution, perf stats |
| `core/textures.js` | `TextureFactory` | all procedural textures |
| `sky/atmosphere.js` | `Atmosphere` | sky dome shader, scattering, ambient colours |
| `sky/celestial.js` | `Celestial` | sun/moon bodies, star field, `state.sun`/`state.moon` |
| `sky/daynight.js` | `DayNightCycle` | `state.time.timeOfDay`, lighting keyframes |
| `sky/clouds.js` | `VolumetricClouds` | raymarched cloud layer |
| `weather/wind.js` | `WindField` | `state.wind`, incl. the real `sample()` |
| `weather/weather.js` | `WeatherSystem` | weather state machine, `state.weather`, `state.clouds` |
| `weather/precipitation.js` | `Precipitation` | rain, snow, splashes |
| `weather/atmosfx.js` | `AtmosphericFX` | fog, mist, lightning, godray occluders |
| `world/terrain.js` | `Terrain` | infinite ground, `getHeight(x,z)`, ground material |
| `world/grass.js` | `GrassField` | instanced grass, chunking, LOD, wind response |
| `world/scatter.js` | `Scatter` | rocks, flowers, fallen-petal ground layer, distant treeline |
| `tree/sakura.js` | `SakuraTree` | tree orchestration, branch geometry, bark, blossoms |
| `tree/petals.js` | `PetalSystem` | falling petal simulation + rendering |
| `player/controller.js` | `PlayerController` | walk + fly movement, camera rig, `state.player` |
| `post/pipeline.js` | `PostPipeline` | postprocessing stack, exposes `render(dt, state)` |
| `ui/hud.js` | `HUD` | HUD, settings panel, keybind help |
| `ui/loading.js` | `LoadingScreen` | loading screen (`show/setProgress/hide/fail`) |
| `ui/styles.css` | - | all CSS |

Shared, already written, **do not edit**: `core/state.js`, `core/math.js`,
`core/input.js`, `src/main.js`, `index.html`, `vite.config.js`.

`core/math.js` gives you: `createNoise(seed)` → `{noise2D, noise3D, fbm2D, fbm3D,
ridged2D, billow2D, curl2D, curl3D}`, plus `worley2D/3D`, `makeRNG`, `halton`,
`fibonacciSphere/Disc`, `clamp/lerp/smoothstep/damp/remap`, `kelvinToRGB`,
`RollingAverage`, `Easing`. **Use it - do not reimplement noise.**

---

## 6. Cross-module interfaces you may rely on

These are guaranteed. Code against them.

```js
// world/terrain.js
terrain.getHeight(x, z) -> number            // synchronous, cheap, analytic
terrain.getNormal(x, z, out) -> Vector3      // synchronous

// weather/wind.js  (also installed onto state.wind.sample)
wind.sample(x, z, t, out) -> {x, y, z}       // world-space wind velocity
wind.getGustAt(x, z, t) -> number            // 0..2 multiplier

// tree/sakura.js
tree.position -> Vector3                     // trunk base, world space
tree.getBlossomSpawnPoints() -> Float32Array // xyz triplets, blossom world positions
tree.canopyRadius -> number
tree.canopyCenter -> Vector3

// sky/atmosphere.js
atmosphere.getSkyColor(direction, out) -> Color   // CPU-side approximation, for IBL/fog
atmosphere.envTexture -> Texture | null           // cube/equirect env map if generated

// core/textures.js
textures.get(name, generatorFn, options) -> Texture   // memoised by name
textures.noise2D(size, opts) -> DataTexture
textures.noise3D(size, opts) -> Data3DTexture
textures.canvas(size, drawFn) -> CanvasTexture

// core/quality.js
quality.tier -> 'low'|'medium'|'high'|'ultra'
quality.set(tier)
quality.registerCost(label, ms)   // optional profiling hook

// post/pipeline.js
post.render(dt, state)                       // performs the actual draw
post.registerGodRayLight(object3D)           // sun mesh for volumetric light shafts
```

---

## 7. Shader conventions

- Write GLSL as tagged template strings or plain string constants inside the module.
  Do not add a GLSL loader plugin.
- GLSL ES 3.0 is available (three r180 uses WebGL2). Use `#version 300 es`? **No** - 
  three injects its own version directive. Write GLSL1-style three shaders and let
  three transpile, OR use `glslVersion: THREE.GLSL3` on `RawShaderMaterial` if you
  need `textureLod`/`texture()` explicitly. Be consistent within a file.
- Prefix custom uniforms with the system, e.g. `uCloudCoverage`, `uGrassWind`.
- Any material that must react to wind/time needs its uniforms updated in `update()`.
- **Extending three's built-in materials:** use `material.onBeforeCompile` and patch
  with `.replace()` on known chunk markers, or use `MeshStandardMaterial` +
  `customProgramCacheKey`. Do not fork three's shader source wholesale.

---

## 8. Performance rules

1. **Instance everything repeated.** `InstancedMesh` or `InstancedBufferGeometry`.
2. **No per-frame allocation.** No `new Vector3()` inside `update()`. Hoist scratch
   objects to module or instance scope. This is the #1 cause of GC hitches.
3. **No `Math.random()` for anything spatial.** Use `makeRNG(seed)` so the world is
   deterministic and chunks are stable across streaming.
4. Reuse geometries and materials. One material per visual family.
5. Frustum-cull manually for instanced chunks; set correct `boundingSphere`.
6. Render targets: allocate once, resize on `resize()`, never per frame.
7. Prefer `HalfFloatType` render targets over `FloatType` - the 780M pays for float.

---

## 9. Code style

- Modern ESM, no TypeScript, no build-time codegen.
- Comments explain **why**, not what. Dense shader math deserves a citation or a
  one-line intuition, not a restatement of the code.
- No placeholder comments (`// TODO: implement`), no stub functions, no truncation.
  **Ship complete, working code.** If a feature is too expensive, implement a
  cheaper real version - never an empty one.
- Guard against WebGL context loss and missing extensions where cheap to do so.
- Keep files under ~900 lines; split helpers into a sibling file **that you own**
  (declare it in your task) if a module gets larger.

---

## 10. Definition of done for a module

- It runs without console errors.
- It looks demonstrably better than a naive implementation - that is the entire
  point of this project.
- It responds correctly to time of day, weather, and wind where relevant.
- It has all four quality tiers wired and LOW genuinely reduces cost.
- It disposes cleanly.
- No allocation in the hot path.
