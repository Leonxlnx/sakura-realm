<div align="center">

# Sakura Realm

**A real-time cherry blossom landscape that runs in your browser.**
Volumetric sky, dynamic weather, an endless wind-driven meadow, and one
procedurally grown sakura tree — with **zero art assets**.

Every texture is generated at runtime. Every mesh is built in code.
The whole world ships as source.

[Quick start](#quick-start) · [Controls](#controls) · [How it works](#how-it-works) · [Performance](#performance)

</div>

---

![The meadow at midday](docs/01-field.png)

## What this is

A single-page WebGL2 scene built on [three.js](https://threejs.org/). There are no
`.glb` models, no downloaded textures, no image files of any kind in the render path —
the bark, the soil, the petals, the star map and the cloud noise are all synthesised
on load, and the tree and every blade of grass are generated procedurally.

- **Procedural sakura** — a recursive branch skeleton grown against a crown envelope,
  swept into welded tapered tubes, carrying ~500,000 instanced blossoms.
- **Volumetric clouds** — raymarched at reduced resolution with temporal reprojection,
  Beer–Powder scattering and a dual-lobe Henyey–Greenstein phase.
- **Endless meadow** — hundreds of thousands of instanced blades streamed in chunks
  around the camera, bending in a shared, divergence-free wind field.
- **Full day/night cycle** — physically-based Rayleigh/Mie sky, sun and moon with real
  phase, star field, and auto-exposure across the whole 24 hours.
- **Weather system** — clear through overcast, rain, storm, fog and a petal storm, all
  continuously blended rather than switched.
- **Falling petals** — simulated with real aerodynamics: flutter, tumble, drag, and
  advection by the same wind field the grass and branches read.

![Golden hour over the field](docs/02-golden-hour.png)

## Quick start

```bash
npm install
npm run dev
```

Open the URL Vite prints. For a production build:

```bash
npm run build && npm run preview
```

Requires a browser with **WebGL2**.

## Controls

| Key | Action |
|---|---|
| Click | Capture mouse (pointer lock) |
| `W` `A` `S` `D` | Move |
| Mouse | Look |
| `Shift` | Sprint |
| `Space` | Jump · ascend when flying |
| `Ctrl` | Crouch · descend when flying |
| `F` | Toggle walk / fly |
| `G` | Cycle grass: lawn → meadow → tall |
| `Tab` | Settings — weather, time of day, wind, quality |
| `H` | Hide the HUD |
| `Esc` | Release the mouse |

![Under the canopy](docs/03-canopy.png)

## How it works

`src/main.js` owns system order and nothing else. Every subsystem is a class following
one lifecycle contract:

```js
constructor(ctx) → async init() → link(systems) → update(dt, state) → dispose()
```

`src/core/state.js` is the single mutable source of truth. Every field has exactly one
owning system, marked `@owner`. Systems read anything and write only what they own —
that one constraint is what keeps a scene this interconnected from collapsing into a
tangle of cross-references.

```
src/
  core/      renderer, quality tiers, procedural textures, input, state, math
  sky/       atmospheric scattering, sun/moon/stars, volumetric clouds, day–night
  weather/   wind field, weather state machine, precipitation, fog and lightning
  world/     infinite terrain, instanced grass, scatter detail, birds
  tree/      the sakura — skeleton, bark, blossoms, falling petals
  player/    walk and fly controller
  post/      post-processing pipeline
  ui/        HUD, settings panel, loading screen
```

[`CONTRACTS.md`](CONTRACTS.md) documents the full build contract: module ownership,
cross-module interfaces, shader conventions and the per-system frame budget.

### A note on the tree

The sakura is not a model — it is grown. A recursive generator produces a branch
skeleton under competing tropisms (gravity, phototropism, crowding avoidance, and a
crown envelope that decides the silhouette), radii follow the pipe model at every fork,
and branch length follows from radius through a single allometric law. The blossoms are
then placed on the resulting twig cloud by a light-transport approximation, so the crown
is dense on its lit shell and thinner inside — the way a real cherry actually flowers.

## Performance

The reference target is an **AMD Radeon 780M integrated GPU**, and that constraint drove
most of the engineering: clouds raymarch at reduced resolution with temporal
reprojection, grass streams in chunks with instance-count LOD, and post-processing is
merged into a single fullscreen pass.

Quality auto-detects on first run and adapts at runtime. All four tiers are real —
`low` genuinely costs less rather than merely looking worse — and an adaptive resolution
controller holds the target framerate under load.

```bash
node scripts/check-integration.mjs
```

Static check for export mismatches, state-ownership violations, hot-path allocation and
syntax errors across every module.

## Built with

[three.js](https://threejs.org/) · [postprocessing](https://github.com/pmndrs/postprocessing) · [Vite](https://vitejs.dev/)

## License

[MIT](LICENSE)
