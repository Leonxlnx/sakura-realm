/**
 * Static integration check for the parallel build.
 *
 * Catches the failure modes that a 20-author fan-out actually produces:
 *  - a module renamed its exported class, so main.js's import is undefined
 *  - a module still contains a generated stub
 *  - a module writes a state field it does not own
 *  - allocation inside an update()/render() hot path
 *  - leftover TODO/placeholder markers
 *  - syntax errors
 *
 * Run: node scripts/check-integration.mjs
 * Exit code is non-zero if any ERROR-level problem is found.
 */
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';

const root = join(import.meta.dirname, '..');
const src = join(root, 'src');

const errors = [];
const warnings = [];
const err = (file, msg) => errors.push(`${file}: ${msg}`);
const warn = (file, msg) => warnings.push(`${file}: ${msg}`);

// ---------------------------------------------------------------------------
// 1. Every class main.js imports must actually be exported by its module.
// ---------------------------------------------------------------------------
const mainSrc = readFileSync(join(src, 'main.js'), 'utf8');
const importRe = /import\s*\{([^}]+)\}\s*from\s*'(\.[^']+)'/g;
const modules = [];
for (const m of mainSrc.matchAll(importRe)) {
  const names = m[1].split(',').map((s) => s.trim().split(/\s+as\s+/)[0]).filter(Boolean);
  const rel = m[2].replace(/^\.\//, '');
  modules.push({ rel, path: join(src, rel), names });
}

for (const mod of modules) {
  if (!existsSync(mod.path)) {
    err(mod.rel, 'FILE MISSING - main.js imports it');
    continue;
  }
  const code = readFileSync(mod.path, 'utf8');

  for (const name of mod.names) {
    const exported =
      new RegExp(`export\\s+(class|function|const|let)\\s+${name}\\b`).test(code) ||
      new RegExp(`export\\s*\\{[^}]*\\b${name}\\b`).test(code);
    if (!exported) err(mod.rel, `does not export "${name}" - main.js import will be undefined`);
  }

  if (code.includes('@generated-stub')) err(mod.rel, 'STILL A STUB - no real implementation');

  // Placeholder markers the contract forbids.
  for (const marker of ['TODO', 'FIXME', 'not implemented', 'placeholder for', 'simplified for now']) {
    if (new RegExp(marker, 'i').test(code)) warn(mod.rel, `contains "${marker}"`);
  }

  if (code.length < 1200) warn(mod.rel, `suspiciously short (${code.length} bytes) - likely under-built`);
}

// ---------------------------------------------------------------------------
// 2. State ownership. Parse @owner annotations out of state.js, then flag any
//    module writing a field owned by a different module.
// ---------------------------------------------------------------------------
const stateSrc = readFileSync(join(src, 'core', 'state.js'), 'utf8');
/** section name -> owning file(s) */
const owners = new Map();
{
  // Matches:  // Sun  @owner sky/celestial.js
  const ownerRe = /\/\/\s*([A-Za-z/ ]+?)\s+@owner\s+([^\n]+)/g;
  for (const m of stateSrc.matchAll(ownerRe)) {
    const section = m[1].trim().toLowerCase().split(/\s+/)[0];
    owners.set(section, m[2].trim());
  }
}

// Map state section -> the file allowed to write it.
const SECTION_OWNER = {
  sun: 'sky/celestial.js',
  moon: 'sky/celestial.js',
  sky: 'sky/atmosphere.js',
  clouds: 'sky/clouds.js',
  weather: 'weather/weather.js',
  wind: 'weather/wind.js',
  player: 'player/controller.js',
  quality: 'core/quality.js',
  perf: 'core/quality.js',
};
// Fields that are legitimately written by more than one system.
const SHARED_WRITES = new Set([
  'state.player.groundHeight', // terrain writes this for the controller
  'state.sky.starIntensity',   // clouds dim stars
  'state.clouds',              // weather drives cloud params by design
  'state.weather.lightning',   // precipitation spikes it
  'state.quality.resolutionScale',
]);

for (const mod of modules) {
  if (!existsSync(mod.path)) continue;
  const code = readFileSync(mod.path, 'utf8');
  // Assignment to a state path: state.sun.intensity = ...  /  state.wind.strength +=
  const writeRe = /\bstate\.(\w+)\.(\w+)\s*(?:=[^=]|\+=|-=|\*=|\/=)/g;
  for (const m of code.matchAll(writeRe)) {
    const [full, section, field] = [m[0], m[1], m[2]];
    const owner = SECTION_OWNER[section];
    if (!owner) continue;
    if (owner === mod.rel) continue;
    const path = `state.${section}.${field}`;
    if (SHARED_WRITES.has(path) || SHARED_WRITES.has(`state.${section}`)) continue;
    warn(mod.rel, `writes ${path}, owned by ${owner}`);
  }
}

// ---------------------------------------------------------------------------
// 3. Allocation in hot paths. Crude but effective: pull the body of update()
//    and render() and look for constructor calls / literals.
// ---------------------------------------------------------------------------
function extractMethodBody(code, methodName) {
  const bodies = [];
  const re = new RegExp(`\\b${methodName}\\s*\\([^)]*\\)\\s*\\{`, 'g');
  for (const m of code.matchAll(re)) {
    let depth = 1;
    let i = m.index + m[0].length;
    const start = i;
    while (i < code.length && depth > 0) {
      const c = code[i];
      if (c === '{') depth++;
      else if (c === '}') depth--;
      i++;
    }
    bodies.push(code.slice(start, i - 1));
  }
  return bodies;
}

const ALLOC_PATTERNS = [
  [/\bnew\s+(Vector[234]|Color|Quaternion|Matrix[34]|Euler|Box3|Sphere|Ray|Plane)\b/g, 'allocates a three.js math object'],
  [/\bnew\s+(Float32Array|Uint8Array|Uint16Array|Uint32Array|Int32Array|Array)\b/g, 'allocates a typed array'],
  [/\.map\(|\.filter\(|\.slice\(|\.concat\(|\.flatMap\(/g, 'allocates via an array method'],
  [/\bObject\.(keys|values|entries|assign)\(/g, 'allocates via Object.*'],
];

for (const mod of modules) {
  if (!existsSync(mod.path)) continue;
  const code = readFileSync(mod.path, 'utf8');
  for (const method of ['update', 'render']) {
    for (const body of extractMethodBody(code, method)) {
      // Strip comments and strings so we do not match inside GLSL or docs.
      const clean = body
        .replace(/\/\*[\s\S]*?\*\//g, '')
        .replace(/\/\/[^\n]*/g, '')
        .replace(/`[\s\S]*?`/g, '``')
        .replace(/'[^']*'/g, "''")
        .replace(/"[^"]*"/g, '""');
      for (const [re, label] of ALLOC_PATTERNS) {
        const hits = clean.match(re);
        if (hits) warn(mod.rel, `${method}() ${label}: ${[...new Set(hits)].join(', ')}`);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 4. Syntax check every source file.
// ---------------------------------------------------------------------------
import { readdirSync, statSync } from 'node:fs';
function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (p.endsWith('.js')) out.push(p);
  }
  return out;
}
for (const file of walk(src)) {
  try {
    execFileSync(process.execPath, ['--check', file], { stdio: 'pipe' });
  } catch (e) {
    err(file.replace(src + '\\', '').replace(/\\/g, '/'), `SYNTAX ERROR\n${e.stderr?.toString().split('\n').slice(0, 4).join('\n')}`);
  }
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------
console.log(`\nChecked ${modules.length} modules imported by main.js\n`);
if (errors.length) {
  console.log(`ERRORS (${errors.length}):`);
  for (const e of errors) console.log('  x ' + e);
  console.log('');
}
if (warnings.length) {
  console.log(`WARNINGS (${warnings.length}):`);
  for (const w of warnings) console.log('  ! ' + w);
  console.log('');
}
if (!errors.length && !warnings.length) console.log('Clean.\n');
process.exit(errors.length ? 1 : 0);
