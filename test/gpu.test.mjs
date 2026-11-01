// GPU mechanism tests. They drive the built page (dist/plate-player.html) in
// Chromium through `window.plate`. Keep them cheap and about mechanisms with
// known answers; high-level behaviour is benchmarked once per design phase,
// not kept here (see README, "What belongs in the test suite").
//
//   npm run build && npm run test:gpu
//
// Needs Playwright with Chromium (`npx playwright install chromium` if it is
// not already installed). By default the browser uses SwiftShader, a CPU
// implementation of WebGPU, so results do not depend on the local GPU; set
// PLATE_GPU=hardware to use the real GPU. Headless Chromium cannot present a
// WebGPU canvas, so the browser runs headed: on a machine without a display,
// wrap the command in `xvfb-run -a`.

import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Edge, Particle, SLOTS, unpack } from '../src/gpu/layout.js';
import { checkEdgeStorage } from '../src/edges/topology.js';
import { readZip } from '../src/record/zip.js';
import { createIcosphere } from '../src/geometry/icosphere.js';
import { CHAIN, chainExact, chainState } from './support/chain.mjs';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const PAGE = pathToFileURL(fileURLToPath(new URL('../dist/plate-player.html', import.meta.url))).href + '?paused';
const MAX_PARTICLES = 25000;
const MAX_EDGES = 131072;         // CAPACITY in src/sim/simulation.js
const ARGS = process.env.PLATE_GPU === 'hardware'
  ? ['--enable-unsafe-webgpu']
  : ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--disable-vulkan-surface'];

let browser;
test.before(async () => { browser = await chromium.launch({ headless: false, args: ARGS }); });
test.after(async () => { await browser?.close(); });

async function openPage() {
  const page = await browser.newPage({ viewport: { width: 640, height: 480 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error' && !m.text().includes('favicon')) errors.push(m.text()); });
  await page.goto(PAGE);
  await page.waitForFunction(() => window.plate?.recorder.samples.length > 0, null, { timeout: 120_000 });
  page.errors = errors;
  return page;
}

// ArrayBuffers cross the page boundary as base64.
const toB64 = `(buf) => { const b = new Uint8Array(buf); let s = ''; for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode(...b.subarray(i, i + 0x8000)); return btoa(s); }`;
const fromB64 = (s) => { const b = Buffer.from(s, 'base64'); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };

async function readState(page) {
  const b64 = await page.evaluate(async (toB64Src) => {
    const toB64 = eval(toB64Src);
    const bufs = await plate.readBuffers(['particles', 'edges', 'particlePool', 'edgePool', 'grid', 'deltas']);
    return Object.fromEntries(Object.entries(bufs).map(([k, v]) => [k, toB64(v)]));
  }, toB64);
  return Object.fromEntries(Object.entries(b64).map(([k, v]) => [k, fromB64(v)]));
}

function checkFreeLists(state) {
  const P = unpack(Particle, state.particles, MAX_PARTICLES);
  const pool = new Uint32Array(state.particlePool);
  const count = pool[0];
  assert.equal(pool[1], 0, 'particle pool: split requests not cleared');
  const free = new Uint8Array(MAX_PARTICLES);
  for (let k = 0; k < count; k++) {
    const idx = pool[4 + k];
    assert.ok(idx < MAX_PARTICLES, `particle pool holds out-of-range index ${idx}`);
    assert.equal(free[idx], 0, `particle slot ${idx} is in the free list twice`);
    assert.equal(P.is_active[idx], 0, `particle slot ${idx} is both free and active`);
    free[idx] = 1;
  }
  let activeCount = 0;
  for (let i = 0; i < MAX_PARTICLES; i++) activeCount += P.is_active[i] ? 1 : 0;
  assert.equal(activeCount + count, MAX_PARTICLES, `particles: ${activeCount} active + ${count} free != ${MAX_PARTICLES}`);

  // Edges: free list, no duplicate pairs, slot lists symmetric (design 3.2).
  const E = unpack(Edge, state.edges, MAX_EDGES);
  const problems = checkEdgeStorage({
    E, pool: new Uint32Array(state.edgePool), slots: new Uint32Array(state.deltas), layout: SLOTS, particleActive: P.is_active,
  });
  assert.deepEqual(problems.slice(0, 10), [], `${problems.length} edge storage problems`);
  let cohesive = 0, loose = 0;
  for (let e = 0; e < MAX_EDGES; e++) if (E.is_active[e]) (E.integrity[e] > 0 ? cohesive++ : loose++);
  assert.ok(cohesive > 0 && loose > 0, `edges: ${cohesive} cohesive, ${loose} without cohesion`);
  return P;
}

test('free lists account for every slot; cached mantle node matches brute force', { timeout: 300_000 }, async () => {
  const page = await openPage();
  await page.evaluate(() => plate.runSteps(20));
  const state = await readState(page);
  const P = checkFreeLists(state);

  // The node is located at the start of a step and the particle then moves
  // a little, so compare against brute force with a small tolerance and
  // require nearly all to match exactly.
  const vertices = createIcosphere(4).vertices;
  let exact = 0, total = 0, worst = 0;
  for (let i = 0; i < MAX_PARTICLES; i++) {
    if (!P.is_active[i]) continue;
    const x = P.position[4 * i], y = P.position[4 * i + 1], z = P.position[4 * i + 2];
    let best = 0, bestD = Infinity;
    vertices.forEach((v, k) => { const d = Math.hypot(v[0] - x, v[1] - y, v[2] - z); if (d < bestD) { bestD = d; best = k; } });
    const v = vertices[P.mantle_node[i]];
    const d = Math.hypot(v[0] - x, v[1] - y, v[2] - z);
    worst = Math.max(worst, d - bestD);
    exact += P.mantle_node[i] === best ? 1 : 0;
    total++;
  }
  console.log(`  node cache: ${exact}/${total} exact, worst excess distance ${worst.toExponential(2)}`);
  assert.ok(exact / total > 0.98, `only ${exact}/${total} cached nodes are the nearest`);
  assert.ok(worst < 0.02, `cached node too far from nearest: ${worst}`);
  assert.deepEqual(page.errors, []);
  await page.close();
});

test('a 10x time step stays finite and bounded', { timeout: 300_000 }, async () => {
  const page = await openPage();
  await page.evaluate(async () => { await plate.setParams({ dt_crust_myr: 0.16 }); await plate.runSteps(10); });
  const state = await readState(page);
  assert.deepEqual(page.errors, []);
  await page.close();

  const P = checkFreeLists(state);
  const speeds = [];
  for (let i = 0; i < MAX_PARTICLES; i++) {
    if (!P.is_active[i]) continue;
    const r = Math.hypot(P.position[4 * i], P.position[4 * i + 1], P.position[4 * i + 2]);
    assert.ok(Math.abs(r - 1) < 1e-4, `particle ${i} left the sphere (r = ${r})`);
    const v = Math.hypot(P.velocity[4 * i], P.velocity[4 * i + 1], P.velocity[4 * i + 2]);
    assert.ok(Number.isFinite(v), `particle ${i} velocity is not finite`);
    speeds.push(v * 637.1); // cm/yr
  }
  // Splits move a particle ~1 apothem within a step, which reads as a large
  // one-step "velocity" for that particle; a blow-up would move many particles.
  speeds.sort((a, b) => a - b);
  const p99 = speeds[Math.floor(0.99 * (speeds.length - 1))];
  console.log(`  particle speed p99 ${p99.toFixed(1)} cm/yr, max ${speeds.at(-1).toFixed(1)} cm/yr`);
  assert.ok(p99 < 30, `p99 particle speed ${p99} cm/yr`);
});

test('two bonded particles relax as implicit Euler with tau = gamma / 2k', { timeout: 300_000 }, async () => {
  // One stretched bond, no mantle flow. Drag-as-mass XPBD should reproduce
  // implicit Euler for  dC/dt = -C / tau  exactly: C_n = C_0 (1 + dt/tau)^-n,
  // for time steps far below, at, and far above tau.
  const page = await openPage();
  const results = await page.evaluate(async () => {
    const a = 0.0175, L = 3 * a, eps = 0.1;
    // base_apothem = L/2: these particles have the reference drag (w = 1).
    await plate.setParams({ mantle_drive: 0, crust_youngs_modulus_gpa: 70, base_apothem: L / 2 });
    const kappa70 = plate.sim.params.bond_rate_myr_inv;
    await plate.setParams({ crust_youngs_modulus_gpa: 70 * 0.5 / kappa70 }); // kappa = 0.5/Myr, tau = 1/(2 kappa) = 1 Myr
    const tau = 1 / (2 * plate.sim.params.bond_rate_myr_inv);
    const half = Math.asin(L * (1 + eps) / 2);
    const state = () => ({
      particles: {
        count: 2,
        position: Float32Array.from([-Math.sin(half), 0, Math.cos(half), Math.sin(half), 0, Math.cos(half)]),
        thickness: Float32Array.of(7, 7), compositionDensity: Float32Array.of(3000, 3000),
        // Age 100 Myr: oceanic lithosphere at its 100 km cap, so the drag equals the reference (w = 1).
        temperature: Float32Array.of(278, 278), age: Float32Array.of(100, 100), apothem: Float32Array.of(L / 2, L / 2), // rest length = apothem sum
      },
      bonds: { count: 1, a: Uint32Array.of(0), b: Uint32Array.of(1), damage: Float32Array.of(0) },
      yieldStressMPa: 1e6,
    });
    const stretch = async () => {
      const buf = (await plate.readBuffers(['particles'])).particles;
      const f = new Float32Array(buf);
      const w = buf.byteLength / 4 / 25000; // words per particle
      return Math.hypot(f[w] - f[0], f[w + 1] - f[1], f[w + 2] - f[2]) - L;
    };
    const out = [];
    for (const [dt, steps] of [[0.1, 10], [1, 3], [10, 1]]) {
      await plate.setParams({ dt_crust_myr: dt });
      await plate.loadState(state());
      const c0 = await stretch();
      await plate.runSteps(steps);
      out.push({ dt, steps, tau, ratio: (await stretch()) / c0, expected: (1 + dt / tau) ** -steps });
    }
    return out;
  });
  for (const r of results) {
    console.log(`  dt/tau = ${(r.dt / r.tau).toFixed(1)}, ${r.steps} steps: C/C0 = ${r.ratio.toFixed(5)}, implicit Euler ${r.expected.toFixed(5)}`);
    assert.ok(Math.abs(r.ratio - r.expected) < 0.01 * r.expected + 2e-3, `dt ${r.dt}: ${r.ratio} vs ${r.expected}`);
  }
  assert.deepEqual(page.errors, []);
  await page.close();
});

test('Chebyshev-accelerated sweeps converge on a stiff chain where plain Jacobi does not', { timeout: 600_000 }, async () => {
  // A pre-stretched chain with kappa * dt = 200 (the stiff regime of a 70 GPa
  // lithosphere): the exact implicit step is a tridiagonal solve. Plain Jacobi
  // propagates the end displacement about one bond per sweep.
  const K = 200;
  const page = await openPage();
  const runs = await page.evaluate(async ([chain, builderSrc, K]) => {
    const build = eval(`(${builderSrc})`);
    await plate.setParams({ mantle_drive: 0, crust_youngs_modulus_gpa: 70, base_apothem: chain.apothem });
    await plate.setParams({ dt_crust_myr: K / plate.sim.params.bond_rate_myr_inv, solver_iterations: 128 });
    const out = {};
    for (const [name, rho] of [['plain', 0], ['chebyshev', 0.999]]) {
      await plate.setParams({ chebyshev_rho: rho });
      await plate.loadState(build(chain));
      await plate.runSteps(1);
      const f = new Float32Array((await plate.readBuffers(['particles'])).particles);
      const w = f.length / 25000;
      out[name] = Array.from({ length: chain.n }, (_, i) => Math.atan2(f[w * i], f[w * i + 2]));
    }
    return out;
  }, [CHAIN, chainState.toString(), K]);
  assert.deepEqual(page.errors, []);
  await page.close();

  const exact = chainExact(CHAIN, K);
  const phi0 = (i) => (i - (CHAIN.n - 1) / 2) * CHAIN.spacing;
  const norm = Math.hypot(...exact);
  const error = (phi) => Math.hypot(...phi.map((p, i) => p - phi0(i) - exact[i])) / norm;
  const plain = error(runs.plain), cheb = error(runs.chebyshev);
  console.log(`  128 sweeps: relative error plain Jacobi ${plain.toFixed(3)}, Chebyshev ${cheb.toFixed(4)}`);
  assert.ok(cheb < 0.02, `Chebyshev error ${cheb}`);
  assert.ok(plain > 10 * cheb, `plain ${plain} vs Chebyshev ${cheb}`);
});

test('a thickened block relaxes to its Airy height (free radius); its neighbours stay put', { timeout: 300_000 }, async () => {
  // Phase 3 acceptance (design 9). A row of bonded columns of old ocean; the
  // middle three are instantly thickened to 35 km of continental crust while
  // still at the ocean's elevation. Each column's top relaxes toward its own
  // Airy height by implicit Euler with the Haskell rebound time. With no
  // flexural rigidity yet (phase 7), columns are independent.
  const page = await openPage();
  const out = await page.evaluate(async () => {
    const a = 0.0175, n = 9, half = Math.asin(a); // chord between neighbours = 2a
    await plate.setParams({ mantle_drive: 0 });
    const p = plate.sim.params;
    // Independent Airy height and rebound time (same formulas as design 4.4).
    const airy = (H, comp, T) => {
      const rho = comp * (1 + p.thermal_expansion_coeff * (p.mantle_temp - T));
      const e0 = H * (1 - rho / p.mantle_density);
      return e0 >= p.sea_level ? e0 : e0 - (p.sea_level - e0) * p.water_density / (p.mantle_density - p.water_density);
    };
    const tauMyr = (eq) => {
      const drho = eq < p.sea_level ? p.mantle_density - p.water_density : p.mantle_density;
      return (4 * Math.PI * p.isostatic_viscosity_pa_s) / (drho * 9.81 * 4 * a * 6.371e6) / 3.15576e13;
    };
    const block = (i) => i >= 3 && i <= 5;
    const T = 278; // at ocean temperature, so densities stay constant
    const ocean = airy(7, 3000, T);
    const position = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) { const phi = (i - (n - 1) / 2) * 2 * half; position.set([Math.sin(phi), 0, Math.cos(phi)], 3 * i); }
    const fill = (f) => Float32Array.from({ length: n }, (_, i) => f(i));
    await plate.loadState({
      particles: {
        count: n, position,
        thickness: fill((i) => (block(i) ? 35 : 7)), compositionDensity: fill((i) => (block(i) ? 2700 : 3000)),
        temperature: fill(() => T), age: fill(() => 100), apothem: fill(() => a),
        elevation: fill(() => ocean),
      },
      bonds: { count: n - 1, a: Uint32Array.from({ length: n - 1 }, (_, j) => j), b: Uint32Array.from({ length: n - 1 }, (_, j) => j + 1), damage: new Float32Array(n - 1) },
      yieldStressMPa: 1e6,
    });
    const read = async () => {
      const f = new Float32Array((await plate.readBuffers(['particles'])).particles);
      const w = f.length / 25000, off = plate.layout.Particle.offsets;
      return { e: Array.from({ length: n }, (_, i) => f[w * i + off.elevation]), x: Array.from({ length: n }, (_, i) => f[w * i]) };
    };
    const start = await read();
    const steps = [];
    for (let s = 0; s < 6; s++) { await plate.runSteps(1); steps.push((await read()).e); }
    const end = await read();
    return { start, steps, end, dt: p.dt_crust_myr, ocean, thick: airy(35, 2700, T), tau: tauMyr(airy(35, 2700, T)), sea: p.sea_level };
  });
  assert.deepEqual(page.errors, []);
  await page.close();

  const { dt, ocean, thick, tau } = out;
  console.log(`  ocean ${(ocean - out.sea).toFixed(2)} km, block Airy ${(thick - out.sea).toFixed(2)} km rel. sea level; tau ${(tau * 1000).toFixed(1)} kyr, dt ${(dt * 1000).toFixed(0)} kyr`);
  let e = ocean;
  out.steps.forEach((row, s) => {
    const k = dt / tau;
    e = (e + k * thick) / (1 + k);
    for (let i = 3; i <= 5; i++) assert.ok(Math.abs(row[i] - e) < 1e-3 * Math.abs(thick - ocean), `step ${s + 1}, column ${i}: ${row[i]} vs ${e}`);
    for (const i of [0, 1, 2, 6, 7, 8]) assert.ok(Math.abs(row[i] - ocean) < 1e-4, `ocean column ${i} moved to ${row[i]}`);
  });
  for (let i = 0; i < 9; i++) assert.ok(Math.abs(out.end.x[i] - out.start.x[i]) < 1e-6, `column ${i} moved sideways`);
});

test('columns touch only if they overlap in map view and in depth', { timeout: 300_000 }, async () => {
  // Two unbonded pairs overlapping by the same amount in map view. In one
  // pair both columns are at the surface: a contact pushes them apart. In the
  // other, one column's top starts 400 km down, below the other's
  // lithosphere: no contact, so neither moves.
  const page = await openPage();
  const out = await page.evaluate(async () => {
    const a = 0.0175, overlap = 0.005, half = Math.asin((2 * a - overlap) / 2);
    await plate.setParams({ mantle_drive: 0 });
    // Each pair straddles the equator at longitude lon, along its meridian (chord = 2 sin(half)).
    const pair = (lon) => [-half, half].map((t) => [Math.sin(lon) * Math.cos(t), Math.sin(t), Math.cos(lon) * Math.cos(t)]);
    const pts = [...pair(0), ...pair(1)];
    const fill = (v) => new Float32Array(4).fill(v);
    await plate.loadState({
      particles: {
        count: 4, position: Float32Array.from(pts.flat()),
        thickness: fill(7), compositionDensity: fill(3000), temperature: fill(278), age: fill(100), apothem: fill(a),
        elevation: Float32Array.of(NaN, NaN, NaN, -400),
      },
      bonds: { count: 0, a: new Uint32Array(0), b: new Uint32Array(0), damage: new Float32Array(0) },
      yieldStressMPa: 1e6,
    });
    await plate.runSteps(1);
    const f = new Float32Array((await plate.readBuffers(['particles'])).particles);
    const w = f.length / 25000;
    const d = (i, j) => Math.hypot(f[w * i] - f[w * j], f[w * i + 1] - f[w * j + 1], f[w * i + 2] - f[w * j + 2]);
    return { before: 2 * a - overlap, surface: d(0, 1), deep: d(2, 3), overlap };
  });
  assert.deepEqual(page.errors, []);
  await page.close();
  console.log(`  separation after one step: surface pair +${((out.surface - out.before) / out.overlap * 100).toFixed(0)}% of the overlap, deep pair +${((out.deep - out.before) / out.overlap * 100).toFixed(2)}%`);
  assert.ok(out.surface - out.before > 0.5 * out.overlap, `surface pair not pushed apart: ${out.surface}`);
  assert.ok(Math.abs(out.deep - out.before) < 1e-6, `deep pair moved: ${out.deep}`);
});

// Page-side helpers for synthetic edge tests: a triangular lattice patch of
// spacing 2a around the north pole, and a writer for particle positions.
const PATCH_HELPERS = `({
  patch(rings, a) {
    const pts = [];
    for (let q = -rings; q <= rings; q++) for (let r = -rings; r <= rings; r++) {
      if (Math.abs(q + r) > rings) continue;
      const x = 2 * a * (q + r / 2), y = 2 * a * r * Math.sqrt(3) / 2;
      pts.push([x, y, Math.sqrt(1 - x * x - y * y)]);
    }
    return pts;
  },
  state(pts, a, bonds = []) {
    const n = pts.length, fill = (v) => new Float32Array(n).fill(v);
    return {
      particles: { count: n, position: Float32Array.from(pts.flat()), thickness: fill(7), compositionDensity: fill(3000), temperature: fill(278), age: fill(100), apothem: fill(a) },
      bonds: { count: bonds.length, a: Uint32Array.from(bonds.map((b) => b[0])), b: Uint32Array.from(bonds.map((b) => b[1])), damage: Float32Array.from(bonds.map((b) => b[2])) },
      yieldStressMPa: 1e9,
    };
  },
  // Set positions (and predicted positions) of particles 0..n-1 to f(i, [x, y, z]), normalised.
  async move(n, f) {
    const buf = (await plate.readBuffers(['particles'])).particles;
    const v = new Float32Array(buf), W = plate.layout.Particle.words, o = plate.layout.Particle.offsets;
    for (let i = 0; i < n; i++) {
      const p = f(i, [v[W * i + o.position], v[W * i + o.position + 1], v[W * i + o.position + 2]]);
      const l = Math.hypot(...p);
      v.set([p[0] / l, p[1] / l, p[2] / l, 1], W * i + o.position);
      v.set([p[0] / l, p[1] / l, p[2] / l, 1], W * i + o.predicted_pos);
    }
    plate.sim.device.queue.writeBuffer(plate.sim.buffers.particles, 0, buf);
  },
  // GPU edge set vs the CPU reference, skipping pairs within rounding of the range boundary.
  compare(r, n, previous) {
    const T = plate.debug.topology, params = plate.sim.params;
    const cols = T.columnsFrom(r.P, params, { count: n });
    const expected = T.updateEdges(cols, previous, params);
    const mismatches = [];
    for (const k of new Set([...expected.edges, ...r.pairs.keys()])) {
      if (expected.edges.has(k) === r.pairs.has(k)) continue;
      const [i, j] = k.split(',').map(Number);
      if (Math.abs(T.rangeMargin(cols, i, j, params)) < 1e-5) continue;
      mismatches.push(k + (r.pairs.has(k) ? ' on GPU only' : ' expected but missing'));
    }
    return { expected, mismatches };
  },
})`;

test('edge storage stays consistent and matches the reference under random motion', { timeout: 600_000 }, async () => {
  // Phase A (design 9): 127 columns on a triangular lattice, moved by a
  // random walk with a pull back to their sites (stationary rms ~0.1 apothem),
  // so pairs keep leaving and re-entering range. No solver iterations and no
  // mantle flow: only the edge kernels act. Every step: slot lists symmetric
  // and sorted, no duplicate pairs, free list exact, and the edge set equal to
  // the CPU reference's.
  const page = await openPage();
  const out = await page.evaluate(async (helperSrc) => {
    const H = eval(helperSrc);
    const a = 0.0175, pts = H.patch(6, a), n = pts.length;
    await plate.setParams({ mantle_drive: 0, solver_iterations: 0 });
    await plate.loadState(H.state(pts, a));
    let seed = 12345;
    const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 - 0.5; };
    let previous = new Set();
    const res = { steps: 0, problems: [], mismatches: [], created: 0, removed: 0, minEdges: Infinity, maxEdges: 0, slabs: 0 };
    for (let s = 0; s < 100; s++) {
      if (s > 0) await H.move(n, (i, p) => p.map((v, c) => v + 0.1 * (pts[i][c] - v) + 0.04 * a * 3.46 * rand()));
      await plate.runSteps(1);
      const r = await plate.debug.edges();
      res.problems.push(...r.problems.map((m) => `step ${s}: ${m}`));
      const { expected, mismatches } = H.compare(r, n, previous);
      res.mismatches.push(...mismatches.map((m) => `step ${s}: ${m}`));
      if (s > 0) { res.created += expected.created.size; res.removed += expected.removed.size; }
      res.minEdges = Math.min(res.minEdges, r.pairs.size);
      res.maxEdges = Math.max(res.maxEdges, r.pairs.size);
      for (let i = 0; i < n; i++) res.slabs += r.P.state[i] !== 1 ? 1 : 0;
      res.overflow = [r.edgePoolEmpty, r.slotOverflow];
      previous = new Set(r.pairs.keys());
      res.steps++;
    }
    return res;
  }, PATCH_HELPERS);
  assert.deepEqual(page.errors, []);
  await page.close();
  console.log(`  ${out.steps} steps: ${out.minEdges}-${out.maxEdges} edges, ${out.created} created and ${out.removed} removed after step 0`);
  assert.deepEqual(out.problems.slice(0, 10), [], `${out.problems.length} storage problems`);
  assert.deepEqual(out.mismatches.slice(0, 10), [], `${out.mismatches.length} mismatches with the reference`);
  assert.equal(out.slabs, 0, 'a column became a slab; the test assumes none');
  assert.deepEqual(out.overflow, [0, 0], 'edge pool or slot list overflowed');
  assert.ok(out.created > 50 && out.removed > 50, 'too little turnover to test anything');
});

test('an edge keeps its state while its pair stays in range and is dropped exactly when it leaves', { timeout: 600_000 }, async () => {
  // Phase A (design 9). Seven columns (a centre and its ring), all 12 lattice
  // pairs bonded at integrity 0.7, with distinct values written into every
  // carried field. 100 steps of small motion within range must leave every
  // edge, its slot and its state untouched. Then one ring column moves
  // outward by 1.5% of the apothem sum per step: its edges must vanish at the
  // step the reference says they leave range, the others stay as they were.
  // Moved back, the pair gets a fresh edge: no cohesion and no history.
  const page = await openPage();
  const out = await page.evaluate(async (helperSrc) => {
    const H = eval(helperSrc);
    const a = 0.0175, pts = H.patch(1, a), n = pts.length; // 7 columns; index 3 is the centre
    const d = (i, j) => Math.hypot(...pts[i].map((v, c) => v - pts[j][c]));
    const bonds = [];
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (d(i, j) < 2.1 * a) bonds.push([i, j, 0.3]);
    await plate.setParams({ mantle_drive: 0, solver_iterations: 0 });
    await plate.loadState(H.state(pts, a, bonds));

    // Distinct values in every carried field.
    const E = plate.layout.Edge, fields = ['stored_stress_n', 'stored_stress_t0', 'stored_stress_t1', 'slip_anchor0', 'slip_anchor1', 'timer'];
    const buf = (await plate.readBuffers(['edges'])).edges, v = new Float32Array(buf);
    bonds.forEach((_, k) => fields.forEach((f, m) => { v[E.words * k + E.offsets[f]] = 10 * (m + 1) + k; }));
    plate.sim.device.queue.writeBuffer(plate.sim.buffers.edges, 0, buf);

    const snapshotEdges = (r) => [...r.pairs.entries()].map(([k, e]) => ({ k, e, integrity: r.E.integrity[e], born: r.E.born_myr[e], ...Object.fromEntries(fields.map((f) => [f, r.E[f][e]])) }));
    let seed = 99;
    const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 - 0.5; };
    let r = null, previous = new Set();
    const problems = [], mismatches = [];
    const step = async (label) => {
      await plate.runSteps(1);
      r = await plate.debug.edges();
      problems.push(...r.problems.map((m) => `${label}: ${m}`));
      const c = H.compare(r, n, previous);
      mismatches.push(...c.mismatches.map((m) => `${label}: ${m}`));
      previous = new Set(r.pairs.keys());
      return c.expected;
    };
    await step('start');
    const start = snapshotEdges(r);
    for (let s = 0; s < 100; s++) {
      await H.move(n, (i, p) => p.map((v, c) => v + 0.5 * (pts[i][c] - v) + 0.01 * a * 3.46 * rand()));
      await step(`jitter ${s}`);
    }
    const afterJitter = snapshotEdges(r);

    // Move column 0 (on the ring) radially away from the centre (3).
    const out = pts[0].map((v, c) => v - pts[3][c]), lo = Math.hypot(...out);
    const drops = [];
    for (let s = 1; s <= 30; s++) {
      await H.move(n, (i, p) => (i === 0 ? pts[0].map((v, c) => v + (s * 0.015 * 2 * a) * out[c] / lo) : pts[i]));
      const expected = await step(`pull ${s}`);
      if (expected.removed.size) drops.push({ s, removed: [...expected.removed] });
    }
    const afterPull = snapshotEdges(r);
    await H.move(n, (i) => pts[i]);
    await step('return');
    const returned = snapshotEdges(r);
    return { start, afterJitter, afterPull, returned, drops, problems, mismatches, time: plate.sim.timeMyr };
  }, PATCH_HELPERS);
  assert.deepEqual(page.errors, []);
  await page.close();

  assert.deepEqual(out.problems.slice(0, 10), []);
  assert.deepEqual(out.mismatches.slice(0, 10), []);
  assert.equal(out.start.length, 12);
  assert.ok(out.start.every((e) => Math.abs(e.integrity - 0.7) < 1e-6 && e.timer >= 60), 'carried state not as written');
  assert.deepEqual(out.afterJitter, out.start, 'edge state changed while every pair stayed in range');

  // The ring column has 3 edges (centre and two ring neighbours); all leave range.
  const touching0 = out.start.filter((e) => e.k.split(',').includes('0'));
  assert.equal(touching0.length, 3);
  const dropped = out.drops.flatMap((d) => d.removed);
  assert.deepEqual(dropped.sort(), touching0.map((e) => e.k).sort());
  console.log(`  ring column pulled out by 1.5% of the apothem sum per step; edges dropped at steps ${out.drops.map((d) => `${d.s} (${d.removed.join(' ')})`).join(', ')}`);
  // The centre edge leaves first (gap 10.5% at step 7); the ring edges see
  // half the motion along their length (gap 10.5% at step 14).
  assert.deepEqual(out.drops.map((d) => [d.s, d.removed.length]), [[7, 1], [14, 2]]);
  const kept = out.start.filter((e) => !e.k.split(',').includes('0'));
  assert.deepEqual(out.afterPull, kept, 'edges of the other columns changed');

  // Back in range: fresh edges with no cohesion and no history, born now.
  for (const e of out.returned.filter((x) => x.k.split(',').includes('0'))) {
    assert.equal(e.integrity, 0);
    for (const f of ['stored_stress_n', 'stored_stress_t0', 'stored_stress_t1', 'slip_anchor0', 'slip_anchor1', 'timer']) assert.equal(e[f], 0);
    assert.ok(e.born > 0, `born_myr ${e.born}`);
  }
  assert.equal(out.returned.length, 12);
});

test('resuming from a record ZIP in a fresh page continues the run', { timeout: 600_000 }, async () => {
  // Bit-identical GPU results are not a goal (README, "Reproducibility");
  // the continuation must match to well below a particle spacing.
  const N = 10;
  const a = await openPage();
  await a.evaluate((n) => plate.runSteps(n), N);
  const zipB64 = await a.evaluate(async (toB64Src) => {
    await plate.recorder.capture({ fields: true, snapshot: true });
    return eval(toB64Src)(await plate.exportRecord());
  }, toB64);
  await a.evaluate((n) => plate.runSteps(n), N);
  const continuous = await readState(a);
  const stepsA = await a.evaluate(() => plate.sim.steps);
  await a.close();

  const b = await openPage();
  await b.evaluate(async (zip) => {
    const bytes = Uint8Array.from(atob(zip), (c) => c.charCodeAt(0));
    await plate.restoreFromZip(bytes, 'test');
  }, zipB64);
  await b.evaluate((n) => plate.runSteps(n), N);
  const restarted = await readState(b);
  const stepsB = await b.evaluate(() => plate.sim.steps);
  assert.deepEqual(b.errors, []);
  await b.close();

  assert.equal(stepsB, stepsA);
  const Pa = unpack(Particle, continuous.particles, MAX_PARTICLES);
  const Pb = unpack(Particle, restarted.particles, MAX_PARTICLES);
  let active = 0, worst = 0;
  for (let i = 0; i < MAX_PARTICLES; i++) {
    assert.equal(Pa.is_active[i], Pb.is_active[i], `particle ${i} active in only one run`);
    if (!Pa.is_active[i]) continue;
    active++;
    for (let c = 0; c < 3; c++) worst = Math.max(worst, Math.abs(Pa.position[4 * i + c] - Pb.position[4 * i + c]));
  }
  console.log(`  ${active} particles, largest position difference ${worst.toExponential(2)} (spacing 0.035)`);
  assert.ok(worst < 1e-4, `resumed run drifted by ${worst}`);
});

test('record ZIP is complete and self-consistent', { timeout: 300_000 }, async () => {
  const page = await openPage();
  await page.evaluate(() => { plate.recorder.options.fieldIntervalMyr = 0.16; }); // every 10 steps
  await page.evaluate(() => plate.runSteps(20));
  const zip = fromB64(await page.evaluate(async (toB64Src) => eval(toB64Src)(await plate.exportRecord()), toB64));
  await page.close();

  const files = readZip(zip);
  const manifest = JSON.parse(new TextDecoder().decode(files.get('manifest.json')));
  assert.equal(manifest.schema.name, 'plate-player-record');
  assert.match(manifest.model.build.commit, /^[0-9a-f]{40}$/);
  assert.equal(manifest.run.seed, 1);

  const bytesPer = { float32: 4, float64: 8, uint32: 4 };
  const check = ({ file, dtype, shape }) => {
    const data = files.get(file);
    assert.ok(data, `missing ${file}`);
    assert.equal(data.length, shape.reduce((a, b) => a * b, 1) * bytesPer[dtype], `${file} size`);
    return data;
  };
  Object.values(manifest.mesh.files).forEach(check);
  const time = new Float64Array(check(manifest.fields.time).slice().buffer);
  assert.equal(time.length, 3); // steps 0, 10, 20
  for (let t = 1; t < time.length; t++) assert.ok(time[t] > time[t - 1]);

  const field = (name) => {
    const v = manifest.fields.variables.find((f) => f.name === name);
    return new Float32Array(check(v).slice().buffer);
  };
  manifest.fields.variables.forEach(check);
  const support = field('support');
  const height = field('surface_height');
  let covered = 0;
  for (let k = 0; k < support.length; k++) {
    assert.ok(support[k] >= 0 && support[k] < 1.5, `support ${support[k]}`);
    if (support[k] > 0.5) {
      covered++;
      assert.ok(height[k] > -8 && height[k] < 8, `surface height ${height[k]} km`);
    }
  }
  assert.ok(covered / support.length > 0.95, `only ${covered}/${support.length} samples well covered`);
  // The first sample (step 0) must already have derived elevations: land and sea.
  let land = 0, sea = 0;
  for (let k = 0; k < manifest.mesh.nodes; k++) if (support[k] > 0.5) (height[k] > 0 ? land++ : sea++);
  assert.ok(land > 0.1 * (land + sea) && sea > 0.4 * (land + sea), `step-0 land/sea split ${land}/${sea}`);

  const platesSeries = JSON.parse(new TextDecoder().decode(files.get(manifest.plates.file)));
  assert.ok(platesSeries.length >= 1 && platesSeries.at(-1).plates.length >= 5, 'plate series missing or too few plates');
  const plateIds = new Set(field('plate_id').filter((v) => v > 0));
  assert.ok(plateIds.size >= 5, `only ${plateIds.size} plate ids in the plate_id field`);
  assert.ok(Array.isArray(JSON.parse(new TextDecoder().decode(files.get(manifest.events.file)))));

  assert.ok(manifest.snapshots.length >= 1);
  for (const snap of manifest.snapshots) {
    for (const { file, bytes } of Object.values(snap.buffers)) assert.equal(files.get(file)?.length, bytes, file);
  }
});
