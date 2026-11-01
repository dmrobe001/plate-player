// GPU mechanism tests. They drive the built page (dist/plate-player.html) in
// Chromium through `window.plate`. Keep them cheap and about mechanisms with
// known answers; high-level behaviour is benchmarked once per design phase,
// not kept here (see README, "What belongs in the test suite").
//
//   npm run build && npm run test:gpu
//
// Under the software GPU every dispatch and readback costs in proportion to
// buffer capacity, and a readback takes ~0.5 s. So mechanism tests run on an
// empty page with small buffers (SMALL_PAGE: 1024 particle slots; the phase E
// sphere tests 4096) and read back as rarely as they can; the whole default
// suite takes about four minutes under SwiftShader.
// Tests on a generated planet (full capacity, minutes each) run only with
// PLATE_PLANET=1.
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
import { tessellate } from '../src/columns/area.js';
import { splitColumn, volumeStep } from '../src/columns/volume.js';

// Typed arrays do not survive page.evaluate's JSON; send plain arrays (restored in the page).
const jsonState = (st) => ({
  ...st,
  particles: Object.fromEntries(Object.entries(st.particles).map(([k, v]) => [k, ArrayBuffer.isView(v) ? Array.from(v) : v])),
  bonds: Object.fromEntries(Object.entries(st.bonds).map(([k, v]) => [k, ArrayBuffer.isView(v) ? Array.from(v) : v])),
});

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const PAGE = pathToFileURL(fileURLToPath(new URL('../dist/plate-player.html', import.meta.url))).href + '?paused';
const SMALL_PAGE = '&empty&capacity=1024';
const PLANET = process.env.PLATE_PLANET === '1' ? {} : { skip: 'planet-scale; set PLATE_PLANET=1' };
const MAX_PARTICLES = 25000;      // planet pages only
const MAX_EDGES = 131072;         // CAPACITY in src/sim/simulation.js
const ARGS = process.env.PLATE_GPU === 'hardware'
  ? ['--enable-unsafe-webgpu']
  : ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--disable-vulkan-surface'];

let browser;
test.before(async () => { browser = await chromium.launch({ headless: false, args: ARGS }); });
test.after(async () => { await browser?.close(); });

/** A page with a generated planet, or (small = true) an empty page with small buffers. */
async function openPage({ small = false, query = '' } = {}) {
  const page = await browser.newPage({ viewport: { width: 320, height: 240 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error' && !m.text().includes('favicon')) errors.push(m.text()); });
  await page.goto(PAGE + (small ? SMALL_PAGE : '') + query);
  await page.waitForFunction(() => window.plate?.ready, null, { timeout: 120_000 });
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

test('free lists account for every slot; cached mantle node matches brute force', { timeout: 300_000 , ...PLANET }, async () => {
  const page = await openPage();
  await page.evaluate(() => plate.runSteps(5));
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

test('a 10x time step stays finite and bounded', { timeout: 300_000 , ...PLANET }, async () => {
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
  // The edge law uses a uniform, purely elastic profile (ELASTIC).
  const page = await openPage({ small: true });
  const results = await page.evaluate(async (elastic) => {
    const a = 0.0175, L = 3 * a, eps = 0.05; // stretch well inside edge range (10%)
    // base_apothem = L/2: these particles have the reference drag (w = 1).
    await plate.setParams({ ...elastic, mantle_drive: 0, crust_youngs_modulus_gpa: 70, base_apothem: L / 2 });
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
      const w = plate.layout.Particle.words;
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
  }, ELASTIC);
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
  const page = await openPage({ small: true });
  const runs = await page.evaluate(async ([chain, builderSrc, K, elastic]) => {
    const build = eval(`(${builderSrc})`);
    await plate.setParams({ ...elastic, mantle_drive: 0, crust_youngs_modulus_gpa: 70, base_apothem: chain.apothem });
    await plate.setParams({ dt_crust_myr: K / plate.sim.params.bond_rate_myr_inv, solver_iterations: 128 });
    const out = {};
    for (const [name, rho] of [['plain', 0], ['chebyshev', 0.999]]) {
      await plate.setParams({ chebyshev_rho: rho });
      await plate.loadState(build(chain));
      await plate.runSteps(1);
      const f = new Float32Array((await plate.readBuffers(['particles'])).particles);
      const w = plate.layout.Particle.words;
      out[name] = Array.from({ length: chain.n }, (_, i) => Math.atan2(f[w * i], f[w * i + 2]));
    }
    return out;
  }, [CHAIN, chainState.toString(), K, ELASTIC]);
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
  // flexural rigidity yet (phase F), columns are independent. Airy heights
  // come from the column profile's CPU reference (design 4.0; checked against
  // analytic answers in the unit tests): old ocean is past the capped age and
  // continents have a steady geotherm, so the targets do not drift with age.
  const page = await openPage({ small: true });
  const out = await page.evaluate(async () => {
    const a = 0.0175, n = 9, half = Math.asin(a); // chord between neighbours = 2a
    await plate.setParams({ mantle_drive: 0 });
    const p = plate.sim.params;
    const airy = (H, comp) => plate.debug.profile.airy({ crustKm: H, compositionDensity: comp, ageMyr: 100, elevationKm: 0 }, p).elevation;
    const tauMyr = (eq) => {
      const drho = eq < p.sea_level ? p.mantle_density - p.water_density : p.mantle_density;
      return (4 * Math.PI * p.isostatic_viscosity_pa_s) / (drho * 9.81 * 4 * a * 6.371e6) / 3.15576e13;
    };
    const block = (i) => i >= 3 && i <= 5;
    const T = 278;
    const ocean = airy(7, 3000);
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
      const w = plate.layout.Particle.words, off = plate.layout.Particle.offsets;
      return { e: Array.from({ length: n }, (_, i) => f[w * i + off.elevation]), x: Array.from({ length: n }, (_, i) => f[w * i]) };
    };
    const start = await read();
    const steps = [];
    for (let s = 0; s < 6; s++) { await plate.runSteps(1); steps.push((await read()).e); }
    const end = await read();
    return { start, steps, end, dt: p.dt_crust_myr, ocean, thick: airy(35, 2700), tau: tauMyr(airy(35, 2700)), sea: p.sea_level };
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

test('column profiles on the GPU match the CPU reference', { timeout: 300_000 }, async () => {
  // Phase B (design 9): the kernel (profile.wgsl) against its reference
  // (src/columns/profile.js), whose physics the unit tests check against
  // analytic answers. Columns span ridge to old ocean, a crustless column, and
  // continents from shelf to plateau; elevations start at their Airy heights.
  const page = await openPage({ small: true });
  const out = await page.evaluate(async () => {
    const cols = [ // crust km, composition density, age Myr
      [7, 3000, 0], [7, 3000, 0.5], [7, 3000, 5], [7, 3000, 30], [7, 3000, 100], [0, 3000, 20],
      [30, 2700, 1000], [35, 2700, 1000], [45, 2750, 1000], [70, 2700, 1000], [12, 2800, 50],
    ];
    const n = cols.length, a = 0.0175;
    const position = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) { const phi = i * 3 * a; position.set([Math.sin(phi), 0, Math.cos(phi)], 3 * i); }
    const f = (k) => Float32Array.from(cols, (c) => c[k]);
    await plate.loadState({
      particles: { count: n, position, thickness: f(0), compositionDensity: f(1), age: f(2), temperature: new Float32Array(n).fill(278), apothem: new Float32Array(n).fill(a) },
      bonds: { count: 0, a: new Uint32Array(0), b: new Uint32Array(0), damage: new Float32Array(0) },
      yieldStressMPa: 1e9,
    });
    const r = await plate.debug.profiles();
    return {
      worst: r.worst, active: r.active.length,
      elevation: r.active.map((i) => r.P.elevation[i]), airy: r.active.map((i) => r.G.airy_elevation_km[i]),
    };
  });
  assert.deepEqual(page.errors, []);
  await page.close();
  assert.equal(out.active, 11);
  console.log(`  worst scaled differences: ${Object.entries(out.worst).map(([k, w]) => `${k} ${w.err.toExponential(1)}`).join(', ')}`);
  for (const [field, w] of Object.entries(out.worst)) {
    assert.ok(w.err < 1e-3, `${field}: particle ${w.particle}, GPU ${w.gpu} vs reference ${w.ref}`);
  }
  // Loading sets each elevation to its Airy height.
  out.elevation.forEach((e, k) => assert.ok(Math.abs(e - out.airy[k]) < 1e-6));
});

test('edge law on the GPU matches the CPU reference: linearised law, caps and slice stresses', { timeout: 300_000 }, async () => {
  // Phase C (design 9). Isolated pairs with real profiles (old ocean,
  // continents, continent against ocean, thick against thin), loaded
  // compressed or stretched, at integrity 1, 0.3 and 0; one step with the
  // solver. The kernels (law.wgsl) must give the reference's (src/edges/law.js)
  // stiffness, zero-force gap, multiplier bounds and overlap at the start of the
  // step, and its slice stresses for the strain the solve produced.
  const page = await openPage({ small: true });
  const out = await page.evaluate(async () => {
    const pairs = [ // [crust km, density, age] x 2, strain, integrity
      [[7, 3000, 100], [7, 3000, 100], -1e-3, 1], [[7, 3000, 100], [7, 3000, 100], 2e-3, 0],
      [[35, 2700, 1000], [35, 2700, 1000], -2e-3, 1], [[35, 2700, 1000], [7, 3000, 100], -1e-3, 0.3],
      [[70, 2700, 1000], [30, 2700, 1000], 1e-3, 1], [[7, 3000, 80], [7, 3000, 100], 3e-3, 0.3],
    ];
    const a = 0.0175, n = 2 * pairs.length, position = new Float32Array(3 * n);
    const cols = pairs.flatMap((q) => [q[0], q[1]]);
    pairs.forEach((q, k) => {
      const c = 6 * a * k, half = Math.asin(a * (1 + q[2])); // chord 2a (1 + strain)
      position.set([Math.sin(c - half), 0, Math.cos(c - half), Math.sin(c + half), 0, Math.cos(c + half)], 6 * k);
    });
    const f = (m) => Float32Array.from(cols, (c) => c[m]);
    await plate.setParams({ mantle_drive: 0 });
    await plate.loadState({
      particles: { count: n, position, thickness: f(0), compositionDensity: f(1), age: f(2), temperature: new Float32Array(n).fill(278), apothem: new Float32Array(n).fill(a) },
      bonds: { count: pairs.length, a: Uint32Array.from(pairs, (_, k) => 2 * k), b: Uint32Array.from(pairs, (_, k) => 2 * k + 1), damage: Float32Array.from(pairs, (q) => 1 - q[3]) },
      yieldStressMPa: 1e9,
    });
    const L = plate.debug.law, Pr = plate.debug.profile, p = plate.sim.params;
    const before = await plate.debug.edges();
    await plate.runSteps(1);
    const after = await plate.debug.edges();
    const worst = {};
    const note = (field, gpu, ref, scale) => { const e = Math.abs(gpu - ref) / scale; if (!(e <= (worst[field]?.e ?? -1))) worst[field] = { e, gpu, ref }; };
    const summary = [];
    pairs.forEach((q, k) => {
      const e = [...after.pairs.entries()].find(([key]) => key === `${2 * k},${2 * k + 1}`)?.[1];
      if (e === undefined) { worst.missing = { e: 1 }; return; }
      const colA = Pr.columnOfParticle(after.P, 2 * k), colB = Pr.columnOfParticle(after.P, 2 * k + 1);
      const edge = L.edgeSlices(colA, colB, after.E.integrity[e], p);
      const stress0 = Array.from(before.E.slice_stress_mpa.subarray(8 * e, 8 * e + 8));
      const rest = before.P.apothem[2 * k] + before.P.apothem[2 * k + 1]; // during the step (columns_flow may change it at the end)
      const lin = L.linearize(stress0, edge, (before.E.length[e] - rest) / rest, p);
      const bounds = L.lambdaBounds(lin, rest, p);
      note('stiffness', after.E.stiffness[e], lin.stiffness, 1);
      note('reference_gap', after.E.reference_gap[e], lin.zeroStrain * rest, Math.max(Math.abs(lin.zeroStrain * rest), 1e-3 * rest));
      note('lambda_min', after.E.lambda_min[e], bounds.lambdaMin, Math.max(bounds.lambdaMax, 1e-12));
      note('lambda_max', after.E.lambda_max[e], bounds.lambdaMax, Math.max(bounds.lambdaMax, 1e-12));
      note('overlap_km', after.E.overlap_km[e], edge.overlap, 100);
      const upd = L.maxwellUpdate(stress0, edge, (after.E.length[e] - before.E.length[e]) / rest, p);
      for (let s = 0; s < 8; s++) note('slice_stress_mpa', after.E.slice_stress_mpa[8 * e + s], upd.stored[s], 70);
      note('stress_normal_mpa', after.E.stress_normal_mpa[e], upd.meanStress, 70);
      summary.push(`${q[0][0]}/${q[1][0]} km, strain ${q[2]}, integrity ${q[3]}: stiffness ${lin.stiffness.toFixed(3)}, stress ${upd.meanStress.toFixed(1)} MPa`);
    });
    return { worst, summary };
  });
  assert.deepEqual(page.errors, []);
  await page.close();
  for (const line of out.summary) console.log(`  ${line}`);
  console.log(`  worst scaled differences: ${Object.entries(out.worst).map(([k, w]) => `${k} ${w.e.toExponential(1)}`).join(', ')}`);
  for (const [field, w] of Object.entries(out.worst)) assert.ok(w.e < 1e-3, `${field}: GPU ${w.gpu} vs reference ${w.ref}`);
});

test('one edge under constant separation rate follows the Maxwell sequence on the GPU', { timeout: 300_000 }, async () => {
  // Phase C (design 9). Creep without temperature dependence gives every
  // slice the same viscosity, eta = 3 E dt (a Maxwell time of 3 steps). The
  // pair is pulled apart by the same strain every step (positions written,
  // no solve): the mean stress must follow sigma_n = (sigma_n-1 + E de) r,
  // r = 1 / (1 + E dt / eta), toward eta x rate.
  const page = await openPage({ small: true });
  const out = await page.evaluate(async (helperSrc) => {
    const p0 = plate.sim.params;
    const E = p0.crust_youngs_modulus_gpa * 1e9, dtS = p0.dt_crust_myr * 3.15576e13, eta = 3 * E * dtS;
    const n = 3.5, edot = p0.reference_strain_rate_s, log10a = Math.log10(edot) - n * Math.log10(4 * eta * edot / 1e6);
    const over = { mantle_drive: 0, solver_iterations: 0, cohesion_mpa: 1e5 };
    for (const m of ['quartzite', 'diabase', 'olivine']) Object.assign(over, { [`${m}_log10_a`]: log10a, [`${m}_n`]: n, [`${m}_q_kj`]: 0 });
    await plate.setParams(over);
    const a = 0.0175, rate = 50e6 / eta, de = rate * dtS; // eta rate = 50 MPa
    const place = (strain) => { const h = Math.asin(a * (1 + strain)); return [[-Math.sin(h), 0, Math.cos(h)], [Math.sin(h), 0, Math.cos(h)]]; };
    const pos = place(0);
    await plate.loadState({
      particles: { count: 2, position: Float32Array.from(pos.flat()), thickness: Float32Array.of(7, 7), compositionDensity: Float32Array.of(3000, 3000), age: Float32Array.of(100, 100), temperature: Float32Array.of(278, 278), apothem: Float32Array.of(a, a) },
      bonds: { count: 1, a: Uint32Array.of(0), b: Uint32Array.of(1), damage: Float32Array.of(0) },
      yieldStressMPa: 1e9,
    });
    const H = eval(helperSrc);
    const stresses = [];
    for (let s = 1; s <= 12; s++) {
      H.move(place(s * de));
      await plate.runSteps(1);
      if (s % 4 === 0) {
        const r = await plate.debug.edges();
        const e = r.pairs.get('0,1');
        stresses.push({ s, mean: r.E.stress_normal_mpa[e], strain: r.E.strain[e] });
      }
    }
    return { stresses, E, de, r: 1 / (1 + E * dtS / eta), target: eta * rate / 1e6 };
  }, PATCH_HELPERS);
  assert.deepEqual(page.errors, []);
  await page.close();
  let sigma = 0;
  const expected = [];
  for (let s = 1; s <= 12; s++) { sigma = (sigma + out.E * out.de / 1e6) * out.r; if (s % 4 === 0) expected.push(sigma); }
  out.stresses.forEach((x, k) => {
    console.log(`  step ${x.s}: mean stress ${x.mean.toFixed(2)} MPa, implicit-Euler Maxwell ${expected[k].toFixed(2)} MPa (eta x rate = ${out.target.toFixed(1)})`);
    assert.ok(Math.abs(x.mean / expected[k] - 1) < 0.02, `step ${x.s}: ${x.mean} vs ${expected[k]}`);
  });
});

test('columns touch only if they overlap in map view and in depth', { timeout: 300_000 }, async () => {
  // Two unbonded pairs overlapping by the same amount in map view. In one
  // pair both columns are at the surface: a contact pushes them apart. In the
  // other, one column's top starts 400 km down, below the other's
  // lithosphere: no contact, so neither moves.
  const page = await openPage({ small: true });
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
    const w = plate.layout.Particle.words;
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
// Edge-law parameters that make every slice a pure spring (no creep, no
// strength limit) over exactly the reference thickness, for an age-100 ocean
// column (100 km deep): edge stiffness is then the old bond stiffness (design 9, phase C).
const ELASTIC = { olivine_log10_a: -60, diabase_log10_a: -60, quartzite_log10_a: -60, cohesion_mpa: 1e5, stiffness_reference_km: 100 };

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
  // Set the positions (and predicted positions, the two leading vec4 fields)
  // of particles 0..n-1 to pts[i], normalised. Writes only those fields, so
  // no readback is needed: the caller keeps the positions.
  move(pts) {
    const P = plate.layout.Particle;
    if (P.offsets.position !== 0 || P.offsets.predicted_pos !== 4) throw new Error('Particle layout changed');
    pts.forEach((p, i) => {
      const l = Math.hypot(...p), q = [p[0] / l, p[1] / l, p[2] / l, 1];
      plate.sim.device.queue.writeBuffer(plate.sim.buffers.particles, 4 * P.words * i, Float32Array.from([...q, ...q]));
    });
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
  // random walk with a pull back to their sites (stationary rms ~0.1 apothem,
  // reached in ~10 steps), so pairs keep leaving and re-entering range. No solver iterations and no
  // mantle flow: only the edge kernels act. Every step: slot lists symmetric
  // and sorted, no duplicate pairs, free list exact, and the edge set equal to
  // the CPU reference's.
  const page = await openPage({ small: true });
  const out = await page.evaluate(async ([helperSrc, elastic]) => {
    const H = eval(helperSrc);
    const a = 0.0175, pts = H.patch(6, a), n = pts.length;
    await plate.setParams({ ...elastic, damage_work_mpa_km: 1e30, heal_time_myr: 1e30, volumetric_flow_factor: 0, mantle_drive: 0, solver_iterations: 0 }); // storage only: no creep, damage, healing or area change
    await plate.loadState(H.state(pts, a));
    let seed = 12345;
    const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 - 0.5; };
    let previous = new Set();
    const res = { steps: 0, problems: [], mismatches: [], created: 0, removed: 0, minEdges: Infinity, maxEdges: 0, slabs: 0 };
    let cur = pts.map((p) => [...p]);
    for (let s = 0; s < 30; s++) {
      if (s > 0) {
        cur = cur.map((p, i) => p.map((v, c) => v + 0.1 * (pts[i][c] - v) + 0.04 * a * 3.46 * rand()));
        H.move(cur);
      }
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
  }, [PATCH_HELPERS, ELASTIC]);
  assert.deepEqual(page.errors, []);
  await page.close();
  console.log(`  ${out.steps} steps: ${out.minEdges}-${out.maxEdges} edges, ${out.created} created and ${out.removed} removed after step 0`);
  assert.deepEqual(out.problems.slice(0, 10), [], `${out.problems.length} storage problems`);
  assert.deepEqual(out.mismatches.slice(0, 10), [], `${out.mismatches.length} mismatches with the reference`);
  assert.equal(out.slabs, 0, 'a column became a slab; the test assumes none');
  assert.deepEqual(out.overflow, [0, 0], 'edge pool or slot list overflowed');
  assert.ok(out.created > 20 && out.removed > 20, 'too little turnover to test anything');
});

test('an edge keeps its state while its pair stays in range and is dropped exactly when it leaves', { timeout: 600_000 }, async () => {
  // Phase A (design 9). Seven columns (a centre and its ring), all 12 lattice
  // pairs bonded at integrity 0.7, with distinct values written into every
  // carried field. 100 steps of small motion within range must leave every
  // edge, its slot and its state untouched. Then one ring column moves
  // outward by 1.5% of the apothem sum per step: its edges must vanish at the
  // step the reference says they leave range, the others stay as they were.
  // Moved back, the pair gets a fresh edge: no cohesion and no history.
  const page = await openPage({ small: true });
  const out = await page.evaluate(async ([helperSrc, elastic]) => {
    const H = eval(helperSrc);
    const a = 0.0175, pts = H.patch(1, a), n = pts.length; // 7 columns; index 3 is the centre
    const d = (i, j) => Math.hypot(...pts[i].map((v, c) => v - pts[j][c]));
    const bonds = [];
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (d(i, j) < 2.1 * a) bonds.push([i, j, 0.3]);
    await plate.setParams({ ...elastic, damage_work_mpa_km: 1e30, heal_time_myr: 1e30, volumetric_flow_factor: 0, mantle_drive: 0, solver_iterations: 0 }); // storage only: no creep, damage, healing or area change
    await plate.loadState(H.state(pts, a, bonds));

    // Distinct values in every carried field.
    const E = plate.layout.Edge, fields = ['stored_stress_t0', 'stored_stress_t1', 'slip_anchor0', 'slip_anchor1', 'timer'];
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
    // 100 steps of jitter, inspected only at the end (a readback costs ~0.5 s).
    let cur = pts.map((p) => [...p]);
    for (let s = 0; s < 100; s++) {
      cur = cur.map((p, i) => p.map((v, c) => v + 0.5 * (pts[i][c] - v) + 0.01 * a * 3.46 * rand()));
      H.move(cur);
      await plate.runSteps(1);
    }
    r = await plate.debug.edges();
    problems.push(...r.problems.map((m) => `after jitter: ${m}`));
    mismatches.push(...H.compare(r, n, previous).mismatches.map((m) => `after jitter: ${m}`));
    const afterJitter = snapshotEdges(r);
    previous = new Set(r.pairs.keys());

    // Move column 0 (on the ring) radially away from the centre (3).
    const out = pts[0].map((v, c) => v - pts[3][c]), lo = Math.hypot(...out);
    const drops = [];
    for (let s = 1; s <= 16; s++) {
      H.move(pts.map((p, i) => (i === 0 ? p.map((v, c) => v + (s * 0.015 * 2 * a) * out[c] / lo) : p)));
      const expected = await step(`pull ${s}`);
      if (expected.removed.size) drops.push({ s, removed: [...expected.removed] });
    }
    const afterPull = snapshotEdges(r);
    H.move(pts);
    await step('return');
    const returned = snapshotEdges(r);
    return { start, afterJitter, afterPull, returned, drops, problems, mismatches, time: plate.sim.timeMyr };
  }, [PATCH_HELPERS, ELASTIC]);
  assert.deepEqual(page.errors, []);
  await page.close();

  assert.deepEqual(out.problems.slice(0, 10), []);
  assert.deepEqual(out.mismatches.slice(0, 10), []);
  assert.equal(out.start.length, 12);
  assert.ok(out.start.every((e) => Math.abs(e.integrity - 0.7) < 1e-6 && e.timer >= 50), 'carried state not as written');
  assert.deepEqual(out.afterJitter, out.start, 'edge state changed while every pair stayed in range');

  // The ring column has 3 edges (centre and two ring neighbours); all leave range.
  const touching0 = out.start.filter((e) => e.k.split(',').includes('0'));
  assert.equal(touching0.length, 3);
  const dropped = out.drops.flatMap((d) => d.removed);
  assert.deepEqual(dropped.sort(), touching0.map((e) => e.k).sort());
  console.log(`  ring column pulled out by 1.5% of the apothem sum per step; edges dropped at steps ${out.drops.map((d) => `${d.s} (${d.removed.join(' ')})`).join(', ')}`);
  // The centre edge leaves first (gap 10.5% at step 7). The ring edges lie at
  // 60 degrees to the motion: with displacement d = 0.03 s apothems their length
  // is sqrt((2 + d/2)^2 + (d sqrt(3)/2)^2) apothems, a 10.1% gap at step 12.
  assert.deepEqual(out.drops.map((d) => [d.s, d.removed.length]), [[7, 1], [12, 2]]);
  const kept = out.start.filter((e) => !e.k.split(',').includes('0'));
  assert.deepEqual(out.afterPull, kept, 'edges of the other columns changed');

  // Back in range: fresh edges with no cohesion and no history, born now.
  for (const e of out.returned.filter((x) => x.k.split(',').includes('0'))) {
    assert.equal(e.integrity, 0);
    for (const f of ['stored_stress_t0', 'stored_stress_t1', 'slip_anchor0', 'slip_anchor1', 'timer']) assert.equal(e[f], 0);
    assert.ok(e.born > 0, `born_myr ${e.born}`);
  }
  assert.equal(out.returned.length, 12);
});

// Cold, brittle columns for the volumetric-flow tests: no creep at all, so
// slices are pure springs up to their Byerlee caps (cohesion as default).
const BRITTLE = { olivine_log10_a: -60, diabase_log10_a: -60, quartzite_log10_a: -60 };

test('volumetric flow: uniform compression thickens at the predicted rate at constant volume; pure shear changes no area', { timeout: 300_000 }, async () => {
  // Phase D (design 9). A 19-column patch of old ocean, bonded, moved by
  // hand (no solve). (1) Uniform shortening by 0.4% per step: once every
  // slice sits at its compression cap, all further shortening is inelastic,
  // so each column's reference area shrinks by 2 x 0.4% per step; its crust
  // (volume / tessellated area, phase E) thickens inversely, with total crust
  // volume constant; integrity falls by
  // exactly the logged plastic work / W_D (cold: no healing). (2) Pure shear
  // (x stretched, y shortened by 0.02% per step) below the compression cap:
  // stretched edges crack, others stay elastic, and no area changes.
  const page = await openPage({ small: true });
  const out = await page.evaluate(async ([helperSrc, brittle]) => {
    const H = eval(helperSrc);
    const a = 0.0175, pts = H.patch(2, a), n = pts.length;
    const d = (i, j) => Math.hypot(...pts[i].map((v, c) => v - pts[j][c]));
    const bonds = [];
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (d(i, j) < 2.1 * a) bonds.push([i, j, 0]);
    const centre = pts.findIndex((p) => Math.hypot(p[0], p[1]) < 1e-9);
    const WD = 1e5;
    await plate.setParams({ ...brittle, damage_work_mpa_km: WD, gpe_force_factor: 0, mantle_drive: 0, solver_iterations: 0 });
    const volume = (r) => { let v = 0; for (let i = 0; i < n; i++) v += r.P.crust_volume[i]; return v; };

    // (1) Uniform shortening.
    await plate.loadState(H.state(pts, a, bonds));
    const delta = 0.004;
    const rows = [];
    const phiExpected = new Map();
    let r = await plate.debug.edges();
    const v0 = volume(r);
    for (const e of r.pairs.values()) phiExpected.set(e, 1);
    for (let s = 1; s <= 12; s++) {
      const k = (1 - delta) ** s;
      H.move(pts.map((p) => [p[0] * k, p[1] * k, p[2]]));
      await plate.runSteps(1);
      r = await plate.debug.edges();
      for (const [e, phi] of phiExpected) phiExpected.set(e, Math.max(phi - r.E.work_mpa_km[e] / WD, 0));
      rows.push({ s, h: r.P.thickness_km[centre], area: r.P.area_ref[centre], volume: volume(r), hv: r.P.crust_volume[centre] / r.P.cell_area[centre] });
    }
    let phiErr = 0, phiMin = 1;
    for (const [e, phi] of phiExpected) { phiErr = Math.max(phiErr, Math.abs(r.E.integrity[e] - phi)); phiMin = Math.min(phiMin, phi); }

    // (2) Pure shear.
    await plate.loadState(H.state(pts, a, bonds));
    r = await plate.debug.edges();
    const area0 = Array.from(r.P.area_ref.subarray(0, n));
    const ds = 2e-4;
    const work = new Map();
    for (let s = 1; s <= 5; s++) {
      H.move(pts.map((p) => [p[0] * (1 + ds) ** s, p[1] * (1 - ds) ** s, p[2]]));
      await plate.runSteps(1);
      r = await plate.debug.edges();
      for (const [key, e] of r.pairs) work.set(key, (work.get(key) ?? 0) + r.E.work_mpa_km[e]);
    }
    let areaChange = 0;
    for (let i = 0; i < n; i++) areaChange = Math.max(areaChange, Math.abs(Math.log(r.P.area_ref[i] / area0[i])));
    const byAngle = { stretched: 0, other: 0, stretchedWork: Infinity, otherWork: 0 };
    for (const [key, w] of work) {
      const [i, j] = key.split(',').map(Number);
      const along = Math.abs(pts[i][1] - pts[j][1]) < 1e-9; // edges along x are stretched
      if (along) { byAngle.stretched++; byAngle.stretchedWork = Math.min(byAngle.stretchedWork, w); }
      else { byAngle.other++; byAngle.otherWork = Math.max(byAngle.otherWork, w); }
    }
    return { rows, v0, phiErr, phiMin, areaChange, byAngle, delta };
  }, [PATCH_HELPERS, BRITTLE]);
  assert.deepEqual(page.errors, []);
  await page.close();

  const at = (s) => out.rows.find((x) => x.s === s);
  const measured = Math.log(at(12).h / at(6).h), expected = -2 * 6 * Math.log(1 - out.delta);
  const flowed = -Math.log(at(12).area / at(6).area);
  console.log(`  uniform shortening 0.4%/step, steps 6-12: centre area_ref shrinks by ln ${flowed.toFixed(4)} (inelastic), crust thickens by ln ${measured.toFixed(4)}, predicted ${expected.toFixed(4)}`);
  assert.ok(Math.abs(flowed / expected - 1) < 0.03, `area_ref flow ${flowed} vs ${expected}`);
  assert.ok(Math.abs(measured / expected - 1) < 1e-3, `thickening ${measured} vs ${expected}`);
  // Thickness is volume over the tessellated area (one step behind the last move).
  for (const x of out.rows) assert.ok(Math.abs(x.h / x.hv - 1) < 1e-6);
  for (const x of out.rows) assert.ok(Math.abs(x.volume / out.v0 - 1) < 1e-5, `step ${x.s}: volume ${x.volume} vs ${out.v0}`);
  console.log(`  integrity: worst difference from 1 - sum(work) / W_D ${out.phiErr.toExponential(1)} (lowest ${out.phiMin.toFixed(4)})`);
  assert.ok(out.phiErr < 1e-4 && out.phiMin < 0.99);
  console.log(`  pure shear: largest area change ${out.areaChange.toExponential(1)}; work on ${out.byAngle.stretched} stretched edges >= ${out.byAngle.stretchedWork.toFixed(2)}, on ${out.byAngle.other} others <= ${out.byAngle.otherWork}`);
  assert.ok(out.areaChange < 1e-6);
  assert.ok(out.byAngle.stretchedWork > 0 && out.byAngle.otherWork === 0);
});

test('below yield nothing irreversible happens; the GPE push matches the reference', { timeout: 300_000 }, async () => {
  // Phase D (design 9). (1) A bonded patch of old ocean with real profiles,
  // slightly jittered, solved for 30 steps (no mantle flow): no plastic work,
  // integrity stays 1, areas do not drift. (2) A patch with an age ramp across
  // it (GPE falls with age): one step with no solve moves each column by the
  // reference's push down the least-squares GPE gradient (flow.js).
  const page = await openPage({ small: true });
  const out = await page.evaluate(async (helperSrc) => {
    const H = eval(helperSrc);
    const a = 0.0175, pts = H.patch(2, a), n = pts.length;
    const d = (i, j) => Math.hypot(...pts[i].map((v, c) => v - pts[j][c]));
    const bonds = [];
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (d(i, j) < 2.1 * a) bonds.push([i, j, 0]);
    await plate.setParams({ mantle_drive: 0 });
    let seed = 5;
    const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 - 0.5; };
    // The flat-built patch is pre-stretched ~3e-4 by the sphere's curvature
    // (~25 MPa, above cohesion): load it, declare that state stress-free (as
    // the generated planet is), then jitter by strains ~1e-5 (~1 MPa), far
    // below the 20 MPa cohesion even after the hot base's load moves to the lid.
    await plate.loadState(H.state(pts, a, bonds));
    {
      const E = plate.layout.Edge, buf = (await plate.readBuffers(['edges'])).edges, f = new Float32Array(buf);
      bonds.forEach((_, k) => { f.fill(0, E.words * k + E.offsets.slice_stress_mpa, E.words * k + E.offsets.slice_stress_mpa + 8); });
      plate.sim.device.queue.writeBuffer(plate.sim.buffers.edges, 0, buf);
    }
    H.move(pts.map((p) => p.map((v) => v + 2e-5 * a * rand())));
    let r = await plate.debug.edges();
    const area0 = Array.from(r.P.area_ref.subarray(0, n));
    await plate.runSteps(30);
    r = await plate.debug.edges();
    let work = 0, phi = 1, areaChange = 0;
    for (const e of r.pairs.values()) { work = Math.max(work, r.E.work_mpa_km[e]); phi = Math.min(phi, r.E.integrity[e]); }
    for (let i = 0; i < n; i++) areaChange = Math.max(areaChange, Math.abs(Math.log(r.P.area_ref[i] / area0[i])));

    // (2) GPE push (off by default until integration; on here).
    await plate.setParams({ solver_iterations: 0, gpe_force_factor: 1 });
    const st = H.state(pts, a, bonds);
    st.particles.age = Float32Array.from(pts, (p) => 30 + 300 * p[0]); // 9 .. 51 Myr across the patch
    await plate.loadState(st);
    await plate.runSteps(1);
    r = await plate.debug.edges();
    const G = (await plate.debug.profiles()).G;
    const F = plate.debug.flow, p = plate.sim.params;
    const nbrs = new Map();
    for (const key of r.pairs.keys()) { const [i, j] = key.split(',').map(Number); for (const [u, v] of [[i, j], [j, i]]) (nbrs.get(u) ?? nbrs.set(u, []).get(u)).push(v); }
    let worst = 0, mean = 0;
    for (let i = 0; i < n; i++) {
      const x0 = pts[i].map((v) => v / Math.hypot(...pts[i]));
      const grad = F.gradient(x0, G.gpe_mpa_km[i], (nbrs.get(i) ?? []).map((j) => ({ x: pts[j].map((v) => v / Math.hypot(...pts[j])), f: G.gpe_mpa_km[j] })));
      if (!grad) continue;
      const want = F.gpeDisplacement(grad, G.lithosphere_km[i], p);
      const got = [0, 1, 2].map((c) => r.P.position[4 * i + c] - x0[c]);
      const size = Math.hypot(...want);
      worst = Math.max(worst, Math.hypot(...got.map((v, c) => v - want[c])) / size);
      mean += size / n;
    }
    return { work, phi, areaChange, worst, meanKmPerStep: mean * 6371, dt: p.dt_crust_myr };
  }, PATCH_HELPERS);
  assert.deepEqual(page.errors, []);
  await page.close();
  console.log(`  30 steps below yield: max plastic work ${out.work}, min integrity ${out.phi}, largest area change ${out.areaChange.toExponential(1)}`);
  assert.equal(out.work, 0);
  assert.equal(out.phi, 1);
  assert.ok(out.areaChange < 1e-5);
  console.log(`  GPE push: mean ${out.meanKmPerStep.toFixed(2)} km per step (${(out.meanKmPerStep * 1e5 / (out.dt * 1e6)).toFixed(0)} cm/yr for a free column); worst relative difference from the reference ${out.worst.toExponential(1)}`);
  assert.ok(out.worst < 1e-2);
});

// --- Phase E: area, volume, melt and splits (design 4.8) ---------------------
// A whole sphere of 2562 columns (icosphere 4, jittered, apothems +-15%), so
// that the tessellation tiles it. Columns are ~4x the planet's, so the page
// uses a coarser hash grid (cells 0.15) and a matching clip radius.
const SPHERE_PAGE = { query: '&empty&capacity=4096&grid=20' };
function sphereState({ jitter = 0.2, spread = 0.3, seed = 4, kind = () => 'old' } = {}) {
  const v = createIcosphere(4).vertices;
  let s = seed;
  const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const spacing = Math.sqrt((4 * Math.PI) / v.length / (Math.sqrt(3) / 2));
  const pts = v.map((x) => { const y = x.map((c) => c + jitter * spacing * (rng() - 0.5)); const l = Math.hypot(...y); return y.map((c) => c / l); });
  const apothem = Float32Array.from(v, () => (spacing / 2) * (1 + spread * (rng() - 0.5)));
  const n = pts.length;
  const K = { young: [7, 3000, 0], old: [7, 3000, 100], continent: [35, 2700, 1000] };
  const col = (f) => Float32Array.from(pts, (p, i) => K[kind(p, i)][f]);
  return {
    pts, spacing, apothem,
    state: {
      particles: { count: n, position: Float32Array.from(pts.flat()), thickness: col(0), compositionDensity: col(1), age: col(2), temperature: new Float32Array(n).fill(278), apothem },
      bonds: { count: 0, a: new Uint32Array(0), b: new Uint32Array(0), damage: new Float32Array(0) },
    },
  };
}
const SPHERE_PARAMS = (spacing) => ({
  cell_clip_radius: 0.065, base_apothem: spacing / 2, mantle_drive: 0, solver_iterations: 0, gpe_force_factor: 0, volumetric_flow_factor: 0,
});
// Read the first n particles' fields as plain arrays, plus the counters.
const READ_COLUMNS = `async (n, fields) => {
  const P = plate.debug.topology && plate.layout.Particle;
  const f = new Float32Array((await plate.readBuffers(['particles'])).particles), u = new Uint32Array(f.buffer);
  const out = {};
  for (const [name, type] of P.fields) {
    if (!fields.includes(name)) continue;
    const o = P.offsets[name], len = type === 'vec4f' ? 4 : 1, src = type === 'u32' ? u : f;
    out[name] = Array.from({ length: n * len }, (_, k) => src[P.words * Math.floor(k / len) + o + (k % len)]);
  }
  out.counters = await plate.debug.counters();
  return out;
}`;

test('tessellated areas tile the sphere and match the reference; crust volume = thickness x area at load', { timeout: 300_000 }, async () => {
  const { pts, spacing, apothem, state } = sphereState({ kind: (p) => (p[2] > 0 ? 'continent' : 'old') });
  const page = await openPage(SPHERE_PAGE);
  const out = await page.evaluate(async ([st, params, readSrc]) => {
    const read = eval(readSrc);
    await plate.setParams(params);
    const restore = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, Array.isArray(v) ? Float32Array.from(v) : v]));
    await plate.loadState({ ...st, particles: restore(st.particles), bonds: restore(st.bonds) });
    return read(st.particles.count, ['cell_area', 'crust_volume', 'thickness_km', 'cell_clipped']);
  }, [jsonState(state), SPHERE_PARAMS(spacing), READ_COLUMNS]);
  assert.deepEqual(page.errors, []);
  await page.close();
  const n = pts.length;
  const total = out.cell_area.reduce((a, b) => a + b, 0);
  const ref = tessellate(Float64Array.from(pts.flat()), Array.from(apothem), 0.065);
  let worst = 0;
  for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(out.cell_area[i] / ref[i].area - 1));
  console.log(`  ${n} cells: sum - 4 pi = ${(total - 4 * Math.PI).toExponential(1)} (relative ${((total - 4 * Math.PI) / (4 * Math.PI)).toExponential(1)}); worst cell vs reference ${worst.toExponential(1)}; clipped ${out.counters.clippedCells}`);
  assert.equal(out.counters.clippedCells, 0);
  assert.equal(ref.filter((c) => c.clipped).length, 0);
  assert.ok(Math.abs(total / (4 * Math.PI) - 1) < 1e-5);
  assert.ok(worst < 1e-4);
  for (let i = 0; i < n; i++) assert.ok(Math.abs(out.crust_volume[i] / (out.thickness_km[i] * out.cell_area[i]) - 1) < 1e-6);
});

test('melt: new area makes 7 km of crust where the lithosphere is thin, none under thick lithosphere; the volume budget closes', { timeout: 300_000 }, async () => {
  // One step with the columns moved by hand (a smooth displacement field, so
  // cells grow and shrink) and one column pushed deep onto a neighbour (the
  // overlap stand-in turns the lower one into a slab: a volume sink). Young
  // ocean (L = 0) in the north, old ocean at mid latitudes, continent in the south.
  const kind = (p) => (p[2] > 0.3 ? 'young' : p[2] < -0.3 ? 'continent' : 'old');
  const { pts, spacing, apothem, state } = sphereState({ kind });
  const n = pts.length;
  const amp = 0.01;
  const moved = pts.map(([x, y, z]) => { const q = [x + amp * Math.sin(5 * y), y + amp * Math.sin(5 * z), z + amp * Math.sin(5 * x)]; const l = Math.hypot(...q); return q.map((c) => c / l); });
  // The sink: an old-ocean column pushed onto its nearest neighbour.
  const k = pts.findIndex((p) => kind(p) === 'old');
  let j = -1, dj = Infinity;
  pts.forEach((p, i) => { const d = Math.hypot(...p.map((c, m) => c - pts[k][m])); if (i !== k && d < dj) { dj = d; j = i; } });
  moved[k] = (() => { const q = moved[j].map((c, m) => c + 0.3 * (moved[k][m] - c)); const l = Math.hypot(...q); return q.map((c) => c / l); })();
  const page = await openPage(SPHERE_PAGE);
  const fields = ['cell_area', 'crust_volume', 'thermal_age_myr', 'composition_density', 'volume_added', 'state', 'age'];
  const out = await page.evaluate(async ([st, params, readSrc, movedPts, fields, helperSrc]) => {
    const read = eval(readSrc), H = eval(helperSrc);
    // Only the pushed pair (overlap ~0.05) may trigger the overlap stand-in for subduction.
    await plate.setParams({ ...params, subduction_overlap: 0.03 });
    const restore = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, Array.isArray(v) ? Float32Array.from(v) : v]));
    await plate.loadState({ ...st, particles: restore(st.particles), bonds: restore(st.bonds) });
    const before = await read(st.particles.count, fields);
    H.move(movedPts);
    await plate.runSteps(1);
    return { before, after: await read(st.particles.count, fields), params: plate.sim.params };
  }, [jsonState(state), SPHERE_PARAMS(spacing), READ_COLUMNS, moved, fields, PATCH_HELPERS]);
  assert.deepEqual(page.errors, []);
  await page.close();

  const { before: B, after: A, params: p } = out;
  const sinkCol = [k, j].find((i) => A.state[i] === 2);
  assert.ok(sinkCol !== undefined, 'the pushed pair should have produced a slab');
  assert.equal(A.state.filter((x) => x === 2).length, 1, 'exactly one slab');
  // Per column vs the reference (volume.js), from the GPU's own cell areas.
  let worst = 0, young = { dA: 0, dV: 0, n: 0 }, thick = 0, added = 0;
  for (let i = 0; i < n; i++) {
    const r = volumeStep({ volume: B.crust_volume[i], area: A.cell_area[i], areaPrev: B.cell_area[i], thermalAgeMyr: B.thermal_age_myr[i], compositionDensity: B.composition_density[i], ageMyr: B.age[i] }, p);
    worst = Math.max(worst, Math.abs(A.volume_added[i] - r.added) / (r.added + 1e-3 * B.crust_volume[i] * 1e-3));
    added += A.volume_added[i];
    const dA = A.cell_area[i] - B.cell_area[i];
    if (kind(pts[i]) === 'young' && dA > 0) { young.dA += dA; young.dV += A.volume_added[i]; young.n++; }
    if (kind(pts[i]) !== 'young') thick = Math.max(thick, A.volume_added[i]);
  }
  const surface = (S) => { let v = 0; for (let i = 0; i < n; i++) if (S.state[i] === 1) v += S.crust_volume[i]; return v; };
  const source = A.counters.volumeSource - B.counters.volumeSource, sink = A.counters.volumeSink - B.counters.volumeSink;
  const dV = surface(A) - surface(B);
  console.log(`  ${young.n} stretched young columns: ${(young.dV / young.dA).toFixed(4)} km of crust per unit new area (F H_m = ${p.melt_fraction * p.melt_depth_km}); old ocean and continent: max melt ${thick}`);
  console.log(`  worst column vs reference ${worst.toExponential(1)}; budget: d(sum V) ${dV.toExponential(6)}, sources ${source.toExponential(6)} (sum of columns ${added.toExponential(6)}), sinks ${sink.toExponential(6)} (slab V ${A.crust_volume[sinkCol].toExponential(6)})`);
  assert.ok(young.n > 50 && Math.abs(young.dV / young.dA / (p.melt_fraction * p.melt_depth_km) - 1) < 1e-4);
  assert.equal(thick, 0);
  assert.ok(worst < 1e-3);
  assert.ok(Math.abs(source / added - 1) < 1e-5, 'logged sources = sum of columns');
  assert.ok(Math.abs(sink / A.crust_volume[sinkCol] - 1) < 1e-6, 'logged sink = slab volume');
  // Identity to the float32 rounding of each column's V (at most 6e-8 V per column touched).
  let rounding = 0;
  for (let i = 0; i < n; i++) if (A.volume_added[i] > 0) rounding += 6e-8 * A.crust_volume[i];
  console.log(`  budget mismatch ${(dV - (source - sink)).toExponential(1)}, float32 rounding bound ${rounding.toExponential(1)}`);
  assert.ok(Math.abs(dV - (source - sink)) < rounding, `budget ${dV} vs ${source - sink}`);
});

test('splits: halves share volume and area_ref, sit on the cell axis, start bonded and stress-free; the tiling stays exact', { timeout: 300_000 }, async () => {
  // Cells larger than 1.15 x the base hexagon split (jitter and apothem spread
  // make a few dozen). No melt, so total volume and reference area are conserved.
  const { pts, spacing, state } = sphereState({ kind: () => 'old' });
  const n = pts.length;
  const page = await openPage(SPHERE_PAGE);
  const fields = ['cell_area', 'crust_volume', 'area_ref', 'position', 'cell_shape', 'cell_clipped', 'is_active'];
  const out = await page.evaluate(async ([st, params, readSrc, fields]) => {
    const read = eval(readSrc);
    await plate.setParams({ ...params, split_area_factor: 1.15, melt_fraction: 0, subduction_overlap: 1 }); // no slabs
    const restore = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, Array.isArray(v) ? Float32Array.from(v) : v]));
    await plate.loadState({ ...st, particles: restore(st.particles), bonds: restore(st.bonds) });
    const cap = 4096;
    const before = await read(cap, fields);
    await plate.runSteps(1);
    const after = await read(cap, fields);
    const r = await plate.debug.edges();
    const E = [];
    for (const [key, e] of r.pairs) E.push({ key, integrity: r.E.integrity[e], stress: Array.from(r.E.slice_stress_mpa.subarray(8 * e, 8 * e + 8)) });
    await plate.setParams({ split_area_factor: 100 });
    await plate.runSteps(1);
    return { before, after, E, next: await read(cap, ['cell_area', 'is_active', 'state']), problems: r.problems, base: Math.sqrt(3) * 2 * params.base_apothem ** 2 };
  }, [jsonState(state), SPHERE_PARAMS(spacing), READ_COLUMNS, fields]);
  assert.deepEqual(page.errors, []);
  await page.close();

  const { before: B, after: A, base } = out;
  const splitters = [];
  for (let i = 0; i < n; i++) if (B.cell_area[i] > 1.15 * base && !B.cell_clipped[i]) splitters.push(i);
  const children = [];
  for (let i = n; i < 4096; i++) if (A.is_active[i]) children.push(i);
  const sum = (S, f, m) => { let v = 0; for (let i = 0; i < m; i++) if (S.is_active[i]) v += S[f][i]; return v; };
  console.log(`  ${splitters.length} cells above 1.15 x base split into ${children.length} children; sum V ${sum(A, 'crust_volume', 4096) / sum(B, 'crust_volume', n) - 1}, sum area_ref ${sum(A, 'area_ref', 4096) / sum(B, 'area_ref', n) - 1} (relative change)`);
  assert.ok(splitters.length > 5 && children.length === splitters.length);
  assert.ok(Math.abs(sum(A, 'crust_volume', 4096) / sum(B, 'crust_volume', n) - 1) < 1e-6);
  assert.ok(Math.abs(sum(A, 'area_ref', 4096) / sum(B, 'area_ref', n) - 1) < 1e-6);
  assert.deepEqual(out.problems, []);
  const edgeOf = new Map(out.E.map((x) => [x.key, x]));
  let worstPos = 0;
  for (const i of splitters) {
    const p = B.position.slice(4 * i, 4 * i + 3), sh = B.cell_shape.slice(4 * i, 4 * i + 4);
    const want = splitColumn(p, { centroid: [sh[0], sh[1]], axis: sh[2], spread: sh[3] }, 1, 1);
    // The child: the new column nearest the reference child position.
    const dist = (c) => Math.hypot(...want.child.map((v, m) => v - A.position[4 * c + m]));
    const c = children.reduce((best, x) => (dist(x) < dist(best) ? x : best));
    const pair = edgeOf.get(`${Math.min(i, c)},${Math.max(i, c)}`);
    assert.ok(pair, `no edge from parent ${i} to its child ${c}`);
    assert.equal(pair.integrity, 1);
    // Stress-free up to the float32 rounding of positions: ~6e-8 over a rest
    // length of ~0.05 is a strain of ~1e-6, x E = 0.1 MPa.
    assert.ok(pair.stress.every((v) => Math.abs(v) < 0.2), `parent-child edge stressed: ${pair.stress}`);
    assert.ok(Math.abs(A.crust_volume[i] / (B.crust_volume[i] / 2) - 1) < 1e-6 && A.crust_volume[c] === A.crust_volume[i]);
    assert.ok(Math.abs(A.area_ref[i] / (B.area_ref[i] / 2) - 1) < 1e-6 && A.area_ref[c] === A.area_ref[i]);
    const got = { parent: A.position.slice(4 * i, 4 * i + 3), child: A.position.slice(4 * c, 4 * c + 3) };
    for (const w of ['parent', 'child']) worstPos = Math.max(worstPos, Math.hypot(...got[w].map((v, m) => v - want[w][m])) / spacing);
  }
  const total = out.next.cell_area.reduce((s, a, i) => s + (out.next.is_active[i] && out.next.state[i] === 1 ? a : 0), 0);
  console.log(`  halves vs reference positions: worst ${worstPos.toExponential(1)} spacings; after the split step, sum of cells - 4 pi = ${(total - 4 * Math.PI).toExponential(1)}`);
  // WGSL's cos and sin are accurate to 2^-11 (absolute): ~1e-5 of the offset.
  assert.ok(worstPos < 2e-4);
  assert.ok(Math.abs(total / (4 * Math.PI) - 1) < 1e-5);
});

test('resuming from a record ZIP in a fresh page continues the run', { timeout: 600_000 , ...PLANET }, async () => {
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

test('record ZIP is complete and self-consistent', { timeout: 300_000 , ...PLANET }, async () => {
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
