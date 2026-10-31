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
import { Constraint, Particle, unpack } from '../src/gpu/layout.js';
import { readZip } from '../src/record/zip.js';
import { createIcosphere } from '../src/geometry/icosphere.js';
import { CHAIN, chainExact, chainState } from './support/chain.mjs';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const PAGE = pathToFileURL(fileURLToPath(new URL('../dist/plate-player.html', import.meta.url))).href + '?paused';
const MAX_PARTICLES = 25000;
const MAX_CONSTRAINTS = 100000;
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
    const bufs = await plate.readBuffers(['particles', 'constraints', 'particlePool', 'constraintPool', 'grid']);
    return Object.fromEntries(Object.entries(bufs).map(([k, v]) => [k, toB64(v)]));
  }, toB64);
  return Object.fromEntries(Object.entries(b64).map(([k, v]) => [k, fromB64(v)]));
}

function checkFreeLists(state) {
  const P = unpack(Particle, state.particles, MAX_PARTICLES);
  const C = unpack(Constraint, state.constraints, MAX_CONSTRAINTS);
  for (const [name, pool, active, cap] of [
    ['particle', new Uint32Array(state.particlePool), P.is_active, MAX_PARTICLES],
    ['constraint', new Uint32Array(state.constraintPool), C.is_active, MAX_CONSTRAINTS],
  ]) {
    const count = pool[0];
    assert.equal(pool[1], 0, `${name} pool: split requests not cleared`);
    const free = new Uint8Array(cap);
    for (let k = 0; k < count; k++) {
      const idx = pool[4 + k];
      assert.ok(idx < cap, `${name} pool holds out-of-range index ${idx}`);
      assert.equal(free[idx], 0, `${name} slot ${idx} is in the free list twice`);
      assert.equal(active[idx], 0, `${name} slot ${idx} is both free and active`);
      free[idx] = 1;
    }
    let activeCount = 0;
    for (let i = 0; i < cap; i++) activeCount += active[i] ? 1 : 0;
    assert.equal(activeCount + count, cap, `${name}: ${activeCount} active + ${count} free != ${cap}`);
  }
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
  const vertices = createIcosphere(3).vertices;
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
  let maxSpeed = 0;
  for (let i = 0; i < MAX_PARTICLES; i++) {
    if (!P.is_active[i]) continue;
    const r = Math.hypot(P.position[4 * i], P.position[4 * i + 1], P.position[4 * i + 2]);
    assert.ok(Math.abs(r - 1) < 1e-4, `particle ${i} left the sphere (r = ${r})`);
    const v = Math.hypot(P.velocity[4 * i], P.velocity[4 * i + 1], P.velocity[4 * i + 2]);
    assert.ok(Number.isFinite(v), `particle ${i} velocity is not finite`);
    maxSpeed = Math.max(maxSpeed, v);
  }
  const psi = new Float32Array(state.grid).filter((_, w) => w % 12 === 8); // GridNode.psi at word 8 of 12
  const maxPsi = Math.max(...psi.map(Math.abs));
  assert.ok(Number.isFinite(maxPsi) && maxPsi < 10, `mantle psi out of bounds: ${maxPsi}`);
  console.log(`  max speed ${(maxSpeed * 637.1).toFixed(1)} cm/yr, max |psi| ${maxPsi.toFixed(3)}`);
  assert.ok(maxSpeed * 637.1 < 100, 'plate speed above 100 cm/yr');
});

test('two bonded particles relax as implicit Euler with tau = gamma / 2k', { timeout: 300_000 }, async () => {
  // One stretched bond, no mantle flow. Drag-as-mass XPBD should reproduce
  // implicit Euler for  dC/dt = -C / tau  exactly: C_n = C_0 (1 + dt/tau)^-n,
  // for time steps far below, at, and far above tau.
  const page = await openPage();
  const results = await page.evaluate(async () => {
    const a = 0.0175, L = 3 * a, eps = 0.1;
    await plate.setParams({ blanket_forcing: 0, mantle_time_per_myr: 0, crust_youngs_modulus_gpa: 70 });
    const kappa70 = plate.sim.params.bond_rate_myr_inv;
    await plate.setParams({ crust_youngs_modulus_gpa: 70 * 0.5 / kappa70 }); // kappa = 0.5/Myr, tau = 1/(2 kappa) = 1 Myr
    const tau = 1 / (2 * plate.sim.params.bond_rate_myr_inv);
    const half = Math.asin(L * (1 + eps) / 2);
    const state = () => ({
      particles: {
        count: 2,
        position: Float32Array.from([-Math.sin(half), 0, Math.cos(half), Math.sin(half), 0, Math.cos(half)]),
        thickness: Float32Array.of(7, 7), compositionDensity: Float32Array.of(3000, 3000),
        temperature: Float32Array.of(278, 278), age: Float32Array.of(0, 0), apothem: Float32Array.of(a, a),
      },
      bonds: { count: 1, a: Uint32Array.of(0), b: Uint32Array.of(1), restLength: Float32Array.of(L), damage: Float32Array.of(0) },
      mantlePsi: new Float32Array(plate.sim.numNodes),
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
  // A pre-compressed chain with kappa * dt = 200 (the stiff regime of a 70 GPa
  // lithosphere): the exact implicit step is a tridiagonal solve. Plain Jacobi
  // propagates the end displacement about one bond per sweep.
  const K = 200;
  const page = await openPage();
  const runs = await page.evaluate(async ([chain, builderSrc, K]) => {
    const build = eval(`(${builderSrc})`);
    await plate.setParams({ blanket_forcing: 0, mantle_time_per_myr: 0, crust_youngs_modulus_gpa: 70 });
    await plate.setParams({ dt_crust_myr: K / plate.sim.params.bond_rate_myr_inv, solver_iterations: 128 });
    const out = {};
    for (const [name, rho] of [['plain', 0], ['chebyshev', 0.999]]) {
      await plate.setParams({ chebyshev_rho: rho });
      await plate.loadState(build(chain, plate.sim.numNodes));
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

  assert.ok(manifest.snapshots.length >= 1);
  for (const snap of manifest.snapshots) {
    for (const { file, bytes } of Object.values(snap.buffers)) assert.equal(files.get(file)?.length, bytes, file);
  }
});
