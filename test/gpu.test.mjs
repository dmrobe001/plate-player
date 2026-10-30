// GPU acceptance tests for design phases 0 and 1. They drive the built page
// (dist/plate-player.html) in Chromium through `window.plate`.
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

test('phase 0: free lists account for every slot; cached mantle node matches brute force', { timeout: 600_000 }, async () => {
  const page = await openPage();
  await page.evaluate(() => plate.runSteps(100));
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

test('phase 0: 10x time step stays finite and bounded; dt sensitivity reported', { timeout: 900_000 }, async () => {
  const T = 1.6; // Myr
  const runAt = async (dt) => {
    const page = await openPage();
    await page.evaluate(async (dt) => { await plate.setParams({ dt_crust_myr: dt }); }, dt);
    await page.evaluate((n) => plate.runSteps(n), Math.round(T / dt));
    const state = await readState(page);
    assert.deepEqual(page.errors, []);
    await page.close();
    return state;
  };
  const fine = await runAt(0.016);
  const coarse = await runAt(0.16);

  for (const state of [fine, coarse]) {
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
  }

  // Sensitivity to dt, as a baseline for phase 2 (PBD stiffness still
  // depends on dt, so this is reported rather than asserted): RMS difference
  // of particle positions present in both runs, relative to RMS displacement.
  const Pf = unpack(Particle, fine.particles, MAX_PARTICLES);
  const Pc = unpack(Particle, coarse.particles, MAX_PARTICLES);
  const P0 = await (async () => {
    const page = await openPage();
    const s = await readState(page);
    await page.close();
    return unpack(Particle, s.particles, MAX_PARTICLES);
  })();
  let diff = 0, disp = 0, n = 0;
  for (let i = 0; i < MAX_PARTICLES; i++) {
    if (!P0.is_active[i] || !Pf.is_active[i] || !Pc.is_active[i] || Pf.state[i] !== 1 || Pc.state[i] !== 1) continue;
    let d2 = 0, m2 = 0;
    for (let c = 0; c < 3; c++) {
      d2 += (Pf.position[4 * i + c] - Pc.position[4 * i + c]) ** 2;
      m2 += (Pf.position[4 * i + c] - P0.position[4 * i + c]) ** 2;
    }
    diff += d2; disp += m2; n++;
  }
  console.log(`  dt 0.16 vs 0.016 over ${T} Myr: rms position difference / rms displacement = ${Math.sqrt(diff / disp).toFixed(3)} (${n} particles)`);
});

test('phase 1: restart from a record ZIP in a fresh page reproduces the continuation', { timeout: 900_000 }, async () => {
  const N = 30;
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
  // Splits draw free slots in thread-timing order (documented
  // nondeterminism). Report whether any happened in the window.
  const freeAt = (s) => new Uint32Array(s.particlePool)[0];
  console.log(`  free particle slots after ${2 * N} steps: ${freeAt(continuous)} (continuous) vs ${freeAt(restarted)} (restarted)`);
  for (const name of ['grid', 'constraints', 'particlePool', 'constraintPool']) {
    const x = new Uint8Array(continuous[name]), y = new Uint8Array(restarted[name]);
    let first = -1;
    for (let k = 0; k < x.length && first < 0; k++) if (x[k] !== y[k]) first = k;
    assert.equal(first, -1, `${name} differs after restart (first differing byte ${first})`);
  }
  // Particles: every field bit for bit, except next_idx, the collision hash
  // grid's linked-list pointer. It is rebuilt each step by atomic exchanges
  // whose order depends on thread timing, and carries no physical state.
  const Pa = unpack(Particle, continuous.particles, MAX_PARTICLES);
  const Pb = unpack(Particle, restarted.particles, MAX_PARTICLES);
  for (const [field] of Particle.fields) {
    if (field === 'next_idx') continue;
    const x = new Uint32Array(Pa[field].buffer), y = new Uint32Array(Pb[field].buffer); // compare bits, not values (NaN-safe)
    const bad = x.findIndex((v, k) => v !== y[k]);
    assert.equal(bad, -1, `particle field ${field} differs after restart (first at element ${bad})`);
  }
});

test('phase 1: record ZIP is complete and self-consistent', { timeout: 600_000 }, async () => {
  const page = await openPage();
  await page.evaluate(() => { plate.recorder.options.fieldIntervalMyr = 0.16; }); // every 10 steps
  await page.evaluate(() => plate.runSteps(30));
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
  assert.equal(time.length, 4); // steps 0, 10, 20, 30
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
