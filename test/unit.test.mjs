// CPU-side tests: run with `npm run test:unit` (no browser or GPU needed).
import test from 'node:test';
import assert from 'node:assert/strict';
import { crc32, readZip, writeZip } from '../src/record/zip.js';
import { eastNorth, latLonDeg, sampleToMesh } from '../src/record/sampling.js';
import { CellList } from '../src/init/cell-list.js';
import { generateInitialState } from '../src/init/initial-state.js';
import { createIcosphere } from '../src/geometry/icosphere.js';
import { mulberry32, randomOnSphere } from '../src/init/random.js';

test('crc32 matches the standard check value', () => {
  assert.equal(crc32(new TextEncoder().encode('123456789')), 0xcbf43926);
});

test('zip round-trips text and binary entries', () => {
  const floats = Float32Array.from({ length: 1000 }, (_, i) => Math.sin(i));
  const zip = writeZip([
    { name: 'manifest.json', data: '{"a": 1}' },
    { name: 'dir/data.f32', data: floats },
  ]);
  const files = readZip(zip);
  assert.equal(new TextDecoder().decode(files.get('manifest.json')), '{"a": 1}');
  const back = new Float32Array(files.get('dir/data.f32').slice().buffer);
  assert.deepEqual(back, floats);
});

test('cell list finds exactly the pairs brute force finds', () => {
  const rng = mulberry32(7);
  const n = 800;
  const pos = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) pos.set(randomOnSphere(rng), 3 * i);
  const cutoff = 0.12;
  const fast = new Set();
  new CellList(pos, cutoff).forEachPair((i, j) => fast.add(`${i},${j}`));
  const slow = new Set();
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const d = Math.hypot(pos[3 * i] - pos[3 * j], pos[3 * i + 1] - pos[3 * j + 1], pos[3 * i + 2] - pos[3 * j + 2]);
      if (d < cutoff) slow.add(`${i},${j}`);
    }
  assert.deepEqual([...fast].sort(), [...slow].sort());
});

test('initial state is a pure function of the seed', () => {
  const a = generateInitialState(642, { seed: 3 });
  const b = generateInitialState(642, { seed: 3 });
  const c = generateInitialState(642, { seed: 4 });
  assert.deepEqual(a.particles.position, b.particles.position);
  assert.deepEqual(a.bonds.damage, b.bonds.damage);
  assert.notDeepEqual(a.particles.position, c.particles.position);
});

test('sampling reproduces constant and linear fields, and support ~ coverage', () => {
  const state = generateInitialState(642, { seed: 1 });
  const P = state.particles;
  const area = Float32Array.from(P.apothem, (a) => 2 * Math.sqrt(3) * a * a);
  const constant = new Float32Array(P.count).fill(5);
  const linear = Float32Array.from({ length: P.count }, (_, i) => P.position[3 * i]); // f = x
  const mesh = createIcosphere(4);
  const nodes = Float32Array.from(mesh.vertices.flat());
  const spacing = 2 * 0.0175;
  const out = sampleToMesh(nodes, { position: P.position, area }, { constant, linear }, spacing);

  for (const v of out.constant) assert.ok(Math.abs(v - 5) < 1e-4);
  // f = x changes by up to one spacing per spacing. Away from holes in the
  // packing (support > 0.5) the estimate should be within ~half a spacing.
  let worst = 0, sq = 0;
  out.linear.forEach((v, k) => {
    const e = Math.abs(v - nodes[3 * k]);
    sq += e * e;
    if (out.support[k] > 0.5) worst = Math.max(worst, e);
  });
  const rms = Math.sqrt(sq / out.linear.length);
  assert.ok(rms < 0.1 * spacing, `linear field rms error ${rms}`);
  assert.ok(worst < 0.6 * spacing, `linear field worst error (support > 0.5) ${worst}`);
  const meanSupport = out.support.reduce((s, v) => s + v, 0) / out.support.length;
  // Hexagons cover 85% of the sphere; the 3-sigma cutoff drops ~1% of the kernel.
  assert.ok(Math.abs(meanSupport - 0.85) < 0.03, `mean support ${meanSupport}`);
});

test('east/north frame is orthonormal and tangent', () => {
  const rng = mulberry32(2);
  for (let k = 0; k < 100; k++) {
    const p = randomOnSphere(rng);
    const [e, n] = eastNorth(...p);
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    assert.ok(Math.abs(dot(e, p)) < 1e-6 && Math.abs(dot(n, p)) < 1e-6 && Math.abs(dot(e, n)) < 1e-6);
    assert.ok(Math.abs(dot(e, e) - 1) < 1e-6 && Math.abs(dot(n, n) - 1) < 1e-6);
    // north points toward increasing latitude
    const [lat] = latLonDeg(...p);
    const [lat2] = latLonDeg(p[0] + 1e-4 * n[0], p[1] + 1e-4 * n[1], p[2] + 1e-4 * n[2]);
    assert.ok(lat2 > lat);
  }
});
