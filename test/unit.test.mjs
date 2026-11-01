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
  const a = generateInitialState({ seed: 3 });
  const b = generateInitialState({ seed: 3 });
  const c = generateInitialState({ seed: 4 });
  assert.deepEqual(a.particles.position, b.particles.position);
  assert.deepEqual(a.bonds.damage, b.bonds.damage);
  assert.notDeepEqual(a.particles.position, c.particles.position);
});

test('sampling reproduces constant and linear fields, and support ~ coverage', () => {
  const state = generateInitialState({ seed: 1 });
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
  // Mean support is the hexagons' coverage of the sphere; the 3-sigma cutoff drops ~1% of the kernel.
  assert.ok(Math.abs(meanSupport - state.stats.coverage) < 0.03, `mean support ${meanSupport} vs coverage ${state.stats.coverage}`);
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

// --- plates ---------------------------------------------------------------
import { eulerPole, labelPlates } from '../src/plates/plates.js';

/** Synthetic network: rows of particles along the equator, bonded in chains. */
function chains(lengths, { damagedBetween = [] } = {}) {
  const n = lengths.reduce((a, b) => a + b, 0);
  const P = { is_active: new Uint32Array(n).fill(1), state: new Uint32Array(n).fill(1), position: new Float32Array(4 * n), velocity: new Float32Array(4 * n) };
  const a = [], b = [], integrity = [];
  let i = 0;
  lengths.forEach((len, c) => {
    for (let k = 0; k < len; k++, i++) {
      const phi = i * 0.03, lat = c * 0.01;
      P.position.set([Math.cos(lat) * Math.sin(phi), Math.sin(lat), Math.cos(lat) * Math.cos(phi)], 4 * i);
      if (k > 0) { a.push(i - 1); b.push(i); integrity.push(1); }
    }
    if (c > 0) { a.push(i - len - 1); b.push(i - len); integrity.push(damagedBetween.includes(c) ? 0 : 1); } // link to previous chain
  });
  const C = { is_active: new Uint32Array(a.length).fill(1), particle_a: Uint32Array.from(a), particle_b: Uint32Array.from(b), integrity: Float32Array.from(integrity) };
  return { P, C };
}

test('plates: components of intact edges, persistent ids, split and merge events', () => {
  // Two chains joined by an intact edge: one plate.
  const joined = chains([30, 30]);
  const first = labelPlates(joined.P, joined.C);
  assert.equal(first.plates.length, 1);
  assert.equal(first.plates[0].size, 60);

  // Damage the joint: two plates; the larger share keeps the id, the other is a split.
  const split = chains([30, 30], { damagedBetween: [1] });
  const second = labelPlates(split.P, split.C, first);
  assert.equal(second.plates.length, 2);
  assert.ok(second.plates.some((p) => p.id === first.plates[0].id));
  assert.deepEqual(second.events.map((e) => e.type), ['plate_split']);

  // Heal it: back to one plate, and the other id is reported as absorbed.
  const third = labelPlates(joined.P, joined.C, second);
  assert.equal(third.plates.length, 1);
  assert.deepEqual(third.events.map((e) => e.type), ['plate_merge']);

  // Components below the minimum size are not plates.
  const small = chains([10]);
  assert.equal(labelPlates(small.P, small.C).plates.length, 0);
});

test('plates: Euler pole recovers a rigid rotation, and the residual measures non-rigid motion', () => {
  const { P, C } = chains([40, 40]);
  const omega = [0.002, -0.001, 0.003]; // rad/Myr
  const n = P.is_active.length;
  for (let i = 0; i < n; i++) {
    const r = P.position.subarray(4 * i, 4 * i + 3);
    P.velocity.set([omega[1] * r[2] - omega[2] * r[1], omega[2] * r[0] - omega[0] * r[2], omega[0] * r[1] - omega[1] * r[0]], 4 * i);
  }
  const [plate] = labelPlates(P, C).plates;
  const rigid = eulerPole(P, plate.members);
  rigid.omega.forEach((w, k) => assert.ok(Math.abs(w - omega[k]) < 1e-6, `omega ${rigid.omega}`));
  assert.ok(rigid.rigidityResidual < 1e-4, `residual ${rigid.rigidityResidual}`);

  // Add a non-rigid stretch: the residual grows accordingly.
  for (let i = 0; i < n; i++) P.velocity[4 * i] += 1e-3 * (i / n - 0.5);
  assert.ok(eulerPole(P, plate.members).rigidityResidual > 0.05);
});

// --- mantle -----------------------------------------------------------------
import { SphereMesh } from '../src/mantle/sphere-mesh.js';
import { Mantle } from '../src/mantle/mantle.js';

test('sphere mesh: Laplacian, Poisson solve and gradient match a degree-2 harmonic', () => {
  const mesh = new SphereMesh(4);
  const f = Float64Array.from({ length: mesh.n }, (_, i) => 3 * mesh.pos[3 * i + 2] ** 2 - 1); // l = 2: lap f = -6 f
  const rel = (a, b) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / b.reduce((s, v) => s + v * v, 0));
  assert.ok(rel(mesh.laplacian(f), f.map((v) => -6 * v)) < 0.02);
  const phi = new Float64Array(mesh.n);
  mesh.solvePoisson(f, phi);
  assert.ok(rel(phi, f.map((v) => -v / 6)) < 0.01);
  const z = Float64Array.from({ length: mesh.n }, (_, i) => mesh.pos[3 * i + 2]); // grad z = e_z - z r
  const expected = new Float64Array(3 * mesh.n);
  for (let i = 0; i < mesh.n; i++) { const r = mesh.point(i); expected.set([-r[2] * r[0], -r[2] * r[1], 1 - r[2] * r[2]], 3 * i); }
  assert.ok(rel(mesh.gradient(z), expected) < 0.01);
});

test('mantle: same seed same state, resumes exactly from saved state, and keeps evolving', () => {
  const opts = { subdivisions: 3, spinUpMyr: 60 }; // small and short, for speed
  const a = new Mantle(7, opts), b = new Mantle(7, opts);
  a.spinUp(); b.spinUp();
  assert.deepEqual(a.b, b.b);

  a.advanceTo(10);
  const saved = JSON.parse(JSON.stringify(a.getState()));
  const c = new Mantle(7, opts);
  c.setState(saved);
  a.advanceTo(20); c.advanceTo(20);
  assert.deepEqual(a.b, c.b);

  // "Lava lamp": the buoyancy pattern decorrelates over a few plume lifetimes.
  const start = Float64Array.from(a.b);
  a.advanceTo(260);
  const dot = (x, y) => x.reduce((s, v, i) => s + v * y[i], 0);
  const corr = dot(start, a.b) / Math.sqrt(dot(start, start) * dot(a.b, a.b));
  assert.ok(Math.abs(corr) < 0.5, `correlation after 250 Myr ${corr}`);
  assert.ok(a.rmsSpeedCmYr() > 0.5 && a.rmsSpeedCmYr() < 30, `rms speed ${a.rmsSpeedCmYr()}`);
});

test('initial bonds are stress-free with rest length = apothem sum, and unbonded pairs do not overlap', () => {
  const { particles: P, bonds: B, stats } = generateInitialState({ seed: 2 });
  const p = P.position, a = P.apothem;
  let worst = 0;
  const bonded = new Set();
  for (let k = 0; k < B.count; k++) {
    const i = B.a[k], j = B.b[k];
    bonded.add(Math.min(i, j) * P.count + Math.max(i, j));
    const d = Math.hypot(p[3 * i] - p[3 * j], p[3 * i + 1] - p[3 * j + 1], p[3 * i + 2] - p[3 * j + 2]);
    worst = Math.max(worst, Math.abs(d - a[i] - a[j]) / (a[i] + a[j]));
  }
  assert.ok(worst < 2e-5, `worst bond strain ${worst} (fit reported ${stats.maxBondStrain})`);
  let overlap = 0;
  new CellList(p, 2 * Math.max(...a)).forEachPair((i, j, dx, dy, dz, d) => {
    if (!bonded.has(Math.min(i, j) * P.count + Math.max(i, j))) overlap = Math.max(overlap, (a[i] + a[j] - d) / (a[i] + a[j]));
  });
  assert.ok(overlap < 2e-5, `worst unbonded overlap ${overlap}`);
});

// --- edge topology (design 3.2, phase A): CPU reference ---------------------
import { columnsFrom, inRange, partnerAngle, rangeMargin, updateEdges, pairKey } from '../src/edges/topology.js';

const EDGE_PARAMS = {
  edge_range_margin: 0.1, subduction_overlap: 0.012,
  continental_lithosphere_km: 150, oceanic_lithosphere_max_km: 100, thermal_diffusivity_m2_s: 1e-6,
};

/** Columns at given unit-sphere positions (stride 3), old ocean at the surface unless overridden. */
function cols(points, { apothem = 0.0175, elevation = null, state = null } = {}) {
  const n = points.length;
  const P = {
    position: Float32Array.from(points.flat()),
    apothem: new Float32Array(n).fill(apothem),
    elevation: elevation ? Float32Array.from(elevation) : new Float32Array(n).fill(-3),
    composition_density: new Float32Array(n).fill(3000),
    age: new Float32Array(n).fill(100),
    state: state ? Uint32Array.from(state) : new Uint32Array(n).fill(1),
    is_active: new Uint32Array(n).fill(1),
  };
  return columnsFrom(P, EDGE_PARAMS, { stride: 3, positionField: 'position' });
}

/** Triangular lattice of spacing 2a around the north pole: `rings` hexagonal rings. */
function hexPatch(rings, a = 0.0175) {
  const pts = [];
  for (let q = -rings; q <= rings; q++)
    for (let r = -rings; r <= rings; r++) {
      if (Math.abs(q + r) > rings) continue;
      const x = 2 * a * (q + r / 2), y = 2 * a * (r * Math.sqrt(3) / 2);
      const z = Math.sqrt(1 - x * x - y * y);
      pts.push([x, y, z]);
    }
  return pts;
}

test('edges: a triangular lattice links each column to its 6 nearest neighbours only', () => {
  // Chords are 2a, so the gap is 0 for nearest neighbours and ~0.73 of the
  // apothem sum for the next shell (sqrt 3 spacing): far outside the 10% margin.
  // (The lattice is flat; on the sphere the chords shrink by ~1e-4, still in range.)
  const c = cols(hexPatch(3));
  const { edges, created } = updateEdges(c, new Set(), EDGE_PARAMS);
  assert.equal(created.size, edges.size);
  const degree = new Uint8Array(c.count);
  for (const k of edges) for (const i of k.split(',').map(Number)) degree[i]++;
  // 37 columns: 3*3*(3+1)+1; interior (2 rings, 19 columns) has 6 neighbours each.
  assert.equal(c.count, 37);
  const interior = hexPatch(2).length;
  assert.equal(degree[18], 6); // the centre
  assert.equal([...degree].filter((d) => d === 6).length, interior);
  // Angles of the centre's partners are 60 degrees apart.
  const centre = 18;
  const angles = [...edges].map((k) => k.split(',').map(Number)).filter(([i, j]) => i === centre || j === centre)
    .map(([i, j]) => partnerAngle(c, centre, i === centre ? j : i)).sort((x, y) => x - y);
  for (let k = 1; k < 6; k++) assert.ok(Math.abs(angles[k] - angles[k - 1] - Math.PI / 3) < 1e-3, `angles ${angles}`);
});

test('edges: a pair is in range exactly while its gap is below the margin and the columns overlap in depth', () => {
  const a = 0.0175;
  const pairAt = (gapFraction, elevation = null) => {
    const half = Math.asin((2 * a * (1 + gapFraction)) / 2); // chord = (1 + gap) * apothem sum
    return cols([[-Math.sin(half), 0, Math.cos(half)], [Math.sin(half), 0, Math.cos(half)]], { elevation });
  };
  assert.ok(inRange(pairAt(0.099), 0, 1, EDGE_PARAMS));
  assert.ok(!inRange(pairAt(0.101), 0, 1, EDGE_PARAMS));
  assert.ok(Math.abs(rangeMargin(pairAt(0.05), 0, 1, EDGE_PARAMS) - 0.05) < 1e-6);
  // Same map view, but one column's top is below the other's 100 km lithosphere.
  assert.ok(inRange(pairAt(-0.05, [-3, -90]), 0, 1, EDGE_PARAMS));
  assert.ok(!inRange(pairAt(-0.05, [-3, -110]), 0, 1, EDGE_PARAMS));
});

test('edges: update keeps, creates and removes by range; deep overlap creates no edge but keeps an old one', () => {
  const a = 0.0175;
  const at = (phi) => [Math.sin(phi), 0, Math.cos(phi)];
  // 0-1 touching, 1-2 far apart; then 2 moves next to 1.
  const before = cols([at(0), at(2 * a), at(10 * a)]);
  const e0 = updateEdges(before, new Set(), EDGE_PARAMS);
  assert.deepEqual([...e0.edges], [pairKey(0, 1)]);
  const after = cols([at(0), at(2 * a * 1.15), at(2 * a * 2.15)]); // 0-1 gap 15%: out; 1-2 gap 0: in
  const e1 = updateEdges(after, e0.edges, EDGE_PARAMS);
  assert.deepEqual([...e1.removed], [pairKey(0, 1)]);
  assert.deepEqual([...e1.created], [pairKey(1, 2)]);
  // Two surface columns overlapping by more than subduction_overlap: no new edge...
  const deep = cols([at(0), at(2 * a - 0.02)]);
  assert.equal(updateEdges(deep, new Set(), EDGE_PARAMS).edges.size, 0);
  // ...but an existing edge between them is kept (removal is purely geometric).
  assert.deepEqual([...updateEdges(deep, new Set([pairKey(0, 1)]), EDGE_PARAMS).kept], [pairKey(0, 1)]);
});
