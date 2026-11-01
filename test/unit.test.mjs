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
  edge_range_margin: 0.1,
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
  // Deep overlap is just range: an edge (subduction starts from edge state, phase F).
  const deep = cols([at(0), at(2 * a - 0.02)]);
  assert.deepEqual([...updateEdges(deep, new Set(), EDGE_PARAMS).edges], [pairKey(0, 1)]);
});

// --- Column profile (design 4.0, phase B) ---------------------------------
import * as profile from '../src/columns/profile.js';
import { SIM_PARAMS } from '../src/sim/params.js';
import { ColumnProfile } from '../src/gpu/layout.js';

const PARAMS = Object.fromEntries(SIM_PARAMS.map(([n, , v]) => [n, v]));
const ocean = (ageMyr, crustKm = 7) => ({ crustKm, compositionDensity: 3000, ageMyr, elevationKm: 0 });
const continent = (crustKm = 35) => ({ crustKm, compositionDensity: 2700, ageMyr: 1000, elevationKm: 3 });

test('profile: geotherms solve the heat equation with their boundary conditions', () => {
  const p = PARAMS, kappa = p.thermal_diffusivity_m2_s, k = p.thermal_conductivity_w_m_k;
  const Ts = p.ambient_ocean_temp, Tm = p.mantle_temp;
  // Oceanic: dT/dt = kappa d2T/dz2 (finite differences in age and depth), T(0) = Ts,
  // 90% isotherm at the lithosphere base, heat flow = k dT/dz at the surface.
  for (const age of [2, 10, 40]) {
    const T = (z, t) => profile.geotherm(ocean(t), p).T(z);
    for (const z of [5, 20, 40]) {
      const dt = 0.01, dz = 0.05; // Myr, km
      const dTdt = (T(z, age + dt) - T(z, age - dt)) / (2 * dt * profile.SECONDS_PER_MYR);
      const d2Tdz2 = (T(z + dz, age) - 2 * T(z, age) + T(z - dz, age)) / (dz * 1e3) ** 2;
      assert.ok(Math.abs(dTdt - kappa * d2Tdz2) < 2e-3 * Math.abs(dTdt) + 1e-16, `age ${age} z ${z}: ${dTdt} vs ${kappa * d2Tdz2}`);
    }
    const g = profile.geotherm(ocean(age), p);
    assert.equal(g.T(0), Ts);
    assert.ok(Math.abs((g.T(g.L) - Ts) / (Tm - Ts) - 0.9) < 2e-3);
    const q = (k * (g.T(0.01) - g.T(0))) / 10; // W/m^2
    assert.ok(Math.abs(q * 1e3 / g.heatFlow - 1) < 2e-3, `heat flow ${q * 1e3} vs ${g.heatFlow}`);
  }
  // Continental: k T'' = -A in the heat-producing layer, T'' = 0 below it,
  // T(0) = Ts, T(L) = 90% isotherm, flux continuous; heat flow = k dT/dz at 0.
  const g = profile.geotherm(continent(35), p);
  const A = p.radiogenic_heat_uw_m3 * 1e-6, h = 0.05;
  const d2 = (z) => (g.T(z + h) - 2 * g.T(z) + g.T(z - h)) / (h * 1e3) ** 2;
  assert.ok(Math.abs(k * d2(7) + A) < 1e-3 * A);
  assert.ok(Math.abs(d2(60)) < 1e-3 * A / k);
  const slope = (z) => (g.T(z + h) - g.T(z - h)) / (2 * h * 1e3);
  assert.ok(Math.abs(slope(p.radiogenic_layer_km - 2 * h) - slope(p.radiogenic_layer_km + 2 * h)) < 0.02 * slope(60));
  assert.equal(g.T(0), Ts);
  assert.ok(Math.abs(g.T(g.L) - (Ts + 0.9 * (Tm - Ts))) < 1e-6);
  assert.ok(Math.abs((k * (g.T(0.01) - g.T(0))) / 10 * 1e3 / g.heatFlow - 1) < 2e-3);
  assert.equal(g.T(g.zMantle + 1), Tm);
});

test('profile: Airy elevation reproduces depth ~ sqrt(age), and flattens at the capped age', () => {
  const p = { ...PARAMS, oceanic_lithosphere_max_km: 1e4 }; // uncapped half-space
  const drho = p.mantle_density - p.water_density;
  // Crustless column of mantle composition: subsidence below the ridge is
  // 2 rho_m alpha (Tm - Ts) sqrt(kappa t / pi) / (rho_m - rho_w) (Turcotte & Schubert 4.209).
  const coeff = (2 * p.mantle_density * p.thermal_expansion_coeff * (p.mantle_temp - p.ambient_ocean_temp)) / drho;
  const ridge = profile.airy(ocean(0, 0), p).elevation;
  for (const t of [1, 4, 16, 64, 150]) {
    const subsidence = ridge - profile.airy(ocean(t, 0), p).elevation;
    const expected = coeff * Math.sqrt((p.thermal_diffusivity_m2_s * t * profile.SECONDS_PER_MYR) / Math.PI) / 1e3;
    assert.ok(Math.abs(subsidence / expected - 1) < 1e-4, `age ${t}: ${subsidence} km vs ${expected} km`);
  }
  // Default parameters: 7 km crust, depth grows to the capped age and then stays.
  const depth = (t) => PARAMS.sea_level - profile.airy(ocean(t), PARAMS).elevation;
  assert.ok(depth(0) < depth(10) && depth(10) < depth(50));
  assert.ok(Math.abs(depth(80) - depth(150)) < 1e-9);
  console.log(`  default hypsometry: ridge ${depth(0).toFixed(2)} km deep, old ocean ${depth(150).toFixed(2)} km deep, `
    + `35 km continent ${(-PARAMS.sea_level + profile.airy(continent(35), PARAMS).elevation).toFixed(2)} km high`);
});

test('profile: GPE differences give the half-space ridge push, independent of the integration depth', () => {
  const p = { ...PARAMS, oceanic_lithosphere_max_km: 1e4 };
  const kt = (t) => p.thermal_diffusivity_m2_s * t * profile.SECONDS_PER_MYR;
  const B = p.mantle_density * p.thermal_expansion_coeff * (p.mantle_temp - p.ambient_ocean_temp);
  for (const t of [10, 50, 100]) {
    // Turcotte & Schubert 6.405: F = g rho_m alpha dT kappa t [1 + 2 rho_m alpha dT / (pi (rho_m - rho_w))], N/m.
    const F = profile.GRAVITY * B * kt(t) * (1 + (2 * B) / (Math.PI * (p.mantle_density - p.water_density)));
    const push = (profile.gpe(ocean(0, 0), p) - profile.gpe(ocean(t, 0), p)) * 1e9;
    assert.ok(Math.abs(push / F - 1) < 1e-4, `age ${t}: ${push.toExponential(3)} vs ${F.toExponential(3)} N/m`);
  }
  for (const col of [ocean(30), continent(35), continent(70)]) {
    const a = profile.gpe(col, PARAMS), b = profile.gpe(col, PARAMS, undefined, undefined, 100);
    assert.ok(Math.abs(a - b) < 2, `GPE ${a} vs ${b} MPa km with a deeper base`); // ridge push ~2000 MPa km
  }
});

test('profile: the yield-strength envelope peaks at the brittle-ductile transition, at textbook temperatures', () => {
  const p = PARAMS;
  for (const [name, col, Tlo, Thi] of [
    ['oceanic mantle (olivine)', ocean(30, 0), 873, 1073],   // 600-800 C
    ['continental crust (quartzite)', continent(35), 523, 723], // 250-450 C
  ]) {
    const prof = profile.columnProfile(col, p);
    const g = profile.geotherm(col, p);
    // Search the first material only: a continent is a "jelly sandwich", and
    // its strongest point is brittle upper mantle below the weak lower crust.
    const zMax = col.crustKm > 0 ? col.crustKm : prof.columnKm;
    let best = { z: 0, s: -1 };
    for (let z = 0.05; z < zMax; z += 0.05) {
      const s = profile.envelope(col, p, z, 1, g).strength;
      if (s > best.s) best = { z, s };
    }
    const bdt = prof.brittleDuctileKm;
    const e = profile.envelope(col, p, bdt, 1, g);
    assert.ok(Math.abs(e.brittle / e.ductile - 1) < 1e-3, `${name}: curves do not cross at ${bdt}`);
    assert.ok(Math.abs(best.z - bdt) < 0.1, `${name}: envelope peaks at ${best.z} km, transition at ${bdt} km`);
    const T = g.T(bdt);
    assert.ok(T > Tlo && T < Thi, `${name}: transition at ${T} K`);
  }
  // Slices tile [0, column] from the top down, and match the struct the kernel writes.
  const prof = profile.columnProfile(ocean(30), p);
  let z = 0;
  for (const s of prof.slices) { assert.ok(Math.abs(s.z - 0.5 * s.dz - z) < 1e-9); z += s.dz; }
  assert.ok(Math.abs(z - prof.columnKm) < 1e-9);
  assert.equal(ColumnProfile.fields.find(([f]) => f === 'depth_km')[1], `array<f32, ${profile.SLICES}>`);
});

// --- Edge law (design 4.5, phase C) ----------------------------------------
import * as law from '../src/edges/law.js';
import * as shape from '../src/columns/shape.js';

// Parameters with the derived bond rate the GPU uses, and optional overrides.
const lawParams = (over = {}) => {
  const p = { ...PARAMS, ...over };
  const k = shape.normalStiffnessPerModulus(p.shear_stiffness_ratio) * p.crust_youngs_modulus_gpa * 1e9 * p.stiffness_reference_km * 1e3;
  const gamma = (2 * Math.sqrt(3) * (p.base_apothem * 6.371e6) ** 2 * p.asthenosphere_viscosity_pa_s) / (p.drag_reference_channel_km * 1e3);
  return { ...p, bond_rate_myr_inv: k / (gamma / profile.SECONDS_PER_MYR) };
};
// Creep with no temperature dependence: every slice has the same viscosity
// eta (Pa s), crust and mantle alike; cohesion out of the way.
const uniform = (eta) => {
  const n = 3.5, edot = PARAMS.reference_strain_rate_s;
  const sigmaMPa = 4 * eta * edot / 1e6; // differential stress = 2 x shear strength = 2 x 2 eta edot
  const log10a = Math.log10(edot) - n * Math.log10(sigmaMPa);
  const o = {};
  for (const m of ['quartzite', 'diabase', 'olivine']) Object.assign(o, { [`${m}_log10_a`]: log10a, [`${m}_n`]: n, [`${m}_q_kj`]: 0 });
  return lawParams({ ...o, cohesion_mpa: 1e5 });
};
const oldOcean = (e) => ({ crustKm: 7, compositionDensity: 3000, ageMyr: 100, elevationKm: e });

test('edge law: under constant separation rate the stress follows the implicit-Euler Maxwell sequence to eta * rate', () => {
  const E = PARAMS.crust_youngs_modulus_gpa * 1e9, dtS = PARAMS.dt_crust_myr * profile.SECONDS_PER_MYR;
  const eta = 3 * E * dtS; // Maxwell time eta / E = 3 steps
  const p = uniform(eta);
  const col = oldOcean(-3);
  const edge = law.edgeSlices(col, col, 1, p);
  const r = 1 / (1 + E * dtS / eta);
  for (const s of edge.slices) assert.ok(Math.abs(s.r / r - 1) < 1e-9, `slice r ${s.r} vs ${r}`);
  const rate = 50e6 / eta; // 1/s: eta * rate = 50 MPa, well below the ductile cap (4 x that)
  const dEps = rate * dtS;
  let stress = new Array(law.EDGE_SLICES).fill(0), exact = 0;
  for (let n = 1; n <= 30; n++) {
    stress = law.maxwellUpdate(stress, edge, dEps, p).stored;
    exact = (exact + E * dEps / 1e6) * r;
    assert.ok(Math.abs(stress[0] / exact - 1) < 1e-12);
  }
  const target = (eta * rate) / 1e6;
  assert.ok(Math.abs(stress[0] / target - 1) < 1e-3, `after 10 Maxwell times: ${stress[0]} vs eta rate ${target} MPa`);
  // The fixed point is exact for any step: eta rate (implicit Euler has no steady-state error).
  assert.ok(Math.abs((E * dEps / 1e6) * r / (1 - r) / target - 1) < 1e-12);
});

test('edge law: a cold pair is a spring, a hot pair a dashpot; real old ocean is both, by depth', () => {
  const E = PARAMS.crust_youngs_modulus_gpa * 1e9, dtS = PARAMS.dt_crust_myr * profile.SECONDS_PER_MYR;
  const col = oldOcean(-3);
  // Cold: eta = 1e6 E dt. A strain held for 100 steps keeps its stress.
  const pc = uniform(1e6 * E * dtS);
  const cold = law.edgeSlices(col, col, 1, pc);
  let s = law.maxwellUpdate(new Array(8).fill(0), cold, 1e-4, pc).stored;
  for (let n = 0; n < 100; n++) s = law.maxwellUpdate(s, cold, 0, pc).stored;
  assert.ok(Math.abs(s[0] / (E * 1e-4 / 1e6) - 1) < 2e-4, `spring lost stress: ${s[0]}`);
  const lc = law.linearize(new Array(8).fill(0), cold, 0, pc);
  assert.ok(Math.abs(lc.stiffness - cold.overlap / pc.stiffness_reference_km) < 1e-5, 'stiffness = overlap / reference');
  // Hot: eta = 1e-3 E dt. Stress is eta x this step's rate and is gone a step later.
  const ph = uniform(1e-3 * E * dtS);
  const hot = law.edgeSlices(col, col, 1, ph);
  const one = law.maxwellUpdate(new Array(8).fill(0), hot, 1e-4, ph);
  const etaRate = (1e-3 * E * dtS) * (1e-4 / dtS) / 1e6;
  assert.ok(Math.abs(one.actual[0] / etaRate - 1) < 2e-3, `dashpot ${one.actual[0]} vs ${etaRate}`);
  assert.ok(Math.abs(law.maxwellUpdate(one.stored, hot, 0, ph).actual[0]) < 2e-3 * etaRate);
  // Real old ocean: elastic on top, viscous below; the elastic thickness emerges.
  const p = lawParams();
  const real = law.edgeSlices(col, col, 1, p);
  const elastic = real.slices.reduce((a, sl) => a + sl.r * sl.dz, 0);
  assert.ok(real.slices[0].r > 0.999 && real.slices[7].r < 0.01, `top r ${real.slices[0].r}, bottom r ${real.slices[7].r}`);
  console.log(`  old ocean pair: elastic thickness ${elastic.toFixed(1)} km of ${real.overlap.toFixed(1)} km`);
});

test('edge law: a cold pair without integrity carries no tension, closes exactly at touching, and yields in compression', () => {
  const col = oldOcean(-3);
  // Real old ocean: at integrity 0 only the ductile base resists tension
  // (its flow stress), unless damage also weakens it.
  {
    const caps = (phi, w) => { const p = lawParams({ ductile_damage_weakening: w }); return law.linearize(new Array(8).fill(0), law.edgeSlices(col, col, phi, p), 0, p); };
    const ci = caps(1, 0), cb = caps(0, 0), cw = caps(0, 1);
    console.log(`  old ocean pair caps (MPa km): compression ${cb.Scomp.toFixed(0)}; tension ${ci.Stens.toFixed(0)} at integrity 1, `
      + `${cb.Stens.toFixed(0)} at 0 (ductile base), ${cw.Stens.toFixed(0)} at 0 with full ductile weakening`);
    assert.ok(ci.Stens > cb.Stens && cw.Stens === 0);
  }
  // A cold pair (no creep at all: every slice brittle) for the crack rules.
  const E0 = PARAMS.crust_youngs_modulus_gpa * 1e9, dtS = PARAMS.dt_crust_myr * profile.SECONDS_PER_MYR;
  const p = { ...uniform(1e6 * E0 * dtS), cohesion_mpa: PARAMS.cohesion_mpa };
  const E = p.crust_youngs_modulus_gpa * 1000;
  const intact = law.edgeSlices(col, col, 1, p), broken = law.edgeSlices(col, col, 0, p);
  const ci = law.linearize(new Array(8).fill(0), intact, 0, p);
  assert.ok(broken.slices.every((s) => s.cracks && s.tens === 0));
  // Pull a broken cold pair apart, then push it back past touching.
  let st = new Array(8).fill(0);
  const pull = law.maxwellUpdate(st, broken, 1e-3, p);
  assert.ok(Math.abs(pull.S) < 0.01 * E * 1e-3 * broken.overlap, `separated pair pulls with ${pull.S}`);
  st = pull.stored;
  // This step's law for the open pair: zero force exactly at touching, and no tension allowed.
  const lin = law.linearize(st, broken, 1e-3, p);
  assert.ok(Math.abs(lin.zeroStrain) < 1e-9 && lin.Stens === 0 && Math.abs(lin.K / ci.K - 1) < 1e-9);
  const back = law.maxwellUpdate(st, broken, -1e-3, p);
  assert.ok(Math.abs(back.S) < 0.01 * E * 1e-3 * broken.overlap, `at touching: ${back.S}`);
  const past = law.maxwellUpdate(back.stored, broken, -1e-4, p);
  assert.ok(past.actual[0] < -0.9 * E * 1e-4, `compression past touching: ${past.actual[0]}`);
  // Compressed far beyond the cap: the top slice sits at its cap, and unloads elastically from there.
  const crush = law.maxwellUpdate(new Array(8).fill(0), intact, -0.05, p);
  assert.ok(Math.abs(crush.actual[0] + intact.slices[0].comp) < 1e-9);
  const unload = law.maxwellUpdate(crush.stored, intact, 1e-5, p);
  assert.ok(Math.abs(unload.actual[0] - (-intact.slices[0].comp + E * 1e-5) * intact.slices[0].r) < 1e-6);
});

// --- Irreversible step (design 4.6, 4.7, phase D) ---------------------------
import * as flow from '../src/columns/flow.js';

test('irreversible: inelastic strain is zero below yield, the full increment at yield, and plastic work = cap x plastic strain', () => {
  const E0 = PARAMS.crust_youngs_modulus_gpa * 1e9, dtS = PARAMS.dt_crust_myr * profile.SECONDS_PER_MYR;
  const p = { ...uniform(1e6 * E0 * dtS), cohesion_mpa: PARAMS.cohesion_mpa }; // cold: every slice brittle and elastic
  const E = p.crust_youngs_modulus_gpa * 1000;
  const edge = law.edgeSlices(oldOcean(-3), oldOcean(-3), 1, p);
  const below = law.maxwellUpdate(new Array(8).fill(0), edge, -1e-4, p);
  assert.ok(Math.abs(below.inelastic) < 1e-9 && below.work === 0, `below yield: ${below.inelastic}, ${below.work}`);
  // Load every slice to its compression cap, then compress further.
  const atCap = edge.slices.map((s) => -s.comp);
  const more = law.maxwellUpdate(atCap, edge, -2e-4, p);
  assert.ok(Math.abs(more.inelastic / -2e-4 - 1) < 1e-6, `at yield: inelastic ${more.inelastic}`);
  const expectWork = edge.slices.reduce((a, s) => a + s.comp * 2e-4 * s.dz, 0);
  assert.ok(Math.abs(more.work / expectWork - 1) < 1e-4, `work ${more.work} vs ${expectWork}`); // r = 1 - 1e-6: a little creep is not plastic
  // A crack opening is recoverable: no inelastic strain, but work against cohesion.
  const open = law.maxwellUpdate(new Array(8).fill(0), edge, 1e-3, p);
  assert.ok(Math.abs(open.inelastic) < 1e-9);
  assert.ok(Math.abs(open.work / (p.cohesion_mpa * 1e-3 * edge.overlap) - 1) < 1e-6);
  void E;
});

test('irreversible: pure shear of a hexagon of edges leaves the area unchanged; uniform compression shrinks it at twice the strain', () => {
  // A column's 6 edges at 0, 60, ..., 300 degrees under the strain field
  // diag(+e, -e): edge strain e cos 2 theta. With compression elastic and
  // tension cracking, no edge has inelastic strain. Then all edges at the
  // compression cap under uniform shortening.
  const E0 = PARAMS.crust_youngs_modulus_gpa * 1e9, dtS = PARAMS.dt_crust_myr * profile.SECONDS_PER_MYR;
  const p = { ...uniform(1e6 * E0 * dtS), cohesion_mpa: PARAMS.cohesion_mpa };
  const edge = law.edgeSlices(oldOcean(-3), oldOcean(-3), 1, p);
  const e = 1e-3;
  const shear = [0, 60, 120, 180, 240, 300].map((deg) => law.maxwellUpdate(new Array(8).fill(0), edge, e * Math.cos(2 * deg * Math.PI / 180), p));
  const col = { area: 1e-3 };
  const sheared = flow.columnFlow(col, shear.map((u) => u.inelastic));
  assert.ok(Math.abs(sheared.area / col.area - 1) < 1e-9, `pure shear changed the area by ${sheared.area / col.area - 1}`);
  assert.ok(shear.some((u) => u.work > 0), 'the stretched edges should crack (work against cohesion)');
  const atCap = edge.slices.map((s) => -s.comp);
  const squeezed = flow.columnFlow(col, new Array(6).fill(law.maxwellUpdate(atCap, edge, -e, p).inelastic));
  assert.ok(Math.abs(Math.log(squeezed.area / col.area) - -2 * e) < 1e-9);
});

test('irreversible: integrity falls by work / W_D and heals exponentially with an Arrhenius time', () => {
  const p = PARAMS;
  // Cold: no healing; damage is exactly proportional to work.
  assert.ok(Math.abs(law.integrityStep(0.8, 10, 400, p) - (0.8 - 10 / p.damage_work_mpa_km)) < 1e-12);
  assert.equal(law.integrityStep(0.1, 1e3, 400, p), 0);
  // Hot: 1 - phi decays as exp(-dt / tau(T)), tau = heal_time at mantle temperature.
  let phi = 0;
  for (let n = 0; n < 10; n++) phi = law.integrityStep(phi, 0, p.mantle_temp, p);
  assert.ok(Math.abs(1 - phi - Math.exp(-10 * p.dt_crust_myr / p.heal_time_myr)) < 1e-12);
  const ratio = law.healTimeMyr(1200, p) / law.healTimeMyr(1400, p);
  assert.ok(Math.abs(Math.log(ratio) - (p.heal_activation_kj / p.gas_constant) * (1 / 1200 - 1 / 1400)) < 1e-9);
  console.log(`  healing time: ${law.healTimeMyr(1600, p)} Myr at 1600 K, ${law.healTimeMyr(1200, p).toFixed(0)} Myr at 1200 K, ${law.healTimeMyr(900, p).toExponential(1)} Myr at 900 K`);
});

test('irreversible: the least-squares gradient is exact for linear fields; the GPE push runs down it', () => {
  const xi = [0.2, -0.1, Math.sqrt(1 - 0.05)];
  const [e1, e2] = flow.tangentBasis(xi);
  const g = [3, -5]; // per unit length in the tangent basis
  const lin = (x) => 7 + [0, 1, 2].reduce((a, c) => a + (g[0] * e1[c] + g[1] * e2[c]) * (x[c] - xi[c]), 0);
  const rng = mulberry32(3);
  const nbrs = Array.from({ length: 5 }, () => {
    const t = [0.03 * (rng() - 0.5), 0.03 * (rng() - 0.5)];
    const x = [0, 1, 2].map((c) => xi[c] + t[0] * e1[c] + t[1] * e2[c]);
    return { x, f: lin(x) };
  });
  const grad = flow.gradient(xi, 7, nbrs);
  const want = [0, 1, 2].map((c) => g[0] * e1[c] + g[1] * e2[c]);
  assert.ok(Math.hypot(...grad.map((v, c) => v - want[c])) < 1e-9 * Math.hypot(...want));
  assert.equal(flow.gradient(xi, 7, nbrs.slice(0, 1)), null);
  const dx = flow.gpeDisplacement(grad, 100, { ...PARAMS, gpe_force_factor: 1 });
  assert.ok(dx.every((v, c) => Math.sign(v) === -Math.sign(grad[c]) || grad[c] === 0), 'push should run down the gradient');
});

// --- area, volume, melt and splits (design 4.8, phase E): CPU reference ----
import * as area from '../src/columns/area.js';
import * as volume from '../src/columns/volume.js';

/** Icosphere vertices, jittered by `jitter` x spacing, with apothems varied by +-`spread`. */
function sphereColumns(subdivisions, { jitter = 0, spread = 0, seed = 1 } = {}) {
  const v = createIcosphere(subdivisions).vertices;
  const rng = mulberry32(seed);
  const spacing = Math.sqrt((4 * Math.PI) / v.length / (Math.sqrt(3) / 2));
  const positions = new Float64Array(3 * v.length);
  v.forEach((x, i) => {
    const y = x.map((c) => c + jitter * spacing * (rng() - 0.5));
    const l = Math.hypot(...y);
    positions.set(y.map((c) => c / l), 3 * i);
  });
  const apothems = Array.from(v, () => (spacing / 2) * (1 + spread * (rng() - 0.5)));
  return { positions, apothems, spacing };
}

test('tessellation: weighted cells tile the sphere (sum 4 pi) on lattice and jittered, weighted configurations', () => {
  for (const [name, cfg] of [
    ['icosphere lattice, equal weights', sphereColumns(3)],
    ['jittered, apothems +-20%', sphereColumns(3, { jitter: 0.3, spread: 0.4, seed: 2 })],
    ['jittered, apothems +-40%', sphereColumns(2, { jitter: 0.4, spread: 0.8, seed: 3 })],
  ]) {
    const cells = area.tessellate(cfg.positions, cfg.apothems, 1.2 * cfg.spacing);
    const total = cells.reduce((s, c) => s + c.area, 0);
    const clipped = cells.filter((c) => c.clipped).length;
    console.log(`  ${name}: ${cells.length} cells, sum - 4 pi = ${(total - 4 * Math.PI).toExponential(1)}, ${clipped} clipped`);
    assert.equal(clipped, 0);
    assert.ok(Math.abs(total / (4 * Math.PI) - 1) < 1e-12, `sum ${total}`);
    assert.ok(cells.every((c) => c.area > 0));
  }
});

test('tessellation: the boundary between touching columns passes through their contact point', () => {
  const r1 = 0.02, r2 = 0.035, D = r1 + r2; // touching: centre distance = apothem sum (as angles)
  const p1 = [0, 0, 1], p2 = [Math.sin(D), 0, Math.cos(D)];
  const [e1, e2] = flow.tangentBasis(p1);
  const { c, n } = area.halfPlane(p2.map((v, k) => v - p1[k]), area.excessWeight(r1), area.excessWeight(r2), e1, e2);
  // Gnomonic distance of the boundary along the pair's direction = tan(r1).
  assert.ok(Math.abs(-c / Math.hypot(...n) / Math.tan(r1) - 1) < 1e-9); // (rounding of p2 - p1)
});

test('tessellation: cell shape gives a rectangle its long axis; a split puts the halves at the half-cells', () => {
  // Rectangular lattice, spacing 2s along x and s along y, near the north pole:
  // the centre column's cell is a 2s x s rectangle.
  const s = 0.01;
  const pts = [];
  for (let i = -3; i <= 3; i++) for (let j = -3; j <= 3; j++) {
    const x = 2 * s * i, y = s * j, l = Math.hypot(x, y, 1);
    pts.push([x / l, y / l, 1 / l]);
  }
  const positions = Float64Array.from(pts.flat());
  const centre = pts.findIndex((q) => q[0] === 0 && q[1] === 0);
  const cell = area.cellOf(centre, (k) => pts[k], () => 0.004, pts.keys(), 0.05);
  assert.ok(!cell.clipped);
  const [e1] = cell.basis;
  const axisAlongX = Math.abs(Math.cos(cell.axis) * e1[0] + Math.sin(cell.axis) * cell.basis[1][0]);
  assert.ok(Math.abs(axisAlongX - 1) < 1e-6, `axis ${cell.axis}`);
  assert.ok(Math.abs(cell.spread - (2 * s) / Math.sqrt(12)) < 1e-3 * s);
  const sp = volume.splitColumn(pts[centre], cell, 1, 2);
  // Halves at x = -+ s/2 (the half-rectangles' centroids), to SPLIT_OFFSET's accuracy.
  assert.ok(Math.abs(Math.abs(sp.parent[0]) - s / 2) < 0.01 * s && Math.abs(sp.child[0] + sp.parent[0]) < 1e-12);
  assert.equal(sp.volume + sp.volume, 1);
  assert.equal(sp.areaRef + sp.areaRef, 2);
  void positions;
});

test('melt: a stretched column makes 7 km of crust per unit new area at L = 0 and none at L >= H_m', () => {
  const p = PARAMS;
  // A ridge column (thermal age 0, L = 0), stretched in 10 steps from area 1e-3 to 2e-3.
  let c = { volume: 7e-3, area: 1e-3, areaPrev: 1e-3, thermalAgeMyr: 0, compositionDensity: 3000, ageMyr: 0 };
  let added = 0;
  for (let s = 1; s <= 10; s++) {
    const r = volume.volumeStep({ ...c, areaPrev: c.area, area: 1e-3 * (1 + s / 10) }, p);
    added += r.added;
    c = { ...c, ...r, area: 1e-3 * (1 + s / 10) };
  }
  assert.ok(Math.abs(added / 1e-3 - p.melt_fraction * p.melt_depth_km) < 1e-9, `${added / 1e-3} km per unit new area`);
  assert.ok(Math.abs(c.thicknessKm - 7) < 1e-9, 'steady ridge crust stays 7 km');
  // Old ocean (L = 100 km) and a continent (L = 150 km): stretching by 10% only thins.
  for (const [rho, t] of [[3000, 100], [2700, 1000]]) {
    const cap = volume.thermalAgeCapMyr(rho, p);
    const r = volume.volumeStep({ volume: 7e-3, area: 1.1e-3, areaPrev: 1e-3, thermalAgeMyr: Math.min(t, cap), compositionDensity: rho, ageMyr: t }, p);
    assert.equal(r.added, 0);
    const L0 = rho < 2850 ? p.continental_lithosphere_km : p.oceanic_lithosphere_max_km;
    assert.ok(Math.abs(r.lithosphereKm - L0 / 1.1) < 1e-9, `lithosphere ${r.lithosphereKm} vs ${L0 / 1.1}`);
    assert.ok(Math.abs(r.thicknessKm - 7 / 1.1) < 1e-12);
  }
  // Partial: halfway between, L = H_m / 2 makes half the melt.
  const tHalf = ((p.melt_depth_km / 2 * 1e3 / 2.32) ** 2 / p.thermal_diffusivity_m2_s) / profile.SECONDS_PER_MYR;
  const half = volume.volumeStep({ volume: 7e-3, area: 1.0001e-3, areaPrev: 1e-3, thermalAgeMyr: tHalf * 1.0001 ** 2, compositionDensity: 3000, ageMyr: 10 }, p);
  assert.ok(Math.abs(half.added / 1e-7 - 3.5) < 1e-6, `${half.added / 1e-7}`);
});

test('melt: a stretched continent thins until it melts; mixing in melt makes it oceanic (breakup)', () => {
  const p = PARAMS;
  let c = { volume: 35e-3, area: 1e-3, thermalAgeMyr: volume.thermalAgeCapMyr(2700, p), compositionDensity: 2700, ageMyr: 1000 };
  let firstMelt = null, breakup = null;
  for (let s = 1; s <= 400 && !breakup; s++) {
    const A = c.area * 1.01;
    const r = volume.volumeStep({ ...c, areaPrev: c.area, area: A }, p);
    if (r.added > 0 && firstMelt === null) firstMelt = { beta: A / 1e-3, L: r.lithosphereKm };
    if (r.breakup) breakup = { beta: A / 1e-3, h: r.thicknessKm, age: r.ageMyr };
    c = { ...c, ...r, area: A };
  }
  console.log(`  continent: melting starts at stretching factor ${firstMelt.beta.toFixed(2)} (L ${firstMelt.L.toFixed(1)} km); breakup at ${breakup.beta.toFixed(2)}, crust ${breakup.h.toFixed(1)} km`);
  assert.ok(Math.abs(firstMelt.beta - p.continental_lithosphere_km / p.melt_depth_km) < 0.02);
  assert.ok(breakup && breakup.age === 0 && c.compositionDensity >= 2850);
});

// --- subduction, suturing, slabs, bending (design 4.3, 6; phase F): CPU reference ---
import * as transitions from '../src/edges/transitions.js';
import * as flexure from '../src/columns/flexure.js';

test('transitions: the timer runs only in sustained compression; dense columns subduct, buoyant pairs weld', () => {
  const p = { ...PARAMS };
  let t = 0;
  for (let s = 0; s < 70; s++) t = transitions.timerStep(t, -2 * p.transition_stress_mpa, p);
  assert.ok(Math.abs(t - 70 * p.dt_crust_myr) < 1e-12);
  assert.equal(transitions.timerStep(t, -0.5 * p.transition_stress_mpa, p), 0);
  const col = (index, ageMyr, rho = 3000, slabNeighbours = 0) => ({
    index, slabNeighbours,
    freeboardKm: profile.columnProfile({ crustKm: rho < 2850 ? 35 : 7, compositionDensity: rho, ageMyr, elevationKm: -3 }, p).freeboardKm,
  });
  const ripe = { timer: p.subduction_window_myr, work: 0 };
  const early = { timer: 0.9 * p.subduction_window_myr, work: 0 };
  // Old ocean against a continent: the ocean goes down, only once the window is past.
  assert.equal(transitions.transition(ripe, col(0, 100), col(1, 1000, 2700), p).subduct, 0);
  assert.equal(transitions.transition(early, col(0, 100), col(1, 1000, 2700), p).subduct, null);
  // Old against younger ocean: the older (denser) goes down.
  assert.equal(transitions.transition(ripe, col(0, 20), col(1, 60), p).subduct, 1);
  // Polarity persists: the column already touching a slab goes down, on a tenth of the window.
  const fast = { timer: p.subduction_window_myr * p.subduction_propagation_factor, work: 0 };
  assert.equal(transitions.transition(fast, col(0, 20, 3000, 1), col(1, 60), p).subduct, 0);
  // Two buoyant columns never subduct, however long; they weld once quiet past the suture window.
  for (const [a, b] of [[col(0, 1000, 2700), col(1, 1000, 2700)], [col(0, 2), col(1, 1000, 2700)]]) {
    const long = { timer: 100, work: 0 };
    assert.equal(transitions.transition(long, a, b, p).subduct, null);
    assert.equal(transitions.transition(long, a, b, p).weld, true);
    assert.equal(transitions.transition({ timer: 100, work: 1 }, a, b, p).weld, false);
    assert.equal(transitions.transition({ timer: 0.9 * p.suture_window_myr, work: 0 }, a, b, p).weld, false);
  }
  const fb = (age) => col(0, age).freeboardKm;
  console.log(`  freeboard: ${fb(2).toFixed(2)} km at 2 Myr, ${fb(10).toFixed(2)} at 10, ${fb(60).toFixed(2)} at 60; continent ${col(0, 1000, 2700).freeboardKm.toFixed(2)}`);
});

test('slabs: sinking rate from negative buoyancy (Stokes-like), zero when buoyant', () => {
  const p = PARAMS;
  const w = transitions.slabSinkRate(-1.27, 0.0175, p);
  const expected = (p.mantle_density * 1.27e3 * 9.81 * 2 * 0.0175 * 6.371e6) / (4 * p.isostatic_viscosity_pa_s) * 3.15576e13 / 1e3;
  assert.ok(Math.abs(w / expected - 1) < 1e-12);
  console.log(`  old ocean slab (freeboard -1.27 km) sinks at ${w.toFixed(0)} km/Myr = ${(w / 10).toFixed(1)} cm/yr`);
  assert.equal(transitions.slabSinkRate(0.3, 0.0175, p), 0);
});

test('bending: a line load on a strip of columns gives the thin-plate flexure profile', () => {
  // A row of 55 columns along the equator, spacing 0.005 (32 km), cells from
  // the tessellation (clip radius 5 spacings: near-rectangular cells). The
  // centre column's Airy height is raised by 1 km: a line load K delta A / b.
  const l = 0.005, half = 27, n = 2 * half + 1, R = flexure.EARTH_RADIUS_M;
  const pos = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) { const t = (i - half) * l; pos.set([Math.sin(t), 0, Math.cos(t)], 3 * i); }
  const cells = area.tessellate(pos, new Array(n).fill(l / 2), 5 * l);
  const P = (k) => [pos[3 * k], pos[3 * k + 1], pos[3 * k + 2]];
  const K = (PARAMS.mantle_density - PARAMS.water_density) * 9.81;
  const D = flexure.rigidity(50, PARAMS); // 50 km elastic plate
  const cols = cells.map((c, i) => ({ A: c.area * R * R, K, r: 0, w0: 0, weq: i === half ? 1 : 0, D, nbrs: flexure.cellWeights(i, c, P) }));
  const w = flexure.solveExact(cols);
  const alpha = flexure.flexuralLength(D, K);
  const boundary = cols[half].nbrs[0].c * l * R; // shared boundary length, m
  const peak = flexure.lineLoadPeak((K * 1e3 * cols[half].A) / boundary, D, alpha) / 1e3;
  let worst = 0, zero = null, bulge = 0;
  for (let i = half; i < n; i++) {
    const x = (i - half) * l * R;
    worst = Math.max(worst, Math.abs(w[i] - peak * flexure.lineLoadShape(x, alpha)) / peak);
    if (zero === null && w[i] < 0) zero = x;
    bulge = Math.min(bulge, w[i]);
  }
  console.log(`  alpha ${(alpha / 1e3).toFixed(0)} km (${(alpha / (l * R)).toFixed(1)} spacings): peak ${w[half].toFixed(4)} vs ${peak.toFixed(4)} km, first zero ${(zero / alpha).toFixed(2)} alpha (3 pi / 4 = 2.36), forebulge ${(-bulge / w[half]).toFixed(3)} of the peak (exp(-pi) = 0.043); worst ${(100 * worst).toFixed(1)}% of the peak`);
  assert.ok(worst < 0.1);
  // The GPU's damped Jacobi converges to the same answer.
  const it = flexure.jacobi(cols, 3000, PARAMS.flexure_damping);
  assert.ok(Math.max(...w.map((v, i) => Math.abs(it[i] - v))) < 1e-3 * w[half]);
  // D = 0: one sweep is exactly the old per-column relaxation (w0 + k weq) / (1 + k), k = 1 / r.
  const one = flexure.jacobi(cols.map((c) => ({ ...c, D: 0, r: 2, w0: 0.5 })), 1, 2);
  assert.ok(Math.abs(one[half] - (0.5 + 0.5 * 1) / 1.5) < 1e-12 && Math.abs(one[0] - 0.5 / 1.5) < 1e-12);
});

// --- Orientation, shape, tangential channel, rolling (design 4.2, 4.5; phase G) ---

test('lattice elasticity: normal and tangential edges with free column rotations give nu = (kn - kt) / (3 kn + kt), isotropically; rigid rotation is free', () => {
  // A triangular lattice (spacing L, one column per cell), homogeneous
  // displacement gradient grad u = e + w (rotation w), every column spinning
  // by theta. Per edge: normal stretch n.e.n L; sliding (t.grad u.n - theta) L.
  // Energy per cell over cell area sqrt(3)/2 L^2, minimised over theta.
  const L = 1, kn = 1;
  const energy = (kt, e, w, phase, theta = null) => {
    const dirs = [0, 1, 2].map((k) => phase + (k * Math.PI) / 3);
    const sliding = (a) => { const n = [Math.cos(a), Math.sin(a)], t = [-n[1], n[0]]; return t[0] * (e[0][0] * n[0] + e[0][1] * n[1]) + t[1] * (e[1][0] * n[0] + e[1][1] * n[1]) + w; };
    // Optimal spin: the mean sliding (each column has the same spin).
    const th = theta ?? dirs.reduce((s, a) => s + sliding(a), 0) / 3;
    let W = 0;
    for (const a of dirs) {
      const n = [Math.cos(a), Math.sin(a)];
      const en = n[0] * (e[0][0] * n[0] + e[0][1] * n[1]) + n[1] * (e[1][0] * n[0] + e[1][1] * n[1]);
      W += 0.5 * L * L * (kn * en * en + kt * (sliding(a) - th) ** 2);
    }
    return { W: W / ((Math.sqrt(3) / 2) * L * L), theta: th };
  };
  for (const kt of [0, 0.2, 1]) {
    const m = shape.latticeModuli(kn, kt);
    for (const phase of [0, 0.3, 1.1]) {
      for (const e of [[[1e-3, 0], [0, 0]], [[2e-3, -1e-3], [-1e-3, 5e-4]], [[0, 1e-3], [1e-3, 0]]]) {
        const tr = e[0][0] + e[1][1], dxx = e[0][0] - tr / 2, dxy = e[0][1];
        const want = 0.5 * m.bulk * tr * tr + m.shear * 2 * (dxx * dxx + dxy * dxy);
        const got = energy(kt, e, 0.003, phase).W;
        assert.ok(Math.abs(got / want - 1) < 1e-9, `kt ${kt} phase ${phase}: ${got} vs ${want}`);
      }
    }
    assert.ok(Math.abs(m.poisson - (kn - kt) / (3 * kn + kt)) < 1e-12);
    assert.ok(Math.abs(m.shear - m.youngs / (2 * (1 + m.poisson))) < 1e-12, 'shear modulus = E / 2(1 + nu)');
  }
  assert.ok(Math.abs(shape.latticeModuli(1, 0.2).poisson - 0.25) < 1e-12);
  // Rigid rotation: free when the columns turn (optimal spin = w); with spins
  // held at 0 it would cost 3 (kt / 2) L^2 w^2 per cell (the phase-2 bug).
  const free = energy(0.2, [[0, 0], [0, 0]], 0.01, 0.2);
  const held = energy(0.2, [[0, 0], [0, 0]], 0.01, 0.2, 0);
  assert.ok(free.W < 1e-20 && Math.abs(free.theta - 0.01) < 1e-15 && held.W > 1e-6);
  // The normal stiffness that keeps E h for any ratio.
  for (const r of [0, 0.2, 1]) assert.ok(Math.abs(shape.latticeModuli(shape.normalStiffnessPerModulus(r), r * shape.normalStiffnessPerModulus(r)).youngs - 1) < 1e-12);
});

test('reference shape: radii are exact for an affinely stretched lattice; elongation is fitted, capped and split into touching halves', () => {
  const a = 0.0175;
  // A hexagon's six neighbours stretched by U = exp(L), long axis at 0.4 rad from b.
  const x = [0, 0, 1], b = [1, 0, 0];
  const s = 0.3, ang = 0.4, p = s * Math.cos(2 * ang), q = s * Math.sin(2 * ang);
  assert.ok(Math.abs(shape.shapeAxes(p, q).aspect - Math.exp(0.6)) < 1e-12 && Math.abs(shape.shapeAxes(p, q).axis - ang) < 1e-12);
  const U = (v) => { const c = Math.cosh(s), f = Math.sinh(s) / s; return [c * v[0] + f * (p * v[0] + q * v[1]), c * v[1] + f * (q * v[0] - p * v[1])]; };
  for (let k = 0; k < 6; k++) {
    const d = U([2 * a * Math.cos((k * Math.PI) / 3), 2 * a * Math.sin((k * Math.PI) / 3)]);
    const r = shape.radiusAlong(a, p, q, x, b, [d[0], d[1], 0]);
    assert.ok(Math.abs((2 * r) / Math.hypot(...d) - 1) < 1e-12, `rest ${2 * r} vs distance ${Math.hypot(...d)}`);
  }
  assert.equal(shape.radiusAlong(a, 0, 0, x, b, [0.3, 0.1, 0]), a);
  // Fit: a uniform strain tensor from hexagon edges (normal and tangential), and from three irregular ones.
  const e = { exx: 2e-3, eyy: -1e-3, exy: 7e-4 };
  // The neighbourhood also turns by w relative to the column (its spin lags): fitted, not read as shear.
  const w = 3e-3;
  const row = (phi) => { const nx = Math.cos(phi), ny = Math.sin(phi); return { nx, ny, en: e.exx * nx * nx + e.eyy * ny * ny + 2 * e.exy * nx * ny, g: -ny * (e.exx * nx + e.exy * ny) + nx * (e.exy * nx + e.eyy * ny) + w }; };
  for (const dirs of [[0, 1, 2, 3, 4, 5].map((k) => (k * Math.PI) / 3), [0.1, 1.9, 4.0], [0.2, 0.5, 0.9]]) {
    const f = shape.fitStrain(dirs.map(row));
    for (const k of ['exx', 'eyy', 'exy', 'w']) assert.ok(Math.abs(f[k] - { ...e, w }[k]) < 1e-12, `${dirs.length} edges: ${k} ${f[k]}`);
  }
  assert.equal(shape.fitStrain([row(0.3)]), null);
  // Deviatoric part only, capped at the aspect ratio limit.
  const u = shape.updateShape(0, 0, e, 1, 3);
  assert.ok(Math.abs(u.p - 1.5e-3) < 1e-15 && Math.abs(u.q - 7e-4) < 1e-15);
  // The shape turns with the material: by w relative to the body frame (2w in (p, q)).
  const turned = shape.updateShape(0.2, 0, { exx: 0, eyy: 0, exy: 0, w: Math.PI / 4 }, 1, 3);
  assert.ok(Math.abs(turned.p) < 1e-12 && Math.abs(turned.q - 0.2) < 1e-12 && Math.abs(shape.shapeAxes(turned.p, turned.q).axis - Math.PI / 4) < 1e-12);
  // Volumetric flow takes the fitted trace; without a fit, 2 x mean only with >= 3 edges.
  const col = { area: 1 };
  assert.ok(Math.abs(flow.columnFlow(col, [0.1, 0.1], 1, { exx: 0.01, eyy: 0.02, exy: 0 }).area - Math.exp(0.03)) < 1e-15);
  assert.equal(flow.columnFlow(col, [0.1, 0.1], 1, null).area, 1);
  assert.ok(Math.abs(flow.columnFlow(col, [0.1, 0.1, 0.1], 1, null).area - Math.exp(0.2)) < 1e-15);
  const big = shape.updateShape(0, 0, { exx: 2, eyy: 0, exy: 0 }, 1, 3);
  assert.ok(Math.abs(shape.shapeAxes(big.p, big.q).aspect - 3) < 1e-12);
  // Split at aspect 2: round halves, at their contact distance.
  const s2 = 0.5 * Math.log(2);
  const h = shape.splitShape(a, s2, 0);
  assert.ok(Math.abs(h.p) < 1e-15 && Math.abs(h.q) < 1e-15 && Math.abs(2 * h.offset - 2 * a / Math.SQRT2) < 1e-15);
  // At aspect 3 the halves keep 3/2 along the same axis; below 2 the axes swap.
  const h3 = shape.splitShape(a, 0.5 * Math.log(3), 0);
  assert.ok(Math.abs(shape.shapeAxes(h3.p, h3.q).aspect - 1.5) < 1e-12 && h3.p > 0);
  const h15 = shape.splitShape(a, 0.5 * Math.log(1.5), 0);
  assert.ok(Math.abs(shape.shapeAxes(h15.p, h15.q).aspect - 4 / 3) < 1e-12 && h15.p < 0);
  // Body axis: carried along a meridian it stays the meridian's tangent; spin turns it.
  const bb = shape.carryBodyAxis([1, 0, 0], [Math.sin(0.1), 0, Math.cos(0.1)], Math.PI / 2);
  assert.ok(Math.abs(bb[1] - 1) < 1e-12, `${bb}`);
});

test('tangential channel: rigid motions slide nothing; friction rises with compression across the face; brittle slip is a fault, ductile shear changes shape', () => {
  // Sliding of a pair: a rigid rotation of the sphere (about z, by w) with
  // each column spinning by its share of it (w z.x), and a translation, give
  // none (to the sphere's curvature over one spacing, (L/2)^2 / 2); holding
  // the spins at 0 under a rotation gives L0 psi.
  const a = [0, 0, 1], bx = 0.035, b = [Math.sin(bx), 0, Math.cos(bx)];
  const rot = (v, w) => [Math.cos(w) * v[0] - Math.sin(w) * v[1], Math.sin(w) * v[0] + Math.cos(w) * v[1], v[2]];
  const L0 = Math.hypot(b[0] - a[0], b[2] - a[2]);
  const w = 0.01;
  assert.ok(Math.abs(law.tangentialDisplacement(a, b, rot(a, w), rot(b, w), L0, 0.5, w * a[2], w * b[2])) < (bx / 2) ** 2 * L0 * w);
  assert.ok(Math.abs(law.tangentialDisplacement(a, b, rot(a, w), rot(b, w), L0, 0.3, 0, 0) / (L0 * w) - 1) < 1e-3);
  // Any rigid rotation of the sphere (axis k): spins w k.x, again none.
  for (const k of [[1, 0, 0], [0, 1, 0], [0.6, 0, 0.8]]) {
    const R = (v) => { // Rodrigues
      const c = Math.cos(w), s = Math.sin(w), kv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
      const kx = [k[1] * v[2] - k[2] * v[1], k[2] * v[0] - k[0] * v[2], k[0] * v[1] - k[1] * v[0]];
      return v.map((x, i) => x * c + kx[i] * s + k[i] * kv * (1 - c));
    };
    const spin = (v) => w * (k[0] * v[0] + k[1] * v[1] + k[2] * v[2]);
    assert.ok(Math.abs(law.tangentialDisplacement(a, b, R(a), R(b), L0, 0.5, spin(a), spin(b))) < (bx / 2) ** 2 * L0 * w, `axis ${k}`);
  }

  // Friction of a cold, broken (phi = 0) pair of old ocean: mu (1 - lambda) P per
  // slice; compression across the face (sigma_n < 0) raises it by mu |sigma_n|,
  // tension lowers it; an open crack is frictionless.
  const E0 = PARAMS.crust_youngs_modulus_gpa * 1e9, dtS = PARAMS.dt_crust_myr * profile.SECONDS_PER_MYR;
  const p = { ...uniform(1e6 * E0 * dtS), cohesion_mpa: PARAMS.cohesion_mpa };
  const col = oldOcean(-3);
  const zero = new Array(8).fill(0);
  const free = law.edgeSlices(col, col, 0, p, zero);
  const pressed = law.edgeSlices(col, col, 0, p, zero.map(() => -50));
  const open = law.edgeSlices(col, col, 0, p, zero.map(() => 1e4));
  free.slices.forEach((s, k) => {
    assert.ok(s.slips, 'cold slices slip by friction');
    assert.ok(Math.abs(pressed.slices[k].shear - s.shear - p.friction_coefficient * 50) < 1e-9);
    assert.equal(open.slices[k].shear, 0);
  });
  // Shear loading: elastic (G_t = ratio x E) up to the depth-integrated cap, then slip.
  const G = p.shear_stiffness_ratio * p.crust_youngs_modulus_gpa * 1000;
  let tau = zero, slip = 0, ductile = 0, Sprev = 0, work = 0;
  const dg = 5e-3; // 70 MPa per step; 60 steps load the deepest (strongest) slice past its cap
  for (let n = 0; n < 60; n++) {
    const u = law.maxwellShear(tau, free, dg, p);
    tau = u.stored; slip += u.slip; ductile += u.ductile; work += u.work;
    if (n === 0) assert.ok(Math.abs(u.S - G * dg * free.overlap) < 1e-5 * u.S, 'elastic slope (r = 1 - 1e-6)');
    Sprev = u.S;
  }
  const cap = free.slices.reduce((s, sl) => s + sl.shear * sl.dz, 0);
  assert.ok(Math.abs(Sprev - cap) < 1e-9 * cap, `plateau ${Sprev} vs cap ${cap}`);
  assert.ok(slip > 0.5 * 60 * dg && ductile < 1e-6 * slip && work > 0, `slip ${slip}, ductile ${ductile}`); // (below the cap a cold slice creeps by (1 - r) ~ 1e-6)
  // A hot (uniformly viscous) pair: the same loading is distributed shear, not slip.
  const ph = uniform(1e-2 * E0 * dtS);
  const hot = law.edgeSlices(col, col, 1, ph, zero);
  const uh = law.maxwellShear(zero, hot, dg, ph);
  assert.ok(uh.slip === 0 && uh.ductile > 0.9 * dg, `hot: slip ${uh.slip}, ductile ${uh.ductile}`);
  // No channel at ratio 0.
  assert.equal(law.linearizeShear(zero, free, 0.035, lawParams({ shear_stiffness_ratio: 0 })).stiffness, 0);
});

test('rolling: flat faces resist relative rotation (K face^2 / 12 rest) up to the kern rule', () => {
  const p = lawParams();
  const col = oldOcean(-3);
  const edge = law.edgeSlices(col, col, 1, p, new Array(8).fill(0));
  const face = 0.0175 * 2 / Math.sqrt(3), rest = 0.035;
  const K = edge.slices.reduce((s, sl) => s + p.crust_youngs_modulus_gpa * 1000 * sl.r * sl.dz, 0);
  const lin = law.linearizeRolling(0, edge, face, rest, -100, 50, p);
  assert.ok(Math.abs(lin.K - K * face * face / (12 * rest)) < 1e-9 * lin.K);
  assert.ok(Math.abs(lin.cap - (100 + 50) * face / 6) < 1e-12);
  assert.ok(Math.abs(law.rollingUpdate(lin, 1e-6) - lin.K * 1e-6) < 1e-12 * lin.K);
  assert.equal(law.rollingUpdate(lin, 1), lin.cap); // rocks open beyond the kern
  // No moment without a shared face, or with tension and no tension cap.
  assert.equal(law.linearizeRolling(0, edge, 0, rest, -100, 50, p).stiffness, 0);
  assert.equal(law.linearizeRolling(0, edge, face, rest, 100, 0, p).cap, 0);
});
