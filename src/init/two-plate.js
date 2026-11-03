// A coarse, ordered test planet (design 9.2): two plates whose relative motion
// is a rotation about an Euler pole P on their shared boundary, the simplest
// plate system on a sphere. Half the boundary is a ridge, half a trench; the
// relative speed goes as sin(angle from P), so both taper to zero at P and its
// antipode and no transform faults are needed.
//
//   - Columns at the 162 vertices of a twice-subdivided icosahedron, bonded
//     along its edges (5 or 6 per column), apothems fitted so every bond
//     starts at its rest length: no overlaps, no gaps but the 12 pentagons'.
//   - Plate A (x . m > 0): oceanic, moving with the mantle beneath it
//     (src/mantle/kinematic.js) about P at omega; it is the subducting plate.
//     Its leading row on the trench half starts as slabs (sinking in place,
//     as subduction leaves them; their edges pull A toward the trench).
//     It carries a microcontinent of 7 columns that reaches the trench after
//     about arrivalMyr at the mantle's rate: a scheduled collision.
//   - Plate B (x . m < 0): at rest with its mantle (the reference frame), with
//     a continent along the trench half (an Andean margin, where arcs grow)
//     and ocean on its ridge half.
//   - Bonds across the boundary are broken (integrity 0); within a plate,
//     intact (a passive margin welds continent to ocean).
//   - Seafloor age = steady state of symmetric spreading at omega: the ridge
//     recedes from each plate at omega / 2, so age = 2 phi / omega, with phi
//     the angle around P from the ridge.
//
// Everything is deterministic (no seed). Parameters that are lengths on the
// unit sphere are scaled with the column size (`params`), so the same rules
// apply as at the default resolution.

import { createIcosphere } from '../geometry/icosphere.js';
import { fitApothems, CRUST } from './initial-state.js';
import { mulberry32, randomOnSphere } from './random.js';

export const TWO_PLATE_DEFAULTS = {
  subdivisions: 2,          // 162 columns
  speedCmYr: 8,             // mantle speed under A at 90 degrees from the pole (relative plate rate)
  transitionWidthRad: 0.05, // width of the mantle's step at the boundary (nearest columns are ~0.04 from it)
  arrivalMyr: 100,          // when the microcontinent's leading edge reaches the trench (at mantle speed)
  oceanAgeCapMyr: 180,      // oldest seafloor (thermal age is capped on the GPU anyway, ~60 Myr)
  continentThicknessKm: 38,
  continentHalfWidthRad: 0.95, // B's continent: within this angle (around P) of the trench
  continentLatitude: 0.7,      // ... and |x . P| below this (away from the pole)
  slabRowMinQ: 0.25,           // A's leading row becomes slabs where x . q < -this (away from P)
  fitTolerance: 1e-5,
  fitIterations: 2000,
};

const BASE_APOTHEM = 0.0175; // default resolution (sim param base_apothem)
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a) => { const l = Math.hypot(...a); return a.map((v) => v / l); };

/**
 * Boundary frame for a set of points: the great circle (normal m) farthest
 * from every point, then the pole P on it farthest from every point.
 * Deterministic (fixed search sequence).
 */
export function chooseFrame(points) {
  const rng = mulberry32(12345);
  let m = null, best = -1;
  for (let k = 0; k < 4000; k++) {
    const c = randomOnSphere(rng);
    let margin = Infinity;
    for (const x of points) margin = Math.min(margin, Math.abs(dot(x, c)));
    if (margin > best) { best = margin; m = c; }
  }
  const e = unit(cross(m, Math.abs(m[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
  const f = cross(m, e);
  let P = null, bestP = -2;
  for (let k = 0; k < 720; k++) {
    const t = (k * Math.PI) / 360;
    const c = e.map((v, i) => Math.cos(t) * v + Math.sin(t) * f[i]);
    let near = 2;
    for (const x of points) near = Math.min(near, 1 - Math.abs(dot(x, c)));
    if (near > bestP) { bestP = near; P = c; }
  }
  return { normal: m, pole: P, q: cross(m, P), margin: best };
}

export function generateTwoPlateState(options = {}) {
  const opt = { ...TWO_PLATE_DEFAULTS, ...options };
  const ico = createIcosphere(opt.subdivisions);
  const n = ico.vertices.length;
  const { normal: m, pole: P, q, margin } = chooseFrame(ico.vertices);
  const omega = (opt.speedCmYr * 10) / 6371; // rad/Myr

  // Bonds along icosphere edges.
  const a = [], b = [];
  ico.neighbours.forEach((list, i) => { for (const j of list) if (i < j) { a.push(i); b.push(j); } });
  const bonds = { count: a.length, a: Uint32Array.from(a), b: Uint32Array.from(b), damage: new Float32Array(a.length) };

  const position = Float32Array.from(ico.vertices.flat());
  const meanSpacing = a.reduce((s, i, k) => s + Math.hypot(...[0, 1, 2].map((c) => position[3 * i + c] - position[3 * b[k] + c])), 0) / a.length;
  const fit = fitApothems(position, meanSpacing / 2, bonds, opt.fitTolerance, opt.fitIterations);

  const x = (i) => [position[3 * i], position[3 * i + 1], position[3 * i + 2]];
  const plateA = Uint8Array.from({ length: n }, (_, i) => (dot(x(i), m) > 0 ? 1 : 0));
  const phi = Float64Array.from({ length: n }, (_, i) => Math.atan2(dot(x(i), m), dot(x(i), q)));

  // Broken bonds across the boundary.
  for (let k = 0; k < bonds.count; k++) if (plateA[a[k]] !== plateA[b[k]]) bonds.damage[k] = 1;

  // Crust: ocean everywhere, then the continents.
  const kind = new Array(n).fill('ocean');
  for (let i = 0; i < n; i++) {
    if (!plateA[i] && Math.abs(phi[i]) > Math.PI - opt.continentHalfWidthRad && Math.abs(dot(x(i), P)) < opt.continentLatitude) kind[i] = 'continent';
  }
  // Microcontinent on A: its leading edge (one spacing ahead of its centre)
  // reaches the trench (phi = pi) after arrivalMyr at omega.
  const alpha = Math.PI - meanSpacing - omega * opt.arrivalMyr;
  const target = q.map((v, c) => Math.cos(alpha) * v + Math.sin(alpha) * m[c]);
  let centre = 0;
  for (let i = 1; i < n; i++) if (dot(x(i), target) > dot(x(centre), target)) centre = i;
  for (const i of [centre, ...ico.neighbours[centre]]) if (plateA[i]) kind[i] = 'micro';

  // Trench: A's leading row on the trench half stays at the surface, each
  // column with a slab hanging from it (an extra particle, numbered after the
  // columns), at its front face toward B, bonded to it, as the trench strip
  // test seeds it. (The row itself was made slab: a whole 1,900 km row of
  // slab with no overriding partner, and the columns behind took their slab
  // as partner and went down at once, leaving chains hanging from free slabs
  // and nothing to pull or consume; design 9.2.)
  const touchesB = (i) => ico.neighbours[i].some((j) => !plateA[j]);
  const trench = [];
  for (let i = 0; i < n; i++) if (plateA[i] && kind[i] === 'ocean' && touchesB(i) && dot(x(i), q) < -opt.slabRowMinQ) trench.push(i);
  const total = n + trench.length;
  const slabOf = new Map(trench.map((i, k) => [i, n + k]));
  const pos = new Float32Array(3 * total);
  pos.set(position);
  const apothem = new Float32Array(total);
  apothem.set(fit.apothem);
  for (const [i, s] of slabOf) {
    // Toward its B neighbours, one apothem out: its front face.
    const d = [0, 0, 0];
    for (const j of ico.neighbours[i]) if (!plateA[j]) x(j).forEach((v, c) => { d[c] += v - x(i)[c]; });
    const t = d.map((v, c) => v - dot(d, x(i)) * x(i)[c]), tl = Math.hypot(...t);
    const y = x(i).map((v, c) => v + (fit.apothem[i] * t[c]) / tl), yl = Math.hypot(...y);
    for (let c = 0; c < 3; c++) pos[3 * s + c] = y[c] / yl;
    apothem[s] = fit.apothem[i];
  }
  const state = new Uint32Array(total).fill(1);
  for (const s of slabOf.values()) state[s] = 2;
  const bondsWithSlabs = {
    count: bonds.count + trench.length,
    a: Uint32Array.from([...bonds.a, ...trench]),
    b: Uint32Array.from([...bonds.b, ...trench.map((i) => slabOf.get(i))]),
    damage: Float32Array.from([...bonds.damage, ...trench.map(() => 0)]),
  };

  const particles = {
    count: total,
    position: pos,
    apothem,
    state,
    thickness: new Float32Array(total),
    compositionDensity: new Float32Array(total),
    temperature: new Float32Array(total),
    age: new Float32Array(total),
  };
  const counts = { A: 0, B: 0, continent: 0, micro: 0, slab: trench.length };
  for (let i = 0; i < n; i++) {
    const crust = kind[i] === 'ocean' ? CRUST.oceanic : CRUST.continental;
    particles.thickness[i] = kind[i] === 'ocean' ? crust.thickness[0] : opt.continentThicknessKm;
    particles.compositionDensity[i] = crust.density;
    particles.temperature[i] = crust.temperature;
    particles.age[i] = kind[i] === 'ocean' ? Math.min((2 * Math.abs(phi[i])) / omega, opt.oceanAgeCapMyr) : crust.age;
    counts[plateA[i] ? 'A' : 'B']++;
    if (kind[i] !== 'ocean') counts[kind[i]]++;
  }
  for (const [i, s] of slabOf) {
    particles.thickness[s] = particles.thickness[i];
    particles.compositionDensity[s] = particles.compositionDensity[i];
    particles.temperature[s] = particles.temperature[i];
    particles.age[s] = particles.age[i];
  }

  // Unit-sphere lengths scaled with the column size (defaults are for BASE_APOTHEM).
  let meanApothem = 0;
  for (let i = 0; i < n; i++) meanApothem += fit.apothem[i] / n; // (the columns)
  const scale = meanApothem / BASE_APOTHEM;
  const clip = 0.04 * scale;
  const params = {
    base_apothem: meanApothem,
    cell_clip_radius: clip,
    arc_reach: 0.05 * scale,
    // Hash cells at least twice the clip radius (tessellation reach, design 4.8).
    hash_grid_dim: Math.max(1, Math.floor(3 / (2 * clip))),
    // Time step (design 9.2, two-plate findings): at this column size a slab
    // lives ~1.3 Myr. The default solve (32 warm-started sweeps, budget-capped
    // Chebyshev rho; design 5.2) holds the velocity within ~6% of a converged
    // one; before the warm start this world needed 256.
    dt_crust_myr: 0.25,
  };

  return {
    particles,
    bonds: bondsWithSlabs,
    params,
    mantle: { pole: P, normal: m, omegaRadMyr: omega, widthRad: opt.transitionWidthRad },
    frame: { pole: P, normal: m, q, phi, plateA, kind, trench, slabOf },
    stats: {
      world: 'two-plate',
      particles: total,
      columns: n,
      bonds: bonds.count,
      bondsPerParticle: (2 * bonds.count) / n,
      continentalFraction: (counts.continent + counts.micro) / n,
      damagedBondFraction: bonds.damage.reduce((s, d) => s + (d > 0.5), 0) / bonds.count,
      plateA: counts.A, plateB: counts.B, continent: counts.continent, microcontinent: counts.micro, slabs: counts.slab,
      boundaryMarginRad: margin,
      maxBondStrain: fit.maxStrain,
      apothemRange: [fit.min / meanApothem, fit.max / meanApothem],
      coverage: fit.coverage,
      meanApothem,
      speedCmYr: opt.speedCmYr,
      collisionMyr: opt.arrivalMyr,
    },
  };
}
