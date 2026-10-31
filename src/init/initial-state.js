// Initial conditions: where the crust particles start, what they are made of,
// how they are bonded, and the initial mantle pattern.
//
// Recipe (every random choice flows from one seed):
//   1. Scatter N points uniformly on the sphere.
//   2. Relax them with pure pairwise repulsion (no bonds yet). N is chosen so
//      the hexagons would cover a fraction `coverage` < 1 of the sphere, which
//      leaves room to remove nearly every overlap; the few that remain are
//      removed by shrinking those particles' apothems to half the distance to
//      their nearest neighbour (typically by well under 1%). Otherwise bonds
//      (holding the current distance) and contacts (pushing to the apothem sum)
//      would fight from the first step.
//   3. Crust type: threshold a fractal-noise field so that a fraction
//      `continentalFraction` of particles is continental. Continental
//      thickness rises from coast to interior.
//   4. Bond each particle to its nearest neighbours (max 6), rest length =
//      current distance, so the starting lattice is stress-free.
//   5. Bond damage: a domain-warped Voronoi ("Worley") pattern of plate
//      centres. Bonds near Voronoi cell borders are damaged, which seeds a
//      network of weak plate boundaries with triple junctions.
//   6. Velocities zero; the mantle field starts as small random noise.
//
// The returned state is plain typed arrays (structure of arrays); the
// Simulation packs it into GPU buffers.

import { mulberry32, randomOnSphere } from './random.js';
import { createPerlin3, fbm, quantile, smoothstep } from './noise.js';
import { CellList } from './cell-list.js';

export const INIT_DEFAULTS = {
  seed: 1,
  apothem: 0.0175,            // unit-sphere length; matches sim param base_apothem
  coverage: 0.85,             // fraction of the sphere the hexagons would tile
  relaxIterations: 100,
  continentalFraction: 0.35,
  continentFrequency: 1.4,    // noise frequency on the unit sphere (higher = smaller continents)
  plateCount: 12,
  boundaryWarp: 0.35,         // how wiggly plate boundaries are
  boundaryWidth: [0.6, 1.6],  // full damage .. no damage, in particle spacings from a border
  bondCutoff: 1.3,            // bond partners within this many contact distances
  maxBondsPerParticle: 6,
  yieldStressMPa: 150,
  mantleNoise: 0.1,
};

export const CRUST = {
  continental: { density: 2700, thickness: [30, 45], temperature: 293, age: 1000 },
  oceanic: { density: 3000, thickness: [7, 7], temperature: 278, age: 60 },
};

export function generateInitialState(numMantleNodes, options = {}) {
  const opt = { ...INIT_DEFAULTS, ...options };
  const rng = mulberry32(opt.seed);
  const contact = 2 * opt.apothem;
  const hexArea = 2 * Math.sqrt(3) * opt.apothem ** 2;
  const n = Math.round((opt.coverage * 4 * Math.PI) / hexArea);

  // 1-2. Uniform scatter, then repulsive relaxation.
  const position = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) position.set(randomOnSphere(rng), 3 * i);
  const relaxResidual = relax(position, contact, opt.relaxIterations);
  const apothem = new Float32Array(n).fill(opt.apothem);
  new CellList(position, contact).forEachPair((i, j, dx, dy, dz, d) => {
    const fit = (d / 2) * (1 - 1e-5); // margin for float32 rounding on the GPU
    apothem[i] = Math.min(apothem[i], fit);
    apothem[j] = Math.min(apothem[j], fit);
  });
  let shrunk = 0, minApothem = opt.apothem;
  for (let i = 0; i < n; i++) {
    if (apothem[i] < opt.apothem) shrunk++;
    minApothem = Math.min(minApothem, apothem[i]);
  }

  // 3. Crust type from thresholded fractal noise.
  const continentNoise = createPerlin3(rng);
  const offset = [rng() * 100, rng() * 100, rng() * 100]; // decorrelate from the noise lattice origin
  const field = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const f = opt.continentFrequency;
    field[i] = fbm(continentNoise,
      position[3 * i] * f + offset[0], position[3 * i + 1] * f + offset[1], position[3 * i + 2] * f + offset[2]);
  }
  const coast = quantile(field, 1 - opt.continentalFraction);
  const interior = quantile(field, 0.99);

  const particles = {
    count: n,
    position,
    thickness: new Float32Array(n),
    compositionDensity: new Float32Array(n),
    temperature: new Float32Array(n),
    age: new Float32Array(n),
    apothem,
  };
  let continental = 0;
  for (let i = 0; i < n; i++) {
    const isContinent = field[i] >= coast;
    const crust = isContinent ? CRUST.continental : CRUST.oceanic;
    const [thin, thick] = crust.thickness;
    particles.thickness[i] = thin + (thick - thin) * smoothstep(coast, interior, field[i]);
    particles.compositionDensity[i] = crust.density;
    particles.temperature[i] = crust.temperature;
    particles.age[i] = crust.age;
    continental += isContinent;
  }

  // 4. Bonds to nearest neighbours.
  const bonds = buildBonds(position, opt.bondCutoff * contact, opt.maxBondsPerParticle);

  // 5. Damage along warped Voronoi borders, evaluated at bond midpoints.
  //    Any bond joining particles in different cells is fully damaged, so
  //    the borders really separate plates; the smooth band around them
  //    weakens their surroundings.
  const plateField = plateBorderField(rng, opt.plateCount, opt.boundaryWarp);
  const cell = new Uint16Array(n);
  for (let i = 0; i < n; i++) cell[i] = plateField(position[3 * i], position[3 * i + 1], position[3 * i + 2]).cell;
  const [w0, w1] = opt.boundaryWidth.map((w) => w * contact);
  let damaged = 0;
  for (let k = 0; k < bonds.count; k++) {
    const i = bonds.a[k], j = bonds.b[k];
    const { border } = plateField(
      position[3 * i] + position[3 * j], position[3 * i + 1] + position[3 * j + 1], position[3 * i + 2] + position[3 * j + 2]);
    bonds.damage[k] = cell[i] !== cell[j] ? 1 : 1 - smoothstep(w0, w1, border);
    damaged += bonds.damage[k] > 0.5;
  }

  // 6. Mantle: small random perturbation from which the pattern grows.
  const mantlePsi = new Float32Array(numMantleNodes);
  for (let i = 0; i < numMantleNodes; i++) mantlePsi[i] = (rng() - 0.5) * opt.mantleNoise;

  return {
    particles,
    bonds,
    mantlePsi,
    yieldStressMPa: opt.yieldStressMPa,
    stats: {
      particles: n,
      bonds: bonds.count,
      bondsPerParticle: (2 * bonds.count) / n,
      continentalFraction: continental / n,
      damagedBondFraction: damaged / bonds.count,
      relaxResidual,
      shrunkFraction: shrunk / n,
      minApothemRatio: minApothem / opt.apothem,
    },
  };
}

/**
 * Pure repulsion: each overlapping pair is pushed apart by half its overlap
 * each, applied immediately (Gauss-Seidel, which converges much faster than
 * averaging all corrections first); after each sweep points are projected
 * back onto the sphere. Returns the largest overlap found in the final sweep
 * as a fraction of the contact distance.
 */
function relax(position, contact, iterations) {
  const n = position.length / 3;
  const p = position;
  let worst = 0;
  for (let it = 0; it < iterations; it++) {
    worst = 0;
    // Pairs are found from positions at the start of the sweep; a cutoff
    // slightly above contact catches pairs that a correction pushes together.
    new CellList(p, 1.1 * contact).forEachPair((i, j) => {
      const dx = p[3 * i] - p[3 * j], dy = p[3 * i + 1] - p[3 * j + 1], dz = p[3 * i + 2] - p[3 * j + 2];
      const d = Math.hypot(dx, dy, dz);
      const overlap = contact - d;
      if (overlap <= 0) return;
      worst = Math.max(worst, overlap);
      const s = (0.5 * overlap) / Math.max(d, 1e-9);
      p[3 * i] += dx * s; p[3 * i + 1] += dy * s; p[3 * i + 2] += dz * s;
      p[3 * j] -= dx * s; p[3 * j + 1] -= dy * s; p[3 * j + 2] -= dz * s;
    });
    for (let i = 0; i < n; i++) {
      const len = Math.hypot(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
      p[3 * i] /= len; p[3 * i + 1] /= len; p[3 * i + 2] /= len;
    }
    if (worst < 1e-3 * contact) break;
  }
  return worst / contact;
}

/** Bond the closest pairs first, skipping any particle that already has maxBonds. */
function buildBonds(position, cutoff, maxBonds) {
  const pairs = [];
  new CellList(position, cutoff).forEachPair((i, j, dx, dy, dz, d) => pairs.push([i, j, d]));
  pairs.sort((p, q) => p[2] - q[2]);

  const n = position.length / 3;
  const count = new Uint8Array(n);
  const a = [], b = [], rest = [];
  for (const [i, j, d] of pairs) {
    if (count[i] >= maxBonds || count[j] >= maxBonds) continue;
    a.push(i); b.push(j); rest.push(d);
    count[i]++; count[j]++;
  }
  return {
    count: a.length,
    a: Uint32Array.from(a),
    b: Uint32Array.from(b),
    restLength: Float32Array.from(rest),
    damage: new Float32Array(a.length),
  };
}

/**
 * Returns f(x, y, z) -> {cell, border} for a point (any length, projected to
 * the sphere): the index of the nearest plate centre, and the difference
 * between the angular distances to the second-nearest and nearest centres
 * (0 exactly on a Voronoi border, growing away from it). The point is first
 * displaced by vector noise so borders are not geodesics.
 */
function plateBorderField(rng, plateCount, warp) {
  const centres = Array.from({ length: plateCount }, () => randomOnSphere(rng));
  const warpNoise = createPerlin3(rng);
  return function plateField(x, y, z) {
    let len = Math.hypot(x, y, z);
    x /= len; y /= len; z /= len;
    const wx = fbm(warpNoise, 2 * x + 11.3, 2 * y, 2 * z, 3);
    const wy = fbm(warpNoise, 2 * x, 2 * y + 23.7, 2 * z, 3);
    const wz = fbm(warpNoise, 2 * x, 2 * y, 2 * z + 37.1, 3);
    x += warp * wx; y += warp * wy; z += warp * wz;
    len = Math.hypot(x, y, z);
    x /= len; y /= len; z /= len;

    let best = -2, second = -2, cell = 0; // largest dot products = smallest angles
    centres.forEach((c, k) => {
      const d = x * c[0] + y * c[1] + z * c[2];
      if (d > best) { second = best; best = d; cell = k; } else if (d > second) second = d;
    });
    return { cell, border: Math.acos(Math.min(1, second)) - Math.acos(Math.min(1, best)) };
  };
}
