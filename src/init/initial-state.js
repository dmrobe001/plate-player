// Initial conditions: where the crust particles start, what they are made of,
// how they are bonded, and the initial mantle pattern.
//
// The returned state is plain typed arrays (structure of arrays); the
// Simulation packs it into GPU buffers.

import { mulberry32 } from './random.js';

export const INIT_DEFAULTS = {
  seed: 1,
  particleCount: 3000,
  apothem: 0.0175,
  yieldStressMPa: 150,
};

/** Phase 5 prototype initial state: a cap of continental crust around the north pole. */
export function generateInitialState(numMantleNodes, options = {}) {
  const opt = { ...INIT_DEFAULTS, ...options };
  const rng = mulberry32(opt.seed);
  const position = [];

  // Rejection sampling within 60 degrees of the pole, minimum spacing 0.04.
  for (let attempts = 0; position.length / 3 < opt.particleCount && attempts < 20000; attempts++) {
    const theta = rng() * Math.PI * 2;
    const phi = Math.acos(rng() * 0.5 + 0.5);
    const x = Math.sin(phi) * Math.cos(theta), y = Math.sin(phi) * Math.sin(theta), z = Math.cos(phi);
    let tooClose = false;
    for (let k = 0; k < position.length && !tooClose; k += 3) {
      tooClose = Math.hypot(x - position[k], y - position[k + 1], z - position[k + 2]) < 0.04;
    }
    if (!tooClose) position.push(x, y, z);
  }
  const n = position.length / 3;

  const particles = {
    count: n,
    position: Float32Array.from(position),
    thickness: new Float32Array(n).fill(35),
    compositionDensity: new Float32Array(n).fill(2700),
    temperature: new Float32Array(n).fill(293),
    age: new Float32Array(n).fill(100),
    apothem: new Float32Array(n).fill(opt.apothem),
  };

  // Greedy bonding: each particle bonds to its nearest neighbours within 0.08, max 6 bonds each.
  const bondCount = new Uint32Array(n);
  const a = [], b = [], rest = [];
  for (let i = 0; i < n; i++) {
    if (bondCount[i] >= 6) continue;
    const near = [];
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const d = Math.hypot(position[3 * i] - position[3 * j], position[3 * i + 1] - position[3 * j + 1], position[3 * i + 2] - position[3 * j + 2]);
      if (d < 0.08) near.push([j, d]);
    }
    near.sort((p, q) => p[1] - q[1]);
    for (const [j, d] of near) {
      if (bondCount[i] >= 6 || bondCount[j] >= 6) break;
      if (i < j) { a.push(i); b.push(j); rest.push(d); bondCount[i]++; bondCount[j]++; }
    }
  }

  const mantlePsi = new Float32Array(numMantleNodes);
  for (let i = 0; i < numMantleNodes; i++) mantlePsi[i] = (rng() - 0.5) * 0.1;

  return {
    particles,
    bonds: {
      count: a.length,
      a: Uint32Array.from(a),
      b: Uint32Array.from(b),
      restLength: Float32Array.from(rest),
      damage: new Float32Array(a.length),
    },
    mantlePsi,
    yieldStressMPa: opt.yieldStressMPa,
    stats: { particles: n, bonds: a.length },
  };
}
