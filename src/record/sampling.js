// Sampling particle quantities onto a fixed mesh (the record's Eulerian grid).
//
// Each mesh node takes a Gaussian-weighted average over nearby particles,
// weights w = area * exp(-d^2 / 2 sigma^2), cut off at 3 sigma. This is an
// intensive (point-value) estimate; it does not conserve integrals exactly.
// `support` = sum(w) / (2 pi sigma^2) is ~1 under continuous crust and falls
// toward 0 in gaps, so consumers can tell "thin crust" from "no crust".
// Categorical channels (ids) are not averaged: each node takes the value of
// its most heavily weighted particle.

import { CellList } from '../init/cell-list.js';

/**
 * @param nodes     Float32Array of mesh node positions (xyz triples, unit sphere)
 * @param particles {position: Float32Array xyz triples, area: Float32Array}
 * @param channels  {name: Float32Array per particle (NaN = exclude from that channel)}
 * @param sigma     kernel width (unit sphere)
 * @returns {support: Float32Array, [name]: Float32Array per node (NaN where no particle within cutoff)}
 */
export function sampleToMesh(nodes, particles, channels, sigma, nearestChannels = {}) {
  const nNodes = nodes.length / 3;
  const nParticles = particles.position.length / 3;
  const cutoff = 3 * sigma;

  // One cell list over particles and nodes together; pairs with exactly one
  // node and one particle are the ones we want.
  const all = new Float32Array(3 * (nParticles + nNodes));
  all.set(particles.position, 0);
  all.set(nodes, 3 * nParticles);
  const names = Object.keys(channels);
  const sum = Object.fromEntries(names.map((n) => [n, new Float64Array(nNodes)]));
  const wsum = Object.fromEntries(names.map((n) => [n, new Float64Array(nNodes)]));
  const support = new Float64Array(nNodes);
  const inv2s2 = 1 / (2 * sigma * sigma);
  const nearestNames = Object.keys(nearestChannels);
  const nearestW = new Float64Array(nNodes);
  const nearestP = new Int32Array(nNodes).fill(-1);

  new CellList(all, cutoff).forEachPair((i, j, dx, dy, dz, d) => {
    const iIsNode = i >= nParticles, jIsNode = j >= nParticles;
    if (iIsNode === jIsNode) return;
    const node = (iIsNode ? i : j) - nParticles;
    const p = iIsNode ? j : i;
    const w = particles.area[p] * Math.exp(-d * d * inv2s2);
    support[node] += w;
    if (w > nearestW[node]) { nearestW[node] = w; nearestP[node] = p; }
    for (const n of names) {
      const v = channels[n][p];
      if (Number.isNaN(v)) continue;
      sum[n][node] += w * v;
      wsum[n][node] += w;
    }
  });

  const out = { support: Float32Array.from(support, (s) => s / (2 * Math.PI * sigma * sigma)) };
  for (const n of names) out[n] = Float32Array.from(sum[n], (s, k) => (wsum[n][k] > 0 ? s / wsum[n][k] : NaN));
  // Categorical channels (ids) take the value of the most heavily weighted particle.
  for (const n of nearestNames) out[n] = Float32Array.from(nearestP, (p) => (p >= 0 ? nearestChannels[n][p] : NaN));
  return out;
}

/** Geographic convention for records: north pole = +y, longitude 0 toward +z, east toward +x. */
export function latLonDeg(x, y, z) {
  return [Math.asin(Math.max(-1, Math.min(1, y))) * 180 / Math.PI, Math.atan2(x, z) * 180 / Math.PI];
}

/** Unit east and north vectors at a point on the unit sphere (same convention). */
export function eastNorth(x, y, z) {
  let ex = z, ez = -x; // (0,1,0) x (x,y,z)
  const len = Math.hypot(ex, ez);
  if (len < 1e-9) return [[1, 0, 0], [0, 0, -1]]; // at a pole, pick any frame
  ex /= len; ez /= len;
  // north = position x east
  return [[ex, 0, ez], [y * ez, z * ex - x * ez, -y * ex]];
}
