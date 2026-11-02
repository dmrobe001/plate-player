// No net rotation (design 5.3; GPU: crust_net_rotation_* in solver.wgsl).
//
// The net rotation of a set of columns is the rotation vector theta that best
// explains their displacements u as one rigid rotation, u ~ theta x x, in the
// area-weighted least-squares sense: it minimises sum a |u - theta x x|^2,
// which gives I theta = L with L = sum a x cross u and I = sum a (1 - x x^T)
// (for unit x; the moment of inertia of the shell). On a fully covered sphere
// I = (2/3) A 1, so theta is the mean of x cross u. Geodesy's no-net-rotation
// frame is the same construction, with plate velocities for u.

import { cross } from './vec.js';

/**
 * positions, displacements: arrays of [x, y, z] (unit positions); weights:
 * areas. Returns theta (same units as u per unit radius), or [0, 0, 0] when
 * the columns cannot define a rotation (I nearly singular).
 */
export function netRotation(positions, displacements, weights) {
  const L = [0, 0, 0];
  const I = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  positions.forEach((x, i) => {
    const a = weights[i];
    const l = cross(x, displacements[i]);
    for (let p = 0; p < 3; p++) {
      L[p] += a * l[p];
      for (let q = 0; q < 3; q++) I[3 * p + q] += a * ((p === q ? 1 : 0) - x[p] * x[q]);
    }
  });
  const r0 = I.slice(0, 3), r1 = I.slice(3, 6), r2 = I.slice(6, 9);
  const det = r0[0] * (r1[1] * r2[2] - r1[2] * r2[1]) - r0[1] * (r1[0] * r2[2] - r1[2] * r2[0]) + r0[2] * (r1[0] * r2[1] - r1[1] * r2[0]);
  const t = (I[0] + I[4] + I[8]) / 3;
  if (!(t > 0) || !(det > 1e-3 * t * t * t)) return [0, 0, 0];
  // Inverse by cofactors (I is symmetric): theta = (c0 L0 + c1 L1 + c2 L2) / det.
  const c = [cross(r1, r2), cross(r2, r0), cross(r0, r1)];
  return [0, 1, 2].map((k) => (c[0][k] * L[0] + c[1][k] * L[1] + c[2][k] * L[2]) / det);
}

/** Turn v by the rotation vector theta (Rodrigues). */
export function rotate(v, theta) {
  const angle = Math.hypot(...theta);
  if (!(angle > 0)) return v.slice();
  const k = theta.map((c) => c / angle);
  const co = Math.cos(angle), s = Math.sin(angle);
  const kv = cross(k, v), kd = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
  return v.map((c, i) => c * co + kv[i] * s + k[i] * kd * (1 - co));
}
