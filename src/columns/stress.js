// Per-column stress tensor and net edge force, for drawing (README, "What you are looking at").
// CPU reference for vs_force and vs_stress in src/shaders/render/crust.wgsl.
//
// Each edge carries a depth-integrated normal stress S = sigma_n x overlap
// (MPa km, + tension) and tangential stress St = tau x overlap (+ when b is
// displaced along +t relative to a, t = up x n, n the unit direction a -> b).
// The lattice force it exerts on a is f = c rest (S n + St t), with
// c = normalStiffnessPerModulus(k_t / k_n) (src/columns/shape.js): the same
// conversion bondRate uses, so these are the forces the solver applies. On b
// it is -f (the formula with n and t both reversed).
//
// Net force: sum f over the column's edges. In this overdamped model it is
// balanced by basal drag (drag x velocity relative to the mantle), so it is
// the push or pull of the neighbours, not an acceleration.
//
// Stress (Love-Weber average over the column): sigma = sym(sum r (x) f) / (A h),
// r = lever x |x_b - x_a| x n the contact point from the column centre, A the
// reference area, h the mean depth overlap of its edges. For a hexagonal
// lattice under uniform strain eps this gives E eps / (1 - nu) (isotropic),
// the plane-stress continuum value: the edges and the drawing agree.

import { normalStiffnessPerModulus } from './shape.js';

/**
 * @param edges [{n: [nx, ny], t?: [tx, ty], sigmaN, tau, overlap, rest, lever, dist}]
 *   in the column's 2D tangent frame, for this column as the edge's `a` end
 *   (n toward the partner, t = up x n; lever = fraction of dist from this column).
 * @returns {force: [fx, fy] (MPa km x unit length), sxx, syy, sxy (MPa), h}
 */
export function columnStress(edges, areaRef, shearRatio) {
  const c = normalStiffnessPerModulus(shearRatio);
  let fx = 0, fy = 0, xx = 0, yy = 0, xy = 0, h = 0, m = 0;
  for (const e of edges) {
    if (!(e.overlap > 0)) continue;
    const [nx, ny] = e.n, [tx, ty] = e.t ?? [-ny, nx];
    const S = e.sigmaN * e.overlap, St = e.tau * e.overlap;
    const gx = c * e.rest * (S * nx + St * tx), gy = c * e.rest * (S * ny + St * ty);
    fx += gx; fy += gy;
    const r = e.lever * e.dist;
    xx += r * nx * gx; yy += r * ny * gy; xy += 0.5 * r * (nx * gy + ny * gx);
    h += e.overlap; m++;
  }
  if (m === 0) return { force: [0, 0], sxx: 0, syy: 0, sxy: 0, h: 0 };
  h /= m;
  const k = 1 / (areaRef * h);
  return { force: [fx, fy], sxx: xx * k, syy: yy * k, sxy: xy * k, h };
}

/** Principal stresses (s1 >= s2) and the angle of the s1 axis from the frame's x axis. */
export function principalStresses(sxx, syy, sxy) {
  const mean = 0.5 * (sxx + syy), R = Math.hypot(0.5 * (sxx - syy), sxy);
  return { s1: mean + R, s2: mean - R, angle: 0.5 * Math.atan2(2 * sxy, sxx - syy) };
}
