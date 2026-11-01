// Column-level irreversible processes, CPU reference (design 4.6, 4.7; phase D).
// Mirrors columns_flow and columns_gpe_force in src/shaders/compute/flow.wgsl.

import { SECONDS_PER_MYR } from './profile.js';

export const EARTH_RADIUS_M = 6.371e6;

/**
 * Volumetric flow at fixed crust volume: a column's reference area changes by
 * the isotropic part of its edges' inelastic strain, d ln area_ref =
 * 2 x mean(inelastic) (area scales as length squared); crust thickness
 * changes inversely. Deviatoric parts (pure shear) cancel in the mean.
 */
export function columnFlow({ area, thicknessKm }, inelastic, factor = 1) {
  if (!inelastic.length) return { area, thicknessKm };
  const mean = inelastic.reduce((a, b) => a + b, 0) / inelastic.length;
  const f = Math.exp(2 * factor * mean);
  return { area: area * f, thicknessKm: thicknessKm / f, apothem: Math.sqrt((area * f) / (2 * Math.sqrt(3))) };
}

/** Tangent basis at unit vector n (the slot-angle basis: e1 = normalize(ref x n), e2 = n x e1). */
export function tangentBasis(n) {
  const ref = Math.abs(n[2]) > 0.9 ? [1, 0, 0] : [0, 0, 1];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  let e1 = cross(ref, n);
  const l = Math.hypot(...e1);
  e1 = e1.map((v) => v / l);
  return [e1, cross(n, e1)];
}

/**
 * Least-squares gradient of a field f over a column's neighbours, in the
 * column's tangent plane: minimise sum_j (f_j - f_i - g . d_j)^2, d_j the
 * tangent projection of x_j - x_i. Returns g as a 3-vector (per unit-sphere
 * length), or null with fewer than 2 neighbours or a degenerate fit.
 */
export function gradient(xi, fi, neighbours) {
  if (neighbours.length < 2) return null;
  const [e1, e2] = tangentBasis(xi);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  let a11 = 0, a12 = 0, a22 = 0, b1 = 0, b2 = 0;
  for (const { x, f } of neighbours) {
    const d = [x[0] - xi[0], x[1] - xi[1], x[2] - xi[2]];
    const u = dot(d, e1), v = dot(d, e2), df = f - fi;
    a11 += u * u; a12 += u * v; a22 += v * v; b1 += u * df; b2 += v * df;
  }
  const det = a11 * a22 - a12 * a12;
  if (!(det > 1e-12 * (a11 + a22) ** 2)) return null;
  const g1 = (a22 * b1 - a12 * b2) / det, g2 = (a11 * b2 - a12 * b1) / det;
  return [0, 1, 2].map((c) => g1 * e1[c] + g2 * e2[c]);
}

/**
 * GPE force (design 4.6) as this step's predicted displacement: F = -A grad
 * Gamma against basal drag gamma = A eta_a / h, so dx = -grad Gamma h dt / eta_a,
 * independent of the column's area. grad in MPa km per unit-sphere length;
 * returns a unit-sphere displacement 3-vector.
 */
export function gpeDisplacement(grad, lithosphereKm, p) {
  const gradPa = (1e9 / EARTH_RADIUS_M); // MPa km per unit length -> N/m per m
  const h = Math.max(p.asthenosphere_base_km - lithosphereKm, p.min_channel_km) * 1e3;
  const s = -p.gpe_force_factor * gradPa * h * p.dt_crust_myr * SECONDS_PER_MYR / p.asthenosphere_viscosity_pa_s / EARTH_RADIUS_M;
  return grad.map((v) => s * v);
}
