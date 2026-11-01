// Free radius with bending (design 4.3, 4.4; phase F). CPU reference for
// src/shaders/compute/flexure.wgsl.
//
// The surface elevations w of a step minimise
//
//   sum_i A_i [ (K_i r_i / 2) (w_i - w0_i)^2 + (K_i / 2) (w_i - weq_i)^2 ]
//     + sum_i (D_i A_i / 2) m_i^2,      m_i = (1 / A_i) sum_j c_ij (w_j - w_i)
//
//   - K = drho g: a Winkler foundation (isostasy) toward the Airy height weq;
//   - r = tau / dt: implicit Euler against viscous rebound (Haskell time tau)
//     from the step's starting elevation w0;
//   - D m^2 / 2: thin-plate bending, (D/2)(laplacian w)^2, with the
//     tessellation's Laplacian: c_ij = (shared cell boundary) / (distance)
//     between the two columns, the standard finite-volume ("cotangent")
//     weights of a Voronoi mesh. The Gaussian-curvature term of the Kirchhoff
//     energy is left out: it only adds boundary terms.
//   - D = E h_e^3 / 12 (1 - nu^2), h_e = the column's slices weighted by their
//     Maxwell factor this step (1 elastic, 0 fluid), the same factor the edge
//     law uses for in-plane stiffness.
//
// The design proposed directors (a slope per column, with a Kirchhoff
// penalty); bending of the elevation field on the tessellation needs no extra
// unknowns and passes the line-load benchmark, so it was used instead.
//
// Units: areas m^2, lengths m, w km, D N m, K Pa/m.

import { SECONDS_PER_MYR } from './profile.js';

export const EARTH_RADIUS_M = 6.371e6;

/** Elastic thickness (km) of a column profile (columnProfile in profile.js). */
export function elasticThicknessKm(prof, p) {
  const edt = p.crust_youngs_modulus_gpa * 1e9 * p.dt_crust_myr * SECONDS_PER_MYR;
  return prof.slices.reduce((h, s) => h + s.dz / (1 + edt / 10 ** Math.min(s.log10Viscosity, 30)), 0);
}

/** Flexural rigidity D (N m) of an elastic thickness h_e (km). */
export function rigidity(heKm, p) {
  const h = heKm * 1e3;
  return (p.flexure_factor * p.crust_youngs_modulus_gpa * 1e9 * h ** 3) / (12 * (1 - p.poisson_ratio ** 2));
}

/** Flexural length alpha = (4 D / K)^(1/4), m. */
export const flexuralLength = (D, K) => ((4 * D) / K) ** 0.25;

/** Deflection shape under a line load on an infinite plate: w(x) / w(0), x >= 0. */
export const lineLoadShape = (x, alpha) => Math.exp(-x / alpha) * (Math.cos(x / alpha) + Math.sin(x / alpha));

/** Peak deflection under a line load V0 (N/m): V0 alpha^3 / 8 D (m). */
export const lineLoadPeak = (V0, D, alpha) => (V0 * alpha ** 3) / (8 * D);

/**
 * Bending weights of a cell (area.js, cellOf): [{j, c}], one per polygon edge
 * shared with a neighbour (clip edges skipped). pos(k) -> unit vector.
 */
export function cellWeights(i, cell, pos) {
  const out = [];
  const pi = pos(i);
  for (let k = 0; k < cell.poly.length; k++) {
    const j = cell.labels[k];
    if (j < 0) continue;
    const u = cell.poly[k], v = cell.poly[(k + 1) % cell.poly.length];
    const pj = pos(j);
    out.push({ j, c: Math.hypot(v[0] - u[0], v[1] - u[1]) / Math.hypot(pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]) });
  }
  return out;
}

/**
 * Exact minimiser (dense solve; for tests). cols[i] = {A (m^2), K, r, w0,
 * weq, D, nbrs: [{j, c}]}. Returns w (km).
 */
export function solveExact(cols) {
  const n = cols.length;
  const H = Array.from({ length: n }, () => new Float64Array(n));
  const b = new Float64Array(n);
  // L_ij = c_ij / A_i, L_ii = -sum c / A_i; H = diag(A K (1 + r)) + L^T diag(D A) L.
  const L = cols.map((col, i) => {
    const row = new Map();
    let s = 0;
    for (const { j, c } of col.nbrs) { row.set(j, (row.get(j) ?? 0) + c / col.A); s += c; }
    row.set(i, (row.get(i) ?? 0) - s / col.A);
    return row;
  });
  cols.forEach((col, i) => {
    H[i][i] += col.A * col.K * (1 + col.r);
    b[i] = col.A * col.K * (col.weq + col.r * col.w0);
    const wgt = col.D * col.A;
    for (const [k, lk] of L[i]) for (const [m, lm] of L[i]) H[k][m] += wgt * lk * lm;
  });
  // Gaussian elimination with partial pivoting.
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(H[r][c]) > Math.abs(H[piv][c])) piv = r;
    [H[c], H[piv]] = [H[piv], H[c]];
    [b[c], b[piv]] = [b[piv], b[c]];
    for (let r = c + 1; r < n; r++) {
      const f = H[r][c] / H[c][c];
      if (f === 0) continue;
      for (let k = c; k < n; k++) H[r][k] -= f * H[c][k];
      b[r] -= f * b[c];
    }
  }
  const w = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < n; k++) s -= H[r][k] * w[k];
    w[r] = s / H[r][r];
  }
  return w;
}

/**
 * The GPU's iteration (columns_moment, then columns_flexure), `iterations`
 * damped Jacobi sweeps from w = w0. Returns w (km).
 */
export function jacobi(cols, iterations, damping) {
  let w = Float64Array.from(cols, (c) => c.w0);
  for (let it = 0; it < iterations; it++) {
    const M = cols.map((col, i) => (col.D * col.nbrs.reduce((s, { j, c }) => s + c * (w[j] - w[i]), 0)) / col.A);
    w = Float64Array.from(cols, (col, i) => {
      let grad = col.A * col.K * ((1 + col.r) * w[i] - col.weq - col.r * col.w0);
      let csum = 0, hb = 0;
      for (const { j, c } of col.nbrs) {
        grad += c * (M[j] - M[i]);
        csum += c;
        hb += (cols[j].D * c * c) / cols[j].A;
      }
      hb += (col.D * csum * csum) / col.A;
      return w[i] - grad / (col.A * col.K * (1 + col.r) + damping * hb);
    });
  }
  return w;
}
