// Column orientation and reference shape (design 4.2, 4.8; phase G). CPU
// reference for the shape and spin code in src/shaders/compute/ (bindings.wgsl:
// column_radius; solver.wgsl: spin; flow.wgsl: columns_flow; split.wgsl).
//
// A column carries
//   - a body axis b: a unit tangent vector, its body x-axis. It moves with the
//     column (projected onto the new tangent plane each step, which is
//     parallel transport to first order) and turns with its spin, a solved
//     rotation about the outward radial (counter-clockwise seen from above).
//     Rigid rotation of a plate turns its columns with it, so the edges'
//     tangential channel (src/edges/law.js) costs nothing for it.
//   - a reference shape: the hexagon of area area_ref stretched by
//     U = exp(L), L = [[p, q], [q, -p]] (traceless, so det U = 1 and the area
//     is kept), in the body frame (b, x x b). Its principal radii are
//     apothem e^{+-s}, s = |(p, q)|: aspect ratio e^{2 s}, long axis at angle
//     (1/2) atan2(q, p) from b. The radius along a unit direction n (body
//     frame) is apothem / sqrt(n^T U^-2 n), so an affinely stretched lattice
//     keeps its rest lengths exact.
//
// Distributed (ductile) inelastic strain of a column, fitted from its edges,
// changes the shape: the deviatoric part adds to L (logarithmic strain), the
// volumetric part changes area_ref (src/columns/flow.js). Brittle slip on an
// edge (a fault) is a jump between columns, not a change of either shape.

/** 1 / sqrt(n^T exp(-2L) n) for a unit body-frame direction (nx, ny). */
export function radiusFactor(p, q, nx, ny) {
  const s = Math.hypot(p, q);
  const f = s > 1e-6 ? Math.sinh(2 * s) / s : 2;
  const nLn = p * (nx * nx - ny * ny) + 2 * q * nx * ny;
  return 1 / Math.sqrt(Math.cosh(2 * s) - f * nLn);
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a) => { const l = Math.hypot(...a); return a.map((v) => v / l); };

/** Components (u, v) of a 3D vector d in the body frame (b, x x b) of a column at unit position x. */
export function bodyComponents(x, b, d) {
  return [dot(d, b), dot(d, cross(x, b))];
}

/** The column's radius toward a 3D direction d (any length): apothem x radiusFactor. */
export function radiusAlong(apothem, p, q, x, b, d) {
  const [u, v] = bodyComponents(x, b, d);
  const l = Math.hypot(u, v);
  return l > 0 ? apothem * radiusFactor(p, q, u / l, v / l) : apothem;
}

/** Aspect ratio and long-axis angle (from b) of a shape. */
export function shapeAxes(p, q) {
  const s = Math.hypot(p, q);
  return { aspect: Math.exp(2 * s), axis: 0.5 * Math.atan2(q, p), s };
}

/**
 * Least-squares strain tensor (exx, eyy, exy; body frame) from a column's
 * edges: rows {nx, ny, en, g}, with n the unit direction to the partner, en
 * the normal strain n^T e n and g the tangential strain (tangential
 * displacement over rest length, t = (-ny, nx)). Each edge gives two
 * equations: en = n^T e n and g = t^T e n + w. The fourth unknown w is the
 * neighbourhood's rotation relative to the column's own spin (zero when the
 * column turns with its surroundings): it is fitted, so a column that lags
 * the local rotation does not read it as shear. Returns {exx, eyy, exy, w},
 * or null when the directions do not determine them: a singular system, or
 * a trace beyond twice the largest normal strain any edge saw (a 2D tensor's trace
 * is at most twice its largest principal strain, and edges spread in angle
 * sample that; a nearly one-directional neighbourhood, like a strip, passes
 * the determinant test but extrapolates without bound).
 */
export function fitStrain(rows) {
  const A = Array.from({ length: 4 }, () => [0, 0, 0, 0]), r = [0, 0, 0, 0];
  const add = (a, y) => { for (let i = 0; i < 4; i++) { r[i] += a[i] * y; for (let j = 0; j < 4; j++) A[i][j] += a[i] * a[j]; } };
  for (const { nx, ny, en, g } of rows) {
    add([nx * nx, ny * ny, 2 * nx * ny, 0], en);
    add([-nx * ny, nx * ny, nx * nx - ny * ny, 1], g);
  }
  const d = det(A);
  const tr = A[0][0] + A[1][1] + A[2][2] + A[3][3];
  if (!(Math.abs(d) > 1e-6 * tr ** 4)) return null;
  const col = (k) => A.map((row, i) => row.map((v, j) => (j === k ? r[i] : v)));
  const fit = { exx: det(col(0)) / d, eyy: det(col(1)) / d, exy: det(col(2)) / d, w: det(col(3)) / d };
  const seen = rows.reduce((m, { en }) => Math.max(m, Math.abs(en)), 0);
  return Math.abs(fit.exx + fit.eyy) <= 2 * seen * (1 + 1e-4) ? fit : null;
}

/** Determinant by elimination with partial pivoting. */
function det(m0) {
  const m = m0.map((row) => [...row]), n = m.length;
  let d = 1;
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[piv][c])) piv = r;
    if (m[piv][c] === 0) return 0;
    if (piv !== c) { [m[c], m[piv]] = [m[piv], m[c]]; d = -d; }
    d *= m[c][c];
    for (let r = c + 1; r < n; r++) { const f = m[r][c] / m[c][c]; for (let k = c; k < n; k++) m[r][k] -= f * m[c][k]; }
  }
  return d;
}

/**
 * Turn a shape with the material, then add a strain increment's deviatoric
 * part (factor x), capped at aspect ratio `cap`. The shape is a finite-strain
 * ellipse, a material marker: it turns with the material around it (the
 * column's spin, which turns the body frame, plus e.w, the neighbourhood's
 * rotation relative to the spin), not with the column alone. Returns {p, q}.
 */
export function updateShape(p, q, e, factor, cap) {
  const w = factor * (e.w ?? 0), c = Math.cos(2 * w), sn = Math.sin(2 * w);
  let np = c * p - sn * q + factor * 0.5 * (e.exx - e.eyy);
  let nq = sn * p + c * q + factor * e.exy;
  const s = Math.hypot(np, nq), smax = 0.5 * Math.log(cap);
  if (s > smax) { np *= smax / s; nq *= smax / s; }
  return { p: np, q: nq };
}

/**
 * Split across the long axis (design 4.8, phase G): each half has half the
 * area and half the aspect ratio, along the same axis (axes swap if it falls
 * below 1), and sits at its own radius along that axis from the parent's
 * position, so the two halves touch at their rest length. Returns the halves'
 * shape {p, q}, the axis angle (from b) and the offset of each half (unit sphere).
 */
export function splitShape(apothem, p, q) {
  const { s, axis } = shapeAxes(p, q);
  const sNew = s - 0.5 * Math.log(2);
  const k = s > 0 ? sNew / s : 0;
  return { p: p * k, q: q * k, axis, offset: (apothem / Math.SQRT2) * Math.exp(sNew) };
}

/** Squared radius of gyration about the column axis (unit sphere): area_ref cosh(2 s) / 2 pi (equal-area ellipse). */
export function gyrationSquared(areaRef, p, q) {
  return (areaRef * Math.cosh(2 * Math.hypot(p, q))) / (2 * Math.PI);
}

/**
 * Carry a body axis to a new position and turn it by `spin` (rad, CCW about
 * the outward radial): project onto the new tangent plane, normalise, rotate.
 */
export function carryBodyAxis(b, xNew, spin) {
  const x = unit(xNew);
  const t = unit(b.map((v, c) => v - dot(b, x) * x[c]));
  const w = cross(x, t);
  return t.map((v, c) => Math.cos(spin) * v + Math.sin(spin) * w[c]);
}

/**
 * Plane elasticity of a triangular lattice of columns joined by normal and
 * tangential springs (k_n, k_t, per edge), each column free to rotate
 * (Bravais lattice: by symmetry every column turns with the local rotation).
 * Energy per area sqrt(3) [k_n tr(e)^2 / 4 + (k_n + k_t) e_dev:e_dev / 4]:
 * 2D bulk modulus K = sqrt(3) k_n / 2, shear modulus mu = sqrt(3) (k_n + k_t) / 4,
 * so nu = (k_n - k_t) / (3 k_n + k_t), E = 4 K mu / (K + mu).
 */
export function latticeModuli(kn, kt) {
  const K = (Math.sqrt(3) * kn) / 2, mu = (Math.sqrt(3) * (kn + kt)) / 4;
  return { bulk: K, shear: mu, poisson: (K - mu) / (K + mu), youngs: (4 * K * mu) / (K + mu) };
}

/**
 * Normal edge stiffness per thickness-integrated Young's modulus, k_n / (E h),
 * for a lattice with tangential ratio r = k_t / k_n: from latticeModuli,
 * E = 2 sqrt(3) k_n (1 + r) / (3 + r).
 */
export const normalStiffnessPerModulus = (r) => (3 + r) / (2 * Math.sqrt(3) * (1 + r));
