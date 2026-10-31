// A bonded chain along the equator, for solver tests: builder (runs in the
// page) and exact one-step solution (runs in Node).
//
// N particles at angles phi_i = (i - (N-1)/2) s, bonds to neighbours with rest
// length (1 + eps) times the current spacing. Rest length is the sum of
// apothems, so apothems are set to half of it; eps < 0 keeps the chain in
// tension, so contacts (which also act at the apothem sum) stay out of it,
// and no mantle flow (tests set mantle_drive = 0). One overdamped implicit step minimises
//   sum_i (gamma/2dt) u_i^2 + sum_j (k/2) C_j^2,   C_j = u_{j+1} - u_j - delta,
// with u the arc displacement and delta the arc rest-length excess. In units
// of the base drag that is the tridiagonal system
//   u_i + K (C_{i-1} - C_i) = 0,  K = bond_rate * dt
// (chord vs. arc differences are O(s^2/24), ~1e-4 here).

export const CHAIN = { n: 61, spacing: 0.04, eps: -0.01 };
// Half the rest length; tests set base_apothem to it so every particle has the reference drag.
CHAIN.apothem = Math.sin(CHAIN.spacing / 2) * (1 + CHAIN.eps);

/** Page-side state builder; pass its source to page.evaluate. */
export function chainState({ n, spacing, eps, apothem }) {
  const phi = (i) => (i - (n - 1) / 2) * spacing;
  const position = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) position.set([Math.sin(phi(i)), 0, Math.cos(phi(i))], 3 * i);
  const fill = (v, m = n) => new Float32Array(m).fill(v);
  return {
    particles: {
      count: n, position, thickness: fill(7), compositionDensity: fill(3000),
      temperature: fill(278), age: fill(100), apothem: fill(apothem), // age 100 and base size: reference drag
    },
    bonds: {
      count: n - 1,
      a: Uint32Array.from({ length: n - 1 }, (_, j) => j),
      b: Uint32Array.from({ length: n - 1 }, (_, j) => j + 1),
      damage: fill(0, n - 1),
    },
    yieldStressMPa: 1e9,
  };
}

/** Exact arc displacements after one step, by the Thomas algorithm. */
export function chainExact({ n, spacing, eps }, K) {
  const delta = 2 * Math.asin(Math.sin(spacing / 2) * (1 + eps)) - spacing;
  const a = new Float64Array(n), b = new Float64Array(n), c = new Float64Array(n), d = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const left = i > 0, right = i < n - 1;
    b[i] = 1 + K * ((left ? 1 : 0) + (right ? 1 : 0));
    a[i] = left ? -K : 0;
    c[i] = right ? -K : 0;
    d[i] = K * delta * ((left ? 1 : 0) - (right ? 1 : 0));
  }
  for (let i = 1; i < n; i++) { const m = a[i] / b[i - 1]; b[i] -= m * c[i - 1]; d[i] -= m * d[i - 1]; }
  const u = new Float64Array(n);
  u[n - 1] = d[n - 1] / b[n - 1];
  for (let i = n - 2; i >= 0; i--) u[i] = (d[i] - c[i] * u[i + 1]) / b[i];
  return u;
}
