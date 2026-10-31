// Plate identification (design.md phase 5, first part): plates are the
// connected components of the network of intact bonds between surface
// particles. Computed on the CPU from a readback, which is exact and cheap
// (union-find over ~10^4 particles), and keeps no extra GPU state.
//
// Plate ids persist: each new component inherits the previous id held by
// most of its particles. Components that lose that contest get fresh ids
// (a split); ids that no component inherits were absorbed (a merge) or
// destroyed. Both are reported as events.

export const PLATE_DEFAULTS = {
  intactDamage: 0.5,   // bonds with damage below this hold plates together
  minPlateSize: 20,    // smaller components get id 0 ("not a plate")
};

/**
 * @param P   particle arrays: is_active, state, position (vec4), velocity (vec4)
 * @param C   constraint arrays: is_active, particle_a, particle_b, damage, and
 *            optionally kind (contacts, kind 1, never join plates)
 * @param previous  {ids: Uint32Array per particle slot, nextId} or null
 * @returns {ids, nextId, plates: [{id, size, members}], events: [...]}
 */
export function labelPlates(P, C, previous = null, options = {}) {
  const opt = { ...PLATE_DEFAULTS, ...options };
  const n = P.is_active.length;
  const surface = (i) => P.is_active[i] && P.state[i] === 1;

  // Union-find with path halving and union by size.
  const parent = Int32Array.from({ length: n }, (_, i) => i);
  const size = new Int32Array(n).fill(1);
  const find = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  for (let k = 0; k < C.is_active.length; k++) {
    if (!C.is_active[k] || C.damage[k] >= opt.intactDamage || C.kind?.[k]) continue;
    const a = C.particle_a[k], b = C.particle_b[k];
    if (!surface(a) || !surface(b)) continue;
    let ra = find(a), rb = find(b);
    if (ra === rb) continue;
    if (size[ra] < size[rb]) [ra, rb] = [rb, ra];
    parent[rb] = ra;
    size[ra] += size[rb];
  }

  // Components big enough to be plates.
  const members = new Map(); // root -> [particle]
  for (let i = 0; i < n; i++) {
    if (!surface(i)) continue;
    const r = find(i);
    if (size[r] < opt.minPlateSize) continue;
    if (!members.has(r)) members.set(r, []);
    members.get(r).push(i);
  }
  const comps = [...members.values()].sort((a, b) => b.length - a.length);

  // Inherit previous ids by majority.
  const prevIds = previous?.ids;
  let nextId = previous?.nextId ?? 1;
  const overlap = comps.map((list) => {
    const counts = new Map();
    if (prevIds) for (const i of list) { const p = prevIds[i]; if (p) counts.set(p, (counts.get(p) ?? 0) + 1); }
    return counts;
  });
  const claims = []; // [count, comp, prevId]
  overlap.forEach((counts, c) => counts.forEach((count, id) => claims.push([count, c, id])));
  claims.sort((x, y) => y[0] - x[0] || x[1] - y[1] || x[2] - y[2]);
  const compId = new Array(comps.length).fill(0);
  const taken = new Set();
  for (const [, c, id] of claims) {
    if (compId[c] || taken.has(id)) continue;
    compId[c] = id;
    taken.add(id);
  }

  const events = [];
  comps.forEach((list, c) => {
    if (compId[c]) return;
    compId[c] = nextId++;
    // Fresh id: a split if its particles mostly came from an existing plate.
    let from = 0, best = 0;
    overlap[c].forEach((count, id) => { if (count > best) { best = count; from = id; } });
    if (from) events.push({ type: 'plate_split', plate: compId[c], from });
  });
  if (prevIds) {
    const before = new Set();
    for (let i = 0; i < n; i++) if (prevIds[i]) before.add(prevIds[i]);
    for (const id of before) {
      if (taken.has(id)) continue;
      // Where did most of its particles go?
      let into = 0, best = 0;
      overlap.forEach((counts, c) => { const k = counts.get(id) ?? 0; if (k > best) { best = k; into = compId[c]; } });
      events.push(into ? { type: 'plate_merge', plate: into, absorbed: id } : { type: 'plate_lost', plate: id });
    }
  }

  const ids = new Uint32Array(n);
  const plates = comps.map((list, c) => {
    for (const i of list) ids[i] = compId[c];
    return { id: compId[c], size: list.length, members: list };
  });
  return { ids, nextId, plates, events };
}

/**
 * Best-fit rigid rotation of a plate: omega minimising sum |v_i - omega x r_i|^2
 * (r on the unit sphere, v in rad/Myr). Returns the Euler pole (unit vector),
 * the rotation rate in degrees per Myr, the rigidity residual
 * sqrt(sum |v - omega x r|^2 / sum |v|^2) (0 for a perfectly rigid plate;
 * meaningless when the plate barely moves), and the rms speed and rms
 * non-rigid speed in the units of v.
 */
export function eulerPole(P, members) {
  const A = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  const b = [0, 0, 0];
  for (const i of members) {
    const r = [P.position[4 * i], P.position[4 * i + 1], P.position[4 * i + 2]];
    const v = [P.velocity[4 * i], P.velocity[4 * i + 1], P.velocity[4 * i + 2]];
    const rr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) A[3 * p + q] += (p === q ? rr : 0) - r[p] * r[q];
    b[0] += r[1] * v[2] - r[2] * v[1];
    b[1] += r[2] * v[0] - r[0] * v[2];
    b[2] += r[0] * v[1] - r[1] * v[0];
  }
  const omega = solve3(A, b);
  let resid = 0, total = 0;
  for (const i of members) {
    const r = [P.position[4 * i], P.position[4 * i + 1], P.position[4 * i + 2]];
    const v = [P.velocity[4 * i], P.velocity[4 * i + 1], P.velocity[4 * i + 2]];
    const w = [omega[1] * r[2] - omega[2] * r[1], omega[2] * r[0] - omega[0] * r[2], omega[0] * r[1] - omega[1] * r[0]];
    for (let c = 0; c < 3; c++) { resid += (v[c] - w[c]) ** 2; total += v[c] ** 2; }
  }
  const rate = Math.hypot(...omega);
  const count = Math.max(members.length, 1);
  return {
    omega,
    pole: rate > 0 ? omega.map((x) => x / rate) : [0, 1, 0],
    rateDegPerMyr: (rate * 180) / Math.PI,
    rigidityResidual: total > 0 ? Math.sqrt(resid / total) : 0,
    rmsSpeed: Math.sqrt(total / count),        // same units as v
    rmsNonRigidSpeed: Math.sqrt(resid / count), // same units as v
  };
}

function solve3(A, b) {
  const det = (m) => m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
  const d = det(A);
  if (Math.abs(d) < 1e-30) return [0, 0, 0];
  return [0, 1, 2].map((col) => {
    const m = A.slice();
    for (let row = 0; row < 3; row++) m[3 * row + col] = b[row];
    return det(m) / d;
  });
}

/** Distinct colour for a plate id (golden-angle hue spacing), as [r, g, b] in 0..1. */
export function plateColor(id) {
  if (!id) return [0.35, 0.35, 0.35];
  const h = (id * 137.508) % 360;
  const s = 0.55, l = 0.55;
  const k = (n) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n) => l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1));
  return [f(0), f(8), f(4)];
}
