// Edge topology, CPU reference (design 3.2, phase A).
//
// Pure functions on plain arrays, mirroring the GPU kernels in
// src/shaders/compute/edges.wgsl, so that each rule can be unit-tested
// against a known answer and the kernels against this reference.
//
// The rule: an edge exists between columns i and j exactly while they are in
// range, i.e. their map-view gap is below `edge_range_margin` times the
// apothem sum AND their depth intervals overlap. Edges are keyed by pair and
// keep their state while the pair stays in range. Exception (until stress-
// based subduction initiation, phase F): two surface columns overlapping by
// more than `subduction_overlap` do not get a *new* edge; the lower one
// becomes a slab instead.

import { CellList } from '../init/cell-list.js';

const SECONDS_PER_MYR = 3.15576e13;
/** Columns lighter than this (kg/m^3 at mantle temperature) are continental. */
export const CONTINENTAL_DENSITY_MAX = 2850;

/** Depth of the column's base below its top, km (lithosphere_thickness_km in solver.wgsl). */
export function lithosphereThicknessKm(compositionDensity, ageMyr, params) {
  if (compositionDensity < CONTINENTAL_DENSITY_MAX) return params.continental_lithosphere_km;
  const ageS = Math.max(ageMyr, 0) * SECONDS_PER_MYR;
  return Math.min((2.32 * Math.sqrt(params.thermal_diffusivity_m2_s * ageS)) / 1000, params.oceanic_lithosphere_max_km);
}

/**
 * Columns as plain arrays, for the functions below.
 * @param P  per-particle arrays: position (stride `stride`), apothem, elevation,
 *           composition_density, age, state, is_active (as unpacked from the
 *           Particle buffer, or built by hand in tests)
 * @param positionField  which position to use (the GPU decides on predicted_pos)
 */
export function columnsFrom(P, params, { stride = 4, positionField = 'predicted_pos', count = P.is_active.length } = {}) {
  const pos = P[positionField] ?? P.position;
  const position = new Float64Array(3 * count);
  const top = new Float64Array(count), bottom = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    for (let c = 0; c < 3; c++) position[3 * i + c] = pos[stride * i + c];
    top[i] = P.elevation[i];
    bottom[i] = P.elevation[i] - lithosphereThicknessKm(P.composition_density[i], P.age[i], params);
  }
  return {
    count, position, top, bottom,
    apothem: Float64Array.from(P.apothem.subarray(0, count)),
    surface: Uint8Array.from({ length: count }, (_, i) => P.state[i] === 1),
    active: Uint8Array.from({ length: count }, (_, i) => P.is_active[i] !== 0),
  };
}

const chord = (cols, i, j) => Math.hypot(
  cols.position[3 * i] - cols.position[3 * j],
  cols.position[3 * i + 1] - cols.position[3 * j + 1],
  cols.position[3 * i + 2] - cols.position[3 * j + 2]);

/** Map-view overlap (apothem sum minus centre distance; negative = gap). */
export function overlap(cols, i, j) {
  return cols.apothem[i] + cols.apothem[j] - chord(cols, i, j);
}

/**
 * Signed distance of pair (i, j) from the range boundary, as a fraction of the
 * apothem sum: positive = in range in map view. Useful to skip pairs within
 * float rounding of the boundary when comparing with the GPU.
 */
export function rangeMargin(cols, i, j, params) {
  const contact = cols.apothem[i] + cols.apothem[j];
  return overlap(cols, i, j) / contact + params.edge_range_margin;
}

export function depthOverlap(cols, i, j) {
  return Math.min(cols.top[i], cols.top[j]) > Math.max(cols.bottom[i], cols.bottom[j]);
}

export function inRange(cols, i, j, params) {
  return cols.active[i] && cols.active[j] && rangeMargin(cols, i, j, params) > 0 && depthOverlap(cols, i, j);
}

export const pairKey = (i, j) => (i < j ? `${i},${j}` : `${j},${i}`);

/** Every in-range pair, as a Set of pairKey strings. */
export function inRangePairs(cols, params) {
  let amax = 0;
  for (let i = 0; i < cols.count; i++) if (cols.active[i]) amax = Math.max(amax, cols.apothem[i]);
  const cutoff = 2 * amax * (1 + params.edge_range_margin) * 1.001;
  const pos32 = Float32Array.from(cols.position);
  const out = new Set();
  new CellList(pos32, cutoff).forEachPair((i, j) => { if (inRange(cols, i, j, params)) out.add(pairKey(i, j)); });
  return out;
}

/**
 * One topology update. `previous` is a Set of pairKeys (the edges before the
 * update). Returns the edge set after it, split into kept / created / removed.
 */
export function updateEdges(cols, previous, params) {
  const now = inRangePairs(cols, params);
  const kept = new Set(), created = new Set(), removed = new Set();
  for (const k of previous) (now.has(k) ? kept : removed).add(k);
  for (const k of now) {
    if (previous.has(k)) continue;
    const [i, j] = k.split(',').map(Number);
    const subducts = cols.surface[i] && cols.surface[j] && overlap(cols, i, j) > params.subduction_overlap;
    if (!subducts) created.add(k);
  }
  return { edges: new Set([...kept, ...created]), kept, created, removed };
}

/**
 * Angle of each partner around column i, in i's tangent plane (same basis as
 * slot_angle in edges.wgsl): e1 = normalize(ref x n), e2 = n x e1, with
 * ref = z, or x near the poles.
 */
export function partnerAngle(cols, i, j) {
  const n = [cols.position[3 * i], cols.position[3 * i + 1], cols.position[3 * i + 2]];
  const ln = Math.hypot(...n);
  for (let c = 0; c < 3; c++) n[c] /= ln;
  const ref = Math.abs(n[2]) > 0.9 ? [1, 0, 0] : [0, 0, 1];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  let e1 = cross(ref, n);
  const l1 = Math.hypot(...e1);
  e1 = e1.map((v) => v / l1);
  const e2 = cross(n, e1);
  const d = [0, 1, 2].map((c) => cols.position[3 * j + c] - cols.position[3 * i + c]);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  return Math.atan2(dot(d, e2), dot(d, e1));
}

/**
 * Check the GPU edge storage against its invariants. Returns a list of
 * problems (empty = all good), so tests can print them all.
 * @param E      unpacked Edge arrays (layout.js)
 * @param pool   Uint32Array of the edge free list (4 header words, then indices)
 * @param slots  Uint32Array of the deltas buffer; layout from SLOTS (layout.js)
 * @param cols   columns (columnsFrom), for the angle order; optional
 */
export function checkEdgeStorage({ E, pool, slots, layout, particleActive, cols = null }) {
  const problems = [];
  const cap = E.is_active.length;
  // Free list: every slot free or active, exactly once.
  const free = new Uint8Array(cap);
  for (let k = 0; k < pool[0]; k++) {
    const e = pool[4 + k];
    if (e >= cap) { problems.push(`free list holds out-of-range ${e}`); continue; }
    if (free[e]) problems.push(`edge ${e} is in the free list twice`);
    if (E.is_active[e]) problems.push(`edge ${e} is both free and active`);
    free[e] = 1;
  }
  let active = 0;
  for (let e = 0; e < cap; e++) active += E.is_active[e] ? 1 : 0;
  if (active + pool[0] !== cap) problems.push(`${active} active + ${pool[0]} free != ${cap}`);

  // No duplicate pairs; both ends active.
  const seen = new Map();
  const listedAt = new Map(); // edge -> count of slot lists naming it
  for (let e = 0; e < cap; e++) {
    if (!E.is_active[e]) continue;
    const a = E.particle_a[e], b = E.particle_b[e];
    if (a === b) problems.push(`edge ${e} joins particle ${a} to itself`);
    if (!particleActive[a] || !particleActive[b]) problems.push(`edge ${e} touches an inactive particle`);
    const k = pairKey(a, b);
    if (seen.has(k)) problems.push(`edges ${seen.get(k)} and ${e} duplicate pair ${k}`);
    seen.set(k, e);
    listedAt.set(e, 0);
  }

  // Slot lists: symmetric (an edge is listed by exactly its two ends), counts
  // match, sorted by angle.
  const W = layout.wordsPerParticle;
  const degree = new Map();
  for (const [e] of listedAt) for (const p of [E.particle_a[e], E.particle_b[e]]) degree.set(p, (degree.get(p) ?? 0) + 1);
  for (let i = 0; i < particleActive.length; i++) {
    if (!particleActive[i]) continue;
    const n = slots[W * i + layout.countWord];
    const want = degree.get(i) ?? 0;
    if (n !== want) problems.push(`particle ${i}: slot count ${n}, edges touching it ${want}`);
    if (n > layout.capacity) { problems.push(`particle ${i}: slot list overflow (${n})`); continue; }
    let prevAngle = -Infinity;
    for (let s = 0; s < n; s++) {
      const e = slots[W * i + layout.firstSlotWord + s];
      if (!listedAt.has(e)) { problems.push(`particle ${i} lists inactive edge ${e}`); continue; }
      if (E.particle_a[e] !== i && E.particle_b[e] !== i) { problems.push(`particle ${i} lists edge ${e}, which is not its own`); continue; }
      listedAt.set(e, listedAt.get(e) + 1);
      if (cols) {
        const j = E.particle_a[e] === i ? E.particle_b[e] : E.particle_a[e];
        const ang = partnerAngle(cols, i, j);
        if (ang < prevAngle - 1e-5) problems.push(`particle ${i}: slots not sorted by angle`);
        prevAngle = ang;
      }
    }
  }
  for (const [e, c] of listedAt) if (c !== 2) problems.push(`edge ${e} is listed ${c} times (want 2)`);
  return problems;
}
