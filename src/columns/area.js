// Tessellated area (design 4.8, phase E). CPU reference for columns_area in
// src/shaders/compute/area.wgsl, which follows this file line by line.
//
// A column's actual area is its cell in a weighted Voronoi diagram of the
// column positions on the unit sphere. Between columns i and j the boundary
// is the great circle in the plane
//
//     x . (a_j p_j - a_i p_i) = 0,   a = 1 / cos(apothem),
//
// so between two touching columns (distance = apothem sum) it passes exactly
// through the contact point, and between equal columns it is the
// perpendicular bisector. Because every boundary plane goes through the
// origin and the three planes of any triple (i, j, k) are linearly dependent
// (their normals sum to zero), the cells meet at common vertices and tile the
// sphere exactly: sum of areas = 4 pi, whatever the weights.
//
// Each cell is computed alone, in the gnomonic projection about its column
// (x proportional to p_i + u1 e1 + u2 e2), where great circles are straight
// lines: the cell is a convex polygon, the intersection of half-planes, found
// by clipping a regular CLIP_SIDES-gon of radius clipRadius. Its spherical
// area is summed over fan triangles with the Van Oosterom-Strackee formula.
// Working in the column's own tangent coordinates keeps the result accurate
// in float32 (on the GPU): no difference of nearly equal unit vectors.
//
// The clip polygon bounds cells with no neighbours on some side; a cell that
// still touches it is "clipped" (counted on the GPU), and then the tiling is
// not exact there. Candidates must include every column whose boundary can
// cut the clip polygon: all columns within about 2 x clipRadius.

import { tangentBasis } from './flow.js';

export const CLIP_SIDES = 12;
export const CLIP = -1; // label of a polygon edge that lies on the clip polygon

/** a - 1 = 1/cos(r) - 1 as a series (exact to ~r^8: 1e-13 at r = 0.04); kept small for float32. */
export function excessWeight(r) {
  const r2 = r * r;
  return r2 * (0.5 + r2 * (5 / 24 + r2 * (61 / 720)));
}

/**
 * Half-plane c + n.u <= 0 of column i's cell against column j, in i's
 * gnomonic coordinates. d = p_j - p_i; with p_i.p_j = 1 - |d|^2 / 2 and
 * b = a - 1, c = (b_j - b_i) - (1 + b_j) |d|^2 / 2, n = (1 + b_j) (d.e1, d.e2).
 */
export function halfPlane(d, bi, bj, e1, e2) {
  const dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  const aj = 1 + bj;
  return {
    c: (bj - bi) - aj * dd * 0.5,
    n: [aj * (d[0] * e1[0] + d[1] * e1[1] + d[2] * e1[2]), aj * (d[0] * e2[0] + d[1] * e2[1] + d[2] * e2[2])],
  };
}

/**
 * Clip a convex polygon (vertices u, labels: the label of the edge from
 * vertex k to k+1) by c + n.u <= 0. Edges created along the clip line get
 * `label`. Sutherland-Hodgman.
 */
export function clipPolygon(poly, labels, c, n, label) {
  const out = [], lab = [];
  const m = poly.length;
  for (let k = 0; k < m; k++) {
    const cur = poly[k], nxt = poly[(k + 1) % m];
    const sc = c + n[0] * cur[0] + n[1] * cur[1];
    const sn = c + n[0] * nxt[0] + n[1] * nxt[1];
    const cut = () => { const t = sc / (sc - sn); return [cur[0] + (nxt[0] - cur[0]) * t, cur[1] + (nxt[1] - cur[1]) * t]; };
    if (sc <= 0) {
      out.push(cur); lab.push(labels[k]);
      if (sn > 0) { out.push(cut()); lab.push(label); }
    } else if (sn <= 0) {
      out.push(cut()); lab.push(labels[k]);
    }
  }
  return { poly: out, labels: lab };
}

/**
 * Solid angle of the gnomonic polygon (signed: counter-clockwise in (e1, e2)
 * is positive), summed over triangles (centre, u_k, u_k+1). For 3-vectors
 * P = (0, 0, 1), X = (u, 1), Y = (v, 1): tan(omega / 2) = det[P, X, Y] /
 * (|X||Y| + |X| + |Y| + X.Y) (Van Oosterom & Strackee 1983).
 */
export function gnomonicArea(poly) {
  let area = 0;
  for (let k = 0; k < poly.length; k++) {
    const u = poly[k], v = poly[(k + 1) % poly.length];
    const det = u[0] * v[1] - u[1] * v[0];
    const lx = Math.hypot(u[0], u[1], 1), ly = Math.hypot(v[0], v[1], 1);
    area += 2 * Math.atan2(det, lx * ly + lx + ly + u[0] * v[0] + u[1] * v[1] + 1);
  }
  return area;
}

/**
 * Centroid and principal axis of the polygon in the (flat) gnomonic plane:
 * {centroid: [x, y], axis: angle of the major axis from e1, spread: sqrt of
 * the major central second moment per area}. Sets where a split puts its halves.
 */
export function polygonShape(poly) {
  let a2 = 0, cx = 0, cy = 0, ixx = 0, iyy = 0, ixy = 0;
  for (let k = 0; k < poly.length; k++) {
    const [x0, y0] = poly[k], [x1, y1] = poly[(k + 1) % poly.length];
    const cr = x0 * y1 - x1 * y0;
    a2 += cr;
    cx += (x0 + x1) * cr;
    cy += (y0 + y1) * cr;
    ixx += (x0 * x0 + x0 * x1 + x1 * x1) * cr;
    iyy += (y0 * y0 + y0 * y1 + y1 * y1) * cr;
    ixy += (x0 * y1 + 2 * x0 * y0 + 2 * x1 * y1 + x1 * y0) * cr;
  }
  if (!(a2 > 0)) return { centroid: [0, 0], axis: 0, spread: 0 };
  const A = a2 / 2;
  cx /= 3 * a2; cy /= 3 * a2;
  const cxx = ixx / 12 / A - cx * cx, cyy = iyy / 12 / A - cy * cy, cxy = ixy / 24 / A - cx * cy;
  const axis = 0.5 * Math.atan2(2 * cxy, cxx - cyy);
  const lambda = 0.5 * (cxx + cyy) + Math.hypot(0.5 * (cxx - cyy), cxy);
  return { centroid: [cx, cy], axis, spread: Math.sqrt(Math.max(lambda, 0)) };
}

/**
 * Cell of column i. pos(k) -> [x, y, z] unit vector; apothem(k); candidates:
 * indices to clip against (i itself is skipped). Returns {area, poly, labels,
 * clipped, centroid, axis, spread, basis}.
 */
export function cellOf(i, pos, apothem, candidates, clipRadius) {
  const p = pos(i);
  const [e1, e2] = tangentBasis(p);
  let poly = [], labels = [];
  for (let k = 0; k < CLIP_SIDES; k++) {
    const t = (2 * Math.PI * k) / CLIP_SIDES;
    poly.push([clipRadius * Math.cos(t), clipRadius * Math.sin(t)]);
    labels.push(CLIP);
  }
  const bi = excessWeight(apothem(i));
  for (const j of candidates) {
    if (j === i) continue;
    const q = pos(j);
    const { c, n } = halfPlane([q[0] - p[0], q[1] - p[1], q[2] - p[2]], bi, excessWeight(apothem(j)), e1, e2);
    if (c + clipRadius * Math.hypot(n[0], n[1]) <= 0) continue; // misses the clip disc
    ({ poly, labels } = clipPolygon(poly, labels, c, n, j));
    if (poly.length < 3) break;
  }
  const area = poly.length >= 3 ? gnomonicArea(poly) : 0;
  return { area, poly, labels, clipped: labels.includes(CLIP), ...polygonShape(poly), basis: [e1, e2] };
}

/**
 * Cells of all columns (brute-force candidates within 4 x clipRadius; for
 * tests). positions: flat array (stride 3); apothems: array.
 */
export function tessellate(positions, apothems, clipRadius) {
  const n = apothems.length;
  const pos = (k) => [positions[3 * k], positions[3 * k + 1], positions[3 * k + 2]];
  const reach = 4 * clipRadius;
  const cells = [];
  for (let i = 0; i < n; i++) {
    const p = pos(i);
    const cand = [];
    for (let j = 0; j < n; j++) {
      if (j !== i && Math.hypot(positions[3 * j] - p[0], positions[3 * j + 1] - p[1], positions[3 * j + 2] - p[2]) < reach) cand.push(j);
    }
    cells.push(cellOf(i, pos, (k) => apothems[k], cand, clipRadius));
  }
  return cells;
}
