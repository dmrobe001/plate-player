// Discrete calculus on an icosphere (a triangulated unit sphere): the
// cotangent Laplacian, a conjugate-gradient Poisson solver, nodal gradients,
// and point location with linear interpolation. Plain JavaScript; the mantle
// grid is small enough (2562 nodes at subdivision 4) that the CPU is fine.

import { createIcosphere } from '../geometry/icosphere.js';

export class SphereMesh {
  constructor(subdivisions) {
    const ico = createIcosphere(subdivisions);
    this.ico = ico;
    const n = (this.n = ico.vertices.length);
    this.pos = Float64Array.from(ico.vertices.flat());
    this.tri = ico.indices;
    this.neighbours = ico.neighbours;
    const nt = this.tri.length / 3;

    // Triangle areas, node areas (one third of incident triangles), incident lists.
    this.triArea = new Float64Array(nt);
    this.area = new Float64Array(n);
    this.incident = Array.from({ length: n }, () => []);
    // Cotangent weights per edge, stored per node as parallel arrays.
    const weights = Array.from({ length: n }, () => new Map());
    for (let t = 0; t < nt; t++) {
      const v = [this.tri[3 * t], this.tri[3 * t + 1], this.tri[3 * t + 2]];
      const p = v.map((i) => this.point(i));
      const A = 0.5 * norm(cross(sub(p[1], p[0]), sub(p[2], p[0])));
      this.triArea[t] = A;
      for (let k = 0; k < 3; k++) {
        this.area[v[k]] += A / 3;
        this.incident[v[k]].push(t);
        // The angle at vertex k is opposite edge (k+1, k+2).
        const a = v[(k + 1) % 3], b = v[(k + 2) % 3];
        const e1 = sub(p[(k + 1) % 3], p[k]), e2 = sub(p[(k + 2) % 3], p[k]);
        const cot = dot(e1, e2) / norm(cross(e1, e2));
        weights[a].set(b, (weights[a].get(b) ?? 0) + cot / 2);
        weights[b].set(a, (weights[b].get(a) ?? 0) + cot / 2);
      }
    }
    this.wIdx = weights.map((m) => Int32Array.from(m.keys()));
    this.wVal = weights.map((m) => Float64Array.from(m.values()));
    this.wSum = Float64Array.from(this.wVal, (w) => w.reduce((s, x) => s + x, 0));
  }

  point(i) {
    return [this.pos[3 * i], this.pos[3 * i + 1], this.pos[3 * i + 2]];
  }

  /** (L f)_i = (1/A_i) sum_j w_ij (f_j - f_i): approximates the surface Laplacian. */
  laplacian(f, out = new Float64Array(this.n)) {
    for (let i = 0; i < this.n; i++) {
      let s = 0;
      const idx = this.wIdx[i], w = this.wVal[i];
      for (let k = 0; k < idx.length; k++) s += w[k] * (f[idx[k]] - f[i]);
      out[i] = s / this.area[i];
    }
    return out;
  }

  /** Area-weighted mean. */
  mean(f) {
    let s = 0, a = 0;
    for (let i = 0; i < this.n; i++) { s += f[i] * this.area[i]; a += this.area[i]; }
    return s / a;
  }

  /**
   * Solve L phi = rhs (rhs is made mean-free first) by conjugate gradients on
   * the symmetric form K phi = -A rhs with K = -A L; phi is returned with zero
   * mean. `phi` is the starting guess and is overwritten.
   */
  solvePoisson(rhs, phi, tolerance = 1e-8, maxIterations = 500) {
    return this.solveScreened(rhs, phi, Infinity, tolerance, maxIterations);
  }

  /**
   * Solve (L - 1/l^2) phi = rhs. With l = Infinity this is the Poisson problem
   * (rhs made mean-free, phi returned with zero mean); with finite l the
   * operator is definite and the response to a point source decays over ~l.
   */
  solveScreened(rhs, phi, screening, tolerance = 1e-8, maxIterations = 500) {
    const n = this.n, area = this.area, wIdx = this.wIdx, wVal = this.wVal;
    const s2 = Number.isFinite(screening) ? 1 / (screening * screening) : 0;
    const wSum = s2 ? Float64Array.from(this.wSum, (w, i) => w + s2 * area[i]) : this.wSum;
    const m = s2 ? 0 : this.mean(rhs);
    const r = new Float64Array(n), z = new Float64Array(n), p = new Float64Array(n), Kp = new Float64Array(n);
    // K x = (sum_j w_ij) x_i - sum_j w_ij x_j  (symmetric positive semi-definite)
    const applyK = (x, out) => {
      for (let i = 0; i < n; i++) {
        let s = wSum[i] * x[i];
        const idx = wIdx[i], w = wVal[i];
        for (let k = 0; k < idx.length; k++) s -= w[k] * x[idx[k]];
        out[i] = s;
      }
    };
    applyK(phi, r);
    let bNorm = 0;
    for (let i = 0; i < n; i++) {
      const b = -(rhs[i] - m) * area[i];
      bNorm += b * b;
      r[i] = b - r[i];
    }
    bNorm = Math.sqrt(bNorm) || 1;
    let rz = 0;
    for (let i = 0; i < n; i++) { z[i] = r[i] / wSum[i]; p[i] = z[i]; rz += r[i] * z[i]; }
    let it = 0;
    for (; it < maxIterations; it++) {
      let rr = 0;
      for (let i = 0; i < n; i++) rr += r[i] * r[i];
      if (Math.sqrt(rr) < tolerance * bNorm) break;
      applyK(p, Kp);
      let pKp = 0;
      for (let i = 0; i < n; i++) pKp += p[i] * Kp[i];
      const alpha = rz / pKp;
      let rzNew = 0;
      for (let i = 0; i < n; i++) {
        phi[i] += alpha * p[i];
        r[i] -= alpha * Kp[i];
        z[i] = r[i] / wSum[i];
        rzNew += r[i] * z[i];
      }
      const beta = rzNew / rz;
      rz = rzNew;
      for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
    }
    if (!s2) {
      const pm = this.mean(phi);
      for (let i = 0; i < n; i++) phi[i] -= pm;
    }
    return it;
  }

  /** Tangent gradient at each node: area-weighted average of the incident triangles' gradients. */
  gradient(f, out = new Float64Array(3 * this.n)) {
    const nt = this.tri.length / 3;
    const tg = new Float64Array(3 * nt);
    for (let t = 0; t < nt; t++) {
      const v = [this.tri[3 * t], this.tri[3 * t + 1], this.tri[3 * t + 2]];
      const p = v.map((i) => this.point(i));
      const N = cross(sub(p[1], p[0]), sub(p[2], p[0]));
      const twoA = norm(N);
      const nHat = N.map((x) => x / twoA);
      // grad f = sum_k f_k (n x e_k) / 2A, e_k the edge opposite vertex k (counterclockwise)
      const g = [0, 0, 0];
      for (let k = 0; k < 3; k++) {
        const e = sub(p[(k + 2) % 3], p[(k + 1) % 3]);
        const c = cross(nHat, e);
        for (let d = 0; d < 3; d++) g[d] += (f[v[k]] * c[d]) / twoA;
      }
      tg.set(g, 3 * t);
    }
    for (let i = 0; i < this.n; i++) {
      const g = [0, 0, 0];
      let a = 0;
      for (const t of this.incident[i]) {
        for (let d = 0; d < 3; d++) g[d] += this.triArea[t] * tg[3 * t + d];
        a += this.triArea[t];
      }
      const r = this.point(i);
      const radial = dot(g, r) / a;
      for (let d = 0; d < 3; d++) out[3 * i + d] = g[d] / a - radial * r[d];
    }
    return out;
  }

  /** Nearest node to a unit vector, by greedy descent from node `start`. */
  nearest(x, start = 0) {
    let node = start, best = dot(this.point(node), x);
    for (;;) {
      let next = node;
      for (const j of this.neighbours[node]) {
        const d = dot(this.point(j), x);
        if (d > best) { best = d; next = j; }
      }
      if (next === node) return node;
      node = next;
    }
  }

  /**
   * Locate unit vector x: the triangle containing it and its barycentric
   * weights, as {nodes: [i, j, k], weights: [a, b, c]} (`hint`: a nearby node).
   * Reuse the result to interpolate several fields at the same point.
   */
  locate(x, hint = 0) {
    const node = this.nearest(x, hint);
    for (const t of this.incident[node]) {
      const v = [this.tri[3 * t], this.tri[3 * t + 1], this.tri[3 * t + 2]];
      const w = barycentric(x, this.point(v[0]), this.point(v[1]), this.point(v[2]));
      if (w && w[0] >= -1e-9 && w[1] >= -1e-9 && w[2] >= -1e-9) return { nodes: v, weights: w };
    }
    return { nodes: [node, node, node], weights: [1, 0, 0] }; // numerically on an edge of the fan
  }

  /** Linear interpolation of node values f at unit vector x (or at a `locate` result). */
  interpolate(f, x, hint = 0) {
    const { nodes, weights } = Array.isArray(x) ? this.locate(x, hint) : x;
    return weights[0] * f[nodes[0]] + weights[1] * f[nodes[1]] + weights[2] * f[nodes[2]];
  }
}

/** Barycentric coordinates of the ray through x in the plane of triangle (a, b, c). */
function barycentric(x, a, b, c) {
  const n = cross(sub(b, a), sub(c, a));
  const denom = dot(n, x);
  if (denom <= 0) return null;
  const s = dot(n, a) / denom; // scale x onto the triangle's plane
  const p = x.map((v) => v * s);
  const area = dot(n, n);
  return [dot(cross(sub(b, p), sub(c, p)), n) / area, dot(cross(sub(c, p), sub(a, p)), n) / area, dot(cross(sub(a, p), sub(b, p)), n) / area];
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);
