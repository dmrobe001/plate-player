// Icosphere: an icosahedron whose triangles are recursively split in four and
// pushed back onto the unit sphere. Its vertices are the mantle grid nodes.
// Vertex count = 10 * 4^subdivisions + 2 (12 of them have 5 neighbours, the rest 6).

import { normalize } from './vec.js';

export function createIcosphere(subdivisions) {
  const t = (1 + Math.sqrt(5)) / 2;
  const vertices = [
    [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
    [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
    [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
  ].map(normalize);

  let faces = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
  ];

  const midpointCache = new Map();
  function midpoint(i, j) {
    const key = Math.min(i, j) * 1e6 + Math.max(i, j);
    let m = midpointCache.get(key);
    if (m === undefined) {
      const a = vertices[i], b = vertices[j];
      m = vertices.push(normalize([a[0] + b[0], a[1] + b[1], a[2] + b[2]])) - 1;
      midpointCache.set(key, m);
    }
    return m;
  }

  for (let s = 0; s < subdivisions; s++) {
    faces = faces.flatMap(([a, b, c]) => {
      const ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
      return [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]];
    });
  }

  const neighbours = Array.from(vertices, () => new Set());
  for (const [a, b, c] of faces) {
    neighbours[a].add(b).add(c);
    neighbours[b].add(a).add(c);
    neighbours[c].add(a).add(b);
  }

  return {
    vertices,                                  // [[x, y, z], ...] on the unit sphere
    neighbours: neighbours.map((s) => [...s]), // adjacency lists (5 or 6 entries)
    indices: Uint32Array.from(faces.flat()),   // triangle list for drawing
  };
}
