// Uniform-grid cell list for finding all pairs of points closer than `cutoff`
// (the standard short-range neighbour search of molecular dynamics).

export class CellList {
  /** @param position Float32Array of xyz triples, all within the cube [-1.1, 1.1]^3 */
  constructor(position, cutoff) {
    const n = position.length / 3;
    this.cutoff = cutoff;
    this.dim = Math.ceil(2.2 / cutoff);
    const { dim } = this;
    this.cellOf = new Int32Array(n);
    this.start = new Int32Array(dim ** 3 + 1);
    this.order = new Int32Array(n);

    const coord = (x) => Math.min(dim - 1, Math.max(0, Math.floor((x + 1.1) / cutoff)));
    for (let i = 0; i < n; i++) {
      const c = coord(position[3 * i]) + dim * (coord(position[3 * i + 1]) + dim * coord(position[3 * i + 2]));
      this.cellOf[i] = c;
      this.start[c + 1]++;
    }
    for (let c = 0; c < dim ** 3; c++) this.start[c + 1] += this.start[c];
    const cursor = this.start.slice(0, -1);
    for (let i = 0; i < n; i++) this.order[cursor[this.cellOf[i]]++] = i;
    this.position = position;
  }

  /** Call fn(i, j, dx, dy, dz, distance) once per unordered pair closer than the cutoff. */
  forEachPair(fn) {
    const { dim, cellOf, start, order, position: p, cutoff } = this;
    const cut2 = cutoff * cutoff;
    for (let i = 0; i < cellOf.length; i++) {
      const c = cellOf[i];
      const cx = c % dim, cy = Math.floor(c / dim) % dim, cz = Math.floor(c / (dim * dim));
      for (let z = Math.max(0, cz - 1); z <= Math.min(dim - 1, cz + 1); z++)
        for (let y = Math.max(0, cy - 1); y <= Math.min(dim - 1, cy + 1); y++)
          for (let x = Math.max(0, cx - 1); x <= Math.min(dim - 1, cx + 1); x++) {
            const cell = x + dim * (y + dim * z);
            for (let k = start[cell]; k < start[cell + 1]; k++) {
              const j = order[k];
              if (j <= i) continue;
              const dx = p[3 * i] - p[3 * j], dy = p[3 * i + 1] - p[3 * j + 1], dz = p[3 * i + 2] - p[3 * j + 2];
              const d2 = dx * dx + dy * dy + dz * dz;
              if (d2 < cut2) fn(i, j, dx, dy, dz, Math.sqrt(d2));
            }
          }
    }
  }
}
