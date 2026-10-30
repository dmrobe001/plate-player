// GPU struct layouts, declared once.
//
// Each struct is a list of [field, type] pairs. From that list we derive both
// the WGSL struct text that the shaders see and the word offsets that the
// JavaScript side uses to fill buffers, so the two cannot drift apart.
// Layout rules follow the WGSL spec for storage/uniform buffers: scalars align
// to 4 bytes, vec4f to 16, and the struct size rounds up to its largest
// alignment.

const TYPES = {
  f32: { size: 4, align: 4 },
  u32: { size: 4, align: 4 },
  i32: { size: 4, align: 4 },
  vec4f: { size: 16, align: 16 },
};

export function defineStruct(name, fields) {
  let offset = 0;
  let maxAlign = 4;
  const offsets = {};
  for (const [field, type] of fields) {
    const { size, align } = TYPES[type];
    offset = Math.ceil(offset / align) * align;
    offsets[field] = offset / 4; // in 32-bit words
    offset += size;
    maxAlign = Math.max(maxAlign, align);
  }
  const byteSize = Math.ceil(offset / maxAlign) * maxAlign;
  return {
    name,
    fields,
    offsets,
    byteSize,
    words: byteSize / 4,
    /** WGSL source for this struct; `overrides` swaps field types, e.g. to atomic<i32>. */
    wgsl(overrides = {}) {
      const body = fields.map(([f, t]) => `    ${f}: ${overrides[f] ?? t},`).join('\n');
      return `struct ${name} {\n${body}\n}\n`;
    },
  };
}

export const NONE = 0xffffffff; // "null index" sentinel shared with WGSL

/**
 * Decode `count` structs from an ArrayBuffer into one typed array per field
 * (vec4f fields become Float32Array of length 4 * count).
 */
export function unpack(struct, buffer, count) {
  const f32 = new Float32Array(buffer);
  const u32 = new Uint32Array(buffer);
  const out = {};
  for (const [field, type] of struct.fields) {
    const o = struct.offsets[field];
    if (type === 'vec4f') {
      const a = new Float32Array(4 * count);
      for (let i = 0; i < count; i++) a.set(f32.subarray(i * struct.words + o, i * struct.words + o + 4), 4 * i);
      out[field] = a;
    } else {
      const a = type === 'f32' ? new Float32Array(count) : type === 'i32' ? new Int32Array(count) : new Uint32Array(count);
      const src = type === 'f32' ? f32 : u32;
      for (let i = 0; i < count; i++) a[i] = src[i * struct.words + o];
      out[field] = a;
    }
  }
  return out;
}

/** Plain-JSON description of a struct layout, for record manifests. */
export function describe(struct) {
  return {
    byte_size: struct.byteSize,
    fields: struct.fields.map(([f, t]) => ({ name: f, type: t, byte_offset: 4 * struct.offsets[f] })),
  };
}

/** One crust particle: a hexagonal column of crust, drawn as a disc of radius `apothem`. */
export const Particle = defineStruct('Particle', [
  ['position', 'vec4f'],        // unit-sphere position (w unused)
  ['predicted_pos', 'vec4f'],   // position-based-dynamics predicted position
  ['velocity', 'vec4f'],        // radians / Myr, tangent to sphere
  ['thickness_km', 'f32'],
  ['composition_density', 'f32'], // kg/m^3 at mantle temperature
  ['temperature', 'f32'],       // K
  ['age', 'f32'],               // Myr
  ['surface_area', 'f32'],      // unit-sphere area = 2*sqrt(3)*apothem^2 (hexagon)
  ['apothem', 'f32'],           // unit-sphere length
  ['state', 'u32'],             // 1 = surface, 2 = subducting slab
  ['is_active', 'u32'],
  ['next_idx', 'u32'],          // linked list through the collision hash grid
  ['density', 'f32'],           // cached thermally-corrected density
  ['elevation', 'f32'],         // cached isostatic elevation, km
  ['split_child', 'u32'],       // index of child created this frame, else NONE
  ['mantle_node', 'u32'],       // nearest mantle grid node, updated by a local walk each step
  ['next_state', 'u32'],        // state requested during collisions, applied afterwards
]);

/** A bond between two particles. */
export const Constraint = defineStruct('Constraint', [
  ['particle_a', 'u32'],
  ['particle_b', 'u32'],
  ['rest_length', 'f32'],
  ['compliance', 'f32'],        // currently unused by the solver
  ['yield_stress_mpa', 'f32'],
  ['is_active', 'u32'],
  ['damage', 'f32'],            // 0 intact .. 1 fully damaged
  ['stress_normal_mpa', 'f32'], // diagnostic: + tension, - compression
  ['stress_shear_mpa', 'f32'],  // diagnostic: magnitude
  ['_pad', 'f32'],
]);

/** One vertex of the icosphere that carries the mantle convection field. */
export const GridNode = defineStruct('GridNode', [
  ['position', 'vec4f'],
  ['velocity', 'vec4f'],
  ['psi', 'f32'],               // Swift-Hohenberg order parameter
  ['laplacian_psi', 'f32'],
  ['thermal_blanket', 'f32'],
  ['thermal_atomic', 'i32'],    // atomic<i32> in compute shaders
]);
