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
// Fixed-size scalar arrays, e.g. 'array<f32, 8>' (stride 4: storage buffers only).
const arrayType = (type) => /^array<(f32|u32|i32), (\d+)>$/.exec(type);
const typeInfo = (type) => {
  const m = arrayType(type);
  return m ? { size: 4 * Number(m[2]), align: 4 } : TYPES[type];
};

export function defineStruct(name, fields) {
  let offset = 0;
  let maxAlign = 4;
  const offsets = {};
  for (const [field, type] of fields) {
    const { size, align } = typeInfo(type);
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
    const arr = arrayType(type);
    if (type === 'vec4f' || arr) {
      const len = arr ? Number(arr[2]) : 4;
      const src = arr && arr[1] !== 'f32' ? (arr[1] === 'u32' ? u32 : new Int32Array(buffer)) : f32;
      const a = new src.constructor(len * count);
      for (let i = 0; i < count; i++) a.set(src.subarray(i * struct.words + o, i * struct.words + o + len), len * i);
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
  ['position', 'vec4f'],        // unit-sphere direction of the column (w unused); its radius is `elevation`
  ['predicted_pos', 'vec4f'],   // position-based-dynamics predicted position
  ['velocity', 'vec4f'],        // radians / Myr, tangent to sphere
  ['iter_start', 'vec4f'],      // solver iterate x_k at the start of the current iteration (Chebyshev)
  ['iter_prev', 'vec4f'],       // solver iterate x_{k-1} (Chebyshev)
  ['cell_shape', 'vec4f'],      // tessellation cell (derived, design 4.8): centroid (x, y) and major-axis
                                //   angle (z) in the gnomonic tangent basis, sqrt of the major second moment (w)
  ['slab_trailing', 'vec4f'],   // slab: unit direction to the plate behind it (integrity-weighted surface partners), w = 1 if any
  ['thickness_km', 'f32'],      // crust thickness, derived: crust_volume / cell_area (slabs: / area_ref)
  ['composition_density', 'f32'], // kg/m^3 at mantle temperature
  ['temperature', 'f32'],       // K
  ['age', 'f32'],               // Myr
  ['area_ref', 'f32'],          // stress-free (reference) area, unit sphere = 2*sqrt(3)*apothem^2 (hexagon)
  ['apothem', 'f32'],           // unit-sphere length; sets rest lengths and range
  ['state', 'u32'],             // 1 = surface, 2 = slab (sinking below the surface)
  ['is_active', 'u32'],
  ['next_idx', 'u32'],          // linked list through the collision hash grid
  ['density', 'f32'],           // cached thermally-corrected density
  ['elevation', 'f32'],         // free radial coordinate: top of the column, km above the mantle datum
  ['split_child', 'u32'],       // index of child created this frame, else NONE
  ['mantle_node', 'u32'],       // nearest mantle grid node, updated by a local walk each step
  ['next_state', 'u32'],        // state requested during contact detection, applied at the end of the step
  ['edge_count', 'u32'],        // edges touching the particle this step; scales Jacobi updates
  // Area, volume and melt (design 4.8, phase E):
  ['crust_volume', 'f32'],      // km x unit-sphere area; changes only at sources (melt) and sinks (slabs)
  ['thermal_age_myr', 'f32'],   // sets the lithosphere (2.32 sqrt(kappa t), capped); scaled by stretching
  ['cell_area', 'f32'],         // actual area: the column's tessellation cell (derived; 0 = just split)
  ['cell_area_prev', 'f32'],    // last step's cell_area (dilatation; 0 = none this step)
  ['volume_added', 'f32'],      // melt added this step (diagnostic)
  ['cell_clipped', 'u32'],      // 1 = the cell touched its clip polygon (a free side): its area is not real, it never splits
  // Subduction, slabs, arcs, bending (design 4.3, 6; phase F):
  ['slab_neighbours', 'u32'],   // edges to slab columns this step (subduction propagation, polarity)
  ['geometry_reset', 'u32'],    // 1 = split or subducted this step: its edges restart their length measure
  ['arc_count', 'u32'],         // slab: surface columns within arc reach this step
  ['arc_supply', 'f32'],        // slab: arc volume it supplies this step (km x unit area)
  ['flexural_rigidity', 'f32'], // D, N m (derived each step)
  ['bending_moment', 'f32'],    // D x laplacian of elevation (solver scratch)
  ['elevation_start', 'f32'],   // elevation at the start of the vertical solve
  // Orientation and shape (design 4.2, phase G; src/columns/shape.js):
  ['body_axis', 'vec4f'],       // unit tangent: the column's body x-axis (xyz; w unused), carried and spun with it
  ['shape_p', 'f32'],           // reference shape: traceless log-stretch L = [[p, q], [q, -p]] in the body
  ['shape_q', 'f32'],           //   frame (body_axis, position x body_axis); aspect ratio exp(2 |(p, q)|)
  ['spin', 'f32'],              // rotation this step about the outward radial, rad, CCW (solver iterate; plain Jacobi)
  ['rotation_total', 'f32'],    // vertical-axis rotation since load, rad (diagnostic: paleomagnetic declination)
  // Trench hinge and hanging slabs (design 6; src/columns/hinge.js):
  ['hinge_angle', 'f32'],       // surface: bend of the flap beyond the hinge, rad (+ down); slab: its dip
  ['hinge_plastic', 'f32'],     // plastic part of the bend, rad (state)
  ['hinge_flap_km', 'f32'],     // flap length beyond the hinge (derived; 0 = no hinge)
  ['hinge_underthrust_km', 'f32'], // part of the column past the trench line (derived)
  ['hinge_partner', 'u32'],     // the column the hinge faces: overriding neighbour or hanging slab (derived), else NONE
  ['attached_to', 'u32'],       // slab: what it hangs from (NONE = free); surface: the column behind it (derived)
  ['slab_child', 'u32'],        // a slab hanging from this column this step (derived), else NONE
  ['slab_extent_km', 'f32'],    // slab: depth range from its top to its bottom
  ['child_load', 'f32'],        // load hanging from this column, N (last step)
  ['hang_load', 'f32'],         // slab: load it passed up last step, N (under-relaxed, thermal.wgsl)
]);

/**
 * An edge: the interface between two neighbouring columns (design 3.2). There
 * is one kind. Whether it acts as a weld, a fault or a gravel contact follows
 * from its state (integrity) and the two columns, not from a type field.
 *
 * Existence is geometric: an edge exists while its pair is in range (map-view
 * gap below edge_range_margin of the apothem sum, and overlapping depth
 * intervals). Edges are keyed by pair and persist across steps in a
 * free-listed array; per-particle slot lists (SLOTS) are an index rebuilt
 * every step. Rest length is the sum of the two columns' radii along the
 * pair (their reference shapes, src/columns/shape.js), set each step by edges_law.
 */
export const Edge = defineStruct('Edge', [
  ['particle_a', 'u32'],
  ['particle_b', 'u32'],
  ['is_active', 'u32'],
  ['integrity', 'f32'],         // 0..1: 0 = no cohesion (pushes only), 1 = intact
  ['lambda', 'f32'],            // XPBD multiplier accumulated over this step's iterations
  ['lambda_prev', 'f32'],       // multiplier before the current iteration (Chebyshev acceleration)
  ['stress_normal_mpa', 'f32'], // diagnostic: depth-averaged normal stress, + tension, - compression
  ['stress_shear_mpa', 'f32'],  // diagnostic: depth-averaged tangential stress, signed (b moving +t relative to a is +)
  // Edge law (design 4.5, phase C; src/edges/law.js). State:
  ['length', 'f32'],            // |x_b - x_a| (unit sphere) at the last update: strain increments are
                                //   measured from it, so a change of rest length (area_ref) adds no stress
  ['slice_stress_mpa', 'array<f32, 8>'], // Maxwell stress per depth slice of the shared interval, top down
  // Diagnostics of the last update (design 4.7, phase D):
  ['strain', 'f32'],            // (|x_b - x_a| - rest) / rest
  ['inelastic_strain', 'f32'],  // strain increment not stored elastically (feeds area_ref)
  ['work_mpa_km', 'f32'],       // plastic work (feeds damage)
  // Derived each step by edges_law, for the solve:
  ['stiffness', 'f32'],         // effective stiffness / (E * stiffness_reference_km), in [0, 1]
  ['reference_gap', 'f32'],     // gap C (unit sphere) at which the linearised force is zero
  ['lambda_min', 'f32'],        // multiplier bounds: depth-integrated tension and compression caps
  ['lambda_max', 'f32'],
  ['overlap_km', 'f32'],        // thickness of the depth interval the two columns share
  ['timer', 'f32'],             // sustained-compression accumulator (subduction, suturing)
  ['born_myr', 'f32'],          // creation time
  // Tangential channel and face rolling (design 4.5, phase G; src/edges/law.js). State:
  ['slice_shear_mpa', 'array<f32, 8>'], // tangential Maxwell stress per depth slice (+: b displaced along +t, t = up x n)
  ['direction_x', 'f32'],       // x_b - x_a at the last update (with `length`): sliding is measured from it
  ['direction_y', 'f32'],
  ['direction_z', 'f32'],
  ['moment', 'f32'],            // face moment against relative rotation, MPa km x unit-sphere length
  // Derived each step by edges_law, for the solve (t = tangential, r = rolling):
  ['rest', 'f32'],              // rest length: the two columns' radii along the pair (shape-dependent)
  ['lever_a', 'f32'],           // contact point: fraction r_a / (r_a + r_b) of the pair distance from a
  ['face_length', 'f32'],       // shared tessellation boundary (unit sphere; 0 = not cell neighbours)
  ['stiffness_t', 'f32'],
  ['reference_t', 'f32'],       // tangential offset (unit sphere) at zero force
  ['lambda_t', 'f32'],           // (tangential and rolling rows are plain Jacobi: no previous multipliers)
  ['lambda_t_max', 'f32'],      // symmetric bound: depth-integrated shear cap (friction / flow stress)
  ['stiffness_r', 'f32'],
  ['reference_r', 'f32'],       // relative rotation (rad) at zero moment
  ['lambda_r', 'f32'],
  ['lambda_r_max', 'f32'],      // symmetric bound: the kern rule, (compression + tension cap) x face / 6
  // Diagnostics of the last update:
  ['inelastic_shear', 'f32'],   // tangential strain not stored, where distributed (ductile): feeds the shape
  ['slip', 'f32'],              // tangential strain slipped at the friction cap (brittle): a fault, not the shape
]);

/**
 * Per-particle slot list, inside the `deltas` buffer (ParticleDelta in
 * bindings.wgsl): `count` edges touching the particle (may exceed `capacity`,
 * which flags an overflow), then up to `capacity` edge ids sorted by the
 * angle of the partner around the particle. Rebuilt every step.
 */
/**
 * A column's vertical profile (design 4.0), recomputed every step from the
 * particle's state by column_profile (profile.wgsl; CPU reference
 * src/columns/profile.js). Derived, not state: never saved. Depths are km
 * below the column top; per-slice arrays run from the top down, the first
 * CRUST_SLICES over the crust.
 */
export const ColumnProfile = defineStruct('ColumnProfile', [
  ['lithosphere_km', 'f32'],        // base of the lithosphere (90% isotherm)
  ['column_km', 'f32'],             // slices span [0, max(lithosphere, crust)]
  ['freeboard_km', 'f32'],          // Airy height without water loading
  ['airy_elevation_km', 'f32'],     // equilibrium elevation (water-loaded below sea level)
  ['gpe_mpa_km', 'f32'],            // gravitational potential energy vs the reference mantle column (1e9 N/m)
  ['heat_flow_mw_m2', 'f32'],       // surface heat flow
  ['moho_temp_k', 'f32'],
  ['brittle_ductile_km', 'f32'],    // shallowest brittle-ductile transition (integrity 1)
  ['depth_km', 'array<f32, 8>'],    // slice midpoints
  ['thickness_km', 'array<f32, 8>'],
  ['temp_k', 'array<f32, 8>'],
  ['density', 'array<f32, 8>'],     // kg/m^3
  ['pressure_mpa', 'array<f32, 8>'], // lithostatic, from the sea surface (or the column top on land)
  ['log10_viscosity', 'array<f32, 8>'], // Pa s, dislocation creep at the reference strain rate
  ['ductile_mpa', 'array<f32, 8>'], // ductile shear strength (capped at 1e6)
]);

export const SLOTS = { wordsPerParticle: 52, countWord: 4, firstSlotWord: 8, capacity: 12 };

/** One node of the mantle grid; written from the CPU mantle model (src/mantle/). */
export const GridNode = defineStruct('GridNode', [
  ['position', 'vec4f'],
  ['velocity', 'vec4f'],        // flow beneath the plates, rad/Myr, tangent
  ['buoyancy', 'f32'],          // mid-mantle temperature anomaly (positive = hot), model units
  ['divergence', 'f32'],        // of the flow, 1/Myr (positive over upwellings)
]);
