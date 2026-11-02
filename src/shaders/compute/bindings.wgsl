// Bindings and helpers shared by every compute entry point.
// (Particle, Edge, ColumnProfile, GridNode and SimParams structs are generated from
// src/gpu/layout.js and src/sim/params.js and prepended to this file.)

const NONE: u32 = 0xFFFFFFFFu;

// Position corrections are accumulated with integer atomics in fixed point,
// because WGSL has no float atomics. 1e8 resolves 1e-8 of the unit sphere
// (6 cm; a bond strain of ~3e-7), and the largest corrections (an overlap of
// ~0.035 from each of ~50 neighbours) stay well inside i32.
const FIXED_POINT: f32 = 100000000.0;

// Stack of free slot indices. `requests` is used by two-phase allocation
// (columns_split reserves, crust_commit_splits pops; see split.wgsl).
struct FreeList { counter: atomic<u32>, requests: atomic<u32>, pad2: u32, pad3: u32, indices: array<u32> }
// Per-particle scratch: summed position and spin corrections (reset every
// iteration) and the particle's slot list: the edges touching it, sorted by
// the partner's angle around it (rebuilt every step; SLOTS in layout.js).
// slot_count is the true number of edges and may exceed MAX_SLOTS, which is
// an overflow (counted in solver.slot_overflow).
const MAX_SLOTS: u32 = 12u;
// Then the column's tessellation neighbours (columns_area) with the
// bending weights c = shared cell boundary / distance (flexure.wgsl).
struct ParticleDelta {
    dx: atomic<i32>, dy: atomic<i32>, dz: atomic<i32>, count: atomic<i32>,
    slot_count: atomic<u32>, cell_count: u32, dspin: atomic<i32>, pad3: u32,
    slots: array<u32, 12>,
    cell_nbr: array<u32, 12>,
    cell_coef: array<f32, 12>,
    // Position corrections from tangential rows (phase G), kept apart: they
    // are not extrapolated by Chebyshev (crust_solve_edges, solver.wgsl).
    tdx: atomic<i32>, tdy: atomic<i32>, tdz: atomic<i32>, pad4: u32,
    // Load of the slabs hanging from this column, N, as f32 bits (atomic_add_f32);
    // copied to Particle.child_load and cleared at the end of the step.
    // Likewise the down-dip weight (W sin dip, unsupported) and the drag
    // coefficient (W / w_free, N s / m) of the slabs hanging from it
    // (Particle.child_pull, child_drag).
    child_load: atomic<u32>, child_pull: atomic<u32>, child_drag: atomic<u32>, pad7: u32,
}

@group(0) @binding(0) var<storage, read_write> grid: array<GridNode>;
@group(0) @binding(1) var<storage, read> adjacency: array<u32>;   // 6 neighbours per node, NONE-padded
@group(0) @binding(2) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(3) var<storage, read_write> edges: array<Edge>;
@group(0) @binding(4) var<storage, read_write> particle_pool: FreeList;
@group(0) @binding(5) var<storage, read_write> edge_pool: FreeList;
@group(0) @binding(6) var<storage, read_write> hash_grid: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> particle_deltas: array<ParticleDelta>;

// Solver iteration counter; counters since load, read by tests and the debug
// panel (SOLVER_HEADER in simulation.js): capacity overflows, tessellation
// cells that touched their clip polygon (coverage not exact there), the crust
// volume budget (design 4.8) as two 64-bit fixed-point sums (lo, hi words;
// VOLUME_FIXED units per km x unit-sphere area), continental breakups; then
// slot high-water marks (one past the highest particle / edge slot ever
// used; the free lists hand out the lowest free slot first, so these track
// the live problem size) and the indirect dispatch sizes and draw instance
// count made from them (solver_dispatch_sizes); then the net rotation sums
// and the rotation removed this step (crust_net_rotation_*, solver.wgsl);
// then the Chebyshev weight for each iteration (written by simulation.js; see
// chebyshevWeights).
struct SolverState {
    iteration: u32, edge_pool_empty: atomic<u32>, slot_overflow: atomic<u32>, clipped_cells: atomic<u32>,
    budget: array<atomic<u32>, 4>,   // source lo, hi; sink lo, hi
    breakups: atomic<u32>, slabs_lost: atomic<u32>, particle_high: atomic<u32>, edge_high: atomic<u32>,
    dispatch: array<u32, 6>,   // workgroups x, y, z over particle slots, then over edge slots
    draw_instances: u32, pad19: u32,
    // Area-weighted sums over surface columns, f32 bits: L = sum a x cross u
    // (3), then I = sum a (1 - x x^T) as xx, yy, zz, xy, xz, yz (6).
    net_rotation_sums: array<atomic<u32>, 9>,
    net_rotation: array<f32, 3>,   // rotation vector removed this step (rad)
    omega: array<f32>,
}
const BUDGET_SOURCE: u32 = 0u;
const BUDGET_SINK: u32 = 2u;
const VOLUME_FIXED: f32 = 1e12;
@group(0) @binding(8) var<storage, read_write> solver: SolverState;
// Per-column vertical profiles (design 4.0), recomputed every step by column_profile (profile.wgsl).
@group(0) @binding(9) var<storage, read_write> profiles: array<ColumnProfile>;

// Float add on an atomic word holding f32 bits (compare-and-swap loop).
fn atomic_add_f32(p: ptr<storage, atomic<u32>, read_write>, v: f32) {
    var old = atomicLoad(p);
    loop {
        let r = atomicCompareExchangeWeak(p, old, bitcast<u32>(bitcast<f32>(old) + v));
        if (r.exchanged) { break; }
        old = r.old_value;
    }
}

// Add a (non-negative) crust volume to a budget sum: 64-bit fixed point,
// split into words, with the carry from the low word.
fn add_budget(slot: u32, v: f32) {
    let q = max(v, 0.0) * VOLUME_FIXED;
    let hi = floor(q / 4294967296.0);
    let lo = u32(q - hi * 4294967296.0);
    let old = atomicAdd(&solver.budget[slot], lo);
    let carry = select(0u, 1u, old + lo < old);
    if (hi > 0.0 || carry > 0u) { atomicAdd(&solver.budget[slot + 1u], u32(hi) + carry); }
}

// A slot handed out by a free list: raise the high-water mark.
fn note_particle_slot(i: u32) { atomicMax(&solver.particle_high, i + 1u); }
fn note_edge_slot(e: u32) { atomicMax(&solver.edge_high, e + 1u); }

fn chebyshev_omega() -> f32 {
    return solver.omega[min(solver.iteration, arrayLength(&solver.omega) - 1u)];
}

@group(1) @binding(1) var<uniform> sim: SimParams;

fn to_fixed(v: vec3f) -> vec3i {
    return vec3i(round(v * FIXED_POINT));
}

fn add_delta_fixed(i: u32, d: vec3i, count: i32) {
    atomicAdd(&particle_deltas[i].dx, d.x);
    atomicAdd(&particle_deltas[i].dy, d.y);
    atomicAdd(&particle_deltas[i].dz, d.z);
    atomicAdd(&particle_deltas[i].count, count);
}

fn add_delta(i: u32, d: vec3f, count: i32) {
    add_delta_fixed(i, to_fixed(d), count);
}

fn add_tangential_delta(i: u32, d: vec3f) {
    let f = to_fixed(d);
    atomicAdd(&particle_deltas[i].tdx, f.x);
    atomicAdd(&particle_deltas[i].tdy, f.y);
    atomicAdd(&particle_deltas[i].tdz, f.z);
}

// Spin corrections (rad), same fixed point.
fn add_spin(i: u32, v: f32) {
    atomicAdd(&particle_deltas[i].dspin, i32(round(v * FIXED_POINT)));
}

// Return an index to a free list (used when particles melt or edges leave range).
// Pushes only ever run in dispatches without pops, so the slot written is
// never being read concurrently.
fn release(pool_is_particles: bool, idx: u32) {
    if (pool_is_particles) {
        let n = atomicAdd(&particle_pool.counter, 1u);
        if (n < sim.max_particles) { particle_pool.indices[n] = idx; } else { atomicSub(&particle_pool.counter, 1u); }
    } else {
        let n = atomicAdd(&edge_pool.counter, 1u);
        if (n < sim.max_edges) { edge_pool.indices[n] = idx; } else { atomicSub(&edge_pool.counter, 1u); }
    }
}

// Uniform 3D hash grid over the cube [-1.5, 1.5]^3.
fn hash_cell(p: vec3f) -> vec3i {
    return vec3i(clamp((p + vec3f(1.5)) / 3.0, vec3f(0.0), vec3f(0.999)) * f32(sim.hash_grid_dim));
}

fn hash_index(c: vec3i) -> u32 {
    let d = sim.hash_grid_dim;
    return u32(c.x) + u32(c.y) * d + u32(c.z) * d * d;
}

// Separation of an edge's two columns (b - a). Map view (unit sphere) between
// surface columns; in 3D when either is a slab, so a sinking slab stretches
// its edges and pulls on the plate behind it (design 6).
const EARTH_RADIUS_KM: f32 = 6371.0;
fn edge_offset(a: u32, b: u32, xa: vec3f, xb: vec3f) -> vec3f {
    if (particles[a].state == 2u || particles[b].state == 2u) {
        return xb * (1.0 + particles[b].elevation / EARTH_RADIUS_KM) - xa * (1.0 + particles[a].elevation / EARTH_RADIUS_KM);
    }
    return xb - xa;
}

// Reference shape (design 4.2, phase G; src/columns/shape.js): the column's
// radius toward direction d (any 3D vector) is its apothem stretched by
// U = exp(L), L = [[p, q], [q, -p]] in its body frame (body_axis, x x body_axis)
// at unit position x: apothem / sqrt(n^T exp(-2L) n).
fn shape_radius_factor(p: f32, q: f32, n: vec2f) -> f32 {
    let s = length(vec2f(p, q));
    let f = select(2.0, sinh(2.0 * s) / s, s > 1e-6);
    let nln = p * (n.x * n.x - n.y * n.y) + 2.0 * q * n.x * n.y;
    return 1.0 / sqrt(cosh(2.0 * s) - f * nln);
}
fn body_frame(i: u32, x: vec3f, d: vec3f) -> vec2f {
    let b = particles[i].body_axis.xyz;
    return vec2f(dot(d, b), dot(d, cross(x, b)));
}
fn column_radius(i: u32, x: vec3f, d: vec3f) -> f32 {
    let p = particles[i].shape_p;
    let q = particles[i].shape_q;
    if (p == 0.0 && q == 0.0) { return particles[i].apothem; }
    let u = body_frame(i, x, d);
    let l = length(u);
    if (!(l > 0.0)) { return particles[i].apothem; }
    return particles[i].apothem * shape_radius_factor(p, q, u / l);
}
// Rest length of a pair: the two radii along it (positions xa, xb).
fn contact_length(a: u32, b: u32, xa: vec3f, xb: vec3f) -> f32 {
    let d = xb - xa;
    return column_radius(a, xa, d) + column_radius(b, xb, -d);
}
// Tangential channel (phase G; tangentialDisplacement in src/edges/law.js).
// Rotation of the pair line since the last update (the edge's stored
// direction -> the current iterate), about the outward radial at the pair,
// CCW positive. Like `length`, the stored direction is the edge's memory: it
// is reset where geometry jumps (splits), so a jump slides nothing.
fn pair_rotation(c: Edge) -> f32 {
    let a = c.particle_a;
    let b = c.particle_b;
    let d0 = vec3f(c.direction_x, c.direction_y, c.direction_z);
    let xa = particles[a].predicted_pos.xyz;
    let xb = particles[b].predicted_pos.xyz;
    let d = xb - xa;
    let up = normalize(xa + xb);
    let y = dot(cross(d0, d), up);
    let x = dot(d0, d);
    // atan(y/x) where the pair turned less than a right angle: SwiftShader's
    // atan2 returns pi for y = -0 (a pair that did not move at all).
    if (x > 0.0) { return atan(y / x); }
    return atan2(y, x);
}
// Sliding of the faces this step (unit sphere): L0 psi - l_a spin_a - l_b spin_b,
// levers from the last-update distance L0 (edge.length) split at the contact point.
fn tangential_displacement(c: Edge) -> f32 {
    let L0 = c.length;
    return L0 * (pair_rotation(c) - c.lever_a * particles[c.particle_a].spin - (1.0 - c.lever_a) * particles[c.particle_b].spin);
}

// An edge's rest length this step (set by edges_law, law.wgsl).
fn rest_length(c: Edge) -> f32 {
    return c.rest;
}

