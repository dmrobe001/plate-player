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
// Per-particle scratch: summed position corrections (reset every
// iteration) and the particle's slot list: the edges touching it, sorted by
// the partner's angle around it (rebuilt every step; SLOTS in layout.js).
// slot_count is the true number of edges and may exceed MAX_SLOTS, which is
// an overflow (counted in solver.slot_overflow).
const MAX_SLOTS: u32 = 12u;
// Then the column's tessellation neighbours (columns_area) with the
// bending weights c = shared cell boundary / distance (flexure.wgsl).
struct ParticleDelta {
    dx: atomic<i32>, dy: atomic<i32>, dz: atomic<i32>, count: atomic<i32>,
    slot_count: atomic<u32>, cell_count: u32, pad2: u32, pad3: u32,
    slots: array<u32, 12>,
    cell_nbr: array<u32, 12>,
    cell_coef: array<f32, 12>,
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
// the Chebyshev weight for each iteration (written by simulation.js; see
// chebyshevWeights).
struct SolverState {
    iteration: u32, edge_pool_empty: atomic<u32>, slot_overflow: atomic<u32>, clipped_cells: atomic<u32>,
    budget: array<atomic<u32>, 4>,   // source lo, hi; sink lo, hi
    breakups: atomic<u32>, pad9: u32, pad10: u32, pad11: u32,
    omega: array<f32>,
}
const BUDGET_SOURCE: u32 = 0u;
const BUDGET_SINK: u32 = 2u;
const VOLUME_FIXED: f32 = 1e12;
@group(0) @binding(8) var<storage, read_write> solver: SolverState;
// Per-column vertical profiles (design 4.0), recomputed every step by column_profile (profile.wgsl).
@group(0) @binding(9) var<storage, read_write> profiles: array<ColumnProfile>;

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

// An edge's rest length is always the sum of its particles' apothems, whatever
// changed them (growth, splitting, ...).
fn rest_length(c: Edge) -> f32 {
    return particles[c.particle_a].apothem + particles[c.particle_b].apothem;
}

