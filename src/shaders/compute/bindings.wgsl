// Bindings and helpers shared by every compute entry point.
// (Particle, Constraint, GridNode and SimParams structs are generated from
// src/gpu/layout.js and src/sim/params.js and prepended to this file.)

const NONE: u32 = 0xFFFFFFFFu;

// Position corrections are accumulated with integer atomics in fixed point,
// because WGSL has no float atomics. Integer addition is order-independent,
// so the summed correction is deterministic.
const FIXED_POINT: f32 = 1000000.0;

struct FreeList { counter: atomic<u32>, pad1: u32, pad2: u32, pad3: u32, indices: array<u32> }
struct ParticleDelta { dx: atomic<i32>, dy: atomic<i32>, dz: atomic<i32>, count: atomic<i32> }

@group(0) @binding(0) var<storage, read_write> grid: array<GridNode>;
@group(0) @binding(1) var<storage, read> adjacency: array<u32>;   // 6 neighbours per node, NONE-padded
@group(0) @binding(2) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(3) var<storage, read_write> constraints: array<Constraint>;
@group(0) @binding(4) var<storage, read_write> particle_pool: FreeList;
@group(0) @binding(5) var<storage, read_write> constraint_pool: FreeList;
@group(0) @binding(6) var<storage, read_write> hash_grid: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> particle_deltas: array<ParticleDelta>;

@group(1) @binding(1) var<uniform> sim: SimParams;

// Brute-force nearest mantle node (O(num_nodes) per call; see README to-do).
fn nearest_node(p: vec3f) -> u32 {
    var best = 0u;
    var best_dist = 1000.0;
    for (var i = 0u; i < sim.num_nodes; i++) {
        let d = distance(p, grid[i].position.xyz);
        if (d < best_dist) { best_dist = d; best = i; }
    }
    return best;
}

fn add_delta(i: u32, d: vec3f, count: i32) {
    atomicAdd(&particle_deltas[i].dx, i32(d.x * FIXED_POINT));
    atomicAdd(&particle_deltas[i].dy, i32(d.y * FIXED_POINT));
    atomicAdd(&particle_deltas[i].dz, i32(d.z * FIXED_POINT));
    atomicAdd(&particle_deltas[i].count, count);
}

// Return an index to a free list (used when particles melt or bonds die).
fn release(pool_is_particles: bool, idx: u32) {
    if (pool_is_particles) {
        let n = atomicAdd(&particle_pool.counter, 1u);
        if (n < sim.max_particles) { particle_pool.indices[n] = idx; } else { atomicSub(&particle_pool.counter, 1u); }
    } else {
        let n = atomicAdd(&constraint_pool.counter, 1u);
        if (n < sim.max_constraints) { constraint_pool.indices[n] = idx; } else { atomicSub(&constraint_pool.counter, 1u); }
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

// Arrhenius factor exp(-Ea / (R T)), used by healing.
fn arrhenius(temp: f32) -> f32 {
    let exponent = -sim.activation_energy / (sim.gas_constant * max(temp, 1.0));
    return select(0.0, exp(exponent), exponent > -50.0);
}

fn bond_temperature(c: Constraint) -> f32 {
    return max((particles[c.particle_a].temperature + particles[c.particle_b].temperature) * 0.5, 1.0);
}
