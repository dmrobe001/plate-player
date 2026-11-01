// Edge topology (design 3.2, phase A). CPU reference: src/edges/topology.js.
//
// An edge exists between two columns exactly while they are in range: map-view
// gap below edge_range_margin of the apothem sum, and overlapping depth
// intervals. Edges are keyed by pair and keep their state (integrity, stored
// stresses, anchors, timers) while the pair stays in range. Each step, on the
// predicted positions:
//   edges_begin          clear every particle's slot list
//   edges_prune          drop edges whose pair left range (or whose particle
//                        died); list the survivors at both ends
//   edges_note_listed    remember how many survivors each particle lists
//   edges_create         each particle i creates edges to in-range partners
//                        j > i that it does not already list
//   edges_finish         final edge counts; sort each slot list by angle
//
// Why no pair is ever created twice: pair (i, j), i < j, is created only by
// thread i, after checking i's surviving edges. Edges that other threads add
// to i's list during edges_create have partners k < i, so they cannot be
// (i, j). If i's list overflowed (more than MAX_SLOTS edges), i cannot check
// and creates nothing until it is back under capacity.

// Radial extent of a column, km in the elevation datum: from its top down
// through the lithosphere that moves with it.
fn column_top(i: u32) -> f32 { return particles[i].elevation; }
fn column_bottom(i: u32) -> f32 { return particles[i].elevation - lithosphere_thickness_km(i); }

// Map-view overlap on predicted positions (negative = gap).
fn pair_overlap(i: u32, j: u32) -> f32 {
    return particles[i].apothem + particles[j].apothem - distance(particles[i].predicted_pos.xyz, particles[j].predicted_pos.xyz);
}

fn in_range(i: u32, j: u32) -> bool {
    let contact = particles[i].apothem + particles[j].apothem;
    let depth_overlap = min(column_top(i), column_top(j)) > max(column_bottom(i), column_bottom(j));
    return pair_overlap(i, j) > -sim.edge_range_margin * contact && depth_overlap;
}

fn other_end(e: u32, i: u32) -> u32 {
    return select(edges[e].particle_a, edges[e].particle_b, edges[e].particle_a == i);
}

fn list_edge(i: u32, e: u32) {
    let k = atomicAdd(&particle_deltas[i].slot_count, 1u);
    if (k < MAX_SLOTS) { particle_deltas[i].slots[k] = e; } else { atomicAdd(&solver.slot_overflow, 1u); }
}

// Pop one edge slot. Only used in a dispatch with no pushes, so the index
// read after a successful decrement is never being written.
fn pop_edge() -> u32 {
    var n = atomicLoad(&edge_pool.counter);
    loop {
        if (n == 0u) { return NONE; }
        let r = atomicCompareExchangeWeak(&edge_pool.counter, n, n - 1u);
        if (r.exchanged) { return edge_pool.indices[n - 1u]; }
        n = r.old_value;
    }
}

// A fresh edge with no history.
fn new_edge(a: u32, b: u32, integrity: f32) -> Edge {
    var c: Edge;
    c.particle_a = a;
    c.particle_b = b;
    c.is_active = 1u;
    c.integrity = integrity;
    c.lambda = 0.0;
    c.lambda_prev = 0.0;
    c.yield_stress_mpa = sim.yield_stress_mpa;
    c.stress_normal_mpa = 0.0;
    c.stress_shear_mpa = 0.0;
    c.stored_stress_n = 0.0;
    c.stored_stress_t0 = 0.0;
    c.stored_stress_t1 = 0.0;
    c.slip_anchor0 = 0.0;
    c.slip_anchor1 = 0.0;
    c.timer = 0.0;
    c.born_myr = sim.time_myr;
    return c;
}

@compute @workgroup_size(64)
fn edges_begin(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < sim.max_particles) { atomicStore(&particle_deltas[id.x].slot_count, 0u); }
}

@compute @workgroup_size(64)
fn edges_prune(@builtin(global_invocation_id) id: vec3<u32>) {
    let e = id.x;
    if (e >= sim.max_edges || edges[e].is_active == 0u) { return; }
    let a = edges[e].particle_a;
    let b = edges[e].particle_b;
    if (particles[a].is_active == 0u || particles[b].is_active == 0u || !in_range(a, b)) {
        edges[e].is_active = 0u;
        release(false, e);
        return;
    }
    edges[e].lambda = 0.0;
    edges[e].lambda_prev = 0.0;
    list_edge(a, e);
    list_edge(b, e);
}

@compute @workgroup_size(64)
fn edges_note_listed(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    particles[idx].edge_count = atomicLoad(&particle_deltas[idx].slot_count);
}

fn listed(i: u32, j: u32, n: u32) -> bool {
    for (var k = 0u; k < n; k++) {
        if (other_end(particle_deltas[i].slots[k], i) == j) { return true; }
    }
    return false;
}

// Also the (overlap-based) subduction trigger, until phase F replaces it with
// a stress-based rule: deep map-view overlap between two surface columns
// makes the lower one a slab at the end of the step, and no edge is created
// between them. Every pair is examined by both threads, so each thread only
// writes its own next_state.
@compute @workgroup_size(64)
fn edges_create(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    let n0 = particles[idx].edge_count;      // survivors listed by edges_prune
    let can_create = n0 <= MAX_SLOTS;
    let surface_i = particles[idx].state == 1u;
    let home = hash_cell(particles[idx].predicted_pos.xyz);
    let dim = i32(sim.hash_grid_dim);

    for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
        let nc = home + vec3i(x, y, z);
        if (any(nc < vec3i(0)) || any(nc >= vec3i(dim))) { continue; }
        var j = atomicLoad(&hash_grid[hash_index(nc)]);
        var iter = 0;
        while (j != NONE && iter < 50) {
            let next = particles[j].next_idx;
            iter++;
            if (j == idx || !in_range(idx, j)) { j = next; continue; }
            if (surface_i && particles[j].state == 1u && pair_overlap(idx, j) > sim.subduction_overlap) {
                let lower = select(j, idx, particles[idx].elevation < particles[j].elevation
                    || (particles[idx].elevation == particles[j].elevation && idx < j));
                if (lower == idx) { particles[idx].next_state = 2u; }
            } else if (idx < j && can_create && !listed(idx, j, n0)) {
                let e = pop_edge();
                if (e == NONE) {
                    atomicAdd(&solver.edge_pool_empty, 1u);
                } else {
                    edges[e] = new_edge(idx, j, 0.0);
                    list_edge(idx, e);
                    list_edge(j, e);
                }
            }
            j = next;
        }
    }}}
}

// Angle of j around i in i's tangent plane: e1 = normalize(ref x n),
// e2 = n x e1, ref = z (x near the poles). Same basis as partnerAngle in
// src/edges/topology.js.
fn slot_angle(i: u32, j: u32) -> f32 {
    let n = normalize(particles[i].predicted_pos.xyz);
    let ref_axis = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(n.z) > 0.9);
    let e1 = normalize(cross(ref_axis, n));
    let e2 = cross(n, e1);
    let d = particles[j].predicted_pos.xyz - particles[i].predicted_pos.xyz;
    return atan2(dot(d, e2), dot(d, e1));
}

// Final edge count (for the Jacobi scaling), slot list sorted by angle
// (insertion sort over <= 12), and the solver's start-of-step iterates.
@compute @workgroup_size(64)
fn edges_finish(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    let count = atomicLoad(&particle_deltas[idx].slot_count);
    particles[idx].edge_count = count;
    let n = min(count, MAX_SLOTS);
    var ids: array<u32, 12>;
    var ang: array<f32, 12>;
    for (var k = 0u; k < n; k++) {
        ids[k] = particle_deltas[idx].slots[k];
        ang[k] = slot_angle(idx, other_end(ids[k], idx));
    }
    for (var k = 1u; k < n; k++) {
        let e = ids[k];
        let a = ang[k];
        var m = k;
        while (m > 0u && ang[m - 1u] > a) {
            ids[m] = ids[m - 1u];
            ang[m] = ang[m - 1u];
            m--;
        }
        ids[m] = e;
        ang[m] = a;
    }
    for (var k = 0u; k < n; k++) { particle_deltas[idx].slots[k] = ids[k]; }
    particles[idx].iter_start = particles[idx].predicted_pos;
    particles[idx].iter_prev = particles[idx].predicted_pos;
}
