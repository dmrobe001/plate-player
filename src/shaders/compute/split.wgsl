// Splits (design 4.8, phase E). CPU reference: splitColumn in
// src/columns/volume.js. A surface column whose actual (tessellated) area
// exceeds split_area_factor x the base hexagon, and whose cell is not clipped, splits along its cell's major
// axis: parent and child go to the centroid -/+ SPLIT_OFFSET x spread along
// that axis, and share volume and area_ref equally. The child copies the
// parent's material (composition, thermal age, age). The edge between them
// starts intact (integrity 1) and stress-free; edges to former neighbours are
// re-attached to whichever half is closer. Splits are stress-neutral: every
// edge touching a split column measures its next strain increment from the
// new geometry (length reset), keeping its stored stresses.
//
// Slot allocation is two-phase so that the free lists never change while
// being read: each splitting thread takes a ticket t from
// `particle_pool.requests` and claims the t-th slot from the top of both
// free lists, provided t is below both counts; crust_commit_splits then pops
// the claimed slots. Which particle gets which slot depends on thread timing,
// but no slot is ever claimed twice and the counters never wrap.

const SPLIT_OFFSET: f32 = 0.87;

@compute @workgroup_size(64)
fn columns_split(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u || particles[idx].state != 1u) { return; }
    let base_area = 3.4641016 * sim.base_apothem * sim.base_apothem;
    // A clipped cell has a free side (no neighbour within reach): its area is
    // not a measure of stretching, so it does not split.
    if (particles[idx].cell_area <= sim.split_area_factor * base_area || particles[idx].cell_clipped != 0u
        || particles[idx].split_child != NONE) { return; }

    let ticket = atomicAdd(&particle_pool.requests, 1u);
    let p_count = atomicLoad(&particle_pool.counter);
    let c_count = atomicLoad(&edge_pool.counter);
    if (ticket >= min(p_count, c_count)) { return; } // out of slots; try again next step
    let child = particle_pool.indices[p_count - 1u - ticket];
    let edge = edge_pool.indices[c_count - 1u - ticket];

    // Halves' positions in the parent's gnomonic tangent coordinates.
    let p = particles[idx].position.xyz;
    let e1 = tangent_e1(p);
    let e2 = cross(p, e1);
    let shape = particles[idx].cell_shape;
    let off = SPLIT_OFFSET * shape.w * vec2f(cos(shape.z), sin(shape.z));
    let up = shape.xy - off;
    let uc = shape.xy + off;
    let xp = vec4f(normalize(p + up.x * e1 + up.y * e2), 1.0);
    let xc = vec4f(normalize(p + uc.x * e1 + uc.y * e2), 1.0);

    particles[idx].area_ref *= 0.5;
    particles[idx].apothem = sqrt(particles[idx].area_ref / 3.4641016);
    particles[idx].crust_volume *= 0.5;
    particles[idx].volume_added = 0.0;
    particles[idx].split_child = child;
    particles[idx].position = xp;
    particles[idx].predicted_pos = xp;

    particles[child] = particles[idx];
    particles[child].position = xc;
    particles[child].predicted_pos = xc;
    particles[child].split_child = child; // marks "just split" for crust_rewire_edges
    particles[child].is_active = 1u;
    // No dilatation (melt, thermal thinning) is counted for either half next step.
    particles[idx].cell_area = 0.0;
    particles[child].cell_area = 0.0;

    var c = new_edge(idx, child, 1.0);
    for (var k = 0u; k < 8u; k++) { c.slice_stress_mpa[k] = 0.0; }
    edges[edge] = c;
}

// Single thread: pop the slots claimed in columns_split.
@compute @workgroup_size(1)
fn crust_commit_splits() {
    let p_count = atomicLoad(&particle_pool.counter);
    let c_count = atomicLoad(&edge_pool.counter);
    let granted = min(atomicLoad(&particle_pool.requests), min(p_count, c_count));
    atomicStore(&particle_pool.counter, p_count - granted);
    atomicStore(&edge_pool.counter, c_count - granted);
    atomicStore(&particle_pool.requests, 0u);
}

// Integrity and stored stresses are kept: re-attaching an edge does not weld
// or break it. An edge re-attached out of range is dropped by edges_prune.
// Every edge touching a split column (parent or child) restarts its strain
// measure from the new geometry, so the split adds no stress.
@compute @workgroup_size(64)
fn crust_rewire_edges(@builtin(global_invocation_id) id: vec3<u32>) {
    let c_idx = id.x;
    if (c_idx >= sim.max_edges || edges[c_idx].is_active == 0u) { return; }
    let pA = edges[c_idx].particle_a;
    let pB = edges[c_idx].particle_b;

    let childA = particles[pA].split_child;
    if (childA != NONE && childA != pB) {
        if (distance(particles[childA].position.xyz, particles[pB].position.xyz) < distance(particles[pA].position.xyz, particles[pB].position.xyz)) {
            edges[c_idx].particle_a = childA;
        }
    }
    let a = edges[c_idx].particle_a;
    let childB = particles[pB].split_child;
    if (childB != NONE && childB != a) {
        if (distance(particles[a].position.xyz, particles[childB].position.xyz) < distance(particles[a].position.xyz, particles[pB].position.xyz)) {
            edges[c_idx].particle_b = childB;
        }
    }
    let b = edges[c_idx].particle_b;
    if (childA != NONE || childB != NONE) {
        edges[c_idx].length = distance(particles[a].position.xyz, particles[b].position.xyz);
    }
}

@compute @workgroup_size(64)
fn crust_clean_split(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx < sim.max_particles) { particles[idx].split_child = NONE; }
}
