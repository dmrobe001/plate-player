// Crust creation at divergent mantle flow: particles over upwellings grow in
// area, and split in two once large enough. The child is fresh oceanic crust.
// Edges that now sit closer to the child than to the parent are re-attached.

// Slot allocation for splits is two-phase so that the free lists never
// change while being read: each splitting thread takes a ticket t from
// `particle_pool.requests` and claims the t-th slot from the top of both
// free lists, provided t is below both counts; crust_commit_splits then
// pops the claimed slots. Which particle gets which slot depends on thread
// timing, but no slot is ever claimed twice and the counters never wrap.
@compute @workgroup_size(64)
fn crust_accumulation(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u || particles[idx].state == 2u) { return; }
    let pos = particles[idx].position.xyz;
    let node = particles[idx].mantle_node;

    // Spreading proxy: over diverging mantle, area grows at the divergence rate.
    let div = grid[node].divergence * sim.mantle_drive;
    if (div > sim.spreading_divergence_threshold) {
        let rate = min(div * sim.spreading_efficiency, sim.max_spreading_rate);
        particles[idx].surface_area *= exp(rate * sim.dt_crust_myr);
        particles[idx].apothem = sqrt(particles[idx].surface_area / 3.4641);
    }

    let base_area = 3.4641 * sim.base_apothem * sim.base_apothem;
    if (particles[idx].surface_area <= sim.split_area_factor * base_area || particles[idx].split_child != NONE) { return; }

    let ticket = atomicAdd(&particle_pool.requests, 1u);
    let p_count = atomicLoad(&particle_pool.counter);
    let c_count = atomicLoad(&edge_pool.counter);
    if (ticket >= min(p_count, c_count)) { return; } // out of slots; try again next step
    let child = particle_pool.indices[p_count - 1u - ticket];
    let edge = edge_pool.indices[c_count - 1u - ticket];

    particles[idx].surface_area *= 0.5;
    particles[idx].apothem = sqrt(particles[idx].surface_area / 3.4641);
    particles[idx].split_child = child;

    particles[child] = particles[idx];
    particles[child].temperature = sim.mantle_temp;
    particles[child].age = 0.0;
    particles[child].thickness_km = 7.0;
    particles[child].composition_density = 3000.0;
    particles[child].split_child = NONE;
    particles[child].is_active = 1u;

    // Separate parent and child along the local mantle flow.
    var spread_dir = normalize(grid[node].velocity.xyz);
    if (length(spread_dir) < 0.001) { spread_dir = normalize(cross(pos, vec3f(0.0, 1.0, 0.0))); }
    let offset = spread_dir * (particles[idx].apothem * 1.05);
    particles[idx].position = vec4f(normalize(particles[idx].position.xyz - offset), 1.0);
    particles[child].position = vec4f(normalize(particles[child].position.xyz + offset), 1.0);

    edges[edge] = new_edge(idx, child, 1.0); // parent and child start welded
}

// Single thread: pop the slots claimed in crust_accumulation.
@compute @workgroup_size(1)
fn crust_commit_splits() {
    let p_count = atomicLoad(&particle_pool.counter);
    let c_count = atomicLoad(&edge_pool.counter);
    let granted = min(atomicLoad(&particle_pool.requests), min(p_count, c_count));
    atomicStore(&particle_pool.counter, p_count - granted);
    atomicStore(&edge_pool.counter, c_count - granted);
    atomicStore(&particle_pool.requests, 0u);
}

// Integrity is kept: re-attaching an edge does not weld or break it. An edge
// re-attached out of range is dropped by edges_prune.
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
}

@compute @workgroup_size(64)
fn crust_clean_split(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx < sim.max_particles) { particles[idx].split_child = NONE; }
}
