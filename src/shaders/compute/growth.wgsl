// Crust creation at divergent mantle flow: particles over upwellings grow in
// area, and split in two once large enough. The child is fresh oceanic crust.
// Bonds that now sit closer to the child than to the parent are re-attached.

@compute @workgroup_size(64)
fn crust_accumulation(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u || particles[idx].state == 2u) { return; }
    let pos = particles[idx].position.xyz;
    let node = nearest_node(pos);

    // velocity = -grad(psi), so div(velocity) = -lap(psi)
    let div = -grid[node].laplacian_psi;
    if (div > 1.5) {
        particles[idx].surface_area += min(div * sim.dt_crust_myr * 0.005, 0.0001);
        particles[idx].apothem = sqrt(particles[idx].surface_area / 3.4641);
    }

    let base_area = 3.4641 * sim.base_apothem * sim.base_apothem;
    if (particles[idx].surface_area > sim.split_area_factor * base_area && particles[idx].split_child == NONE) {
        // Pop one particle slot and one bond slot from the free lists.
        let p_count = atomicSub(&particle_pool.counter, 1u);
        if (p_count == 0u) { atomicAdd(&particle_pool.counter, 1u); return; }
        let c_count = atomicSub(&constraint_pool.counter, 1u);
        if (c_count == 0u) { atomicAdd(&particle_pool.counter, 1u); atomicAdd(&constraint_pool.counter, 1u); return; }
        let child = particle_pool.indices[p_count - 1u];
        let bond = constraint_pool.indices[c_count - 1u];

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

        constraints[bond].particle_a = idx;
        constraints[bond].particle_b = child;
        constraints[bond].rest_length = particles[idx].apothem * 2.0;
        constraints[bond].compliance = 0.001;
        constraints[bond].yield_stress_mpa = sim.yield_stress_mpa;
        constraints[bond].is_active = 1u;
        constraints[bond].damage = 0.0;
        constraints[bond].stress_normal_mpa = 0.0;
        constraints[bond].stress_shear_mpa = 0.0;
    }
}

@compute @workgroup_size(64)
fn crust_rewire_bonds(@builtin(global_invocation_id) id: vec3<u32>) {
    let c_idx = id.x;
    if (c_idx >= sim.max_constraints || constraints[c_idx].is_active == 0u) { return; }
    var pA = constraints[c_idx].particle_a;
    var pB = constraints[c_idx].particle_b;
    var rewired = false;

    let childA = particles[pA].split_child;
    if (childA != NONE && childA != pB) {
        if (distance(particles[childA].position.xyz, particles[pB].position.xyz) < distance(particles[pA].position.xyz, particles[pB].position.xyz)) {
            pA = childA; constraints[c_idx].particle_a = childA; rewired = true;
        }
    }
    let childB = particles[pB].split_child;
    if (childB != NONE && childB != pA) {
        if (distance(particles[pA].position.xyz, particles[childB].position.xyz) < distance(particles[pA].position.xyz, particles[pB].position.xyz)) {
            pB = childB; constraints[c_idx].particle_b = childB; rewired = true;
        }
    }
    if (rewired) {
        constraints[c_idx].rest_length = particles[pA].apothem + particles[pB].apothem;
        constraints[c_idx].damage = 0.0;
    }
}

@compute @workgroup_size(64)
fn crust_clean_split(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx < sim.max_particles) { particles[idx].split_child = NONE; }
}
