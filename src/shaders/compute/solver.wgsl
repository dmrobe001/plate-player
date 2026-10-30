// Crust motion: position-based dynamics (PBD).
//   1. predict:   drag velocity toward the mantle flow, predict new positions
//   2. hash:      bin predicted positions into a uniform grid for collisions
//   3. iterate:   bond constraints + collisions accumulate position corrections,
//                 which are averaged and applied (Jacobi iteration)
//   4. integrate: velocity = (predicted - old) / dt

@compute @workgroup_size(64)
fn crust_predict(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    let pos = particles[idx].position.xyz;
    var vel = particles[idx].velocity.xyz;
    let node = particles[idx].mantle_node;

    let grid_vel = grid[node].velocity.xyz * sim.sh_grad_to_cm_yr * sim.cm_yr_to_sim_vel;
    var target_vel = grid_vel;

    // Slab pull: a slab denser than the mantle is dragged along the flow.
    if (particles[idx].state == 2u && particles[idx].density > sim.mantle_density) {
        let pull = (particles[idx].density - sim.mantle_density) * 0.02;
        target_vel += normalize(grid_vel + vec3f(0.001)) * pull;
    }

    // Exact exponential relaxation toward the target: stable for any dt.
    vel = target_vel + (vel - target_vel) * exp(-sim.drag_coeff_myr_inv * sim.dt_crust_myr);
    particles[idx].predicted_pos = vec4f(normalize(pos + vel * sim.dt_crust_myr), 1.0);
    particles[idx].velocity = vec4f(vel, 0.0);
}

@compute @workgroup_size(64)
fn crust_hash_clear(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < sim.hash_grid_dim * sim.hash_grid_dim * sim.hash_grid_dim) { atomicStore(&hash_grid[id.x], NONE); }
}

// Each cell holds the head of a linked list threaded through particles[].next_idx.
@compute @workgroup_size(64)
fn crust_hash_build(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx < sim.max_particles && particles[idx].is_active == 1u) {
        let cell = hash_index(hash_cell(particles[idx].predicted_pos.xyz));
        particles[idx].next_idx = atomicExchange(&hash_grid[cell], idx);
    }
}

@compute @workgroup_size(64)
fn crust_clear_deltas(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx < sim.max_particles) {
        atomicStore(&particle_deltas[idx].dx, 0i);
        atomicStore(&particle_deltas[idx].dy, 0i);
        atomicStore(&particle_deltas[idx].dz, 0i);
        atomicStore(&particle_deltas[idx].count, 0i);
    }
}

@compute @workgroup_size(64)
fn crust_solve_constraints(@builtin(global_invocation_id) id: vec3<u32>) {
    let c_idx = id.x;
    if (c_idx >= sim.max_constraints || constraints[c_idx].is_active == 0u) { return; }
    let c = constraints[c_idx];
    let pA = c.particle_a;
    let pB = c.particle_b;

    // A bond to a melted particle dies with it.
    if (particles[pA].is_active == 0u || particles[pB].is_active == 0u) {
        constraints[c_idx].is_active = 0u;
        release(false, c_idx);
        return;
    }

    let dir = particles[pB].predicted_pos.xyz - particles[pA].predicted_pos.xyz;
    let dist = max(length(dir), 0.0001);
    let n = dir / dist;

    // Normal error: stretch (+) or compression (-) relative to rest length.
    let err_n = dist - c.rest_length;
    // "Shear" error: relative motion *this step* perpendicular to the bond.
    let rel_move = (particles[pB].predicted_pos.xyz - particles[pB].position.xyz)
                 - (particles[pA].predicted_pos.xyz - particles[pA].position.xyz);
    let err_t = rel_move - dot(rel_move, n) * n;

    // Damage weakens tension and shear, but never compression.
    let tension_factor = select(1.0, 1.0 - c.damage, err_n > 0.0);
    let shear_factor = max(1.0 - c.damage, 0.1);
    let total_corr = n * (err_n * tension_factor) + err_t * shear_factor;

    // Strain -> stress (MPa) for yielding and for the stress display.
    let modulus_mpa = sim.crust_youngs_modulus_gpa * 1000.0;
    let sigma_n = err_n / c.rest_length * modulus_mpa;
    let sigma_t = length(err_t) / c.rest_length * modulus_mpa;
    constraints[c_idx].stress_normal_mpa = sigma_n * tension_factor;
    constraints[c_idx].stress_shear_mpa = sigma_t * shear_factor;

    // Arrhenius-weakened yield stress; overstress accumulates damage.
    let stress = sqrt(sigma_n * sigma_n + sigma_t * sigma_t);
    let temp = bond_temperature(c);
    let y = c.yield_stress_mpa * exp(clamp((sim.activation_energy / sim.gas_constant) * (1.0 / temp - 1.0 / sim.reference_temp), -10.0, 10.0));
    if (stress > y) {
        constraints[c_idx].damage = min(c.damage + ((stress - y) / y) * 50.0 * sim.dt_crust_myr, 1.0);
    }

    let corr = total_corr * 0.5;
    add_delta(pA, corr, 1i);
    add_delta(pB, -corr, 1i);
}

@compute @workgroup_size(64)
fn crust_solve_collisions(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    let pos_i = particles[idx].predicted_pos.xyz;
    let state_i = particles[idx].state;
    // Summed in fixed point pair by pair, so the result does not depend on
    // the (thread-timing-dependent) order of the hash grid's linked lists.
    var total_delta = vec3i(0);
    var coll_count = 0i;
    let home = hash_cell(pos_i);
    let dim = i32(sim.hash_grid_dim);

    for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
        let nc = home + vec3i(x, y, z);
        if (any(nc < vec3i(0)) || any(nc >= vec3i(dim))) { continue; }
        var j = atomicLoad(&hash_grid[hash_index(nc)]);
        var iter = 0;
        while (j != NONE && iter < 50) {
            if (j != idx) {
                let pos_j = particles[j].predicted_pos.xyz;
                let state_j = particles[j].state;
                let dir = pos_i - pos_j;
                let dist = max(length(dir), 0.0001);
                let overlap = (particles[idx].apothem + particles[j].apothem) - dist;

                if (overlap > 0.0) {
                    if (state_i == 1u && state_j == 1u && overlap > sim.subduction_overlap) {
                        // Deep overlap between surface particles: the lower one subducts.
                        // Only this thread writes next_state[idx], and other threads read
                        // `state`, which does not change until crust_apply_deltas.
                        if (particles[idx].elevation < particles[j].elevation) { particles[idx].next_state = 2u; }
                    } else if (state_i != state_j) {
                        // Slab under surface crust: no normal push, only basal friction.
                        let rel = (pos_i - particles[idx].position.xyz) - (pos_j - particles[j].position.xyz);
                        total_delta -= to_fixed((rel - dot(rel, dir / dist) * (dir / dist)) * 0.2);
                        coll_count++;
                    } else {
                        total_delta += to_fixed((dir / dist) * overlap * 0.5);
                        coll_count++;
                    }
                }
            }
            j = particles[j].next_idx;
            iter++;
        }
    }}}
    if (coll_count > 0i) { add_delta_fixed(idx, total_delta, coll_count); }
}

@compute @workgroup_size(64)
fn crust_apply_deltas(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    if (particles[idx].next_state == 2u) { particles[idx].state = 2u; }
    let count = atomicLoad(&particle_deltas[idx].count);
    if (count > 0i) {
        let sum = vec3f(f32(atomicLoad(&particle_deltas[idx].dx)), f32(atomicLoad(&particle_deltas[idx].dy)), f32(atomicLoad(&particle_deltas[idx].dz)));
        particles[idx].predicted_pos = vec4f(normalize(particles[idx].predicted_pos.xyz + sum / (FIXED_POINT * f32(count))), 1.0);
    }
}

@compute @workgroup_size(64)
fn crust_integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    particles[idx].velocity = vec4f((particles[idx].predicted_pos.xyz - particles[idx].position.xyz) / sim.dt_crust_myr, 0.0);
    particles[idx].position = vec4f(particles[idx].predicted_pos.xyz, 1.0);
}
