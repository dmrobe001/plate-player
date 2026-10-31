// Crust motion: overdamped XPBD (design.md 5.2).
// Each step minimises  sum_i (gamma_i / 2 dt) |x_i - x~_i|^2 + U(x):
// basal drag gamma_i plays the role of mass, and there is no inertia.
//   1. predict:   x~ = where the mantle flow alone would carry the particle
//   2. hash:      bin predicted positions into a uniform grid for collisions
//   3. iterate:   bonds (XPBD, with accumulated multipliers lambda), then
//                 collisions (position projection), each applied as a Jacobi sweep;
//                 the result is extrapolated by Chebyshev acceleration
//   4. integrate: velocity = (x - x_old) / dt, a diagnostic only
//
// Units inside the solver: drag is relative to a base-size particle
// (w_i = base_area / area_i is the inverse drag), and bond stiffness is
// sim.bond_rate_myr_inv in those units, so compliance~ = 1 / (rate * dt).

fn base_area() -> f32 {
    return 3.4641 * sim.base_apothem * sim.base_apothem;
}

// Depth of the plate's base (lithosphere-asthenosphere boundary), km.
// Oceanic: half-space cooling, 2.32 sqrt(kappa age), capped; continental: fixed keel.
fn lithosphere_thickness_km(i: u32) -> f32 {
    if (particles[i].composition_density < 2850.0) { return sim.continental_lithosphere_km; }
    let age_s = max(particles[i].age, 0.0) * 3.15576e13;
    return min(2.32 * sqrt(sim.thermal_diffusivity_m2_s * age_s) / 1000.0, sim.oceanic_lithosphere_max_km);
}

// Couette shear across the asthenosphere: drag per area = viscosity / h, with
// h from the plate's base to the base of the low-viscosity layer. Returned
// relative to a base-size particle over the reference channel (w = 1).
fn inverse_drag(i: u32) -> f32 {
    let channel = max(sim.asthenosphere_base_km - lithosphere_thickness_km(i), sim.min_channel_km);
    return (base_area() / particles[i].surface_area) * (channel / sim.drag_reference_channel_km);
}

@compute @workgroup_size(64)
fn crust_predict(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    let pos = particles[idx].position.xyz;
    let node = particles[idx].mantle_node;

    let grid_vel = grid[node].velocity.xyz * sim.mantle_drive;
    var target_vel = grid_vel;

    // Slab pull, as an extra velocity (force / drag).
    if (particles[idx].state == 2u && particles[idx].density > sim.mantle_density) {
        let pull = (particles[idx].density - sim.mantle_density) * 0.02;
        target_vel += normalize(grid_vel + vec3f(0.001)) * pull;
    }

    particles[idx].predicted_pos = vec4f(normalize(pos + target_vel * sim.dt_crust_myr), 1.0);
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

// Start of the solve: reset multipliers, and count each particle's bonds
// (into particle_deltas.count, copied to bond_count by the next pass).
@compute @workgroup_size(64)
fn crust_begin_constraints(@builtin(global_invocation_id) id: vec3<u32>) {
    let c_idx = id.x;
    if (c_idx >= sim.max_constraints || constraints[c_idx].is_active == 0u) { return; }
    constraints[c_idx].lambda = 0.0;
    constraints[c_idx].lambda_prev = 0.0;
    atomicAdd(&particle_deltas[constraints[c_idx].particle_a].count, 1i);
    atomicAdd(&particle_deltas[constraints[c_idx].particle_b].count, 1i);
}

@compute @workgroup_size(64)
fn crust_store_bond_count(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    particles[idx].bond_count = u32(atomicLoad(&particle_deltas[idx].count));
    particles[idx].iter_start = particles[idx].predicted_pos;
    particles[idx].iter_prev = particles[idx].predicted_pos;
}

@compute @workgroup_size(1)
fn solver_begin() {
    solver.iteration = 0u;
}

@compute @workgroup_size(1)
fn solver_next_iteration() {
    solver.iteration += 1u;
}

// Chebyshev semi-iterative acceleration (Wang 2015): with x^ the result of
// one plain sweep from x_k,  x_{k+1} = omega_k (x^ - x_{k-1}) + x_{k-1}.
// The multipliers get the same combination in crust_solve_constraints, so
// x - x~ stays consistent with lambda.
@compute @workgroup_size(64)
fn crust_chebyshev_combine(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    let omega = chebyshev_omega();
    var x = particles[idx].predicted_pos;
    if (omega != 1.0) {
        let prev = particles[idx].iter_prev.xyz;
        x = vec4f(normalize(omega * (x.xyz - prev) + prev), 1.0);
        particles[idx].predicted_pos = x;
    }
    particles[idx].iter_prev = particles[idx].iter_start;
    particles[idx].iter_start = x;
}

// One XPBD update per bond, C = |x_b - x_a| - (apothem_a + apothem_b):
//   dlambda = (-C - a~ lambda) / (w_a + w_b + a~),  a~ = 1 / (stiffness * dt)
//   dx_a = -w_a n dlambda,  dx_b = +w_b n dlambda
// Bonds are updated simultaneously (Jacobi), so each dlambda is scaled by
// 1 / max(bonds at either end). Scaling dlambda, not the applied dx, keeps
// lambda consistent with the displacement it caused, so a converged solve is
// the true minimum of drag + elastic energy.
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
    let dist = max(length(dir), 1e-6);
    let n = dir / dist;
    let C = dist - rest_length(c); // + stretched, - compressed

    // Damage softens tension, never compression. Whether the bond counts as
    // in tension is decided by last step's stress, not the current iterate:
    // switching stiffness mid-solve makes the iteration nonlinear, and
    // Chebyshev acceleration then diverges (design.md, Chebyshev findings).
    let stiffness_factor = select(1.0, max(1.0 - c.damage, 1e-4), c.stress_normal_mpa > 0.0);
    let alpha = 1.0 / (sim.bond_rate_myr_inv * stiffness_factor * sim.dt_crust_myr);
    let wA = inverse_drag(pA);
    let wB = inverse_drag(pB);
    let degree = f32(max(max(particles[pA].bond_count, particles[pB].bond_count), 1u));
    let dlambda = (-C - alpha * c.lambda) / (wA + wB + alpha) / degree;
    let omega = chebyshev_omega();
    constraints[c_idx].lambda = omega * (c.lambda + dlambda - c.lambda_prev) + c.lambda_prev;
    constraints[c_idx].lambda_prev = c.lambda;

    // count 0: bond corrections are applied in full (see crust_apply_deltas).
    add_delta(pA, -wA * dlambda * n, 0i);
    add_delta(pB, wB * dlambda * n, 0i);
}

// After the solve: stress from the converged positions, then yield. Damage
// is an irreversible change between solves (design.md 4.7); updating it
// inside the iterations lets transient overshoot (which Chebyshev
// acceleration produces on purpose) soften bonds and run away.
@compute @workgroup_size(64)
fn crust_bond_stress(@builtin(global_invocation_id) id: vec3<u32>) {
    let c_idx = id.x;
    if (c_idx >= sim.max_constraints || constraints[c_idx].is_active == 0u) { return; }
    let c = constraints[c_idx];
    let rest = rest_length(c);
    let C = distance(particles[c.particle_b].predicted_pos.xyz, particles[c.particle_a].predicted_pos.xyz) - rest;
    let stiffness_factor = select(1.0, max(1.0 - c.damage, 1e-4), C > 0.0);

    // Central bonds carry no shear; a real shear measure arrives with the
    // per-particle strain fit (design phase 4).
    let sigma_n = sim.crust_youngs_modulus_gpa * 1000.0 * stiffness_factor * C / rest;
    constraints[c_idx].stress_normal_mpa = sigma_n;
    constraints[c_idx].stress_shear_mpa = 0.0;

    // Arrhenius-weakened yield stress; overstress accumulates damage.
    let temp = bond_temperature(c);
    let y = c.yield_stress_mpa * exp(clamp((sim.activation_energy / sim.gas_constant) * (1.0 / temp - 1.0 / sim.reference_temp), -10.0, 10.0));
    if (abs(sigma_n) > y) {
        constraints[c_idx].damage = min(c.damage + ((abs(sigma_n) - y) / y) * 50.0 * sim.dt_crust_myr, 1.0);
    }
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
                        // Non-penetration, split in proportion to inverse drag.
                        let w_i = inverse_drag(idx);
                        let share = w_i / (w_i + inverse_drag(j));
                        total_delta += to_fixed((dir / dist) * overlap * share);
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
    // Collision corrections are averaged over their count (position-based
    // dynamics); bond corrections arrive with count 0 and are applied in full.
    let count = max(atomicLoad(&particle_deltas[idx].count), 1i);
    let sum = vec3f(f32(atomicLoad(&particle_deltas[idx].dx)), f32(atomicLoad(&particle_deltas[idx].dy)), f32(atomicLoad(&particle_deltas[idx].dz)));
    if (any(sum != vec3f(0.0))) {
        particles[idx].predicted_pos = vec4f(normalize(particles[idx].predicted_pos.xyz + sum / (FIXED_POINT * f32(count))), 1.0);
    }
}

@compute @workgroup_size(64)
fn crust_integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    // Overdamped: velocity is not state, only a record of this step's motion.
    particles[idx].velocity = vec4f((particles[idx].predicted_pos.xyz - particles[idx].position.xyz) / sim.dt_crust_myr, 0.0);
    particles[idx].position = vec4f(particles[idx].predicted_pos.xyz, 1.0);
}
