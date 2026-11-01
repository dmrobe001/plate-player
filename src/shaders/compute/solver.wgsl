// Crust motion: overdamped XPBD (design.md 5.2).
// Each step minimises  sum_i (gamma_i / 2 dt) |x_i - x~_i|^2 + U(x):
// basal drag gamma_i plays the role of mass, and there is no inertia.
//   1. predict:   x~ = where the mantle flow alone would carry the particle
//   2. hash:      bin predicted positions into a uniform grid
//   3. edges:     drop edges whose pair left range, create edges for pairs
//                 that came into range (edges.wgsl; design 3.2)
//   4. iterate:   all edges (XPBD, with accumulated multipliers lambda) as one
//                 Jacobi sweep, extrapolated by Chebyshev acceleration
//   5. integrate: velocity = (x - x_old) / dt, a diagnostic only
//
// Units inside the solver: drag is relative to a base-size particle
// (w_i = base_area / area_i is the inverse drag), and edge stiffness is
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
// The multipliers get the same combination in crust_solve_edges, so
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

// One XPBD update per edge, C = |x_b - x_a| - (apothem_a + apothem_b):
//   dlambda = (-C - a~ lambda) / (w_a + w_b + a~),  a~ = 1 / (stiffness * dt)
//   dx_a = -w_a n dlambda,  dx_b = +w_b n dlambda
// Edges are updated simultaneously (Jacobi), so each dlambda is scaled by
// 1 / max(edges at either end). Scaling dlambda, not the applied dx, keeps
// lambda consistent with the displacement it caused, so a converged solve is
// the true minimum of drag + elastic energy.
@compute @workgroup_size(64)
fn crust_solve_edges(@builtin(global_invocation_id) id: vec3<u32>) {
    let c_idx = id.x;
    if (c_idx >= sim.max_edges || edges[c_idx].is_active == 0u) { return; }
    let c = edges[c_idx];
    let pA = c.particle_a;
    let pB = c.particle_b;

    let dir = particles[pB].predicted_pos.xyz - particles[pA].predicted_pos.xyz;
    let dist = max(length(dir), 1e-6);
    let n = dir / dist;
    let C = dist - rest_length(c); // + stretched, - compressed

    // Lost integrity softens tension, never compression. Whether the edge
    // counts as in tension is decided by last step's stress, not the current
    // iterate: switching stiffness mid-solve makes the iteration nonlinear,
    // and Chebyshev acceleration then diverges (design.md, Chebyshev
    // findings). (Stand-in until the edge law of phase C.)
    let stiffness_factor = select(1.0, max(c.integrity, 1e-4), c.stress_normal_mpa > 0.0);
    let alpha = 1.0 / (sim.bond_rate_myr_inv * stiffness_factor * sim.dt_crust_myr);
    let wA = inverse_drag(pA);
    let wB = inverse_drag(pB);
    let degree = f32(max(max(particles[pA].edge_count, particles[pB].edge_count), 1u));
    var dlambda = (-C - alpha * c.lambda) / (wA + wB + alpha) / degree;
    let omega = chebyshev_omega();
    var lambda = omega * (c.lambda + dlambda - c.lambda_prev) + c.lambda_prev;
    if (c.integrity <= 0.0) {
        // No cohesion: one-sided, the total multiplier (push) never goes negative.
        dlambda = max(c.lambda + dlambda, 0.0) - c.lambda;
        lambda = max(lambda, 0.0);
    }
    edges[c_idx].lambda = lambda;
    edges[c_idx].lambda_prev = c.lambda;

    add_delta(pA, -wA * dlambda * n, 0i);
    add_delta(pB, wB * dlambda * n, 0i);
}

// After the solve: stress from the converged positions, then yield. Loss of
// integrity is an irreversible change between solves (design.md 4.7);
// updating it inside the iterations lets transient overshoot (which
// Chebyshev acceleration produces on purpose) soften edges and run away.
// An edge at zero integrity is not removed: it stays while its pair is in
// range and pushes only (design 3.2). (Damage rule: stand-in until phase D.)
@compute @workgroup_size(64)
fn crust_edge_stress(@builtin(global_invocation_id) id: vec3<u32>) {
    let c_idx = id.x;
    if (c_idx >= sim.max_edges || edges[c_idx].is_active == 0u) { return; }
    let c = edges[c_idx];
    let rest = rest_length(c);
    let C = distance(particles[c.particle_b].predicted_pos.xyz, particles[c.particle_a].predicted_pos.xyz) - rest;
    let stiffness_factor = select(1.0, max(c.integrity, 1e-4), C > 0.0);

    // Central edges carry no shear; a real shear measure arrives with the
    // per-particle strain fit (design 4.2).
    var sigma_n = sim.crust_youngs_modulus_gpa * 1000.0 * stiffness_factor * C / rest;
    if (c.integrity <= 0.0) { sigma_n = min(sigma_n, 0.0); } // no cohesion: carries no tension
    edges[c_idx].stress_normal_mpa = sigma_n;
    edges[c_idx].stress_shear_mpa = 0.0;
    if (c.integrity <= 0.0) { return; } // nothing left to damage

    // Arrhenius-weakened yield stress; overstress removes integrity.
    let temp = edge_temperature(c);
    let y = c.yield_stress_mpa * exp(clamp((sim.activation_energy / sim.gas_constant) * (1.0 / temp - 1.0 / sim.reference_temp), -10.0, 10.0));
    if (abs(sigma_n) > y) {
        edges[c_idx].integrity = max(c.integrity - ((abs(sigma_n) - y) / y) * 50.0 * sim.dt_crust_myr, 0.0);
    }
}

@compute @workgroup_size(64)
fn crust_apply_deltas(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    // Edge corrections arrive already scaled (dlambda / degree) and are applied in full.
    let sum = vec3f(f32(atomicLoad(&particle_deltas[idx].dx)), f32(atomicLoad(&particle_deltas[idx].dy)), f32(atomicLoad(&particle_deltas[idx].dz)));
    if (any(sum != vec3f(0.0))) {
        particles[idx].predicted_pos = vec4f(normalize(particles[idx].predicted_pos.xyz + sum / FIXED_POINT), 1.0);
    }
}

@compute @workgroup_size(64)
fn crust_integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    // Overdamped: velocity is not state, only a record of this step's motion.
    particles[idx].velocity = vec4f((particles[idx].predicted_pos.xyz - particles[idx].position.xyz) / sim.dt_crust_myr, 0.0);
    particles[idx].position = vec4f(particles[idx].predicted_pos.xyz, 1.0);
    // Subduction requested during contact detection: the column's top drops
    // below any lithosphere (a stand-in until slabs bend, design phase 7).
    if (particles[idx].next_state == 2u && particles[idx].state == 1u) {
        particles[idx].state = 2u;
        particles[idx].elevation = -sim.slab_initial_depth_km;
    }
    particles[idx].next_state = particles[idx].state;
}
