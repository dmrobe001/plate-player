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

// Depth of the plate's base (lithosphere-asthenosphere boundary), km:
// half-space cooling 2.32 sqrt(kappa t) of the thermal age, capped at the full
// thickness for the composition (CPU reference: lithosphereThicknessKm in
// src/edges/topology.js). Stretching thins it by scaling t (area.wgsl).
fn lithosphere_cap_km(rho: f32) -> f32 {
    return select(sim.oceanic_lithosphere_max_km, sim.continental_lithosphere_km, rho < 2850.0);
}
fn lithosphere_thickness_km(i: u32) -> f32 {
    let age_s = max(particles[i].thermal_age_myr, 0.0) * 3.15576e13;
    return min(2.32 * sqrt(sim.thermal_diffusivity_m2_s * age_s) / 1000.0, lithosphere_cap_km(particles[i].composition_density));
}
// Thermal age at which the lithosphere reaches its cap, Myr (thermalAgeCapMyr in src/columns/volume.js).
fn thermal_age_cap_myr(rho: f32) -> f32 {
    let x = lithosphere_cap_km(rho) * 1000.0 / 2.32;
    return x * x / sim.thermal_diffusivity_m2_s / 3.15576e13;
}

// Couette shear across the asthenosphere: drag per area = viscosity / h, with
// h from the plate's base to the base of the low-viscosity layer. Returned
// relative to a base-size particle over the reference channel (w = 1).
fn inverse_drag(i: u32) -> f32 {
    let channel = max(sim.asthenosphere_base_km - lithosphere_thickness_km(i), sim.min_channel_km);
    return (base_area() / particles[i].area_ref) * (channel / sim.drag_reference_channel_km);
}

// The same basal shear resists the column's spin: drag x squared radius of
// gyration, area_ref cosh(2 s) / 2 pi (equal-area ellipse; gyrationSquared in
// src/columns/shape.js). Summed over a plate, spin and orbital drag are the
// two terms of the parallel-axis theorem: a plate rotating rigidly feels both.
fn inverse_spin_drag(i: u32) -> f32 {
    let s = length(vec2f(particles[i].shape_p, particles[i].shape_q));
    return inverse_drag(i) * 6.2831853 / (particles[i].area_ref * cosh(2.0 * s));
}

@compute @workgroup_size(64)
fn crust_predict(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    let pos = particles[idx].position.xyz;
    let node = particles[idx].mantle_node;

    // (Slab pull is not a velocity here: sinking slabs pull through their
    // edges, which are measured in 3D; bindings.wgsl, edge_offset.)
    let grid_vel = grid[node].velocity.xyz * sim.mantle_drive;
    particles[idx].predicted_pos = vec4f(normalize(pos + grid_vel * sim.dt_crust_myr), 1.0);
    // The mantle flow is a potential flow (no vertical vorticity, src/mantle/),
    // so basal drag alone does not spin a column: spin starts at 0.
    particles[idx].spin = 0.0;
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
        atomicStore(&particle_deltas[idx].dspin, 0i);
        atomicStore(&particle_deltas[idx].tdx, 0i);
        atomicStore(&particle_deltas[idx].tdy, 0i);
        atomicStore(&particle_deltas[idx].tdz, 0i);
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
    // Tangential rows (phase G) are plain Jacobi: their position corrections
    // are added after the extrapolation, and spins and their multipliers are
    // not extrapolated, so each subsystem stays consistent with its
    // multipliers. (Extrapolating them with the normal rows' weights slows
    // them: for rho near 1, Chebyshev converges every mode at the slow rate.)
    // The previous iterate is shifted by the same correction, so the next
    // extrapolation sees only the normal rows' progress (no momentum is given
    // to the tangential corrections).
    let td = vec3f(f32(atomicLoad(&particle_deltas[idx].tdx)), f32(atomicLoad(&particle_deltas[idx].tdy)), f32(atomicLoad(&particle_deltas[idx].tdz))) / FIXED_POINT;
    var prev = particles[idx].iter_start;
    if (any(td != vec3f(0.0))) {
        x = vec4f(normalize(x.xyz + td), 1.0);
        particles[idx].predicted_pos = x;
        prev = vec4f(normalize(prev.xyz + td), 1.0);
    }
    particles[idx].iter_prev = prev;
    particles[idx].iter_start = x;
}

// One XPBD update per edge on its linearised law (edges_law, law.wgsl):
//   C~ = |x_b - x_a| - reference_gap, a~ = 1 / (rate * stiffness * dt)
//   dlambda = (-C~ - a~ lambda) / (w_a + w_b + a~)
//   dx_a = -w_a n dlambda,  dx_b = +w_b n dlambda
// lambda > 0 pushes. It is clamped to [lambda_min, lambda_max], the
// depth-integrated tension and compression caps of the yield-strength
// envelope. Edges are updated simultaneously (Jacobi), so each dlambda is
// scaled by 1 / max(edges at either end). Scaling dlambda, not the applied
// dx, keeps lambda consistent with the displacement it caused, so a
// converged solve is the true minimum of drag + edge energy.
//
// Then, in the same thread, the tangential row (phase G): C_t = delta_t -
// reference_t, delta_t = L0 psi - l_a spin_a - l_b spin_b (bindings.wgsl),
// gradients -t, +t on the positions and -l_a, -l_b on the spins, spins
// weighted by the inverse spin drag; clamped to the shear caps. Then the face
// moment: C_r = spin_b - spin_a - reference_r, clamped by the kern rule.
// Both are plain Jacobi (not extrapolated; crust_chebyshev_combine).
@compute @workgroup_size(64)
fn crust_solve_edges(@builtin(global_invocation_id) id: vec3<u32>) {
    let c_idx = id.x;
    if (c_idx >= sim.max_edges || edges[c_idx].is_active == 0u) { return; }
    let c = edges[c_idx];
    let pA = c.particle_a;
    let pB = c.particle_b;

    let dir = edge_offset(pA, pB, particles[pA].predicted_pos.xyz, particles[pB].predicted_pos.xyz);
    let dist = max(length(dir), 1e-6);
    let n = dir / dist;
    let C = dist - c.reference_gap - rest_length(c); // + stretched, - compressed, vs this step's zero-force length

    let alpha = 1.0 / (sim.bond_rate_myr_inv * max(c.stiffness, 1e-6) * sim.dt_crust_myr);
    let wA = inverse_drag(pA);
    let wB = inverse_drag(pB);
    let degree = f32(max(max(particles[pA].edge_count, particles[pB].edge_count), 1u));
    var dlambda = (-C - alpha * c.lambda) / (wA + wB + alpha) / degree;
    let omega = chebyshev_omega();
    let lambda = clamp(omega * (c.lambda + dlambda - c.lambda_prev) + c.lambda_prev, c.lambda_min, c.lambda_max);
    dlambda = clamp(c.lambda + dlambda, c.lambda_min, c.lambda_max) - c.lambda;
    edges[c_idx].lambda = lambda;
    edges[c_idx].lambda_prev = c.lambda;

    add_delta(pA, -wA * dlambda * n, 0i);
    add_delta(pB, wB * dlambda * n, 0i);

    if (c.stiffness_t <= 0.0 && c.stiffness_r <= 0.0) { return; }
    let sA = inverse_spin_drag(pA);
    let sB = inverse_spin_drag(pB);
    if (c.stiffness_t > 0.0) {
        let up = normalize(particles[pA].predicted_pos.xyz + particles[pB].predicted_pos.xyz);
        let t = normalize(cross(up, n));
        let la = c.lever_a * c.length;
        let lb = c.length - la;
        let Ct = tangential_displacement(c) - c.reference_t;
        let at = 1.0 / (sim.bond_rate_myr_inv * c.stiffness_t * sim.dt_crust_myr);
        var dl = (-Ct - at * c.lambda_t) / (wA + wB + sA * la * la + sB * lb * lb + at) / degree;
        let lt = clamp(c.lambda_t + dl, -c.lambda_t_max, c.lambda_t_max);
        dl = clamp(c.lambda_t + dl, -c.lambda_t_max, c.lambda_t_max) - c.lambda_t;
        edges[c_idx].lambda_t = lt;
        add_tangential_delta(pA, -wA * dl * t);
        add_tangential_delta(pB, wB * dl * t);
        add_spin(pA, -sA * la * dl);
        add_spin(pB, -sB * lb * dl);
    }
    if (c.stiffness_r > 0.0) {
        let Cr = particles[pB].spin - particles[pA].spin - c.reference_r;
        let ar = 1.0 / (sim.bond_rate_myr_inv * c.stiffness_r * sim.dt_crust_myr);
        var dl = (-Cr - ar * c.lambda_r) / (sA + sB + ar) / degree;
        let lr = clamp(c.lambda_r + dl, -c.lambda_r_max, c.lambda_r_max);
        dl = clamp(c.lambda_r + dl, -c.lambda_r_max, c.lambda_r_max) - c.lambda_r;
        edges[c_idx].lambda_r = lr;
        add_spin(pA, -sA * dl);
        add_spin(pB, sB * dl);
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
    particles[idx].spin += f32(atomicLoad(&particle_deltas[idx].dspin)) / FIXED_POINT;
}

@compute @workgroup_size(64)
fn crust_integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    // Overdamped: velocity is not state, only a record of this step's motion.
    particles[idx].velocity = vec4f((particles[idx].predicted_pos.xyz - particles[idx].position.xyz) / sim.dt_crust_myr, 0.0);
    let x = particles[idx].predicted_pos.xyz;
    particles[idx].position = vec4f(x, 1.0);
    // Body axis: carried to the new tangent plane and turned by the spin
    // (carryBodyAxis in src/columns/shape.js).
    let b0 = particles[idx].body_axis.xyz;
    var b = b0 - dot(b0, x) * x;
    b = select(tangent_e1(x), normalize(b), dot(b, b) > 1e-12);
    let spin = particles[idx].spin;
    particles[idx].body_axis = vec4f(cos(spin) * b + sin(spin) * cross(x, b), 0.0);
    particles[idx].rotation_total += spin;
    // Subduction initiated by an edge (edges_maxwell): the column becomes a
    // slab where it is; it then sinks under its own weight (thermal.wgsl), and
    // its edges are measured in 3D from the next step (length reset, split.wgsl).
    if (particles[idx].next_state == 2u && particles[idx].state == 1u) {
        particles[idx].state = 2u;
        particles[idx].geometry_reset = 1u;
        add_budget(BUDGET_SINK, particles[idx].crust_volume); // leaves the surface: a volume sink (design 4.8)
    }
    particles[idx].next_state = particles[idx].state;
}
