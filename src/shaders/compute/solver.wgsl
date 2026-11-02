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
// relative to a base-size particle over the reference channel (w = 1). A
// trench column also drags its slabs (and its flap) down dip through the
// mantle: slab_drag_ratio times its basal drag (hinge.wgsl, design 6).
fn inverse_drag(i: u32) -> f32 {
    let channel = max(sim.asthenosphere_base_km - lithosphere_thickness_km(i), sim.min_channel_km);
    return (base_area() / particles[i].area_ref) * (channel / sim.drag_reference_channel_km) / (1.0 + particles[i].slab_drag_ratio);
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

// Problem size (simulation.js): every per-particle and per-edge kernel is
// dispatched over the slots below the high-water marks, not the capacity.
// solver_find_high rebuilds the marks from scratch (after a load or restore,
// dispatched over the full capacity); solver_dispatch_sizes turns them into
// indirect dispatch arguments and the draw instance count.
@compute @workgroup_size(64)
fn solver_find_high(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < sim.max_particles && particles[id.x].is_active != 0u) { note_particle_slot(id.x); }
    if (id.x < sim.max_edges && edges[id.x].is_active != 0u) { note_edge_slot(id.x); }
}

@compute @workgroup_size(1)
fn solver_dispatch_sizes() {
    let p = atomicLoad(&solver.particle_high);
    let e = atomicLoad(&solver.edge_high);
    solver.dispatch[0] = max((p + 63u) / 64u, 1u);
    solver.dispatch[1] = 1u;
    solver.dispatch[2] = 1u;
    solver.dispatch[3] = max((e + 63u) / 64u, 1u);
    solver.dispatch[4] = 1u;
    solver.dispatch[5] = 1u;
    solver.draw_instances = p;
}

@compute @workgroup_size(1)
fn solver_begin() {
    solver.iteration = 0u;
}

@compute @workgroup_size(1)
fn solver_next_iteration() {
    solver.iteration += 1u;
}


// Warm start: positions consistent with the starting multipliers (edges_law),
// x = x~ + W B^T lambda_0, so the solve begins from last step's forces.
@compute @workgroup_size(64)
fn crust_warm_start(@builtin(global_invocation_id) id: vec3<u32>) {
    let e = id.x;
    if (e >= sim.max_edges || edges[e].is_active == 0u || edges[e].lambda == 0.0) { return; }
    let pA = edges[e].particle_a;
    let pB = edges[e].particle_b;
    let dir = edge_offset(pA, pB, particles[pA].predicted_pos.xyz, particles[pB].predicted_pos.xyz);
    let n = dir / max(length(dir), 1e-6);
    add_delta(pA, -inverse_drag(pA) * edges[e].lambda * n, 0i);
    add_delta(pB, inverse_drag(pB) * edges[e].lambda * n, 0i);
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
// Both are plain Jacobi (not extrapolated).
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
    let dlambda = (-C - alpha * c.lambda) / (wA + wB + alpha) / degree;
    // Chebyshev extrapolation of the multiplier, clamped to the caps; the
    // positions move by exactly the change applied, so x - x~ = W B^T lambda
    // holds at every iteration, clamped or not (design 5.2).
    let omega = chebyshev_omega();
    let lambda = clamp(omega * (c.lambda + dlambda - c.lambda_prev) + c.lambda_prev, c.lambda_min, c.lambda_max);
    let applied = lambda - c.lambda;
    edges[c_idx].lambda = lambda;
    edges[c_idx].lambda_prev = c.lambda;

    add_delta(pA, -wA * applied * n, 0i);
    add_delta(pB, wB * applied * n, 0i);

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
    // Edge corrections arrive already scaled (dlambda / degree) and are applied
    // in full: the normal rows' (from the Chebyshev-extrapolated multipliers)
    // and the tangential rows' (plain Jacobi; their own accumulator).
    let sum = vec3f(f32(atomicLoad(&particle_deltas[idx].dx)), f32(atomicLoad(&particle_deltas[idx].dy)), f32(atomicLoad(&particle_deltas[idx].dz)))
        + vec3f(f32(atomicLoad(&particle_deltas[idx].tdx)), f32(atomicLoad(&particle_deltas[idx].tdy)), f32(atomicLoad(&particle_deltas[idx].tdz)));
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
    particles[idx].velocity = vec4f((particles[idx].predicted_pos.xyz - particles[idx].position.xyz) / sim.dt_crust_myr, particles[idx].velocity.w); // (w: a hanging slab's feed rate down dip, thermal.wgsl)
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
        // Its trench hinge passed its back face (hinge.wgsl): the whole column is
        // flap, dipping at its bend. It hangs from the column behind it (found by
        // the hinge; attached_to) from the next step, its top at that column's flap tip.
        let dip = max(particles[idx].hinge_angle, 0.2);
        particles[idx].hinge_angle = dip;
        particles[idx].hinge_plastic = 0.0; // a slab's neck (thermal.wgsl, slab_hang)
        particles[idx].slab_extent_km = 2.0 * particles[idx].apothem * EARTH_RADIUS_KM * sin(dip);
        // (hinge_partner kept: the overriding column it went under, which the
        // column behind inherits as its trench partner, hinge.wgsl.)
        particles[idx].hinge_flap_km = 0.0;
        particles[idx].hinge_underthrust_km = 0.0;
        particles[idx].slab_drag_ratio = 0.0; // (a slab's drag is its own: it sinks, thermal.wgsl)
        if (sim.slab_attach_factor <= 0.0) { particles[idx].attached_to = NONE; }
    }
    particles[idx].next_state = particles[idx].state;
    // What hung from this column this step loads its hinge (or the slab above) next step.
    particles[idx].child_load = bitcast<f32>(atomicLoad(&particle_deltas[idx].child_load));
    atomicStore(&particle_deltas[idx].child_load, 0u);
    particles[idx].child_pull = bitcast<f32>(atomicLoad(&particle_deltas[idx].child_pull));
    atomicStore(&particle_deltas[idx].child_pull, 0u);
    particles[idx].child_drag = bitcast<f32>(atomicLoad(&particle_deltas[idx].child_drag));
    atomicStore(&particle_deltas[idx].child_drag, 0u);
    particles[idx].slab_child = NONE;
}

// No net rotation (design 5.3; netRotation in src/geometry/net-rotation.js).
// The rotation vector theta that best explains this step's displacements u
// (predicted - position) of the surface columns as one rigid rotation, in
// the area-weighted least-squares sense: minimise sum a |u - theta x x|^2,
// so I theta = L with L = sum a x cross u and I = sum a (1 - x x^T). Each
// workgroup sums its columns in shared memory, then adds its partial sums
// to the solver header (float atomics: one per workgroup and word).
var<workgroup> net_rotation_partial: array<array<f32, 9>, 64>;

@compute @workgroup_size(64)
fn crust_net_rotation_sum(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = id.x;
    var s: array<f32, 9>;
    if (i < sim.max_particles && particles[i].is_active != 0u && particles[i].state == 1u) {
        let x = particles[i].position.xyz;
        let u = particles[i].predicted_pos.xyz - x;
        let a = select(particles[i].area_ref, particles[i].cell_area, particles[i].cell_area > 0.0);
        let l = a * cross(x, u);
        s = array<f32, 9>(l.x, l.y, l.z,
            a * (1.0 - x.x * x.x), a * (1.0 - x.y * x.y), a * (1.0 - x.z * x.z),
            -a * x.x * x.y, -a * x.x * x.z, -a * x.y * x.z);
    }
    net_rotation_partial[li] = s;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if (li < stride) {
            for (var k = 0u; k < 9u; k++) { net_rotation_partial[li][k] += net_rotation_partial[li + stride][k]; }
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        for (var k = 0u; k < 9u; k++) { atomic_add_f32(&solver.net_rotation_sums[k], net_rotation_partial[0][k]); }
    }
}

// Solve I theta = L (one thread) and clear the sums for the next step. A
// patch too small to define a rotation (I nearly singular) removes nothing.
@compute @workgroup_size(1)
fn crust_net_rotation_solve() {
    var v: array<f32, 9>;
    for (var k = 0u; k < 9u; k++) {
        v[k] = bitcast<f32>(atomicLoad(&solver.net_rotation_sums[k]));
        atomicStore(&solver.net_rotation_sums[k], 0u);
    }
    let L = vec3f(v[0], v[1], v[2]);
    let I = mat3x3f(vec3f(v[3], v[6], v[7]), vec3f(v[6], v[4], v[8]), vec3f(v[7], v[8], v[5]));
    let t = (v[3] + v[4] + v[5]) / 3.0;
    let det = determinant(I);
    var theta = vec3f(0.0);
    if (t > 0.0 && det > 1e-3 * t * t * t) {
        // Inverse by cofactors (I is symmetric).
        let c0 = cross(I[1], I[2]);
        let c1 = cross(I[2], I[0]);
        let c2 = cross(I[0], I[1]);
        theta = (c0 * L.x + c1 * L.y + c2 * L.z) / det;
    }
    solver.net_rotation[0] = theta.x;
    solver.net_rotation[1] = theta.y;
    solver.net_rotation[2] = theta.z;
}

// Turn every column (surface and slab) by -theta: a change of frame, so it
// changes no length, and each column's spin loses theta's radial component,
// which keeps the faces' sliding unchanged (bindings.wgsl, tangential_displacement).
@compute @workgroup_size(64)
fn crust_remove_net_rotation(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= sim.max_particles || particles[i].is_active == 0u) { return; }
    let theta = vec3f(solver.net_rotation[0], solver.net_rotation[1], solver.net_rotation[2]);
    let angle = length(theta);
    if (!(angle > 0.0)) { return; }
    let k = theta / angle;
    let p = particles[i].predicted_pos.xyz;
    // Rodrigues, angle -angle about k. WGSL's sin and cos may err by ~5e-4
    // (absolute), more than a step's whole rotation, so small angles use
    // their series (error ~angle^5 / 120).
    let a2 = angle * angle;
    let s = select(sin(angle), angle * (1.0 - a2 / 6.0 * (1.0 - a2 / 20.0)), angle < 0.1);
    let one_minus_c = select(1.0 - cos(angle), 0.5 * a2 * (1.0 - a2 / 12.0 * (1.0 - a2 / 30.0)), angle < 0.1);
    let q = p - cross(k, p) * s + cross(k, cross(k, p)) * one_minus_c;
    particles[i].predicted_pos = vec4f(normalize(q), particles[i].predicted_pos.w);
    particles[i].spin -= dot(theta, particles[i].position.xyz);
}
