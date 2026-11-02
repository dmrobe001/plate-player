// Free radius with bending (design 4.3, 4.4; phase F) and arcs (design 6).
// CPU reference: src/columns/flexure.js (it documents the discretisation).
//
// Each step's surface elevations w minimise
//   sum_i A_i [ (K_i tau_i / 2 dt) (w_i - w_start,i)^2 + (K_i / 2) (w_i - w_eq,i)^2 ]
//     + sum_i (D_i A_i / 2) m_i^2,   m_i = (1 / A_i) sum_j c_ij (w_j - w_i)
// (implicit Euler against the Haskell rebound time tau, a Winkler foundation
// K = drho g toward the Airy height w_eq, and plate bending with the
// tessellation's Laplacian: c_ij = shared cell boundary / distance). Solved by
// Jacobi sweeps with the bending part of the diagonal scaled by
// flexure_damping (convergence). With D = 0 one sweep is exact: each column
// relaxes alone toward its Airy height, as before phase F.
//
//   columns_flexure_begin   D_i = E h_e^3 / 12 (1 - nu^2), h_e = sum of the
//                           column's slices x their Maxwell factor this step
//   columns_moment          M_i = D_i m_i
//   columns_flexure         one Jacobi update of w_i
//   slabs_arc_count         per slab supplying an arc: surface columns within reach
//   columns_arc             arc volume, hot arc lithosphere

const R_M: f32 = 6.371e6;

fn winkler_k(eq: f32) -> f32 {
    return select(sim.mantle_density, sim.mantle_density - sim.water_density, eq < sim.sea_level) * 9.81;
}

@compute @workgroup_size(64)
fn columns_flexure_begin(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    particles[idx].elevation_start = particles[idx].elevation;
    particles[idx].bending_moment = 0.0;
    if (particles[idx].state != 1u) { particles[idx].flexural_rigidity = 0.0; return; }
    let E = sim.crust_youngs_modulus_gpa * 1e9;
    let edt = E * sim.dt_crust_myr * 3.15576e13;
    var he = 0.0;
    for (var k = 0u; k < SLICES; k++) {
        let eta = pow(10.0, min(profiles[idx].log10_viscosity[k], 30.0));
        he += profiles[idx].thickness_km[k] / (1.0 + edt / eta);
    }
    let h = he * 1000.0;
    particles[idx].flexural_rigidity = sim.flexure_factor * E * h * h * h / (12.0 * (1.0 - sim.poisson_ratio * sim.poisson_ratio));
}

@compute @workgroup_size(64)
fn columns_moment(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u || particles[idx].state != 1u) { return; }
    let n = particle_deltas[idx].cell_count;
    let w = particles[idx].elevation;
    var s = 0.0;
    for (var k = 0u; k < n; k++) {
        s += particle_deltas[idx].cell_coef[k] * (particles[particle_deltas[idx].cell_nbr[k]].elevation - w);
    }
    let A = particles[idx].cell_area * R_M * R_M;
    particles[idx].bending_moment = particles[idx].flexural_rigidity * s / A;
}

@compute @workgroup_size(64)
fn columns_flexure(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u || particles[idx].state != 1u) { return; }
    let eq = airy_elevation(idx);
    let K = winkler_k(eq);
    let r = isostatic_time_myr(idx, eq) / sim.dt_crust_myr; // tau / dt
    let A = particles[idx].cell_area * R_M * R_M;
    let w = particles[idx].elevation;
    let D = particles[idx].flexural_rigidity;
    let Mi = particles[idx].bending_moment;
    var grad = A * K * ((1.0 + r) * w - eq - r * particles[idx].elevation_start);
    var csum = 0.0;
    var hb = 0.0;
    let n = particle_deltas[idx].cell_count;
    for (var k = 0u; k < n; k++) {
        let j = particle_deltas[idx].cell_nbr[k];
        let c = particle_deltas[idx].cell_coef[k];
        grad += c * (particles[j].bending_moment - Mi);
        csum += c;
        hb += particles[j].flexural_rigidity * c * c / (particles[j].cell_area * R_M * R_M);
    }
    hb += D * csum * csum / A;
    particles[idx].elevation = w - grad / (A * K * (1.0 + r) + sim.flexure_damping * hb);
}

// Arcs (design 6). A slab's arc_supply was set as it sank through the arc
// depth window this step (thermal.wgsl); here each supplying slab counts the
// surface columns within arc_reach on the far side from the plate behind it,
// and each of those takes an equal share.
// Surface column j is in slab s's arc: within reach, and not on the side of
// the plate behind the slab (when it has one).
fn arc_column(s: u32, j: u32) -> bool {
    let d = particles[j].position.xyz - particles[s].position.xyz;
    let t = particles[s].slab_trailing;
    return particles[j].is_active != 0u && particles[j].state == 1u && length(d) < sim.arc_reach && !(t.w > 0.0 && dot(d, t.xyz) >= 0.0);
}

@compute @workgroup_size(64)
fn slabs_arc_count(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u || particles[idx].state != 2u) { return; }
    particles[idx].arc_count = 0u;
    particles[idx].slab_trailing = vec4f(0.0);
    if (!(particles[idx].arc_supply > 0.0)) { return; }
    var count = 0u;
    let p = particles[idx].position.xyz;
    // The plate behind the slab: its surface edge partners, weighted by
    // integrity (slot lists from last step's edges). Arcs grow on the other side.
    var t = vec3f(0.0);
    let ns = min(atomicLoad(&particle_deltas[idx].slot_count), MAX_SLOTS);
    for (var k = 0u; k < ns; k++) {
        let e = particle_deltas[idx].slots[k];
        if (edges[e].is_active == 0u) { continue; }
        let j = other_end(e, idx);
        if (particles[j].state == 1u) { t += edges[e].integrity * (particles[j].position.xyz - p); }
    }
    // A hanging slab has no edges: the plate behind it is what it hangs from.
    if (particles[idx].attached_to != NONE) { t = particles[particles[idx].attached_to].position.xyz - p; }
    if (length(t) > 0.0) { particles[idx].slab_trailing = vec4f(normalize(t), 1.0); }
    let home = hash_cell(p);
    let dim = i32(sim.hash_grid_dim);
    for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
        let nc = home + vec3i(x, y, z);
        if (any(nc < vec3i(0)) || any(nc >= vec3i(dim))) { continue; }
        var j = atomicLoad(&hash_grid[hash_index(nc)]);
        var iter = 0;
        while (j != NONE && iter < HASH_LIST_CAP) {
            iter++;
            if (arc_column(idx, j)) { count++; }
            j = particles[j].next_idx;
        }
    }}}
    particles[idx].arc_count = count;
}

@compute @workgroup_size(64)
fn columns_arc(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u || particles[idx].state != 1u) { return; }
    let p = particles[idx].position.xyz;
    let home = hash_cell(p);
    let dim = i32(sim.hash_grid_dim);
    var added = 0.0;
    for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
        let nc = home + vec3i(x, y, z);
        if (any(nc < vec3i(0)) || any(nc >= vec3i(dim))) { continue; }
        var j = atomicLoad(&hash_grid[hash_index(nc)]);
        var iter = 0;
        while (j != NONE && iter < HASH_LIST_CAP) {
            iter++;
            if (particles[j].is_active != 0u && particles[j].state == 2u && particles[j].arc_count > 0u && particles[j].arc_supply > 0.0 && arc_column(j, idx)) {
                added += particles[j].arc_supply / f32(particles[j].arc_count);
            }
            j = particles[j].next_idx;
        }
    }}}
    if (!(added > 0.0)) { return; }
    let V = particles[idx].crust_volume;
    particles[idx].composition_density = (particles[idx].composition_density * V + sim.arc_density * added) / (V + added);
    particles[idx].crust_volume = V + added;
    particles[idx].thickness_km = (V + added) / particles[idx].cell_area;
    particles[idx].volume_added += added;
    particles[idx].thermal_age_myr = 0.0;
    add_budget(BUDGET_SOURCE, added);
}
