// Column-level irreversible processes (design 4.6, 4.7; phase D). CPU
// reference: src/columns/flow.js.
//
//   columns_gpe_force  after the edge update: push each surface column down
//                      the least-squares gradient of GPE over its listed
//                      neighbours (ridge push, plateau collapse), as a change
//                      of this step's predicted position against basal drag
//   columns_flow       after edges_maxwell: volumetric flow at fixed crust
//                      volume, d ln area_ref = 2 x mean inelastic strain of
//                      the column's edges; thickness changes inversely

@compute @workgroup_size(64)
fn columns_flow(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    let n = atomicLoad(&particle_deltas[idx].count);
    if (n == 0i) { return; }
    let mean = f32(atomicLoad(&particle_deltas[idx].dx)) / FIXED_POINT / f32(n);
    let f = exp(2.0 * sim.volumetric_flow_factor * mean);
    particles[idx].surface_area *= f;
    particles[idx].thickness_km /= f;
    particles[idx].apothem = sqrt(particles[idx].surface_area / 3.4641016);
}

@compute @workgroup_size(64)
fn columns_gpe_force(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u || particles[idx].state != 1u) { return; }
    let n = min(atomicLoad(&particle_deltas[idx].slot_count), MAX_SLOTS);
    if (n < 2u) { return; }
    // Geometry from start-of-step positions: predicted positions are being
    // pushed by other threads of this dispatch.
    let xi = particles[idx].position.xyz;
    let nrm = normalize(xi);
    let ref_axis = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(nrm.z) > 0.9);
    let e1 = normalize(cross(ref_axis, nrm));
    let e2 = cross(nrm, e1);
    let fi = profiles[idx].gpe_mpa_km;
    var a11 = 0.0; var a12 = 0.0; var a22 = 0.0; var b1 = 0.0; var b2 = 0.0;
    for (var k = 0u; k < n; k++) {
        let j = other_end(particle_deltas[idx].slots[k], idx);
        let d = particles[j].position.xyz - xi;
        let u = dot(d, e1);
        let v = dot(d, e2);
        let df = profiles[j].gpe_mpa_km - fi;
        a11 += u * u; a12 += u * v; a22 += v * v; b1 += u * df; b2 += v * df;
    }
    let det = a11 * a22 - a12 * a12;
    if (!(det > 1e-12 * (a11 + a22) * (a11 + a22))) { return; }
    let grad = ((a22 * b1 - a12 * b2) / det) * e1 + ((a11 * b2 - a12 * b1) / det) * e2;
    // dx = -grad Gamma h dt / eta_a (flow.js, gpeDisplacement), in unit-sphere lengths.
    let h = max(sim.asthenosphere_base_km - lithosphere_thickness_km(idx), sim.min_channel_km) * 1000.0;
    let s = -sim.gpe_force_factor * (1e9 / 6.371e6) * h * sim.dt_crust_myr * 3.15576e13 / sim.asthenosphere_viscosity_pa_s / 6.371e6;
    let x = vec4f(normalize(particles[idx].predicted_pos.xyz + s * grad), 1.0);
    particles[idx].predicted_pos = x;
    particles[idx].iter_start = x;
    particles[idx].iter_prev = x;
}
