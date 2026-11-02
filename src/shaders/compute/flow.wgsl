// Column-level irreversible processes (design 4.6, 4.7; phase D). CPU
// reference: src/columns/flow.js.
//
//   columns_gpe_force  after the edge update: push each surface column down
//                      the least-squares gradient of GPE over its listed
//                      neighbours (ridge push, plateau collapse), as a change
//                      of this step's predicted position against basal drag
//   columns_flow       after edges_maxwell: the column's distributed inelastic
//                      strain tensor, fitted from its edges (normal and
//                      ductile tangential parts; phase G, src/columns/shape.js).
//                      Volumetric flow: d ln area_ref = its trace (2 x the
//                      edges' mean inelastic strain when it cannot be fitted).
//                      Crust volume is unchanged; thickness follows the actual
//                      (tessellated) area, area.wgsl. Its deviatoric part
//                      elongates the reference shape, up to the aspect cap

@compute @workgroup_size(64)
fn columns_flow(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    let n = atomicLoad(&particle_deltas[idx].count);
    if (n == 0i) { return; }
    let mean = f32(atomicLoad(&particle_deltas[idx].dx)) / FIXED_POINT / f32(n);

    // The column's inelastic strain tensor (fitStrain in shape.js): least
    // squares over its surface edges, two equations each, in its body frame;
    // unknowns exx, eyy, exy and w, the neighbourhood's rotation relative to
    // the column's spin.
    let x = particles[idx].predicted_pos.xyz;
    let m = min(atomicLoad(&particle_deltas[idx].slot_count), MAX_SLOTS);
    var A = mat4x4f(vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));
    var r = vec4f(0.0);
    for (var k = 0u; k < m; k++) {
        let e = particle_deltas[idx].slots[k];
        let j = other_end(e, idx);
        if (particles[j].state != 1u) { continue; }
        let u = body_frame(idx, x, particles[j].predicted_pos.xyz - x);
        let l = length(u);
        if (!(l > 0.0)) { continue; }
        let nx = u.x / l;
        let ny = u.y / l;
        let an = vec4f(nx * nx, ny * ny, 2.0 * nx * ny, 0.0);
        let at = vec4f(-nx * ny, nx * ny, nx * nx - ny * ny, 1.0);
        A += mat4x4f(an * an.x, an * an.y, an * an.z, an * an.w) + mat4x4f(at * at.x, at * at.y, at * at.z, at * at.w);
        r += an * edges[e].inelastic_strain + at * edges[e].inelastic_shear;
    }
    let det = determinant(A);
    let tr = A[0][0] + A[1][1] + A[2][2] + A[3][3];
    let fitted = particles[idx].state == 1u && abs(det) > 1e-6 * tr * tr * tr * tr;

    // Volumetric flow (columnFlow in flow.js): the fitted trace. Where the
    // tensor cannot be fitted (fewer than two edge directions), 2 x the mean
    // only for a column with at least three edges; else no change: the mean
    // of a one-sided neighbourhood is not its dilatation.
    var dln = select(0.0, 2.0 * mean, n >= 3i);
    var exx = 0.0;
    var eyy = 0.0;
    var exy = 0.0;
    var w = 0.0;
    if (fitted) {
        w = determinant(mat4x4f(A[0], A[1], A[2], r)) / det;
        exx = determinant(mat4x4f(r, A[1], A[2], A[3])) / det;
        eyy = determinant(mat4x4f(A[0], r, A[2], A[3])) / det;
        exy = determinant(mat4x4f(A[0], A[1], r, A[3])) / det;
        dln = exx + eyy;
    }
    particles[idx].area_ref *= exp(sim.volumetric_flow_factor * dln);
    particles[idx].apothem = sqrt(particles[idx].area_ref / 3.4641016);

    // Shape (updateShape in shape.js): turned with the material (by w
    // relative to the body frame, which turns with the spin), then the
    // deviatoric part, up to the aspect cap.
    if (sim.shape_flow_factor <= 0.0 || !fitted) { return; }
    let ang = 2.0 * sim.shape_flow_factor * w;
    let p0 = particles[idx].shape_p;
    let q0 = particles[idx].shape_q;
    var sp = cos(ang) * p0 - sin(ang) * q0 + sim.shape_flow_factor * 0.5 * (exx - eyy);
    var sq = sin(ang) * p0 + cos(ang) * q0 + sim.shape_flow_factor * exy;
    let sl = length(vec2f(sp, sq));
    let smax = 0.5 * log(sim.shape_aspect_cap);
    if (sl > smax) { sp *= smax / sl; sq *= smax / sl; }
    particles[idx].shape_p = sp;
    particles[idx].shape_q = sq;
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
}
