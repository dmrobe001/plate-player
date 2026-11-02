// Edge law (design 4.5, phase C). CPU reference: src/edges/law.js, which
// documents the rules; this file follows it line by line.
//
//   edges_law      before the solve: the edge's slices over the depth interval
//                  both columns share; from the stored slice stresses, this
//                  step's linear force law (stiffness, zero-force gap) and the
//                  multiplier bounds (depth-integrated caps); the same for the
//                  tangential channel and the face moment (phase G)
//   edges_maxwell  after the solve: advance each slice's Maxwell stress by the
//                  step's strain, cap it, store it; mean stress; then the
//                  irreversible part (phase D): inelastic strain (accumulated
//                  on both columns for volumetric flow, flow.wgsl), plastic
//                  work, and the integrity law (damage by work, thermal healing);
//                  then the edge timer and suturing (src/edges/transitions.js;
//                  subduction is the trench hinge's, hinge.wgsl)

const EDGE_SLICES: u32 = 8u;

// Depth extent of a column below its top, km: crust and lithosphere.
fn column_depth_km(i: u32) -> f32 { return max(lithosphere_thickness_km(i), particles[i].thickness_km); }

// Lithostatic pressure at depth d (km) in column i, from its profile: Taylor
// expansion about the nearest slice midpoint (first of equals).
fn slice_pressure(i: u32, d: f32) -> f32 {
    var k = 0u;
    var best = 1e30;
    for (var m = 0u; m < SLICES; m++) {
        let e = abs(d - profiles[i].depth_km[m]);
        if (e < best) { best = e; k = m; }
    }
    return profiles[i].pressure_mpa[k] + profiles[i].density[k] * GRAVITY * (d - profiles[i].depth_km[k]) * 1000.0 / 1e6;
}

struct EdgeSlices {
    overlap: f32,
    dz: f32,
    r: array<f32, 8>,       // Maxwell factor 1 / (1 + E dt / eta)
    comp: array<f32, 8>,    // caps, MPa
    tens: array<f32, 8>,
    cracks: array<u32, 8>,  // 1 = brittle (cracks in tension), 0 = ductile (flows)
    temp: array<f32, 8>,    // K, mean of the two columns at this depth
    shear: array<f32, 8>,   // tangential cap, MPa (phase G)
    slips: array<u32, 8>,   // 1 = the tangential cap is friction (slip is a fault)
}

// sn: the slices' stored normal stresses (+ tension), which load the faces in friction.
fn edge_slices(a: u32, b: u32, phi: f32, sn: array<f32, 8>) -> EdgeSlices {
    var s: EdgeSlices;
    let ea = particles[a].elevation;
    let eb = particles[b].elevation;
    let top = min(ea, eb);
    let bottom = max(ea - column_depth_km(a), eb - column_depth_km(b));
    s.overlap = max(top - bottom, 0.0);
    s.dz = s.overlap / f32(EDGE_SLICES);
    let ca = column_of(a);
    let cb = column_of(b);
    let ln_edt = log(sim.crust_youngs_modulus_gpa * 1e9 * sim.dt_crust_myr * 3.15576e13);
    // Subduction interface (exactly one slab end): lubricated in compression.
    // At a trench hinge the contact is a fault dipping at the flap's bend: a
    // vertical face cannot slip under horizontal push, and the resistance falls
    // to the lubricated interface as the dip reaches the optimal thrust angle
    // (design 6; contactWeakening in hinge.js).
    var lubricated = select(1.0, sim.slab_interface_weakening, (particles[a].state == 2u) != (particles[b].state == 2u));
    if (particles[a].state == 1u && particles[a].hinge_partner == b) { lubricated = min(lubricated, contact_weakening(particles[a].hinge_angle)); }
    if (particles[b].state == 1u && particles[b].hinge_partner == a) { lubricated = min(lubricated, contact_weakening(particles[b].hinge_angle)); }
    let ln_eta_shift = log(1e6) - log(2.0 * sim.reference_strain_rate_s);
    for (var k = 0u; k < EDGE_SLICES; k++) {
        let z = top - (f32(k) + 0.5) * s.dz;
        let da = ea - z;
        let db = eb - z;
        let ln_a = ln_ductile(ca, da, temp_at(ca, da));
        let ln_b = ln_ductile(cb, db, temp_at(cb, db));
        // Harmonic means of viscosity (uncapped) and of ductile strength (capped).
        let edt_over_eta = 0.5 * (exp(ln_edt - ln_a - ln_eta_shift) + exp(ln_edt - ln_b - ln_eta_shift));
        let ductile_cap = 1.0 / (0.5 * (exp(-min(ln_a, LN_STRENGTH_CAP)) + exp(-min(ln_b, LN_STRENGTH_CAP))));
        let pressure = 0.5 * (slice_pressure(a, da) + slice_pressure(b, db));
        let brittle_comp = phi * sim.cohesion_mpa + sim.friction_coefficient * (1.0 - sim.pore_fluid_factor) * pressure;
        let brittle = brittle_comp < ductile_cap;
        s.r[k] = 1.0 / (1.0 + edt_over_eta);
        s.comp[k] = min(brittle_comp, ductile_cap) * lubricated;
        // Brittle: Anderson's fraction of the frictional strength (deviatoric extension; law.js).
        let q = (sqrt(1.0 + sim.friction_coefficient * sim.friction_coefficient) + sim.friction_coefficient);
        let brittle_tens = phi * (sim.cohesion_mpa + sim.friction_coefficient * (1.0 - sim.pore_fluid_factor) * pressure) / (q * q);
        s.tens[k] = select(ductile_cap * (1.0 - sim.ductile_damage_weakening * (1.0 - phi)), brittle_tens, brittle);
        s.cracks[k] = select(0u, 1u, brittle);
        s.temp[k] = 0.5 * (temp_at(ca, da) + temp_at(cb, db));
        let brittle_shear = phi * sim.cohesion_mpa + sim.friction_coefficient * max((1.0 - sim.pore_fluid_factor) * pressure - sn[k], 0.0);
        s.shear[k] = min(brittle_shear, ductile_cap);
        s.slips[k] = select(0u, 1u, brittle_shear < ductile_cap);
    }
    return s;
}

@compute @workgroup_size(64)
fn edges_law(@builtin(global_invocation_id) id: vec3<u32>) {
    let e = id.x;
    if (e >= sim.max_edges || edges[e].is_active == 0u) { return; }
    let c = edges[e];
    let a = c.particle_a;
    let b = c.particle_b;
    let s = edge_slices(a, b, c.integrity, c.slice_stress_mpa);
    let E = sim.crust_youngs_modulus_gpa * 1000.0;
    var S0 = 0.0;
    var K = 0.0;
    var scomp = 0.0;
    var stens = 0.0;
    var S0t = 0.0;
    var scap_t = 0.0;
    for (var k = 0u; k < EDGE_SLICES; k++) {
        S0 += c.slice_stress_mpa[k] * s.r[k] * s.dz;
        K += E * s.r[k] * s.dz;
        scomp += s.comp[k] * s.dz;
        stens += s.tens[k] * s.dz;
        S0t += c.slice_shear_mpa[k] * s.r[k] * s.dz;
        scap_t += s.shear[k] * s.dz;
    }
    // Rest length from the two reference shapes along the pair (phase G);
    // contact point at the lever fraction r_a / rest from a.
    let xa = particles[a].predicted_pos.xyz;
    let xb = particles[b].predicted_pos.xyz;
    let rest = contact_length(a, b, xa, xb);
    edges[e].rest = rest;
    edges[e].lever_a = column_radius(a, xa, xb - xa) / rest;
    let keff = max(K, 1e-6 * E * sim.stiffness_reference_km);
    let scale = sim.bond_rate_myr_inv * sim.dt_crust_myr * rest / (E * sim.stiffness_reference_km);
    edges[e].stiffness = keff / (E * sim.stiffness_reference_km);
    // Zero-force length: the law is S = S0 + K (d - length) / rest.
    edges[e].reference_gap = c.length - S0 / keff * rest - rest;
    edges[e].lambda_min = -scale * stens;
    edges[e].lambda_max = scale * scomp;
    // Warm start (design 5.2): the force this edge applied in last step's
    // solve (edges_maxwell keeps it, MPa km), as this step's multiplier,
    // within the caps; crust_warm_start moves the positions to match. So the
    // solve starts from last step's motion relative to the mantle. (Not from
    // the stored stress: positions moved by that are an explicit Euler step of
    // the stiff edges, which amplifies whatever a short solve left unconverged.)
    let warm = select(0.0, clamp(scale * c.lambda, -scale * stens, scale * scomp), sim.solver_warm_start != 0u);
    edges[e].lambda = warm;
    edges[e].lambda_prev = warm;
    edges[e].overlap_km = s.overlap;

    // Tangential channel and face moment (linearizeShear, linearizeRolling in
    // law.js): surface pairs only (slab interfaces are normal-only).
    let surface_pair = particles[a].state == 1u && particles[b].state == 1u;
    var face = 0.0;
    for (var k = 0u; k < particle_deltas[a].cell_count; k++) {
        if (particle_deltas[a].cell_nbr[k] == b) { face = particle_deltas[a].cell_coef[k] * distance(particles[a].position.xyz, particles[b].position.xyz); }
    }
    edges[e].face_length = face;
    let ratio = sim.shear_stiffness_ratio;
    edges[e].stiffness_t = 0.0;
    edges[e].reference_t = 0.0;
    edges[e].lambda_t_max = 0.0;
    if (surface_pair && ratio > 0.0 && s.overlap > 0.0) {
        let kt = max(ratio * K, 1e-6 * ratio * E * sim.stiffness_reference_km);
        edges[e].stiffness_t = kt / (E * sim.stiffness_reference_km);
        edges[e].reference_t = -S0t / kt * rest;
        edges[e].lambda_t_max = scale * scap_t;
    }
    edges[e].stiffness_r = 0.0;
    edges[e].reference_r = 0.0;
    edges[e].lambda_r_max = 0.0;
    if (surface_pair && sim.rolling_stiffness_factor > 0.0 && face > 0.0 && s.overlap > 0.0) {
        let geom = sim.rolling_stiffness_factor * face * face / 12.0;
        let kr = keff * geom / rest;
        let m0 = c.moment * K / (E * s.overlap);
        edges[e].stiffness_r = edges[e].stiffness * geom;
        edges[e].reference_r = -m0 / kr;
        edges[e].lambda_r_max = scale * (max(-S0, 0.0) + stens) * face / 6.0;
    }
}

// After the solve (CPU reference: law.js, maxwellUpdate and integrityStep).
// Expects particle_deltas cleared: it accumulates each edge's inelastic
// strain on both columns (fixed point in dx, count in count) for columns_flow.
@compute @workgroup_size(64)
fn edges_maxwell(@builtin(global_invocation_id) id: vec3<u32>) {
    let e = id.x;
    if (e >= sim.max_edges || edges[e].is_active == 0u) { return; }
    let c = edges[e];
    let rest = rest_length(c);
    let offset = edge_offset(c.particle_a, c.particle_b, particles[c.particle_a].predicted_pos.xyz, particles[c.particle_b].predicted_pos.xyz);
    let len = length(offset);
    let d_strain = (len - c.length) / rest;
    let s = edge_slices(c.particle_a, c.particle_b, c.integrity, c.slice_stress_mpa);
    let E = sim.crust_youngs_modulus_gpa * 1000.0;
    var S = 0.0;
    var work = 0.0;
    var inelastic = 0.0;
    var r_max = -1.0;
    var w_sum = 0.0;
    var wt_sum = 0.0;
    for (var k = 0u; k < EDGE_SLICES; k++) {
        let elastic = c.slice_stress_mpa[k] + E * d_strain;
        let relaxed = elastic * s.r[k];
        let capped = clamp(relaxed, -s.comp[k], s.tens[k]);
        let crack = relaxed > s.tens[k] && s.cracks[k] == 1u;
        var kept = capped;
        if (crack) {
            kept = elastic - (1.0 / s.r[k] - 1.0) * s.tens[k];
            work += s.tens[k] * max(d_strain, 0.0) * s.dz;
        } else if (relaxed != capped) {
            work += abs(capped * (relaxed - capped) / (E * max(s.r[k], 1e-30))) * s.dz;
        }
        edges[e].slice_stress_mpa[k] = kept;
        S += capped * s.dz;
        // Inelastic strain of the stiffest slice (law.js, maxwellUpdate).
        if (s.r[k] > r_max) { r_max = s.r[k]; inelastic = d_strain - (kept - c.slice_stress_mpa[k]) / E; }
        let w = (s.r[k] + 1e-9) * s.dz;
        w_sum += w;
        wt_sum += w * s.temp[k];
    }
    // The force this step's solve applied, MPa km (+ push), kept for the
    // next step's warm start (edges_law): lambda / scale, the inverse of the
    // caps' scaling.
    edges[e].lambda = c.lambda * E * sim.stiffness_reference_km / (sim.bond_rate_myr_inv * sim.dt_crust_myr * rest);
    edges[e].length = len;
    edges[e].direction_x = offset.x;
    edges[e].direction_y = offset.y;
    edges[e].direction_z = offset.z;
    edges[e].strain = (len - rest) / rest;
    edges[e].stress_normal_mpa = select(0.0, S / s.overlap, s.overlap > 0.0);
    // An edge in its first step has deformed nothing: what its slices relax
    // is the load it was born with (from the rest length), not flow, so it is
    // not inelastic strain of the columns (phase G: in hot rock, gaps of new
    // neighbours read as dilatation, e.g. after neighbour exchange in shear).
    let newborn = c.born_myr == sim.time_myr;
    // Nor is underthrust at a trench hinge: the flap goes under, it is not compacted.
    let trench = hinge_pair(c.particle_a, c.particle_b);
    if (!(s.overlap > 0.0) || newborn || trench) { inelastic = 0.0; }
    // As a strain of the columns, per length, not per rest length (they differ
    // where a shape has met its cap): the fitted tensor is then the true
    // (logarithmic) strain increment, and area-preserving flow keeps the area.
    let per_length = rest / max(c.length, 1e-9);
    edges[e].inelastic_strain = inelastic * per_length;

    // Tangential channel (maxwellShear in law.js): the faces' sliding this
    // step; inelastic shear of the stiffest slice is fault slip if that slice
    // is brittle and at its cap, else distributed (it changes the shapes).
    let surface_pair = particles[c.particle_a].state == 1u && particles[c.particle_b].state == 1u;
    var St = 0.0;
    var shear_inelastic = 0.0;
    var fault = false;
    let G = sim.shear_stiffness_ratio * E;
    if (c.stiffness_t > 0.0 && G > 0.0) {
        let dg = tangential_displacement(c) / rest;
        var rt = -1.0;
        for (var k = 0u; k < EDGE_SLICES; k++) {
            let relaxed = (c.slice_shear_mpa[k] + G * dg) * s.r[k];
            let capped = clamp(relaxed, -s.shear[k], s.shear[k]);
            edges[e].slice_shear_mpa[k] = capped;
            St += capped * s.dz;
            if (relaxed != capped) { work += abs(capped * (relaxed - capped) / (G * max(s.r[k], 1e-30))) * s.dz; }
            if (s.r[k] > rt) { rt = s.r[k]; shear_inelastic = dg - (capped - c.slice_shear_mpa[k]) / G; fault = s.slips[k] == 1u && relaxed != capped; }
        }
    } else {
        for (var k = 0u; k < EDGE_SLICES; k++) { edges[e].slice_shear_mpa[k] = 0.0; }
    }
    if (!(s.overlap > 0.0) || newborn || trench) { shear_inelastic = 0.0; }
    edges[e].stress_shear_mpa = select(0.0, St / s.overlap, s.overlap > 0.0);
    edges[e].slip = select(0.0, shear_inelastic, fault) * per_length;
    edges[e].inelastic_shear = select(shear_inelastic, 0.0, fault) * per_length;
    // Face moment (rollingUpdate in law.js): the kern-rule bound in lambda
    // units over the scale gives the cap.
    if (c.stiffness_r > 0.0) {
        var K = 0.0;
        for (var k = 0u; k < EDGE_SLICES; k++) { K += E * s.r[k] * s.dz; }
        let keff = max(K, 1e-6 * E * sim.stiffness_reference_km);
        let kr = keff * sim.rolling_stiffness_factor * c.face_length * c.face_length / 12.0 / rest;
        let scale = sim.bond_rate_myr_inv * sim.dt_crust_myr * rest / (E * sim.stiffness_reference_km);
        let cap = c.lambda_r_max / scale;
        let dtheta = particles[c.particle_b].spin - particles[c.particle_a].spin;
        edges[e].moment = clamp(c.moment * K / (E * s.overlap) + kr * dtheta, -cap, cap);
    } else {
        edges[e].moment = 0.0;
    }
    edges[e].work_mpa_km = work;

    // Volumetric flow: both columns collect this edge's inelastic strain
    // (surface pairs only: slabs are not part of the area budget).
    if (surface_pair) {
        let fixed = i32(round(inelastic * FIXED_POINT));
        atomicAdd(&particle_deltas[c.particle_a].dx, fixed);
        atomicAdd(&particle_deltas[c.particle_a].count, 1i);
        atomicAdd(&particle_deltas[c.particle_b].dx, fixed);
        atomicAdd(&particle_deltas[c.particle_b].count, 1i);
    }

    // Integrity law: damage by plastic work, then exact exponential healing
    // at the temperature of the load-bearing part of the interface.
    let t_heal = select(sim.mantle_temp, wt_sum / w_sum, w_sum > 0.0);
    let damaged = max(c.integrity - work / sim.damage_work_mpa_km, 0.0);
    let exponent = min((sim.heal_activation_kj / sim.gas_constant) * (1.0 / max(t_heal, 1.0) - 1.0 / sim.mantle_temp), 80.0);
    let tau = sim.heal_time_myr * exp(exponent);
    edges[e].integrity = 1.0 - (1.0 - damaged) * exp(-sim.dt_crust_myr / tau);

    // Suturing (design 6; transitions.js). Subduction is not decided here: a
    // column goes down when its trench hinge passes its back face (hinge.wgsl).
    let timer = select(0.0, c.timer + sim.dt_crust_myr, edges[e].stress_normal_mpa < -sim.transition_stress_mpa);
    edges[e].timer = timer;
    if (!surface_pair) { return; }
    let buoyant = profiles[c.particle_a].freeboard_km >= 0.0 && profiles[c.particle_b].freeboard_km >= 0.0;
    if (buoyant && timer >= sim.suture_window_myr && work == 0.0) {
        edges[e].integrity = 1.0;
        edges[e].timer = 0.0;
    }
}
