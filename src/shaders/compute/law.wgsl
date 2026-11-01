// Edge law (design 4.5, phase C). CPU reference: src/edges/law.js, which
// documents the rules; this file follows it line by line.
//
//   edges_law      before the solve: the edge's slices over the depth interval
//                  both columns share; from the stored slice stresses, this
//                  step's linear force law (stiffness, zero-force gap) and the
//                  multiplier bounds (depth-integrated caps)
//   edges_maxwell  after the solve: advance each slice's Maxwell stress by the
//                  step's strain, cap it, store it; mean stress; stand-in damage

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
}

fn edge_slices(a: u32, b: u32, phi: f32) -> EdgeSlices {
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
        s.comp[k] = min(brittle_comp, ductile_cap);
        s.tens[k] = select(ductile_cap * (1.0 - sim.ductile_damage_weakening * (1.0 - phi)), phi * sim.cohesion_mpa, brittle);
        s.cracks[k] = select(0u, 1u, brittle);
    }
    return s;
}

@compute @workgroup_size(64)
fn edges_law(@builtin(global_invocation_id) id: vec3<u32>) {
    let e = id.x;
    if (e >= sim.max_edges || edges[e].is_active == 0u) { return; }
    let c = edges[e];
    let s = edge_slices(c.particle_a, c.particle_b, c.integrity);
    let E = sim.crust_youngs_modulus_gpa * 1000.0;
    var S0 = 0.0;
    var K = 0.0;
    var scomp = 0.0;
    var stens = 0.0;
    for (var k = 0u; k < EDGE_SLICES; k++) {
        S0 += c.slice_stress_mpa[k] * s.r[k] * s.dz;
        K += E * s.r[k] * s.dz;
        scomp += s.comp[k] * s.dz;
        stens += s.tens[k] * s.dz;
    }
    let keff = max(K, 1e-6 * E * sim.stiffness_reference_km);
    let rest = rest_length(c);
    let scale = sim.bond_rate_myr_inv * sim.dt_crust_myr * rest / (E * sim.stiffness_reference_km);
    edges[e].stiffness = keff / (E * sim.stiffness_reference_km);
    edges[e].reference_gap = (c.strain - S0 / keff) * rest;
    edges[e].lambda_min = -scale * stens;
    edges[e].lambda_max = scale * scomp;
    edges[e].overlap_km = s.overlap;
}

// After the solve; replaces the stand-in stress rule. Damage is still the
// stand-in (overstress of the mean normal stress vs an Arrhenius yield)
// until the integrity law of phase D.
@compute @workgroup_size(64)
fn edges_maxwell(@builtin(global_invocation_id) id: vec3<u32>) {
    let e = id.x;
    if (e >= sim.max_edges || edges[e].is_active == 0u) { return; }
    let c = edges[e];
    let rest = rest_length(c);
    let strain = (distance(particles[c.particle_b].predicted_pos.xyz, particles[c.particle_a].predicted_pos.xyz) - rest) / rest;
    let s = edge_slices(c.particle_a, c.particle_b, c.integrity);
    let E = sim.crust_youngs_modulus_gpa * 1000.0;
    var S = 0.0;
    for (var k = 0u; k < EDGE_SLICES; k++) {
        let elastic = c.slice_stress_mpa[k] + E * (strain - c.strain);
        let relaxed = elastic * s.r[k];
        let capped = clamp(relaxed, -s.comp[k], s.tens[k]);
        var stored = capped;
        if (relaxed > s.tens[k] && s.cracks[k] == 1u) { stored = elastic - (1.0 / s.r[k] - 1.0) * s.tens[k]; }
        edges[e].slice_stress_mpa[k] = stored;
        S += capped * s.dz;
    }
    edges[e].strain = strain;
    let sigma_n = select(0.0, S / s.overlap, s.overlap > 0.0);
    edges[e].stress_normal_mpa = sigma_n;
    edges[e].stress_shear_mpa = 0.0;
    if (c.integrity <= 0.0) { return; }

    // Stand-in damage (phase D replaces it): overstress lowers integrity.
    let temp = edge_temperature(c);
    let y = c.yield_stress_mpa * exp(clamp((sim.activation_energy / sim.gas_constant) * (1.0 / temp - 1.0 / sim.reference_temp), -10.0, 10.0));
    if (abs(sigma_n) > y) {
        edges[e].integrity = max(c.integrity - ((abs(sigma_n) - y) / y) * 50.0 * sim.dt_crust_myr, 0.0);
    }
}
