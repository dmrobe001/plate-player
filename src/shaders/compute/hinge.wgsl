// The trench hinge and hanging slabs (design 6). CPU reference:
// src/columns/hinge.js, which documents the mechanics.
//
//   columns_hinge   (after the edges are listed, before the edge law) each
//                   surface column finds its trench partner, solves the bend
//                   of its flap, decides whether the hinge has passed its back
//                   face (it then becomes a slab, crust_integrate), finds the
//                   column behind it (which a new slab will hang from), and
//                   pushes the column toward the trench with the down-dip
//                   pull of what hangs from it.
// Hanging slabs (attached_to != NONE) are handled in thermal.wgsl; they have
// no edges (edges.wgsl, in_range).

const PI_F: f32 = 3.14159265;

fn is_hanging(i: u32) -> bool { return particles[i].state == 2u && particles[i].attached_to != NONE; }

// Depth of the flap tip below the column's top, km.
fn flap_tip_depth_km(i: u32) -> f32 {
    return particles[i].hinge_flap_km * sin(particles[i].hinge_angle);
}

// Elevation where a slab hanging from p has its top: p's flap tip (surface)
// or p's bottom (a slab).
fn hang_point_elevation(p: u32) -> f32 {
    if (particles[p].state == 1u) { return particles[p].elevation - flap_tip_depth_km(p); }
    return particles[p].elevation - particles[p].slab_extent_km;
}

// Compression cap factor of a trench contact dipping at beta (contactWeakening):
// from 1 (a vertical face) to the interface's slab_interface_weakening at the
// optimal thrust dip, theta* = atan(1 / mu) / 2.
fn contact_weakening(beta: f32) -> f32 {
    let opt = 0.5 * atan(1.0 / sim.friction_coefficient);
    return mix(1.0, sim.slab_interface_weakening, clamp(beta / opt, 0.0, 1.0));
}

// The hinge pair: i's flap goes under j.
fn hinge_pair(a: u32, b: u32) -> bool {
    return (particles[a].state == 1u && particles[a].hinge_partner == b) || (particles[b].state == 1u && particles[b].hinge_partner == a);
}

// Plastic moment (N m / m) and tensile strength (N / m) of column i from its
// profile slices (columnStrength in hinge.js).
struct ColumnStrength { moment: f32, tension: f32 }
fn column_strength(i: u32) -> ColumnStrength {
    let mu = sim.friction_coefficient;
    var s: array<f32, 8>;
    var total = 0.0;
    for (var k = 0u; k < SLICES; k++) {
        let brittle = sim.cohesion_mpa + mu * (1.0 - sim.pore_fluid_factor) * profiles[i].pressure_mpa[k];
        s[k] = min(brittle, profiles[i].ductile_mpa[k]);
        total += s[k] * profiles[i].thickness_km[k];
    }
    var acc = 0.0;
    var zc = profiles[i].depth_km[SLICES - 1u];
    for (var k = 0u; k < SLICES; k++) {
        let w = s[k] * profiles[i].thickness_km[k];
        if (acc + w >= 0.5 * total) {
            zc = profiles[i].depth_km[k] - 0.5 * profiles[i].thickness_km[k] + select(0.0, (0.5 * total - acc) / w * profiles[i].thickness_km[k], w > 0.0);
            break;
        }
        acc += w;
    }
    var out: ColumnStrength;
    for (var k = 0u; k < SLICES; k++) {
        out.moment += s[k] * profiles[i].thickness_km[k] * abs(profiles[i].depth_km[k] - zc);
        // The edge law's tension cap at full integrity: brittle / q (Anderson), else ductile.
        let brittle = sim.cohesion_mpa + mu * (1.0 - sim.pore_fluid_factor) * profiles[i].pressure_mpa[k];
        let q = (sqrt(1.0 + mu * mu) + mu) * (sqrt(1.0 + mu * mu) + mu);
        out.tension += select(profiles[i].ductile_mpa[k], brittle / q, brittle < profiles[i].ductile_mpa[k]) * profiles[i].thickness_km[k];
    }
    out.moment *= 1e12;
    out.tension *= 1e9;
    return out;
}

@compute @workgroup_size(64)
fn columns_hinge(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u || particles[idx].state != 1u) { return; }
    // Geometry from start-of-step positions (predicted ones are being pushed by other threads).
    let xi = particles[idx].position.xyz;
    let fi = profiles[idx].freeboard_km;
    let n = min(atomicLoad(&particle_deltas[idx].slot_count), MAX_SLOTS);

    // Trench partner: a more buoyant surface neighbour across a broken edge
    // (this column can sink), the one it is furthest under; else a slab
    // hanging from this column.
    var partner = NONE;
    var edge = NONE;
    var best = -1e30;
    if (fi < 0.0) {
        for (var k = 0u; k < n; k++) {
            let e = particle_deltas[idx].slots[k];
            if (edges[e].is_active == 0u || edges[e].integrity >= 0.5) { continue; }
            let j = other_end(e, idx);
            if (particles[j].state != 1u) { continue; }
            let fj = profiles[j].freeboard_km;
            if (!(fi < fj)) { continue; } // equal columns have no polarity: neither goes down
            let xj = particles[j].position.xyz;
            let ov = contact_length(idx, j, xi, xj) - distance(xi, xj);
            if (ov > best) { best = ov; partner = j; edge = e; }
        }
    }
    var u_unit = max(best, 0.0);
    if (partner == NONE && particles[idx].slab_child != NONE) {
        partner = particles[idx].slab_child;
        let xs = particles[partner].position.xyz;
        u_unit = max(contact_length(idx, partner, xi, xs) - distance(xi, xs), 0.0);
    }
    particles[idx].hinge_partner = partner;
    if (partner == NONE) {
        particles[idx].hinge_angle = 0.0;
        particles[idx].hinge_plastic = 0.0;
        particles[idx].hinge_flap_km = 0.0;
        particles[idx].hinge_underthrust_km = 0.0;
        particles[idx].attached_to = NONE;
        return;
    }

    // Flexural setback and stiffness (hinge.js: flexuralLength, hingeSetback, hingeStiffness).
    let D = particles[idx].flexural_rigidity;
    let K = select(sim.mantle_density, sim.mantle_density - sim.water_density, profiles[idx].airy_elevation_km < sim.sea_level) * 9.81;
    let ell = sqrt(sqrt(4.0 * D / K));
    let xs = PI_F * ell / 4.0;
    let kh = select(0.0, D / ell, ell > 0.0);
    let u = u_unit * R_M;
    let Lf = xs + u;
    let width = 2.0 * particles[idx].apothem * R_M;

    // Torques (hingeTorques): what hangs from the tip; the underthrust part's
    // negative buoyancy; compression across the trench.
    var P = 0.0;
    if (edge != NONE) { P = max(-edges[edge].stress_normal_mpa * edges[edge].overlap_km, 0.0) * 1e9; }
    let Fs = particles[idx].child_load / width;
    // The underthrust part hangs like a slab: its negative buoyancy, less the
    // viscous support of its descent (the flap sinks at du/dt sin(beta) against
    // its free rate; hangSlab in hinge.js).
    let free_rate = slab_sink_rate_km_myr(idx);
    let sink = max(u / 1000.0 - particles[idx].hinge_underthrust_km, 0.0) / sim.dt_crust_myr * sin(particles[idx].hinge_angle);
    let support = select(0.0, clamp(sink / free_rate, 0.0, 1.0), free_rate > 0.0);
    let wu = select(0.0, -sim.mantle_density * fi * 1000.0 * 9.81, fi < 0.0) * (1.0 - support);
    let drive = Fs * Lf + wu * (Lf * Lf - xs * xs) * 0.5;
    let stiff = K * xs * xs * xs / 3.0 - P * Lf;
    let My = column_strength(idx).moment;

    // Quasi-static bend (solveHinge).
    let bmax = sim.hinge_max_angle;
    let bp = particles[idx].hinge_plastic;
    var beta = 0.0;
    var beta_p = bp;
    var done = false;
    let den = stiff + kh;
    if (den > 0.0) {
        let b = (drive + kh * bp) / den;
        if (abs(kh * (b - bp)) <= My) { beta = b; done = true; }
    }
    if (!done) {
        let sgn = select(1.0, -1.0, den > 0.0 && drive + kh * bp < 0.0);
        if (stiff > 0.0) {
            beta = clamp((drive - sgn * My) / stiff, 0.0, bmax);
            if (kh > 0.0) { beta_p = beta - sgn * My / kh; }
        } else {
            beta = bmax;
            if (kh > 0.0) { beta_p = max(bp, bmax - My / kh); }
        }
    }
    beta = clamp(beta, 0.0, bmax);
    particles[idx].hinge_angle = beta;
    particles[idx].hinge_plastic = beta_p;
    particles[idx].hinge_flap_km = Lf / 1000.0;
    particles[idx].hinge_underthrust_km = u / 1000.0;

    // Toward the trench, in the tangent plane.
    var nh = particles[partner].position.xyz - xi;
    nh = nh - dot(nh, xi) * xi;
    let nl = length(nh);
    if (nl > 0.0) { nh /= nl; }

    // The column behind: an intact surface neighbour, the one most opposite
    // the trench. A slab made from this column hangs from it.
    // Also what the plate behind can hold: the tension caps of the edges
    // behind the column (last step's bounds), projected on the pull.
    var behind = NONE;
    var most = 0.0;
    var hold = 0.0;
    let r = sim.shear_stiffness_ratio;
    let c_lattice = (3.0 + r) / (2.0 * sqrt(3.0) * (1.0 + r)); // normalStiffnessPerModulus
    for (var k = 0u; k < n; k++) {
        let e = particle_deltas[idx].slots[k];
        if (edges[e].is_active == 0u) { continue; }
        let j = other_end(e, idx);
        if (j == partner || particles[j].state != 1u) { continue; }
        let back = -dot(normalize(particles[j].position.xyz - xi), nh);
        if (back <= 0.0) { continue; }
        // lambda_min = -scale stens, scale = rate dt rest / (E h_ref) (law.wgsl): force = c stens rest R.
        let rest = max(edges[e].rest, 1e-9);
        let scale = sim.bond_rate_myr_inv * sim.dt_crust_myr * rest / (sim.crust_youngs_modulus_gpa * 1000.0 * sim.stiffness_reference_km);
        hold += c_lattice * (-edges[e].lambda_min / scale) * 1e9 * rest * R_M * back;
        if (edges[e].integrity >= 0.5 && back > most) { most = back; behind = j; }
    }
    particles[idx].attached_to = behind;

    // Subducts once the hinge has passed its back face (becomesSlab).
    if (u + xs >= width) { particles[idx].next_state = 2u; }

    // Slab pull: the down-dip component of what hangs from the flap (and of
    // the underthrust part itself), a body force toward the trench, applied
    // against basal drag like the GPE push (flow.wgsl).
    // At most what the column can transmit (its tensile strength across its
    // width) and what the plate behind it can hold (its edges' tension caps):
    // a bigger pull would tear the plate behind the hinge.
    let pull = min((particles[idx].child_load + wu * u * width) * sin(beta), min(column_strength(idx).tension * width, 0.9 * hold));
    if (pull > 0.0 && nl > 0.0) {
        let h = max(sim.asthenosphere_base_km - lithosphere_thickness_km(idx), sim.min_channel_km) * 1000.0;
        let gamma = particles[idx].area_ref * R_M * R_M * sim.asthenosphere_viscosity_pa_s / h;
        let dx = pull * sim.dt_crust_myr * 3.15576e13 / gamma / R_M;
        let x = vec4f(normalize(particles[idx].predicted_pos.xyz + dx * nh), 1.0);
        particles[idx].predicted_pos = x;
        particles[idx].iter_start = x;
        particles[idx].iter_prev = x;
    }
}
