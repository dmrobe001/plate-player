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

// Dip of the subducting sheet at trench column i's contact: its bend, or the
// dip of the slab hanging from it, whichever is steeper (with consumption the
// column passes straight into the slab, and the flap stays short; trenchDip in
// hinge.js). (The bend alone: a consumed column's short flap barely bent, the
// contact stayed a vertical face and pushed the overriding plate away.)
fn trench_dip(i: u32) -> f32 {
    let s = particles[i].slab_child;
    let slab_dip = select(0.0, particles[s].hinge_angle, s != NONE && particles[s].state == 2u && particles[s].attached_to == i);
    return max(particles[i].hinge_angle, slab_dip);
}

// The hinge pair: i's flap goes under j.
fn hinge_pair(a: u32, b: u32) -> bool {
    return (particles[a].state == 1u && particles[a].hinge_partner == b) || (particles[b].state == 1u && particles[b].hinge_partner == a);
}

// Plastic moment (N m / m) and tensile strength (N / m) of column i from its
// profile slices (columnStrength in hinge.js): the yield-strength envelope in
// horizontal differential stress, tension above the neutral plane and
// compression below it, the neutral plane where the two balance.
struct ColumnStrength { moment: f32, tension: f32 }
fn column_strength(i: u32) -> ColumnStrength {
    var st: array<f32, 8>;
    var sc: array<f32, 8>;
    var below: array<f32, 9>;
    for (var k = 0u; k < SLICES; k++) {
        let d = 2.0 * profiles[i].ductile_mpa[k];
        st[k] = min(brittle_tension(profiles[i].pressure_mpa[k], 1.0), d);
        sc[k] = min(brittle_compression(profiles[i].pressure_mpa[k], 1.0), d);
    }
    below[SLICES] = 0.0;
    for (var k = SLICES; k > 0u; k--) { below[k - 1u] = below[k] + sc[k - 1u] * profiles[i].thickness_km[k - 1u]; }
    var above = 0.0;
    var m = SLICES - 1u;
    var f = 1.0;
    for (var k = 0u; k < SLICES; k++) {
        let t = st[k] * profiles[i].thickness_km[k];
        let c = sc[k] * profiles[i].thickness_km[k];
        if (above + t >= below[k + 1u]) {
            m = k;
            f = select(0.5, clamp((below[k + 1u] + c - above) / (t + c), 0.0, 1.0), t + c > 0.0);
            break;
        }
        above += t;
    }
    let zc = profiles[i].depth_km[m] - 0.5 * profiles[i].thickness_km[m] + f * profiles[i].thickness_km[m];
    var out: ColumnStrength;
    for (var k = 0u; k < SLICES; k++) {
        let h = profiles[i].thickness_km[k];
        out.tension += st[k] * h;
        if (k < m) { out.moment += st[k] * h * (zc - profiles[i].depth_km[k]); }
        else if (k > m) { out.moment += sc[k] * h * (profiles[i].depth_km[k] - zc); }
        else { let u = f * h; let l = (1.0 - f) * h; out.moment += 0.5 * (st[k] * u * u + sc[k] * l * l); }
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
            // Polarity needs a buoyancy contrast: near-equal columns (within 50 m of
            // freeboard, e.g. two old ocean columns) have none, and neither goes down.
            if (!(fi < fj - 0.05)) { continue; }
            let xj = particles[j].position.xyz;
            let ov = contact_length(idx, j, xi, xj) - distance(xi, xj);
            if (ov > best) { best = ov; partner = j; edge = e; }
        }
    }
    var u_unit = max(best, 0.0);
    // No such neighbour, but a slab hangs from this column: the column it
    // came from has subducted, and the trench is still the overriding column
    // that one went under (the slab keeps it as its hinge_partner): this
    // column inherits it, its underthrust measured against it (a gap at
    // first, closing as the plate converges; trenchPartner in hinge.js).
    // (It took the slab itself as partner, at full overlap: it subducted at
    // once, and the trench died; design 9.2.)
    let child = particles[idx].slab_child;
    if (partner == NONE && child != NONE) {
        let j = particles[child].hinge_partner;
        if (j != NONE && j != idx && particles[j].is_active != 0u && particles[j].state == 1u) {
            partner = j;
            let xj = particles[j].position.xyz;
            u_unit = max(contact_length(idx, j, xi, xj) - distance(xi, xj), 0.0);
        } else {
            // No overriding column known (a seeded slab): the slab gives the
            // trench's direction only; its overlap with the column it hangs
            // from (it hangs at the front face) is not underthrust. (Taken as
            // underthrust, the column subducted in its first step.)
            partner = child;
            u_unit = 0.0;
        }
    }
    particles[idx].hinge_partner = partner;
    if (partner == NONE) {
        particles[idx].hinge_angle = 0.0;
        particles[idx].hinge_plastic = 0.0;
        particles[idx].hinge_flap_km = 0.0;
        particles[idx].hinge_underthrust_km = 0.0;
        particles[idx].attached_to = NONE;
        particles[idx].slab_drag_ratio = 0.0;
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
    let wu0 = select(0.0, -sim.mantle_density * fi * 1000.0 * 9.81, fi < 0.0); // per area, unsupported
    let wu = wu0 * (1.0 - support);
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

    // Slab pull (hingePull in hinge.js): the tension the slab carries, the
    // down-dip weight of what hangs from the flap (each slab at its own dip,
    // thermal.wgsl) and of the underthrust part itself (at the bend), a body
    // force toward the trench. Moving, the column drags them down dip through
    // the mantle: their drag coefficient, sum of W / w_free, is added to its
    // basal drag, in the solve (inverse_drag) and here; so a plate fed by a
    // vertical slab moves at about the slab's free sinking speed, and the
    // vertical support the slabs get (thermal.wgsl) is the vertical part of
    // the same drag. The pull is at most what the column can transmit
    // (its tensile strength across its width) and what the plate behind it
    // can hold (its edges' tension caps): more would tear the plate behind.
    // What it transmits is the slabs' tension at the trench, the down-dip
    // weight less their drag at the column's speed: gamma v. So the speed is
    // v = min(W_d / (gamma + drag), cap / gamma), exactly (hingePull in
    // hinge.js). (They bounded the gross down-dip weight: a plate could not
    // outrun the cap, its slab never reached its free speed, was never held
    // by the mantle, and necked. Then pull <= cap + drag x last step's speed:
    // with drag >> gamma that relaxed by a factor drag / (gamma + drag) per
    // step, ~0.9997 on the two-plate planet, and its slabs necked off; 9.2.)
    let h = max(sim.asthenosphere_base_km - lithosphere_thickness_km(idx), sim.min_channel_km) * 1000.0;
    let gamma = particles[idx].area_ref * R_M * R_M * sim.asthenosphere_viscosity_pa_s / h;
    let flap_weight = wu0 * u * width;
    let flap_rate = free_rate * 1000.0 / 3.15576e13;
    let drag = particles[idx].child_drag + select(0.0, flap_weight / flap_rate, flap_rate > 0.0);
    particles[idx].slab_drag_ratio = drag / gamma;
    let down_dip = particles[idx].child_pull + flap_weight * sin(beta);
    let cap = min(column_strength(idx).tension * width, 0.9 * hold);
    let v = min(down_dip / (gamma + drag), cap / gamma); // m/s
    if (v > 0.0 && nl > 0.0) {
        let dx = v * sim.dt_crust_myr * 3.15576e13 / R_M;
        let x = vec4f(normalize(particles[idx].predicted_pos.xyz + dx * nh), 1.0);
        particles[idx].predicted_pos = x;
    }
}

// Continuous consumption through the hinge (design 6; consumeStep and
// feedSlab in hinge.js). After the solve, before positions are committed: a
// trench column whose front has gone past touching its overriding partner,
// with a slab hanging from it, hands that strip (c, at most half the column
// per step) to the slab: the fraction c / 2a of its area and crust volume,
// which leaves the surface budget. It stays round with the smaller area and
// moves away from the trench by a - a', its back face fixed, position and
// predicted position alike (its velocity is unchanged); geometry_reset makes
// its edges take the new geometry as their memory next step, so the jump
// stores no stress. Once shorter than its setback the hinge makes the remnant
// a slab (becomesSlab), and the next column's material feeds it. (A column
// had to go its whole width under first: at 1,900 km, longer than its slab
// stayed heavy; design 9.2.)
@compute @workgroup_size(64)
fn columns_consume(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u || particles[idx].state != 1u || particles[idx].next_state == 2u) { return; }
    let j = particles[idx].hinge_partner;
    let s = particles[idx].slab_child;
    if (j == NONE || s == NONE) { return; }
    if (particles[j].is_active == 0u || particles[j].state != 1u || particles[s].state != 2u || particles[s].attached_to != idx) { return; }
    let xi = particles[idx].predicted_pos.xyz;
    let xj = particles[j].predicted_pos.xyz;
    let over = contact_length(idx, j, xi, xj) - distance(xi, xj);
    if (!(over > 0.0)) { return; }
    let a = particles[idx].apothem;
    let c = min(over, a);
    let f = c / (2.0 * a);
    let A = particles[idx].area_ref;
    let V = particles[idx].crust_volume;
    let dA = f * A;
    let dV = f * V;
    let A2 = A - dA;
    let a2 = sqrt(A2 / 3.4641016);
    // Away from the trench, in the tangent plane.
    var nh = xj - xi;
    nh = nh - dot(nh, xi) * xi;
    let nl = length(nh);
    if (!(nl > 0.0)) { return; }
    nh /= nl;
    let shift = a - a2;
    particles[idx].predicted_pos = vec4f(normalize(xi - shift * nh), particles[idx].predicted_pos.w);
    let x0 = particles[idx].position.xyz;
    particles[idx].position = vec4f(normalize(x0 - shift * nh), particles[idx].position.w);
    particles[idx].area_ref = A2;
    particles[idx].apothem = a2;
    particles[idx].crust_volume = V - dV;
    particles[idx].geometry_reset = 1u;
    add_budget(BUDGET_SINK, dV);
    // The slab gains it: volume-weighted composition, area-weighted thermal age.
    let Vs = particles[s].crust_volume;
    let As = particles[s].area_ref;
    particles[s].composition_density = (particles[s].composition_density * Vs + particles[idx].composition_density * dV) / max(Vs + dV, 1e-30);
    particles[s].thermal_age_myr = (particles[s].thermal_age_myr * As + particles[idx].thermal_age_myr * dA) / max(As + dA, 1e-30);
    particles[s].crust_volume = Vs + dV;
    particles[s].area_ref = As + dA;
    particles[s].apothem = sqrt((As + dA) / 3.4641016);
}
