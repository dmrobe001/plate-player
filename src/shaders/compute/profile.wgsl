// Column profile (design 4.0, phase B). CPU reference: src/columns/profile.js,
// which documents every formula; this file follows it line by line.
//
// column_profile: per column, from its state (crust thickness and
// composition, age, elevation): geotherm T(z), density, lithostatic
// pressure, ductile strength and viscosity on SLICES depth slices, and the
// column totals (lithosphere base, Airy elevation, GPE, heat flow, Moho
// temperature, brittle-ductile transition). Written to `profiles`; read by
// isostasy now, and by the edge law (phase C) and GPE forces (phase D).

const SLICES: u32 = 8u;
const CRUST_SLICES: u32 = 3u;
const CRUST_INTERVALS: u32 = 6u;
const MANTLE_INTERVALS: u32 = 16u;
const GRAVITY: f32 = 9.81;
const GL0: f32 = 0.21132487;  // 2-point Gauss-Legendre nodes on [0, 1]
const GL1: f32 = 0.78867513;
const LN10: f32 = 2.30258509;
const LN_STRENGTH_CAP: f32 = 13.8155106; // ln(1e6 MPa)

fn erf_approx(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return sign(x) * y;
}

struct Column {
    crust_km: f32,
    rho_crust: f32,      // composition density at mantle temperature
    continental: bool,
    elevation: f32,      // current elevation (water load for pressure)
    slab: bool,          // overburden is mantle rock, not water
    // geotherm
    lith_km: f32,
    z_mantle_km: f32,    // below this, T = mantle temperature
    s_km: f32,           // oceanic: 2 sqrt(kappa t_eff)
    hr_m: f32,           // continental: heat-producing layer
    a_w_m3: f32,         // continental: heat production
    qm: f32,             // continental: heat flow from below, W/m^2
    heat_flow: f32,      // surface, mW/m^2
}

fn column_of(i: u32) -> Column {
    var c: Column;
    c.crust_km = particles[i].thickness_km;
    c.rho_crust = particles[i].composition_density;
    c.continental = c.rho_crust < 2850.0;  // CONTINENTAL_DENSITY_MAX
    c.elevation = particles[i].elevation;
    c.slab = particles[i].state == 2u;
    // A slab's oceanic crust turns to eclogite (crustDensity in profile.js).
    if (c.slab && !c.continental) {
        let f = clamp((-c.elevation - (sim.eclogite_depth_km - 10.0)) / 20.0, 0.0, 1.0);
        c.rho_crust = mix(c.rho_crust, sim.eclogite_density, f);
    }
    c.lith_km = lithosphere_thickness_km(i);
    let ts = sim.ambient_ocean_temp;
    let tm = sim.mantle_temp;
    let k = sim.thermal_conductivity_w_m_k;
    if (c.continental) {
        c.hr_m = min(sim.radiogenic_layer_km, c.crust_km) * 1000.0;
        c.a_w_m3 = sim.radiogenic_heat_uw_m3 * 1e-6;
        let tl = ts + 0.9 * (tm - ts);
        let produced = 0.5 * c.a_w_m3 * c.hr_m * c.hr_m;
        c.qm = max((k * (tl - ts) - produced) / (max(c.lith_km, 1.0) * 1000.0), 1e-3);
        c.z_mantle_km = max((k * (tm - ts) - produced) / c.qm / 1000.0, c.lith_km);
        c.heat_flow = (c.qm + c.a_w_m3 * c.hr_m) * 1000.0;
    } else {
        c.s_km = c.lith_km / 1.16;
        c.z_mantle_km = 3.6 * c.s_km;
        let half_m = c.s_km * 500.0;
        let t_eff = max(half_m * half_m / sim.thermal_diffusivity_m2_s, 1e-3 * 3.15576e13);
        c.heat_flow = k * (tm - ts) / sqrt(3.14159265 * sim.thermal_diffusivity_m2_s * t_eff) * 1000.0;
    }
    return c;
}

fn temp_at(c: Column, z_km: f32) -> f32 {
    let ts = sim.ambient_ocean_temp;
    let tm = sim.mantle_temp;
    if (c.continental) {
        let z = z_km * 1000.0;
        let k = sim.thermal_conductivity_w_m_k;
        var t: f32;
        if (z < c.hr_m) { t = ts + ((c.qm + c.a_w_m3 * c.hr_m) * z - 0.5 * c.a_w_m3 * z * z) / k; }
        else { t = ts + (c.qm * z + 0.5 * c.a_w_m3 * c.hr_m * c.hr_m) / k; }
        return min(t, tm);
    }
    if (c.s_km > 0.0) { return ts + (tm - ts) * erf_approx(z_km / c.s_km); }
    return select(ts, tm, z_km > 0.0);
}

fn density_at(c: Column, z_km: f32) -> f32 {
    let rho0 = select(sim.mantle_density, c.rho_crust, z_km < c.crust_km);
    return rho0 * (1.0 + sim.thermal_expansion_coeff * (sim.mantle_temp - temp_at(c, z_km)));
}

fn column_end(c: Column) -> f32 { return max(c.crust_km, c.z_mantle_km); }

// What to integrate (no closures in WGSL): 0 = rho, 1 = 1 - rho / rho_m,
// 2 = GPE integrand (rho - rho_ref) (z_end - z) with the datum at depth e.
fn integrand(c: Column, kind: u32, z: f32, e: f32, z_end: f32) -> f32 {
    let rho = density_at(c, z);
    if (kind == 0u) { return rho; }
    if (kind == 1u) { return 1.0 - rho / sim.mantle_density; }
    return (rho - select(0.0, sim.mantle_density, z > e)) * (z_end - z);
}

fn integrate(c: Column, kind: u32, a: f32, b: f32, n: u32, e: f32, z_end: f32) -> f32 {
    if (!(b > a)) { return 0.0; }
    let h = (b - a) / f32(n);
    var s = 0.0;
    for (var i = 0u; i < n; i++) {
        let z0 = a + f32(i) * h;
        s += integrand(c, kind, z0 + GL0 * h, e, z_end) + integrand(c, kind, z0 + GL1 * h, e, z_end);
    }
    return 0.5 * h * s;
}

// One segment of the GPE integral, split at the datum (depth e) where rho_ref jumps.
fn gpe_segment(c: Column, a: f32, b: f32, n: u32, e: f32, z_end: f32) -> f32 {
    if (e > a && e < b) { return integrate(c, 2u, a, e, n, e, z_end) + integrate(c, 2u, e, b, n, e, z_end); }
    return integrate(c, 2u, a, b, n, e, z_end);
}

fn pressure_at(c: Column, z_km: f32) -> f32 {
    let water = select(sim.water_density, sim.mantle_density, c.slab) * max(sim.sea_level - c.elevation, 0.0) * 1000.0;
    let rock = integrate(c, 0u, 0.0, min(z_km, c.crust_km), 4u, 0.0, 0.0) + integrate(c, 0u, c.crust_km, z_km, 8u, 0.0, 0.0);
    return GRAVITY * (water + rock * 1000.0) / 1e6;
}

// Ductile shear strength, as ln(MPa), for the material at depth z.
fn ln_ductile(c: Column, z_km: f32, t: f32) -> f32 {
    var log10a = sim.olivine_log10_a;
    var n = sim.olivine_n;
    var q = sim.olivine_q_kj;
    if (z_km < c.crust_km) {
        if (c.continental) { log10a = sim.quartzite_log10_a; n = sim.quartzite_n; q = sim.quartzite_q_kj; }
        else { log10a = sim.diabase_log10_a; n = sim.diabase_n; q = sim.diabase_q_kj; }
    }
    let ln_sigma = (log(sim.reference_strain_rate_s) - log10a * LN10) / n + q / (n * sim.gas_constant * t);
    return ln_sigma + log(0.5);
}

// Anderson's theory of faulting (brittleCompression, brittleTension in
// src/columns/profile.js): horizontal differential stress at failure under
// the vertical effective stress, thrust and normal faults; integrity scales
// the cohesion in compression and the whole strength in tension (design 4.5).
fn anderson_q() -> f32 {
    let r = sqrt(1.0 + sim.friction_coefficient * sim.friction_coefficient) + sim.friction_coefficient;
    return r * r;
}
fn brittle_compression(p_mpa: f32, phi: f32) -> f32 {
    let q = anderson_q();
    return phi * 2.0 * sim.cohesion_mpa * sqrt(q) + (q - 1.0) * (1.0 - sim.pore_fluid_factor) * p_mpa;
}
fn brittle_tension(p_mpa: f32, phi: f32) -> f32 {
    let q = anderson_q();
    return phi * (2.0 * sim.cohesion_mpa / sqrt(q) + (1.0 - 1.0 / q) * (1.0 - sim.pore_fluid_factor) * p_mpa);
}

// Envelope in horizontal differential stress: ductile flow stress = 2 x shear strength.
fn is_ductile(c: Column, z_km: f32) -> bool {
    let d = 2.0 * exp(min(ln_ductile(c, z_km, temp_at(c, z_km)), LN_STRENGTH_CAP));
    return d < brittle_compression(pressure_at(c, z_km), 1.0);
}

@compute @workgroup_size(64)
fn column_profile(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    let c = column_of(idx);
    let z_end = column_end(c);

    // Airy isostasy (profile.js, airy).
    let freeboard = integrate(c, 1u, 0.0, min(c.crust_km, z_end), CRUST_INTERVALS, 0.0, 0.0)
        + integrate(c, 1u, c.crust_km, z_end, MANTLE_INTERVALS, 0.0, 0.0);
    let s = sim.sea_level;
    var e = freeboard;
    if (freeboard < s) { e = freeboard - (s - freeboard) * sim.water_density / (sim.mantle_density - sim.water_density); }

    // GPE at the Airy elevation (profile.js, gpe). Heights above the datum: top e, bottom rb.
    let rb = e - z_end;
    var gpe = gpe_segment(c, 0.0, c.crust_km, CRUST_INTERVALS, e, z_end)
        + gpe_segment(c, c.crust_km, z_end, MANTLE_INTERVALS, e, z_end);
    if (e < s) {
        // water from height e to s: rho_w, minus rho_m where below the datum
        gpe += sim.water_density * (s - e) * (0.5 * (s + e) - rb);
        let lo = e;
        let hi = min(0.0, s);
        if (hi > lo) { gpe -= sim.mantle_density * (hi - lo) * (0.5 * (hi + lo) - rb); }
    }

    var p: ColumnProfile;
    p.lithosphere_km = c.lith_km;
    p.column_km = max(c.lith_km, c.crust_km);
    p.freeboard_km = freeboard;
    p.airy_elevation_km = e;
    p.gpe_mpa_km = GRAVITY * gpe * 1e6 / 1e9;
    p.heat_flow_mw_m2 = c.heat_flow;
    p.moho_temp_k = temp_at(c, c.crust_km);

    // Slices (profile.js, sliceLayout), and the brittle-ductile transition:
    // the first ductile slice midpoint, then bisection from the one above.
    var lo = 0.0;
    var bdt = -1.0;
    for (var k = 0u; k < SLICES; k++) {
        let crust = k < CRUST_SLICES;
        let top = select(c.crust_km, 0.0, crust);
        let bottom = select(p.column_km, c.crust_km, crust);
        let n = select(SLICES - CRUST_SLICES, CRUST_SLICES, crust);
        let m = select(k - CRUST_SLICES, k, crust);
        let dz = (bottom - top) / f32(n);
        let z = top + (f32(m) + 0.5) * dz;
        let t = temp_at(c, z);
        let ln_d = ln_ductile(c, z, t);
        let pressure = pressure_at(c, z);
        let ductile = exp(min(ln_d, LN_STRENGTH_CAP));
        p.depth_km[k] = z;
        p.thickness_km[k] = dz;
        p.temp_k[k] = t;
        p.density[k] = density_at(c, z);
        p.pressure_mpa[k] = pressure;
        p.log10_viscosity[k] = (ln_d + log(1e6) - log(2.0 * sim.reference_strain_rate_s)) / LN10;
        p.ductile_mpa[k] = ductile;
        if (bdt < 0.0) {
            if (2.0 * ductile < brittle_compression(pressure, 1.0)) {
                var hi = z;
                for (var it = 0; it < 30; it++) {
                    let mid = 0.5 * (lo + hi);
                    if (is_ductile(c, mid)) { hi = mid; } else { lo = mid; }
                }
                bdt = 0.5 * (lo + hi);
            } else {
                lo = z;
            }
        }
    }
    if (bdt < 0.0) { bdt = p.depth_km[SLICES - 1u] + 0.5 * p.thickness_km[SLICES - 1u]; }
    p.brittle_ductile_km = bdt;
    profiles[idx] = p;
}
