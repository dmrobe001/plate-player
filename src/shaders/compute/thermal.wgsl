// Crust thermal state, isostasy (the free radial coordinate) and edge healing.

@compute @workgroup_size(64)
fn crust_thermo_isostasy(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }

    particles[idx].age += sim.dt_crust_myr;
    var temp = particles[idx].temperature;

    if (particles[idx].state == 1u) {
        // Surface crust relaxes toward ocean-bottom temperature (exact exponential step).
        temp = sim.ambient_ocean_temp + (temp - sim.ambient_ocean_temp) * exp(-sim.surface_cooling_rate * sim.dt_crust_myr);
    } else if (particles[idx].state == 2u) {
        // Subducting slab heats toward mantle temperature and loses thickness
        // in proportion to the heat it absorbs (assimilation).
        let temp_inc = (sim.mantle_temp - temp) * (1.0 - exp(-sim.slab_heating_rate * sim.dt_crust_myr));
        temp += temp_inc;
        particles[idx].thickness_km -= temp_inc * sim.slab_assimilation_km_per_k;
        if (particles[idx].thickness_km <= 1.0) {
            particles[idx].is_active = 0u;
            release(true, idx);
            return;
        }
    }
    particles[idx].temperature = temp;
    update_density(idx);
    if (particles[idx].state == 1u) {
        // Free radius (design 4.4): the column top relaxes toward its Airy
        // height e_eq. Energy (1/2) drho g A (e - e_eq)^2 against radial drag
        // with the Haskell rebound time tau; one implicit Euler step, as the
        // XPBD solve would take once bending couples neighbouring columns.
        let eq = airy_elevation(idx);
        let k = sim.dt_crust_myr / isostatic_time_myr(idx, eq);
        particles[idx].elevation = (particles[idx].elevation + k * eq) / (1.0 + k);
    } else {
        // Slab: no restoring equilibrium; sinks at a fixed rate (stand-in, design phase 7).
        particles[idx].elevation -= sim.slab_sink_rate_km_myr * sim.dt_crust_myr;
    }
}

// Colder rock is denser; composition_density is defined at mantle temperature.
fn update_density(idx: u32) {
    particles[idx].density = particles[idx].composition_density * (1.0 + sim.thermal_expansion_coeff * (sim.mantle_temp - particles[idx].temperature));
}

// Airy isostasy: freeboard of a floating column above the mantle datum,
// e0 = H (1 - rho / rho_m). Where that is below sea level the column is also
// loaded by water of depth (sea_level - e): balancing
// rho H + rho_w (s - e) = rho_m (H - e) gives e = e0 - (s - e0) rho_w / (rho_m - rho_w).
fn airy_elevation(idx: u32) -> f32 {
    let e0 = particles[idx].thickness_km * (1.0 - particles[idx].density / sim.mantle_density);
    if (e0 >= sim.sea_level) { return e0; }
    return e0 - (sim.sea_level - e0) * sim.water_density / (sim.mantle_density - sim.water_density);
}

// Viscous rebound time of a load of wavelength lambda = 4 apothems (Haskell):
// tau = 4 pi eta / (drho g lambda), drho = rho_m (or rho_m - rho_w under water).
fn isostatic_time_myr(idx: u32, eq: f32) -> f32 {
    let drho = select(sim.mantle_density, sim.mantle_density - sim.water_density, eq < sim.sea_level);
    let wavelength_m = 4.0 * particles[idx].apothem * 6.371e6;
    return 4.0 * 3.14159265 * sim.isostatic_viscosity_pa_s / (drho * 9.81 * wavelength_m) / 3.15576e13;
}

// Recompute derived quantities without advancing time (after loading a
// state). Columns loaded without an elevation (NaN) start at their Airy
// height. NaN is tested on the bits: WGSL compilers may assume floats are
// never NaN and fold `e != e` away.
@compute @workgroup_size(64)
fn crust_refresh_derived(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    update_density(idx);
    if ((bitcast<u32>(particles[idx].elevation) & 0x7f800000u) == 0x7f800000u) { particles[idx].elevation = airy_elevation(idx); }
}

// Healing restores integrity, faster when hot. An edge at zero integrity
// does not heal (no re-welding until suturing, phase F); stand-in until the
// integrity law of phase D.
@compute @workgroup_size(64)
fn crust_edge_healing(@builtin(global_invocation_id) id: vec3<u32>) {
    let c_idx = id.x;
    if (c_idx >= sim.max_edges || edges[c_idx].is_active == 0u || edges[c_idx].integrity <= 0.0) { return; }
    let rate = 100000.0 * arrhenius(edge_temperature(edges[c_idx]));
    edges[c_idx].integrity = clamp(edges[c_idx].integrity + rate * sim.dt_crust_myr, 0.0, 1.0);
}
