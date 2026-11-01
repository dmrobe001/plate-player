// Crust thermal state; slab sinking and removal (design 6, phase F). Surface
// elevations (the free radius with bending) are solved in flexure.wgsl.

@compute @workgroup_size(64)
fn crust_thermo_isostasy(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }

    particles[idx].age += sim.dt_crust_myr;
    var temp = particles[idx].temperature;

    if (particles[idx].state == 1u) {
        particles[idx].thermal_age_myr = min(particles[idx].thermal_age_myr + sim.dt_crust_myr, thermal_age_cap_myr(particles[idx].composition_density));
        // Surface crust relaxes toward ocean-bottom temperature (exact exponential step; display only).
        temp = sim.ambient_ocean_temp + (temp - sim.ambient_ocean_temp) * exp(-sim.surface_cooling_rate * sim.dt_crust_myr);
    } else {
        // Slab: warms (its thermal age, hence its cold lithosphere, decays)
        // and sinks under its own negative buoyancy (slabSinkRate in
        // src/edges/transitions.js). Removed once neutral or past the cutoff
        // (its volume left the surface budget when it subducted).
        let decay = exp(-sim.slab_heating_rate * sim.dt_crust_myr);
        particles[idx].thermal_age_myr *= decay;
        temp = sim.mantle_temp + (temp - sim.mantle_temp) * decay;
        let w = slab_sink_rate_km_myr(idx);
        let z0 = -particles[idx].elevation;
        particles[idx].elevation -= w * sim.dt_crust_myr;
        // Arc supply (flexure.wgsl): arc_efficiency x the slab's volume, in
        // proportion to the part of the arc depth window crossed this step.
        let crossed = max(min(-particles[idx].elevation, sim.arc_depth_max_km) - max(z0, sim.arc_depth_min_km), 0.0);
        particles[idx].arc_supply = sim.arc_efficiency * particles[idx].crust_volume * crossed / (sim.arc_depth_max_km - sim.arc_depth_min_km);
        if (w <= 0.0 || particles[idx].elevation < -sim.slab_cutoff_km) {
            particles[idx].is_active = 0u;
            release(true, idx);
            return;
        }
    }
    particles[idx].temperature = temp;
    update_density(idx);
}

// Stokes-like sinking of a slab column of width 2 x apothem: buoyancy per
// area B g, B = -rho_m x freeboard (kg/m^2), against upper-mantle viscosity:
// w = B g width / (4 eta), km/Myr; 0 if not negatively buoyant.
fn slab_sink_rate_km_myr(idx: u32) -> f32 {
    let excess = -sim.mantle_density * profiles[idx].freeboard_km * 1000.0;
    let width_m = 2.0 * particles[idx].apothem * EARTH_RADIUS_KM * 1000.0;
    let w = excess * 9.81 * width_m / (4.0 * sim.isostatic_viscosity_pa_s);
    return max(w, 0.0) * 3.15576e13 / 1000.0;
}

// Colder rock is denser; composition_density is defined at mantle temperature.
fn update_density(idx: u32) {
    particles[idx].density = particles[idx].composition_density * (1.0 + sim.thermal_expansion_coeff * (sim.mantle_temp - particles[idx].temperature));
}

// Airy isostasy of the whole column (crust and lithospheric mantle, with
// temperature-dependent densities), from its profile (design 4.0;
// column_profile must have run this step).
fn airy_elevation(idx: u32) -> f32 {
    return profiles[idx].airy_elevation_km;
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
