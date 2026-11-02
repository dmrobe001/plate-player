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
        if (particles[idx].attached_to != NONE && (sim.slab_attach_factor <= 0.0 || particles[particles[idx].attached_to].is_active == 0u)) {
            particles[idx].attached_to = NONE; // nothing to hang from: free
        }
        if (particles[idx].attached_to != NONE) {
            if (slab_hang(idx)) {
                particles[idx].temperature = temp;
                update_density(idx);
            }
            return;
        }
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

// A hanging slab (design 6; hangSlab in src/columns/hinge.js): in map view at
// the trench (its parent's front face); its top at the hang point of what it hangs from (that column's flap tip, or the
// bottom of the slab above); it passes its load up the chain (its weight above
// the cutoff, less the viscous support of its descent, plus what hangs from
// it), at most the tensile strength of the weaker of the two columns; any
// excess makes it neck, and necked through it breaks off and sinks freely
// from the next step. Removed
// below the cutoff or once neutrally buoyant. A slab lost while hanging from
// a surface column breaks the heuristic of design 9.2 and is counted.
// Returns false if the slab was removed.
fn slab_hang(idx: u32) -> bool {
    let p = particles[idx].attached_to;
    // In map view it hangs at the trench: at its parent's front face toward
    // the parent's trench partner (a surface column) or, the step it starts
    // hanging, before the parent's hinge has found one, toward the overriding
    // column it went under itself; or under its parent slab. (It was put on
    // top of its parent: the arc then grew on both sides of the trench and
    // reset the plate behind it; design 9.2.)
    var x = particles[p].position.xyz;
    var toward = select(NONE, particles[p].hinge_partner, particles[p].state == 1u);
    if (particles[p].state == 1u && toward == NONE && particles[idx].hinge_partner != NONE && particles[particles[idx].hinge_partner].state == 1u) { toward = particles[idx].hinge_partner; }
    if (toward != NONE) {
        var nh = particles[toward].position.xyz - x;
        nh = nh - dot(nh, x) * x;
        if (length(nh) > 0.0) { x = normalize(x + normalize(nh) * column_radius(p, x, nh)); }
    }
    particles[idx].position = vec4f(x, 1.0);
    particles[idx].predicted_pos = vec4f(x, 1.0);
    let z0 = particles[idx].elevation;
    let top = hang_point_elevation(p);
    let ext = particles[idx].slab_extent_km;
    particles[idx].elevation = top;
    let lost = particles[p].state == 1u;
    // Arc supply as the slab's bottom crosses the arc window (the top stays at the trench).
    let b0 = -(z0 - ext);
    let b1 = -(top - ext);
    let crossed = max(min(b1, sim.arc_depth_max_km) - max(b0, sim.arc_depth_min_km), 0.0);
    particles[idx].arc_supply = sim.arc_efficiency * particles[idx].crust_volume * crossed / (sim.arc_depth_max_km - sim.arc_depth_min_km);
    let excess = -sim.mantle_density * profiles[idx].freeboard_km * 1000.0;
    if (excess <= 0.0 || -top > sim.slab_cutoff_km) {
        if (lost) { atomicAdd(&solver.slabs_lost, 1u); }
        particles[idx].is_active = 0u;
        release(true, idx);
        return false;
    }
    let above = clamp((sim.slab_cutoff_km + top) / max(ext, 1e-3), 0.0, 1.0);
    let weight = excess * 9.81 * particles[idx].area_ref * EARTH_RADIUS_KM * EARTH_RADIUS_KM * 1e6 * above;
    // Its material moves down dip as fast as its plate feeds the trench (the
    // hang point itself moves with the bend, quasi-statically: that is not
    // flow past the slab): the parent's speed toward its trench partner; a
    // slab hanging from a slab, its parent's (stored as its velocity.w).
    var feed = 0.0;
    if (particles[p].state == 1u) {
        let hp = particles[p].hinge_partner;
        if (hp != NONE) {
            let xp = particles[p].position.xyz;
            var nh = particles[hp].position.xyz - xp;
            nh = nh - dot(nh, xp) * xp;
            if (length(nh) > 0.0) { feed = max(dot(particles[p].velocity.xyz, normalize(nh)), 0.0) * EARTH_RADIUS_KM; }
        }
    } else {
        feed = particles[p].velocity.w;
    }
    particles[idx].velocity.w = feed; // km/Myr down dip, for slabs hanging from this one
    let free_rate = slab_sink_rate_km_myr(idx);
    // Its dip (hinge_angle): pinned at the top, it turns toward vertical under
    // its own weight like a pendulum in a viscous fluid, d dip / dt = (w / L)
    // cos dip, L its length (slabDipStep in hinge.js); its vertical extent follows.
    let L = 2.0 * particles[idx].apothem * EARTH_RADIUS_KM;
    var dip = particles[idx].hinge_angle;
    dip = min(dip + sim.dt_crust_myr * free_rate / max(L, 1e-3) * cos(dip), 1.5707963);
    particles[idx].hinge_angle = dip;
    particles[idx].slab_extent_km = L * sin(dip);
    let descent = feed * sin(dip);
    let support = select(0.0, clamp(descent / free_rate, 0.0, 1.0), free_rate > 0.0);
    // Support, descent and bend depend on each other through the hinge, one
    // step apart: the load is under-relaxed (half the change per step), so
    // the lagged fixed point converges instead of flipping (hangSlab, relax).
    var load = 0.5 * (particles[idx].hang_load + weight * (1.0 - support) + particles[idx].child_load);
    let width = 2.0 * min(particles[p].apothem, particles[idx].apothem) * EARTH_RADIUS_KM * 1000.0;
    let cap = sim.slab_attach_factor * min(column_strength(p).tension, column_strength(idx).tension) * width;
    // The tension at its top, down dip (slabTension in hinge.js): the
    // down-dip weight of it and what hangs from it, less the mantle's drag
    // on them as they move down dip at the feed rate. A slab fed at its free
    // speed is held by the mantle, not by its attachment. Beyond the
    // attachment's strength the slab necks: its top falls behind the hang
    // point at the excess rate, w_free (1 - cap / T), while the hang point
    // carries the strength. Necked through its thickness, it breaks off.
    // (The vertical load was compared before: it stays large, W cos^2 dip,
    // even for a slab the mantle holds; design 9.2.)
    let w_si = free_rate * 1000.0 / 3.15576e13; // m/s
    let feed_si = feed * 1000.0 / 3.15576e13;
    let tension = weight * sin(dip) + particles[idx].child_pull - (select(0.0, weight / w_si, w_si > 0.0) + particles[idx].child_drag) * feed_si;
    var neck = particles[idx].hinge_plastic;
    if (tension > cap) {
        neck += free_rate * (1.0 - cap / tension) * sim.dt_crust_myr;
        load *= cap / tension;
    }
    particles[idx].hinge_plastic = neck;
    particles[idx].elevation = top - neck;
    if (neck >= profiles[idx].column_km) {
        particles[idx].attached_to = NONE;
        if (lost) { atomicAdd(&solver.slabs_lost, 1u); }
        return true;
    }
    particles[idx].hang_load = load;
    atomic_add_f32(&particle_deltas[p].child_load, load);
    // Down-dip weight and drag coefficient up the chain (hingePull in
    // hinge.js): the trench column pulls with the slabs' down-dip weight and
    // drags them through the mantle as it moves (inverse_drag, solver.wgsl).
    atomic_add_f32(&particle_deltas[p].child_pull, weight * sin(dip) + particles[idx].child_pull);
    atomic_add_f32(&particle_deltas[p].child_drag, select(0.0, weight / w_si, w_si > 0.0) + particles[idx].child_drag);
    particles[p].slab_child = idx;
    return true;
}

// Free sinking of a slab (slabSinkRate in src/edges/transitions.js): buoyancy
// per area B g, B = -rho_m x freeboard (kg/m^2), against upper-mantle
// viscosity, w = B g h / (4 eta) with h the slab's thickness (its column
// extent): a sheet sinks edge-first, whatever its width in map view. km/Myr;
// 0 if not negatively buoyant.
fn slab_sink_rate_km_myr(idx: u32) -> f32 {
    let excess = -sim.mantle_density * profiles[idx].freeboard_km * 1000.0;
    let w = excess * 9.81 * profiles[idx].column_km * 1000.0 / (4.0 * sim.isostatic_viscosity_pa_s);
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
