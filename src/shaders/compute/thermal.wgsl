// Crust thermal state, isostasy, bond healing, and the crust -> mantle
// "thermal blanket" feedback.

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
    update_isostasy(idx);
}

// Density and elevation follow from temperature and thickness.
fn update_isostasy(idx: u32) {
    // Colder rock is denser; composition_density is defined at mantle temperature.
    let rho = particles[idx].composition_density * (1.0 + sim.thermal_expansion_coeff * (sim.mantle_temp - particles[idx].temperature));
    particles[idx].density = rho;

    // Airy isostasy: freeboard of a floating column above the mantle datum,
    // with extra depression where the column is loaded by sea water.
    var elevation = particles[idx].thickness_km * (1.0 - rho / sim.mantle_density);
    if (elevation < sim.sea_level) {
        elevation -= (sim.sea_level - elevation) * (1000.0 / sim.mantle_density);
    }
    particles[idx].elevation = elevation;
}

// Recompute derived quantities without advancing time (after loading a state).
@compute @workgroup_size(64)
fn crust_refresh_derived(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    update_isostasy(idx);
}

@compute @workgroup_size(64)
fn crust_constraint_healing(@builtin(global_invocation_id) id: vec3<u32>) {
    let c_idx = id.x;
    if (c_idx >= sim.max_constraints || constraints[c_idx].is_active == 0u) { return; }
    let rate = 100000.0 * arrhenius(bond_temperature(constraints[c_idx]));
    constraints[c_idx].damage = clamp(constraints[c_idx].damage - rate * sim.dt_crust_myr, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn grid_clear_thermal(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.num_nodes) { return; }
    atomicStore(&grid[idx].thermal_atomic, 0i);
    grid[idx].thermal_blanket = 0.0;
}

// Thick crust insulates (warms) the mantle beneath it; slabs cool it.
@compute @workgroup_size(64)
fn crust_apply_thermal(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    let node = particles[idx].mantle_node;
    var heat = i32(particles[idx].thickness_km * 5.0);
    if (particles[idx].state == 2u) { heat = -i32(particles[idx].thickness_km * 20.0); }
    atomicAdd(&grid[node].thermal_atomic, heat);
}

@compute @workgroup_size(64)
fn grid_finalize_thermal(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.num_nodes) { return; }
    grid[idx].thermal_blanket = clamp(f32(atomicLoad(&grid[idx].thermal_atomic)) * 0.001, -1.5, 1.5);
}
