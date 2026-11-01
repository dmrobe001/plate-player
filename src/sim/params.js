// Simulation parameters, uploaded to the GPU as one uniform buffer (`sim` in WGSL).
//
// The order of this list *is* the memory layout; the WGSL struct is generated
// from it (see simParamsWGSL), so add, remove, or reorder entries freely.
// Values of `null` are filled in at runtime (capacities, derived step sizes).
//
// Units: lengths on the unit sphere unless marked _km; time in Myr (million
// years); temperatures in K; densities in kg/m^3; energies in kJ/mol.

export const SIM_PARAMS = [
  // --- mantle: flow comes from the CPU mantle model (src/mantle/mantle.js) ---
  ['num_nodes', 'u32', null],
  ['mantle_drive', 'f32', 1.0],         // multiplies the mantle velocity seen by the crust (0 = no flow; tests)

  // --- basal drag: Couette shear across the asthenosphere (design.md, mantle and drag) ---
  // Traction per area = viscosity * (v_mantle - v_plate) / h, where the channel
  // h runs from the plate's base (LAB) down to the base of the low-viscosity
  // layer. Deep keels shear a thin channel and are dragged hard; young, thin
  // oceanic lithosphere floats on a thick channel and is dragged weakly.
  ['asthenosphere_viscosity_pa_s', 'f32', 1e19],
  ['asthenosphere_base_km', 'f32', 300.0],      // depth where viscosity rises again
  ['min_channel_km', 'f32', 50.0],              // floor on h (keels never touch the base)
  ['drag_reference_channel_km', 'f32', 200.0],  // h of old ocean; normalises drag in the solver
  ['continental_lithosphere_km', 'f32', 150.0], // LAB depth under continents (cratons reach ~250)
  ['oceanic_lithosphere_max_km', 'f32', 100.0], // oceanic LAB: 2.32 sqrt(kappa age), capped here
  ['thermal_diffusivity_m2_s', 'f32', 1e-6],

  // --- crust: overdamped XPBD (see design.md 5.2) ---
  // Each step minimises  sum_i (gamma_i / 2 dt) |x_i - x~_i|^2 + U(x):
  // basal drag gamma plays the role of mass. Bond stiffness and drag come from
  // the physical values below; only their ratio enters (bond_rate_myr_inv).
  ['dt_crust_myr', 'f32', 0.016],
  ['time_myr', 'f32', 0],               // current time, written by simulation.js each step
  ['solver_iterations', 'u32', 32],     // Jacobi sweeps over all edges per step (read by simulation.js)
  // Chebyshev semi-iterative acceleration of the Jacobi sweeps (Wang 2015).
  // rho is the spectral radius of one plain sweep: negative = derive it
  // (simulation.js, jacobiSpectralRadius), 0 = off (plain Jacobi), else as given.
  // The first `delay` iterations are plain Jacobi. Read by simulation.js.
  ['chebyshev_rho', 'f32', -1],
  ['chebyshev_delay', 'u32', 3],
  // Lithosphere Young's modulus. With 32 Chebyshev-accelerated iterations the
  // solve converges well enough that residual stresses stay at a few MPa
  // (design.md, Chebyshev findings); plain Jacobi needed a 2 GPa stop-gap.
  ['crust_youngs_modulus_gpa', 'f32', 70.0],
  // Edge stiffness is E times the elastic part of the shared depth interval
  // (design 4.5); this thickness normalises it so the per-edge factor is
  // <= 1 (the thickest column), which keeps the Chebyshev bound valid.
  ['stiffness_reference_km', 'f32', 150.0],
  ['bond_rate_myr_inv', 'f32', null],   // derived: bond stiffness / reference drag of a base-size particle, 1/Myr
  ['base_apothem', 'f32', 0.0175],      // reference particle apothem
  ['split_area_factor', 'f32', 3.0],    // particle splits when area > factor * base hexagon area
  ['subduction_overlap', 'f32', 0.012], // overlap depth that triggers subduction
  // An edge exists while its pair's map-view gap is below this fraction of the
  // apothem sum (and their depth intervals overlap); design 3.2.
  ['edge_range_margin', 'f32', 0.1],
  // Spreading proxy (until design phase 8): crust over diverging mantle grows
  // in area at the divergence rate. With ~1%/Myr divergence over upwellings,
  // global crust production is ~0.5%/Myr of the surface (Earth: ~0.6%/Myr).
  ['spreading_divergence_threshold', 'f32', 0.002], // 1/Myr
  ['spreading_efficiency', 'f32', 1.0],             // area growth rate / mantle divergence
  ['max_spreading_rate', 'f32', 0.05],              // cap on fractional area growth, 1/Myr
  ['max_particles', 'u32', null],
  ['max_edges', 'u32', null],
  ['hash_grid_dim', 'u32', null],

  // --- constants ---
  ['gas_constant', 'f32', 0.008314],    // kJ/(mol K)

  // --- column profile (design 4.0; src/columns/profile.js) ---
  // Geotherms: oceanic half-space cooling; continental steady conduction with
  // heat production in the upper crust. Strength: Byerlee friction with
  // hydrostatic pore pressure plus cohesion (brittle), dislocation creep at a
  // reference strain rate (ductile). Flow laws are laboratory values (A in
  // MPa^-n s^-1); wet quartzite after Gleason & Tullis (1995), diabase after
  // Mackwell et al. (1998), dry olivine after Hirth & Kohlstedt (2003).
  // Provisional: calibrated at integration (design 9).
  ['thermal_conductivity_w_m_k', 'f32', 2.5],
  ['radiogenic_heat_uw_m3', 'f32', 1.5],     // continental upper crust
  ['radiogenic_layer_km', 'f32', 15.0],
  ['friction_coefficient', 'f32', 0.6],      // Byerlee
  ['pore_fluid_factor', 'f32', 0.4],         // pore pressure / lithostatic (hydrostatic ~0.4)
  ['cohesion_mpa', 'f32', 20.0],             // of intact rock (integrity 1)
  ['reference_strain_rate_s', 'f32', 1e-15],
  // Edge law (design 4.5): how much lost integrity weakens a ductile slice in
  // tension, cap x (1 - w (1 - phi)). 0 = the design table (ductile rock
  // resists separation whatever phi); 1 = a broken edge's ductile part is as
  // weak in tension as its brittle part (shear-zone damage, as in grain-damage theory).
  ['ductile_damage_weakening', 'f32', 0.0],
  // Integrity law (design 4.7, phase D): damage by plastic work, thermally
  // activated healing. Provisional; calibrated at integration.
  ['damage_work_mpa_km', 'f32', 50.0],     // plastic work (per unit edge length and depth) that removes all integrity
  ['heal_time_myr', 'f32', 0.1],           // healing time at mantle temperature
  ['heal_activation_kj', 'f32', 250.0],    // Arrhenius: ~50 Myr at 1200 K, effectively never below ~900 K
  // GPE force (design 4.6): push down the gradient of gravitational potential
  // energy (ridge push, plateau collapse). 1 = physical; 0 = off. Off by
  // default until integration calibrates basal drag against it: at 1, a
  // column at a continental margin is pushed at metres per year against the
  // present drag (design, phase D findings).
  ['gpe_force_factor', 'f32', 0.0],
  // Volumetric flow (design 4.7): multiplies the area_ref change from edges'
  // inelastic strain. 1 = physical; 0 = off (storage tests).
  ['volumetric_flow_factor', 'f32', 1.0],
  ['quartzite_log10_a', 'f32', -3.959],      // log10(1.1e-4)
  ['quartzite_n', 'f32', 4.0],
  ['quartzite_q_kj', 'f32', 223.0],
  ['diabase_log10_a', 'f32', 0.903],         // log10(8)
  ['diabase_n', 'f32', 4.7],
  ['diabase_q_kj', 'f32', 485.0],
  ['olivine_log10_a', 'f32', 5.041],         // log10(1.1e5)
  ['olivine_n', 'f32', 3.5],
  ['olivine_q_kj', 'f32', 530.0],

  // --- thermal state and isostasy ---
  ['mantle_density', 'f32', 3300.0],
  ['thermal_expansion_coeff', 'f32', 3e-5], // 1/K
  ['ambient_ocean_temp', 'f32', 278.0],
  ['mantle_temp', 'f32', 1600.0],
  ['surface_cooling_rate', 'f32', 0.05],  // 1/Myr, relaxation of surface crust to ocean temperature
  ['slab_heating_rate', 'f32', 2.5],      // 1/Myr, relaxation of slab to mantle temperature
  ['slab_assimilation_km_per_k', 'f32', 0.1], // slab thickness lost per kelvin of heating
  // Sea level in the same datum as particle elevation (the top of a dry
  // mantle at mantle temperature). With the column profile's Airy isostasy
  // (design 4.0), 2.6 km puts a ridge (7 km crust, age 0) 2.9 km below sea
  // level, old ocean (capped age) 5.6 km below, and 35 km continental crust
  // 0.8 km above; 30 km continental crust is a shelf just under water.
  ['sea_level', 'f32', 2.6],
  ['water_density', 'f32', 1030.0],
  // Free radius (design 4.4): each column's elevation relaxes toward its Airy
  // height at the rate of viscous isostatic rebound of a load one column pair
  // wide (Haskell): tau = 4 pi eta / (drho g lambda), lambda = 4 apothems.
  // 1e21 Pa s is the upper-mantle viscosity from post-glacial rebound; tau is
  // then ~30 kyr at the base particle size, so columns track Airy closely.
  ['isostatic_viscosity_pa_s', 'f32', 1e21],
  // Stand-ins until slabs get real buoyancy and bending (design phase 7): a
  // new slab's top drops below any lithosphere, so its column no longer
  // overlaps surface columns in depth, then sinks at a fixed rate.
  ['slab_initial_depth_km', 'f32', 160.0],
  ['slab_sink_rate_km_myr', 'f32', 20.0],
];

export function simParamsWGSL() {
  const words = paddedWordCount();
  const body = SIM_PARAMS.map(([n, t]) => `    ${n}: ${t},`);
  for (let i = SIM_PARAMS.length; i < words; i++) body.push(`    _pad${i}: u32,`);
  return `struct SimParams {\n${body.join('\n')}\n}\n`;
}

/** Uniform buffers must be a multiple of 16 bytes. */
function paddedWordCount() {
  return Math.ceil(SIM_PARAMS.length / 4) * 4;
}

export const SIM_PARAMS_BYTES = paddedWordCount() * 4;

/** Pack parameter values (defaults overridden by `runtime`) into an ArrayBuffer. */
export function packSimParams(runtime = {}) {
  const buf = new ArrayBuffer(SIM_PARAMS_BYTES);
  const f32 = new Float32Array(buf);
  const u32 = new Uint32Array(buf);
  SIM_PARAMS.forEach(([name, type, value], i) => {
    const v = name in runtime ? runtime[name] : value;
    if (v === null || v === undefined) throw new Error(`sim param ${name} has no value`);
    if (type === 'u32') u32[i] = v;
    else f32[i] = v;
  });
  return buf;
}

export function simParamDefault(name) {
  const entry = SIM_PARAMS.find(([n]) => n === name);
  if (!entry) throw new Error(`unknown sim param ${name}`);
  return entry[2];
}
