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
  ['solver_iterations', 'u32', 32],     // constraint + collision iterations per step (read by simulation.js)
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
  ['elastic_thickness_km', 'f32', 30.0],     // thickness that carries in-plane stress
  ['bond_rate_myr_inv', 'f32', null],   // derived: bond stiffness / reference drag of a base-size particle, 1/Myr
  ['yield_stress_mpa', 'f32', 150.0],
  ['base_apothem', 'f32', 0.0175],      // reference particle apothem
  ['split_area_factor', 'f32', 3.0],    // particle splits when area > factor * base hexagon area
  ['subduction_overlap', 'f32', 0.012], // overlap depth that triggers subduction
  ['contact_margin', 'f32', 0.1],       // contacts are created for gaps below this fraction of the apothem sum
  // Spreading proxy (until design phase 8): crust over diverging mantle grows
  // in area at the divergence rate. With ~1%/Myr divergence over upwellings,
  // global crust production is ~0.5%/Myr of the surface (Earth: ~0.6%/Myr).
  ['spreading_divergence_threshold', 'f32', 0.002], // 1/Myr
  ['spreading_efficiency', 'f32', 1.0],             // area growth rate / mantle divergence
  ['max_spreading_rate', 'f32', 0.05],              // cap on fractional area growth, 1/Myr
  ['max_particles', 'u32', null],
  ['max_constraints', 'u32', null],     // bonds + contacts
  ['max_bonds', 'u32', null],           // bond slots come first; contact slots follow
  ['hash_grid_dim', 'u32', null],

  // --- rheology: Arrhenius temperature dependence ---
  ['activation_energy', 'f32', 250.0],  // kJ/mol
  ['gas_constant', 'f32', 0.008314],    // kJ/(mol K)
  ['reference_temp', 'f32', 293.0],     // K, temperature at which yield = yield_stress_mpa

  // --- thermal state and isostasy ---
  ['mantle_density', 'f32', 3300.0],
  ['thermal_expansion_coeff', 'f32', 3e-5], // 1/K
  ['ambient_ocean_temp', 'f32', 278.0],
  ['mantle_temp', 'f32', 1600.0],
  ['surface_cooling_rate', 'f32', 0.05],  // 1/Myr, relaxation of surface crust to ocean temperature
  ['slab_heating_rate', 'f32', 2.5],      // 1/Myr, relaxation of slab to mantle temperature
  ['slab_assimilation_km_per_k', 'f32', 0.1], // slab thickness lost per kelvin of heating
  // Sea level in the same datum as particle elevation (the top of an
  // uncompensated mantle surface). 3.5 km puts cold 7 km oceanic crust about
  // 4.5 km below sea level and 35 km continental crust about 1.7 km above it.
  ['sea_level', 'f32', 3.5],
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
