// Simulation parameters, uploaded to the GPU as one uniform buffer (`sim` in WGSL).
//
// The order of this list *is* the memory layout; the WGSL struct is generated
// from it (see simParamsWGSL), so add, remove, or reorder entries freely.
// Values of `null` are filled in at runtime (capacities, derived step sizes).
//
// Units: lengths on the unit sphere unless marked _km; time in Myr (million
// years); temperatures in K; densities in kg/m^3; energies in kJ/mol.

export const SIM_PARAMS = [
  // --- mantle: Swift-Hohenberg pattern on the icosphere ---
  ['mantle_rayleigh', 'f32', 0.2],      // control parameter r: growth rate of the convection pattern
  ['mantle_wavenumber', 'f32', 4.0],    // preferred wavenumber k0 of the pattern
  ['mantle_time_per_myr', 'f32', 0.015625], // pattern-time units per Myr (= 50 steps of 5e-6 per 0.016 Myr, as in Phase 5)
  ['dt_mantle', 'f32', null],           // explicit-Euler substep for psi; derived from a stability bound (simulation.js)
  ['laplacian_weight', 'f32', 45.0],    // 1/h^2-like scale of the graph Laplacian
  ['blanket_forcing', 'f32', 0.4],      // how strongly crust heat (thermal blanket) forces psi
  ['num_nodes', 'u32', null],

  // --- crust: overdamped XPBD (see design.md 5.2) ---
  // Each step minimises  sum_i (gamma_i / 2 dt) |x_i - x~_i|^2 + U(x):
  // basal drag gamma plays the role of mass. Bond stiffness and drag come from
  // the physical values below; only their ratio enters (bond_rate_myr_inv).
  ['dt_crust_myr', 'f32', 0.016],
  ['solver_iterations', 'u32', 4],      // constraint + collision iterations per step (read by simulation.js)
  // Stop-gap: real lithosphere is ~70 GPa, but until plate-scale stress
  // converges (design phase 5), Jacobi residual strain times 70 GPa reads as
  // >150 MPa in ~13% of bonds and spreads spurious damage. 2 GPa (the Phase 5
  // prototype's value) keeps residual stresses at a few MPa. See design.md, phase 2.
  ['crust_youngs_modulus_gpa', 'f32', 2.0],
  ['elastic_thickness_km', 'f32', 30.0],     // thickness that carries in-plane stress
  ['asthenosphere_viscosity_pa_s', 'f32', 1e19],
  ['asthenosphere_thickness_km', 'f32', 100.0], // basal drag per area = viscosity / thickness
  ['bond_rate_myr_inv', 'f32', null],   // derived: bond stiffness / drag of a base-size particle, 1/Myr
  ['yield_stress_mpa', 'f32', 150.0],
  ['cm_yr_to_sim_vel', 'f32', 10.0 / 6371.0], // 1 cm/yr = 10 km/Myr; / Earth radius -> rad/Myr
  ['sh_grad_to_cm_yr', 'f32', 25.0],    // grad(psi) -> plate speed in cm/yr
  ['base_apothem', 'f32', 0.0175],      // reference particle apothem
  ['split_area_factor', 'f32', 3.0],    // particle splits when area > factor * base hexagon area
  ['subduction_overlap', 'f32', 0.012], // overlap depth that triggers subduction
  ['divergence_threshold', 'f32', 1.5], // -lap(psi) above which crust grows in area
  ['area_growth_coeff', 'f32', 0.005],  // area growth per Myr per unit divergence (unit-sphere area)
  ['max_area_growth_rate', 'f32', 0.00625], // cap on area growth, unit-sphere area per Myr
  ['max_particles', 'u32', null],
  ['max_constraints', 'u32', null],
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
  // 4 km below sea level and 35 km continental crust about 1.7 km above it.
  ['sea_level', 'f32', 3.5],
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
