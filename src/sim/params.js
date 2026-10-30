// Simulation parameters, uploaded to the GPU as one uniform buffer (`sim` in WGSL).
//
// The order of this list *is* the memory layout; the WGSL struct is generated
// from it (see simParamsWGSL), so add, remove, or reorder entries freely.
// Values of `null` are filled in at runtime from buffer capacities.
//
// Units: lengths on the unit sphere unless marked _km; time in Myr (million
// years); temperatures in K; densities in kg/m^3; energies in kJ/mol.

export const SIM_PARAMS = [
  // --- mantle: Swift-Hohenberg pattern on the icosphere ---
  ['mantle_rayleigh', 'f32', 0.2],      // control parameter r: growth rate of the convection pattern
  ['mantle_wavenumber', 'f32', 4.0],    // preferred wavenumber k0 of the pattern
  ['dt_mantle', 'f32', 0.000005],       // explicit-Euler step for psi (pattern time, not Myr)
  ['laplacian_weight', 'f32', 45.0],    // 1/h^2-like scale of the graph Laplacian
  ['blanket_forcing', 'f32', 0.4],      // how strongly crust heat (thermal blanket) forces psi
  ['num_nodes', 'u32', null],

  // --- crust: position-based dynamics ---
  ['dt_crust_myr', 'f32', 0.016],
  ['drag_coeff_myr_inv', 'f32', 50.0],  // relaxation rate of crust velocity toward mantle velocity
  ['crust_youngs_modulus_gpa', 'f32', 2.0], // used only to convert strain to stress (diagnostics, yielding)
  ['yield_stress_mpa', 'f32', 150.0],
  ['cm_yr_to_sim_vel', 'f32', 10.0 / 6371.0], // 1 cm/yr = 10 km/Myr; / Earth radius -> rad/Myr
  ['sh_grad_to_cm_yr', 'f32', 25.0],    // grad(psi) -> plate speed in cm/yr
  ['base_apothem', 'f32', 0.0175],      // reference particle apothem
  ['split_area_factor', 'f32', 3.0],    // particle splits when area > factor * base hexagon area
  ['subduction_overlap', 'f32', 0.012], // overlap depth that triggers subduction
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
