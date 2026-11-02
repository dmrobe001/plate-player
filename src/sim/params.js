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
  // rho is the spectral radius of one plain sweep: negative = derive it, capped
  // by the iteration budget and at 0.99 (simulation.js, chebyshevRho; design
  // 5.2), 0 = off (plain Jacobi), else as given.
  // The first `delay` iterations are plain Jacobi.
  ['chebyshev_rho', 'f32', -1],
  ['chebyshev_delay', 'u32', 3],
  // Warm start: each edge's multiplier starts at the force of its stored
  // elastic stress (and positions move to match), so a step's solve corrects
  // the increment instead of rebuilding the whole stress from zero (design 5.2).
  ['solver_warm_start', 'u32', 1],
  // No net rotation (design 5.3): each step, the area-weighted mean rigid
  // rotation of the surface columns is removed from the whole planet (a frame
  // choice, as in geodesy's no-net-rotation frame). 0 = keep it (mechanism
  // tests on a patch, where "the whole planet" is not meant).
  ['remove_net_rotation', 'u32', 1],
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
  // A surface column splits when its actual (tessellated) area exceeds this
  // factor x the base hexagon area (design 4.8).
  ['split_area_factor', 'f32', 2.0],
  // ... and when its reference shape is elongated past this aspect ratio while
  // its reference area is at least split_area_guard x the base hexagon (design
  // 4.8, phase G): it splits across its long axis, into two halves that are
  // round at aspect ratio 2. Constant-area elongation (shear) never splits.
  ['split_aspect_ratio', 'f32', 2.0],
  ['split_area_guard', 'f32', 1.3],

  // --- tangential channel, rotation and shape (design 4.2, 4.5; phase G) ---
  // Edges resist relative sliding of their faces, with the columns' own
  // rotations (spins) as unknowns, so rigid rotation is free. On a triangular
  // lattice, normal and tangential stiffnesses k_n, k_t give Poisson's ratio
  // (k_n - k_t) / (3 k_n + k_t) and shear modulus E / 2 (1 + nu): 0.2 gives
  // nu = 0.25; 0 = no tangential channel (nu = 1/3). The tangential caps are
  // friction, mu (1 - lambda_pore)(P - sigma_n) + phi c, or the flow stress.
  ['shear_stiffness_ratio', 'f32', 0.2],
  // Flat faces resist relative rotation of the two columns (k_n face^2 / 12),
  // up to the kern rule: the face opens at one end once the moment exceeds
  // (compression + tension cap) x face / 6. Scales that stiffness; 0 = off.
  ['rolling_stiffness_factor', 'f32', 1.0],
  // Distributed (ductile) inelastic strain changes a column's reference
  // shape: the deviatoric part elongates it (the volumetric part changes its
  // area, volumetric_flow_factor). 1 = physical; 0 = shapes stay round.
  ['shape_flow_factor', 'f32', 1.0],
  // Largest aspect ratio of a reference shape. Beyond it, deviatoric flow is
  // taken up by neighbour exchange alone.
  ['shape_aspect_cap', 'f32', 3.0],
  // An edge exists while its pair's map-view gap is below this fraction of the
  // apothem sum (and their depth intervals overlap); design 3.2.
  ['edge_range_margin', 'f32', 0.1],
  ['max_particles', 'u32', null],
  ['max_edges', 'u32', null],
  ['hash_grid_dim', 'u32', null],

  // --- area, volume and melt (design 4.8; src/columns/area.js, volume.js) ---
  // Tessellation cells are clipped to a disc of this radius (unit sphere).
  // Exact tiling needs every neighbour that can cut it to be in the hash
  // cells scanned: keep it below half the hash cell size, 3 / hash_grid_dim.
  ['cell_clip_radius', 'f32', 0.04],
  // Decompression melting (McKenzie & Bickle 1988): new area makes
  // melt_fraction x melt_depth_km of crust (7 km) where the lithosphere is
  // thin, none where it is thicker than melt_depth_km.
  ['melt_depth_km', 'f32', 70.0],         // where rising mantle crosses its solidus
  ['melt_fraction', 'f32', 0.1],          // mean melt fraction of the melting column
  ['melt_density', 'f32', 3000.0],        // composition of melt (basalt), mixed into the column by volume

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
  // Integrity law (design 4.7): slip weakening, thermally activated healing.
  // Integrity falls by the plastic displacement across the edge (weighted by
  // the share of its strength that yields) over this distance: the cumulative
  // slip at which faults become structurally mature, ~10 km (Wesnousky 1988).
  ['damage_slip_km', 'f32', 10.0],
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
  ['slab_heating_rate', 'f32', 0.05],     // 1/Myr: a slab's thermal age decays at this rate (it warms; design 6)
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
  // --- suturing, subduction, slabs, arcs (design 6; src/edges/transitions.js, src/columns/hinge.js) ---
  // One timer per edge: it accumulates while the edge's depth-averaged normal
  // stress is more compressive than transition_stress_mpa, and resets
  // otherwise. Past suture_window_myr, with both columns buoyant and no
  // plastic work that step, the edge welds (integrity 1).
  ['transition_stress_mpa', 'f32', 50.0],
  ['suture_window_myr', 'f32', 5.0],
  // Subduction is not a timer: a column goes down when the hinge of its trench
  // has passed its back face (design 6, the trench hinge; src/columns/hinge.js).
  ['hinge_max_angle', 'f32', 1.0],        // rad: the bend's limit (the runaway branch goes here)
  // A slab hangs from the column behind it until its load exceeds the tensile
  // strength of the weaker of the two (x this factor); then it breaks off and
  // sinks freely. 0 = slabs never hang (each sinks freely from the start).
  ['slab_attach_factor', 'f32', 1.0],
  // The subduction interface is lubricated: compression caps of edges with
  // exactly one slab end are scaled by this.
  ['slab_interface_weakening', 'f32', 0.1],
  // Slabs sink under their own negative buoyancy B = -rho_m x freeboard
  // (kg/m^2) at the Stokes-like rate w = B g width / (4 eta), eta =
  // isostatic_viscosity_pa_s (upper mantle). Removed below this depth or once
  // warmed to neutral buoyancy.
  ['slab_cutoff_km', 'f32', 660.0],
  // Eclogite (design 6): a slab's oceanic crust turns from basalt to eclogite
  // at ~1.5-2 GPa (50-70 km; delayed in cold slabs), denser than the mantle
  // (Hacker et al. 2003). Linear over +-10 km about this depth of its middle.
  // Continental crust stays buoyant.
  ['eclogite_depth_km', 'f32', 60.0],
  ['eclogite_density', 'f32', 3500.0],
  // Arcs: surface columns within arc_reach (unit sphere) of a slab whose top
  // is between these depths gain crust volume: arc_efficiency x the slab's
  // volume over its passage through the window, shared equally; their thermal
  // age resets to 0 (hot, weak arc lithosphere).
  ['arc_depth_min_km', 'f32', 80.0],
  ['arc_depth_max_km', 'f32', 160.0],
  ['arc_reach', 'f32', 0.05],
  ['arc_efficiency', 'f32', 0.2],
  ['arc_density', 'f32', 2750.0],

  // --- bending (design 4.3, 4.4; phase F; src/columns/flexure.js) ---
  // Plate energy (D/2) (laplacian of elevation)^2 per area on the tessellation
  // (cotangent-like weights: shared cell boundary / distance), with a Winkler
  // foundation (isostasy). D = E h_e^3 / 12 (1 - nu^2), h_e = the column's
  // Maxwell-elastic thickness this step. flexure_factor scales D (0 = Airy).
  ['flexure_factor', 'f32', 1.0],
  ['poisson_ratio', 'f32', 0.25],
  ['flexure_iterations', 'u32', 8],       // damped Jacobi sweeps per step (read by simulation.js)
  ['flexure_damping', 'f32', 2.0],        // bending part of the Jacobi diagonal x this (> 1.4 for convergence)
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
