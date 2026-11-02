// Edge state transitions: suturing (design 6). CPU reference for the end of
// edges_maxwell in src/shaders/compute/law.wgsl. (Subduction is the trench
// hinge's: src/columns/hinge.js.)
//
// One timer per edge. It accumulates while the edge's depth-averaged normal
// stress is more compressive than transition_stress_mpa and resets to 0
// otherwise. If both columns of a surface pair are buoyant (continents, young
// ocean), once the timer passes suture_window_myr in a step with no plastic
// work on the edge, the edge welds (integrity 1): collision sutures the plates.

/** Timer after this step: + dt in compression beyond the threshold, else 0. */
export function timerStep(timer, stressMpa, p) {
  return stressMpa < -p.transition_stress_mpa ? timer + p.dt_crust_myr : 0;
}

/**
 * a, b: {freeboardKm}; edge: {timer, work}. Returns {weld: bool}.
 */
export function transition(edge, a, b, p) {
  const buoyant = a.freeboardKm >= 0 && b.freeboardKm >= 0;
  return { weld: buoyant && edge.timer >= p.suture_window_myr && edge.work === 0 };
}

/**
 * Free sinking rate of a slab column, km/Myr (slab_sink_rate_km_myr in
 * thermal.wgsl): buoyancy per area B g with B = -rho_m x freeboard (kg/m^2),
 * against upper-mantle viscosity, w = B g h / (4 eta). The Stokes size of a
 * slab is its thickness h (km; the column's depth extent), not its width in
 * map view: a sheet sinks edge-first, and its speed must not depend on how
 * coarsely the plate is cut into columns. 0 if not negatively buoyant.
 */
export function slabSinkRate(freeboardKm, thicknessKm, p) {
  const excess = -p.mantle_density * freeboardKm * 1e3;
  return Math.max((excess * 9.81 * thicknessKm * 1e3) / (4 * p.isostatic_viscosity_pa_s), 0) * 3.15576e13 / 1e3;
}
