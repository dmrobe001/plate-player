// Edge state transitions: subduction initiation and suturing (design 6,
// phase F). CPU reference for the end of edges_maxwell in
// src/shaders/compute/law.wgsl.
//
// One timer per edge. It accumulates while the edge's depth-averaged normal
// stress is more compressive than transition_stress_mpa and resets to 0
// otherwise. Then, for a pair of surface columns:
//   - If either column is negatively buoyant (freeboard < 0: the column is
//     denser than the mantle it floats on; old ocean), subduction starts once
//     the timer passes subduction_window_myr. The window shrinks by
//     subduction_propagation_factor when the down column already touches a
//     slab, so a trench propagates and keeps its polarity.
//     Which column goes down: the one touching a slab, if only one can sink
//     and does; else the one with the lower freeboard (denser; older ocean);
//     ties go to the lower index.
//   - If both are buoyant (continents, young ocean), nothing subducts; once
//     the timer passes suture_window_myr in a step with no plastic work on the
//     edge, the edge welds (integrity 1): collision sutures the plates.

/** Timer after this step: + dt in compression beyond the threshold, else 0. */
export function timerStep(timer, stressMpa, p) {
  return stressMpa < -p.transition_stress_mpa ? timer + p.dt_crust_myr : 0;
}

/**
 * a, b: {index, freeboardKm, slabNeighbours}; edge: {timer, work}.
 * Returns {subduct: index of the column that becomes a slab, or null; weld: bool}.
 */
export function transition(edge, a, b, p) {
  const sinkA = a.freeboardKm < 0, sinkB = b.freeboardKm < 0;
  if (sinkA || sinkB) {
    let down;
    if (sinkA && a.slabNeighbours > 0 && !(sinkB && b.slabNeighbours > 0)) down = a;
    else if (sinkB && b.slabNeighbours > 0 && !(sinkA && a.slabNeighbours > 0)) down = b;
    else if (a.freeboardKm !== b.freeboardKm) down = a.freeboardKm < b.freeboardKm ? a : b;
    else down = a.index < b.index ? a : b;
    const window = p.subduction_window_myr * (down.slabNeighbours > 0 ? p.subduction_propagation_factor : 1);
    return { subduct: edge.timer >= window ? down.index : null, weld: false };
  }
  return { subduct: null, weld: edge.timer >= p.suture_window_myr && edge.work === 0 };
}

/**
 * Sinking rate of a slab column, km/Myr (slab_sink_rate_km_myr in
 * thermal.wgsl): buoyancy per area B g with B = -rho_m x freeboard (kg/m^2),
 * over a column of width 2 x apothem (unit sphere), against upper-mantle
 * viscosity: w = B g width / (4 eta). 0 if not negatively buoyant.
 */
export function slabSinkRate(freeboardKm, apothem, p) {
  const excess = -p.mantle_density * freeboardKm * 1e3;
  const width = 2 * apothem * 6.371e6;
  return Math.max((excess * 9.81 * width) / (4 * p.isostatic_viscosity_pa_s), 0) * 3.15576e13 / 1e3;
}
