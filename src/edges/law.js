// Edge law, CPU reference (design 4.5, phase C). Mirrors edges_law and
// edges_maxwell in src/shaders/compute/edges.wgsl.
//
// An edge couples two columns over the depth interval they share, cut into
// EDGE_SLICES equal slices. Slices act in parallel (columns are vertically
// rigid); each is a Maxwell element (spring E in series with a dashpot of the
// pair's viscosity at that depth), stepped by implicit Euler:
//     sigma' = (sigma + E d_eps) r,   r = 1 / (1 + E dt / eta)
// so cold slices act as springs (r ~ 1) and hot ones as dashpots (r ~ 0),
// and the elastic thickness emerges. Each slice's stress is capped by the
// yield-strength envelope (design 4.0). A slice is brittle where friction is
// the weaker (mu (1 - lambda_pore) P + phi c < ductile strength), else ductile:
//   compression cap: min(friction + phi c, ductile)
//   tension cap:     phi c if brittle; the ductile flow stress if ductile
//                    (sign-blind: hot rock resists separation), reduced by
//                    lost integrity if ductile_damage_weakening > 0
//
// What happens above a cap:
//   - ductile slices flow, and brittle compression slips: the stored stress
//     is the capped stress (the slice's stress-free length moves; the pair's
//     rest length does not; the isotropic part goes to area_ref in phase D);
//   - brittle tension cracks: the slice carries its cap (phi c, so nothing at
//     phi = 0) and stays referenced to the rest length: its stored stress is
//     the elastic trial stress, less only the creep driven by the stress it
//     actually carries. So a crack closes, and a contact re-engages, exactly
//     at touching.
//
// Within a step the law is linear (Chebyshev needs that): every slice
// contributes its decayed stored stress and its stiffness E r, and the total
// is clamped to the depth-integrated caps (lambda bounds). So a cracked pair
// is pulled together at most at its tension cap until its slices close, at
// touching, where the law is stiff again: no overshoot.
//
// Stresses are normal stresses on the interface (MPa, + tension), integrated
// over depth as S (MPa km). Normal only: a tangential (shear) channel needs
// particle orientations to avoid resisting rigid rotation (design 10).

import { geotherm, ductile, flowLaw, columnProfile, GRAVITY, SECONDS_PER_MYR } from '../columns/profile.js';

export const EDGE_SLICES = 8;
const LN_STRENGTH_CAP = Math.log(1e6);

/** The column's depth extent below its top, km: crust and lithosphere (edges.wgsl, column_bottom). */
export const columnDepthKm = (col, p) => Math.max(geotherm(col, p).L, col.crustKm);

/**
 * Lithostatic pressure (MPa) at depth d in a column, from its profile slices:
 * first-order Taylor expansion about the nearest slice midpoint (first of
 * equals), P_k + rho_k g (d - z_k). Robust to zero-thickness slices.
 */
export function slicePressure(prof, d) {
  let k = 0, best = Infinity;
  prof.slices.forEach((s, m) => { const e = Math.abs(d - s.z); if (e < best) { best = e; k = m; } });
  const s = prof.slices[k];
  return s.pressure + (s.density * GRAVITY * (d - s.z) * 1e3) / 1e6;
}

/**
 * The edge's slices for this step. `a`, `b`: column inputs
 * ({ crustKm, compositionDensity, ageMyr, elevationKm }), phi: integrity.
 * Per slice: dz (km), r (Maxwell factor), comp and tens caps (MPa), and
 * whether tension is brittle-limited (cracks) rather than ductile (flows).
 */
export function edgeSlices(a, b, phi, p) {
  const top = Math.min(a.elevationKm, b.elevationKm);
  const bottom = Math.max(a.elevationKm - columnDepthKm(a, p), b.elevationKm - columnDepthKm(b, p));
  const overlap = Math.max(top - bottom, 0);
  const dz = overlap / EDGE_SLICES;
  const ends = [a, b].map((col) => ({ col, g: geotherm(col, p), prof: columnProfile(col, p) }));
  const lnEdt = Math.log(p.crust_youngs_modulus_gpa * 1e9 * p.dt_crust_myr * SECONDS_PER_MYR);
  const slices = [];
  for (let k = 0; k < EDGE_SLICES; k++) {
    const z = top - (k + 0.5) * dz; // height above the datum
    let edtOverEta = 0, invStrength = 0, pressure = 0, T = 0;
    for (const { col, g, prof } of ends) {
      const d = col.elevationKm - z; // depth in this column
      const lnStrength = ductile(flowLaw(col, d, p), g.T(d), p).lnStrength;
      const lnS = Math.min(lnStrength, LN_STRENGTH_CAP); // the cap keeps strengths finite in f32; viscosity is uncapped
      const lnEta = lnStrength + Math.log(1e6) - Math.log(2 * p.reference_strain_rate_s);
      // Harmonic means of viscosity and of ductile strength (series: the weaker dominates).
      edtOverEta += 0.5 * Math.exp(lnEdt - lnEta);
      invStrength += 0.5 * Math.exp(-lnS);
      pressure += 0.5 * slicePressure(prof, d);
      T += 0.5 * g.T(d);
    }
    const ductileCap = 1 / invStrength;
    const brittleComp = phi * p.cohesion_mpa + p.friction_coefficient * (1 - p.pore_fluid_factor) * pressure;
    const brittle = brittleComp < ductileCap;
    slices.push({
      z, dz, T,
      r: 1 / (1 + edtOverEta),
      comp: Math.min(brittleComp, ductileCap),
      tens: brittle ? phi * p.cohesion_mpa : ductileCap * (1 - p.ductile_damage_weakening * (1 - phi)),
      cracks: brittle,
    });
  }
  return { top, bottom, overlap, slices };
}

const E_MPA = (p) => p.crust_youngs_modulus_gpa * 1000;

/**
 * Linearisation for this step's solve, from the stored slice stresses:
 * S(strain) = S0 + K (strain - strain0), clamped to [-Scomp, Stens].
 * Returns S0, K (MPa km per unit strain), the stiffness factor
 * K / (E h_ref), the strain at zero force, and the total caps (MPa km).
 */
export function linearize(stress, edge, strain0, p) {
  const E = E_MPA(p);
  let S0 = 0, K = 0, Scomp = 0, Stens = 0;
  edge.slices.forEach((s, k) => {
    S0 += stress[k] * s.r * s.dz;
    K += E * s.r * s.dz;
    Scomp += s.comp * s.dz;
    Stens += s.tens * s.dz;
  });
  const Kfloor = 1e-6 * E * p.stiffness_reference_km;
  const Keff = Math.max(K, Kfloor);
  return { S0, K, stiffness: Keff / (E * p.stiffness_reference_km), zeroStrain: strain0 - S0 / Keff, Scomp, Stens };
}

/**
 * The multiplier bounds for the solve (solver.wgsl: lambda > 0 pushes).
 * lambda = -rate dt rest S / (E h_ref), with rest on the unit sphere.
 */
export function lambdaBounds(lin, rest, p) {
  const scale = (p.bond_rate_myr_inv * p.dt_crust_myr * rest) / (E_MPA(p) * p.stiffness_reference_km);
  return { lambdaMin: -scale * lin.Stens, lambdaMax: scale * lin.Scomp };
}

/**
 * After the solve: advance each slice by the strain increment (implicit
 * Euler), cap, and store. An open crack stores its elastic trial stress less
 * the creep its actual (capped) stress drives over the step, E dt / eta x cap.
 *
 * Also the irreversible bookkeeping of the step (design 4.7, phase D):
 *   inelastic: the edge's strain increment not stored elastically (creep and
 *     plastic flow; not crack opening, which closes again) in its stiffest
 *     slice (largest r; the first of equals). A column shortens permanently
 *     only as fast as its strongest layer lets it: a cold lid that does not
 *     yield holds it however much the hot base creeps (stress relaxing below
 *     is load moving to the lid, not shortening); a lid at its cap lets it
 *     flow; an all-hot column flows by its coldest part. Columns turn it into
 *     area_ref change (volumetric flow).
 *   work: plastic work this step, MPa km (cap x plastic strain x dz, summed):
 *     slices at a cap, and cracks opening against cohesion. Creep below the
 *     cap is not counted. Drives damage.
 * Returns the stored stresses, the actual (capped) ones, S, inelastic, work.
 */
export function maxwellUpdate(stress, edge, dStrain, p) {
  const E = E_MPA(p);
  const stored = [], actual = [];
  let S = 0, work = 0, inelastic = 0, rMax = -1;
  edge.slices.forEach((s, k) => {
    const elastic = stress[k] + E * dStrain;       // no creep
    const relaxed = elastic * s.r;                 // Maxwell, implicit Euler
    const capped = Math.min(Math.max(relaxed, -s.comp), s.tens);
    const crack = relaxed > s.tens && s.cracks;
    const kept = crack ? elastic - (1 / s.r - 1) * s.tens : capped;
    actual.push(capped);
    stored.push(kept);
    S += capped * s.dz;
    if (crack) work += s.tens * Math.max(dStrain, 0) * s.dz;
    else if (relaxed !== capped) work += Math.abs(capped * (relaxed - capped) / (E * Math.max(s.r, 1e-30))) * s.dz;
    if (s.r > rMax) { rMax = s.r; inelastic = dStrain - (kept - stress[k]) / E; }
  });
  return {
    stored, actual, S, work,
    inelastic: edge.overlap > 0 ? inelastic : 0,
    meanStress: edge.overlap > 0 ? S / edge.overlap : 0,
  };
}

/** Temperature of the edge's load-bearing part: slice temperatures weighted by elastic weight r dz. */
export function healingTemperature(edge) {
  let w = 0, wt = 0;
  for (const s of edge.slices) { const x = (s.r + 1e-9) * s.dz; w += x; wt += x * s.T; }
  return w > 0 ? wt / w : 0;
}

/** Healing time (Myr) at temperature T: heal_time_myr at mantle temperature, Arrhenius below it. */
export function healTimeMyr(T, p) {
  return p.heal_time_myr * Math.exp(Math.min((p.heal_activation_kj / p.gas_constant) * (1 / Math.max(T, 1) - 1 / p.mantle_temp), 80));
}

/**
 * Integrity law (design 4.7), one step: damage by plastic work,
 * phi -= W / W_D (floored at 0), then healing toward 1 with time scale
 * tau_heal(T), integrated exactly: 1 - phi decays by exp(-dt / tau).
 */
export function integrityStep(phi, work, T, p) {
  const damaged = Math.max(phi - work / p.damage_work_mpa_km, 0);
  return 1 - (1 - damaged) * Math.exp(-p.dt_crust_myr / healTimeMyr(T, p));
}
