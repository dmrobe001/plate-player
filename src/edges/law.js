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
//   tension cap:     if brittle, phi (c + friction) / q, q = ((1 + mu^2)^(1/2) + mu)^2:
//                    slice stresses are deviatoric (from the lithostatic
//                    state), and a brittle slice in deviatoric extension fails
//                    by normal faulting at Anderson's fraction 1 / q of its
//                    compressive strength, not at cohesion (that is a crack
//                    in absolute tension, near the surface only). So intact
//                    plates can transmit slab pull (design 6); broken edges
//                    (phi = 0) still carry none. If ductile: the ductile flow
//                    stress (sign-blind: hot rock resists separation), reduced
//                    by lost integrity if ductile_damage_weakening > 0
//
// What happens above a cap:
//   - ductile slices flow, and brittle compression slips: the stored stress
//     is the capped stress (the slice's stress-free length moves; the pair's
//     rest length does not; the isotropic part goes to area_ref in phase D);
//   - brittle tension: the slice carries its cap; what it is stretched beyond
//     is split by integrity. The intact share phi is normal faulting:
//     permanent, stored at the cap, inelastic strain (it thins the column).
//     The broken share 1 - phi is a crack: it stays referenced to the rest
//     length (the elastic trial stress, less only the creep driven by the
//     stress it actually carries), so it closes, and a contact re-engages,
//     exactly at touching. (All crack before calibration, design 9.2: an
//     intact plate stretched past its strength then pulled back at the cap
//     once the load went.)
//
// Within a step the law is linear (Chebyshev needs that): every slice
// contributes its decayed stored stress and its stiffness E r, and the total
// is clamped to the depth-integrated caps (lambda bounds). So a cracked pair
// is pulled together at most at its tension cap until its slices close, at
// touching, where the law is stiff again: no overshoot.
//
// Stresses are normal stresses on the interface (MPa, + tension), integrated
// over depth as S (MPa km).
//
// Tangential channel (phase G). Each slice also carries a tangential stress
// tau, a Maxwell element of stiffness G_t = shear_stiffness_ratio x E and the
// same Maxwell factor r. Its strain increment is the faces' relative sliding
// over the step, over the rest length:
//     delta_t = L0 psi - l_a theta_a - l_b theta_b
// psi the rotation of the pair line, theta the columns' spins, l the levers
// from each centre to the contact point (l_a + l_b = L0). Rigid rotation
// (psi = theta_a = theta_b) slides nothing. Caps per slice, symmetric:
//     friction  phi c + mu max((1 - lambda_pore) P - sigma_n, 0)
//     (sigma_n the slice's stored normal stress, + tension: compression across
//     the face strengthens it, tension weakens it, an open crack is frictionless)
//     or the ductile flow stress, whichever is weaker.
// Above the cap a brittle slice slips: that is a fault, a jump between the
// columns, and goes to `slip`. Inelastic shear of a ductile slice is spread
// through the columns and goes to `ductile`, which changes their shapes
// (src/columns/shape.js). Plastic work of both kinds feeds damage.
//
// Rolling (phase G). Flat faces resist relative rotation of the two columns:
// a rotation dtheta opens one end of the face and closes the other, a moment
// K face^2 / (12 rest) dtheta (K the normal stiffness). It decays with the
// mean Maxwell factor and is capped by the kern rule: the face opens at one
// end once |M| > (compression + tension cap) x face / 6. Without it a column
// in a slipping zone could spin like a ball bearing.

import { geotherm, ductile, flowLaw, columnProfile, andersonRatio, brittleCompression, brittleTension, GRAVITY, SECONDS_PER_MYR } from '../columns/profile.js';

export const EDGE_SLICES = 8;
const LN_STRENGTH_CAP = Math.log(1e6);


/** Anderson's ratio of frictional strength in compression to that in extension, ((1 + mu^2)^(1/2) + mu)^2. */
export { andersonRatio };

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
 * ({ crustKm, compositionDensity, ageMyr, elevationKm, slab? }), phi: integrity.
 * Per slice: dz (km), r (Maxwell factor), comp and tens caps (MPa), and
 * whether tension is brittle-limited (cracks) rather than ductile (flows).
 */
export function edgeSlices(a, b, phi, p, normalStress = null) {
  const top = Math.min(a.elevationKm, b.elevationKm);
  const bottom = Math.max(a.elevationKm - columnDepthKm(a, p), b.elevationKm - columnDepthKm(b, p));
  const overlap = Math.max(top - bottom, 0);
  const dz = overlap / EDGE_SLICES;
  const ends = [a, b].map((col) => ({ col, g: geotherm(col, p), prof: columnProfile(col, p) }));
  const lnEdt = Math.log(p.crust_youngs_modulus_gpa * 1e9 * p.dt_crust_myr * SECONDS_PER_MYR);
  // Subduction interface (exactly one slab end): lubricated in compression.
  const iface = !!a.slab !== !!b.slab ? p.slab_interface_weakening : 1;
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
    // Shear strength (the tangential channel's) and, for the normal channel,
    // horizontal differential stress: twice it when ductile, Anderson's
    // thrust / normal-fault strength when brittle (profile.js).
    const ductileCap = 1 / invStrength;
    const ductileN = 2 * ductileCap;
    const brittleComp = brittleCompression(pressure, phi, p);
    const brittleTens = brittleTension(pressure, phi, p);
    const brittle = brittleTension(pressure, 1, p) < ductileN; // the mechanism, from intact rock
    const sigmaN = normalStress ? normalStress[k] : 0;
    const brittleShear = phi * p.cohesion_mpa + p.friction_coefficient * Math.max((1 - p.pore_fluid_factor) * pressure - sigmaN, 0);
    slices.push({
      z, dz, T,
      r: 1 / (1 + edtOverEta),
      comp: Math.min(brittleComp, ductileN) * iface,
      tens: brittle ? brittleTens : ductileN * (1 - p.ductile_damage_weakening * (1 - phi)),
      cracks: brittle,
      shear: Math.min(brittleShear, ductileCap),
      slips: brittleShear < ductileCap,
    });
  }
  return { top, bottom, overlap, slices, phi };
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
  let S = 0, work = 0, inelastic = 0, rMax = -1, capSum = 0;
  edge.slices.forEach((s, k) => {
    const elastic = stress[k] + E * dStrain;       // no creep
    const relaxed = elastic * s.r;                 // Maxwell, implicit Euler
    const capped = Math.min(Math.max(relaxed, -s.comp), s.tens);
    const crack = relaxed > s.tens && s.cracks;
    // Beyond a brittle tension cap: the broken share 1 - phi is a crack (trial stress kept), the rest is permanent.
    const kept = crack ? s.tens + (1 - edge.phi) * (elastic - (1 / s.r - 1) * s.tens - s.tens) : capped;
    actual.push(capped);
    stored.push(kept);
    S += capped * s.dz;
    if (crack) work += s.tens * Math.max(dStrain, 0) * s.dz;
    else if (relaxed !== capped) work += Math.abs(capped * (relaxed - capped) / (E * Math.max(s.r, 1e-30))) * s.dz;
    if (s.r > rMax) { rMax = s.r; inelastic = dStrain - (kept - stress[k]) / E; }
    capSum += (dStrain > 0 ? s.tens : s.comp) * s.dz;
  });
  return {
    stored, actual, S, work,
    // Plastic strain weighted by the share of the strength that yields (work over the caps).
    plasticStrain: capSum > 0 ? work / capSum : 0,
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
 * Integrity law (design 4.7), one step: slip weakening. `slipKm` is the
 * step's plastic displacement across the edge, weighted by the share of its
 * strength that yields (plasticStrain of both channels x rest length, km);
 * phi -= slip / damage_slip_km (floored at 0), then healing toward 1 with
 * time scale tau_heal(T), integrated exactly: 1 - phi decays by exp(-dt / tau).
 */
export function integrityStep(phi, slipKm, T, p) {
  const damaged = Math.max(phi - slipKm / p.damage_slip_km, 0);
  return 1 - (1 - damaged) * Math.exp(-p.dt_crust_myr / healTimeMyr(T, p));
}

// --- Tangential channel and rolling (phase G) ------------------------------

/**
 * Sliding of the faces over the step (unit sphere): positions at the last
 * update (xa0, xb0; the GPU keeps their difference as the edge's `direction`)
 * and now (xa, xb), unit vectors; L0 the start distance; f the
 * lever fraction r_a / (r_a + r_b); spins (rad, CCW about the outward radial).
 * Positive when b moves along +t = up x n relative to a.
 */
export function tangentialDisplacement(xa0, xb0, xa, xb, L0, f, spinA, spinB) {
  const d0 = xb0.map((v, c) => v - xa0[c]), d = xb.map((v, c) => v - xa[c]);
  const up = xa.map((v, c) => v + xb[c]), lu = Math.hypot(...up);
  const cr = [d0[1] * d[2] - d0[2] * d[1], d0[2] * d[0] - d0[0] * d[2], d0[0] * d[1] - d0[1] * d[0]];
  const psi = Math.atan2((cr[0] * up[0] + cr[1] * up[1] + cr[2] * up[2]) / lu, d0[0] * d[0] + d0[1] * d[1] + d0[2] * d[2]);
  return L0 * psi - f * L0 * spinA - (1 - f) * L0 * spinB;
}

/**
 * Linear tangential law for this step: S_t = S0 + K (delta_t / rest),
 * clamped to +-Scap. Returns S0, K (MPa km per unit strain), the stiffness
 * factor K / (E h_ref) (0 = no channel), the offset at zero force (unit
 * sphere) and the multiplier bound.
 */
export function linearizeShear(tau, edge, rest, p) {
  const E = E_MPA(p), ratio = p.shear_stiffness_ratio;
  let S0 = 0, K = 0, Scap = 0;
  edge.slices.forEach((s, k) => {
    S0 += tau[k] * s.r * s.dz;
    K += ratio * E * s.r * s.dz;
    Scap += s.shear * s.dz;
  });
  if (!(ratio > 0) || !(edge.overlap > 0)) return { S0: 0, K: 0, stiffness: 0, zeroOffset: 0, Scap: 0, lambdaMax: 0 };
  const Keff = Math.max(K, 1e-6 * ratio * E * p.stiffness_reference_km);
  const scale = (p.bond_rate_myr_inv * p.dt_crust_myr * rest) / (E * p.stiffness_reference_km);
  return { S0, K, stiffness: Keff / (E * p.stiffness_reference_km), zeroOffset: (-S0 / Keff) * rest, Scap, lambdaMax: scale * Scap };
}

/**
 * After the solve: advance each slice's tangential stress by the strain
 * dGamma = delta_t / rest, cap, store. The stiffest slice decides what the
 * inelastic part is: slip on a fault if it is brittle and at its cap,
 * distributed (ductile) otherwise. Returns stored stresses, S, work, slip,
 * ductile and the mean stress.
 */
export function maxwellShear(tau, edge, dGamma, p) {
  const G = p.shear_stiffness_ratio * E_MPA(p);
  if (!(G > 0)) return { stored: tau.map(() => 0), S: 0, work: 0, plasticStrain: 0, slip: 0, ductile: 0, meanStress: 0 };
  const stored = [];
  let S = 0, work = 0, rMax = -1, inelastic = 0, faultSlip = false, capSum = 0;
  edge.slices.forEach((s, k) => {
    const relaxed = (tau[k] + G * dGamma) * s.r;
    const capped = Math.min(Math.max(relaxed, -s.shear), s.shear);
    stored.push(capped);
    S += capped * s.dz;
    capSum += s.shear * s.dz;
    if (relaxed !== capped) work += Math.abs(capped * (relaxed - capped) / (G * Math.max(s.r, 1e-30))) * s.dz;
    if (s.r > rMax) { rMax = s.r; inelastic = dGamma - (capped - tau[k]) / G; faultSlip = s.slips && relaxed !== capped; }
  });
  if (!(edge.overlap > 0)) inelastic = 0;
  return {
    stored, S, work,
    plasticStrain: capSum > 0 ? work / capSum : 0,
    slip: faultSlip ? inelastic : 0, ductile: faultSlip ? 0 : inelastic,
    meanStress: edge.overlap > 0 ? S / edge.overlap : 0,
  };
}

/**
 * Rolling law for this step: M = M0 + K_r (spin_b - spin_a), clamped to
 * +-cap. `moment` the stored moment, face the shared boundary (unit sphere),
 * S0n the start-of-step normal force (MPa km, + tension), Stens the tension
 * cap. Returns M0, K_r, cap, the solver stiffness factor and offset, and the bound.
 */
export function linearizeRolling(moment, edge, face, rest, S0n, Stens, p) {
  const E = E_MPA(p);
  let K = 0;
  edge.slices.forEach((s) => { K += E * s.r * s.dz; });
  if (!(face > 0) || !(edge.overlap > 0) || !(p.rolling_stiffness_factor > 0)) return { M0: 0, K: 0, cap: 0, stiffness: 0, zeroRotation: 0, lambdaMax: 0 };
  const rbar = K / (E * edge.overlap);
  const geom = (p.rolling_stiffness_factor * face * face) / 12;
  const Keff = Math.max(K, 1e-6 * E * p.stiffness_reference_km);
  const Kr = (Keff * geom) / rest;
  const M0 = moment * rbar;
  const cap = ((Math.max(-S0n, 0) + Stens) * face) / 6;
  const scale = (p.bond_rate_myr_inv * p.dt_crust_myr * rest) / (E * p.stiffness_reference_km);
  return { M0, K: Kr, cap, stiffness: (Keff / (E * p.stiffness_reference_km)) * geom, zeroRotation: -M0 / Kr, lambdaMax: scale * cap };
}

/** The stored moment after the step: M0 + K_r dTheta, capped (the face rocks open beyond it). */
export function rollingUpdate(lin, dTheta) {
  return Math.min(Math.max(lin.M0 + lin.K * dTheta, -lin.cap), lin.cap);
}
