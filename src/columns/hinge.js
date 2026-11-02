// The trench hinge (design 6): a bend inside a surface column, where a plate
// goes down at a trench. CPU reference for columns_hinge (hinge.wgsl) and the
// slab attachment in thermal.wgsl.
//
// A plate on a buoyant (Winkler) foundation, loaded at its end, bends within
// the flexural length l = (4 D / K)^(1/4) of the end (D the flexural rigidity,
// K = drho g). With a vertical end load, the moment peaks at x* = pi l / 4
// behind the end (Hetenyi 1946): that is where the hinge sits, behind the
// trench line. A column much wider than l bends there as a hinge; the
// rotational stiffness of such an end is D / l. Up to the plastic moment M_y
// (the column's integrated strength) the hinge is elastic; beyond it, plastic.
//
// The flap is the part of the column beyond the hinge: x* outboard of the
// trench, plus u, the part already past the trench line (under the overriding
// plate, or past the hanging slab). Its length L_f = x* + u, its tip depth
// L_f sin(beta). Torques about the hinge per unit hinge length (N m / m), for
// a small bend beta:
//   drive  = F_s L_f + w_u (L_f^2 - x*^2) / 2
//            (F_s: weight of the slabs hanging from the tip, per unit length;
//             w_u: negative buoyancy per area of the underthrust part, which
//             is immersed in mantle rather than floating)
//   resist = (K x*^3 / 3 - P L_f) beta + M(beta)
//            (the foundation under the outboard part; in-plane compression P
//             destabilises once the flap dips; M the hinge moment, elastic
//             D / l (beta - beta_p) up to M_y)
// Quasi-static each step (the hinge relaxes on the isostatic time scale,
// ~30 kyr). Where the net stiffness is negative the bend runs away to
// beta_max: that is the instability that initiates subduction.

import { brittleCompression, brittleTension } from './profile.js';

export const EARTH_RADIUS_M = 6.371e6;

/** Flexural length (m) from rigidity D (N m) and foundation stiffness K (N/m^3). */
export const flexuralLength = (D, K) => (4 * D / K) ** 0.25;

/** Distance (m) of the hinge behind the trench line: where the end-loaded plate's moment peaks. */
export const hingeSetback = (ell) => (Math.PI * ell) / 4;

/** Rotational stiffness of the hinge, N m / m per radian. */
export const hingeStiffness = (D, ell) => (ell > 0 ? D / ell : 0);

/**
 * Plastic moment (N m / m) and tensile strength (N / m) of a column from its
 * slices [{depthKm, thicknessKm, pressureMpa, ductileMpa}], from its
 * yield-strength envelope in horizontal differential stress, as the edge
 * law's normal channel at full integrity (src/edges/law.js): per slice the
 * weaker of Anderson's brittle strength and the ductile flow stress (2 x
 * ductileMpa, a shear strength). A plate bending down at a trench is in
 * extension above its neutral plane and in compression below it; fully
 * plastic, the neutral plane is where the two forces balance, and the moment
 * is taken about it. Tensile strength (what holds a hanging slab, and the
 * most slab pull the column can pass to its plate): the tension envelope over
 * the whole column, so what a column transmits and what its edges hold agree.
 */
export function columnStrength(slices, p) {
  const st = slices.map((k) => Math.min(brittleTension(k.pressureMpa, 1, p), 2 * k.ductileMpa));
  const sc = slices.map((k) => Math.min(brittleCompression(k.pressureMpa, 1, p), 2 * k.ductileMpa));
  const n = slices.length;
  // Neutral plane: tension above it = compression below it. Slice m holds it,
  // a fraction f of the slice above it.
  const below = new Array(n + 1).fill(0);
  for (let i = n - 1; i >= 0; i--) below[i] = below[i + 1] + sc[i] * slices[i].thicknessKm;
  let above = 0, m = n - 1, f = 1;
  for (let i = 0; i < n; i++) {
    const t = st[i] * slices[i].thicknessKm, c = sc[i] * slices[i].thicknessKm;
    if (above + t >= below[i + 1]) { m = i; f = t + c > 0 ? Math.min(Math.max((below[i + 1] + c - above) / (t + c), 0), 1) : 0.5; break; }
    above += t;
  }
  const top = (i) => slices[i].depthKm - slices[i].thicknessKm / 2;
  const zc = n ? top(m) + f * slices[m].thicknessKm : 0;
  let moment = 0, tension = 0;
  slices.forEach((k, i) => {
    tension += st[i] * k.thicknessKm;
    if (i < m) moment += st[i] * k.thicknessKm * (zc - k.depthKm);
    else if (i > m) moment += sc[i] * k.thicknessKm * (k.depthKm - zc);
    else { const u = f * k.thicknessKm, l = (1 - f) * k.thicknessKm; moment += (st[i] * u * u + sc[i] * l * l) / 2; }
  });
  return { plasticMoment: moment * 1e12, tensileStrength: tension * 1e9, neutralKm: zc };
}

/** Drive (N m / m) and net linear stiffness (N m / m / rad) of the flap. */
export function hingeTorques({ slabLoad, underthrustWeight, K, compression, setback, flap }) {
  return {
    drive: slabLoad * flap + underthrustWeight * (flap * flap - setback * setback) / 2,
    stiffness: (K * setback ** 3) / 3 - compression * flap,
  };
}

/**
 * Quasi-static bend: drive = stiffness beta + M(beta), M = kh (beta - betaPlastic)
 * capped at +-My. Returns {beta, betaPlastic, branch: 'elastic' | 'plastic' | 'runaway'},
 * beta in [0, betaMax].
 */
export function solveHinge({ drive, stiffness, kh, My, betaPlastic = 0, betaMax = 1 }) {
  const clamp = (b) => Math.min(Math.max(b, 0), betaMax);
  const den = stiffness + kh;
  if (den > 0) {
    const beta = (drive + kh * betaPlastic) / den;
    const M = kh * (beta - betaPlastic);
    if (Math.abs(M) <= My) return { beta: clamp(beta), betaPlastic, branch: 'elastic' };
  }
  // Yielding: the hinge carries +-My, the rest is stiffness beta.
  const sign = den > 0 && drive + kh * betaPlastic < 0 ? -1 : 1;
  if (stiffness > 0) {
    const beta = clamp((drive - sign * My) / stiffness);
    return { beta, betaPlastic: kh > 0 ? beta - sign * My / kh : betaPlastic, branch: 'plastic' };
  }
  return { beta: betaMax, betaPlastic: kh > 0 ? Math.max(betaPlastic, betaMax - My / kh) : betaPlastic, branch: 'runaway' };
}

/**
 * Compression cap factor of the trench contact, a fault dipping at the flap's
 * bend beta: a vertical face (beta = 0) cannot slip under horizontal push, so
 * it keeps the full face strength (1); Coulomb resistance is least at the
 * optimal thrust dip theta* = atan(1 / mu) / 2, where the contact is the
 * lubricated interface (`weakening`). Linear in between.
 */
export function contactWeakening(beta, mu, weakening) {
  const opt = 0.5 * Math.atan(1 / mu);
  return 1 + (weakening - 1) * Math.min(Math.max(beta / opt, 0), 1);
}

/** A column becomes a slab when its hinge has passed its back face: the flap covers it. */
export const becomesSlab = (underthrust, setback, width) => underthrust + setback >= width;

/** Seconds per Myr, for rates in km/Myr. */
const MYR_S = 3.15576e13;

/**
 * A hanging slab's dip, one step (slab_hang in thermal.wgsl): pinned at its
 * top, it turns toward vertical under its own weight like a pendulum in a
 * viscous fluid, d dip / dt = (w / L) cos dip, w its free sinking rate, L its
 * length (km, km/Myr, Myr); at most pi / 2.
 */
export const slabDipStep = (dip, freeRate, lengthKm, dt) => Math.min(dip + (dt * freeRate) / Math.max(lengthKm, 1e-3) * Math.cos(dip), Math.PI / 2);

/** A slab's viscous drag coefficient down dip, N s / m: its weight over its free sinking rate (km/Myr). */
export const slabDrag = (weight, freeRate) => (freeRate > 0 ? weight / ((freeRate * 1000) / MYR_S) : 0);

/**
 * The tension at a slab's top, down dip, N: the down-dip weight of it and of
 * what hangs from it (childPull = sum W sin dip), less the mantle's drag on
 * them (drag coefficients, N s / m) as they move down dip at the feed rate
 * (km/Myr). Zero for a slab fed at its free speed (held by the mantle).
 */
export const slabTension = ({ weight, dip, freeRate, feed, childPull = 0, childDrag = 0 }) =>
  weight * Math.sin(dip) + childPull - (slabDrag(weight, freeRate) + childDrag) * ((feed * 1000) / MYR_S);

/**
 * A hanging slab (slab_hang in thermal.wgsl), one step. `load` is the
 * vertical load it passes up for the hinge's bend: its weight (the part above
 * the cutoff) less the viscous support of its descent (feed x sin dip, against
 * its free sinking rate: the vertical part of its drag), plus what hangs from
 * it. It necks when the down-dip tension at its top (slabTension) exceeds the
 * attachment's tensile strength `cap`: its top falls behind the hang point at
 * w_free (1 - cap / T), and the load passes up in proportion; necked through
 * its thickness, it breaks off. With `previous` (last step's load) the load is
 * under-relaxed, half the change per step: support, descent and bend depend
 * on each other one step apart, and unrelaxed the lagged fixed point flips.
 * Also returns what it passes up the chain: its pull (W sin dip + childPull)
 * and drag coefficient (W / w_free + childDrag), and its tension.
 */
export function hangSlab({ weight, dip, feed, freeRate, childLoad = 0, childPull = 0, childDrag = 0, cap, neck = 0, dt, thicknessKm, previous = null }) {
  const support = freeRate > 0 ? Math.min(Math.max((feed * Math.sin(dip)) / freeRate, 0), 1) : 0;
  let load = weight * (1 - support) + childLoad;
  if (previous !== null) load = 0.5 * (previous + load);
  const tension = slabTension({ weight, dip, freeRate, feed, childPull, childDrag });
  if (tension > cap) {
    neck += freeRate * (1 - cap / tension) * dt;
    load *= cap / tension;
  }
  return {
    load, tension, neck, broken: neck >= thicknessKm,
    pull: weight * Math.sin(dip) + childPull, drag: slabDrag(weight, freeRate) + childDrag,
  };
}

/**
 * The trench column's in-plane pull (columns_hinge in hinge.wgsl), N: the
 * down-dip weight of its slabs (childPull) and of its underthrust flap
 * (flapWeight sin beta), applied against its basal drag plus the slabs' and
 * flap's drag coefficient `drag`. What it transmits is the slabs' tension at
 * the trench, pull - drag x speed (speed toward the trench, m/s), and that is
 * what the column's and the plate's strength `cap` bound.
 */
export const hingePull = ({ childPull, flapWeight, beta, cap, drag, speed }) => Math.min(childPull + flapWeight * Math.sin(beta), cap + drag * speed);
