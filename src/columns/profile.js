// Column profile, CPU reference (design 4.0, phase B).
//
// Pure functions mirroring src/shaders/compute/profile.wgsl. A column is
// described by a few numbers (crust thickness and composition, thermal age,
// elevation); from them follow its temperature T(z), density rho(z),
// lithostatic pressure P(z), ductile flow strength and viscosity, brittle
// strength, and the column totals the rest of the model reads: lithosphere
// thickness, Airy elevation, gravitational potential energy (GPE).
//
// Depth z is measured in km below the column top. Strengths are shear
// stresses in MPa. Order-one geometric factors between laboratory
// (uniaxial) flow laws and tensor invariants are ignored; strengths are
// calibrated at integration (design 9).

import { lithosphereThicknessKm, CONTINENTAL_DENSITY_MAX } from '../edges/topology.js';

export const SECONDS_PER_MYR = 3.15576e13;
export const GRAVITY = 9.81;
/** Slices per column: CRUST_SLICES over the crust, the rest over the mantle lithosphere. */
export const SLICES = 8;
export const CRUST_SLICES = 3;
/** Quadrature: 2-point Gauss-Legendre on this many equal sub-intervals per segment. */
export const CRUST_INTERVALS = 6;
export const MANTLE_INTERVALS = 16;
const GL = [0.5 - 0.5 / Math.sqrt(3), 0.5 + 0.5 / Math.sqrt(3)];

/** erf, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7); the kernel uses the same formula. */
export function erf(x) {
  const s = Math.sign(x), a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}

export const isContinental = (compositionDensity) => compositionDensity < CONTINENTAL_DENSITY_MAX;

/**
 * Steady or cooling geotherm of a column. Returns T(z) (K, z in km), the
 * lithosphere base L (km; where T reaches 90% of the way to mantle
 * temperature), the depth below which T = mantle temperature to within
 * 4e-7 (zMantle, km), and the surface heat flow (mW/m^2).
 *
 * Oceanic: half-space cooling, T = Ts + (Tm - Ts) erf(z / 2 sqrt(kappa t)),
 * with the age capped where L reaches oceanic_lithosphere_max_km (a plate
 * model's flattening, approximated).
 * Continental: steady conduction with heat production A in the top
 * min(radiogenic_layer_km, crust) of the crust, T(0) = Ts and T(L) = T_L,
 * L = 2.32 sqrt(kappa t) of the thermal age t, capped at continental_lithosphere_km
 * (stretching thins it; src/columns/volume.js); below L the gradient continues until T = Tm.
 */
export function geotherm(col, p) {
  const Ts = p.ambient_ocean_temp, Tm = p.mantle_temp, dT = Tm - Ts;
  const k = p.thermal_conductivity_w_m_k;
  if (isContinental(col.compositionDensity)) {
    const L = lithosphereThicknessKm(col.compositionDensity, col.ageMyr, p);
    const hr = Math.min(p.radiogenic_layer_km, col.crustKm) * 1e3;
    const A = p.radiogenic_heat_uw_m3 * 1e-6;
    const TL = Ts + 0.9 * dT;
    const qm = Math.max((k * (TL - Ts) - 0.5 * A * hr * hr) / (Math.max(L, 1) * 1e3), 1e-3); // W/m^2 from below
    const T = (zKm) => {
      const z = zKm * 1e3;
      const t = z < hr ? Ts + ((qm + A * hr) * z - 0.5 * A * z * z) / k : Ts + (qm * z + 0.5 * A * hr * hr) / k;
      return Math.min(t, Tm);
    };
    const zMantle = (k * (Tm - Ts) - 0.5 * A * hr * hr) / qm / 1e3;
    return { T, L, zMantle: Math.max(zMantle, L), heatFlow: (qm + A * hr) * 1e3 };
  }
  const L = lithosphereThicknessKm(col.compositionDensity, col.ageMyr, p);
  const s = L / 1.16; // 2 sqrt(kappa t_eff), km: L = 2.32 sqrt(kappa t_eff)
  const T = (zKm) => (s > 0 ? Ts + dT * erf(zKm / s) : zKm > 0 ? Tm : Ts);
  const tEff = Math.max(((s * 1e3) / 2) ** 2 / p.thermal_diffusivity_m2_s, 1e-3 * SECONDS_PER_MYR);
  return { T, L, zMantle: 3.6 * s, heatFlow: (k * dT) / Math.sqrt(Math.PI * p.thermal_diffusivity_m2_s * tEff) * 1e3 };
}

/**
 * Crust density at mantle temperature: its composition, or for a slab's
 * oceanic crust below eclogite_depth_km (the depth of the slab's middle,
 * top + slabExtentKm / 2, +-10 km) eclogite.
 */
export function crustDensity(col, p) {
  if (!col.slab || isContinental(col.compositionDensity)) return col.compositionDensity;
  const depth = -col.elevationKm + 0.5 * (col.slabExtentKm ?? 0);
  const f = Math.min(Math.max((depth - (p.eclogite_depth_km - 10)) / 20, 0), 1);
  return col.compositionDensity + f * (p.eclogite_density - col.compositionDensity);
}

/** Density at depth z: composition density at mantle temperature, contracted by cooling. */
function densityAt(col, g, p, z) {
  const rho0 = z < col.crustKm ? crustDensity(col, p) : p.mantle_density;
  return rho0 * (1 + p.thermal_expansion_coeff * (p.mantle_temp - g.T(z)));
}

/** Sum of f over [a, b] by 2-point Gauss-Legendre on n equal sub-intervals. */
function integrate(f, a, b, n) {
  if (!(b > a)) return 0;
  const h = (b - a) / n;
  let s = 0;
  for (let i = 0; i < n; i++) for (const x of GL) s += f(a + (i + x) * h);
  return 0.5 * h * s;
}

/** Integral of f over the column [0, zEnd], segment by segment (crust, mantle). */
function overColumn(col, zEnd, f) {
  return integrate(f, 0, Math.min(col.crustKm, zEnd), CRUST_INTERVALS)
    + integrate(f, col.crustKm, zEnd, MANTLE_INTERVALS);
}

/** Bottom of the integration: below it the column is mantle at mantle temperature. */
const columnEnd = (col, g) => Math.max(col.crustKm, g.zMantle);

/**
 * Airy isostasy. freeboard e0 = integral of (1 - rho / rho_m) dz (km above the
 * datum, the top of a dry mantle at mantle temperature); under water the
 * column is also loaded by the water above it:
 * e = e0 - (s - e0) rho_w / (rho_m - rho_w).
 */
export function airy(col, p, g = geotherm(col, p)) {
  const freeboard = overColumn(col, columnEnd(col, g), (z) => 1 - densityAt(col, g, p, z) / p.mantle_density);
  const s = p.sea_level;
  const elevation = freeboard >= s ? freeboard
    : freeboard - ((s - freeboard) * p.water_density) / (p.mantle_density - p.water_density);
  return { freeboard, elevation };
}

/** Lithostatic pressure at depth z (MPa), from the sea surface or the column top. */
export function pressureAt(col, p, zKm, g = geotherm(col, p), elevation = col.elevationKm) {
  // Overburden: sea water above a surface column, mantle rock above a slab.
  const water = (col.slab ? p.mantle_density : p.water_density) * Math.max(p.sea_level - elevation, 0) * 1e3;
  const rock = integrate((z) => densityAt(col, g, p, z), 0, Math.min(zKm, col.crustKm), 4)
    + integrate((z) => densityAt(col, g, p, z), col.crustKm, zKm, 8);
  return (GRAVITY * (water + rock * 1e3)) / 1e6;
}

/**
 * Gravitational potential energy per unit area, relative to the reference
 * mantle column (MPa km = 1e9 N/m), at the Airy elevation:
 * Gamma = g * integral over height r of (rho(r) - rho_ref(r)) (r - r_b) dr,
 * rho_ref = mantle density below the datum and 0 above, r_b any height
 * below the thermal layer (in equilibrium the result does not depend on it).
 * Differences in Gamma between columns are the horizontal forces of ridge
 * push and plateau collapse (design 4.6).
 * `extraDepthKm` moves r_b deeper (tests check the result does not change).
 */
export function gpe(col, p, g = geotherm(col, p), elevation = airy(col, p, g).elevation, extraDepthKm = 0) {
  const zEnd = columnEnd(col, g) + extraDepthKm;
  const e = elevation, rb = e - zEnd; // heights above the datum, km
  const rhoRef = (z) => (z > e ? p.mantle_density : 0); // depth z below the column top is below the datum
  const f = (z) => (densityAt(col, g, p, z) - rhoRef(z)) * (zEnd - z);
  // Split the segments at the datum, where rho_ref jumps.
  const cuts = [0, col.crustKm, zEnd, e].filter((z) => z >= 0 && z <= zEnd).sort((a, b) => a - b);
  let rock = 0;
  for (let k = 0; k + 1 < cuts.length; k++) rock += integrate(f, cuts[k], cuts[k + 1], cuts[k] < col.crustKm ? CRUST_INTERVALS : MANTLE_INTERVALS);
  // Water from the column top (height e) to sea level: rho_w - rho_ref, exact.
  const lin = (lo, hi) => (hi > lo ? 0.5 * ((hi - rb) ** 2 - (lo - rb) ** 2) : 0);
  const s = p.sea_level;
  const water = e < s ? p.water_density * lin(e, s) - p.mantle_density * lin(e, Math.min(0, s)) : 0;
  return (GRAVITY * (rock + water) * 1e6) / 1e9; // km^2 -> m^2, Pa m -> MPa km
}

/** Flow law of the material at depth z: crust (quartzite or diabase) or mantle (olivine). */
export function flowLaw(col, z, p) {
  const name = z >= col.crustKm ? 'olivine' : isContinental(col.compositionDensity) ? 'quartzite' : 'diabase';
  return { name, log10A: p[`${name}_log10_a`], n: p[`${name}_n`], Q: p[`${name}_q_kj`] };
}

/**
 * Ductile (dislocation creep) strength at temperature T, as natural logs:
 * strain rate = A sigma^n exp(-Q/RT) at the reference strain rate gives the
 * differential stress sigma; shear strength = sigma / 2 (MPa); viscosity
 * eta = strength / (2 strain rate) (Pa s).
 */
export function ductile(law, T, p) {
  const lnSigma = (Math.log(p.reference_strain_rate_s) - law.log10A * Math.LN10) / law.n + law.Q / (law.n * p.gas_constant * T);
  const lnStrength = lnSigma + Math.log(0.5);
  return { lnStrength, log10Viscosity: (lnStrength + Math.log(1e6) - Math.log(2 * p.reference_strain_rate_s)) / Math.LN10 };
}

/** Ductile strength capped at 1e6 MPa (beyond any brittle strength; keeps f32 finite). */
export const ductileStrength = (law, T, p) => Math.exp(Math.min(ductile(law, T, p).lnStrength, Math.log(1e6)));

/** Brittle (frictional) shear strength of a plane under the effective pressure (Coulomb: cohesion scaled by integrity phi, plus Byerlee friction). */
export const brittleStrength = (P, phi, p) => phi * p.cohesion_mpa + p.friction_coefficient * (1 - p.pore_fluid_factor) * P;

/**
 * Anderson's theory of faulting: what a brittle layer transmits horizontally
 * is the principal stress difference at failure, sigma_1 = C0 + q sigma_3
 * (effective stresses), q = ((1 + mu^2)^(1/2) + mu)^2 (3.12 at mu = 0.6), C0 =
 * 2 c q^(1/2) the uniaxial strength from the cohesion c. Under the vertical
 * effective stress P' = (1 - lambda) P: compression (thrust faults)
 * sigma_xx - sigma_zz = C0 + (q - 1) P'; extension (normal faults)
 * sigma_zz - sigma_xx = C0 / q + (1 - 1 / q) P' (Brace & Kohlstedt 1980; the
 * brittle part of a yield-strength envelope). Integrity scales the cohesion
 * in compression, and the whole strength in tension (a broken cold pair
 * carries no tension: design 4.5).
 */
export const andersonRatio = (mu) => (Math.sqrt(1 + mu * mu) + mu) ** 2;
export function brittleCompression(P, phi, p) {
  const q = andersonRatio(p.friction_coefficient);
  return phi * 2 * p.cohesion_mpa * Math.sqrt(q) + (q - 1) * (1 - p.pore_fluid_factor) * P;
}
export function brittleTension(P, phi, p) {
  const q = andersonRatio(p.friction_coefficient);
  return phi * (2 * p.cohesion_mpa / Math.sqrt(q) + (1 - 1 / q) * (1 - p.pore_fluid_factor) * P);
}

/**
 * Yield-strength envelope at depth z, in horizontal differential stress
 * (what the normal channel of an edge carries): the weaker of brittle
 * compression (Anderson) and ductile flow stress (twice the shear strength).
 */
export function envelope(col, p, z, phi = 1, g = geotherm(col, p)) {
  const b = brittleCompression(pressureAt(col, p, z, g), phi, p);
  const d = 2 * ductileStrength(flowLaw(col, z, p), g.T(z), p);
  return { brittle: b, ductile: d, strength: Math.min(b, d) };
}

/**
 * Brittle-ductile transition (km): the shallowest depth where ductile
 * strength falls below brittle strength (phi = 1). Found as the kernel does:
 * the first slice whose midpoint is ductile, then bisection between the
 * previous midpoint (or the top) and it. The column bottom if every slice is brittle.
 */
export function brittleDuctileDepth(col, p, slices, g = geotherm(col, p)) {
  const ductileAt = (z) => { const e = envelope(col, p, z, 1, g); return e.ductile < e.brittle; };
  let lo = 0;
  for (const s of slices) {
    if (ductileAt(s.z)) {
      let hi = s.z;
      for (let it = 0; it < 30; it++) { const m = 0.5 * (lo + hi); if (ductileAt(m)) hi = m; else lo = m; }
      return 0.5 * (lo + hi);
    }
    lo = s.z;
  }
  const last = slices[slices.length - 1];
  return last.z + 0.5 * last.dz;
}

/**
 * Slice layout: CRUST_SLICES equal slices over the crust, the rest over the
 * mantle lithosphere [crust, D], D = max(L, crust) (zero thickness if the
 * lithosphere is no thicker than the crust).
 */
export function sliceLayout(crustKm, columnKm) {
  const out = [];
  for (let k = 0; k < SLICES; k++) {
    const crust = k < CRUST_SLICES;
    const top = crust ? 0 : crustKm, bottom = crust ? crustKm : columnKm;
    const n = crust ? CRUST_SLICES : SLICES - CRUST_SLICES, m = crust ? k : k - CRUST_SLICES;
    const dz = (bottom - top) / n;
    out.push({ z: top + (m + 0.5) * dz, dz });
  }
  return out;
}

/**
 * The full profile of one column, as the kernel writes it (ColumnProfile in
 * layout.js). `col` = { crustKm, compositionDensity, ageMyr, elevationKm }.
 */
export function columnProfile(col, p) {
  const g = geotherm(col, p);
  const { freeboard, elevation } = airy(col, p, g);
  const columnKm = Math.max(g.L, col.crustKm);
  const slices = sliceLayout(col.crustKm, columnKm).map(({ z, dz }) => {
    const T = g.T(z), law = flowLaw(col, z, p), d = ductile(law, T, p);
    return {
      z, dz, T,
      density: densityAt(col, g, p, z),
      pressure: pressureAt(col, p, z, g),
      log10Viscosity: d.log10Viscosity,
      ductile: Math.exp(Math.min(d.lnStrength, Math.log(1e6))),
    };
  });
  return {
    lithosphereKm: g.L,
    columnKm,
    freeboardKm: freeboard,
    airyElevationKm: elevation,
    gpe: gpe(col, p, g, elevation),
    heatFlow: g.heatFlow,
    mohoTemp: g.T(col.crustKm),
    brittleDuctileKm: brittleDuctileDepth(col, p, slices, g),
    slices,
  };
}

/** The column inputs of particle i, from unpacked Particle arrays (layout.js). */
export const columnOfParticle = (P, i) => ({
  crustKm: P.thickness_km[i], compositionDensity: P.composition_density[i], ageMyr: (P.thermal_age_myr ?? P.age)[i], elevationKm: P.elevation[i], slabExtentKm: P.slab_extent_km?.[i] ?? 0,
  slab: P.state ? P.state[i] === 2 : false,
});

/**
 * Compare GPU profiles (unpacked ColumnProfile arrays) with this reference
 * for the given particles. Returns, per field, the worst error relative to
 * `scale` (a typical magnitude, so near-zero values do not blow up) and where.
 */
export function compareProfiles(P, G, p, indices) {
  const scale = {
    lithosphere_km: 100, column_km: 100, freeboard_km: 1, airy_elevation_km: 1, gpe_mpa_km: 1000,
    heat_flow_mw_m2: 50, moho_temp_k: 1000, brittle_ductile_km: 10,
    depth_km: 10, thickness_km: 10, temp_k: 1000, density: 3000, pressure_mpa: 100, log10_viscosity: 1, ductile_mpa: null,
  };
  const worst = {};
  const note = (field, i, gpu, ref) => {
    // Ductile strength spans many decades: compare its logarithm.
    const err = scale[field] === null ? Math.abs(Math.log(gpu / ref)) : Math.abs(gpu - ref) / scale[field];
    if (!(err <= (worst[field]?.err ?? -1))) worst[field] = { err, particle: i, gpu, ref };
  };
  for (const i of indices) {
    const r = columnProfile(columnOfParticle(P, i), p);
    note('lithosphere_km', i, G.lithosphere_km[i], r.lithosphereKm);
    note('column_km', i, G.column_km[i], r.columnKm);
    note('freeboard_km', i, G.freeboard_km[i], r.freeboardKm);
    note('airy_elevation_km', i, G.airy_elevation_km[i], r.airyElevationKm);
    note('gpe_mpa_km', i, G.gpe_mpa_km[i], r.gpe);
    note('heat_flow_mw_m2', i, G.heat_flow_mw_m2[i], r.heatFlow);
    note('moho_temp_k', i, G.moho_temp_k[i], r.mohoTemp);
    note('brittle_ductile_km', i, G.brittle_ductile_km[i], r.brittleDuctileKm);
    r.slices.forEach((s, k) => {
      const at = SLICES * i + k;
      note('depth_km', i, G.depth_km[at], s.z);
      note('thickness_km', i, G.thickness_km[at], s.dz);
      note('temp_k', i, G.temp_k[at], s.T);
      note('density', i, G.density[at], s.density);
      note('pressure_mpa', i, G.pressure_mpa[at], s.pressure);
      note('log10_viscosity', i, G.log10_viscosity[at], s.log10Viscosity);
      note('ductile_mpa', i, G.ductile_mpa[at], s.ductile);
    });
  }
  return worst;
}
