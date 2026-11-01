// Crust volume, melt and splits (design 4.8, phase E). CPU reference for
// columns_volume (src/shaders/compute/area.wgsl) and columns_split
// (src/shaders/compute/split.wgsl).
//
// State per column: crust volume V (km x unit-sphere area) and thermal age.
// Thickness is derived, h = V / area, area being the tessellated cell
// (src/columns/area.js). Per step, from last step's cell area A0 to this
// step's A:
//   - Thermal age scales by (A0 / A)^2, so the lithosphere (2.32 sqrt(kappa t))
//     thins by A0 / A: pure-shear stretching thins crust and mantle lithosphere
//     alike (McKenzie 1978). It is capped where the lithosphere reaches its
//     full thickness for the composition (oceanic / continental).
//   - New area brings decompression melt (McKenzie & Bickle 1988):
//     dV = F H_m max(A - A0, 0) max(1 - L / H_m, 0), with L the lithosphere
//     after thinning. F H_m = 7 km: a steady ridge makes 7 km of crust.
//   - Melt is mixed into the column's composition by volume; a continental
//     column that crosses into oceanic density this way has broken up (logged).

import { CONTINENTAL_DENSITY_MAX, lithosphereThicknessKm } from '../edges/topology.js';
import { SECONDS_PER_MYR } from './profile.js';
import { tangentBasis } from './flow.js';

/** Thermal age at which the lithosphere reaches its cap for this composition, Myr. */
export function thermalAgeCapMyr(compositionDensity, p) {
  const cap = compositionDensity < CONTINENTAL_DENSITY_MAX ? p.continental_lithosphere_km : p.oceanic_lithosphere_max_km;
  return ((cap * 1e3) / 2.32) ** 2 / p.thermal_diffusivity_m2_s / SECONDS_PER_MYR;
}

/**
 * One step of a surface column's volume. c = {volume, area, areaPrev,
 * thermalAgeMyr, compositionDensity, ageMyr}; areaPrev <= 0 means no
 * dilatation this step (just loaded or split). Returns the new state plus
 * {added, lithosphereKm, thicknessKm, breakup}.
 */
export function volumeStep(c, p) {
  let { volume, thermalAgeMyr: t, compositionDensity: rho, ageMyr: age } = c;
  let added = 0, breakup = false;
  if (c.areaPrev > 0) {
    const ratio = c.areaPrev / c.area;
    t = Math.min(t, thermalAgeCapMyr(rho, p)) * ratio * ratio;
    const L = lithosphereThicknessKm(rho, t, p);
    const dA = c.area - c.areaPrev;
    if (dA > 0) added = p.melt_fraction * p.melt_depth_km * dA * Math.max(1 - L / p.melt_depth_km, 0);
    if (added > 0) {
      const mixed = (rho * volume + p.melt_density * added) / (volume + added);
      breakup = rho < CONTINENTAL_DENSITY_MAX && mixed >= CONTINENTAL_DENSITY_MAX;
      rho = mixed;
      if (breakup) age = 0;
      t = Math.min(t, thermalAgeCapMyr(rho, p));
      volume += added;
    }
  }
  return {
    volume, thermalAgeMyr: t, compositionDensity: rho, ageMyr: age, added, breakup,
    lithosphereKm: lithosphereThicknessKm(rho, t, p), thicknessKm: volume / c.area,
  };
}

/**
 * Where a split puts its two halves: at the cell centroid -/+ SPLIT_OFFSET x
 * spread along the cell's major axis (gnomonic tangent coordinates of the
 * parent; see polygonShape in area.js). For a cell made of two equal hexagons
 * the halves' centroids are at 0.885 spread, for a rectangle 0.866, for an
 * ellipse 0.849.
 */
export const SPLIT_OFFSET = 0.87;

/**
 * Split a column (design 4.8): p its unit position, shape {centroid, axis,
 * spread} of its cell. Parent and child share volume and area_ref equally.
 * Returns {parent: position, child: position, volume, areaRef} (each half's).
 */
export function splitColumn(p, shape, volume, areaRef) {
  const [e1, e2] = tangentBasis(p);
  const off = SPLIT_OFFSET * shape.spread;
  const ax = [Math.cos(shape.axis), Math.sin(shape.axis)];
  const at = (s) => {
    const u = [shape.centroid[0] + s * off * ax[0], shape.centroid[1] + s * off * ax[1]];
    const x = [0, 1, 2].map((k) => p[k] + u[0] * e1[k] + u[1] * e2[k]);
    const l = Math.hypot(...x);
    return x.map((v) => v / l);
  };
  return { parent: at(-1), child: at(1), volume: volume / 2, areaRef: areaRef / 2 };
}
