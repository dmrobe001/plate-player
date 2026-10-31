// A "lava lamp" mantle: a cheap, living stand-in for mantle convection that
// supplies the flow beneath the plates (the basal forcing of design.md 8.4).
//
// State: a buoyancy field b on an icosphere, read as the temperature anomaly
// of the mid-mantle (positive = hot). Each step:
//   1. flow: hot material rises and spreads, cold material converges and
//      sinks, so the horizontal flow beneath the plates is poloidal,
//      v = s grad(Phi) with (lap - 1/l^2) Phi = b. Hot is divergent. The
//      screening length l is the mantle depth: in a convecting layer of depth
//      H, the flow driven by a buoyancy anomaly dies off over ~H. Without it
//      the flow is dominated by the planet-wide imbalance of where plumes
//      happen to be, and its rms speed wanders by factors of several.
//      (Toroidal motion is not imposed: on Earth it comes mostly from the
//      plates themselves, through their rigidity and boundaries.)
//   2. advection of b by v (semi-Lagrangian, so stable for any step):
//      upwellings broaden, downwellings narrow, as in real convection.
//   3. a little diffusion, and decay with the lifetime of a plume.
//   4. new hot plumes and cold drips are born at random (Poisson in time,
//      uniform on the sphere), which keeps the pattern evolving instead of
//      settling into a fixed map.
// The velocity scale s is not a free parameter: it is set so the *expected*
// rms speed (an average over time and over random plume positions) equals
// `rmsSpeedCmYr`. Plumes are born at rate r and their buoyancy decays as
// exp(-t/tau), so the expected mean-square gradient of Phi is
// sum over plume kinds of r (tau/2) E / (4 pi), with E the integral of
// |grad Phi|^2 for one unit plume (one screened solve each, at start-up).
// Advection reshapes plumes, so realised speeds differ somewhat (see
// design.md, mantle findings); they also fluctuate by ~2x over time.
//
// The model runs alone for `spinUpMyr` before the crust starts, so the crust
// never sees the artificial initial state. Crust -> mantle feedback (thermal
// blanketing under continents, slabs as cold sinks) is not coupled yet.

import { mulberry32, randomOnSphere } from '../init/random.js';
import { SphereMesh } from './sphere-mesh.js';

export const MANTLE_DEFAULTS = {
  subdivisions: 4,          // 2562 nodes, spacing ~0.075 rad (~480 km)
  rmsSpeedCmYr: 4,          // target rms flow speed beneath the plates (Earth's rms plate speed is ~4 cm/yr)
  stepMyr: 1,               // mantle time step
  spinUpMyr: 400,           // run alone before the crust starts (~3 plume lifetimes)
  spinUpStepMyr: 2,
  plumeLifetimeMyr: 120,    // decay time of buoyancy anomalies
  hotPlumesPerMyr: 0.3,     // births; with the lifetime, ~35 active plumes (Earth: ~40 hotspots)
  coldDripsPerMyr: 0.3,     // cold downwellings (stand-ins for slabs until the crust feeds them)
  plumeRadiusRad: 0.12,     // Gaussian radius of a new plume (~760 km)
  dripRadiusRad: 0.08,      // cold drips are narrower
  plumeStrength: 1,         // buoyancy amplitude of a new plume or drip (sets units of b only)
  diffusivity: 2e-4,        // rad^2/Myr; effective smoothing (~900 km over 100 Myr), not molecular
  screeningKm: 2900,        // mantle depth: range over which a buoyancy anomaly drives flow
};

const CM_YR_TO_RAD_MYR = 10 / 6371; // 1 cm/yr = 10 km/Myr

// Measured once (design.md, mantle findings): with the defaults, advection
// sharpens downwellings and the realised long-run rms speed is 1.69 times the
// estimate from plume statistics alone (6 seeds x 400 Myr: 6.75 vs 4 cm/yr).
// Re-measure if the plume parameters change much.
const ADVECTION_GAIN = 1.69;

export class Mantle {
  /** `mesh`: an existing SphereMesh of options.subdivisions to share (building one takes ~0.1 s). */
  constructor(seed, options = {}, mesh = null) {
    this.options = { ...MANTLE_DEFAULTS, ...options };
    this.mesh = mesh ?? new SphereMesh(this.options.subdivisions);
    if (this.mesh.n !== 10 * 4 ** this.options.subdivisions + 2) throw new Error('mantle mesh has the wrong subdivision level');
    const n = this.mesh.n;
    this.rng = mulberry32(seed ^ 0x5eed);
    this.b = new Float64Array(n);           // buoyancy
    this.phi = new Float64Array(n);         // flow potential (warm start for the solver)
    this.velocity = new Float64Array(3 * n); // rad/Myr, tangent (xyz interleaved)
    this.vComp = [new Float64Array(n), new Float64Array(n), new Float64Array(n)]; // same, by component
    this.divergence = new Float64Array(n);  // 1/Myr
    this.scale = this.#expectedScale();     // s
    this.time = 0;                          // Myr since the crust started (negative during spin-up)
    this.births = { hot: 0, cold: 0 };
  }

  /** Run alone from a quiet state for options.spinUpMyr, ending at time 0. */
  spinUp() {
    const { spinUpMyr, spinUpStepMyr } = this.options;
    this.time = -spinUpMyr;
    while (this.time < -1e-9) this.step(Math.min(spinUpStepMyr, -this.time));
    this.#updateFlow();
  }

  /** Velocity scale giving the expected rms speed (see the header comment). */
  #expectedScale() {
    const o = this.options, mesh = this.mesh, n = mesh.n;
    const energy = (radius) => {
      const b = new Float64Array(n), phi = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const ang = Math.acos(Math.min(1, mesh.pos[3 * i + 2])); // plume at the pole
        b[i] = o.plumeStrength * Math.exp(-(ang * ang) / (2 * radius * radius));
      }
      mesh.solveScreened(b, phi, o.screeningKm / 6371, 1e-10);
      const g = mesh.gradient(phi);
      let e = 0;
      for (let i = 0; i < n; i++) e += (g[3 * i] ** 2 + g[3 * i + 1] ** 2 + g[3 * i + 2] ** 2) * mesh.area[i];
      return e;
    };
    const meanSquare = (o.plumeLifetimeMyr / 2) * (o.hotPlumesPerMyr * energy(o.plumeRadiusRad) + o.coldDripsPerMyr * energy(o.dripRadiusRad)) / (4 * Math.PI);
    return (o.rmsSpeedCmYr * CM_YR_TO_RAD_MYR) / (ADVECTION_GAIN * Math.sqrt(meanSquare));
  }

  /** Advance to time t (Myr) in steps of options.stepMyr; returns true if anything changed. */
  advanceTo(t) {
    let changed = false;
    while (this.time + this.options.stepMyr <= t + 1e-9) {
      this.step(this.options.stepMyr);
      changed = true;
    }
    return changed;
  }

  step(dt) {
    const o = this.options, mesh = this.mesh, n = mesh.n;
    this.#updateFlow();

    // Advection, semi-Lagrangian with a midpoint departure estimate.
    const old = this.b, next = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const x = mesh.point(i);
      const v = [this.velocity[3 * i], this.velocity[3 * i + 1], this.velocity[3 * i + 2]];
      const mid = unit([x[0] - 0.5 * dt * v[0], x[1] - 0.5 * dt * v[1], x[2] - 0.5 * dt * v[2]]);
      const atMid = mesh.locate(mid, i);
      const vm = this.vComp.map((c) => mesh.interpolate(c, atMid));
      const departure = unit([x[0] - dt * vm[0], x[1] - dt * vm[1], x[2] - dt * vm[2]]);
      next[i] = mesh.interpolate(old, departure, atMid.nodes[0]);
    }

    // Diffusion (explicit, substepped for stability) and decay.
    const lambdaMax = 8 / (0.6 * mesh.area.reduce((a, v) => Math.min(a, v), Infinity)); // generous bound on |L|
    const sub = Math.max(1, Math.ceil((o.diffusivity * dt * lambdaMax) / 1.5));
    const lap = new Float64Array(n);
    for (let k = 0; k < sub; k++) {
      mesh.laplacian(next, lap);
      for (let i = 0; i < n; i++) next[i] += (o.diffusivity * dt / sub) * lap[i];
    }
    const decay = Math.exp(-dt / o.plumeLifetimeMyr);
    for (let i = 0; i < n; i++) next[i] *= decay;

    // Births: Poisson numbers of hot plumes and cold drips in this step.
    for (const [kind, rate, radius, sign] of [['hot', o.hotPlumesPerMyr, o.plumeRadiusRad, 1], ['cold', o.coldDripsPerMyr, o.dripRadiusRad, -1]]) {
      for (let k = poisson(rate * dt, this.rng); k > 0; k--) {
        const c = randomOnSphere(this.rng);
        for (let i = 0; i < n; i++) {
          const x = mesh.point(i);
          const ang = Math.acos(Math.min(1, x[0] * c[0] + x[1] * c[1] + x[2] * c[2]));
          next[i] += sign * o.plumeStrength * Math.exp(-(ang * ang) / (2 * radius * radius));
        }
        this.births[kind]++;
      }
    }
    const m = mesh.mean(next);
    for (let i = 0; i < n; i++) next[i] -= m;
    this.b = next;
    this.time += dt;
  }

  #updateFlow() {
    const mesh = this.mesh;
    mesh.solveScreened(this.b, this.phi, this.options.screeningKm / 6371, 1e-6);
    const g = mesh.gradient(this.phi);
    const lapPhi = mesh.laplacian(this.phi);
    for (let i = 0; i < mesh.n; i++) {
      for (let d = 0; d < 3; d++) this.velocity[3 * i + d] = this.vComp[d][i] = this.scale * g[3 * i + d];
      this.divergence[i] = this.scale * lapPhi[i]; // div v = s lap(Phi)
    }
  }

  /** Area-weighted rms speed, cm/yr. */
  rmsSpeedCmYr() {
    let s2 = 0, a = 0;
    for (let i = 0; i < this.mesh.n; i++) {
      const v = this.velocity;
      s2 += (v[3 * i] ** 2 + v[3 * i + 1] ** 2 + v[3 * i + 2] ** 2) * this.mesh.area[i];
      a += this.mesh.area[i];
    }
    return Math.sqrt(s2 / a) / CM_YR_TO_RAD_MYR;
  }

  /** Velocity (rad/Myr) and buoyancy at an arbitrary unit vector, by linear interpolation. */
  sample(x, hint = 0) {
    const node = this.mesh.nearest(x, hint);
    return { velocity: this.vComp.map((c) => this.mesh.interpolate(c, x, node)), buoyancy: this.mesh.interpolate(this.b, x, node), node };
  }

  /** Everything needed to resume (for snapshots). */
  getState() {
    return { options: this.options, b: Array.from(this.b), phi: Array.from(this.phi), scale: this.scale, time: this.time, rng: this.rng.state, births: { ...this.births } };
  }

  setState(state) {
    Object.assign(this.options, state.options);
    this.b = Float64Array.from(state.b);
    this.phi = Float64Array.from(state.phi);
    this.scale = state.scale;
    this.time = state.time;
    this.rng.state = state.rng;
    this.births = { ...state.births };
    this.#updateFlow();
  }
}

function unit(v) {
  const l = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / l, v[1] / l, v[2] / l];
}

/** Poisson-distributed count with mean `mean` (Knuth; fine for small means). */
function poisson(mean, rng) {
  const L = Math.exp(-mean);
  let k = 0, p = 1;
  do { k++; p *= rng(); } while (p > L);
  return k - 1;
}
