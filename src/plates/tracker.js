// Keeps plate labels up to date during a run: relabels from a readback every
// `intervalMyr`, uploads the ids for drawing, and keeps a plate time series
// and an event log for the record.
//
// Euler poles are fitted to the displacement since the previous labelling
// (a finite rotation, as in plate kinematics), not to the one-step velocity:
// one step moves a plate ~160 m at 1 cm/yr, comparable to the solver's
// per-step tolerance, which would otherwise dominate the fit.

import { Edge, Particle, unpack } from '../gpu/layout.js';
import { CAPACITY } from '../sim/simulation.js';
import { latLonDeg } from '../record/sampling.js';
import { eulerPole, labelPlates } from './plates.js';

export class PlateTracker {
  constructor(sim, { intervalMyr = 0.25 } = {}) {
    this.sim = sim;
    this.intervalMyr = intervalMyr;
    this.reset();
  }

  reset() {
    this.state = null;   // last labelPlates result (ids, nextId, plates)
    this.summary = [];   // per plate: {id, size, pole, rateDegPerMyr, rigidityResidual}
    this.series = [];    // [{step, time_myr, plates: summary}]
    this.events = [];    // [{step, time_myr, type, ...}]
    this.lastStep = -1;
    this.lastPositions = null; // Float32Array (vec4 per slot) at the previous labelling
    this.lastTime = 0;
  }

  due() {
    const every = Math.max(1, Math.round(this.intervalMyr / this.sim.params.dt_crust_myr));
    return this.sim.steps % every === 0 && this.lastStep !== this.sim.steps;
  }

  /** Relabel from a snapshot's buffers (see Simulation.snapshot). */
  update(snap) {
    const P = unpack(Particle, snap.buffers.particles, CAPACITY.maxParticles);
    const C = unpack(Edge, snap.buffers.edges, snap.counts?.edges ?? CAPACITY.maxEdges); // (edge slots in use)
    const result = labelPlates(P, C, this.state);
    // The first labelling of a run is not a sequence of splits.
    if (this.state) for (const e of result.events) this.events.push({ step: snap.step, time_myr: snap.timeMyr, ...e });
    this.state = result;
    // Mean velocity since the last labelling, for particles present at both times.
    const dt = snap.timeMyr - this.lastTime;
    const motion = { position: P.position, velocity: new Float32Array(P.position.length) };
    const moved = (i) => this.lastPositions && dt > 0 && this.lastActive[i];
    for (let i = 0; i < CAPACITY.maxParticles; i++) {
      if (!moved(i)) continue;
      for (let c = 0; c < 3; c++) motion.velocity[4 * i + c] = (P.position[4 * i + c] - this.lastPositions[4 * i + c]) / dt;
    }
    this.summary = result.plates.map(({ id, size, members }) => {
      const fit = eulerPole(motion, members.filter(moved));
      const [lat, lon] = latLonDeg(...fit.pole);
      const mmPerYr = 6371; // rad/Myr on the unit sphere -> km/Myr = mm/yr
      return {
        id, size, pole_lat_deg: lat, pole_lon_deg: lon, rate_deg_per_myr: fit.rateDegPerMyr,
        rigidity_residual: fit.rigidityResidual,
        rms_speed_mm_per_yr: fit.rmsSpeed * mmPerYr,
        rms_nonrigid_speed_mm_per_yr: fit.rmsNonRigidSpeed * mmPerYr,
      };
    });
    this.series.push({ step: snap.step, time_myr: snap.timeMyr, plates: this.summary });
    this.lastPositions = P.position;
    this.lastActive = P.is_active;
    this.lastTime = snap.timeMyr;
    this.lastStep = snap.step;
    this.sim.setPlateIds(result.ids);
  }

  /** Size-weighted median of a summary column over plates of >= 100 particles. */
  median(column) {
    const big = this.summary.filter((p) => p.size >= 100).sort((a, b) => a[column] - b[column]);
    const total = big.reduce((s, p) => s + p.size, 0);
    let acc = 0;
    for (const p of big) { acc += p.size; if (acc >= total / 2) return p[column]; }
    return NaN;
  }
}
