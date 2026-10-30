// Records a run (design.md section 8.3, tiers a-c):
//   a. manifest.json       how to interpret and regenerate everything below
//   b. field series        particle quantities sampled on a fixed icosphere every fieldIntervalMyr
//   c. snapshots           complete raw GPU state every snapshotIntervalMyr (restartable)
// exportZip() bundles them; snapshotFromZip() reads a snapshot back for Simulation.restore.

import { createIcosphere } from '../geometry/icosphere.js';
import { Constraint, describe, GridNode, Particle, unpack } from '../gpu/layout.js';
import { CAPACITY, STATE_BUFFERS } from '../sim/simulation.js';
import { BUILD_INFO } from '../build-info.js';
import { eastNorth, latLonDeg, sampleToMesh } from './sampling.js';
import { readZip, writeZip } from './zip.js';

export const SCHEMA = { name: 'plate-player-record', version: '0.1.0' };
const EARTH_RADIUS_KM = 6371;

/** Field series written for every sample. Order here is order in the manifest. */
export const FIELDS = [
  { name: 'surface_height', units: 'km', long_name: 'surface height relative to sea level', kind: 'intensive' },
  { name: 'crust_thickness', units: 'km', long_name: 'crust thickness', kind: 'intensive', note: 'area-weighted average; not yet conservatively remapped' },
  { name: 'crust_density', units: 'kg m-3', long_name: 'crust composition density at mantle temperature', kind: 'intensive' },
  { name: 'crust_age', units: 'Myr', long_name: 'time since crust formed', kind: 'intensive' },
  { name: 'temperature', units: 'K', long_name: 'crust temperature', kind: 'intensive' },
  { name: 'velocity_east', units: 'mm yr-1', long_name: 'eastward surface velocity (1 mm/yr = 1 km/Myr)', kind: 'intensive' },
  { name: 'velocity_north', units: 'mm yr-1', long_name: 'northward surface velocity', kind: 'intensive' },
  { name: 'bond_damage', units: '1', long_name: 'mean damage of bonds attached to the crust', kind: 'intensive' },
  { name: 'slab_fraction', units: '1', long_name: 'fraction of nearby crust that is subducting slab', kind: 'intensive' },
  { name: 'support', units: '1', long_name: 'crust coverage: ~1 under continuous crust, 0 in gaps', kind: 'diagnostic' },
];

export const RECORD_DEFAULTS = {
  meshSubdivisions: 5,      // 10242 nodes, spacing ~0.035 rad (~220 km), matching base particles
  fieldIntervalMyr: 1,
  snapshotIntervalMyr: 25,
};

export class Recorder {
  constructor(sim, options = {}) {
    this.sim = sim;
    this.options = { ...RECORD_DEFAULTS, ...options };
    const ico = createIcosphere(this.options.meshSubdivisions);
    this.mesh = {
      positions: Float32Array.from(ico.vertices.flat()),
      triangles: ico.indices,
      latLon: Float32Array.from(ico.vertices.flatMap((v) => latLonDeg(v[0], v[1], v[2]))),
      frames: ico.vertices.map((v) => eastNorth(v[0], v[1], v[2])),
    };
    this.reset({});
  }

  /** Start a new record. `run` describes how the initial state was made (seed, init options, ...). */
  reset(run) {
    this.run = run;
    this.created = new Date().toISOString();
    this.samples = [];   // {step, timeMyr, fields: {name: Float32Array}}
    this.snapshots = []; // {step, timeMyr, overrides, buffers}
  }

  get sizeBytes() {
    const fieldBytes = this.samples.length * FIELDS.length * (this.mesh.positions.length / 3) * 4;
    const snapBytes = this.snapshots.reduce((s, snap) => s + STATE_BUFFERS.reduce((t, n) => t + snap.buffers[n].byteLength, 0), 0);
    return fieldBytes + snapBytes;
  }

  #every(intervalMyr) {
    return intervalMyr > 0 ? Math.max(1, Math.round(intervalMyr / this.sim.params.dt_crust_myr)) : Infinity;
  }

  /** Which tiers are due at the simulation's current step. */
  due() {
    const s = this.sim.steps;
    const last = (list) => list.at(-1)?.step;
    return {
      fields: s % this.#every(this.options.fieldIntervalMyr) === 0 && last(this.samples) !== s,
      snapshot: s % this.#every(this.options.snapshotIntervalMyr) === 0 && last(this.snapshots) !== s,
    };
  }

  /** Read back GPU state and store the requested tiers. */
  async capture({ fields = true, snapshot = false }) {
    if (!fields && !snapshot) return;
    const snap = await this.sim.snapshot(); // one readback serves both tiers
    if (snapshot) this.snapshots.push(snap);
    if (fields) this.samples.push({ step: snap.step, timeMyr: snap.timeMyr, fields: this.#sampleFields(snap.buffers) });
  }

  #sampleFields(buffers) {
    const P = unpack(Particle, buffers.particles, CAPACITY.maxParticles);
    const C = unpack(Constraint, buffers.constraints, CAPACITY.maxConstraints);
    const params = this.sim.params;

    // Mean damage of active bonds at each particle.
    const damageSum = new Float32Array(CAPACITY.maxParticles);
    const bondCount = new Uint16Array(CAPACITY.maxParticles);
    for (let k = 0; k < CAPACITY.maxConstraints; k++) {
      if (!C.is_active[k]) continue;
      for (const p of [C.particle_a[k], C.particle_b[k]]) { damageSum[p] += C.damage[k]; bondCount[p]++; }
    }

    const active = [];
    for (let i = 0; i < CAPACITY.maxParticles; i++) if (P.is_active[i]) active.push(i);
    const n = active.length;
    const position = new Float32Array(3 * n);
    const area = new Float32Array(n);
    const ch = Object.fromEntries(FIELDS.filter((f) => f.name !== 'support').map((f) => [f.name, new Float32Array(n)]));
    active.forEach((i, k) => {
      const x = P.position[4 * i], y = P.position[4 * i + 1], z = P.position[4 * i + 2];
      position.set([x, y, z], 3 * k);
      area[k] = P.surface_area[i];
      const surface = P.state[i] === 1;
      const nan = (v) => (surface ? v : NaN); // slabs only contribute to slab_fraction
      const [east, north] = eastNorth(x, y, z);
      const vx = P.velocity[4 * i], vy = P.velocity[4 * i + 1], vz = P.velocity[4 * i + 2];
      ch.surface_height[k] = nan(P.elevation[i] - params.sea_level);
      ch.crust_thickness[k] = nan(P.thickness_km[i]);
      ch.crust_density[k] = nan(P.composition_density[i]);
      ch.crust_age[k] = nan(P.age[i]);
      ch.temperature[k] = nan(P.temperature[i]);
      ch.velocity_east[k] = nan((vx * east[0] + vy * east[1] + vz * east[2]) * EARTH_RADIUS_KM);
      ch.velocity_north[k] = nan((vx * north[0] + vy * north[1] + vz * north[2]) * EARTH_RADIUS_KM);
      ch.bond_damage[k] = surface && bondCount[i] > 0 ? damageSum[i] / bondCount[i] : NaN;
      ch.slab_fraction[k] = surface ? 0 : 1;
    });
    // Kernel width = one base particle spacing (~ the record mesh spacing).
    return sampleToMesh(this.mesh.positions, { position, area }, ch, 2 * params.base_apothem);
  }

  manifest() {
    const nNodes = this.mesh.positions.length / 3;
    const T = this.samples.length;
    const { sim } = this;
    return {
      schema: SCHEMA,
      model: { name: 'plate-player', level: 'L1 global plates', build: BUILD_INFO },
      created: this.created,
      exported: new Date().toISOString(),
      run: {
        ...this.run,
        sim_params: sim.params,
        param_overrides: sim.overrides,
        capacity: CAPACITY,
        mantle: { subdivisions_nodes: sim.numNodes, substeps_per_step: sim.mantleSubsteps, stability: sim.mantleStability },
      },
      conventions: {
        position: 'unit sphere, radius 1 = 6371 km, body-fixed',
        geographic: 'north pole = +y; longitude 0 toward +z; east toward +x; degrees',
        time: 'Myr since the start of the run (t_model)',
        heights: 'km relative to sea level; sea level is sim_params.sea_level in the particle elevation datum',
        byte_order: 'little-endian',
        missing: 'NaN where no crust particle lies within the sampling cutoff',
      },
      nondeterminism: [
        'Which free slot a splitting particle receives depends on GPU thread timing; physics is unaffected but indices differ between runs after the first split.',
        'Collision neighbour lists are capped at 50 entries per hash cell, and list order depends on thread timing; the cap is rarely reached.',
      ],
      mesh: {
        type: 'icosphere',
        subdivisions: this.options.meshSubdivisions,
        nodes: nNodes,
        files: {
          positions: { file: 'mesh/positions.f32', dtype: 'float32', shape: [nNodes, 3] },
          lat_lon: { file: 'mesh/lat_lon_deg.f32', dtype: 'float32', shape: [nNodes, 2] },
          triangles: { file: 'mesh/triangles.u32', dtype: 'uint32', shape: [this.mesh.triangles.length / 3, 3] },
        },
      },
      fields: {
        interval_myr: this.options.fieldIntervalMyr,
        sampling: {
          method: 'gaussian kernel average over particles, weight = area * exp(-d^2 / 2 sigma^2), cutoff 3 sigma',
          sigma_unit_sphere: 2 * sim.params.base_apothem,
        },
        time: { file: 'fields/time_myr.f64', dtype: 'float64', shape: [T] },
        step: { file: 'fields/step.u32', dtype: 'uint32', shape: [T] },
        variables: FIELDS.map((f) => ({ ...f, file: `fields/${f.name}.f32`, dtype: 'float32', shape: [T, nNodes] })),
      },
      snapshots: this.snapshots.map((s) => ({
        step: s.step,
        time_myr: s.timeMyr,
        param_overrides: s.overrides,
        buffers: Object.fromEntries(STATE_BUFFERS.map((b) => [b, { file: `${snapshotDir(s.step)}/${b}.bin`, bytes: s.buffers[b].byteLength }])),
      })),
      structs: {
        Particle: describe(Particle),
        Constraint: describe(Constraint),
        GridNode: describe(GridNode),
        FreeList: { layout: 'u32 counter, u32 requests, 2 x u32 padding, then u32 indices[capacity]; entries [0, counter) are free slots' },
      },
      buffer_structs: { grid: 'GridNode', particles: 'Particle', constraints: 'Constraint', particlePool: 'FreeList', constraintPool: 'FreeList' },
    };
  }

  exportZip() {
    const entries = [{ name: 'manifest.json', data: JSON.stringify(this.manifest(), null, 2) }];
    entries.push({ name: 'mesh/positions.f32', data: this.mesh.positions });
    entries.push({ name: 'mesh/lat_lon_deg.f32', data: this.mesh.latLon });
    entries.push({ name: 'mesh/triangles.u32', data: this.mesh.triangles });
    entries.push({ name: 'fields/time_myr.f64', data: Float64Array.from(this.samples, (s) => s.timeMyr) });
    entries.push({ name: 'fields/step.u32', data: Uint32Array.from(this.samples, (s) => s.step) });
    const nNodes = this.mesh.positions.length / 3;
    for (const f of FIELDS) {
      const series = new Float32Array(this.samples.length * nNodes);
      this.samples.forEach((s, t) => series.set(s.fields[f.name], t * nNodes));
      entries.push({ name: `fields/${f.name}.f32`, data: series });
    }
    for (const s of this.snapshots) {
      for (const b of STATE_BUFFERS) entries.push({ name: `${snapshotDir(s.step)}/${b}.bin`, data: s.buffers[b] });
    }
    return writeZip(entries);
  }
}

function snapshotDir(step) {
  return `snapshots/step_${String(step).padStart(9, '0')}`;
}

/**
 * Read a snapshot from a record ZIP, checking that its memory layouts match
 * this build. `which` is 'latest' or a step number. Returns {snapshot, manifest}.
 */
export function snapshotFromZip(bytes, which = 'latest') {
  const files = readZip(bytes);
  const manifestBytes = files.get('manifest.json');
  if (!manifestBytes) throw new Error('not a plate-player record (no manifest.json)');
  const manifest = JSON.parse(new TextDecoder().decode(manifestBytes));
  if (manifest.schema?.name !== SCHEMA.name) throw new Error('not a plate-player record');
  if (manifest.schema.version.split('.')[0] !== SCHEMA.version.split('.')[0]) {
    throw new Error(`record schema ${manifest.schema.version} is incompatible with ${SCHEMA.version}`);
  }
  for (const [name, struct] of [['Particle', Particle], ['Constraint', Constraint], ['GridNode', GridNode]]) {
    if (JSON.stringify(manifest.structs[name]) !== JSON.stringify(describe(struct))) {
      throw new Error(`record's ${name} layout differs from this build; it was written by build ${manifest.model.build?.commit ?? '?'}`);
    }
  }
  const list = manifest.snapshots;
  if (!list.length) throw new Error('record contains no snapshots');
  const entry = which === 'latest' ? list.at(-1) : list.find((s) => s.step === which);
  if (!entry) throw new Error(`no snapshot at step ${which}`);
  const buffers = {};
  for (const [name, { file }] of Object.entries(entry.buffers)) {
    const view = files.get(file);
    if (!view) throw new Error(`record is missing ${file}`);
    buffers[name] = view.slice().buffer; // own, aligned copy
  }
  return {
    manifest,
    snapshot: { step: entry.step, timeMyr: entry.time_myr, overrides: entry.param_overrides, buffers },
  };
}
