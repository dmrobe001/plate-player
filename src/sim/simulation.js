// GPU-side simulation state and the per-frame compute schedule.

import { ColumnProfile, Edge, GridNode, NONE, Particle, SLOTS } from '../gpu/layout.js';
import { createShaderModule } from '../gpu/device.js';
import { packSimParams, SIM_PARAMS, SIM_PARAMS_BYTES, simParamsWGSL } from './params.js';

import bindingsWGSL from '../shaders/compute/bindings.wgsl';
import locateWGSL from '../shaders/compute/locate.wgsl';
import thermalWGSL from '../shaders/compute/thermal.wgsl';
import growthWGSL from '../shaders/compute/growth.wgsl';
import solverWGSL from '../shaders/compute/solver.wgsl';
import edgesWGSL from '../shaders/compute/edges.wgsl';
import profileWGSL from '../shaders/compute/profile.wgsl';
import lawWGSL from '../shaders/compute/law.wgsl';
import flowWGSL from '../shaders/compute/flow.wgsl';

export const CAPACITY = {
  maxParticles: 25000,
  maxEdges: 131072,      // edge slots, managed by the edge free list
  hashGridDim: 32,       // neighbour-search grid cells per axis over [-1.5, 1.5]
  adjacencyWidth: 6,     // neighbours stored per mantle node
};

const EARTH_RADIUS_M = 6.371e6;
const SECONDS_PER_MYR = 3.15576e13;
const FREE_LIST_HEADER_WORDS = 4;
const MAX_SOLVER_ITERATIONS = 1024; // size of the Chebyshev weight table
const WORKGROUP = 64;
// Solver iterations per GPU submission. Long stretches of GPU work can exceed
// the browser's GPU watchdog on slow GPUs (the device is then lost), so a
// step with many iterations is submitted in pieces, waiting for each.
const ITERATIONS_PER_SUBMIT = 16;

/** Buffers that together are the complete simulation state (snapshot / restore). */
// (Slot lists in `deltas` are an index rebuilt every step, not state.)
export const STATE_BUFFERS = ['grid', 'particles', 'edges', 'particlePool', 'edgePool'];

/** WGSL struct definitions shared by compute and render shaders. */
export function structsWGSL() {
  return Particle.wgsl() + Edge.wgsl() + ColumnProfile.wgsl()
    + GridNode.wgsl()
    + simParamsWGSL();
}

export class Simulation {
  /** `mantleMesh`: the SphereMesh shared with the Mantle model (src/mantle/). */
  constructor(device, mantleMesh, onError) {
    this.device = device;
    this.mantleMesh = mantleMesh;
    this.icosphere = mantleMesh.ico;
    const icosphere = this.icosphere;
    this.mantle = null;
    this.numNodes = icosphere.vertices.length;
    this.steps = 0;
    this.timeMyr = 0;
    const { maxParticles, maxEdges, hashGridDim, adjacencyWidth } = CAPACITY;
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

    this.buffers = {
      grid: device.createBuffer({ label: 'grid', size: this.numNodes * GridNode.byteSize, usage: S }),
      adjacency: device.createBuffer({ label: 'adjacency', size: this.numNodes * adjacencyWidth * 4, usage: S }),
      particles: device.createBuffer({ label: 'particles', size: maxParticles * Particle.byteSize, usage: S }),
      edges: device.createBuffer({ label: 'edges', size: maxEdges * Edge.byteSize, usage: S }),
      particlePool: device.createBuffer({ label: 'particle free list', size: (FREE_LIST_HEADER_WORDS + maxParticles) * 4, usage: S }),
      edgePool: device.createBuffer({ label: 'edge free list', size: (FREE_LIST_HEADER_WORDS + maxEdges) * 4, usage: S }),
      hashGrid: device.createBuffer({ label: 'hash grid', size: hashGridDim ** 3 * 4, usage: S }),
      // ParticleDelta in bindings.wgsl: position corrections + slot list (SLOTS in layout.js)
      deltas: device.createBuffer({ label: 'particle deltas and slot lists', size: maxParticles * SLOTS.wordsPerParticle * 4, usage: S }),
      solver: device.createBuffer({ label: 'solver state', size: (4 + MAX_SOLVER_ITERATIONS) * 4, usage: S }),
      profiles: device.createBuffer({ label: 'column profiles (derived)', size: maxParticles * ColumnProfile.byteSize, usage: S }),
      plateIds: device.createBuffer({ label: 'plate ids (drawing only)', size: maxParticles * 4, usage: S }),
      params: device.createBuffer({ label: 'sim params', size: SIM_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    };

    const adjacency = new Uint32Array(this.numNodes * adjacencyWidth).fill(NONE);
    icosphere.neighbours.forEach((list, i) => adjacency.set(list.slice(0, adjacencyWidth), i * adjacencyWidth));
    device.queue.writeBuffer(this.buffers.adjacency, 0, adjacency);

    this.overrides = {};
    this.setParams({});
    this.#createPipelines(onError);
  }

  /** Set parameter overrides (merged with previous ones), recompute derived values, upload. */
  setParams(overrides) {
    Object.assign(this.overrides, overrides);
    const value = (name) => (name in this.overrides ? this.overrides[name] : SIM_PARAMS.find(([n]) => n === name)[2]);

    this.params = {};
    for (const [name] of SIM_PARAMS) this.params[name] = value(name);
    Object.assign(this.params, {
      bond_rate_myr_inv: bondRate(this.params),
      num_nodes: this.numNodes,
      max_particles: CAPACITY.maxParticles,
      max_edges: CAPACITY.maxEdges,
      hash_grid_dim: CAPACITY.hashGridDim,
    });
    if (this.params.solver_iterations > MAX_SOLVER_ITERATIONS) throw new Error(`solver_iterations > ${MAX_SOLVER_ITERATIONS}`);
    this.device.queue.writeBuffer(this.buffers.params, 0, packSimParams(this.params));
    const rho = this.params.chebyshev_rho < 0 ? jacobiSpectralRadius(this.params) : this.params.chebyshev_rho;
    this.chebyshevRho = rho;
    const omega = chebyshevWeights(rho, this.params.chebyshev_delay, MAX_SOLVER_ITERATIONS);
    this.device.queue.writeBuffer(this.buffers.solver, 16, omega);
  }

  #createPipelines(onError) {
    const { device, buffers } = this;
    const code = [structsWGSL(), bindingsWGSL, locateWGSL, thermalWGSL, growthWGSL, solverWGSL, edgesWGSL, profileWGSL, lawWGSL, flowWGSL].join('\n');
    const module = createShaderModule(device, 'compute', code, onError);

    const order = ['grid', 'adjacency', 'particles', 'edges', 'particlePool', 'edgePool', 'hashGrid', 'deltas', 'solver', 'profiles'];
    const memoryLayout = device.createBindGroupLayout({
      entries: order.map((name, binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: name === 'adjacency' ? 'read-only-storage' : 'storage' },
      })),
    });
    // Group 1 mirrors the render uniform group: binding 0 is the view (unused
    // here), binding 1 the sim params. Only binding 1 is declared for compute.
    const uniformLayout = device.createBindGroupLayout({
      entries: [{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }],
    });
    this.memoryBindGroup = device.createBindGroup({
      layout: memoryLayout,
      entries: order.map((name, binding) => ({ binding, resource: { buffer: buffers[name] } })),
    });
    this.uniformBindGroup = device.createBindGroup({
      layout: uniformLayout,
      entries: [{ binding: 1, resource: { buffer: buffers.params } }],
    });

    const layout = device.createPipelineLayout({ bindGroupLayouts: [memoryLayout, uniformLayout] });
    const entryPoints = [
      'crust_locate', 'column_profile',
      'crust_thermo_isostasy', 'crust_refresh_derived', 'columns_flow', 'columns_gpe_force',
      'crust_accumulation', 'crust_commit_splits', 'crust_rewire_edges', 'crust_clean_split',
      'crust_predict', 'crust_hash_clear', 'crust_hash_build',
      'edges_begin', 'edges_prune', 'edges_note_listed', 'edges_create', 'edges_finish', 'edges_law', 'edges_maxwell',
      'solver_begin', 'solver_next_iteration', 'crust_chebyshev_combine',
      'crust_clear_deltas', 'crust_solve_edges', 'crust_apply_deltas',
      'crust_integrate',
    ];
    this.pipelines = Object.fromEntries(entryPoints.map((entryPoint) => [
      entryPoint,
      device.createComputePipeline({ label: entryPoint, layout, compute: { module, entryPoint } }),
    ]));
  }

  /**
   * Upload an initial crust state (see src/init/initial-state.js), reset the
   * free lists, and attach a spun-up Mantle (src/mantle/mantle.js).
   */
  load(state, mantle) {
    const { device, buffers } = this;
    const { maxParticles, maxEdges } = CAPACITY;
    const { particles: P, bonds: B } = state;
    if (P.count > maxParticles) throw new Error(`initial particles ${P.count} exceed capacity ${maxParticles}`);
    if (B.count > maxEdges) throw new Error(`initial bonds ${B.count} exceed edge capacity ${maxEdges}`);
    if (mantle.mesh !== this.mantleMesh) throw new Error('mantle must use the simulation\'s mantle mesh');
    this.mantle = mantle;
    this.uploadMantle();

    // Particles
    const pbuf = new ArrayBuffer(maxParticles * Particle.byteSize);
    const pf = new Float32Array(pbuf);
    const pu = new Uint32Array(pbuf);
    const po = Particle.offsets;
    for (let i = 0; i < P.count; i++) {
      const base = i * Particle.words;
      const x = P.position[3 * i], y = P.position[3 * i + 1], z = P.position[3 * i + 2];
      pf.set([x, y, z, 1], base + po.position);
      pf.set([x, y, z, 1], base + po.predicted_pos);
      pf[base + po.thickness_km] = P.thickness[i];
      pf[base + po.composition_density] = P.compositionDensity[i];
      pf[base + po.temperature] = P.temperature[i];
      pf[base + po.age] = P.age[i];
      pf[base + po.surface_area] = 2 * Math.sqrt(3) * P.apothem[i] ** 2;
      pf[base + po.apothem] = P.apothem[i];
      pu[base + po.state] = 1;
      pu[base + po.next_state] = 1;
      pu[base + po.is_active] = 1;
      pu[base + po.next_idx] = NONE;
      pf[base + po.density] = P.compositionDensity[i];
      pf[base + po.elevation] = P.elevation ? P.elevation[i] : NaN; // NaN: start at the Airy height
      pu[base + po.split_child] = NONE;
      pu[base + po.mantle_node] = nearestVertex(this.icosphere.vertices, x, y, z);
    }
    device.queue.writeBuffer(buffers.particles, 0, pbuf);

    // Initial bonds become edges with integrity 1 - damage. Other in-range
    // pairs get edges (integrity 0) in the first step.
    const ebuf = new ArrayBuffer(maxEdges * Edge.byteSize);
    const ef = new Float32Array(ebuf);
    const eu = new Uint32Array(ebuf);
    const eo = Edge.offsets;
    for (let k = 0; k < B.count; k++) {
      const base = k * Edge.words;
      eu[base + eo.particle_a] = B.a[k];
      eu[base + eo.particle_b] = B.b[k];
      eu[base + eo.is_active] = 1;
      ef[base + eo.integrity] = 1 - B.damage[k];
      // Elastic loading from the rest length (apothem sum), in every slice.
      const i = B.a[k], j = B.b[k];
      const d = Math.hypot(...[0, 1, 2].map((c) => P.position[3 * i + c] - P.position[3 * j + c]));
      const rest = P.apothem[i] + P.apothem[j];
      ef[base + eo.length] = d;
      ef[base + eo.strain] = (d - rest) / rest;
      ef.fill(this.params.crust_youngs_modulus_gpa * 1000 * ((d - rest) / rest), base + eo.slice_stress_mpa, base + eo.slice_stress_mpa + 8);
    }
    device.queue.writeBuffer(buffers.edges, 0, ebuf);

    // Free lists hold every slot not used by the initial state.
    device.queue.writeBuffer(buffers.particlePool, 0, freeList(P.count, maxParticles));
    device.queue.writeBuffer(buffers.edgePool, 0, freeList(B.count, maxEdges));
    device.queue.writeBuffer(buffers.solver, 0, new Uint32Array(4)); // iteration and overflow counters
    this.steps = 0;
    this.timeMyr = 0;

    // Density and elevation are derived; compute them now so step 0 is complete.
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass({ label: 'refresh derived' });
    pass.setBindGroup(0, this.memoryBindGroup);
    pass.setBindGroup(1, this.uniformBindGroup);
    pass.setPipeline(this.pipelines.column_profile);
    pass.dispatchWorkgroups(Math.ceil(maxParticles / WORKGROUP));
    pass.setPipeline(this.pipelines.crust_refresh_derived);
    pass.dispatchWorkgroups(Math.ceil(maxParticles / WORKGROUP));
    // Again, now that every elevation is set (pressures include the water load).
    pass.setPipeline(this.pipelines.column_profile);
    pass.dispatchWorkgroups(Math.ceil(maxParticles / WORKGROUP));
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  /** Copy the mantle's current flow into the GPU grid (positions, velocity, buoyancy, divergence). */
  uploadMantle() {
    const m = this.mantle, n = this.numNodes, o = GridNode.offsets;
    const grid = new Float32Array(n * GridNode.words);
    for (let i = 0; i < n; i++) {
      const base = i * GridNode.words;
      grid.set([m.mesh.pos[3 * i], m.mesh.pos[3 * i + 1], m.mesh.pos[3 * i + 2], 1], base + o.position);
      grid.set([m.velocity[3 * i], m.velocity[3 * i + 1], m.velocity[3 * i + 2], 0], base + o.velocity);
      grid[base + o.buoyancy] = m.b[i];
      grid[base + o.divergence] = m.divergence[i];
    }
    this.device.queue.writeBuffer(this.buffers.grid, 0, grid);
  }

  /**
   * Run one simulation step: records and submits its GPU work. Steps with more
   * than ITERATIONS_PER_SUBMIT iterations are split, and the returned promise
   * waits for each piece before submitting the next.
   */
  async step() {
    // The mantle steps on the CPU (every options.stepMyr) and its flow is uploaded.
    if (this.mantle.advanceTo(this.timeMyr)) this.uploadMantle();

    const { device } = this;
    const { maxParticles, maxEdges, hashGridDim } = CAPACITY;
    const groups = (n) => Math.ceil(n / WORKGROUP);
    const crust = groups(maxParticles), cells = groups(hashGridDim ** 3), edges = groups(maxEdges);
    this.#writeTime();

    let encoder, pass;
    const open = () => {
      encoder = device.createCommandEncoder();
      pass = encoder.beginComputePass({ label: 'simulation step' });
      pass.setBindGroup(0, this.memoryBindGroup);
      pass.setBindGroup(1, this.uniformBindGroup);
    };
    const submit = () => {
      pass.end();
      device.queue.submit([encoder.finish()]);
    };
    const run = (name, count) => { pass.setPipeline(this.pipelines[name]); pass.dispatchWorkgroups(count); };
    open();

    run('crust_locate', crust);

    // Column profiles (design 4.0), then thermal state and isostasy, healing,
    // growth and splitting
    run('column_profile', crust);
    run('crust_thermo_isostasy', crust);
    run('crust_accumulation', crust);
    run('crust_commit_splits', 1);
    run('crust_rewire_edges', edges);
    run('crust_clean_split', crust);

    // Crust motion: overdamped XPBD. Predict where the mantle alone would
    // carry each particle, update the edges (pairs in range), then iterate
    // all edges toward the minimum of drag + elastic energy.
    run('crust_predict', crust);
    run('crust_hash_clear', cells);
    run('crust_hash_build', crust);
    run('solver_begin', 1);
    run('edges_begin', crust);       // clear slot lists
    run('edges_prune', edges);       // drop pairs out of range; list survivors; reset lambda
    run('edges_note_listed', crust);
    run('edges_create', crust);      // new pairs in range
    run('edges_finish', crust);      // edge counts, angle-sorted slots, solver iterates
    run('columns_gpe_force', crust); // GPE push into the predicted positions (design 4.6)
    run('edges_law', edges);         // this step's linear force law and caps (design 4.5)
    for (let i = 0; i < this.params.solver_iterations; i++) {
      if (i > 0 && i % ITERATIONS_PER_SUBMIT === 0) {
        submit();
        await device.queue.onSubmittedWorkDone();
        open();
      }
      run('crust_clear_deltas', crust);
      run('crust_solve_edges', edges);
      run('crust_apply_deltas', crust);
      run('crust_chebyshev_combine', crust);
      run('solver_next_iteration', 1);
    }
    run('crust_clear_deltas', crust);
    run('edges_maxwell', edges);     // slice stresses, inelastic strain, plastic work, integrity (design 4.5, 4.7)
    run('columns_flow', crust);      // volumetric flow at fixed crust volume (design 4.7)
    run('crust_integrate', crust);
    submit();
    this.steps++;
    this.timeMyr += this.params.dt_crust_myr;
  }

  /** Upload the current time (stamps new edges' born_myr). */
  #writeTime() {
    const index = SIM_PARAMS.findIndex(([n]) => n === 'time_myr');
    this.device.queue.writeBuffer(this.buffers.params, 4 * index, Float32Array.of(this.timeMyr));
  }

  /** Plate id per particle slot, from PlateTracker; used only for drawing. */
  setPlateIds(ids) {
    this.device.queue.writeBuffer(this.buffers.plateIds, 0, ids);
  }

  /** Copy named GPU buffers to the CPU. Resolves to {name: ArrayBuffer}. */
  async readBuffers(names) {
    const { device } = this;
    const encoder = device.createCommandEncoder();
    const staging = names.map((name) => {
      const src = this.buffers[name];
      const dst = device.createBuffer({ size: src.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      encoder.copyBufferToBuffer(src, 0, dst, 0, src.size);
      return dst;
    });
    device.queue.submit([encoder.finish()]);
    await Promise.all(staging.map((b) => b.mapAsync(GPUMapMode.READ)));
    const out = {};
    names.forEach((name, i) => {
      out[name] = staging[i].getMappedRange().slice(0);
      staging[i].destroy();
    });
    return out;
  }

  /** Complete state: raw buffers plus the CPU-side counters. Restoring it resumes exactly. */
  async snapshot() {
    return {
      step: this.steps,
      timeMyr: this.timeMyr,
      overrides: { ...this.overrides },
      mantle: this.mantle.getState(),
      buffers: await this.readBuffers(STATE_BUFFERS),
    };
  }

  restore(snap) {
    this.setParams(snap.overrides ?? {});
    for (const name of STATE_BUFFERS) {
      const data = snap.buffers[name];
      if (data.byteLength !== this.buffers[name].size) {
        throw new Error(`snapshot buffer ${name} has ${data.byteLength} bytes, expected ${this.buffers[name].size}`);
      }
      this.device.queue.writeBuffer(this.buffers[name], 0, data);
    }
    this.steps = snap.step;
    this.timeMyr = snap.timeMyr;
    this.mantle.setState(snap.mantle);
    this.uploadMantle();
  }
}

function freeList(used, capacity) {
  const data = new Uint32Array(FREE_LIST_HEADER_WORDS + capacity);
  let n = 0;
  for (let i = used; i < capacity; i++) data[FREE_LIST_HEADER_WORDS + n++] = i;
  data[0] = n;
  return data;
}

/**
 * Spectral radius of one plain Jacobi sweep of the bond solve. The sweep
 * acts on the multipliers with matrix I - (B W B^T + a I) / (d (w_a + w_b + a)),
 * where B is the bond-particle incidence, a = 1 / (bond_rate dt) the scaled
 * compliance and d the largest bond count. B W B^T has eigenvalues in
 * [0, 2d] (for unit drag weights), so the slowest factor is
 * 1 - a / (d (2 + a)), from bond-force patterns that exert no net force.
 */
export function jacobiSpectralRadius(p, maxDegree = 6) {
  const a = 1 / (p.bond_rate_myr_inv * p.dt_crust_myr);
  return 1 - a / (maxDegree * (2 + a));
}

/**
 * Chebyshev weights omega_k for iteration k (Wang 2015): 1 for the first
 * `delay` iterations, then 2 / (2 - rho^2), then 4 / (4 - rho^2 omega_{k-1}).
 * rho = 0 gives plain Jacobi (all ones).
 */
export function chebyshevWeights(rho, delay, n) {
  const w = new Float32Array(n).fill(1);
  const r2 = rho * rho;
  for (let k = delay; k < n && rho > 0; k++) w[k] = k === delay ? 2 / (2 - r2) : 4 / (4 - r2 * w[k - 1]);
  return w;
}

/**
 * Bond stiffness over basal drag for a base-size particle, in 1/Myr.
 * For a triangular network of central springs, the 2D (thickness-integrated)
 * Young's modulus is E_2D = (2/sqrt 3) k, so k = (sqrt 3 / 2) E h_e,
 * independent of spacing. Basal drag of a column is gamma = A eta / h_a.
 * Positions are on the unit sphere, but the radius cancels in the ratio:
 * both the drag term and the elastic energy scale with R^2.
 */
export function bondRate(p) {
  const k = (Math.sqrt(3) / 2) * p.crust_youngs_modulus_gpa * 1e9 * p.stiffness_reference_km * 1e3; // N/m
  const area = 2 * Math.sqrt(3) * (p.base_apothem * EARTH_RADIUS_M) ** 2;                          // m^2
  const gamma = (area * p.asthenosphere_viscosity_pa_s) / (p.drag_reference_channel_km * 1e3);     // N s/m
  return k / (gamma / SECONDS_PER_MYR);
}

export function nearestVertex(vertices, x, y, z) {
  let best = 0, bestDot = -2;
  for (let i = 0; i < vertices.length; i++) {
    const v = vertices[i];
    const d = v[0] * x + v[1] * y + v[2] * z;
    if (d > bestDot) { bestDot = d; best = i; }
  }
  return best;
}
