// GPU-side simulation state and the per-frame compute schedule.

import { Constraint, GridNode, NONE, Particle } from '../gpu/layout.js';
import { createShaderModule } from '../gpu/device.js';
import { packSimParams, SIM_PARAMS, SIM_PARAMS_BYTES, simParamsWGSL } from './params.js';

import bindingsWGSL from '../shaders/compute/bindings.wgsl';
import locateWGSL from '../shaders/compute/locate.wgsl';
import mantleWGSL from '../shaders/compute/mantle.wgsl';
import thermalWGSL from '../shaders/compute/thermal.wgsl';
import growthWGSL from '../shaders/compute/growth.wgsl';
import solverWGSL from '../shaders/compute/solver.wgsl';

export const CAPACITY = {
  maxParticles: 25000,
  maxConstraints: 100000,
  hashGridDim: 32,       // collision grid cells per axis over [-1.5, 1.5]
  adjacencyWidth: 6,     // neighbours stored per mantle node
};

const EARTH_RADIUS_M = 6.371e6;
const SECONDS_PER_MYR = 3.15576e13;
const MANTLE_STABILITY = 0.5;  // fraction of the explicit-Euler stability limit used per substep
const FREE_LIST_HEADER_WORDS = 4;
const MAX_SOLVER_ITERATIONS = 1024; // size of the Chebyshev weight table
const WORKGROUP = 64;

/** Buffers that together are the complete simulation state (snapshot / restore). */
export const STATE_BUFFERS = ['grid', 'particles', 'constraints', 'particlePool', 'constraintPool'];

/** WGSL struct definitions shared by compute and render shaders. */
export function structsWGSL({ atomicGrid = false } = {}) {
  return Particle.wgsl() + Constraint.wgsl()
    + GridNode.wgsl(atomicGrid ? { thermal_atomic: 'atomic<i32>' } : {})
    + simParamsWGSL();
}

export class Simulation {
  constructor(device, icosphere, onError) {
    this.device = device;
    this.icosphere = icosphere;
    this.numNodes = icosphere.vertices.length;
    this.steps = 0;
    this.timeMyr = 0;
    const { maxParticles, maxConstraints, hashGridDim, adjacencyWidth } = CAPACITY;
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

    this.buffers = {
      grid: device.createBuffer({ label: 'grid', size: this.numNodes * GridNode.byteSize, usage: S }),
      adjacency: device.createBuffer({ label: 'adjacency', size: this.numNodes * adjacencyWidth * 4, usage: S }),
      particles: device.createBuffer({ label: 'particles', size: maxParticles * Particle.byteSize, usage: S }),
      constraints: device.createBuffer({ label: 'constraints', size: maxConstraints * Constraint.byteSize, usage: S }),
      particlePool: device.createBuffer({ label: 'particle free list', size: (FREE_LIST_HEADER_WORDS + maxParticles) * 4, usage: S }),
      constraintPool: device.createBuffer({ label: 'constraint free list', size: (FREE_LIST_HEADER_WORDS + maxConstraints) * 4, usage: S }),
      hashGrid: device.createBuffer({ label: 'hash grid', size: hashGridDim ** 3 * 4, usage: S }),
      deltas: device.createBuffer({ label: 'particle deltas', size: maxParticles * 16, usage: S }),
      solver: device.createBuffer({ label: 'solver state', size: (4 + MAX_SOLVER_ITERATIONS) * 4, usage: S }),
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

  /**
   * Set parameter overrides (merged with previous ones) and upload. Derived
   * values are recomputed: the mantle substep is the largest step within
   * MANTLE_STABILITY of the explicit-Euler limit that divides the mantle
   * time per crust step evenly.
   */
  setParams(overrides) {
    Object.assign(this.overrides, overrides);
    const value = (name) => (name in this.overrides ? this.overrides[name] : SIM_PARAMS.find(([n]) => n === name)[2]);

    const mantleTimePerStep = value('mantle_time_per_myr') * value('dt_crust_myr');
    const limit = mantleStabilityLimit(this.icosphere, value('laplacian_weight'), value('mantle_wavenumber'), value('mantle_rayleigh'));
    this.mantleSubsteps = Math.max(1, Math.ceil(mantleTimePerStep / (MANTLE_STABILITY * limit.dtMax)));
    this.mantleStability = { ...limit, dtUsed: mantleTimePerStep / this.mantleSubsteps };

    this.params = {};
    for (const [name] of SIM_PARAMS) this.params[name] = value(name);
    Object.assign(this.params, {
      bond_rate_myr_inv: bondRate(this.params),
      dt_mantle: this.mantleStability.dtUsed,
      num_nodes: this.numNodes,
      max_particles: CAPACITY.maxParticles,
      max_constraints: CAPACITY.maxConstraints,
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
    const code = [structsWGSL({ atomicGrid: true }), bindingsWGSL, locateWGSL, mantleWGSL, thermalWGSL, growthWGSL, solverWGSL].join('\n');
    const module = createShaderModule(device, 'compute', code, onError);

    const order = ['grid', 'adjacency', 'particles', 'constraints', 'particlePool', 'constraintPool', 'hashGrid', 'deltas', 'solver'];
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
      'crust_locate',
      'grid_clear_thermal', 'crust_apply_thermal', 'grid_finalize_thermal',
      'mantle_laplacian', 'mantle_integrate', 'mantle_velocity',
      'crust_thermo_isostasy', 'crust_refresh_derived', 'crust_constraint_healing',
      'crust_accumulation', 'crust_commit_splits', 'crust_rewire_bonds', 'crust_clean_split',
      'crust_predict', 'crust_hash_clear', 'crust_hash_build',
      'crust_begin_constraints', 'crust_store_bond_count',
      'solver_begin', 'solver_next_iteration', 'crust_chebyshev_combine', 'crust_bond_stress',
      'crust_clear_deltas', 'crust_solve_constraints', 'crust_solve_collisions', 'crust_apply_deltas',
      'crust_integrate',
    ];
    this.pipelines = Object.fromEntries(entryPoints.map((entryPoint) => [
      entryPoint,
      device.createComputePipeline({ label: entryPoint, layout, compute: { module, entryPoint } }),
    ]));
  }

  /**
   * Upload an initial state (see src/init/initial-state.js) and reset the free lists.
   */
  load(state) {
    const { device, buffers, numNodes } = this;
    const { maxParticles, maxConstraints } = CAPACITY;
    const { particles: P, bonds: B, mantlePsi } = state;
    if (P.count > maxParticles) throw new Error(`initial particles ${P.count} exceed capacity ${maxParticles}`);
    if (B.count > maxConstraints) throw new Error(`initial bonds ${B.count} exceed capacity ${maxConstraints}`);

    // Mantle grid
    const grid = new ArrayBuffer(numNodes * GridNode.byteSize);
    const gf = new Float32Array(grid);
    const o = GridNode.offsets;
    this.icosphere.vertices.forEach((v, i) => {
      const base = i * GridNode.words;
      gf.set([v[0], v[1], v[2], 1], base + o.position);
      gf[base + o.psi] = mantlePsi[i];
    });
    device.queue.writeBuffer(buffers.grid, 0, grid);

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
      pu[base + po.split_child] = NONE;
      pu[base + po.mantle_node] = nearestVertex(this.icosphere.vertices, x, y, z);
    }
    device.queue.writeBuffer(buffers.particles, 0, pbuf);

    // Bonds
    const cbuf = new ArrayBuffer(maxConstraints * Constraint.byteSize);
    const cf = new Float32Array(cbuf);
    const cu = new Uint32Array(cbuf);
    const co = Constraint.offsets;
    for (let k = 0; k < B.count; k++) {
      const base = k * Constraint.words;
      cu[base + co.particle_a] = B.a[k];
      cu[base + co.particle_b] = B.b[k];
      cf[base + co.rest_length] = B.restLength[k];
      cf[base + co.yield_stress_mpa] = state.yieldStressMPa;
      cu[base + co.is_active] = 1;
      cf[base + co.damage] = B.damage[k];
    }
    device.queue.writeBuffer(buffers.constraints, 0, cbuf);

    // Free lists hold every slot not used by the initial state.
    device.queue.writeBuffer(buffers.particlePool, 0, freeList(P.count, maxParticles));
    device.queue.writeBuffer(buffers.constraintPool, 0, freeList(B.count, maxConstraints));
    this.steps = 0;
    this.timeMyr = 0;

    // Density and elevation are derived; compute them now so step 0 is complete.
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass({ label: 'refresh derived' });
    pass.setBindGroup(0, this.memoryBindGroup);
    pass.setBindGroup(1, this.uniformBindGroup);
    pass.setPipeline(this.pipelines.crust_refresh_derived);
    pass.dispatchWorkgroups(Math.ceil(maxParticles / WORKGROUP));
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  /** Record one simulation step into a command encoder. */
  encodeStep(encoder) {
    const { maxParticles, maxConstraints, hashGridDim } = CAPACITY;
    const groups = (n) => Math.ceil(n / WORKGROUP);
    const nodes = groups(this.numNodes), crust = groups(maxParticles);
    const bonds = groups(maxConstraints), cells = groups(hashGridDim ** 3);

    const pass = encoder.beginComputePass({ label: 'simulation step' });
    pass.setBindGroup(0, this.memoryBindGroup);
    pass.setBindGroup(1, this.uniformBindGroup);
    const run = (name, count) => { pass.setPipeline(this.pipelines[name]); pass.dispatchWorkgroups(count); };

    run('crust_locate', crust);

    // Crust -> mantle heat feedback
    run('grid_clear_thermal', nodes);
    run('crust_apply_thermal', crust);
    run('grid_finalize_thermal', nodes);

    // Mantle pattern and the velocity field it implies
    for (let i = 0; i < this.mantleSubsteps; i++) {
      run('mantle_laplacian', nodes);
      run('mantle_integrate', nodes);
    }
    run('mantle_velocity', nodes);

    // Crust thermal state, healing, growth and splitting
    run('crust_thermo_isostasy', crust);
    run('crust_constraint_healing', bonds);
    run('crust_accumulation', crust);
    run('crust_commit_splits', 1);
    run('crust_rewire_bonds', bonds);
    run('crust_clean_split', crust);

    // Crust motion: overdamped XPBD. Predict where the mantle alone would
    // carry each particle, then iterate bonds and collisions toward the
    // minimum of drag + elastic energy.
    run('crust_predict', crust);
    run('crust_hash_clear', cells);
    run('crust_hash_build', crust);
    run('crust_clear_deltas', crust);
    run('crust_begin_constraints', bonds); // reset lambda, count bonds per particle
    run('crust_store_bond_count', crust);
    run('solver_begin', 1);
    for (let i = 0; i < this.params.solver_iterations; i++) {
      run('crust_clear_deltas', crust);
      run('crust_solve_constraints', bonds);
      run('crust_apply_deltas', crust);
      run('crust_clear_deltas', crust);
      run('crust_solve_collisions', crust);
      run('crust_apply_deltas', crust);
      run('crust_chebyshev_combine', crust);
      run('solver_next_iteration', 1);
    }
    run('crust_bond_stress', bonds);
    run('crust_integrate', crust);
    pass.end();
    this.steps++;
    this.timeMyr += this.params.dt_crust_myr;
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
  const k = (Math.sqrt(3) / 2) * p.crust_youngs_modulus_gpa * 1e9 * p.elastic_thickness_km * 1e3; // N/m
  const area = 2 * Math.sqrt(3) * (p.base_apothem * EARTH_RADIUS_M) ** 2;                          // m^2
  const gamma = (area * p.asthenosphere_viscosity_pa_s) / (p.asthenosphere_thickness_km * 1e3);     // N s/m
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

/**
 * Explicit-Euler stability limit of the linearised Swift-Hohenberg operator
 * r - (L + k^2)^2 on the icosphere, where L is the weighted graph Laplacian
 * used by mantle.wgsl (eigenvalues -mu, mu in [0, muMax]). The most negative
 * eigenvalue of the operator is r - (muMax - k^2)^2; Euler is stable while
 * dt * |that| < 2. muMax comes from power iteration.
 */
export function mantleStabilityLimit(icosphere, weight, k0, r) {
  const n = icosphere.vertices.length;
  let x = new Float64Array(n).map((_, i) => Math.sin(1.7 * i + 0.3)); // deterministic start
  let mu = 0;
  for (let it = 0; it < 200; it++) {
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const nb = icosphere.neighbours[i].slice(0, CAPACITY.adjacencyWidth);
      let s = nb.length * x[i];
      for (const j of nb) s -= x[j];
      y[i] = weight * s;
    }
    const norm = Math.hypot(...y);
    mu = norm / Math.hypot(...x);
    x = y.map((v) => v / norm);
  }
  const lambda = (mu - k0 * k0) ** 2 - r;
  return { muMax: mu, lambdaMax: lambda, dtMax: 2 / lambda };
}
