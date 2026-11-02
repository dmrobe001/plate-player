// GPU-side simulation state and the per-frame compute schedule.

import { ColumnProfile, Edge, GridNode, NONE, Particle, SLOTS } from '../gpu/layout.js';
import { createShaderModule } from '../gpu/device.js';
import { normalStiffnessPerModulus } from '../columns/shape.js';
import { tangentBasis } from '../columns/flow.js';
import { KinematicMantle } from '../mantle/kinematic.js';
import { Mantle } from '../mantle/mantle.js';
import { packSimParams, SIM_PARAMS, SIM_PARAMS_BYTES, simParamsWGSL } from './params.js';

import bindingsWGSL from '../shaders/compute/bindings.wgsl';
import locateWGSL from '../shaders/compute/locate.wgsl';
import thermalWGSL from '../shaders/compute/thermal.wgsl';
import splitWGSL from '../shaders/compute/split.wgsl';
import solverWGSL from '../shaders/compute/solver.wgsl';
import edgesWGSL from '../shaders/compute/edges.wgsl';
import profileWGSL from '../shaders/compute/profile.wgsl';
import lawWGSL from '../shaders/compute/law.wgsl';
import flowWGSL from '../shaders/compute/flow.wgsl';
import areaWGSL from '../shaders/compute/area.wgsl';
import flexureWGSL from '../shaders/compute/flexure.wgsl';
import hingeWGSL from '../shaders/compute/hinge.wgsl';

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
/**
 * Words before the Chebyshev weights in the solver buffer (SolverState in
 * bindings.wgsl): counters since load. Volume sums are 64-bit fixed point.
 */
export const SOLVER_HEADER = {
  words: 20, edgePoolEmpty: 1, slotOverflow: 2, clippedCells: 3, source: 4, sink: 6, breakups: 8, slabsLost: 9,
  particleHigh: 10, edgeHigh: 11, dispatch: 12, drawInstances: 18, volumeFixed: 1e12,
};
const WORKGROUP = 64;
// GPU submissions. Long stretches of GPU work can exceed the browser's GPU
// watchdog on slow GPUs (the device is then lost), so a step with many solver
// (or flexure) iterations is submitted in pieces, waiting for each. A piece
// holds about WORK_PER_SUBMIT thread launches: at the default resolution
// (~10k columns, ~40k edges) that is the 16 iterations that were always safe;
// a coarse world (162 columns) does its whole step in one piece.
const WORK_PER_SUBMIT = 1_000_000;
const MIN_ITERATIONS_PER_SUBMIT = 16;
const MIN_FLEXURE_ITERATIONS_PER_SUBMIT = 64;

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
      solver: device.createBuffer({ label: 'solver state', size: (SOLVER_HEADER.words + MAX_SOLVER_ITERATIONS) * 4, usage: S }),
      profiles: device.createBuffer({ label: 'column profiles (derived)', size: maxParticles * ColumnProfile.byteSize, usage: S }),
      plateIds: device.createBuffer({ label: 'plate ids (drawing only)', size: maxParticles * 4, usage: S }),
      params: device.createBuffer({ label: 'sim params', size: SIM_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      // Indirect dispatch sizes over the particle and edge slots in use, copied
      // from the solver header (solver_dispatch_sizes) between passes: a buffer
      // bound writable in a dispatch cannot also be its indirect argument.
      dispatchArgs: device.createBuffer({ label: 'dispatch sizes', size: 6 * 4, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST }),
    };
    // Lagged readback of the high-water marks, for sizing submissions only.
    this.highWater = { particles: 0, edges: 0 };
    this.highWaterStaging = device.createBuffer({ label: 'high-water readback', size: 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.highWaterPending = false;

    const adjacency = new Uint32Array(this.numNodes * adjacencyWidth).fill(NONE);
    icosphere.neighbours.forEach((list, i) => adjacency.set(list.slice(0, adjacencyWidth), i * adjacencyWidth));
    device.queue.writeBuffer(this.buffers.adjacency, 0, adjacency);

    this.overrides = {};
    this.setParams({});
    this.#createPipelines(onError);
  }

  /** Replace all parameter overrides (a new world or a restored record). */
  resetParams(overrides = {}) {
    this.overrides = {};
    this.setParams(overrides);
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
      // A world may use fewer, larger hash cells (coarse columns); the buffer holds CAPACITY.hashGridDim^3.
      hash_grid_dim: this.overrides.hash_grid_dim ?? CAPACITY.hashGridDim,
    });
    if (this.params.hash_grid_dim > CAPACITY.hashGridDim) throw new Error(`hash_grid_dim > ${CAPACITY.hashGridDim}`);
    if (this.params.solver_iterations > MAX_SOLVER_ITERATIONS) throw new Error(`solver_iterations > ${MAX_SOLVER_ITERATIONS}`);
    this.device.queue.writeBuffer(this.buffers.params, 0, packSimParams(this.params));
    const rho = this.params.chebyshev_rho < 0 ? chebyshevRho(this.params) : this.params.chebyshev_rho;
    this.chebyshevRho = rho;
    const omega = chebyshevWeights(rho, this.params.chebyshev_delay, MAX_SOLVER_ITERATIONS);
    this.device.queue.writeBuffer(this.buffers.solver, 4 * SOLVER_HEADER.words, omega);
  }

  #createPipelines(onError) {
    const { device, buffers } = this;
    const code = [structsWGSL(), bindingsWGSL, locateWGSL, thermalWGSL, solverWGSL, edgesWGSL, profileWGSL, lawWGSL, flowWGSL, areaWGSL, splitWGSL, flexureWGSL, hingeWGSL].join('\n');
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
      'columns_split', 'crust_commit_splits', 'crust_rewire_edges', 'crust_clean_split',
      'columns_area', 'columns_volume', 'columns_init_volume',
      'slabs_arc_count', 'columns_arc', 'columns_flexure_begin', 'columns_moment', 'columns_flexure',
      'crust_predict', 'crust_hash_clear', 'crust_hash_build',
      'edges_begin', 'edges_prune', 'edges_note_listed', 'edges_create', 'edges_finish', 'edges_law', 'edges_maxwell',
      'columns_hinge',
      'solver_find_high', 'solver_dispatch_sizes',
      'solver_begin', 'solver_next_iteration', 'crust_warm_start',
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

    // Slabs hang from an intact bonded surface partner (the first found).
    const hangFrom = {};
    if (P.state) {
      for (let k = 0; k < B.count; k++) {
        const i = B.a[k], j = B.b[k];
        if (B.damage[k] >= 0.5) continue;
        if (P.state[i] === 2 && P.state[j] !== 2 && hangFrom[i] === undefined) hangFrom[i] = j;
        if (P.state[j] === 2 && P.state[i] !== 2 && hangFrom[j] === undefined) hangFrom[j] = i;
      }
    }

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
      pf[base + po.thermal_age_myr] = P.thermalAge ? P.thermalAge[i] : P.age[i]; // clamped to its cap on the GPU
      pf[base + po.area_ref] = 2 * Math.sqrt(3) * P.apothem[i] ** 2;
      pf[base + po.apothem] = P.apothem[i];
      pu[base + po.state] = P.state ? P.state[i] : 1; // 1 surface, 2 slab (tests may load slabs)
      pu[base + po.next_state] = pu[base + po.state];
      pu[base + po.geometry_reset] = pu[base + po.state] === 2 ? 1 : 0; // slab edges: length measured in 3D from step 1
      pu[base + po.is_active] = 1;
      pu[base + po.next_idx] = NONE;
      pf[base + po.density] = P.compositionDensity[i];
      pf[base + po.elevation] = P.elevation ? P.elevation[i] : NaN; // NaN: start at the Airy height
      pu[base + po.split_child] = NONE;
      pu[base + po.mantle_node] = nearestVertex(this.icosphere.vertices, x, y, z);
      // Body axis (phase G): the tangent basis e1 (tangent_e1 in area.wgsl) unless given; round shape unless given.
      const b = P.bodyAxis ? [P.bodyAxis[3 * i], P.bodyAxis[3 * i + 1], P.bodyAxis[3 * i + 2]] : tangentBasis([x, y, z])[0];
      pf.set([...b, 0], base + po.body_axis);
      pf[base + po.shape_p] = P.shapeP ? P.shapeP[i] : 0;
      pf[base + po.shape_q] = P.shapeQ ? P.shapeQ[i] : 0;
      // Trench hinge (design 6): none yet. A loaded slab hangs from the given
      // column, else from its first intact bonded surface partner, at a dip of 0.5 rad.
      pu[base + po.hinge_partner] = NONE;
      pu[base + po.slab_child] = NONE;
      pu[base + po.attached_to] = pu[base + po.state] === 2 ? (P.attachedTo ? P.attachedTo[i] : (hangFrom[i] ?? NONE)) : NONE;
      if (pu[base + po.state] === 2) {
        pf[base + po.hinge_angle] = 0.5;
        pf[base + po.slab_extent_km] = 2 * P.apothem[i] * 6371 * Math.sin(0.5);
      }
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
      const rest = P.apothem[i] + P.apothem[j]; // (loaded columns start round unless shapes are given; edges_law sets the shaped rest)
      ef[base + eo.length] = d;
      [eo.direction_x, eo.direction_y, eo.direction_z].forEach((o, c) => { ef[base + o] = P.position[3 * j + c] - P.position[3 * i + c]; });
      ef[base + eo.rest] = rest;
      ef[base + eo.lever_a] = 0.5;
      ef[base + eo.strain] = (d - rest) / rest;
      ef.fill(this.params.crust_youngs_modulus_gpa * 1000 * ((d - rest) / rest), base + eo.slice_stress_mpa, base + eo.slice_stress_mpa + 8);
    }
    device.queue.writeBuffer(buffers.edges, 0, ebuf);

    // Free lists hold every slot not used by the initial state.
    device.queue.writeBuffer(buffers.particlePool, 0, freeList(P.count, maxParticles));
    device.queue.writeBuffer(buffers.edgePool, 0, freeList(B.count, maxEdges));
    device.queue.writeBuffer(buffers.solver, 0, new Uint32Array(SOLVER_HEADER.words)); // iteration and counters since load (high-water marks: refresh pass)
    this.highWater = { particles: P.count, edges: B.count };
    this.steps = 0;
    this.timeMyr = 0;

    // Cells, volume, density and elevation are derived; compute them now so
    // step 0 is complete. Crust volume = loaded thickness x tessellated area.
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass({ label: 'refresh derived' });
    pass.setBindGroup(0, this.memoryBindGroup);
    pass.setBindGroup(1, this.uniformBindGroup);
    pass.setPipeline(this.pipelines.crust_hash_clear);
    pass.dispatchWorkgroups(Math.ceil(CAPACITY.hashGridDim ** 3 / WORKGROUP));
    pass.setPipeline(this.pipelines.crust_hash_build);
    pass.dispatchWorkgroups(Math.ceil(maxParticles / WORKGROUP));
    pass.setPipeline(this.pipelines.columns_area);
    pass.dispatchWorkgroups(Math.ceil(maxParticles / WORKGROUP));
    pass.setPipeline(this.pipelines.columns_init_volume);
    pass.dispatchWorkgroups(Math.ceil(maxParticles / WORKGROUP));
    pass.setPipeline(this.pipelines.column_profile);
    pass.dispatchWorkgroups(Math.ceil(maxParticles / WORKGROUP));
    pass.setPipeline(this.pipelines.crust_refresh_derived);
    pass.dispatchWorkgroups(Math.ceil(maxParticles / WORKGROUP));
    // Again, now that every elevation is set (pressures include the water load).
    pass.setPipeline(this.pipelines.column_profile);
    pass.dispatchWorkgroups(Math.ceil(maxParticles / WORKGROUP));
    this.#encodeFindHigh(pass);
    pass.end();
    this.#encodeCopySizes(encoder);
    device.queue.submit([encoder.finish()]);
  }

  /** Rebuild the high-water marks by scanning every slot, then the dispatch sizes. */
  #encodeFindHigh(pass) {
    const { maxParticles, maxEdges } = CAPACITY;
    pass.setPipeline(this.pipelines.solver_find_high);
    pass.dispatchWorkgroups(Math.ceil(Math.max(maxParticles, maxEdges) / WORKGROUP));
    pass.setPipeline(this.pipelines.solver_dispatch_sizes);
    pass.dispatchWorkgroups(1);
  }

  /** Copy the dispatch sizes from the solver header to the indirect-argument buffer (outside a pass). */
  #encodeCopySizes(encoder) {
    encoder.copyBufferToBuffer(this.buffers.solver, 4 * SOLVER_HEADER.dispatch, this.buffers.dispatchArgs, 0, 6 * 4);
  }

  /**
   * Read the high-water marks back without waiting (one readback in flight).
   * They size the submissions (watchdog), so a step or two of lag is fine.
   */
  #pollHighWater() {
    if (this.highWaterPending) return;
    this.highWaterPending = true;
    const { device } = this;
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.buffers.solver, 4 * SOLVER_HEADER.particleHigh, this.highWaterStaging, 0, 8);
    device.queue.submit([encoder.finish()]);
    this.highWaterStaging.mapAsync(GPUMapMode.READ).then(() => {
      const [particles, edges] = new Uint32Array(this.highWaterStaging.getMappedRange());
      this.highWater = { particles, edges };
      this.highWaterStaging.unmap();
    }).catch(() => {}).finally(() => { this.highWaterPending = false; });
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
    const groups = (n) => Math.ceil(n / WORKGROUP);
    const cells = groups(this.params.hash_grid_dim ** 3);
    // Per-particle and per-edge kernels run over the slots in use (the
    // high-water marks), as indirect dispatches; see resize().
    const crust = 'particles', edges = 'edges';
    this.#writeTime();

    // Submission sizes from the (lagged) problem size; 1.5x for growth since.
    const P = 1.5 * Math.max(this.highWater.particles, 1), E = 1.5 * Math.max(this.highWater.edges, 1);
    const iterationsPerSubmit = Math.max(MIN_ITERATIONS_PER_SUBMIT, Math.floor(WORK_PER_SUBMIT / (3 * P + E)));
    const flexurePerSubmit = Math.max(MIN_FLEXURE_ITERATIONS_PER_SUBMIT, Math.floor(WORK_PER_SUBMIT / (2 * P)));

    let encoder, pass;
    const beginPass = () => {
      pass = encoder.beginComputePass({ label: 'simulation step' });
      pass.setBindGroup(0, this.memoryBindGroup);
      pass.setBindGroup(1, this.uniformBindGroup);
    };
    const open = () => {
      encoder = device.createCommandEncoder();
      beginPass();
    };
    const submit = () => {
      pass.end();
      device.queue.submit([encoder.finish()]);
    };
    const run = (name, count) => {
      pass.setPipeline(this.pipelines[name]);
      if (count === crust) pass.dispatchWorkgroupsIndirect(this.buffers.dispatchArgs, 0);
      else if (count === edges) pass.dispatchWorkgroupsIndirect(this.buffers.dispatchArgs, 12);
      else pass.dispatchWorkgroups(count);
    };
    // New dispatch sizes after slots were handed out (splits, new edges).
    const resize = () => {
      run('solver_dispatch_sizes', 1);
      pass.end();
      this.#encodeCopySizes(encoder);
      beginPass();
    };
    open();
    resize();

    run('crust_locate', crust);

    // Splits (design 4.8), on last step's cells.
    run('columns_split', crust);
    resize();
    run('crust_commit_splits', 1);
    run('crust_rewire_edges', edges);
    run('crust_clean_split', crust);

    // Predict where the mantle alone would carry each particle; hash grid.
    run('crust_predict', crust);
    run('crust_hash_clear', cells);
    run('crust_hash_build', crust);

    // Tessellated area; dilatation thins the lithosphere and brings melt;
    // crust thickness = volume / area (design 4.8). Then column profiles
    // (design 4.0), thermal state, slab sinking (design 6) and arcs.
    run('columns_area', crust);
    run('columns_volume', crust);
    run('column_profile', crust);
    run('crust_thermo_isostasy', crust);
    run('slabs_arc_count', crust);
    run('columns_arc', crust);

    // Surface elevations: Winkler isostasy with plate bending (design 4.3, 4.4).
    run('columns_flexure_begin', crust);
    for (let i = 0; i < this.params.flexure_iterations; i++) {
      if (i > 0 && i % flexurePerSubmit === 0) {
        submit();
        await device.queue.onSubmittedWorkDone();
        open();
      }
      run('columns_moment', crust);
      run('columns_flexure', crust);
    }

    // Crust motion: overdamped XPBD. Update the edges (pairs in range), then
    // iterate all edges toward the minimum of drag + elastic energy.
    run('solver_begin', 1);
    run('edges_begin', crust);       // clear slot lists
    run('edges_prune', edges);       // drop pairs out of range; list survivors; reset lambda
    run('edges_note_listed', crust);
    run('edges_create', crust);      // new pairs in range
    resize();
    run('edges_finish', crust);      // edge counts, angle-sorted slots, solver iterates
    run('columns_gpe_force', crust); // GPE push into the predicted positions (design 4.6)
    run('columns_hinge', crust);     // trench hinges: bend, subduction, slab pull (design 6)
    run('edges_law', edges);         // this step's linear force law and caps (design 4.5); warm-start multipliers
    if (this.params.solver_iterations > 0) { // (0: no solve, positions stay predicted)
      run('crust_clear_deltas', crust);
      run('crust_warm_start', edges);  // positions to match the warm-start multipliers (design 5.2)
      run('crust_apply_deltas', crust);
    }
    for (let i = 0; i < this.params.solver_iterations; i++) {
      if (i > 0 && i % iterationsPerSubmit === 0) {
        submit();
        await device.queue.onSubmittedWorkDone();
        open();
      }
      run('crust_clear_deltas', crust);
      run('crust_solve_edges', edges);
      run('crust_apply_deltas', crust);
      run('solver_next_iteration', 1);
    }
    run('crust_clear_deltas', crust);
    run('edges_maxwell', edges);     // slice stresses, inelastic strain, plastic work, integrity (design 4.5, 4.7)
    run('columns_flow', crust);      // volumetric flow at fixed crust volume (design 4.7)
    run('crust_integrate', crust);
    submit();
    this.#pollHighWater();
    this.steps++;
    this.timeMyr += this.params.dt_crust_myr;
  }

  /** Upload the current time (stamps new edges' born_myr). */
  #writeTime() {
    const index = SIM_PARAMS.findIndex(([n]) => n === 'time_myr');
    this.device.queue.writeBuffer(this.buffers.params, 4 * index, Float32Array.of(this.timeMyr));
  }

  /**
   * Counters since load (SOLVER_HEADER): crust volume added by sources (melt)
   * and removed by sinks (slabs), km x unit-sphere area; breakups; clipped
   * tessellation cells (cell-steps); capacity overflows.
   */
  async counters() {
    const h = new Uint32Array((await this.readBuffers(['solver'])).solver, 0, SOLVER_HEADER.words);
    const sum = (k) => (h[k + 1] * 4294967296 + h[k]) / SOLVER_HEADER.volumeFixed;
    return {
      volumeSource: sum(SOLVER_HEADER.source), volumeSink: sum(SOLVER_HEADER.sink), breakups: h[SOLVER_HEADER.breakups],
      slabsLost: h[SOLVER_HEADER.slabsLost], // slabs broken off or removed while hanging from a surface column (design 9.2)
      clippedCells: h[SOLVER_HEADER.clippedCells], edgePoolEmpty: h[SOLVER_HEADER.edgePoolEmpty], slotOverflow: h[SOLVER_HEADER.slotOverflow],
    };
  }

  /** Plate id per particle slot, from PlateTracker; used only for drawing. */
  setPlateIds(ids) {
    this.device.queue.writeBuffer(this.buffers.plateIds, 0, ids);
  }

  /**
   * Copy named GPU buffers to the CPU. Resolves to {name: ArrayBuffer}, each
   * the buffer's full size. `prefix` (bytes per name) copies only the start
   * of a buffer; the rest reads as zeros.
   */
  async readBuffers(names, prefix = {}) {
    const { device } = this;
    const encoder = device.createCommandEncoder();
    const staging = names.map((name) => {
      const src = this.buffers[name];
      const size = Math.max(4, Math.min(src.size, prefix[name] ?? src.size));
      const dst = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      encoder.copyBufferToBuffer(src, 0, dst, 0, size);
      return dst;
    });
    device.queue.submit([encoder.finish()]);
    await Promise.all(staging.map((b) => b.mapAsync(GPUMapMode.READ)));
    const out = {};
    names.forEach((name, i) => {
      const src = staging[i].getMappedRange();
      if (src.byteLength === this.buffers[name].size) out[name] = src.slice(0);
      else {
        out[name] = new ArrayBuffer(this.buffers[name].size);
        new Uint8Array(out[name]).set(new Uint8Array(src));
      }
      staging[i].destroy();
    });
    return out;
  }

  /** Slots in use: one past the highest particle and edge slot ever handed out (exact; waits for the GPU). */
  async slotsInUse() {
    const h = new Uint32Array((await this.readBuffers(['solver'], { solver: 4 * SOLVER_HEADER.words })).solver, 0, SOLVER_HEADER.words);
    return { particles: h[SOLVER_HEADER.particleHigh], edges: h[SOLVER_HEADER.edgeHigh] };
  }

  /**
   * Complete state: raw buffers plus the CPU-side counters. Restoring it
   * resumes exactly. Only the slots in use are read from the GPU (the rest
   * are zeros); `counts` says how many, for consumers that loop over slots.
   */
  async snapshot() {
    const counts = await this.slotsInUse();
    const prefix = { particles: counts.particles * Particle.byteSize, edges: counts.edges * Edge.byteSize };
    return {
      step: this.steps,
      timeMyr: this.timeMyr,
      overrides: { ...this.overrides },
      mantle: this.mantle.getState(),
      buffers: await this.readBuffers(STATE_BUFFERS, prefix),
      counts,
    };
  }

  restore(snap) {
    this.resetParams(snap.overrides ?? {});
    for (const name of STATE_BUFFERS) {
      const data = snap.buffers[name];
      if (data.byteLength !== this.buffers[name].size) {
        throw new Error(`snapshot buffer ${name} has ${data.byteLength} bytes, expected ${this.buffers[name].size}`);
      }
      // (Free lists re-sorted lowest-slot-first: records made before the order changed.)
      this.device.queue.writeBuffer(this.buffers[name], 0, name === 'particlePool' || name === 'edgePool' ? sortFreeList(data) : data);
    }
    // High-water marks from a scan of the restored slots.
    this.device.queue.writeBuffer(this.buffers.solver, 4 * SOLVER_HEADER.particleHigh, new Uint32Array(2));
    {
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass({ label: 'restore: high-water marks' });
      pass.setBindGroup(0, this.memoryBindGroup);
      pass.setBindGroup(1, this.uniformBindGroup);
      this.#encodeFindHigh(pass);
      pass.end();
      this.#encodeCopySizes(encoder);
      this.device.queue.submit([encoder.finish()]);
    }
    this.highWater = { particles: CAPACITY.maxParticles, edges: CAPACITY.maxEdges }; // until the first poll
    this.#pollHighWater();
    this.steps = snap.step;
    this.timeMyr = snap.timeMyr;
    if (snap.mantle.kind === 'kinematic' && this.mantle.kind !== 'kinematic') this.mantle = new KinematicMantle(snap.mantle.options, this.mantleMesh);
    if (snap.mantle.kind !== 'kinematic' && this.mantle.kind === 'kinematic') this.mantle = new Mantle(1, snap.mantle.options, this.mantleMesh);
    this.mantle.setState(snap.mantle);
    this.uploadMantle();
  }
}

// A free list is a stack (pops take the top, indices[counter - 1]). Stacked
// highest first, so the lowest free slot is handed out first and the slots
// in use stay packed below the high-water marks.
function freeList(used, capacity) {
  const data = new Uint32Array(FREE_LIST_HEADER_WORDS + capacity);
  let n = 0;
  for (let i = capacity - 1; i >= used; i--) data[FREE_LIST_HEADER_WORDS + n++] = i;
  data[0] = n;
  return data;
}

/** A free list's buffer with its stack re-sorted highest first (as freeList). */
function sortFreeList(buffer) {
  const data = new Uint32Array(buffer.slice(0));
  const stack = data.subarray(FREE_LIST_HEADER_WORDS, FREE_LIST_HEADER_WORDS + data[0]);
  stack.sort().reverse();
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
 * The spectral radius the Chebyshev weights are built for (chebyshev_rho < 0;
 * design 5.2, Chebyshev findings): the derived one, but at most what the
 * iteration budget can use, and at most CHEBYSHEV_MAX_RHO.
 * After N accelerated sweeps Chebyshev bounds every mode in [-rho, rho] by
 * 1 / T_N(1 / rho) ~ 1 / cosh(N sqrt(2 (1 - rho))). With rho near 1 and N
 * below ~1 / sqrt(2 (1 - rho)) that bound is near 1 for every mode, fast ones
 * included, which plain Jacobi would have removed in a sweep: so rho is capped
 * where N sqrt(2 (1 - rho)) = CHEBYSHEV_REACH (1 / cosh 4 = 4%). Slower modes
 * are left to the warm start, which carries them from step to step.
 * Overestimating rho is harmless for a linear solve but not with caps: the
 * momentum of rows that hit a cap derails the iteration (a strip under
 * concentrated slab pull tore at 0.99987, stayed exact at 0.999).
 */
export const CHEBYSHEV_REACH = 4;
export const CHEBYSHEV_MAX_RHO = 0.999;
export function chebyshevRho(p) {
  const n = p.solver_iterations - p.chebyshev_delay;
  if (n <= 1) return 0;
  return Math.min(jacobiSpectralRadius(p), 1 - 0.5 * (CHEBYSHEV_REACH / n) ** 2, CHEBYSHEV_MAX_RHO);
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
 * For a triangular network of normal springs k and tangential springs r k
 * (shear_stiffness_ratio), the 2D (thickness-integrated) Young's modulus is
 * E_2D = 2 sqrt(3) k (1 + r) / (3 + r) (latticeModuli in src/columns/shape.js;
 * (2/sqrt 3) k for central springs alone), so k = E h_e (3 + r) / (2 sqrt(3) (1 + r)),
 * independent of spacing. Basal drag of a column is gamma = A eta / h_a.
 * Positions are on the unit sphere, but the radius cancels in the ratio:
 * both the drag term and the elastic energy scale with R^2.
 */
export function bondRate(p) {
  const k = normalStiffnessPerModulus(p.shear_stiffness_ratio) * p.crust_youngs_modulus_gpa * 1e9 * p.stiffness_reference_km * 1e3; // N/m
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
