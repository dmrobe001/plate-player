// GPU-side simulation state and the per-frame compute schedule.

import { Constraint, GridNode, NONE, Particle } from '../gpu/layout.js';
import { createShaderModule } from '../gpu/device.js';
import { packSimParams, SIM_PARAMS_BYTES, simParamsWGSL } from './params.js';

import bindingsWGSL from '../shaders/compute/bindings.wgsl';
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

const MANTLE_SUBSTEPS = 50;   // Swift-Hohenberg steps per frame
const SOLVER_ITERATIONS = 3;  // PBD Jacobi iterations per frame
const FREE_LIST_HEADER_WORDS = 4;
const WORKGROUP = 64;

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
    const { maxParticles, maxConstraints, hashGridDim, adjacencyWidth } = CAPACITY;
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    this.buffers = {
      grid: device.createBuffer({ label: 'grid', size: this.numNodes * GridNode.byteSize, usage: S }),
      adjacency: device.createBuffer({ label: 'adjacency', size: this.numNodes * adjacencyWidth * 4, usage: S }),
      particles: device.createBuffer({ label: 'particles', size: maxParticles * Particle.byteSize, usage: S }),
      constraints: device.createBuffer({ label: 'constraints', size: maxConstraints * Constraint.byteSize, usage: S }),
      particlePool: device.createBuffer({ label: 'particle free list', size: (FREE_LIST_HEADER_WORDS + maxParticles) * 4, usage: S }),
      constraintPool: device.createBuffer({ label: 'constraint free list', size: (FREE_LIST_HEADER_WORDS + maxConstraints) * 4, usage: S }),
      hashGrid: device.createBuffer({ label: 'hash grid', size: hashGridDim ** 3 * 4, usage: S }),
      deltas: device.createBuffer({ label: 'particle deltas', size: maxParticles * 16, usage: S }),
      params: device.createBuffer({ label: 'sim params', size: SIM_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    };

    device.queue.writeBuffer(this.buffers.params, 0, packSimParams({
      num_nodes: this.numNodes,
      max_particles: maxParticles,
      max_constraints: maxConstraints,
      hash_grid_dim: hashGridDim,
    }));

    const adjacency = new Uint32Array(this.numNodes * adjacencyWidth).fill(NONE);
    icosphere.neighbours.forEach((list, i) => adjacency.set(list.slice(0, adjacencyWidth), i * adjacencyWidth));
    device.queue.writeBuffer(this.buffers.adjacency, 0, adjacency);

    this.#createPipelines(onError);
  }

  #createPipelines(onError) {
    const { device, buffers } = this;
    const code = [structsWGSL({ atomicGrid: true }), bindingsWGSL, mantleWGSL, thermalWGSL, growthWGSL, solverWGSL].join('\n');
    const module = createShaderModule(device, 'compute', code, onError);

    const order = ['grid', 'adjacency', 'particles', 'constraints', 'particlePool', 'constraintPool', 'hashGrid', 'deltas'];
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
      'grid_clear_thermal', 'crust_apply_thermal', 'grid_finalize_thermal',
      'mantle_laplacian', 'mantle_integrate', 'mantle_velocity',
      'crust_thermo_isostasy', 'crust_constraint_healing',
      'crust_accumulation', 'crust_rewire_bonds', 'crust_clean_split',
      'crust_predict', 'crust_hash_clear', 'crust_hash_build',
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
      pu[base + po.is_active] = 1;
      pu[base + po.next_idx] = NONE;
      pf[base + po.density] = P.compositionDensity[i];
      pu[base + po.split_child] = NONE;
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
      cf[base + co.compliance] = 0.001;
      cf[base + co.yield_stress_mpa] = state.yieldStressMPa;
      cu[base + co.is_active] = 1;
      cf[base + co.damage] = B.damage[k];
    }
    device.queue.writeBuffer(buffers.constraints, 0, cbuf);

    // Free lists hold every slot not used by the initial state.
    device.queue.writeBuffer(buffers.particlePool, 0, freeList(P.count, maxParticles));
    device.queue.writeBuffer(buffers.constraintPool, 0, freeList(B.count, maxConstraints));
    this.steps = 0;
  }

  /** Record one simulation step (one frame) into a command encoder. */
  encodeStep(encoder) {
    const { maxParticles, maxConstraints, hashGridDim } = CAPACITY;
    const groups = (n) => Math.ceil(n / WORKGROUP);
    const nodes = groups(this.numNodes), crust = groups(maxParticles);
    const bonds = groups(maxConstraints), cells = groups(hashGridDim ** 3);

    const pass = encoder.beginComputePass({ label: 'simulation step' });
    pass.setBindGroup(0, this.memoryBindGroup);
    pass.setBindGroup(1, this.uniformBindGroup);
    const run = (name, count) => { pass.setPipeline(this.pipelines[name]); pass.dispatchWorkgroups(count); };

    // Crust -> mantle heat feedback
    run('grid_clear_thermal', nodes);
    run('crust_apply_thermal', crust);
    run('grid_finalize_thermal', nodes);

    // Mantle pattern and the velocity field it implies
    for (let i = 0; i < MANTLE_SUBSTEPS; i++) {
      run('mantle_laplacian', nodes);
      run('mantle_integrate', nodes);
    }
    run('mantle_velocity', nodes);

    // Crust thermal state, healing, growth and splitting
    run('crust_thermo_isostasy', crust);
    run('crust_constraint_healing', bonds);
    run('crust_accumulation', crust);
    run('crust_rewire_bonds', bonds);
    run('crust_clean_split', crust);

    // Crust motion (position-based dynamics)
    run('crust_predict', crust);
    run('crust_hash_clear', cells);
    run('crust_hash_build', crust);
    for (let i = 0; i < SOLVER_ITERATIONS; i++) {
      run('crust_clear_deltas', crust);
      run('crust_solve_constraints', bonds);
      run('crust_solve_collisions', crust);
      run('crust_apply_deltas', crust);
    }
    run('crust_integrate', crust);
    pass.end();
    this.steps++;
  }
}

function freeList(used, capacity) {
  const data = new Uint32Array(FREE_LIST_HEADER_WORDS + capacity);
  let n = 0;
  for (let i = used; i < capacity; i++) data[FREE_LIST_HEADER_WORDS + n++] = i;
  data[0] = n;
  return data;
}
