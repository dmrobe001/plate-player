// Draws the mantle sphere, crust particles and per-column force and stress
// arrows from the simulation buffers.

import { SOLVER_HEADER, structsWGSL } from '../sim/simulation.js';
import { createShaderModule } from '../gpu/device.js';
import { SLOTS } from '../gpu/layout.js';

import commonWGSL from '../shaders/render/common.wgsl';
import mantleWGSL from '../shaders/render/mantle.wgsl';
import crustWGSL from '../shaders/render/crust.wgsl';

const VIEW_BYTES = 96; // must match struct View in common.wgsl
const DEPTH_FORMAT = 'depth24plus';

export class Renderer {
  constructor(device, context, format, sim, onError) {
    this.device = device;
    this.context = context;
    this.sim = sim;
    this.indexCount = sim.icosphere.indices.length;
    this.depthTexture = null;

    // Instanced draws over the particle slots in use: each frame the instance
    // count is copied from the solver header (solver_dispatch_sizes) into these
    // indirect arguments (vertex count, instance count, first vertex, first instance).
    this.drawVertices = { discs: 6, stress: 36, velocity: 18, force: 9 };
    this.drawArgs = device.createBuffer({ label: 'draw arguments', size: 16 * 4, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.drawArgs, 0, Uint32Array.from(Object.values(this.drawVertices).flatMap((v) => [v, 0, 0, 0])));
    this.indexBuffer = device.createBuffer({ size: sim.icosphere.indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.indexBuffer, 0, sim.icosphere.indices);
    this.viewBuffer = device.createBuffer({ size: VIEW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.viewData = new ArrayBuffer(VIEW_BYTES);

    const visibility = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    const storageLayout = device.createBindGroupLayout({
      entries: [0, 1, 2, 3, 4, 5].map((binding) => ({ binding, visibility, buffer: { type: 'read-only-storage' } })),
    });
    const uniformLayout = device.createBindGroupLayout({
      entries: [0, 1].map((binding) => ({ binding, visibility, buffer: { type: 'uniform' } })),
    });
    this.storageBindGroup = device.createBindGroup({
      layout: storageLayout,
      entries: [sim.buffers.particles, sim.buffers.edges, sim.buffers.grid, sim.buffers.plateIds, sim.buffers.deltas, sim.buffers.profiles]
        .map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    this.uniformBindGroup = device.createBindGroup({
      layout: uniformLayout,
      entries: [this.viewBuffer, sim.buffers.params].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });

    const slotConsts = `const SLOT_WORDS = ${SLOTS.wordsPerParticle}u;\nconst SLOT_COUNT_WORD = ${SLOTS.countWord}u;\n`
      + `const SLOT_FIRST_WORD = ${SLOTS.firstSlotWord}u;\nconst SLOT_CAPACITY = ${SLOTS.capacity}u;\n`;
    const code = [structsWGSL(), slotConsts, commonWGSL, mantleWGSL, crustWGSL].join('\n');
    const module = createShaderModule(device, 'render', code, onError);
    const layout = device.createPipelineLayout({ bindGroupLayouts: [storageLayout, uniformLayout] });
    const pipeline = (vertex, fragment, topology) => device.createRenderPipeline({
      label: vertex,
      layout,
      vertex: { module, entryPoint: vertex },
      fragment: { module, entryPoint: fragment, targets: [{ format }] },
      primitive: { topology, cullMode: 'none' },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
    });
    this.pipelines = {
      mantle: pipeline('vs_mantle', 'fs_mantle', 'triangle-list'),
      discs: pipeline('vs_disc', 'fs_disc', 'triangle-list'),
      force: pipeline('vs_force', 'fs_glyph', 'triangle-list'),
      stress: pipeline('vs_stress', 'fs_glyph', 'triangle-list'),
      velocity: pipeline('vs_velocity', 'fs_glyph', 'triangle-list'),
    };
  }

  #ensureDepth(width, height) {
    const t = this.depthTexture;
    if (t && t.width === width && t.height === height) return;
    t?.destroy();
    this.depthTexture = this.device.createTexture({ size: [width, height], format: DEPTH_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT });
  }

  /**
   * @param view {viewProj: Float32Array(16), lightDir: [x,y,z], stressRefMPa, showMantle, showForce, showStress, showVelocity, colorByPlate}
   */
  encodeDraw(encoder, view) {
    const colorTexture = this.context.getCurrentTexture();
    this.#ensureDepth(colorTexture.width, colorTexture.height);
    for (let k = 0; k < Object.keys(this.drawVertices).length; k++) {
      encoder.copyBufferToBuffer(this.sim.buffers.solver, 4 * SOLVER_HEADER.drawInstances, this.drawArgs, 16 * k + 4, 4);
    }

    const f = new Float32Array(this.viewData);
    const u = new Uint32Array(this.viewData);
    f.set(view.viewProj, 0);
    f.set([...view.lightDir, 0], 16);
    f[20] = view.stressRefMPa;
    u[21] = (view.showForce ? 1 : 0) | (view.showStress ? 2 : 0) | (view.colorByPlate ? 4 : 0) | (view.showMantle ? 0 : 8) | (view.showVelocity ? 16 : 0);
    this.device.queue.writeBuffer(this.viewBuffer, 0, this.viewData);

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: colorTexture.createView(),
        clearValue: { r: 0.06, g: 0.09, b: 0.16, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    pass.setBindGroup(0, this.storageBindGroup);
    pass.setBindGroup(1, this.uniformBindGroup);

    // The mantle sphere is always drawn (black when hidden, flag bit 3), so
    // the far side of the globe never shows through.
    pass.setPipeline(this.pipelines.mantle);
    pass.setIndexBuffer(this.indexBuffer, 'uint32');
    pass.drawIndexed(this.indexCount);
    // One instanced quad per particle; arrows per particle (9 vertices for the
    // force, 36 for the two principal axes).
    const draw = (name) => {
      pass.setPipeline(this.pipelines[name]);
      pass.drawIndirect(this.drawArgs, 16 * Object.keys(this.drawVertices).indexOf(name));
    };
    draw('discs');
    if (view.showStress) draw('stress');
    if (view.showVelocity) draw('velocity');
    if (view.showForce) draw('force');
    pass.end();
  }
}
