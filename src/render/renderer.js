// Draws the mantle sphere, crust particles and bonds from the simulation buffers.

import { CAPACITY, structsWGSL } from '../sim/simulation.js';
import { createShaderModule } from '../gpu/device.js';

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

    this.indexBuffer = device.createBuffer({ size: sim.icosphere.indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.indexBuffer, 0, sim.icosphere.indices);
    this.viewBuffer = device.createBuffer({ size: VIEW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.viewData = new ArrayBuffer(VIEW_BYTES);

    const visibility = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    const storageLayout = device.createBindGroupLayout({
      entries: [0, 1, 2, 3].map((binding) => ({ binding, visibility, buffer: { type: 'read-only-storage' } })),
    });
    const uniformLayout = device.createBindGroupLayout({
      entries: [0, 1].map((binding) => ({ binding, visibility, buffer: { type: 'uniform' } })),
    });
    this.storageBindGroup = device.createBindGroup({
      layout: storageLayout,
      entries: [sim.buffers.particles, sim.buffers.edges, sim.buffers.grid, sim.buffers.plateIds]
        .map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    this.uniformBindGroup = device.createBindGroup({
      layout: uniformLayout,
      entries: [this.viewBuffer, sim.buffers.params].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });

    const code = [structsWGSL(), commonWGSL, mantleWGSL, crustWGSL].join('\n');
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
      bonds: pipeline('vs_bond', 'fs_bond', 'triangle-list'),
    };
  }

  #ensureDepth(width, height) {
    const t = this.depthTexture;
    if (t && t.width === width && t.height === height) return;
    t?.destroy();
    this.depthTexture = this.device.createTexture({ size: [width, height], format: DEPTH_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT });
  }

  /**
   * @param view {viewProj: Float32Array(16), lightDir: [x,y,z], stressRefMPa, showMantle, showNormal, showShear}
   */
  encodeDraw(encoder, view) {
    const colorTexture = this.context.getCurrentTexture();
    this.#ensureDepth(colorTexture.width, colorTexture.height);

    const f = new Float32Array(this.viewData);
    const u = new Uint32Array(this.viewData);
    f.set(view.viewProj, 0);
    f.set([...view.lightDir, 0], 16);
    f[20] = view.stressRefMPa;
    u[21] = (view.showNormal ? 1 : 0) | (view.showShear ? 2 : 0) | (view.colorByPlate ? 4 : 0);
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

    if (view.showMantle) {
      pass.setPipeline(this.pipelines.mantle);
      pass.setIndexBuffer(this.indexBuffer, 'uint32');
      pass.drawIndexed(this.indexCount);
    }
    // One instanced quad per particle; two quads (line + tick) per bond.
    pass.setPipeline(this.pipelines.discs);
    pass.draw(6, CAPACITY.maxParticles);
    if (view.showNormal || view.showShear) {
      pass.setPipeline(this.pipelines.bonds);
      pass.draw(12, CAPACITY.maxEdges);
    }
    pass.end();
  }
}
